import os
import time
import logging
import joblib
import warnings
import numpy as np
import pandas as pd
import requests
import duckdb
import yfinance as yf
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional, Tuple, Any
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

# =========================================================
# ロギングと警告の設定
# =========================================================
class JSTFormatter(logging.Formatter):
    """ログのタイムスタンプをJSTで出力するためのカスタムフォーマッター"""
    converter = lambda *args: datetime.now(timezone(timedelta(hours=9))).timetuple()

    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, tz=timezone.utc) + timedelta(hours=9)
        if datefmt:
            return dt.strftime(datefmt)
        else:
            return dt.isoformat(sep=" ", timespec="milliseconds")

warnings.filterwarnings("ignore")

handler = logging.StreamHandler()
handler.setFormatter(JSTFormatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S,%f'))
logging.basicConfig(level=logging.INFO, handlers=[handler])
logger = logging.getLogger(__name__)

# yfinanceのキャッシュ警告対策（GitHub Actions環境などでの権限エラー回避）
yf.set_tz_cache_location(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache"))

# =========================================================
# Configuration
# =========================================================
class Config:
    """
    システム全体の設定（定数）を管理するクラスです。
    パス、APIトークン、対象市場、モデルの再学習頻度などをここで一元管理します。
    """
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DB_PATH = os.path.join(BASE_DIR, "market.db")
    MODEL_PATH = os.path.join(BASE_DIR, "model_v2.pkl")
    HISTORY_PATH = os.path.join(BASE_DIR, "recommendation_history.csv")
    LINE_ACCESS_TOKEN = os.getenv("LINE_BOT_TOKEN")
    LINE_USER_ID = os.getenv("LINE_USER_ID") or "dummy"
    TARGET_MARKETS = ["プライム", "スタンダード", "グロース"]
    RETRAIN_DAYS = 7
    THRESHOLD_STRICT = 0.50  # 高い確信度
    THRESHOLD_NORMAL = 0.45  # 標準

# =========================================================
# Feature Engineering (Unified)
# =========================================================
class FeatureFactory:
    """
    株価データからAIが学習・推論するために必要な「テクニカル指標（特徴量）」を生成するクラスです。
    学習時と推論時で同じ計算ロジックを使用することで、AIの精度低下（計算の乖離）を防ぎます。
    """
    
    FEATURE_COLS = [
        "SMA5", "SMA25", "SMA75", "Bias5", "Bias25", "Bias75",
        "BB_UP1", "BB_LOW1", "BB_UP2", "BB_LOW2", "VolRatio",
        "Bull", "BigBull", "BigBear", "Slope10", "Slope20", "SlopeAccel", "ret10", "RSI", "MACD_Hist",
        "ret1", "ret3", "ret5", "ret20", "atr_ratio",
        "VolVCP"
    ] # AIが判断に使用する項目のリスト

    @staticmethod
    def calculate_metrics(df: pd.DataFrame) -> pd.DataFrame:
        """
        移動平均、ボリンジャーバンド、出来高比率、スロープ（傾き）などの
        テクニカル指標を計算します。
        """
        df = df.copy()
        close = df["Close"]
        
        # 移動平均と乖離率
        for n in [5, 25, 75]:
            df[f"SMA{n}"] = close.rolling(n).mean()
            df[f"Bias{n}"] = (close - df[f"SMA{n}"]) / df[f"SMA{n}"].replace(0, np.nan)

        # ボリンジャーバンド
        # screening_ai.py の計算式に合わせる
        df["BB_MID"] = df["SMA25"]
        df["BB_STD"] = close.rolling(25).std()
        std25 = df["BB_STD"]

        df["BB_UP1"] = df["SMA25"] + std25
        df["BB_LOW1"] = df["SMA25"] - std25
        df["BB_UP2"] = df["SMA25"] + 2 * std25
        df["BB_LOW2"] = df["SMA25"] - 2 * std25

        # 出来高とリターン
        df["VolRatio"] = df["Volume"] / df["Volume"].rolling(25).mean().replace(0, np.nan)
        for n in [1, 3, 5, 10, 20]:
            df[f"ret{n}"] = close.pct_change(n)

        # ボラティリティの収束 (VCP: Volatility Contraction Pattern)
        # 短期のボラティリティが長期に対して低下しているか（＝エネルギーが溜まっているか）
        vol_short = close.pct_change().rolling(10).std()
        vol_long = close.pct_change().rolling(60).std()
        df["VolVCP"] = vol_short / vol_long.replace(0, np.nan)

        # RSI (14日間)
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-9)
        df["RSI"] = 100 - (100 / (1 + rs))

        # MACD (12, 26, 9)
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        df["MACD_Hist"] = (ema12 - ema26) - (ema12 - ema26).ewm(span=9, adjust=False).mean()

        # スロープ (線形回帰によるトレンド検知)
        def calc_slope(series):
            if len(series) < 10: return 0
            y = series.values
            x = np.arange(len(y))
            slope = np.polyfit(x, y, 1)[0]
            return slope

        df["Slope10"] = close.rolling(10).apply(calc_slope, raw=False)
        df["Slope20"] = df["Slope10"].rolling(20).mean()
        df["SlopeAccel"] = df["Slope10"].diff()

        # ATR比率
        atr = (df["High"] - df["Low"]).rolling(14).mean()
        df["atr_ratio"] = atr / close.replace(0, np.nan)

        # ローソク足
        df["Bull"] = (close > df["Open"]).astype(int)
        df["BigBull"] = ((close - df["Open"]) / df["Open"].replace(0, np.nan) > 0.03).astype(int)
        df["BigBear"] = ((df["Open"] - close) / df["Open"].replace(0, np.nan) > 0.03).astype(int)

        # 無限大をNaNに変換
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # 指標が計算できていない初期の行（SMA75などがNaNの期間）を削除してから、残りを0埋め
        return df.dropna(subset=["SMA75", "Slope20"]).fillna(0).replace([np.inf, -np.inf], 0)

    @staticmethod
    def add_target_label(df: pd.DataFrame) -> pd.DataFrame:
        """
        学習用：AIに「正解」を教えるためのラベルを作成します。
        「出来高が静かな状態（仕込み時）から5日以内に急騰したケース」のみを正解と定義します。
        これにより、爆上げした後の銘柄ではなく、爆上げ前の予兆を学習させます。
        """
        # 未来の最大上昇ポテンシャル（明日から5日間の高値）
        future_max = df["High"].shift(-5).rolling(5).max()
        future_gain = future_max / df["Close"] - 1
        
        # 仕込み時の条件（現在が静かであること）
        is_quiet = (df["VolRatio"] < 1.0) # 出来高が平均以下
        is_stable = (df["ret5"].between(-0.05, 0.02)) # 5日間で暴騰も暴落もしていない
        
        # 5日以内に5%以上の利確チャンスがあるか
        will_breakout = (future_gain >= 0.05)

        df["Target"] = np.where(future_gain.notna(), (is_quiet & is_stable & will_breakout).astype(int), np.nan)
        return df

# =========================================================
# Data Management
# =========================================================
class DatabaseManager:
    """
    ローカルデータベース（DuckDB）への接続と、Yahoo Financeからのデータ取得を管理するクラスです。
    株価データの保存・読み込みを一手に引き受けます。
    """
    def __init__(self, db_path: str):
        self.db_path = db_path

    def _get_connection(self):
        """データベースへの接続を取得します。"""
        return duckdb.connect(self.db_path)

    def get_market_regime(self) -> bool:
        """
        日経平均(^N225)のデータを取得し、市場全体が上昇トレンド(SMA75以上)か判定します。
        """
        try:
            n225 = yf.download("^N225", period="150d", progress=False)
            if n225.empty:
                return True # 取得失敗時は安全のためTrue
            
            close = n225["Close"]
            # 1銘柄のみの場合のSeries変換
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
                
            sma75 = close.rolling(75).mean().iloc[-1]
            current = close.iloc[-1]
            return current > sma75
        except Exception as e:
            logger.error(f"地合い判定エラー: {e}")
            return True

    def update_prices(self, symbols_df: pd.DataFrame):
        """
        銘柄リストに基づいて、最新の株価をYahoo Financeからダウンロードし、
        データベースに一括で保存（INSERT OR IGNORE）します。
        """
        logger.info("DuckDB価格更新開始...")
        codes = symbols_df["コード"].tolist()
        failed_codes = []
        total_processed_symbols = 0

        # デバッグ用：ファイル存在確認
        if os.path.exists(self.db_path):
            logger.info(f"既存のDBファイルを検出: {self.db_path} ({os.path.getsize(self.db_path)} bytes)")
        else:
            logger.info("既存のDBファイルが見つかりません。新規作成します。")

        # yfinance自体のログ出力を抑制して、エラーログの煩雑さを抑える
        yf_logger = logging.getLogger('yfinance')
        yf_logger.setLevel(logging.CRITICAL)

        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS prices (
                    code TEXT, date DATE, open DOUBLE, high DOUBLE, 
                    low DOUBLE, close DOUBLE, volume DOUBLE, PRIMARY KEY (code, date)
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_prices_date ON prices (date)")
            
            # データベースに既にデータがあるか確認
            try:
                res = conn.execute("SELECT COUNT(*) FROM prices").fetchone()
                has_data = res[0] > 0
                logger.info(f"現在のDB内レコード数: {res[0]}件")
            except Exception:
                has_data = False

            # データがあれば直近5日分のみ、なければ1年分を取得
            period_setting = "5d" if has_data else "1y"
            logger.info(f"データ取得モード: {'差分(5d)' if has_data else 'フル(1y)'}")
            
            batch_size = 100
            for i in range(0, len(codes), batch_size):
                batch_codes = codes[i:i+batch_size]
                tickers = " ".join([f"{c}.T" for c in batch_codes])
                
                try:
                    df = yf.download(
                        tickers,
                        period=period_setting,
                        group_by="ticker",
                        progress=False,
                        threads=True,
                        timeout=20 # Add timeout to prevent indefinite hangs
                    )
                    if df.empty: continue

                    # 1銘柄のみの場合、MultiIndexにならないケースがあるための正規化
                    # これを行わないと、最後のバッチ（端数）がスキップされる可能性があります。
                    if not isinstance(df.columns, pd.MultiIndex) and len(batch_codes) == 1:
                        symbol = f"{batch_codes[0]}.T"
                        df.columns = pd.MultiIndex.from_product([[symbol], df.columns])

                    dfs_to_insert = []
                    for code in batch_codes:
                        symbol = f"{code}.T"
                        if symbol not in df.columns.get_level_values(0):
                            failed_codes.append(symbol)
                            continue
                        
                        df_s = df[symbol].dropna().reset_index()
                        if df_s.empty: continue
                        
                        df_s.columns = ["date", "open", "high", "low", "close", "volume"]
                        df_s["code"] = code
                        dfs_to_insert.append(df_s[["code", "date", "open", "high", "low", "close", "volume"]])
                    
                    if dfs_to_insert:
                        merged = pd.concat(dfs_to_insert)
                        conn.register("tmp_df", merged)
                        conn.execute("""
                            INSERT INTO prices (code, date, open, high, low, close, volume)
                            SELECT code, date, open, high, low, close, volume FROM tmp_df
                            ON CONFLICT DO NOTHING
                        """)
                        conn.unregister("tmp_df")
                        total_processed_symbols += len(dfs_to_insert)
                    logger.debug(f"Batch {i}-{i+len(batch_codes)-1} processed. Inserted/updated {len(dfs_to_insert)} symbols.")
                    time.sleep(1)  # Yahoo APIのレートリミットを回避するための待機
                except Exception as e:
                    logger.error(f"Batch {i} download error: {e}")
            
            logger.info(f"株価データの更新完了: 合計 {total_processed_symbols} 銘柄を処理しました。")

            # 古いデータのクリーンアップ（2年以上前のデータを削除）
            if datetime.now().weekday() == 6:  # 日曜日のみ実行して負荷軽減
                logger.info("古いデータのクリーンアップを実行中...")
                conn.execute("DELETE FROM prices WHERE date < CURRENT_DATE - INTERVAL 2 YEAR")

        if failed_codes:
            logger.warning(f"取得失敗銘柄数: {len(failed_codes)} (例: {failed_codes[:5]}...)")

    def load_all_data(self, symbols_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """データベースから指定された銘柄の過去400日分程度のデータを一括で読み込みます。"""
        codes = symbols_df["コード"].tolist()
        if not codes:
            return {}

        if not os.path.exists(self.db_path):
            return {}

        query = """
            SELECT code, date, open as Open, high as High, low as Low, close as Close, volume as Volume
            FROM prices 
            WHERE code IN ? 
              AND date >= CURRENT_DATE - INTERVAL 400 DAY
            ORDER BY code, date
        """
        with self._get_connection() as conn:
            df = conn.execute(query, [codes]).df()
        
        return {f"{str(code).zfill(4)}.T": group.set_index("date") for code, group in df.groupby("code")}

# =========================================================
# Notification
# =========================================================
def send_line(message: str):
    """
    分析結果やシステムエラーを、LINE Messaging APIを通じて
    指定したユーザーのLINEにプッシュ通知します。
    """
    if not Config.LINE_ACCESS_TOKEN or not Config.LINE_USER_ID:
        logger.warning("LINE設定が不足しています。")
        return

    url = "https://api.line.me/v2/bot/message/push"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {Config.LINE_ACCESS_TOKEN}"}
    data = {"to": Config.LINE_USER_ID, "messages": [{"type": "text", "text": message}]}
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=15)
        if response.status_code != 200: logger.error(f"LINE送信失敗: {response.text}")
    except Exception as e:
        logger.error(f"LINE送信エラー: {e}")

# =========================================================
# Core Screener Logic
# =========================================================
class StockScreener:
    """
    スクリーニングの全体工程（ワークフロー）を制御するメインクラスです。
    データの更新、特徴量計算、モデルの準備、AI推論、通知の順に実行します。
    """
    def __init__(self):
        self.db = DatabaseManager(Config.DB_PATH)
        self.factory = FeatureFactory()
        self.model = None

    def run(self):
        """スクリーニングの全工程を順番に実行します。"""
        logger.info("=== スクリーニング開始 ===")
        symbols = self._load_symbols()

        # 地合いチェック
        is_market_good = self.db.get_market_regime()
        if not is_market_good:
            logger.warning("【注意】地合いが悪化しています（日経平均が75日線以下）。厳選モードで動作します。")
            # 地合いが悪い時は閾値を上げる
            current_threshold = Config.THRESHOLD_STRICT
        else:
            current_threshold = Config.THRESHOLD_NORMAL
        
        # データ更新
        self.db.update_prices(symbols)
        all_data = self.db.load_all_data(symbols)
        
        # 特徴量生成（並列）
        processed_data = self._parallel_feature_engineering(all_data)
        
        # モデル準備
        if not self._prepare_model(all_data):
            logger.error("モデルの準備に失敗したため、スクリーニングを中断します。")
            send_line("【システム通知】AIモデルの準備に失敗しました。データ量を確認してください。")
            return
        
        # 推論とランキング
        buy_results, sell_results = self._inference(processed_data, current_threshold)
        
        # 結果通知
        self._notify((buy_results, sell_results), symbols, is_market_good)

    def _load_symbols(self) -> pd.DataFrame:
        """JPXの銘柄リストCSVを読み込み、対象とする市場（プライム等）で絞り込みます。"""
        if not os.path.exists("japan_stocks_jpx.csv"):
            logger.error("japan_stocks_jpx.csv が見つかりません。")
            raise FileNotFoundError("japan_stocks_jpx.csv")
        df = pd.read_csv("japan_stocks_jpx.csv", dtype=str)
        df.columns = df.columns.str.strip()
        df["市場"] = df["市場・商品区分"].str.extract(r"(プライム|スタンダード|グロース)")
        return df[df["市場"].isin(Config.TARGET_MARKETS)][["コード", "銘柄名", "市場"]].dropna()

    def _parallel_feature_engineering(self, all_data: Dict) -> Dict:
        logger.info("特徴量生成（並列処理）を開始します...")
        results = Parallel(n_jobs=-1)(delayed(self._feature_worker)(s, d) for s, d in all_data.items())
        processed_data = {r[0]: r[1] for r in results if r is not None}
        logger.info(f"特徴量生成（並列処理）が完了しました。処理済み銘柄数: {len(processed_data)}")
        return processed_data

    def _feature_worker(self, symbol, df):
        """個別の銘柄の特徴量計算ワーカー関数"""
        if len(df) < 80:
            return None
        
        now_jst = datetime.now(timezone.utc) + timedelta(hours=9)
        
        # 9:00〜9:05の間は価格が極めて不安定なため、当日データが含まれている場合はそれを除外して前日ベースで判定する
        if now_jst.hour == 9 and now_jst.minute < 5:
            if df.index[-1].date() == now_jst.date():
                df = df.iloc[:-1]
        # 9:05以降の9時台は、始値を暫定的な終値として扱うことで、寄り付きの勢いを反映させる（既存ロジックの改善）
        elif now_jst.hour == 9 and df.index[-1].date() == now_jst.date():
            df.iloc[-1, df.columns.get_loc("Close")] = df.iloc[-1, df.columns.get_loc("Open")]

        feat_df = self.factory.calculate_metrics(df)
        if len(feat_df) < 10:
            return None
        return symbol, feat_df.iloc[-1]

    def _prepare_model(self, all_data: Dict) -> bool:
        """
        AIモデル（RandomForest）を準備します。
        前回の学習から一定期間が経過している場合は再学習を行い、
        そうでなければ保存されたモデルファイルを読み込みます。
        """
        def train_worker(symbol, df):
            if len(df) < 120: return None
            logger.debug(f"Preparing training features for {symbol}")
            feat_df = self.factory.calculate_metrics(df)
            logger.debug(f"Finished preparing training features for {symbol}")
            feat_df = self.factory.add_target_label(feat_df)
            # 特徴量計算後の有効データが少ない銘柄は、学習の質を下げるため除外
            if len(feat_df) < 30:
                return None
            # 学習に必要なカラムのみを抽出してメモリを節約
            cols = self.factory.FEATURE_COLS + ["Target"]
            return feat_df.iloc[:-5][cols]

        # モデルの読み込みと整合性チェック
        need_training = False
        if not os.path.exists(Config.MODEL_PATH):
            need_training = True
        elif (datetime.now() - datetime.fromtimestamp(os.path.getmtime(Config.MODEL_PATH))).days >= Config.RETRAIN_DAYS:
            need_training = True
        else:
            try:
                self.model = joblib.load(Config.MODEL_PATH)
                # 学習時の特徴量リストを取得して比較
                trained_features = []
                if hasattr(self.model, "feature_names_in_"):
                    trained_features = list(self.model.feature_names_in_)
                elif hasattr(self.model, "calibrated_classifiers_"):
                    trained_features = list(self.model.calibrated_classifiers_[0].estimator.feature_names_in_)
                
                if trained_features and trained_features != self.factory.FEATURE_COLS:
                    logger.warning(f"特徴量構成の変更を検知しました（旧:{len(trained_features)}種 -> 新:{len(self.factory.FEATURE_COLS)}種）。再学習を強制します。")
                    need_training = True
            except Exception as e:
                logger.warning(f"モデルチェック中にエラーが発生しました: {e}。再学習を実行します。")
                need_training = True

        if need_training:
            logger.info("モデルを新規学習します...")
            
            # 学習データの準備を並列化
            results = Parallel(n_jobs=2)(delayed(train_worker)(s, d) for s, d in all_data.items())
            training_dfs = [r for r in results if r is not None]
            
            if not training_dfs:
                logger.error("学習に使用できる有効なデータがありませんでした。")
                return False

            full_train = pd.concat(training_dfs).dropna(subset=["Target"])
            X = full_train[self.factory.FEATURE_COLS]
            y = full_train["Target"]

            logger.info(f"学習データの内訳 - Target=1 (上昇): {int(y.sum())}件")
            logger.info(f"学習データの内訳 - Target=0 (その他): {int((y == 0).sum())}件")
            
            logger.info(f"AIモデルの学習を開始します (データ件数: {len(X)})...")
            base_model = RandomForestClassifier(
                n_estimators=300,
                max_depth=12,
                min_samples_leaf=5,
                n_jobs=2,
                random_state=42,
                class_weight="balanced"
            )
            self.model = CalibratedClassifierCV(
                estimator=base_model,
                method="sigmoid",
                cv=3
            )
            self.model.fit(X, y)
            joblib.dump(self.model, Config.MODEL_PATH)
            logger.info("モデルの学習と保存が完了しました。")
        else:
            try:
                self.model = joblib.load(Config.MODEL_PATH)
            except Exception as e:
                logger.error(f"モデルの読み込みに失敗しました: {e}")
                return False
        return self.model is not None

    def _inference(self, feature_dict: Dict, threshold: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        最新の指標データに基づいてAIが「上昇確率」を予測します。
        確率が高い銘柄に対し、期待値（EV）を計算してランキングを作成します。
        """
        if not feature_dict:
            logger.error("特徴量データが空です。有効な株価データが不足している可能性があります。")
            return pd.DataFrame()

        # モデルが正常に準備できていない場合のガード
        if self.model is None:
            logger.error("AIモデルが準備できていないため、推論をスキップします。")
            return pd.DataFrame()

        symbols = list(feature_dict.keys())
        features = pd.DataFrame([feature_dict[s] for s in symbols])
        
        # クラス数のチェック（学習データにTarget=1が不在の場合のIndexErrorを防止）
        proba = self.model.predict_proba(features[self.factory.FEATURE_COLS])
        if proba.shape[1] < 2:
            logger.error("AIモデルの学習データに正解（Target=1）が含まれていなかったため、推論をスキップします。")
            return pd.DataFrame()
            
        probs = proba[:, 1]
        
        res_df = pd.DataFrame({"symbol": symbols, "prob": probs})
        res_df = pd.concat([res_df, features.reset_index(drop=True)], axis=1)
        
        # 期待値(EV)計算の刷新
        res_df["norm_slope"] = res_df["Slope10"] / res_df["Close"].replace(0, np.nan)
        res_df["SlopeScore"] = res_df["norm_slope"].clip(-0.01, 0.01) * 100
        
        # 期待リターンの推計 (過去のトレンド + 現在の勢い)
        # SlopeScore * 0.1 により、強い上昇トレンドを+10%のリターン期待値として加算
        res_df["ExpectedReturn"] = (
            res_df["ret20"].clip(-0.03, 0.15) 
            + (res_df["SlopeScore"] * 0.1)
        )

        # EV = AI確率 × 期待リターン × 出来高の静寂性
        # SilenceScoreを乗算することで、出来高が急増している「飛びつき」を抑制し、仕込み銘柄を優先
        res_df["SilenceScore"] = (1.2 - res_df["VolRatio"]).clip(0.7, 1.3)
        res_df["EV"] = res_df["prob"] * res_df["ExpectedReturn"] * res_df["SilenceScore"]
        
        # フィルタリング
        logger.info(f"推論完了: {len(res_df)} 銘柄を評価中... (最大確率: {res_df['prob'].max():.3f})")

        # 【勝率向上のためのフィルタ強化】
        # 1. 確率閾値
        # 2. ボラティリティ収束 (VolVCP < 1.0 は短期の値動きが落ち着いている証拠)
        # 3. 過熱感の抑制 (Bias25 > 0.08 などの高値掴みを防止)
        cond_prob = (res_df["prob"] >= threshold) & (res_df["VolVCP"] < 1.1) & (res_df["Bias25"] < 0.06)
        # 傾きの条件を緩和し、仕込み段階の銘柄も拾えるようにする（わずかな下降も許容）
        cond_slope = res_df["Slope20"] > -0.003
        cond_ret = res_df["ret10"].between(-0.05, 0.07)
        cond_vol = res_df["VolRatio"] < 1.2

        logger.info(f"【条件別ヒット数】 AI確率＆収束フィルタ: {cond_prob.sum()}, 傾き(Slope>-0.003): {cond_slope.sum()}, 安定性(ret10): {cond_ret.sum()}, 出来高(静寂): {cond_vol.sum()}")

        filtered = res_df[cond_prob & cond_slope & cond_ret & cond_vol].sort_values("EV", ascending=False)

        # 厳選フィルタで0件の場合、AI確率上位を「準候補」として救い出す（閾値を少し柔軟に）
        if filtered.empty and not res_df.empty:
            # AI確率の閾値を超えているが、テクニカルフィルタで落ちた銘柄を探す
            potential_candidates = res_df[cond_prob].sort_values("prob", ascending=False).head(3).copy()
            
            if not potential_candidates.empty:
                logger.info(f"厳選フィルタは0件ですが、高確率銘柄({len(potential_candidates)}件)を準候補として保持します。")
                potential_candidates["is_potential"] = True

                # 準候補になった理由を特定
                reasons = []
                for idx, row in potential_candidates.iterrows():
                    reason_str = []
                    if not (row["Slope10"] > 0):
                        reason_str.append("傾き不足")
                    if not (-0.03 <= row["ret10"] <= 0.05):
                        reason_str.append("安定性不足")
                    if not (row["VolRatio"] < 1.2):
                        reason_str.append(f"出来高急増({row['VolRatio']:.2f})")
                    potential_candidates.loc[idx, "potential_reason"] = "、".join(reason_str) if reason_str else "その他"
                
                filtered = potential_candidates

        logger.info(f"フィルタ最終通過: {len(filtered)} 銘柄")

        # 【プロ視点】売り・警戒銘柄の検知ロジック
        # 1. RSIが75以上で反落開始 (買われすぎからの調整)
        # 2. MACDヒストグラムが負 (勢いの低下)
        # 3. 重要な節目(SMA25)を割り込んだ (トレンド崩れ)
        # 4. 急激な陰線 (ボラティリティ・ストップ)
        cond_sell = (
            ((res_df["RSI"] > 80) & (res_df["ret1"] < -0.02)) |  # 超買われすぎからの反落
            (res_df["MACD_Hist"] / res_df["Close"] < -0.003) |  # 価格比での勢い低下（正規化）
            (res_df["Close"] < res_df["SMA25"] * 0.97) |        # 25日線を3%以上明確に割り込み
            (res_df["ret1"] < -0.05)                            # 5%以上の急落（ストップロス）
        )
        exit_candidates = res_df[cond_sell & (res_df["Slope10"] < 0.05)].sort_values("ret1", ascending=True)

        return filtered.head(5), exit_candidates

    def _notify(self, results: Tuple[pd.DataFrame, pd.DataFrame], symbols_df: pd.DataFrame, is_market_good: bool):
        """最終的なランキング結果を整形し、LINEへ送信します。"""
        buy_results, sell_results = results
        name_map = dict(zip(symbols_df["コード"] + ".T", symbols_df["銘柄名"]))
        
        msg = ["【AI厳選銘柄ランキング】"]
        if not is_market_good:
            msg.append("（⚠️地合い弱気・厳選モード）")

        # 準候補（フィルタ落ちだが高確率）がある場合のヘッダー追加
        if not buy_results.empty and "is_potential" in buy_results.columns and buy_results["is_potential"].any():
            msg.append("【AI準候補・監視推奨】")
            if "potential_reason" in buy_results.columns and not buy_results["potential_reason"].isna().all():
                msg.append(f"（高確率だが{buy_results['potential_reason'].iloc[0]}のため厳選フィルタ除外）")

        if buy_results.empty:
            msg.append("該当なし")
        else:
            for i, (_, row) in enumerate(buy_results.iterrows(), 1):
                name = name_map.get(row['symbol'], "不明")
                
                # ATRに基づく損切り目安 (2 * ATR)
                # atr_ratio は ATR / Close なので、Close * (1 - atr_ratio * 2) が損切り価格
                stop_loss_price = row['Close'] * (1 - row['atr_ratio'] * 2)
                
                msg.append(f"{i}位 {row['symbol']} {name[:8]}\n  価格:{row['Close']:.1f} (目安損切:{stop_loss_price:.1f})\n  確率:{row['prob']:.1%} EV:{row['EV']:.2f}\n  Slope:{row['norm_slope']:.4f} Vol:{row['VolRatio']:.2f}")
                # ログに詳細な分析根拠を出力
                logger.info(f"分析詳細 {i}位: {row['symbol']} ({name}) - 確率: {row['prob']:.3f}, EV: {row['EV']:.3f}, 傾き: {row['norm_slope']:.4f}, 出来高比: {row['VolRatio']:.2f}")

        # --- 過去の推奨銘柄の管理ロジック ---
        # 実行環境のタイムゾーンに依らず日本時間(JST)で日付を管理します。
        jst = timezone(timedelta(hours=9))
        today_jst = datetime.now(jst).date()

        history_df = pd.DataFrame(columns=["date", "symbol"])
        if os.path.exists(Config.HISTORY_PATH):
            try:
                history_df = pd.read_csv(Config.HISTORY_PATH)
                history_df["date"] = pd.to_datetime(history_df["date"]).dt.date
            except Exception as e:
                logger.error(f"履歴ファイルの読み込みに失敗しました: {e}")
        
        # 監視対象：本日より前に推奨された銘柄のみを抽出
        sell_results_for_notified_buys = pd.DataFrame()
        if not history_df.empty:
            # 本日分を追加する前のリストで売り判定を行う
            monitored_symbols = set(history_df[history_df["date"] < today_jst]['symbol'].tolist())
            sell_results_for_notified_buys = sell_results[sell_results['symbol'].isin(monitored_symbols)]

        # 本日の新規推奨銘柄を履歴に追加
        if not buy_results.empty:
            new_history = pd.DataFrame({"date": [today_jst] * len(buy_results), "symbol": buy_results["symbol"].tolist()})
            history_df = pd.concat([history_df, new_history]).drop_duplicates(subset=["date", "symbol"])

        if not sell_results_for_notified_buys.empty:
            msg.append("\n【⚠️ 売り・手仕舞い警戒】")
            # 買い銘柄として通知されたものの中から、売りシグナルが出ているものを表示
            # ここでは、該当する銘柄全てを表示するように変更（head(3)は削除）
            for _, row in sell_results_for_notified_buys.iterrows():
                name = name_map.get(row['symbol'], "不明")
                
                # 理由の特定
                reason = "トレンド転換"
                if row["ret1"] < -0.05: reason = f"急落(前日比{row['ret1']:.1%})"
                elif row["RSI"] > 80: reason = f"買われすぎ(RSI:{row['RSI']:.0f})"
                elif row["Close"] < row["SMA25"] * 0.97: reason = f"25日線割れ({row['SMA25']:.1f})"
                elif row["MACD_Hist"] / row["Close"] < -0.003: reason = "勢い低下"
                
                msg.append(f"・{row['symbol']} {name[:8]}\n  価格:{row['Close']:.1f} {reason} (RSI:{row['RSI']:.0f})")
            msg.append("※保有銘柄が含まれる場合は要注意")

            # 売りのシグナルが出た銘柄を監視リストから除外
            sold_symbols = sell_results_for_notified_buys['symbol'].tolist()
            history_df = history_df[~history_df['symbol'].isin(sold_symbols)]

        # 最終的な履歴を保存（新規買いの追加と売り銘柄の削除を反映）
        history_df.to_csv(Config.HISTORY_PATH, index=False)

        full_msg = "\n".join(msg)
        logger.info(f"LINE通知内容:\n{full_msg}")
        send_line(full_msg)
        logger.info("通知完了")

if __name__ == "__main__":
    try:
        StockScreener().run()
    except Exception as e:
        logger.exception("致命的なエラーが発生しました")
        send_line(f"システム停止エラー: {e}")
