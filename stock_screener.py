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
from sklearn.model_selection import TimeSeriesSplit
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

# yfinanceの内部ログ出力を抑制してエラーログの煩雑さを抑える
logging.getLogger('yfinance').setLevel(logging.CRITICAL)
logging.getLogger('urllib3').setLevel(logging.CRITICAL)

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
    RETRAIN_DAYS = 0
    THRESHOLD_STRICT = 0.40  # 地合いが悪い時
    THRESHOLD_NORMAL = 0.35  # 標準（ログの0.357に合わせて緩和）
    TRAILING_STOP_ATR_MULT = 2.5  # ATRの何倍下落したらトレールストップを発動するか（推奨: 2.0-3.0）
    
    PORTFOLIO_SIZE = 30000    # 運用予算（S株/少額運用 3万円）
    RISK_PER_TRADE = 0.05     # 1トレードの許容損失（資金の5%：3万円なら1500円まで）
    MAX_HOLDING_DAYS = 10     # タイムストップ（10日間動かなければ撤退）
    
    # 財務・マクロ・イベント用設定
    MACRO_TICKERS_JP = {"VXJ": "^JNIV", "JPY": "JPY=X"} # 日経平均ボラティリティ・インデックス, USD/JPY
    FUNDAMENTAL_COLS = ["days_to_earnings"] # 現時点では決算日までの日数のみ


# =========================================================
# Feature Engineering (Unified)
# =========================================================
class FeatureFactory:
    """
    株価データからAIが学習・推論するために必要な「テクニカル指標（特徴量）」を生成するクラスです。
    学習時と推論時で同じ計算ロジックを使用することで、AIの精度低下（計算の乖離）を防ぎます。
    """
    
    FEATURE_COLS = [
        "SMA5", "SMA25", "SMA75", "SMA200", "Bias5", "Bias25", "Bias75", "Bias200",
        "BB_UP1", "BB_LOW1", "BB_UP2", "BB_LOW2", "VolRatio",
        "Bull", "BigBull", "BigBear", "Slope10", "Slope20", "SlopeAccel", "ret10", "RSI", "MACD_Hist", "Momentum_Change",
        "ret1", "ret3", "ret5", "ret20", "atr_ratio", "Stage2_Score",
        "VolVCP",
        "RelativeStrength",
        "Days_To_Earnings", "Macro_VXJ", "Macro_JPY" # 新規追加
    ] # AIが判断に使用する項目のリスト

    @staticmethod
    def calculate_metrics(df: pd.DataFrame, fundamentals: Dict = None, macro_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        移動平均、ボリンジャーバンド、出来高比率、スロープ（傾き）などの
        テクニカル指標を計算します。
        """
        df = df.copy()
        close = df["Close"]
        
        # 移動平均と乖離率 (長期トレンド確認用に200日を追加)
        for n in [5, 25, 75, 200]:
            df[f"SMA{n}"] = close.rolling(n).mean()
            df[f"Bias{n}"] = (close - df[f"SMA{n}"]) / df[f"SMA{n}"].replace(0, np.nan)

        # 上昇トレンドの土台（Stage2）スコアリング
        # 200日線が上向き、かつ価格がその上にあり、短期>中期>長期の順に並んでいるか
        df["is_long_uptrend"] = (close > df["SMA200"]) & (df["SMA200"] > df["SMA200"].shift(1))
        df["is_alignment"] = (df["SMA25"] > df["SMA75"]) & (df["SMA75"] > df["SMA200"])
        df["Stage2_Score"] = (df["is_long_uptrend"].astype(int) + df["is_alignment"].astype(int))

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
        df["VolVCP"] = vol_short / (vol_long + 1e-9)
        # Relative Strength（過去約3か月の強さ）
        # 値が大きいほど最近の株価が強い
        df["RelativeStrength"] = (
            close / close.rolling(63).mean()
        ).replace([np.inf, -np.inf], np.nan)

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
            return slope / (y[-1] + 1e-9) # 価格で割って正規化（％表記）

        df["Slope10"] = close.rolling(10).apply(calc_slope, raw=False)
        df["Slope20"] = close.rolling(20).apply(calc_slope, raw=False)
        df["SlopeAccel"] = df["Slope10"].diff()

        # ATR比率
        tr = pd.concat([
            df["High"] - df["Low"],
            (df["High"] - close.shift()).abs(),
            (df["Low"] - close.shift()).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        df["atr_ratio"] = atr / close.replace(0, np.nan)

        # 市場連動性（β）の簡易代用として日経平均の移動平均乖離などを入れるのが理想ですが
        # ここでは個別銘柄の「直近の勢いの変化」を強調します
        df["Momentum_Change"] = df["ret1"] - df["ret5"] / 5

        # ローソク足
        df["Bull"] = (close > df["Open"]).astype(int)
        df["BigBull"] = ((close - df["Open"]) / df["Open"].replace(0, np.nan) > 0.03).astype(int)
        df["BigBear"] = ((df["Open"] - close) / df["Open"].replace(0, np.nan) > 0.03).astype(int)
        
        # 財務データの統合 (決算発表日までの日数)
        if fundamentals:
            df["Days_To_Earnings"] = fundamentals.get("days_to_earnings", 30)
        else:
            # 学習時など、財務データがない場合はデフォルト値
            df["Days_To_Earnings"] = 30

        # マクロデータの統合
        if macro_df is not None and not macro_df.empty:
            df = df.join(macro_df, how="left").ffill()
        
        # カラムが存在しない、または取得失敗時のデフォルト値補完
        if "Macro_VXJ" not in df.columns: df["Macro_VXJ"] = 20
        if "Macro_JPY" not in df.columns: df["Macro_JPY"] = 150
        
        # 個別銘柄の期間中にマクロデータが欠落している場合を埋める
        df["Macro_VXJ"] = df["Macro_VXJ"].fillna(20)
        df["Macro_JPY"] = df["Macro_JPY"].fillna(150)

        # 無限大をNaNに変換
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # 指標が計算できていない初期の行（SMA75などがNaNの期間）を削除してから、残りを0埋め
        # 新しい特徴量もNaNになりうるので、dropnaのsubsetに追加
        return df.dropna(
            subset=[
                "SMA200",
                "Slope20",
                "RelativeStrength",
                "Days_To_Earnings",
                "Macro_VXJ",
                "Macro_JPY"
            ]
        ).fillna(0).replace([np.inf, -np.inf], 0)

    @staticmethod
    def add_target_label(df: pd.DataFrame) -> pd.DataFrame:
        
        logger.info("★★ add_target_label 実行 ★★")
        """
        学習用：AIに「正解」を教えるためのラベルを作成します。
        「出来高が静かな状態（仕込み時）から5日以内に急騰したケース」のみを正解と定義します。
        これにより、爆上げした後の銘柄ではなく、爆上げ前の予兆を学習させます。
        """
        # 未来の最大上昇ポテンシャル（明日から5日間の高値）
        indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=5)
        future_max = df["High"].shift(-1).rolling(window=indexer, min_periods=5).max()
        future_gain = future_max / df["Close"] - 1
        
        # 未来の最大ドローダウン（明日から5日間の安値）
        future_min = df["Low"].shift(-1).rolling(window=indexer, min_periods=5).min()
        future_drawdown = future_min / df["Close"] - 1

        # 20日（約1ヶ月）先の持続性も確認
        future_20d_gain = df["Close"].shift(-20) / df["Close"] - 1

        # 仕込み時の条件（現在が静かであること）
        # 1. 急騰予兆パターン: 出来高が平均的で価格が安定
        is_precursor = (df["VolRatio"] < 1.3) & (df["ret5"].between(-0.06, 0.03)) & (df["Stage2_Score"] >= 1)
        
        # 2. トレンド継続パターン: すでに動き出しているが、過熱しすぎていない
        is_trend = (df["VolRatio"].between(1.0, 2.2)) & (df["ret5"].between(0.01, 0.10)) & (df["Stage2_Score"] >= 1)
        
        # 未来のパフォーマンス条件
        # A. 期間内最高値が5%以上上昇（利確チャンス）
        will_breakout = (future_gain >= 0.05)
        # B. 20日後も価格が維持または上昇している（長期持続性）
        will_sustain = (future_20d_gain >= 0.02)
        
        # C. 【改善】逆行リスクをATRの1.5倍までに緩和（一律2.5%は厳しすぎた）
        is_clean_move = (future_drawdown > -(df["atr_ratio"] * 1.5).fillna(0.025))

        # いずれかのセットアップ条件を満たし、かつ未来で上昇したものを正解とする
        is_setup = is_precursor | is_trend

        # 両方の条件を満たすものを「質の高い上昇」として学習させる
        df["Target"] = np.where(future_20d_gain.notna(), (is_setup & will_breakout & will_sustain & is_clean_move).astype(int), np.nan)
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

    def update_macro_data_jp(self) -> pd.DataFrame:
        """
        日本株向けのマクロ経済指標（日経平均ボラティリティ・インデックス、USD/JPY）を更新します。
        """
        logger.info("マクロデータの更新中...")
        dfs = []
        
        tickers = {
            "VXJ": ["^JNIV", "^VIX"], # 日経VIがダメなら米国VIXを試す
            "JPY": ["JPY=X"]
        }

        for name, ticker_list in tickers.items():
            d = pd.Series(dtype='float64')
            for ticker in ticker_list:
                data = yf.download(ticker, period="2y", progress=False, timeout=10)
                if not data.empty:
                    d = data["Close"]
                    if isinstance(d, pd.DataFrame):
                        d = d.iloc[:, 0]
                    break # 取得できたら次の項目へ
                else:
                    logger.warning(f"マクロデータ取得失敗: {ticker} はデータがありませんでした。")

            if d.empty:
                continue
            
            # FeatureFactoryが期待する名称に合わせる (例: VXJ -> Macro_VXJ)
            d.name = f"Macro_{name}"
            dfs.append(d)
        
        if not dfs:
            logger.warning("すべてのマクロデータ取得に失敗しました。デフォルト値を使用します。")
            # 全て失敗した場合、ダミーのDataFrameを返す
            return pd.DataFrame(columns=[f"Macro_{k}" for k in Config.MACRO_TICKERS_JP.keys()])

        # 全てのSeriesを結合し、欠損値を前方補完
        macro_df = pd.concat(dfs, axis=1).ffill()
        macro_df.index.name = "date" # インデックス名を 'date' に設定
        return macro_df

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
                min_date, max_date = conn.execute("""
                    SELECT MIN(date), MAX(date)
                    FROM prices
                """).fetchone()
                
                logger.info(f"DB保存期間: {min_date} ～ {max_date}")
                
            except Exception:
                has_data = False
                
            if has_data:
                period_setting = "2y"
                logger.info("データ取得モード：直近2年を再取得")
            else:
                period_setting = "20y"
                logger.info("データ取得モード：初回20年取得")
            
            # 直近データ削除済みかどうか
            delete_done = False
                        
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
                        if df_s.empty:
                            continue
                        
                        # 列名を小文字に統一
                        df_s.columns = [str(c).lower().replace(" ", "_") for c in df_s.columns]
                        
                        # Adj Close がある場合は削除
                        if "adj_close" in df_s.columns:
                            df_s = df_s.drop(columns=["adj_close"])
                        
                        df_s["code"] = code
                        
                        dfs_to_insert.append(
                            df_s[["code", "date", "open", "high", "low", "close", "volume"]]
                        )
                    
                    if dfs_to_insert:
                        merged = pd.concat(dfs_to_insert)
                        logger.info(
                            f"period_setting={period_setting}"
                        )
                    
                        logger.info(
                            f"merged期間: {merged['date'].min()} ～ {merged['date'].max()}"
                        )

                        oldest_date = merged["date"].min()

                        if has_data and not delete_done:
                            delete_count = conn.execute("""
                                SELECT COUNT(*)
                                FROM prices
                                WHERE date >= ?
                            """, [oldest_date]).fetchone()[0]
                            
                            logger.info(
                                f"{oldest_date.date()}以降のデータを削除します（対象: {delete_count:,}件）"
                            )
                        
                            conn.execute("""
                                DELETE FROM prices
                                WHERE date >= ?
                            """, [oldest_date])
                        
                            delete_done = True                    
                    
                        conn.register("tmp_df", merged)
                    
                        conn.execute("""
                            INSERT INTO prices (
                                code,
                                date,
                                open,
                                high,
                                low,
                                close,
                                volume
                            )
                            SELECT
                                code,
                                date,
                                open,
                                high,
                                low,
                                close,
                                volume
                            FROM tmp_df
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
        
        if df.empty:
            return {}

        # code列を確実に文字列型にキャストし、数値推論による先頭ゼロの欠落 (e.g. "0001" -> 1) を防止
        df["code"] = df["code"].astype(str)

        return {f"{code.zfill(4)}.T": group.set_index("date") for code, group in df.groupby("code")}

    def fetch_fundamentals(self, ticker: str) -> Dict:
        """
        財務データ（決算発表日までの日数）を取得します。
        """
        try:
            t = yf.Ticker(ticker)
            
            # 決算日までの日数を取得
            days_to_earnings = 30 # デフォルト値
            calendar = t.calendar
            if calendar is not None:
                next_event_date = None
                # DataFrameで返る場合と辞書で返る場合の両方に対応
                if isinstance(calendar, pd.DataFrame) and not calendar.empty:
                    next_event_date = calendar.iloc[0, 0]
                elif isinstance(calendar, dict):
                    ed = calendar.get('Earnings Date')
                    if ed and isinstance(ed, list) and len(ed) > 0:
                        next_event_date = ed[0]
                
                if isinstance(next_event_date, (pd.Timestamp, datetime)):
                    days_to_earnings = (next_event_date.date() - datetime.now().date()).days
            
            return {"days_to_earnings": days_to_earnings}
        except Exception as e:
            logger.warning(f"財務データ取得失敗 for {ticker}: {e}")
            return {"days_to_earnings": 30} # 取得失敗時はデフォルト値

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
        macro_df = self.db.update_macro_data_jp() # マクロデータ取得
        self.db.update_prices(symbols)
        all_data = self.db.load_all_data(symbols)
        
        # 特徴量生成（並列）と財務データ取得
        processed_data = self._parallel_feature_engineering(all_data, macro_df)
        
        # モデル準備
        if not self._prepare_model(all_data):
            logger.error("モデルの準備に失敗したため、スクリーニングを中断します。")
            send_line("【システム通知】AIモデルの準備に失敗しました。データ量を確認してください。")
            return
        
        # 推論とランキング
        buy_results, sell_results, max_prob = self._inference(processed_data, current_threshold)
        
        # 結果通知
        self._notify((buy_results, sell_results), symbols, is_market_good, max_prob)

    def _load_symbols(self) -> pd.DataFrame:
        """JPXの銘柄リストCSVを読み込み、対象とする市場（プライム等）で絞り込みます。"""
        if not os.path.exists("japan_stocks_jpx.csv"):
            logger.error("japan_stocks_jpx.csv が見つかりません。")
            raise FileNotFoundError("japan_stocks_jpx.csv")
        df = pd.read_csv("japan_stocks_jpx.csv", dtype=str)
        df.columns = df.columns.str.strip()
        df["市場"] = df["市場・商品区分"].str.extract(r"(プライム|スタンダード|グロース)")
        return df[df["市場"].isin(Config.TARGET_MARKETS)][["コード", "銘柄名", "市場"]].dropna()

    def _parallel_feature_engineering(self, all_data: Dict, macro_df: pd.DataFrame) -> Dict:
        """【高速化版】決算日データを事前に一括取得してマージします"""
        logger.info(f"特徴量生成と財務・マクロ指標の統合を開始... (対象: {len(all_data)} 銘柄)")
        
        ticker_list = list(all_data.keys())
        earnings_cache = self._fetch_all_fundamentals_batch(ticker_list)
        
        results = []
        for s, d in all_data.items():
            f_data = earnings_cache.get(s, {"days_to_earnings": 30})
            
            worker_res = self._feature_worker(s, d, f_data, macro_df)
            if worker_res is not None:
                results.append(worker_res)

        processed_data = {r[0]: r[1] for r in results if r is not None}
        logger.info(f"指標の統合が完了しました。処理済み銘柄数: {len(processed_data)}")
        return processed_data

    def _fetch_all_fundamentals_batch(self, tickers: List[str]) -> Dict[str, Dict]:
        """yfinanceの制限を回避しつつ、複数銘柄の決算スケジュールを100件ずつ一括取得するメソッド"""
        logger.info("決算スケジュールのバッチ取得を開始します...")
        earnings_map = {}
        
        batch_size = 100
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i+batch_size]
            
            try:
                yf_tickers = yf.Tickers(" ".join(batch))
                for ticker_name in batch:
                    try:
                        t = yf_tickers.tickers[ticker_name]
                        calendar = t.calendar
                        days_to_earnings = 30
                        
                        if calendar is not None:
                            next_event_date = None
                            if isinstance(calendar, pd.DataFrame) and not calendar.empty:
                                next_event_date = calendar.iloc[0, 0]
                            elif isinstance(calendar, dict):
                                ed = calendar.get('Earnings Date')
                                if ed and isinstance(ed, list) and len(ed) > 0:
                                    next_event_date = ed[0]
                            
                            if isinstance(next_event_date, (pd.Timestamp, datetime)):
                                days_to_earnings = (next_event_date.date() - datetime.now().date()).days
                        
                        earnings_map[ticker_name] = {"days_to_earnings": days_to_earnings}
                    except Exception:
                        earnings_map[ticker_name] = {"days_to_earnings": 30}
                        
                time.sleep(0.5)
                logger.info(f"決算データ取得進捗: {min(i + batch_size, len(tickers))}/{len(tickers)}")
                
            except Exception as e:
                logger.error(f"決算バッチ {i} の取得中にエラーが発生しました: {e}")
                for ticker_name in batch:
                    earnings_map[ticker_name] = {"days_to_earnings": 30}
                    
        return earnings_map

    def _feature_worker(self, symbol, df, fundamentals: Dict = None, macro_df: pd.DataFrame = None):
        """個別の銘柄の特徴量計算ワーカー関数"""
        if len(df) < 80:
            return None
        
        now_jst = datetime.now(timezone.utc) + timedelta(hours=9)
        
        
        # 9:00〜9:15の間は価格が極めて不安定（寄り付きノイズ）なため、当日データが含まれている場合は除外
        if now_jst.hour == 9 and now_jst.minute < 15:
            if df.index[-1].date() == now_jst.date():
                df = df.iloc[:-1]
        # 9:15以降の9時台は、始値を暫定的な終値として扱うことで、寄り付き後の勢いを反映させる
        elif now_jst.hour == 9 and now_jst.minute >= 15 and df.index[-1].date() == now_jst.date():
            df.iloc[-1, df.columns.get_loc("Close")] = df.iloc[-1, df.columns.get_loc("Open")]
        
        feat_df = self.factory.calculate_metrics(df, fundamentals, macro_df)
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
            if len(df) < 120: return None # 過去データが少ない銘柄は学習から除外
            logger.debug(f"Preparing training features for {symbol}")
            # 学習時は財務データとマクロデータは利用しない（またはデフォルト値）
            feat_df = self.factory.calculate_metrics(df, fundamentals=None, macro_df=None)
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
                    est = self.model.calibrated_classifiers_[0].estimator
                    if hasattr(est, "feature_names_in_"):
                        trained_features = list(est.feature_names_in_)
                
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

            logger.info(f"学習対象銘柄数: {len(training_dfs)}")
            
            if not training_dfs:
                logger.error("学習に使用できる有効なデータがありませんでした。")
                return False

            # 全銘柄を日付順にソートすることで、TimeSeriesSplitが「過去から未来」を正しく分割できるようにする
            full_train = pd.concat(training_dfs).sort_index().dropna(subset=["Target"])

            logger.info(f"Target作成後データ件数: {len(full_train):,}")
            
            X = full_train[self.factory.FEATURE_COLS]
            y = full_train["Target"]

            logger.info(f"特徴量作成後データ件数: {len(X):,}")
            logger.info(f"特徴量数: {len(self.factory.FEATURE_COLS)}")

            logger.info(f"学習データの内訳 - Target=1 (上昇): {int(y.sum())}件")
            logger.info(f"学習データの内訳 - Target=0 (その他): {int((y == 0).sum())}件")
            
            logger.info(f"AIモデルの学習を開始します (データ件数: {len(X)})...")
            base_model = RandomForestClassifier(
                n_estimators=300,
                max_depth=12,
                min_samples_leaf=5,
                n_jobs=-1,
                random_state=42,
                class_weight="balanced"
            )
            self.model = CalibratedClassifierCV(
                estimator=base_model,
                method="sigmoid",
                cv=TimeSeriesSplit(n_splits=3) # 時系列を考慮し、検証回数を増やして精度向上
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

    def _inference(self, feature_dict: Dict, threshold: float) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
        """
        最新の指標データに基づいてAIが「上昇確率」を予測します。
        確率が高い銘柄に対し、期待値（EV）を計算してランキングを作成します。
        """
        if not feature_dict:
            logger.error("特徴量データが空です。有効な株価データが不足している可能性があります。")
            return pd.DataFrame(), pd.DataFrame(), 0.0

        # モデルが正常に準備できていない場合のガード
        if self.model is None:
            logger.error("AIモデルが準備できていないため、推論をスキップします。")
            return pd.DataFrame(), pd.DataFrame(), 0.0

        symbols = list(feature_dict.keys())
        features = pd.DataFrame([feature_dict[s] for s in symbols])
        
        # クラス数のチェック（学習データにTarget=1が不在の場合のIndexErrorを防止）
        proba = self.model.predict_proba(features[self.factory.FEATURE_COLS])
        if proba.shape[1] < 2:
            logger.error("AIモデルの学習データに正解（Target=1）が含まれていなかったため、推論をスキップします。")
            return pd.DataFrame(), pd.DataFrame(), 0.0
            
        probs = proba[:, 1]
        
        res_df = pd.DataFrame({"symbol": symbols, "prob": probs})
        res_df = pd.concat([res_df, features.reset_index(drop=True)], axis=1)
        
        # 期待値(EV)計算の刷新：乖離（Bias）を反発エネルギーとして加算
        # SlopeScoreは正規化済みSlope10を使用
        res_df["SlopeScore"] = res_df["Slope10"].clip(-0.01, 0.01) * 100
        
        # --- リスクリワード中心の期待値(EV)計算 ---
        # Risk (想定損失): ATR(14)の2倍を標準的な損切り幅として定義
        res_df["RiskWidth"] = (res_df["atr_ratio"] * 2.0).clip(lower=0.03) # 最低3%は確保
        
        # Reward (想定利益): 確率が低い銘柄には、より大きなリワードがないと期待値がプラスにならないよう調整
        res_df["RewardTarget"] = (0.04 + (res_df["prob"] * 0.14) + (res_df["SlopeScore"] * 0.05)).clip(lower=0.05)

        # 本来の期待値公式: (P_win * Reward) - (P_loss * Risk)
        res_df["EV_Raw"] = (res_df["prob"] * res_df["RewardTarget"]) - ((1.0 - res_df["prob"]) * res_df["RiskWidth"])

        # 出来高確認スコア：1.2倍付近をピークにしつつ、下限を0.9に底上げして過度な除外を防止
        res_df["VolExpansionScore"] = (1.5 - (res_df["VolRatio"] - 1.2).abs()).clip(0.9, 1.5)
        
        # 加速度ボーナス：確率が低い銘柄(0.4未満)の場合、加速度がマイナスなら評価を大幅に下げる
        res_df["AccelBonus"] = np.where(res_df["SlopeAccel"] > 0, 1.1, np.where(res_df["prob"] < 0.4, 0.7, 1.0))
        
        # 長期トレンドボーナス：Stage2（土台ができている）であれば評価を上乗せ
        res_df["SustainabilityBonus"] = 1.0 + (res_df["Stage2_Score"] * 0.1)
        
        res_df["EV"] = res_df["EV_Raw"] * res_df["VolExpansionScore"] * res_df["AccelBonus"] * res_df["SustainabilityBonus"]

        max_prob = res_df['prob'].max()

        # ログの状況（最大確率が閾値以下）に対応するため、閾値を市場の最高値に合わせる動的調整
        adjusted_threshold = min(threshold, max_prob * 0.95) if max_prob > 0.3 else threshold
        if adjusted_threshold < threshold:
            logger.info(f"市場全体の確率が低いため、閾値を {threshold:.3f} -> {adjusted_threshold:.3f} に調整しました。")
        
        logger.info(f"推論完了: {len(res_df)} 銘柄を評価中... (最大確率: {max_prob:.3f}, 動的閾値: {adjusted_threshold:.3f})")

        # 売り・警戒銘柄の検知ロジック
        cond_sell = (
            ((res_df["RSI"] > 80) & (res_df["ret1"] < -0.02)) |  # 超買われすぎからの反落
            (res_df["MACD_Hist"] < 0) |                         # デッドクロス（勢いの低下）
            (res_df["MACD_Hist"] < -res_df["BB_STD"] * 0.2) |   # 勢いの明確な低下
            ((res_df["Close"] < res_df["SMA25"] * 0.97) & (res_df["ret1"] < -0.01)) | # 25日線下抜け
            (res_df["ret1"] < -0.05)                             # 5%以上の急落
        )

        # 基本条件：出来高が極端に細りすぎているものは除外（流動性リスク回避）
        cond_tech = (res_df["VolRatio"] > 0.25) & (res_df["VolVCP"] < 1.5) & (res_df["Bias25"].between(-0.20, 0.12))
        cond_slope = res_df["Slope20"] > -0.01
        cond_prob = (res_df["prob"] >= adjusted_threshold)
        
        logger.info(f"【条件別ヒット数】 AI確率({adjusted_threshold:.2f}以上): {cond_prob.sum()}, テクニカル合致: {(cond_tech & cond_slope).sum()}")
       
        # 売りシグナルフラグを付与
        res_df["is_sell_signal"] = cond_sell & (res_df["Slope10"] < 0.05)

        # AIが非常に高い確率を出している場合、Slope条件を緩和して「下げ止まりからの反発」を拾う
        cond_slope_flexible = (res_df["Slope20"] > -0.015) if max_prob > 0.35 else cond_slope
        
        # ===========================
        # デバッグ（最終条件）
        # ===========================
        high_prob = res_df[cond_prob].copy()
        
        logger.info(f"AI高確率銘柄数: {len(high_prob)}")
        
        if not high_prob.empty:
            high_prob["Tech"] = cond_tech.loc[high_prob.index]
            high_prob["Slope"] = cond_slope_flexible.loc[high_prob.index]
            high_prob["Sell"] = (~cond_sell).loc[high_prob.index]
        
            logger.info(
                "\n" +
                high_prob[
                    ["symbol", "prob", "Tech", "Slope", "Sell", "EV"]
                ].sort_values("prob", ascending=False).to_string(index=False)
            )
        
        mask = cond_prob.values & cond_tech.values & cond_slope_flexible.values & (~cond_sell).values

        logger.info(f"mask True件数 = {mask.sum()}")
        
        filtered = res_df[mask].sort_values("EV", ascending=False)

        if not filtered.empty:
            filtered["is_potential"] = False
            # 銘柄タイプの判定
            filtered["signal_type"] = np.where(
                (filtered["VolRatio"] < 1.2) & (filtered["ret5"] < 0.02),
                "急騰予兆", "トレンド継続"
            )

        # 厳選フィルタで0件の場合の救済ロジック
        if filtered.empty and not res_df.empty:
            logger.info("厳選条件に合致する買い銘柄が0件のため、条件を緩和して潜在候補（Potential）を抽出します。")
            
            # 売りシグナルが出ておらず、最低限の流動性がある上位5銘柄
            cond_potential = (~res_df["is_sell_signal"]) & (res_df["VolRatio"] > 0.1)
            potential_df = res_df[cond_potential].sort_values("prob", ascending=False).head(5)
            
            if not potential_df.empty:
                potential_df["is_potential"] = True
                potential_df["signal_type"] = "潜在候補(要監視)"
                filtered = potential_df
            else:
                filtered = pd.DataFrame()

        # 売り結果のまとめ
        sell_results = res_df[res_df["is_sell_signal"] == True].sort_values("ret1", ascending=True)

        return filtered, sell_results, max_prob

    def _notify(self, results: Tuple[pd.DataFrame, pd.DataFrame], symbols_df: pd.DataFrame, is_market_good: bool, max_prob: float):
        """最終的なランキング結果を整形し、LINEへ送信します。"""
        buy_results, sell_results = results
        name_map = dict(zip(symbols_df["コード"] + ".T", symbols_df["銘柄名"]))
        
        msg = [f"【AI厳選銘柄ランキング】(最大確率: {max_prob:.1%})"]
        if not is_market_good:
            msg.append("（⚠️地合い弱気・厳選モード）")

        is_potential_rescue = not buy_results.empty and buy_results.get("is_potential", pd.Series([False]*len(buy_results))).any()

        if buy_results.empty:
            msg.append("厳選・準候補ともに該当なし")
        elif is_potential_rescue:
            msg.append("厳選基準の該当なし")

        # 準候補（フィルタ落ちだが高確率）がある場合のヘッダー追加
        if is_potential_rescue:
            msg.append("\n【AI準候補・監視推奨】")
            if "summary_reason" in buy_results.columns:
                msg.append(f"（{buy_results['summary_reason'].iloc[0]}のため厳選除外）")

        if not buy_results.empty:
            for i, (_, row) in enumerate(buy_results.iterrows(), 1):
                name = name_map.get(row['symbol'], "不明")

                # ポジションサイジングの計算
                # リスク額 = 総資金 * リスク率（地合いが悪い時は半分に）
                risk_amount = Config.PORTFOLIO_SIZE * (Config.RISK_PER_TRADE if is_market_good else Config.RISK_PER_TRADE * 0.5)
                # 損切り幅 = Close * (atr_ratio * ATR_MULT) ※ボラティリティに応じたリスク許容
                stop_width = row['Close'] * (row['atr_ratio'] * Config.TRAILING_STOP_ATR_MULT)
                # 推奨株数（S株対応：1株単位）
                # 予算内で買える最大数と、リスク管理上の推奨数の小さい方を採用
                recommended_units = max(1, int(min(risk_amount / stop_width, Config.PORTFOLIO_SIZE / row['Close'])))

                # ATRに基づく損切り目安 (2 * ATR)
                stop_loss_price = row['Close'] * (1 - row['atr_ratio'] * 2)
                sig_type = row.get("signal_type", "不明")
                
                msg.append(f"{i}位 【{sig_type}】\n  {row['symbol']} {name[:8]}\n  価格:{row['Close']:.1f} (損切:{stop_loss_price:.1f})\n  S株推奨:{recommended_units}株\n  確率:{row['prob']:.1%} EV:{row['EV']:.2f}")

        # --- 過去の推奨銘柄の管理ロジック ---
        # 実行環境のタイムゾーンに依らず日本時間(JST)で日付を管理します。
        jst = timezone(timedelta(hours=9))
        today_jst = datetime.now(jst).date()

        # 履歴の読み込み（最高値とエントリー価格を管理）
        history_df = pd.DataFrame(columns=["date", "symbol", "highest_price", "entry_price"])
        if os.path.exists(Config.HISTORY_PATH):
            try:
                history_df = pd.read_csv(Config.HISTORY_PATH)
                history_df["date"] = pd.to_datetime(history_df["date"]).dt.date
                # 必要なカラムが不足している場合の補完
                if "highest_price" not in history_df.columns:
                    history_df["highest_price"] = np.nan
                if "entry_price" not in history_df.columns:
                    history_df["entry_price"] = np.nan
            except Exception as e:
                logger.error(f"履歴ファイルの読み込みに失敗しました: {e}")
        
        # 監視対象：本日より前に推奨された銘柄のみを抽出
        sell_results_for_notified_buys = pd.DataFrame()
        if not history_df.empty and not sell_results.empty:
            mask = history_df["date"] < today_jst
            monitored_symbols = history_df.loc[mask, "symbol"].tolist()
            
            # 現在の情報を抽出
            current_info = sell_results[sell_results['symbol'].isin(monitored_symbols)][
                ['symbol', 'Close', 'is_sell_signal', 'RSI', 'ret1', 'SMA25', 'MACD_Hist', 'atr_ratio']
            ]
            
            # 履歴データとマージ
            merged_monitored = history_df[mask].merge(current_info, on='symbol', how='inner')
            
            if not merged_monitored.empty:
                # 最高値の更新
                merged_monitored['highest_price'] = merged_monitored[['highest_price', 'Close']].max(axis=1)
                
                # タイムストップ判定（保有日数の計算）
                merged_monitored['holding_days'] = (today_jst - merged_monitored['date']).apply(lambda x: x.days)
                merged_monitored['is_time_stop'] = merged_monitored['holding_days'] >= Config.MAX_HOLDING_DAYS
                
                # 動的トレールストップ判定（ATRに基づく）
                merged_monitored['dynamic_stop_ratio'] = (merged_monitored['atr_ratio'] * Config.TRAILING_STOP_ATR_MULT).clip(0.04, 0.12)
                merged_monitored['is_trailing_stop'] = merged_monitored['Close'] < merged_monitored['highest_price'] * (1 - merged_monitored['dynamic_stop_ratio'])
                
                # 売り条件の統合（静的シグナル or トレール or タイムストップ）
                merged_monitored['should_sell'] = merged_monitored['is_sell_signal'] | merged_monitored['is_trailing_stop'] | merged_monitored['is_time_stop']
                sell_results_for_notified_buys = merged_monitored[merged_monitored['should_sell']]
                
                # 履歴側の最高値を更新して保存に備える
                for _, row in merged_monitored.iterrows():
                    history_df.loc[(history_df['symbol'] == row['symbol']) & (history_df['date'] == row['date']), 'highest_price'] = row['highest_price']

        # 本日の新規推奨銘柄を履歴に追加
        if not buy_results.empty:
            new_history = pd.DataFrame({
                "date": [today_jst] * len(buy_results), 
                "symbol": buy_results["symbol"].tolist(),
                "highest_price": buy_results["Close"].tolist(),
                "entry_price": buy_results["Close"].tolist()
            })
            history_df = pd.concat([history_df, new_history]).drop_duplicates(subset=["date", "symbol"])

        if not sell_results_for_notified_buys.empty:
            msg.append("\n【⚠️ 売り・手仕舞い警戒】")
            # 買い銘柄として通知されたものの中から、売りシグナルが出ているものを表示
            # ここでは、該当する銘柄全てを表示するように変更（head(3)は削除）
            for _, row in sell_results_for_notified_buys.iterrows():
                name = name_map.get(row['symbol'], "不明")
                
                # 理由の特定
                reason = "トレンド転換"
                if row.get("is_time_stop"): reason = f"タイムストップ({Config.MAX_HOLDING_DAYS}日経過)"
                elif row.get("is_trailing_stop"): reason = f"トレール下落({row['dynamic_stop_ratio']:.1%})"
                elif row["ret1"] < -0.05: reason = f"急落(前日比{row['ret1']:.1%})"
                elif row["RSI"] > 80: reason = f"買われすぎ(RSI:{row['RSI']:.0f})"
                elif row["Close"] < row["SMA25"] * 0.97: reason = f"25日線割れ({row['SMA25']:.1f})"
                elif row["MACD_Hist"] < 0: reason = "勢い低下"
                
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
