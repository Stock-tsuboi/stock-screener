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
class ETFormatter(logging.Formatter):
    """ログのタイムスタンプを米国東部時間(ET)に近い形で出力（あるいはJST）"""
    def formatTime(self, record, datefmt=None):
        # 日本のユーザーを想定し、ログはJSTで表示
        dt = datetime.fromtimestamp(record.created, tz=timezone.utc) + timedelta(hours=9)
        if datefmt:
            return dt.strftime(datefmt)
        else:
            return dt.isoformat(sep=" ", timespec="milliseconds")

warnings.filterwarnings("ignore")

handler = logging.StreamHandler()
handler.setFormatter(ETFormatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S,%f'))
logging.basicConfig(level=logging.INFO, handlers=[handler])
logger = logging.getLogger(__name__)

yf.set_tz_cache_location(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache_us"))

# =========================================================
# Configuration
# =========================================================
class Config:
    """米国株用設定"""
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DB_PATH = os.path.join(BASE_DIR, "us_market.db")
    MODEL_PATH = os.path.join(BASE_DIR, "us_model_v2.pkl")
    HISTORY_PATH = os.path.join(BASE_DIR, "us_recommendation_history.csv")
    LINE_ACCESS_TOKEN = os.getenv("LINE_BOT_TOKEN")
    LINE_USER_ID = os.getenv("LINE_USER_ID") or "dummy"
    
    # 米国株の場合、特定のセクターや大型株（S&P500構成銘柄など）を対象にすることを推奨
    TARGET_EXCHANGES = ["NASDAQ", "NYSE", "AMEX"]
    RETRAIN_DAYS = 7
    THRESHOLD_STRICT = 0.52  # 米国株はボラティリティが高いため少し厳しめに設定
    THRESHOLD_NORMAL = 0.45
    TRAILING_STOP_ATR_MULT = 3.0  # 米国株はノイズが多いためATR倍率を少し広めに
    
    PORTFOLIO_SIZE = 1000  # 運用予算（ドルベース想定）
    RISK_PER_TRADE = 0.05
    MAX_HOLDING_DAYS = 15

    # 財務・マクロ・イベント用設定
    MACRO_TICKERS = {"10Y_Yield": "^TNX", "VIX": "^VIX", "SPY": "SPY"}
    FUNDAMENTAL_COLS = [
        "trailingPE", "priceToBook", "returnOnEquity", "revenueGrowth", 
        "earningsGrowth", "operatingMargins", "debtToEquity", "marketCap"
    ]

# =========================================================
# Feature Engineering (Unified)
# =========================================================
class FeatureFactory:
    """米国株の特性に合わせた特徴量生成"""
    
    FEATURE_COLS = [
        "SMA5", "SMA25", "SMA75", "Bias5", "Bias25", "Bias75",
        "BB_UP1", "BB_LOW1", "BB_UP2", "BB_LOW2", "VolRatio",
        "Bull", "BigBull", "BigBear", "Slope10", "Slope20", "SlopeAccel", "ret10", "RSI", "MACD_Hist",
        "ret1", "ret3", "ret5", "ret20", "atr_ratio",
        "VolVCP",
        # 追加される多角的特徴量
        "PE_Ratio", "PB_Ratio", "ROE", "Rev_Growth", "EPS_Growth", 
        "Op_Margin", "Debt_Equity", "Macro_VIX", "Macro_10Y", "Days_To_Earnings"
    ]

    @staticmethod
    def calculate_metrics(df: pd.DataFrame, fundamentals: Dict = None, macro_df: pd.DataFrame = None) -> pd.DataFrame:
        df = df.copy()
        close = df["Close"]
        
        for n in [5, 25, 75]:
            df[f"SMA{n}"] = close.rolling(n).mean()
            df[f"Bias{n}"] = (close - df[f"SMA{n}"]) / df[f"SMA{n}"].replace(0, np.nan)

        df["BB_MID"] = df["SMA25"]
        df["BB_STD"] = close.rolling(25).std()
        std25 = df["BB_STD"]

        df["BB_UP1"] = df["SMA25"] + std25
        df["BB_LOW1"] = df["SMA25"] - std25
        df["BB_UP2"] = df["SMA25"] + 2 * std25
        df["BB_LOW2"] = df["SMA25"] - 2 * std25

        df["VolRatio"] = df["Volume"] / df["Volume"].rolling(25).mean().replace(0, np.nan)
        for n in [1, 3, 5, 10, 20]:
            df[f"ret{n}"] = close.pct_change(n)

        vol_short = close.pct_change().rolling(10).std()
        vol_long = close.pct_change().rolling(60).std()
        df["VolVCP"] = vol_short / (vol_long + 1e-9)

        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-9)
        df["RSI"] = 100 - (100 / (1 + rs))

        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        df["MACD_Hist"] = (ema12 - ema26) - (ema12 - ema26).ewm(span=9, adjust=False).mean()

        def calc_slope(series):
            if len(series) < 10: return 0
            y = series.values
            x = np.arange(len(y))
            slope = np.polyfit(x, y, 1)[0]
            return slope / (y[-1] + 1e-9)

        df["Slope10"] = close.rolling(10).apply(calc_slope, raw=False)
        df["Slope20"] = close.rolling(20).apply(calc_slope, raw=False)
        df["SlopeAccel"] = df["Slope10"].diff()

        tr = pd.concat([
            df["High"] - df["Low"],
            (df["High"] - close.shift()).abs(),
            (df["Low"] - close.shift()).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        df["atr_ratio"] = atr / close.replace(0, np.nan)

        df["Bull"] = (close > df["Open"]).astype(int)
        df["BigBull"] = ((close - df["Open"]) / df["Open"].replace(0, np.nan) > 0.04).astype(int) # 米国株は4%以上
        df["BigBear"] = ((df["Open"] - close) / df["Open"].replace(0, np.nan) > 0.04).astype(int)

        # 財務データの統合
        if fundamentals:
            df["PE_Ratio"] = fundamentals.get("trailingPE", 0)
            df["PB_Ratio"] = fundamentals.get("priceToBook", 0)
            df["ROE"] = fundamentals.get("returnOnEquity", 0)
            df["Rev_Growth"] = fundamentals.get("revenueGrowth", 0)
            df["EPS_Growth"] = fundamentals.get("earningsGrowth", 0)
            df["Op_Margin"] = fundamentals.get("operatingMargins", 0)
            df["Debt_Equity"] = fundamentals.get("debtToEquity", 0)
            # 決算までの日数（簡易版：直近データから計算）
            df["Days_To_Earnings"] = fundamentals.get("days_to_earnings", 30)
        else:
            for col in ["PE_Ratio", "PB_Ratio", "ROE", "Rev_Growth", "EPS_Growth", "Op_Margin", "Debt_Equity", "Days_To_Earnings"]:
                df[col] = 0

        # マクロデータの統合 (日付でマージ)
        if macro_df is not None:
            df = df.join(macro_df, how="left").fillna(method="ffill")
        else:
            df["Macro_VIX"] = 20
            df["Macro_10Y"] = 4.0

        df = df.replace([np.inf, -np.inf], np.nan)
        return df.dropna(subset=["SMA75", "Slope20"]).fillna(0).replace([np.inf, -np.inf], 0)

    @staticmethod
    def add_target_label(df: pd.DataFrame) -> pd.DataFrame:
        indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=5)
        future_max = df["High"].shift(-1).rolling(window=indexer, min_periods=5).max()
        future_gain = future_max / df["Close"] - 1
        
        future_min = df["Low"].shift(-1).rolling(window=indexer, min_periods=5).min()
        future_drawdown = future_min / df["Close"] - 1

        is_precursor = (df["VolRatio"] < 1.3) & (df["ret5"].between(-0.06, 0.03))
        is_trend = (df["VolRatio"].between(1.2, 2.5)) & (df["ret5"].between(0.02, 0.12)) & (df["Bias25"] < 0.15)
        
        will_breakout = (future_gain >= 0.07) # 米国株は7%上昇をターゲット
        future_close_gain = df["Close"].shift(-5) / df["Close"] - 1
        will_hold = (future_close_gain >= 0.04)
        is_clean_move = (future_drawdown > -(df["atr_ratio"] * 2.0).fillna(0.03))

        is_setup = is_precursor | is_trend
        df["Target"] = np.where(future_gain.notna(), (is_setup & will_breakout & will_hold & is_clean_move).astype(int), np.nan)
        return df

# =========================================================
# Data Management
# =========================================================
class DatabaseManager:
    def __init__(self, db_path: str):
        self.db_path = db_path

    def _get_connection(self):
        return duckdb.connect(self.db_path)

    def update_macro_data(self):
        """マクロ経済指標（金利、VIX）を更新"""
        logger.info("マクロデータの更新中...")
        dfs = []
        for name, ticker in Config.MACRO_TICKERS.items():
            d = yf.download(ticker, period="2y", progress=False)["Close"]
            d = d.rename(f"Macro_{name.split('_')[-1]}")
            dfs.append(d)
        macro_df = pd.concat(dfs, axis=1).fillna(method="ffill")
        
        with self._get_connection() as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS macro (date DATE PRIMARY KEY, VIX DOUBLE, Y10 DOUBLE)")
            # DuckDBへの保存処理（省略可、ここではメモリ保持でも可）
        return macro_df

    def fetch_fundamentals(self, ticker: str) -> Dict:
        """財務データと決算予定の取得（API負荷軽減のためキャッシュ推奨）"""
        try:
            t = yf.Ticker(ticker)
            info = t.info
            # 決算日
            calendar = t.calendar
            days_to_earnings = 30
            if calendar is not None and not calendar.empty:
                next_event = calendar.iloc[0, 0]
                days_to_earnings = (next_event.date() - datetime.now().date()).days
            
            return {
                "trailingPE": info.get("trailingPE", 0),
                "priceToBook": info.get("priceToBook", 0),
                "returnOnEquity": info.get("returnOnEquity", 0),
                "revenueGrowth": info.get("revenueGrowth", 0),
                "earningsGrowth": info.get("earningsGrowth", 0),
                "operatingMargins": info.get("operatingMargins", 0),
                "debtToEquity": info.get("debtToEquity", 0),
                "days_to_earnings": days_to_earnings
            }
        except:
            return {}

    def get_market_regime(self) -> bool:
        """S&P 500で地合い判定"""
        try:
            sp500 = yf.download("^GSPC", period="150d", progress=False)
            if sp500.empty: return True
            close = sp500["Close"]
            if isinstance(close, pd.DataFrame): close = close.iloc[:, 0]
            sma75 = close.rolling(75).mean().iloc[-1]
            return close.iloc[-1] > sma75
        except Exception as e:
            logger.error(f"地合い判定エラー: {e}")
            return True

    def update_prices(self, symbols_df: pd.DataFrame):
        logger.info("DuckDB米国株価格更新開始...")
        tickers_list = symbols_df["Ticker"].tolist()
        
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS prices (
                    code TEXT, date DATE, open DOUBLE, high DOUBLE, 
                    low DOUBLE, close DOUBLE, volume DOUBLE, PRIMARY KEY (code, date)
                )
            """)
            
            res = conn.execute("SELECT COUNT(*) FROM prices").fetchone()
            # 初回は2010年からの長期データを取得して学習精度を向上させる
            period_setting = "5d" if res[0] > 0 else "max" 
            
            batch_size = 50 # 米国株は1ティッカーあたりのデータ量が多いため少し小さめに
            for i in range(0, len(tickers_list), batch_size):
                batch = tickers_list[i:i+batch_size]
                try:
                    # auto_adjust=Trueを適用し、分割併合調整後の価格を取得
                    df = yf.download(
                        batch, 
                        period=period_setting, 
                        group_by="ticker", 
                        auto_adjust=True, 
                        progress=False, 
                        threads=True
                    )
                    if df.empty: continue

                    if not isinstance(df.columns, pd.MultiIndex) and len(batch) == 1:
                        df.columns = pd.MultiIndex.from_product([[batch[0]], df.columns])

                    dfs_to_insert = []
                    for ticker in batch:
                        if ticker not in df.columns.get_level_values(0): continue
                        df_s = df[ticker].dropna()
                        # yfinanceが返すAdj Close列を削除して列数をDBに合わせる
                        if "Adj Close" in df_s.columns:
                            df_s = df_s.drop(columns=["Adj Close"])
                        df_s = df_s.reset_index()
                        if df_s.empty: continue
                        df_s.columns = ["date", "open", "high", "low", "close", "volume"]
                        df_s["code"] = ticker
                        dfs_to_insert.append(df_s[["code", "date", "open", "high", "low", "close", "volume"]])
                    
                    if dfs_to_insert:
                        merged = pd.concat(dfs_to_insert)
                        conn.register("tmp_df", merged)
                        conn.execute("INSERT INTO prices SELECT * FROM tmp_df ON CONFLICT DO NOTHING")
                        conn.unregister("tmp_df")
                    time.sleep(0.5)
                except Exception as e:
                    logger.error(f"Batch processing error at index {i}: {e}")

    def load_all_data(self, symbols_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        tickers = symbols_df["Ticker"].tolist()
        query = "SELECT code, date, open as Open, high as High, low as Low, close as Close, volume as Volume FROM prices WHERE code IN ? AND date >= CURRENT_DATE - INTERVAL 500 DAY ORDER BY code, date"
        with self._get_connection() as conn:
            df = conn.execute(query, [tickers]).df()
        return {ticker: group.set_index("date") for ticker, group in df.groupby("code")} if not df.empty else {}

# =========================================================
# Core Screener Logic
# =========================================================
class USStockScreener:
    def __init__(self):
        self.db = DatabaseManager(Config.DB_PATH)
        self.factory = FeatureFactory()
        self.model = None

    def run(self):
        logger.info("=== 米国株スクリーニング開始 ===")
        symbols = self._load_symbols()
        is_market_good = self.db.get_market_regime()
        current_threshold = Config.THRESHOLD_STRICT if not is_market_good else Config.THRESHOLD_NORMAL
        
        # マクロデータの取得
        macro_df = self.db.update_macro_data()
        
        self.db.update_prices(symbols)
        all_data = self.db.load_all_data(symbols)
        processed_data = self._parallel_feature_engineering(all_data, macro_df)
        
        if not self._prepare_model(all_data): return
        buy_results, sell_results, max_prob = self._inference(processed_data, current_threshold)
        self._notify((buy_results, sell_results), symbols, is_market_good, max_prob)

    def _load_symbols(self) -> pd.DataFrame:
        """
        us_stocks.csv を読み込む。
        例: Ticker, Name, Exchange
        """
        path = os.path.join(Config.BASE_DIR, "us_stocks.csv")
        if not os.path.exists(path):
            # ファイルがない場合は主要なハイテク株をデフォルトにする例
            logger.warning("us_stocks.csv がないため、デフォルトリストを使用します。")
            return pd.DataFrame({
                "Ticker": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX", "AMD", "INTC"],
                "Name": ["Apple", "Microsoft", "Alphabet", "Amazon", "Tesla", "Nvidia", "Meta", "Netflix", "AMD", "Intel"]
            })
        df = pd.read_csv(path, dtype=str)
        df = df.dropna(subset=["Ticker"])
        # 特殊記号を含む銘柄を念のため再度除外
        df = df[df["Ticker"].str.match(r"^[A-Z]+$", na=False)]
        return df

    def _parallel_feature_engineering(self, all_data: Dict, macro_df: pd.DataFrame) -> Dict:
        # 財務データは逐次取得（またはDBキャッシュから）
        logger.info("財務データとテクニカル指標の統合を開始...")
        results = []
        for s, d in all_data.items():
            f_data = self.db.fetch_fundamentals(s)
            results.append(self._feature_worker(s, d, f_data, macro_df))
            # Rate Limit対策: 銘柄数が多い場合は待機を入れる
            if len(all_data) > 100:
                time.sleep(0.5)
            
        return {r[0]: r[1] for r in results if r is not None}

    def _feature_worker(self, symbol, df, fundamentals=None, macro_df=None):
        if len(df) < 100: return None
        
        feat_df = self.factory.calculate_metrics(df, fundamentals, macro_df)
        return (symbol, feat_df.iloc[-1]) if len(feat_df) > 10 else None

    def _prepare_model(self, all_data: Dict) -> bool:
        def train_worker(symbol, df):
            if len(df) < 150: return None
            feat_df = self.factory.calculate_metrics(df)
            feat_df = self.factory.add_target_label(feat_df)
            if len(feat_df) < 50: return None
            cols = self.factory.FEATURE_COLS + ["Target"]
            return feat_df.iloc[:-5][cols]

        need_training = not os.path.exists(Config.MODEL_PATH) or \
                        (datetime.now() - datetime.fromtimestamp(os.path.getmtime(Config.MODEL_PATH))).days >= Config.RETRAIN_DAYS

        if need_training:
            logger.info("米国株モデル再学習中...")
            results = Parallel(n_jobs=2)(delayed(train_worker)(s, d) for s, d in all_data.items())
            training_dfs = [r for r in results if r is not None]
            if not training_dfs: return False

            full_train = pd.concat(training_dfs).sort_index().dropna(subset=["Target"])
            X, y = full_train[self.factory.FEATURE_COLS], full_train["Target"]
            
            base_model = RandomForestClassifier(n_estimators=400, max_depth=14, n_jobs=-1, random_state=42, class_weight="balanced")
            self.model = CalibratedClassifierCV(estimator=base_model, method="sigmoid", cv=TimeSeriesSplit(n_splits=3))
            self.model.fit(X, y)
            joblib.dump(self.model, Config.MODEL_PATH)
        else:
            self.model = joblib.load(Config.MODEL_PATH)
        return self.model is not None

    def _inference(self, feature_dict: Dict, threshold: float) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
        if not feature_dict: return pd.DataFrame(), pd.DataFrame(), 0.0

        symbols = list(feature_dict.keys())
        features = pd.DataFrame([feature_dict[s] for s in symbols])
        probs = self.model.predict_proba(features[self.factory.FEATURE_COLS])[:, 1]
        
        res_df = pd.DataFrame({"symbol": symbols, "prob": probs})
        res_df = pd.concat([res_df, features.reset_index(drop=True)], axis=1)
        
        res_df["RiskWidth"] = (res_df["atr_ratio"] * 2.5).clip(lower=0.04)
        res_df["RewardTarget"] = (0.07 + (res_df["prob"] * 0.15)).clip(lower=0.07)
        res_df["EV_Raw"] = (res_df["prob"] * res_df["RewardTarget"]) - ((1.0 - res_df["prob"]) * res_df["RiskWidth"])
        res_df["EV"] = res_df["EV_Raw"] * (2.0 - res_df["VolRatio"]).clip(0.5, 1.5)
        
        cond_sell = ((res_df["RSI"] > 82) & (res_df["ret1"] < -0.03)) | (res_df["MACD_Hist"] < 0) | (res_df["ret1"] < -0.07)
        cond_tech = (res_df["VolVCP"] < 1.2) & (res_df["Bias25"].between(-0.15, 0.07))
        cond_prob = (res_df["prob"] >= threshold)
        
        res_df["is_sell_signal"] = cond_sell
        filtered = res_df[cond_prob & cond_tech & ~cond_sell].sort_values("EV", ascending=False)

        if filtered.empty and not res_df.empty:
            filtered = res_df[cond_tech & ~cond_sell].sort_values("prob", ascending=False).head(3)
            if not filtered.empty: filtered["is_potential"] = True

        return filtered.head(5), res_df, res_df['prob'].max()

    def _notify(self, results: Tuple[pd.DataFrame, pd.DataFrame], symbols_df: pd.DataFrame, is_market_good: bool, max_prob: float):
        buy_results, sell_results = results
        name_map = dict(zip(symbols_df["Ticker"], symbols_df["Name"]))
        
        msg = [f"【US株AI厳選ランキング】(Max Prob: {max_prob:.1%})"]
        if not is_market_good: msg.append("（⚠️US市場 弱気トレンド）")

        if buy_results.empty:
            msg.append("該当なし")
        else:
            for i, (_, row) in enumerate(buy_results.iterrows(), 1):
                name = name_map.get(row['symbol'], "Unknown")
                stop_loss = row['Close'] * (1 - row['atr_ratio'] * 2.2)
                msg.append(f"{i}位 {row['symbol']} {name[:10]}\n  Price:${row['Close']:.2f} (SL:${stop_loss:.2f})\n  Prob:{row['prob']:.1%} EV:{row['EV']:.2f}")

        # 履歴管理 (JSTで記録)
        jst = timezone(timedelta(hours=9))
        today_jst = datetime.now(jst).date()
        history_df = pd.DataFrame(columns=["date", "symbol", "highest_price", "entry_price"])
        if os.path.exists(Config.HISTORY_PATH):
            history_df = pd.read_csv(Config.HISTORY_PATH)
            history_df["date"] = pd.to_datetime(history_df["date"]).dt.date

        # 売りシグナル・トレールストップ
        if not history_df.empty and not sell_results.empty:
            monitored = history_df[history_df["date"] < today_jst].merge(sell_results, on='symbol', how='inner')
            if not monitored.empty:
                monitored['is_trailing'] = monitored['Close'] < monitored['highest_price'] * (1 - (monitored['atr_ratio'] * 3.0).clip(0.05, 0.15))
                to_sell = monitored[monitored['is_sell_signal'] | monitored['is_trailing']]
                if not to_sell.empty:
                    msg.append("\n【⚠️ US Sell Signal】")
                    for _, s_row in to_sell.iterrows():
                        msg.append(f"・{s_row['symbol']} Price:${s_row['Close']:.2f} (Exit Recommendation)")
                    history_df = history_df[~history_df['symbol'].isin(to_sell['symbol'])]

        if not buy_results.empty:
            new_recs = pd.DataFrame({"date": [today_jst]*len(buy_results), "symbol": buy_results["symbol"].tolist(), 
                                     "highest_price": buy_results["Close"].tolist(), "entry_price": buy_results["Close"].tolist()})
            history_df = pd.concat([history_df, new_recs]).drop_duplicates(subset=["date", "symbol"])
        
        history_df.to_csv(Config.HISTORY_PATH, index=False)
        
        full_msg = "\n".join(msg)
        logger.info(full_msg)
        if Config.LINE_ACCESS_TOKEN:
            requests.post("https://api.line.me/v2/bot/message/push", 
                          headers={"Authorization": f"Bearer {Config.LINE_ACCESS_TOKEN}"},
                          json={"to": Config.LINE_USER_ID, "messages": [{"type": "text", "text": full_msg}]})

if __name__ == "__main__":
    try:
        USStockScreener().run()
    except Exception as e:
        logger.exception("US Screener Error")
