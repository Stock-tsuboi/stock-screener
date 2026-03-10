# =========================================================
# Step0　Import
# =========================================================
import os
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import duckdb
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression

import datetime
import joblib
import warnings
warnings.filterwarnings("ignore")

# =========================================================
# STEP1　パス設定（絶対パス固定）
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model_new.pkl")
DB_PATH = os.path.join(BASE_DIR, "market.db")


def need_retrain(model_path, days=7):
    if not os.path.exists(model_path):
        return True

    mtime = datetime.datetime.fromtimestamp(os.path.getmtime(model_path))
    return (datetime.datetime.now() - mtime).days >= days


# =========================================================
# Step2　設定：対象市場（全市場)
# =========================================================
TARGET_MARKETS = ["プライム", "スタンダード", "グロース"]


# =========================================================
# Step3　安全な割り算
# =========================================================
def safe_div(a, b):
    if b in [0, None] or pd.isna(b):
        return 0
    return a / b


# =========================================================
# Step4　RSI
# =========================================================
def calc_rsi(close, period=14):
    delta = close.diff()

    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)

    ma_up = up.rolling(period).mean()
    ma_down = down.rolling(period).mean()

    rs = ma_up / ma_down.replace(0, np.nan)

    rsi = 100 - (100 / (1 + rs))
    return rsi


# =========================================================
# Step5　MACD
# =========================================================
def calc_macd(close):
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal


# =========================================================
# Step6　ADX
# =========================================================
def calc_adx(data, period=14):
    high = data["High"]
    low = data["Low"]
    close = data["Close"]

    plus_dm = high.diff()
    minus_dm = -low.diff()

    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

    tr = pd.concat(
        [
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = tr.rolling(period).mean().replace(0, np.nan)
    
    pdi = 100 * (plus_dm.rolling(period).mean() / atr)
    mdi = 100 * (minus_dm.rolling(period).mean() / atr)
    
    dx = (abs(pdi - mdi) / (pdi + mdi)).replace(0, np.nan) * 100
    adx = dx.rolling(period).mean()
    
    return adx


# =========================================================
# Step7　銘柄リスト読み込み
# =========================================================
def load_symbol_list():
    print("銘柄CSV読み込み中...")

    df = pd.read_csv("japan_stocks_jpx.csv", dtype=str)

    # 列名の空白除去（事故防止）
    df.columns = df.columns.str.strip()

    required_cols = ["コード", "銘柄名", "市場・商品区分"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"CSVに必要な列がありません: {col}")

    df["市場"] = df["市場・商品区分"].str.extract(
        r"(プライム|スタンダード|グロース)"
    )

    df = df[["コード", "銘柄名", "市場"]].dropna()

    print(f"銘柄数ロード完了: {len(df)}")

    return df


# =========================================================
# Step8　AIモデル読み込み（旧ロジック用）
# =========================================================
def load_ai_model():
    import joblib
    import zipfile as zf

    print("AIモデル読み込み中...")

    if os.path.exists("model.pkl"):
        print("✔ model.pkl を検出")
        return joblib.load("model.pkl")

    if os.path.exists("model_2.zip"):
        print("✔ model_2.zip を検出 → 展開")

        with zf.ZipFile("model_2.zip") as z:
            if "model.pkl" not in z.namelist():
                raise FileNotFoundError("ZIP内に model.pkl が存在しません")

            z.extract("model.pkl")

        print("✔ ZIP展開完了")
        return joblib.load("model.pkl")

    raise FileNotFoundError("model.pkl / model_2.zip が見つかりません")


# =========================================================
# Step9　特徴量生成（新AI用・安定版）
# =========================================================
def create_features(df):
    df = df.copy()

    df["SMA5"] = df["Close"].rolling(5).mean()
    df["SMA25"] = df["Close"].rolling(25).mean()
    df["SMA75"] = df["Close"].rolling(75).mean()

    df["Bias5"] = (df["Close"] - df["SMA5"]) / df["SMA5"].replace(0, np.nan)
    df["Bias25"] = (df["Close"] - df["SMA25"]) / df["SMA25"].replace(0, np.nan)
    df["Bias75"] = (df["Close"] - df["SMA75"]) / df["SMA75"].replace(0, np.nan)

    df["BB_MID"] = df["SMA25"]
    df["BB_STD"] = df["Close"].rolling(25).std()

    df["BB_UP1"] = df["BB_MID"] + df["BB_STD"]
    df["BB_LOW1"] = df["BB_MID"] - df["BB_STD"]
    df["BB_UP2"] = df["BB_MID"] + 2 * df["BB_STD"]
    df["BB_LOW2"] = df["BB_MID"] - 2 * df["BB_STD"]

    df["VolRatio"] = (
        df["Volume"] / df["Volume"].rolling(25).mean().replace(0, np.nan)
    )

    df["Bull"] = (df["Close"] > df["Open"]).astype(int)

    df["BigBull"] = (
        (df["Close"] - df["Open"]) / df["Open"].replace(0, np.nan) > 0.03
    ).astype(int)

    df["BigBear"] = (
        (df["Open"] - df["Close"]) / df["Open"].replace(0, np.nan) > 0.03
    ).astype(int)

    def calc_slope(series):
        if len(series) < 10:
            return np.nan
        y = series.values.reshape(-1, 1)
        x = np.arange(len(series)).reshape(-1, 1)
        model = LinearRegression().fit(x, y)
        return model.coef_[0][0]

    df["Slope10"] = df["Close"].rolling(10).apply(calc_slope, raw=False)

    df["Target"] = (df["Close"].shift(-5) / df["Close"] - 1 > 0.03).astype(int)

    feature_cols = [
        "SMA5","SMA25","SMA75",
        "Bias5","Bias25","Bias75",
        "BB_UP1","BB_LOW1","BB_UP2","BB_LOW2",
        "VolRatio","Bull","BigBull","BigBear",
        "Slope10","Target"
    ]

    df = df.dropna(subset=feature_cols)

    return df

# =========================================================
# Step10　特徴量生成（推論専用・超軽量版）
# =========================================================
def create_features_fast(df):

    df = df.tail(100).copy()  # ← 最新だけ使う

    close = df["Close"]
    volume = df["Volume"]
    open_ = df["Open"]

    sma5 = close.rolling(5).mean()
    sma25 = close.rolling(25).mean()
    sma75 = close.rolling(75).mean()

    bb_mid = sma25
    bb_std = close.rolling(25).std()

    vol_ratio = volume.iloc[-1] / volume.rolling(25).mean().iloc[-1]

    # ---- Slope10 軽量計算 ----
    if len(close) >= 10:
        y = close.iloc[-10:].values
        x = np.arange(10)
        slope10 = np.polyfit(x, y, 1)[0]
    else:
        slope10 = 0

    latest = {
        "SMA5": sma5.iloc[-1],
        "SMA25": sma25.iloc[-1],
        "SMA75": sma75.iloc[-1],
        "Bias5": (close.iloc[-1] - sma5.iloc[-1]) / sma5.iloc[-1] if sma5.iloc[-1] != 0 else 0,
        "Bias25": (close.iloc[-1] - sma25.iloc[-1]) / sma25.iloc[-1] if sma25.iloc[-1] != 0 else 0,
        "Bias75": (close.iloc[-1] - sma75.iloc[-1]) / sma75.iloc[-1] if sma75.iloc[-1] != 0 else 0,
        "BB_UP1": bb_mid.iloc[-1] + bb_std.iloc[-1],
        "BB_LOW1": bb_mid.iloc[-1] - bb_std.iloc[-1],
        "BB_UP2": bb_mid.iloc[-1] + 2 * bb_std.iloc[-1],
        "BB_LOW2": bb_mid.iloc[-1] - 2 * bb_std.iloc[-1],
        "VolRatio": vol_ratio,
        "Bull": int(close.iloc[-1] > open_.iloc[-1]),
        "BigBull": int((close.iloc[-1] - open_.iloc[-1]) / open_.iloc[-1] > 0.03),
        "BigBear": int((open_.iloc[-1] - close.iloc[-1]) / open_.iloc[-1] > 0.03),
        "Slope10": slope10,
    }

    return latest

# =========================================================
# Step11　学習処理（精度最大化版・安定化）
# =========================================================
def train_ai_model(all_data):
    print("AI学習データ生成中...")

    dfs = []
    used_symbols = 0

    for symbol, df in all_data.items():
        if df is None or len(df) < 120:
            continue

        try:
            df2 = create_features(df)
            if df2.empty:
                continue

            df2["symbol"] = symbol
            dfs.append(df2)
            used_symbols += 1

        except Exception as e:
            print(f"[FEATURE ERROR] {symbol}: {e}")

    if not dfs:
        raise RuntimeError("学習用データがありません。")

    print(f"✔ 学習対象銘柄数: {used_symbols}")

    data = pd.concat(dfs, ignore_index=True)

    feature_cols = [
        "SMA5", "SMA25", "SMA75",
        "Bias5", "Bias25", "Bias75",
        "BB_UP1", "BB_LOW1",
        "BB_UP2", "BB_LOW2",
        "VolRatio",
        "Bull", "BigBull", "BigBear",
        "Slope10",
    ]

    print(f"✔ 学習データ件数: {len(data)}")

    X = data[feature_cols].fillna(0)
    y = data["Target"]

    print("RandomForest 学習開始...")

    model = RandomForestClassifier(
        n_estimators=300,          # ← 400→300で高速化
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=3,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X, y)

    print("✔ 学習完了")

    return model, feature_cols


# =========================================================
# Step12　推論処理（新AI・超高速版）
# =========================================================
def ai_predict(model, feature_cols, all_data, threshold=0.55, top_n=20):

    print("新AI一括推論中...")

    rows = []

    for symbol, df in all_data.items():

        if df is None or len(df) < 80:
            continue

        try:
            features = create_features_fast(df)

            features["symbol"] = symbol
            rows.append(features)

        except Exception as e:
            print(f"[FEATURE ERROR] {symbol}: {e}")

    if not rows:
        print("推論対象なし")
        return []

    df_all = pd.DataFrame(rows)

    X = df_all[feature_cols].fillna(0)

    print("predict_proba 一括実行...")
    probs = model.predict_proba(X)[:, 1]

    df_all["prob"] = probs

    df_all = df_all.sort_values("prob", ascending=False)

    df_filtered = df_all[df_all["prob"] >= threshold]

    print(f"✔ 推論対象: {len(df_all)}銘柄")
    print(f"✔ 閾値 {threshold} 以上: {len(df_filtered)}銘柄")

    return list(zip(df_filtered["symbol"], df_filtered["prob"]))[:top_n]
# =========================================================
# Step13　最強AIランキング（年利最大化）
# =========================================================
def strongest_ai_ranking_V1(model, feature_cols, all_data):

    print("最強AIランキング計算中...")

    rows = []

    for symbol, df in all_data.items():

        if df is None or len(df) < 120:
            continue

        try:
            feat = create_features_fast(df)
        except:
            continue

        X = pd.DataFrame([feat])[feature_cols].fillna(0)

        prob = model.predict_proba(X)[0][1]

        high = df["High"]
        low = df["Low"]
        close = df["Close"]

        atr = (high - low).rolling(14).mean().iloc[-1]
        price = close.iloc[-1]

        if price <= 0:
            continue

        expected_move = atr / price

        ev = prob * expected_move
        win = prob
        moon = prob * (expected_move ** 2)

        rows.append(
            {
                "symbol": symbol,
                "EV": ev,
                "WIN": win,
                "MOON": moon
            }
        )

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    df["EV_rank"] = df["EV"].rank(ascending=False)
    df["WIN_rank"] = df["WIN"].rank(ascending=False)
    df["MOON_rank"] = df["MOON"].rank(ascending=False)

    df["TOTAL_SCORE"] = (
        df["EV_rank"]
        + df["WIN_rank"]
        + df["MOON_rank"]
    )

    df = df.sort_values("TOTAL_SCORE")

    return df.head(50)

# =========================================================
# Step14　パラメータ設定
# =========================================================
BEST_TH = 0.55
EXCLUDE_CODES = []


# =========================================================
# Step15　DuckDB差分更新（バッチDL版・最終安定版）
# =========================================================
def update_duckdb_from_yfinance(symbols, retrain=False):

    print("DuckDBバッチ更新開始...")

    import yfinance as yf
    import duckdb

    conn = duckdb.connect(DB_PATH)

    # --- テーブル保証（既存DBを壊さない） ---
    conn.execute("""
    CREATE TABLE IF NOT EXISTS prices (
        code TEXT,
        date DATE,
        open DOUBLE,
        high DOUBLE,
        low DOUBLE,
        close DOUBLE,
        volume DOUBLE,
        PRIMARY KEY (code, date)
    )
    """)

    codes = symbols["コード"].tolist()

    batch_size = 100
    total_inserted = 0

    period_setting = "1y" if retrain else "5d"

    # =====================================================
    # Step15-1 バッチ取得ループ（高速版）
    # =====================================================
    for i in range(0, len(codes), batch_size):

        batch_codes = codes[i:i+batch_size]
        tickers = " ".join([f"{c}.T" for c in batch_codes])

        print(f"取得中: {i} - {i+len(batch_codes)}")

        # -----------------------------
        # Step15-1-1 Yahooから取得
        # -----------------------------
        try:
            df = yf.download(
                tickers,
                period=period_setting,
                group_by="ticker",
                progress=False,
                threads=True
            )
        except Exception as e:
            print(f"⚠ ダウンロード失敗: {e}")
            continue

        if df is None or df.empty:
            continue

        # -----------------------------
        # Step15-1-2 DataFrame整形
        # -----------------------------
        dfs = []

        for code in batch_codes:

            symbol = f"{code}.T"

            if symbol not in df.columns.get_level_values(0):
                continue

            try:
                df_symbol = df[symbol].dropna().reset_index()
            except Exception:
                continue

            if df_symbol.empty:
                continue

            df_symbol.columns = [c.lower() for c in df_symbol.columns]

            df_symbol["code"] = code

            df_symbol = df_symbol[
                ["code","date","open","high","low","close","volume"]
            ]

            dfs.append(df_symbol)

        if not dfs:
            continue

        merged_df = pd.concat(dfs, ignore_index=True)

        # -----------------------------
        # Step15-1-3 DuckDB一括INSERT
        # -----------------------------
        conn.register("tmp_df", merged_df)

        conn.execute("""
            INSERT OR IGNORE INTO prices
            SELECT * FROM tmp_df
        """)

        conn.unregister("tmp_df")

        total_inserted += len(merged_df)

    conn.close()

    print(f"✔ 更新完了 追加件数: {total_inserted}")
    
# =========================================================
# Step16　並列DL用：1銘柄更新関数
# =========================================================
def update_one_symbol(code):

    import duckdb
    import yfinance as yf
    from datetime import timedelta

    conn = duckdb.connect(DB_PATH)

    last_date = conn.execute("""
        SELECT MAX(date)
        FROM prices
        WHERE code = ?
    """, [code]).fetchone()[0]

    if last_date is None:
        start_date = "2020-01-01"
    else:
        start_date = str(last_date + timedelta(days=1))

    symbol = f"{code}.T"

    df = yf.download(symbol, start=start_date, progress=False)

    if df.empty:
        conn.close()
        return 0

    df = df.reset_index()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.columns = [str(c).lower() for c in df.columns]

    required = {"date","open","high","low","close","volume"}
    if not required.issubset(df.columns):
        conn.close()
        return 0

    df["code"] = code
    df = df[["code","date","open","high","low","close","volume"]]

    conn.register("tmp_df", df)

    conn.execute("""
        INSERT OR IGNORE INTO prices
        SELECT * FROM tmp_df
    """)

    conn.unregister("tmp_df")
    conn.close()

    return len(df)

# =========================================================
# Step17　DuckDB一括ロード高速版
# =========================================================
def load_all_data_from_duckdb(symbols):

    conn = duckdb.connect(DB_PATH)

    print("DuckDBから株価一括ロード中...")

    # ① 対象コード取得
    codes = tuple(symbols["コード"].tolist())

    query = f"""
        SELECT code, date, open, high, low, close, volume
        FROM prices
        WHERE code IN {codes}
        ORDER BY code, date
    """

    df = conn.execute(query).df()
    conn.close()

    if df.empty:
        print("データなし")
        return {}

    df["date"] = pd.to_datetime(df["date"])

    all_data = {}

    # ② groupbyで分割
    for code, g in df.groupby("code"):

        g = g.set_index("date")

        g = g[["open","high","low","close","volume"]]
        g.columns = ["Open","High","Low","Close","Volume"]

        all_data[f"{code}.T"] = g

    print(f"ロード完了: {len(all_data)}銘柄")
    import gc
    gc.collect()
    return all_data

# =========================================================
# Step18　銘柄解析（旧ロジック）
# =========================================================
def analyze_symbol(code, name, model, all_data):
    if code in EXCLUDE_CODES:
        return None

    symbol = f"{code}.T"

    try:
        data = all_data[symbol].dropna()
    except KeyError:
        return None

    if len(data) < 50:
        return None

    close = data["Close"]
    high = data["High"]
    low = data["Low"]
    volume = data["Volume"]

    sma5 = close.rolling(5).mean()
    sma25 = close.rolling(25).mean()
    sma30 = close.rolling(30).mean()
    sma75 = close.rolling(75).mean()

    rsi = calc_rsi(close)
    macd, signal = calc_macd(close)
    adx = calc_adx(data)

    c_y = float(close.iloc[-2])
    h_y = float(high.iloc[-2])
    s25_y = float(sma25.iloc[-2])
    s30_y = float(sma30.iloc[-2])
    rsi_y = float(rsi.iloc[-2])
    rsi_yy = float(rsi.iloc[-3])
    macd_y = float(macd.iloc[-2])
    macd_yy = float(macd.iloc[-3])
    macd_yyy = float(macd.iloc[-4])
    ad_y = float(adx.iloc[-2])
    vol_y = float(volume.iloc[-2])
    vol_avg5_y = float(volume.iloc[-7:-2].mean())

    c_t = float(close.iloc[-1])
    h_t = float(high.iloc[-1])
    l_t = float(low.iloc[-1])
    vol_t = float(volume.iloc[-1])
    s25_t = float(sma25.iloc[-1])

    sma25_touch = (s25_y != 0) and abs(c_y - s25_y) / s25_y <= 0.03
    sma30_touch = (s30_y != 0) and abs(c_y - s30_y) / s30_y <= 0.03

    rsi_rebound = (rsi_yy < 30 and rsi_y > 30 and rsi_y > rsi_yy)
    macd_rebound = (macd_yyy > macd_yy < macd_y and macd_y > macd_yy)
    vol_increase = vol_y >= vol_avg5_y * 0.9
    strong_trend = ad_y >= 20

    buy_sma25 = (
        rsi_rebound and sma25_touch and macd_rebound and vol_increase and strong_trend
    )
    buy_sma30 = (
        rsi_rebound and sma30_touch and macd_rebound and vol_increase and strong_trend
    )

    cond_initial = buy_sma25 or buy_sma30

    cont_score = 0
    if c_t > h_y:
        cont_score += 2
    if vol_t == max(volume.iloc[-6:-1]):
        cont_score += 2
    if c_t > c_y:
        cont_score += 1
    if c_t > s25_t:
        cont_score += 1
    if vol_t > vol_y:
        cont_score += 1
    if vol_t > volume.iloc[-6:-1].mean():
        cont_score += 1

    candle_range = h_t - l_t
    if candle_range > 0 and (c_t - l_t) / candle_range > 0.3:
        cont_score += 1
    if c_t > h_y * 0.95:
        cont_score += 1

    cond_continue = cont_score >= 3

    sma5_val = float(sma5.iloc[-1]) if sma5.iloc[-1] != 0 else None
    sma25_val = float(sma25.iloc[-1]) if sma25.iloc[-1] != 0 else None
    sma75_val = float(sma75.iloc[-1]) if sma75.iloc[-1] != 0 else None

    vol_avg = float(volume.iloc[-6:-1].mean())
    vol_ratio = vol_t / vol_avg if vol_avg != 0 else 0

    features = pd.DataFrame(
        [
            {
                "終値": c_t,
                "高値": h_t,
                "出来高": vol_t,
                "RSI": float(rsi.iloc[-1]),
                "MACD": float(macd.iloc[-1]),
                "MACD_signal": float(signal.iloc[-1]),
                "MACD_hist": float(macd.iloc[-1] - signal.iloc[-1]),
                "ADX": float(adx.iloc[-1]),
                "SMA5乖離": safe_div(c_t, sma5_val),
                "SMA25乖離": safe_div(c_t, sma25_val),
                "SMA75乖離": safe_div(c_t, sma75_val),
                "出来高比率": vol_ratio,
            }
        ]
    )

    ai_prob = model.predict_proba(features.values)[0][1]
    
    if ai_prob >= BEST_TH:
        return {
            "route": "ai_only",
            "コード": code,
            "銘柄名": name,
            "終値": c_t,
            "RSI": float(rsi.iloc[-1]),
            "MACD": float(macd.iloc[-1]),
            "ADX": float(adx.iloc[-1]),
            "出来高": vol_t,
            "継続スコア": cont_score,
            "AI上昇確率": round(ai_prob, 4),
        }

    if not cond_initial and not cond_continue:
        return None

    if cond_initial and cond_continue:
        signal_type = "初動→継続"
    elif cond_initial:
        signal_type = "初動"
    else:
        signal_type = "継続"

    return {
        "route": "normal",
        "コード": code,
        "銘柄名": name,
        "タイプ": signal_type,
        "終値": c_t,
        "RSI": rsi_y,
        "MACD": macd_y,
        "ADX": ad_y,
        "出来高": vol_t,
        "継続スコア": cont_score,
        "AI上昇確率": round(ai_prob, 4),
    }


# =========================================================
# Step19　バックテスト（all_data 再利用）
# =========================================================
def backtest_ai_only(ai_list, all_data, days=200):
    rets = []

    for symbol in ai_list:
        if symbol not in all_data:
            continue

        df = all_data[symbol].sort_index()
        if len(df) < 10:
            continue

        end_date = df.index.max()
        start_date = end_date - timedelta(days=days)

        df_win = df[df.index >= start_date]
        if len(df_win) < 5:
            continue

        start_price = df_win["Close"].iloc[0]
        end_price = df_win["Close"].iloc[-1]
        if start_price <= 0:
            continue

        ret = (end_price - start_price) / start_price
        rets.append(ret)

    if not rets:
        print("バックテスト結果：該当なし")
        return

    avg_return = sum(rets) / len(rets)
    print(f"バックテスト銘柄数：{len(rets)}")
    print(f"平均リターン：{avg_return*100:.2f}%")
    print()

# =========================================================
# Step20　最強AIランキング（本物の期待値AI）
# =========================================================
def strongest_ai_ranking(model, feature_cols, all_data):
    
    rows = []

    for symbol, df in all_data.items():

        if df is None or len(df) < 120:
            continue

        try:

            df = create_features(df)

            if df.empty:
                continue

            last = df.iloc[-1]

            feat = last.to_dict()

            X = pd.DataFrame([feat]).reindex(columns=feature_cols).fillna(0)

            prob = model.predict_proba(X)[0][1]

            # -------------------------
            # STEP20-1 過去リターン計算
            # -------------------------
            future_returns = (
                df["Close"].shift(-5) / df["Close"] - 1
            )

            avg_up = future_returns[future_returns > 0].mean()
            avg_down = future_returns[future_returns < 0].mean()

            if pd.isna(avg_up):
                avg_up = 0

            if pd.isna(avg_down):
                avg_down = 0

            # -------------------------
            # STEP20-2 期待値
            # -------------------------
            expectancy = prob * avg_up + (1 - prob) * avg_down

            rows.append({
                "symbol": symbol,
                "AI上昇確率": prob,
                "平均上昇率": avg_up,
                "平均下落率": avg_down,
                "期待値": expectancy
            })

        except Exception:
            continue

    df_rank = pd.DataFrame(rows)

    if df_rank.empty:
        return df_rank

    df_rank = df_rank.sort_values(
        "期待値",
        ascending=False
    )

    return df_rank
# =========================================================
# Step21　超高速AIランキングエンジン
# =========================================================
def fastest_ai_ranking(model, feature_cols, all_data):

    import numpy as np
    import pandas as pd
    import warnings

    warnings.filterwarnings("ignore")

    rows = []

    for symbol, df in all_data.items():

        # データ不足スキップ
        if len(df) < 120:
            continue

        # 特徴量生成
        df_feat = create_features(df)

        if len(df_feat) == 0:
            continue

        # 最新行
        last = df_feat.iloc[-1]

        try:
            X = pd.DataFrame([last])[feature_cols].fillna(0)

            prob = model.predict_proba(X)[0][1]

        except:
            continue

        # ===== 実データベース期待値 =====

        returns = df["close"].pct_change().dropna()

        if len(returns) < 20:
            continue

        vol = returns.std()

        expected_move = vol * 3

        expected_value = prob * expected_move

        rows.append({
            "symbol": symbol,
            "AI上昇確率": prob,
            "期待上昇率": expected_move,
            "期待値": expected_value
        })

    if len(rows) == 0:
        return pd.DataFrame()

    df_rank = pd.DataFrame(rows)

    df_rank = df_rank.sort_values(
        "期待値",
        ascending=False
    )

    return df_rank    
# =========================================================
# Step22　メイン処理（完全修正版）
# =========================================================
def run_screening():

    print("日本株銘柄リストを読み込み中...")
    symbols = load_symbol_list()

    # ★ 市場フィルタ
    symbols = symbols[symbols["市場"].isin(TARGET_MARKETS)]
    # テスト用
    # symbols = symbols.head(200)

    print(f"対象市場: {TARGET_MARKETS}")
    print(f"対象銘柄数: {len(symbols)}")

    # =====================================================
    # Step22-1 データ更新
    # =====================================================
    print("\nDuckDB + yfinance 差分更新...")
    update_duckdb_from_yfinance(symbols, retrain=need_retrain(MODEL_PATH))

    print("\nDuckDBから株価読み込み...")
    all_data = load_all_data_from_duckdb(symbols)
    
    # =====================================================
    # STEP22-2 新AIモデル準備
    # =====================================================
    
    if need_retrain(MODEL_PATH):
    
        print("\n===== 新AI 学習 =====")
    
        model_new, feature_cols = train_ai_model(all_data)
    
        joblib.dump((model_new, feature_cols), MODEL_PATH)
    
    else:
    
        print("\n===== 新AI 読み込み =====")
    
        model_new, feature_cols = joblib.load(MODEL_PATH)

    # =====================================================
    # Step22-3 旧ロジック
    # =====================================================
    print("\n===== 旧ロジック（初動→継続）解析中 =====")

    model_old = load_ai_model()
    
    # =====================================================
    # Step22-4 新AIモデル
    # =====================================================
    print("\n===== 新AIモデル準備 =====")
    
    if need_retrain(MODEL_PATH):
    
        print("AI再学習開始...")
    
        model_new, feature_cols = train_ai_model(all_data)
    
        joblib.dump((model_new, feature_cols), MODEL_PATH)
    
        print("✔ 新AIモデル保存")
    
    else:
    
        print("保存モデル読み込み")
    
        model_new, feature_cols = joblib.load(MODEL_PATH)
        
    print("旧ロジック解析開始...")

    symbol_list = [
        (row["コード"], row["銘柄名"])
        for _, row in symbols.iterrows()
    ]

    results = Parallel(
        n_jobs=-1,
        backend="threading",
        batch_size=50,
        prefer="threads"
    )(
        delayed(analyze_symbol)(code, name, model_old, all_data)
        for code, name in symbol_list
    )

    results = [r for r in results if r is not None]

    df_old = pd.DataFrame(results)

    if not df_old.empty:
        df_old["旧ロジック判定"] = df_old["route"]
        df_old["旧AI確率"] = df_old["AI上昇確率"]
        df_old["symbol"] = df_old["コード"] + ".T"

        df_old = df_old[
            ["symbol", "銘柄名", "旧ロジック判定", "旧AI確率"]
        ]

    else:
        df_old = pd.DataFrame(
            columns=["symbol","銘柄名","旧ロジック判定","旧AI確率"]
        )

    # =====================================================
    # Step22-5 新AIロジック
    # =====================================================
    print("\n===== 新AIロジック（精度最大化AI） =====")
    print("新AIモデル確認中...")

    if need_retrain(MODEL_PATH, days=7):

        print("🔄 週次再学習を実行")

        model_new, feature_cols = train_ai_model(all_data)

        joblib.dump((model_new, feature_cols), MODEL_PATH)

    else:

        print("📦 既存モデルを使用")

        model_new, feature_cols = joblib.load(MODEL_PATH)

    print("新AI推論中...")

    ai_list = ai_predict(
        model_new,
        feature_cols,
        all_data,
        threshold=0.0,
        top_n=50
    )

    print("\n===== 新AI 上位 =====\n")

    for symbol, prob in ai_list:
        print(f"{symbol}: {prob:.3f}")

    df_new = pd.DataFrame(ai_list, columns=["symbol","新AI確率"])

    if not df_new.empty:

        df_new["新AI順位"] = (
            df_new["新AI確率"]
            .rank(ascending=False, method="min")
            .astype(int)
        )

    else:

        df_new = pd.DataFrame(
            columns=["symbol","新AI確率","新AI順位"]
        )
    # =====================================================
    # Step22-6 最強AI（年利最大化）
    # =====================================================
    print("\n===== 最強AI（年利最大化ランキング） =====")
    print("===== 超高速AIランキング =====")
        
    df_strong = strongest_ai_ranking(
        model_new,
        feature_cols,
        all_data
    )
    
    print(df_strong.head(20))
    # symbol列を作る
    df_strong["symbol"] = df_strong["symbol"]
    
    # =====================================================
    # Step22-7 統合ビュー
    # =====================================================
    print("\n===== 統合ビュー（旧 × 新AI） =====")

    df_merge = pd.merge(
        df_old,
        df_new,
        on="symbol",
        how="outer"
    )
    # 最強AIを統合
    df_merge = pd.merge(
        df_merge,
        df_strong,
        on="symbol",
        how="left"
    )
    

    df_merge["銘柄名"] = df_merge["銘柄名"].fillna("不明")
    df_merge["旧ロジック判定"] = df_merge["旧ロジック判定"].fillna("該当なし")
    df_merge["旧AI確率"] = df_merge["旧AI確率"].fillna(0)
    df_merge["新AI確率"] = df_merge["新AI確率"].fillna(0)
    df_merge["新AI順位"] = df_merge["新AI順位"].fillna(999).astype(int)

    df_merge = df_merge.sort_values("新AI順位").head(50)

    print(df_merge.to_string(index=False))

    df_merge = df_merge.sort_values("新AI順位").head(50)

    print(df_merge.to_string(index=False))

# =========================================================
# Step23　実行
# =========================================================
if __name__ == "__main__":
    run_screening()


















































