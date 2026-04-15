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
# Step0-2　LINE送信関数（環境変数版）
# =========================================================
LINE_ACCESS_TOKEN = os.getenv("LINE_BOT_TOKEN")
LINE_USER_ID = os.getenv("LINE_USER_ID")

if not LINE_ACCESS_TOKEN or not LINE_USER_ID:
    raise ValueError("LINE環境変数が設定されていません")

def send_line(message):

    url = "https://api.line.me/v2/bot/message/push"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LINE_ACCESS_TOKEN}"
    }

    data = {
        "to": LINE_USER_ID,
        "messages": [
            {
                "type": "text",
                "text": message
            }
        ]
    }

    try:
        response = requests.post(url, headers=headers, json=data)

        if response.status_code != 200:
            print("LINE送信失敗:", response.text)

    except Exception as e:
        print("LINE送信エラー:", e)

# =========================================================
# STEP1　パス設定（絶対パス固定）
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

#MODEL_PATH = os.path.join(BASE_DIR, "model_new.pkl")下に書き換えた
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
DB_PATH = os.path.join(BASE_DIR, "market.db")
OLD_MODEL_PATH = os.path.join(BASE_DIR, "model_old.pkl")

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
# Step7-5　model.pkl 自動ダウンロード
# =========================================================
def download_model():

    if os.path.exists("model.pkl"):
        print("model.pkl 既に存在")
        return

    print("model.pkl ダウンロード開始...")

    url = "ここにReleaseのURL"

    import requests

    r = requests.get(url)

    with open("model.pkl", "wb") as f:
        f.write(r.content)

    print("model.pkl ダウンロード完了")

# =========================================================
# Step8　AIモデル読み込み（旧ロジック用）
# =========================================================
def load_ai_model():
    import joblib

    print("AIモデル読み込み中...")

    if os.path.exists(OLD_MODEL_PATH):
        print("✔ model_old.pkl を検出")
        return joblib.load(OLD_MODEL_PATH)

    raise FileNotFoundError("旧モデル（model_old.pkl）が存在しません")


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

    df["Slope10"] = df["Close"].pct_change(10)
    
    # ===== 追加（期待値AI用特徴量） =====
    df["ret3"] = df["Close"].pct_change(3)
    df["ret5"] = df["Close"].pct_change(5)
    df["ret20"] = df["Close"].pct_change(20)

    atr = (df["High"] - df["Low"]).rolling(14).mean()
    df["atr_ratio"] = atr / df["Close"].replace(0, np.nan)
    
    # ===== AI学習ラベル（トレンド特化型：初動＋継続）=====
    future_max = df["High"].shift(-1).rolling(5).max()
    future_close_5 = df["Close"].shift(-5)

    future_return_max = future_max / df["Close"] - 1
    future_return_5 = future_close_5 / df["Close"] - 1

    df["Target"] = np.where(
        future_return_max.notna(),
        (
            (future_return_max > 0.04) &   # 初動 +4%
            (future_return_5 > 0.03)       # 継続 +3%
        ).astype(int),
        np.nan
    )

    feature_cols = [
        "SMA5","SMA25","SMA75",
        "Bias5","Bias25","Bias75",
        "BB_UP1","BB_LOW1","BB_UP2","BB_LOW2",
        "VolRatio",
        "Bull","BigBull","BigBear",
        "Slope10",
        "ret3",
        "ret5",
        "ret20",
        "atr_ratio"
    ]

    df = df.dropna(subset=["SMA75", "BB_STD"])

    return df
    
# =========================================================
# STEP10 特徴量生成ワーカー
# =========================================================
def feature_worker(item):

    symbol, df = item

    if df is None or len(df) < 80:
        return None

    try:
        feat = create_features_fast(df)

        if feat is None:
            return None

        return symbol, feat

    except Exception as e:
        print("FEATURE ERROR:", symbol, e)
        return None
# =========================================================
# Step11　特徴量生成（推論専用・超軽量版）
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

    vol_mean = volume.rolling(25).mean().iloc[-1]
    vol_ratio = volume.iloc[-1] / vol_mean if vol_mean and not np.isnan(vol_mean) else 0

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
        "ret3": close.pct_change(3).iloc[-1] if len(close) >= 3 else 0,
        "ret5": close.pct_change(5).iloc[-1] if len(close) >= 5 else 0,
        "ret20": close.pct_change(20).iloc[-1] if len(close) >= 20 else 0,
        "atr_ratio": (
            ((df["High"] - df["Low"]).rolling(14).mean().iloc[-1]) 
            / close.iloc[-1]
        ) if close.iloc[-1] != 0 else 0,
    }

    return latest

# =========================================================
# Step12　学習処理（安定版・修正版）
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

            print(symbol, "rows:", len(df2) if df2 is not None else 0)

            # データ無し防止
            if df2 is None or len(df2) == 0:
                continue

            # Target列が無い銘柄防止
            if "Target" not in df2.columns:
                continue

            # inf対策
            df2 = df2.replace([np.inf, -np.inf], np.nan)

            # =========================
            # Step12-1：未来データ分を先に削除
            # =========================
            if len(df2) > 5:
                df2 = df2.iloc[:-5]

            # TargetがNaNの行削除
            df2 = df2[df2["Target"].notna()]
            print(symbol, "Target有効件数:", df2["Target"].notna().sum())

            if len(df2) == 0:
                continue

            df2["symbol"] = symbol

            dfs.append(df2)
            used_symbols += 1

        except Exception as e:
            print(f"[FEATURE ERROR] {symbol}: {e}")

    # 学習データ確認
    if len(dfs) == 0:
        print("⚠ 学習データなし → 今回はスキップ")
        return None, None

    print(f"✔ 学習対象銘柄数: {used_symbols}")

    data = pd.concat(dfs, ignore_index=True)

    feature_cols = [
        "SMA5","SMA25","SMA75",
        "Bias5","Bias25","Bias75",
        "BB_UP1","BB_LOW1",
        "BB_UP2","BB_LOW2",
        "VolRatio",
        "Bull","BigBull","BigBear",
        "Slope10",
        "ret3",
        "ret5",
        "ret20",
        "atr_ratio"
    ]

    # =========================
    # Step12-2：欠損はdropnaで除外
    # =========================
    X = data[feature_cols].fillna(0)
    y = data["Target"]

    if "Target" not in data.columns:
        raise RuntimeError("Target列がありません")

    y = data.loc[X.index, "Target"]

    print(f"✔ 学習データ件数: {len(X)}")

    if len(X) == 0:
        raise RuntimeError("学習データが0件です")

    print("RandomForest 学習開始...")

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=3,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X, y)

    print("✔ 学習完了")

    return model, feature_cols


# =========================================================
# Step12-OLD　旧AIモデル学習（旧ロジック専用）
# =========================================================
def train_old_model(all_data):

    print("旧AIモデル学習中...")

    rows = []

    for symbol, df in all_data.items():

        if df is None or len(df) < 120:
            continue

        try:
            close = df["Close"]
            high = df["High"]
            low = df["Low"]
            volume = df["Volume"]

            sma5 = close.rolling(5).mean()
            sma25 = close.rolling(25).mean()
            sma75 = close.rolling(75).mean()

            rsi = calc_rsi(close)
            macd, signal = calc_macd(close)
            adx = calc_adx(df)

            for i in range(80, len(df)-5):

                future = close.iloc[i+5]
                now = close.iloc[i]

                if now <= 0:
                    continue

                ret = (future - now) / now
                target = 1 if ret > 0.03 else 0

                rows.append({
                    "終値": close.iloc[i],
                    "高値": high.iloc[i],
                    "出来高": volume.iloc[i],
                    "RSI": rsi.iloc[i],
                    "MACD": macd.iloc[i],
                    "MACD_signal": signal.iloc[i],
                    "MACD_hist": macd.iloc[i] - signal.iloc[i],
                    "ADX": adx.iloc[i],
                    "SMA5乖離": safe_div(close.iloc[i], sma5.iloc[i]),
                    "SMA25乖離": safe_div(close.iloc[i], sma25.iloc[i]),
                    "SMA75乖離": safe_div(close.iloc[i], sma75.iloc[i]),
                    "出来高比率": safe_div(volume.iloc[i], volume.iloc[i-5:i].mean()),
                    "Target": target
                })

        except Exception:
            continue

    if len(rows) == 0:
        raise RuntimeError("旧モデル用データなし")

    df = pd.DataFrame(rows).dropna()

    X = df.drop("Target", axis=1)
    y = df["Target"]

    print(f"旧モデルデータ件数: {len(X)}")

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X, y)

    joblib.dump(model, OLD_MODEL_PATH)

    print("✔ model_old.pkl 保存完了")

    return model


# =========================================================
# Step13　推論処理（新AI・超高速版）
# =========================================================
def ai_predict(model, feature_cols, all_data, threshold=0.55, top_n=20):

    print("新AI一括推論中...")

    rows = []

    for symbol, feat in all_data.items():

        if feat is None:
            continue

        try:
            row = feat.copy()
            row["symbol"] = symbol
            rows.append(row)

        except Exception as e:
            print(f"[FEATURE ERROR] {symbol}: {e}")

    if not rows:
        print("推論対象なし")
        return []

    df_all = pd.DataFrame(rows)

    # ===== 推論用データ =====
    X = df_all[feature_cols].fillna(0)

    print("predict_proba 一括実行...")
    probs = model.predict_proba(X)[:, 1]

    df_all["prob"] = probs

    # 期待値計算
    df_all["expected_move"] = (
        df_all["ret20"].fillna(0) +
        df_all["atr_ratio"].fillna(0)
    ) / 2

    df_all["risk"] = df_all["atr_ratio"].replace(0, 0.0001)

    df_all["EV"] = (df_all["prob"] * df_all["ret20"]) / df_all["risk"]

    df_all = df_all.sort_values("EV", ascending=False)

    df_filtered = df_all[df_all["prob"] >= threshold]

    print(f"✔ 推論対象: {len(df_all)}銘柄")
    print(f"✔ 閾値 {threshold} 以上: {len(df_filtered)}銘柄")

    return list(zip(df_filtered["symbol"], df_filtered["prob"]))[:top_n]


# =========================================================
# Step14　最強AIランキング（年利最大化）
# =========================================================
def strongest_ai_ranking(model, feature_cols, all_data, feature_data):

    print("最強AIランキング計算中...")

    rows = []

    for symbol, df in all_data.items():

        if df is None or len(df) < 120:
            continue

        try:     
            feat = feature_data.get(symbol)
            if feat is None:
                continue
        except:
            continue
            
        # ===== デバッグ（必要なら使用）=====
        # print("DEBUG keys:", list(feat.keys())[:20])
        
        X = pd.DataFrame([feat])[feature_cols].fillna(0)
        prob = model.predict_proba(X)[0][1]

        high = df["High"]
        low = df["Low"]
        close = df["Close"]

        atr = (high - low).rolling(14).mean().iloc[-1]
        price = close.iloc[-1]

        if price <= 0:
            continue

        ret20 = close.pct_change(20).iloc[-1]
        volatility = atr / price

        expected_move = (ret20 + volatility) / 2

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
# Step15　パラメータ設定
# =========================================================

BEST_TH = 0.40  # 数値を調整する
EXCLUDE_CODES = []


# =========================================================
# Step16　DuckDB差分更新（バッチDL版・最終安定版）
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
    # Step16-1 バッチ取得ループ（高速版）
    # =====================================================
    for i in range(0, len(codes), batch_size):

        batch_codes = codes[i:i+batch_size]
        tickers = " ".join([f"{c}.T" for c in batch_codes])

        print(f"取得中: {i} - {i+len(batch_codes)}")

        # -----------------------------
        # Step16-1-1 Yahooから取得
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
        # Step16-1-2 DataFrame整形
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
        # Step16-1-3 DuckDB一括INSERT
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
# Step17　並列DL用：1銘柄更新関数
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
# Step18　DuckDB一括ロード高速版
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
        AND date >= CURRENT_DATE - INTERVAL 400 DAY
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
    df = df.sort_values(["code","date"])

    for code in df["code"].unique():

        g = df[df["code"] == code]

        g = g.set_index("date")

        g = g[["open","high","low","close","volume"]]
        g.columns = ["Open","High","Low","Close","Volume"]

        all_data[f"{code}.T"] = g

    print(f"ロード完了: {len(all_data)}銘柄")
    import gc
    gc.collect()
    return all_data

# =========================================================
# Step19　銘柄解析（旧ロジック）
# =========================================================
def analyze_symbol(code, name, model, all_data):
    if code in EXCLUDE_CODES:
        return None

    symbol = f"{code}.T"

    try:
        data = all_data[symbol].copy()

        # 必要カラムのNaN除去
        data = data.dropna(subset=["Close","High","Low","Volume"])
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
    # 長期トレンド判定
    c_t = float(close.iloc[-1])  # ← 先に定義
    
    sma75_t = float(sma75.iloc[-1]) if sma75.iloc[-1] != 0 else None
    uptrend = c_t > sma75_t if sma75_t else False

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

    ret1 = (c_t / c_y) - 1
    ret3 = (c_t / float(close.iloc[-4])) - 1

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

    # ★ tuple対策（最重要）
    if isinstance(model, tuple):
        model = model[0]

    ai_prob = model.predict_proba(features)[0][1]
    
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
        
            "ret1": ret1,
            "ret3": ret3,
            "vol_ratio": vol_ratio,
        }

    if (not cond_initial and not cond_continue) or not uptrend:
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
        
        "ret1": ret1,
        "ret3": ret3,
        "vol_ratio": vol_ratio,
    }
    
# =========================================================
# Step20　バックテスト（all_data 再利用）
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
# Step20-2　トレードシミュレーション（追加）
# =========================================================
def simulate_trade(df, entry_index, prob):

    STOP_LOSS = -0.03
    TAKE_PROFIT = 0.02 + (prob * 0.08)

    entry_price = df["Close"].iloc[entry_index]

    for i in range(entry_index + 1, len(df)):

        high = df["High"].iloc[i]
        low = df["Low"].iloc[i]

        # 利確
        if (high - entry_price) / entry_price >= TAKE_PROFIT:
            return TAKE_PROFIT

        # 損切り
        if (low - entry_price) / entry_price <= STOP_LOSS:
            return STOP_LOSS

    # 最後まで行った場合
    exit_price = df["Close"].iloc[-1]
    return (exit_price - entry_price) / entry_price

# =========================================================
# Step21　閾値最適化バックテスト（追加）
# =========================================================
def backtest_threshold(model, feature_cols, all_data, thresholds):

    results = []

    for th in thresholds:

        selected = []

        for symbol, df in all_data.items():

            if df is None or len(df) < 120:
                continue

            try:
                df_feat = create_features(df)

                if len(df_feat) < 10:
                    continue

                # ★未来リーク防止（ここ重要）
                last = df_feat.iloc[-6]

                X = pd.DataFrame([last])[feature_cols].fillna(0)
                prob = model.predict_proba(X)[0][1]

                if prob >= th:
                    selected.append((symbol, df, prob))

            except:
                continue

        # ===== リターン計算 =====
        rets = []

        for symbol, df, prob in selected:

            try:
                ret = simulate_trade(df, -6, prob)
                rets.append(ret)

            except:
                continue

        avg_ret = np.mean(rets) if len(rets) > 0 else 0

        results.append({
            "threshold": th,
            "count": len(rets),
            "avg_return": avg_ret
        })

        print(f"TH={th:.2f} 件数={len(rets)} 平均リターン={avg_ret:.4f}")

    return pd.DataFrame(results)

# =========================================================
# Step22　最強AIランキング（本物の期待値AI）
# =========================================================
def strongest_ai_ranking(model, feature_cols, feature_data):
    
    rows = []

    for symbol, feat in feature_data.items():

        if feat is None:
            continue

        try:

            X = pd.DataFrame([feat])[feature_cols].fillna(0)
            if isinstance(model, tuple):
                model = model[0]
                
            # ===== AI確率 =====
            prob = model.predict_proba(X)[0][1]

            # ===== 爆上げ候補フィルタ（ここから追加） =====
            ret1 = feat.get("ret1", 0)
            ret3 = feat.get("ret3", 0)
            vol_ratio = feat.get("vol_ratio", 1)

            # パターン①：初動ブレイク
            breakout = (ret1 > 0.03 and vol_ratio > 1.5)

            # パターン②：仕込み後ブレイク
            pre_break = (ret3 > 0.05 and ret1 > 0.02)

            # パターン③：AIトレンド強
            #trend = (prob > 0.45 and ret3 > 0.03)　厳しいトレンド用
            trend = (prob > 0.42)

            # 条件外は即除外（←これが一番重要）
            if not (breakout or pre_break or trend):
                continue

            # ===== 崩壊検知フィルタ（ここに追加） =====
            recent_ret5 = feat.get("ret5", 0)
            recent_ret3_check = feat.get("ret3", 0)

            # フィルタ緩和（完全に外す）（上少しきつめ、下ゆるめ）
            # if (recent_ret5 < -0.03) or (recent_ret3 < -0.02):
            # if (recent_ret5 < -0.07) or (recent_ret3 < -0.05):
            #     continue

            # ===== 期待値ロジック修正（未来整合型） =====

            vol = feat.get("atr_ratio", 0)

            # ===== 期待値ロジック修正（未来整合型） =====

            # 最低上昇15%、上昇伸び+10%
            avg_up = 0.015 + (prob * 0.10)

            # ===== 損切り固定（実戦仕様）=====
            avg_down = -0.03  # -3%

            # 期待値
            expectancy = prob * avg_up - (1 - prob) * abs(avg_down)

            # ★ここに追加
            if expectancy <= 0:
                continue
                
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
        return pd.DataFrame(columns=["symbol","AI上昇確率","平均上昇率","平均下落率","期待値"])

    df_rank = df_rank.sort_values("期待値", ascending=False)

    # ==========================================
    # df_rank 空対策（超重要）
    # ==========================================
    if df_rank is None or len(df_rank) == 0:
        print("⚠ df_rank が空 → スキップ")
    
        df_rank = pd.DataFrame(columns=[
            "symbol",
            "AI上昇確率",
            "平均上昇率",
            "平均下落率",
            "期待値"
        ])

    # ===== パターンB（現実向け）：上位5銘柄 =====
    df_rank_top5 = df_rank.head(5).copy()

    # 識別用ラベル付与（任意だが分かりやすくする）
    df_rank_top5["戦略"] = "現実向け_TOP5"

    # ===== 従来（パターンA）はそのまま返す =====
    return df_rank
# =========================================================
# Step23　超高速AIランキングエンジン
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

        returns = df["Close"].pct_change().dropna()

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
        print("⚠ データ0件 → 空DFで返す")
        return pd.DataFrame(columns=[
            "symbol",
            "AI上昇確率",
            "期待上昇率",
            "期待値"
        ])

    df_rank = pd.DataFrame(rows)

    df_rank = df_rank.sort_values(
        "期待値",
        ascending=False
    )

    # ==========================================
    # df_rank 空対策（超重要）
    # ==========================================
    if df_rank is None or len(df_rank) == 0:
        print("⚠ df_rank が空 → スキップ")
    
        df_rank = pd.DataFrame(columns=[
            "symbol",
            "AI上昇確率",
            "期待上昇率",
            "期待値"
        ])
        
    return df_rank    
# =========================================================
# Step24　メイン処理（完全修正版）
# =========================================================
def run_screening():

    # =====================================================
    # Step1 実行時間判定（ここに入れる）
    # =====================================================
    now = datetime.datetime.now()
    hour = now.hour

    if hour < 12:
        RUN_MODE = "OPEN"
    else:
        RUN_MODE = "CLOSE"

    print(f"\n===== 実行モード: {RUN_MODE} =====")
    
    global BEST_TH   # ← これ追加（絶対）
    print("日本株銘柄リストを読み込み中...")
    symbols = load_symbol_list()

    # ★ 市場フィルタ
    symbols = symbols[symbols["市場"].isin(TARGET_MARKETS)]
    # テスト用
    # symbols = symbols.head(200)

    print(f"対象市場: {TARGET_MARKETS}")
    print(f"対象銘柄数: {len(symbols)}")

    # =====================================================
    # Step24-1 データ更新
    # =====================================================
    print("\nDuckDB + yfinance 差分更新...")
    update_duckdb_from_yfinance(symbols, retrain=need_retrain(MODEL_PATH))

    print("\nDuckDBから株価読み込み...")
    all_data = load_all_data_from_duckdb(symbols)

    # ==============================
    # Step24-2 特徴量を事前生成（並列高速化）
    # ==============================
    print("\n特徴量生成（並列処理）...")

    results = Parallel(
        n_jobs=-1,
        backend="loky",
        batch_size=50
    )(
        delayed(feature_worker)(item)
        for item in all_data.items()
    )

    feature_data = {}

    for r in results:

        if r is None:
            continue

        symbol, feat = r
        feature_data[symbol] = feat

    print(f"特徴量生成完了: {len(feature_data)}銘柄")
    
    # =====================================================
    # STEP24-3 新AIモデル準備
    # =====================================================
    
    if need_retrain(MODEL_PATH):
    
        print("\n===== 新AI 学習 =====")
    
        model_new, feature_cols = train_ai_model(all_data)
        
        if model_new is None:
            print("AI学習スキップ → 既存モデルを使用")
    
            if os.path.exists(MODEL_PATH):
                loaded = joblib.load(MODEL_PATH)

                if isinstance(loaded, tuple):
                    model_new, feature_cols = loaded
                else:
                    model_new = loaded
                    feature_cols = [
                        "SMA5","SMA25","SMA75",
                        "Bias5","Bias25","Bias75",
                        "BB_UP1","BB_LOW1","BB_UP2","BB_LOW2",
                        "VolRatio",
                        "Bull","BigBull","BigBear",
                        "Slope10",
                        "ret3","ret5","ret20","atr_ratio"
                    ]
                    
                print("既存モデル読み込み完了")
            else:
                print("❌ モデルが存在しないため処理終了")
                return
        
        joblib.dump((model_new, feature_cols), MODEL_PATH)
    
    else:

        print("\n===== 新AI 読み込み =====")

        # =========================
        # ★ローカルに無ければDL
        # =========================
        if not os.path.exists(MODEL_PATH):

            print("model.pkl が無い → GitHubからDL")

            import requests

            url = "https://github.com/Stock-tsuboi/stock-screener/releases/download/v1.0/model.pkl"

            r = requests.get(url)

            with open(MODEL_PATH, "wb") as f:
                f.write(r.content)

            print("✔ ダウンロード完了")

        loaded = joblib.load(MODEL_PATH)

        if isinstance(loaded, tuple):
            model_new, feature_cols = loaded
        else:
            model_new = loaded
            feature_cols = [
                "SMA5","SMA25","SMA75",
                "Bias5","Bias25","Bias75",
                "BB_UP1","BB_LOW1","BB_UP2","BB_LOW2",
                "VolRatio",
                "Bull","BigBull","BigBear",
                "Slope10",
                "ret3","ret5","ret20","atr_ratio"
            ]

    # ==============================
    # ★ここに追加（これ1回だけ）閾値最適化したいときに実行
    # ==============================
    
    #print("\n===== 閾値最適化バックテスト =====")

    #thresholds = np.arange(0.20, 0.60, 0.05)

    #df_th = backtest_threshold(
        #model_new,
        #feature_cols,
        #all_data,
        #thresholds
    #)

    #best_row = df_th.sort_values("avg_return", ascending=False).iloc[0]
    #BEST_TH = best_row["threshold"]

    #print("\n===== 閾値ランキング =====")
    #print(df_th.sort_values("avg_return", ascending=False))

    #print(f"\n🔥 最適閾値: {BEST_TH:.2f}")
 
    # =====================================================
    # Step24-4 旧ロジック
    # =====================================================
    print("\n===== 旧ロジック（初動→継続）解析中 =====")
    
    # ★旧モデル自動生成対応
    if not os.path.exists(OLD_MODEL_PATH):
        print("旧モデルが無い → 新規作成")
        model_old = train_old_model(all_data)
    else:
        model_old = load_ai_model()
    
    symbol_list = [
    (row["コード"], row["銘柄名"])
    for _, row in symbols.iterrows()
    ]

    # ==============================
    # Step24-5 新AI推論
    # ==============================
    ai_list = ai_predict(
        model_new,
        feature_cols,
        feature_data,
        threshold=BEST_TH,
        top_n=50
    )

    ai_dict = dict(ai_list)

    # =====================================================
    # Step24-6 新AIロジック (旧Step22-5)
    # =====================================================

    symbol_list = [
    (code, name)
    for code, name in symbol_list
    if f"{code}.T" in ai_dict.keys()
    ]
    
    print("\n===== 新AI 上位 =====\n")

    for symbol, prob in ai_list:
        print(f"{symbol}: {prob:.3f}")

    # 銘柄名マッピング作成
    name_map = {
        f"{row['コード']}.T": row["銘柄名"]
        for _, row in symbols.iterrows()
    }

    df_new = pd.DataFrame(ai_list, columns=["symbol","新AI確率"])

    # 銘柄名を付与
    df_new["銘柄名"] = df_new["symbol"].map(name_map)
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
    # Step24-7 新AIモデル　#旧Step22-4
    # =====================================================

    results = Parallel(
        n_jobs=-1,
        backend="loky",
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
    # Step24-8 AI候補銘柄だけに絞る
    # =====================================================
    candidate_symbols = set(ai_dict.keys())

    filtered_all_data = {
        s: all_data[s]
        for s in candidate_symbols
        if s in all_data
    }

    filtered_feature_data = {
        s: feature_data[s]
        for s in candidate_symbols
        if s in feature_data
    }

    # =====================================================
    # Step24-9 最強AI（年利最大化）
    # =====================================================

    # ===== 先に作る =====
    df_rank = strongest_ai_ranking(
        model_new,
        feature_cols,
        filtered_feature_data
    )

    # ★空対策
    if df_rank.empty:
        print("⚠ 最強AI 該当銘柄なし")
        send_line("本日シグナルなし（最強AI）")
        return

    # ===== 表示 =====
    print("\n===== 最強AI（年利最大化ランキング） =====")
    print(df_rank.head(20))

    print("\n===== 最強AI（現実向け_TOP5） =====")
    print(df_rank.head(5))

    # =========================
    # 強制終了ガード（超重要）
    # =========================
    if "AI上昇確率" not in df_rank.columns:
        print("⚠ AI上昇確率カラムなし → 強制停止")
        send_line("エラー：AI上昇確率カラムなし")
        return

    # =========================
    # モード分岐（朝 or 引け）
    # =========================
    if RUN_MODE == "OPEN":
        print("\n>>> 朝モード処理（勝率重視）")

        if df_rank.empty:
            print("⚠ 該当銘柄なし")
            send_line("本日該当銘柄なし")
            return
        
        # ★件数ゼロ防止（重要）
        if "AI上昇確率" in df_rank.columns:
            df_tmp = df_rank[df_rank["AI上昇確率"] > 0.5]
        else:
            print("⚠ フィルタスキップ（列なし）")
            df_tmp = df_rank

        if len(df_tmp) > 0:
            df_rank = df_tmp
        else:
            print("⚠ 朝フィルタで0件 → フィルタ無効化")

    else:
        print("\n>>> 引けモード処理（今まで通り）")

        
    # =====================================================
    # Step24-10 統合ビュー
    # =====================================================
    print("\n===== 統合ビュー（旧 × 新AI）超厳選型 =====")

    df_merge = pd.merge(
        df_old,
        df_new,
        on="symbol",
        how="outer"
    )
    # 最強AIを統合
    df_merge = pd.merge(
        df_merge,
        df_rank,
        on="symbol",
        how="left"
    )
    
    # ===== 銘柄名統合（これ追加）=====
    if "銘柄名_x" in df_merge.columns and "銘柄名_y" in df_merge.columns:
        df_merge["銘柄名"] = df_merge["銘柄名_x"].combine_first(df_merge["銘柄名_y"])
        df_merge = df_merge.drop(columns=["銘柄名_x", "銘柄名_y"])

    df_merge["銘柄名"] = df_merge["銘柄名"].fillna("不明")
    df_merge["旧ロジック判定"] = df_merge["旧ロジック判定"].fillna("該当なし")
    df_merge["旧AI確率"] = df_merge["旧AI確率"].fillna(0)
    df_merge["新AI確率"] = df_merge["新AI確率"].fillna(0)
    df_merge["新AI順位"] = df_merge["新AI順位"].fillna(999).astype(int)

    # ★最強AI（期待値）も補完
    df_merge["期待値"] = df_merge["期待値"].fillna(-999)

    # ★ここ追加（これが本質）
    df_merge = df_merge[df_merge["期待値"] > 0]

    # =========================
    # Step24-10-1 総合スコア作成
    # =========================

    # スコア正規化（0〜1）
    df_merge["score_prob"] = df_merge["新AI確率"]
    df_merge["score_old"] = df_merge["旧AI確率"]
    df_merge["score_ev"] = df_merge["期待値"]

    # 重み付き合計（ここが戦略）
    df_merge["TOTAL_SCORE"] = (
        df_merge["score_prob"] * 0.5 +
        df_merge["score_ev"] * 0.4 +
        df_merge["score_old"] * 0.1
    )
    
    # ★ここが最重要：期待値でソート

    # ★新AIが強い場合は通す（ハイブリッド化）
    df_merge = df_merge[
        (df_merge["旧AI確率"] >= 0.3) |
        (df_merge["新AI確率"] >= 0.4)
    ]
    
    df_merge = df_merge.sort_values(
        "期待値",
        ascending=False
    ).head(50)
    
    #上記の変更が問題なければ削除、一旦保留
    #df_merge = df_merge.sort_values("新AI順位").head(50)
    #print(df_merge.to_string(index=False))
    #df_merge = df_merge.sort_values("新AI順位").head(50)

    # =====================================================
    # Step24-10-2 総合ランキングTOP5
    # =====================================================

    df_total = df_merge.copy()

    # TOTAL_SCOREでランキング
    df_total = df_total.sort_values(
        "TOTAL_SCORE",
        ascending=False
    ).head(5)

    # =========================
    # Step24-11 表示用に整形
    # =========================
    df_view = df_merge.copy()

    # 表示列を厳選
    df_view = df_view[
        ["symbol", "銘柄名", "新AI確率", "旧AI確率", "期待値"]
    ]

    # =========================
    # Step24-12 見やすい固定幅表示
    # =========================
    
    print("\n symbol   銘柄名        新AI   上昇AI   期待値")
    print("-" * 50)

    lines = []

    for _, row in df_view.iterrows():

        line = (
            f"{row['symbol']} "
            f"{str(row['銘柄名'])[:10]} "
            f"{float(row['新AI確率']):.2f} "
            f"{float(row['期待値']):.3f}"
        )

        print(line)
        lines.append(line)
    # =========================
    # 総合ランキング TOP5（最終表示）
    # =========================
    from datetime import datetime

    today = datetime.now().strftime("%Y-%m-%d")
    df_merge.to_csv(f"log_{today}.csv", index=False)
    
    print("\n===== 総合ランキング TOP5 =====\n")

    for i, (_, row) in enumerate(df_total.iterrows(), start=1):
        print(
            f"{i}位 {row['symbol']} "
            f"{str(row['銘柄名'])[:12]} "
            f"確率:{row['新AI確率']:.3f} "
            f"期待値:{row['期待値']:.3f}"
        )

    # =========================
    # LINE送信（総合TOP5）
    # =========================

    lines_total = ["【総合ランキング TOP5】"]

    for i, (_, row) in enumerate(df_total.iterrows(), start=1):

        line = (
            f"{i}位 {row['symbol']} "
            f"{str(row['銘柄名'])[:10]} "
            f"{row['新AI確率']:.2f} "
            f"{row['期待値']:.3f}"
        )

        lines_total.append(line)

    message = "\n".join(lines_total)

    if len(lines_total) <= 1:
        print("⚠ 送信内容なし → 強制メッセージ送信")
        message = "本日シグナルなし"

    send_line(message)

    print("\nLINE送信完了（総合TOP5）")

# =========================================================
# Step25　実行
# =========================================================
if __name__ == "__main__":
    try:
        run_screening()
    except Exception as e:
        print("致命的エラー:", e)
        send_line(f"システムエラー: {str(e)}")





































































