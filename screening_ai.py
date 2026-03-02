# =========================================================
# Step0　Import
# =========================================================
import os
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression


# =========================================================
# Step1　設定：対象市場（全市場)
# =========================================================
TARGET_MARKETS = ["プライム", "スタンダード", "グロース"]


# =========================================================
# Step2　安全な割り算
# =========================================================
def safe_div(a, b):
    if b in [0, None] or pd.isna(b):
        return 0
    return a / b


# =========================================================
# Step3　RSI
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
# Step4　MACD
# =========================================================
def calc_macd(close):
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal


# =========================================================
# Step5　ADX
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
# Step6　銘柄リスト読み込み
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
# Step7　AIモデル読み込み（旧ロジック用）
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
# Step8　特徴量生成（新AI用・安定版）
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
# Step9　学習処理（精度最大化版・安定化）
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
# Step10　推論処理（新AI・安定化）
# =========================================================
def ai_predict(model, feature_cols, all_data, threshold=0.55, top_n=20):

    print("新AIスコア計算中...")

    results = []
    total = len(all_data)
    success = 0

    for idx, (symbol, df) in enumerate(all_data.items(), 1):

        if df is None or len(df) < 80:
            continue

        try:
            df2 = create_features(df)
            if df2.empty:
                continue

            latest = df2.iloc[-1]

            X_pred = (
                latest[feature_cols]
                .fillna(0)
                .values
                .reshape(1, -1)
            )

            prob = model.predict_proba(X_pred)[0][1]

            results.append((symbol, prob))
            success += 1

        except Exception as e:
            print(f"[PREDICT ERROR] {symbol}: {e}")

        # 🔥 進捗表示（50銘柄ごと）
        if idx % 50 == 0:
            print(f"進捗: {idx}/{total} 推論成功: {success}")

    print(f"✔ 推論完了: {success}銘柄")

    results.sort(key=lambda x: x[1], reverse=True)
    filtered = [(s, p) for s, p in results if p >= threshold]

    print(f"✔ 閾値 {threshold} 以上: {len(filtered)}銘柄")

    return filtered[:top_n]


# =========================================================
# Step10b　パラメータ設定
# =========================================================
BEST_TH = 0.55
EXCLUDE_CODES = []


# =========================================================
# Step11　本物のDuckDB差分更新（重複完全防止）
# =========================================================

import duckdb
import yfinance as yf
import pandas as pd
from datetime import timedelta

DB_PATH = "market.db"


def update_duckdb_from_yfinance(symbols):

    conn = duckdb.connect(DB_PATH)

    # ① テーブル作成（主キー付き）
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

    print("DuckDB更新開始...")

    for code in symbols["コード"]:
        symbol = f"{code}.T"

        # ② 銘柄ごとの最終日取得
        last_date = conn.execute("""
            SELECT MAX(date)
            FROM prices
            WHERE code = ?
        """, [code]).fetchone()[0]

        if last_date is None:
            start_date = "2020-01-01"
        else:
            start_date = str(last_date + timedelta(days=1))

        print(f"{symbol} → {start_date} 以降取得")

        df = yf.download(symbol, start=start_date, progress=False)

        if df.empty:
            print("  → 更新なし")
            continue

        df = df.reset_index()
        df.columns = [c.lower() for c in df.columns]

        required = {"date", "open", "high", "low", "close", "volume"}
        if not required.issubset(df.columns):
            print("  → 必要列不足スキップ")
            continue

        df["code"] = code
        df = df[["code", "date", "open", "high", "low", "close", "volume"]]

        conn.register("tmp_df", df)

        # ③ 重複を無視してINSERT
        conn.execute("""
            INSERT OR IGNORE INTO prices
            SELECT * FROM tmp_df
        """)

        conn.unregister("tmp_df")

        print(f"  → {len(df)}件処理")

    conn.close()
    print("DuckDB更新完了")


# =========================================================
# Step12　銘柄解析（旧ロジック）
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
# Step13　バックテスト（all_data 再利用）
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
# Step14　メイン処理
# =========================================================
def run_screening():
    print("日本株銘柄リストを読み込み中...")
    symbols = load_symbol_list()

    # ★ グロース市場だけに絞る
    symbols = symbols[symbols["市場"].isin(TARGET_MARKETS)]
    print(f"対象市場: {TARGET_MARKETS}")
    print(f"対象銘柄数: {len(symbols)}")

    api_key = os.getenv("JQ_API_KEY")
    if not api_key:
        raise RuntimeError("環境変数 JQ_API_KEY が設定されていません。")
    headers = {"x-api-key": api_key}

    print("\nDuckDB + yfinance 差分更新...")
    update_duckdb_from_yfinance(symbols)

    print("\nDuckDBから株価読み込み...")
    all_data = load_all_data_from_duckdb(symbols)

    print("\n===== 旧ロジック（初動→継続）解析中 =====")
    model_old = load_ai_model()

    results = Parallel(n_jobs=-1, backend="loky")(
        delayed(analyze_symbol)(row["コード"], row["銘柄名"], model_old, all_data)
        for _, row in symbols.iterrows()
    )

    results = [r for r in results if r is not None]

    print("\n===== 初動→継続（旧ロジック）上位20 =====\n")
    normal_signals = [r for r in results if r["route"] == "normal"]
    if normal_signals:
        df_normal = (
            pd.DataFrame(normal_signals)
            .sort_values("AI上昇確率", ascending=False)
            .head(20)
        )
        print(df_normal.to_string(index=False))
    else:
        print("該当なし")

    print("\n===== AI単独（旧ロジック）上位20 =====\n")
    ai_only_signals = [r for r in results if r["route"] == "ai_only"]
    if ai_only_signals:
        df_ai = (
            pd.DataFrame(ai_only_signals)
            .sort_values("AI上昇確率", ascending=False)
            .head(20)
        )
        print(df_ai.to_string(index=False))
    else:
        print("該当なし")

    if ai_only_signals:
        print("\n===== 旧ロジック AI単独 バックテスト =====")
        codes_old = [r["コード"] + ".T" for r in ai_only_signals]
        backtest_ai_only(codes_old, all_data, days=200)

    print("\n\n===== 新AIロジック（精度最大化AI） =====")
    print("新AIモデル学習中...")

    model_new, feature_cols = train_ai_model(all_data)

    print("新AI推論中...")
    ai_list = ai_predict(model_new, feature_cols, all_data, threshold=0.55, top_n=20)

    print("\n===== 新AI（精度最大化）上位20 =====\n")
    for symbol, prob in ai_list:
        print(f"{symbol}: {prob:.3f}")

    if ai_list:
        print("\n===== 新AI バックテスト =====")
        codes_new = [s for s, p in ai_list]
        backtest_ai_only(codes_new, all_data, days=200)

    print("\n\n===== 統合ビュー（旧ロジック × 新AIロジック） =====")

    df_old = pd.DataFrame(results)

    def convert_route(row):
        if row["route"] == "normal":
            return row.get("タイプ", "初動/継続")
        elif row["route"] == "ai_only":
            return "AI単独（旧）"
        return "該当なし"

    if not df_old.empty:
        df_old["旧ロジック判定"] = df_old.apply(convert_route, axis=1)
        df_old["旧AI確率"] = df_old["AI上昇確率"]
        df_old["symbol"] = df_old["コード"] + ".T"
        df_old = df_old[["symbol", "銘柄名", "旧ロジック判定", "旧AI確率"]]
    else:
        df_old = pd.DataFrame(
            columns=["symbol", "銘柄名", "旧ロジック判定", "旧AI確率"]
        )

    df_new = pd.DataFrame(ai_list, columns=["symbol", "新AI確率"])
    df_new["新AI順位"] = df_new["新AI確率"].rank(ascending=False).astype(int)

    df_merge = pd.merge(df_old, df_new, on="symbol", how="outer")

    df_merge["銘柄名"] = df_merge["銘柄名"].fillna("不明")
    df_merge["旧ロジック判定"] = df_merge["旧ロジック判定"].fillna("該当なし")
    df_merge["旧AI確率"] = df_merge["旧AI確率"].fillna(0)
    df_merge["新AI確率"] = df_merge["新AI確率"].fillna(0)
    df_merge["新AI順位"] = df_merge["新AI順位"].fillna(999).astype(int)

    df_merge = df_merge.sort_values("新AI順位").head(50)

    print(df_merge.to_string(index=False))


# =========================================================
# Step15　実行
# =========================================================
if __name__ == "__main__":
    run_screening()















