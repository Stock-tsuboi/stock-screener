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
# 設定：対象市場（グロースのみ）
# =========================================================
TARGET_MARKETS = ["グロース"]


# =========================================================
# 安全な割り算
# =========================================================
def safe_div(a, b):
    return a / b if b not in [0, None] else 0


# =========================================================
# RSI
# =========================================================
def calc_rsi(close, period=14):
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.rolling(period).mean()
    ma_down = down.rolling(period).mean()
    rsi = 100 - (100 / (1 + ma_up / ma_down))
    return rsi


# =========================================================
# MACD
# =========================================================
def calc_macd(close):
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal


# =========================================================
# ADX
# =========================================================
def calc_adx(data, period=14):
    high = data["High"]
    low = data["Low"]
    close = data["Close"]

    plus_dm = high.diff()
    minus_dm = low.diff() * -1

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

    atr = tr.rolling(period).mean()
    pdi = 100 * (plus_dm.rolling(period).mean() / atr)
    mdi = 100 * (minus_dm.rolling(period).mean() / atr)
    dx = (abs(pdi - mdi) / (pdi + mdi)) * 100
    adx = dx.rolling(period).mean()
    return adx


# =========================================================
# 銘柄リスト読み込み
# =========================================================
def load_symbol_list():
    df = pd.read_csv("japan_stocks_jpx.csv", dtype={"コード": str})
    df["市場"] = df["市場・商品区分"].str.extract(r"(プライム|スタンダード|グロース)")
    df = df[["コード", "銘柄名", "市場"]].dropna()
    return df


# =========================================================
# AIモデル読み込み（旧ロジック用）
# =========================================================
def load_ai_model():
    import joblib
    import zipfile as zf

    if os.path.exists("model.pkl"):
        return joblib.load("model.pkl")

    if os.path.exists("model_2.zip"):
        with zf.ZipFile("model_2.zip") as z:
            z.extract("model.pkl")
        return joblib.load("model.pkl")

    raise FileNotFoundError("model.pkl / model_2.zip が見つかりません")


# =========================================================
# 特徴量生成（新AI用）
# =========================================================
def create_features(df):
    df = df.copy()

    df["SMA5"] = df["Close"].rolling(5).mean()
    df["SMA25"] = df["Close"].rolling(25).mean()
    df["SMA75"] = df["Close"].rolling(75).mean()

    df["Bias5"] = (df["Close"] - df["SMA5"]) / df["SMA5"]
    df["Bias25"] = (df["Close"] - df["SMA25"]) / df["SMA25"]
    df["Bias75"] = (df["Close"] - df["SMA75"]) / df["SMA75"]

    df["BB_MID"] = df["SMA25"]
    df["BB_STD"] = df["Close"].rolling(25).std()
    df["BB_UP1"] = df["BB_MID"] + df["BB_STD"]
    df["BB_LOW1"] = df["BB_MID"] - df["BB_STD"]
    df["BB_UP2"] = df["BB_MID"] + 2 * df["BB_STD"]
    df["BB_LOW2"] = df["BB_MID"] - 2 * df["BB_STD"]

    df["VolRatio"] = df["Volume"] / df["Volume"].rolling(25).mean()

    df["Bull"] = (df["Close"] > df["Open"]).astype(int)
    df["BigBull"] = ((df["Close"] - df["Open"]) / df["Open"] > 0.03).astype(int)
    df["BigBear"] = ((df["Open"] - df["Close"]) / df["Open"] > 0.03).astype(int)

    def calc_slope(series):
        if len(series) < 10:
            return np.nan
        y = series.values.reshape(-1, 1)
        x = np.arange(len(series)).reshape(-1, 1)
        model = LinearRegression().fit(x, y)
        return model.coef_[0][0]

    df["Slope10"] = df["Close"].rolling(10).apply(calc_slope, raw=False)

    df["Target"] = (df["Close"].shift(-5) / df["Close"] - 1 > 0.03).astype(int)

    df = df.dropna()

    return df


# =========================================================
# 学習処理（精度最大化版）
# =========================================================
def train_ai_model(all_data):
    dfs = []

    for symbol, df in all_data.items():
        if len(df) < 120:
            continue

        df2 = create_features(df)
        df2["symbol"] = symbol
        dfs.append(df2)

    if not dfs:
        raise RuntimeError("学習用データがありません。")

    data = pd.concat(dfs)

    feature_cols = [
        "SMA5",
        "SMA25",
        "SMA75",
        "Bias5",
        "Bias25",
        "Bias75",
        "BB_UP1",
        "BB_LOW1",
        "BB_UP2",
        "BB_LOW2",
        "VolRatio",
        "Bull",
        "BigBull",
        "BigBear",
        "Slope10",
    ]

    X = data[feature_cols].fillna(0)
    y = data["Target"]

    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=3,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X, y)

    return model, feature_cols


# =========================================================
# 推論処理（新AI）
# =========================================================
def ai_predict(model, feature_cols, all_data, threshold=0.55, top_n=20):
    results = []

    for symbol, df in all_data.items():
        df2 = create_features(df)
        if df2.empty:
            continue

        latest = df2.iloc[-1]
        X_pred = latest[feature_cols].fillna(0).values.reshape(1, -1)
        prob = model.predict_proba(X_pred)[0][1]

        results.append((symbol, prob))

    results.sort(key=lambda x: x[1], reverse=True)
    filtered = [(s, p) for s, p in results if p >= threshold]

    return filtered[:top_n]


BEST_TH = 0.55
EXCLUDE_CODES = []


# =========================================================
# bars/daily 単発取得（429完全回避）
# =========================================================
V2_BARS_URL = "https://api.jquants.com/v2/equities/bars/daily"


def fetch_single(code, headers, start, end):
    params = {
        "code": code,
        "from": start.strftime("%Y-%m-%d"),
        "to": end.strftime("%Y-%m-%d"),
    }

    wait_times = [0, 10, 20, 30]

    for wait in wait_times:
        if wait > 0:
            print(f"[429] {code} → {wait}秒待機して再試行")
            time.sleep(wait)

        try:
            r = requests.get(V2_BARS_URL, headers=headers, params=params, timeout=15)

            if r.status_code == 429:
                continue

            if r.status_code == 403:
                print(f"[403] {code} → 権限なし（Freeプラン制限）")
                return None

            if r.status_code != 200:
                print(f"[ERROR] {code} status={r.status_code}")
                return None

            js = r.json()
            rows = js.get("data", [])
            if not rows:
                return None

            df = pd.DataFrame(rows)
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.sort_values("Date").set_index("Date")

            needed = ["Open", "High", "Low", "Close", "Volume"]
            if not all(col in df.columns for col in needed):
                return None

            return df

        except Exception as e:
            print(f"[EXCEPTION] {code}: {e}")
            time.sleep(5)

    print(f"[SKIP] {code} → 3回失敗")
    return None


def download_all_data(symbols, headers):
    end = datetime.today().date() - timedelta(days=1)
    start = end - timedelta(days=150)

    codes = list(symbols["コード"])
    total = len(codes)
    success = 0

    all_data = {}

    print(f"取得対象銘柄数: {total}")

    for idx, code in enumerate(codes, 1):
        df = fetch_single(code, headers, start, end)
        if df is not None:
            all_data[f"{code}.T"] = df
            success += 1

        time.sleep(1.2)

        if idx % 50 == 0:
            print(f"進捗: {idx}/{total} 取得成功: {success}")

    print(f"\n最終取得成功数: {success}/{total}")

    return all_data


# =========================================================
# 銘柄解析（旧ロジック）
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
# バックテスト（all_data 再利用）
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
# メイン処理
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

    print("\n株価データを一括ダウンロード中（bars/daily 単発取得）...")
    all_data = download_all_data(symbols, headers)

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
# 実行
# =========================================================
if __name__ == "__main__":
    run_screening()

