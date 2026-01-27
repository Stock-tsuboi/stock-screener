import pandas as pd
import numpy as np
from joblib import Parallel, delayed
import time
import os
import requests
from datetime import datetime, timedelta
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

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)

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
# AIモデル読み込み
# =========================================================
def load_ai_model():
    import os
    import joblib
    import zipfile

    # ① すでに展開済みなら model.pkl をそのまま読む
    if os.path.exists("model.pkl"):
        return joblib.load("model.pkl")

    # ② model_2.zip から model.pkl を展開して読む
    if os.path.exists("model_2.zip"):
        with zipfile.ZipFile("model_2.zip") as z:
            # 中に model.pkl が入っている前提
            z.extract("model.pkl")
        return joblib.load("model.pkl")

    # ③ どちらも無ければ明示的に落とす
    raise FileNotFoundError("model.pkl / model_2.zip が見つかりません")

BEST_TH = 0.55
EXCLUDE_CODES = []

# =========================================================
# ★ JPX（J-Quants）API で全銘柄を取得
# =========================================================
def download_all_data(symbols):
    api_key = os.getenv("JQ_API_KEY")
    if not api_key:
        raise RuntimeError("環境変数 JQ_API_KEY が設定されていません。")

    headers = {
        "Authorization": f"Bearer {api_key}"
    }

    # 未来日付を避けるため「昨日」まで
    end = datetime.today().date() - timedelta(days=1)
    start = end - timedelta(days=90)  # 3ヶ月で十分

    base_url = "https://api.jquants.com/v1/prices/daily_quotes"

    all_data = {}

    for code in symbols["コード"]:
        print(f"Downloading: {code}")  # ← どこで止まるか分かる

        params = {
            "code": code,
            "from": start.strftime("%Y-%m-%d"),
            "to": end.strftime("%Y-%m-%d"),
        }

        try:
            r = requests.get(base_url, headers=headers, params=params, timeout=2)
        except:
            continue

        if r.status_code != 200:
            continue

        js = r.json()
        rows = js.get("daily_quotes", [])
        if not rows:
            continue

        df = pd.DataFrame(rows)

        # J-Quants のキー名に合わせて整形
        rename_map = {
            "Date": "Date",
            "Open": "Open",
            "High": "High",
            "Low": "Low",
            "Close": "Close",
            "Volume": "Volume",
        }
        df = df.rename(columns=rename_map)

        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").set_index("Date")

        all_data[f"{code}.T"] = df

    return all_data

# =========================================================
# 銘柄解析（高速版）
# =========================================================
def analyze_symbol(code, name, model, all_data):
    if code in EXCLUDE_CODES:
        return None

    symbol = f"{code}.T"

    # 一括データから抽出
    try:
        data = all_data[symbol].dropna()
    except:
        return None

    if data is None or len(data) < 50:
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

    # 昨日
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

    # 今日
    c_t = float(close.iloc[-1])
    h_t = float(high.iloc[-1])
    l_t = float(low.iloc[-1])
    vol_t = float(volume.iloc[-1])
    s25_t = float(sma25.iloc[-1])

    # 初動判定
    sma25_touch = (s25_y != 0) and abs(c_y - s25_y) / s25_y <= 0.03
    sma30_touch = (s30_y != 0) and abs(c_y - s30_y) / s30_y <= 0.03

    rsi_rebound = (rsi_yy < 30 and rsi_y > 30 and rsi_y > rsi_yy)
    macd_rebound = (macd_yyy > macd_yy < macd_y and macd_y > macd_yy)
    vol_increase = vol_y >= vol_avg5_y * 0.9
    strong_trend = ad_y >= 20

    buy_sma25 = rsi_rebound and sma25_touch and macd_rebound and vol_increase and strong_trend
    buy_sma30 = rsi_rebound and sma30_touch and macd_rebound and vol_increase and strong_trend

    # 継続スコア
    cont_score = 0
    if c_t > h_y: cont_score += 2
    if vol_t == max(volume.iloc[-6:-1]): cont_score += 2
    if c_t > c_y: cont_score += 1
    if c_t > s25_t: cont_score += 1
    if vol_t > vol_y: cont_score += 1
    if vol_t > volume.iloc[-6:-1].mean(): cont_score += 1
    candle_range = h_t - l_t
    if candle_range > 0 and (c_t - l_t) / candle_range > 0.3:
        cont_score += 1
    if c_t > h_y * 0.95:
        cont_score += 1

    # AI予測
    sma5_val = float(sma5.iloc[-1]) if sma5.iloc[-1] != 0 else None
    sma25_val = float(sma25.iloc[-1]) if sma25.iloc[-1] != 0 else None
    sma75_val = float(sma75.iloc[-1]) if sma75.iloc[-1] != 0 else None

    vol_avg = float(volume.iloc[-6:-1].mean())
    vol_ratio = vol_t / vol_avg if vol_avg != 0 else 0

    features = pd.DataFrame([{
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
        "出来高比率": vol_ratio
    }])

    ai_prob = model.predict_proba(features)[0][1]

    # AI単独
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
            "AI上昇確率": round(ai_prob, 4)
        }

    # 初動＋継続
    if not (buy_sma25 or buy_sma30):
        return None
    if cont_score < 3:
        return None

    signal_type = "浅押し（SMA25）" if buy_sma25 else "深押し（SMA30）"

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
        "AI上昇確率": round(ai_prob, 4)
    }

# =========================================================
# バックテスト（JPX API 版）
# =========================================================
def backtest_ai_only(ai_list):
    import os
    import requests
    from datetime import datetime, timedelta

    api_key = os.getenv("JQ_API_KEY")
    if not api_key:
        print("環境変数 JQ_API_KEY が設定されていません")
        return

    headers = {
        "Authorization": f"Bearer {api_key}"
    }

    results = []

    for row in ai_list:
        code = row["コード"]

        # 1年分のデータ取得
        end = datetime.today().date()
        start = end - timedelta(days=365)

        params = {
            "code": code,
            "from": start.strftime("%Y-%m-%d"),
            "to": end.strftime("%Y-%m-%d"),
        }

        url = "https://api.jquants.com/v1/prices/daily_quotes"
        r = requests.get(url, headers=headers, params=params)

        if r.status_code != 200:
            continue

        js = r.json()
        rows = js.get("daily_quotes", [])
        if not rows or len(rows) < 10:
            continue

        df = pd.DataFrame(rows)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").set_index("Date")

        close = df["Close"]

        today_close = close.iloc[-1]
        if len(close) >= 6:
            future_close = close.iloc[-6]
        else:
            continue

        ret = (future_close - today_close) / today_close
        results.append(ret)

    if not results:
        print("\nバックテスト結果：データ不足\n")
        return

    results = np.array(results)
    win_rate = (results >= 0.05).mean()
    avg_return = results.mean()

    print("\n===== AI単独ルート バックテスト結果（過去1年） =====")
    print(f"勝率（5日後+5%以上）：{win_rate*100:.2f}%")
    print(f"平均リターン：{avg_return*100:.2f}%")

# =========================================================
# メイン処理（高速版）
# =========================================================
def run_screening():
    print("日本株銘柄リストを読み込み中...")
    symbols = load_symbol_list()

    print("AIモデル読み込み中...")
    model = load_ai_model()

    print("株価データを一括ダウンロード中...")
    all_data = download_all_data(symbols)

    print(f"AI閾値（自動調整）: {BEST_TH}")
    print("スクリーニング開始...")

    results = Parallel(
        n_jobs=-1,
        backend="loky",
        verbose=0
    )(
        delayed(analyze_symbol)(row["コード"], row["銘柄名"], model, all_data)
        for _, row in symbols.iterrows()
    )

    results = [r for r in results if r is not None]

    normal_signals = [r for r in results if r["route"] == "normal"]
    ai_only_signals = [r for r in results if r["route"] == "ai_only"]

    print("\n===== 初動＋継続シグナル（上位20） =====\n")
    if normal_signals:
        df_normal = pd.DataFrame(normal_signals).sort_values("AI上昇確率", ascending=False).head(20)
        print(df_normal.to_string(index=False))
    else:
        print("該当なし")

    print("\n===== AI単独（上位20） =====\n")
    if ai_only_signals:
        df_ai = pd.DataFrame(ai_only_signals).sort_values("AI上昇確率", ascending=False).head(20)
        print(df_ai.to_string(index=False))
    else:
        print("該当なし")

    if ai_only_signals:
        backtest_ai_only(ai_only_signals)

# =========================================================
# 実行
# =========================================================
if __name__ == "__main__":
    run_screening()







