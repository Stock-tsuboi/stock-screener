# =========================================================
# screening_ai.py  完全版（Parallel統一＋警告ゼロ＋AI閾値自動調整）
# =========================================================

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ResourceWarning)

import time
import pandas as pd
import yfinance as yf
import joblib
import numpy as np
from joblib import Parallel, delayed

# =========================================================
# 設定
# =========================================================

CSV_FILE = "japan_stocks_jpx.csv"
MODEL_FILE = "model.pkl"
THRESHOLD_FILE = "best_threshold.txt"
EXCLUDE_CODES = {"6576", "130A", "9999"}

# =========================================================
# 最適AI閾値の読み込み
# =========================================================

try:
    with open(THRESHOLD_FILE) as f:
        BEST_TH = float(f.read().strip())
except:
    BEST_TH = 0.60  # フォールバック
    print(f"警告: {THRESHOLD_FILE} が見つからないため、AI閾値を {BEST_TH} に設定")

# =========================================================
# MultiIndex → Series 変換
# =========================================================

def smart_close(series):
    if isinstance(series, pd.DataFrame):
        return series.iloc[:, 0].astype(float)
    return series.astype(float)

# =========================================================
# テクニカル指標
# =========================================================

def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calc_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def calc_adx(data, period=14):
    high = smart_close(data["High"])
    low = smart_close(data["Low"])
    close = smart_close(data["Close"])

    plus_dm = high.diff()
    minus_dm = low.diff().abs()

    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.rolling(period).mean()

    plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr)

    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.rolling(period).mean()

    return adx

# =========================================================
# 銘柄リスト読み込み
# =========================================================

def load_symbol_list():
    return pd.read_csv(CSV_FILE)

# =========================================================
# AIモデル読み込み
# =========================================================

def load_ai_model():
    return joblib.load(MODEL_FILE)

# =========================================================
# 安全な割り算
# =========================================================

def safe_div(a, b):
    if b is None or b == 0 or np.isnan(b):
        return 0
    return (a - b) / b

# =========================================================
# 銘柄解析（初動＋継続＋AI単独）
# =========================================================

def analyze_symbol(code, name, model):
    if code in EXCLUDE_CODES:
        return None

    symbol = f"{code}.T"
    time.sleep(0.15)

    try:
        data = yf.download(symbol, period="6mo", interval="1d", progress=False)
    except:
        return None

    if data is None or len(data) < 50:
        return None

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    close = smart_close(data["Close"])
    high = smart_close(data["High"])
    low = smart_close(data["Low"])
    volume = smart_close(data["Volume"])

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

    # AI単独ルート（自動閾値）
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

    # 初動＋継続ルート
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
# AI単独ルートのバックテスト
# =========================================================

def backtest_ai_only(ai_list):
    results = []

    for row in ai_list:
        code = row["コード"]
        symbol = f"{code}.T"

        try:
            data = yf.download(symbol, period="1y", interval="1d", progress=False)
        except:
            continue

        if len(data) < 10:
            continue

        close = data["Close"]

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
# メイン処理（Parallel 完全統一）
# =========================================================

def run_screening():
    print("日本株銘柄リストを読み込み中...")
    symbols = load_symbol_list()

    print("AIモデル読み込み中...")
    model = load_ai_model()

    print(f"AI閾値（自動調整）: {BEST_TH}")

    print("スクリーニング開始...")

    results = Parallel(
        n_jobs=-1,
        backend="loky",
        verbose=0
    )(
        delayed(analyze_symbol)(row["コード"], row["銘柄名"], model)
        for _, row in symbols.iterrows()
    )

    results = [r for r in results if r is not None]

    normal_signals = [r for r in results if r["route"] == "normal"]
    ai_only_signals = [r for r in results if r["route"] == "ai_only"]

    print("\n===== 初動＋継続シグナル =====\n")
    if normal_signals:
        print(pd.DataFrame(normal_signals).sort_values("AI上昇確率", ascending=False).to_string(index=False))
    else:
        print("該当なし")

    print("\n===== AI単独（AI確率・自動閾値以上） =====\n")
    if ai_only_signals:
        print(pd.DataFrame(ai_only_signals).sort_values("AI上昇確率", ascending=False).to_string(index=False))
    else:
        print("該当なし")

    if ai_only_signals:
        backtest_ai_only(ai_only_signals)

# =========================================================
# 実行
# =========================================================

if __name__ == "__main__":
    run_screening()
