import os
import time
import duckdb
import pandas as pd
import numpy as np
import requests
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier

# =========================================================
# 0. 基本設定
# =========================================================
DB_NAME = "stock_analytics.duckdb"
BASE_URL = "https://api.jquants.com/v2"

JQ_MAIL = os.getenv("JQ_API_KEY")
JQ_PASS = os.getenv("JQ_PASSWORD")

# =========================================================
# 1. J-Quants 認証（V2準拠・過去データ用）
# =========================================================
def get_id_token():
    url = f"{BASE_URL}/token/generate"
    payload = {
        "mailaddress": JQ_MAIL,
        "password": JQ_PASS
    }
    r = requests.post(url, json=payload, timeout=15)
    if r.status_code != 200:
        raise RuntimeError(f"J-Quants auth failed: {r.text}")
    return r.json()["idToken"]

# =========================================================
# 2. yfinance 安定取得（直近用）
# =========================================================
def fetch_yf_daily(code, days=90, retry=3):
    ticker = f"{code}.T"

    for i in range(retry):
        try:
            df = yf.download(
                ticker,
                period=f"{days}d",
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=False
            )
            if not df.empty:
                df = df.reset_index()
                df["Code"] = code
                return df[["Date", "Code", "Open", "High", "Low", "Close", "Volume"]]
        except Exception as e:
            print(f"[yfinance retry {i+1}] {code} : {e}")
            time.sleep(5)

    return pd.DataFrame()

# =========================================================
# 3. データベース同期（安定版）
# =========================================================
def sync_database():
    print("=== STEP1: DB SYNC START ===")
    conn = duckdb.connect(DB_NAME)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS daily_quotes (
            Date DATE,
            Code VARCHAR,
            Open DOUBLE,
            High DOUBLE,
            Low DOUBLE,
            Close DOUBLE,
            Volume DOUBLE,
            PRIMARY KEY (Date, Code)
        )
    """)

    # --- 銘柄一覧（J-Quants：過去専用）
    token = get_id_token()
    headers = {"Authorization": f"Bearer {token}"}

    r = requests.get(f"{BASE_URL}/listed/info", headers=headers, timeout=30)
    codes = [x["Code"] for x in r.json()["info"]]

    # --- 直近データ補完（yfinance）
    for code in codes:
        try:
            df = fetch_yf_daily(code, days=90)
            if df.empty:
                continue

            conn.register("tmp", df)
            conn.execute("""
                INSERT OR IGNORE INTO daily_quotes
                SELECT * FROM tmp
            """)
            conn.unregister("tmp")

            time.sleep(0.3)  # GitHub Actions 安定化
        except Exception as e:
            print(f"[SYNC ERROR] {code}: {e}")

    conn.close()
    print("=== STEP1: DB SYNC END ===")

# =========================================================
# 4. AI分析（元ロジック維持）
# =========================================================
def run_analysis():
    print("=== STEP2: AI ANALYSIS START ===")
    conn = duckdb.connect(DB_NAME)

    df = conn.execute("""
        SELECT
            Date,
            Code,
            Close,
            Volume,
            AVG(Close) OVER (
                PARTITION BY Code
                ORDER BY Date
                ROWS BETWEEN 74 PRECEDING AND CURRENT ROW
            ) AS SMA75,
            AVG(Volume) OVER (
                PARTITION BY Code
                ORDER BY Date
                ROWS BETWEEN 24 PRECEDING AND CURRENT ROW
            ) AS Vol25,
            LEAD(Close, 10) OVER (
                PARTITION BY Code
                ORDER BY Date
            ) AS FutureClose
        FROM daily_quotes
    """).df()

    conn.close()
    if df.empty:
        print("No data for analysis")
        return

    df["Bias75"] = (df["Close"] - df["SMA75"]) / df["SMA75"]
    df["VolRatio"] = df["Volume"] / df["Vol25"]
    df["Target"] = (df["FutureClose"] / df["Close"] > 1.05).astype(int)

    train_df = df.dropna(subset=["Bias75", "VolRatio", "Target"]).tail(100000)
    if len(train_df) < 500:
        print("Not enough training data")
        return

    model = RandomForestClassifier(
        n_estimators=100,
        n_jobs=-1,
        random_state=42
    )
    model.fit(train_df[["Bias75", "VolRatio"]], train_df["Target"])

    latest_date = df["Date"].max()
    latest = df[df["Date"] == latest_date].dropna(subset=["Bias75", "VolRatio"])

    if latest.empty:
        return

    latest = latest.copy()
    latest["AI_Score"] = model.predict_proba(
        latest[["Bias75", "VolRatio"]]
    )[:, 1]

    result = latest.sort_values("AI_Score", ascending=False).head(20)

    print(f"\n🚀 AI上昇期待銘柄ランキング ({latest_date})")
    print(result[["Code", "Close", "AI_Score"]].to_string(index=False))

# =========================================================
# 5. メイン実行（落ちない構成）
# =========================================================
if __name__ == "__main__":
    try:
        sync_database()
    except Exception as e:
        print("DB SYNC FAILED:", e)

    try:
        run_analysis()
    except Exception as e:
        print("ANALYSIS FAILED:", e)
