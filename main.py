import os
import io
import time
import duckdb
import pandas as pd
import numpy as np
import requests
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier

# =========================================================
# 1. 基本設定と認証ロジック (J-Quants V2 最終確定版)
# =========================================================
DB_NAME = "stock_analytics.duckdb"
BASE_URL = "https://api.jquants.com/v2"
API_KEY = os.getenv("JQ_API_KEY") 

def get_id_token():
    """
    J-Quants V2 最新仕様:
    - エンドポイント: /token/generate (または /idtoken/get)
    - メソッド: POST
    - 認証: x-api-key ヘッダーを使用
    """
    # V2の最も標準的なエンドポイント
    url = f"{BASE_URL}/token/generate"
    
    # APIキーはURLパラメータではなく、ヘッダーに入れます
    headers = {
        "x-api-key": API_KEY
    }
    
    # メソッドは POST です
    res = requests.post(url, headers=headers)
    
    if res.status_code != 200:
        print(f"【認証エラー】ステータスコード: {res.status_code}")
        print(f"詳細: {res.text}")
        # もしこれでも 403 になる場合は /idtoken/get を POST で試します
        if res.status_code == 403:
            url_alt = f"{BASE_URL}/idtoken/get"
            res = requests.post(url_alt, headers=headers)
            if res.status_code == 200:
                return res.json().get("idToken")
        res.raise_for_status()
        
    return res.json().get("idToken")

# =========================================================
# 2. データベース蓄積・同期ロジック
# =========================================================
def sync_database():
    print("--- ステップ1: データの同期を開始します ---")
    conn = duckdb.connect(DB_NAME)
    
    conn.execute("""
        CREATE TABLE IF NOT EXISTS daily_quotes (
            Date TIMESTAMP, Code VARCHAR, Open DOUBLE, High DOUBLE, 
            Low DOUBLE, Close DOUBLE, Volume DOUBLE
        )
    """)
    
    token = get_id_token()
    # 取得したIDトークンは 'Authorization: Bearer' ヘッダーで使用
    headers = {"Authorization": f"Bearer {token}"}

    # A. 銘柄一覧取得
    m_res = requests.get(f"{BASE_URL}/listed/info", headers=headers)
    all_codes = [item['Code'] for item in m_res.json().get('info', [])]

    # B. yfinanceによる過去補完
    existing_codes = conn.execute("SELECT DISTINCT Code FROM daily_quotes").df()['Code'].tolist()
    remaining_codes = [c for c in all_codes if c not in existing_codes]

    if remaining_codes:
        print(f">> {len(remaining_codes)} 銘柄を yfinance から補完...")
        start_date = (datetime.now() - timedelta(days=3*365)).strftime('%Y-%m-%d')
        batch_size = 50
        for i in range(0, len(remaining_codes), batch_size):
            batch = remaining_codes[i:i+batch_size]
            yf_codes = [f"{c}.T" for c in batch]
            try:
                raw_data = yf.download(yf_codes, start=start_date, group_by='ticker', threads=False, timeout=30)
                batch_dfs = []
                for ticker in yf_codes:
                    if ticker in raw_data and not raw_data[ticker].dropna().empty:
                        df_t = raw_data[ticker].copy().reset_index()
                        df_t['Code'] = ticker.replace('.T', '')
                        df_t = df_t[['Date', 'Code', 'Open', 'High', 'Low', 'Close', 'Volume']]
                        batch_dfs.append(df_t)
                if batch_dfs:
                    combined = pd.concat(batch_dfs)
                    conn.register("tmp_batch", combined)
                    conn.execute("INSERT INTO daily_quotes SELECT * FROM tmp_batch")
                    conn.unregister("tmp_batch")
                time.sleep(1.0)
            except Exception as e:
                continue

    # C. J-Quants V2による最新日更新
    print(">> 最新株価を同期中...")
    try:
        today_str = datetime.now().strftime('%Y-%m-%d')
        r = requests.get(f"{BASE_URL}/prices/daily_quotes", headers=headers, params={"date": today_str})
        if r.status_code == 200:
            data = r.json().get("daily_quotes", [])
            if data:
                new_df = pd.DataFrame(data)
                new_df["Date"] = pd.to_datetime(new_df["Date"])
                conn.register("new_data", new_df)
                conn.execute("""
                    INSERT INTO daily_quotes 
                    SELECT Date, Code, Open, High, Low, Close, Volume FROM new_data
                    WHERE NOT EXISTS (
                        SELECT 1 FROM daily_quotes 
                        WHERE daily_quotes.Code = new_data.Code AND daily_quotes.Date = new_data.Date
                    )
                """)
                conn.unregister("new_data")
    except Exception as e:
        pass

    conn.close()

# =========================================================
# 3. AI分析
# =========================================================
def run_analysis():
    print("--- ステップ2: AI分析を開始します ---")
    conn = duckdb.connect(DB_NAME)
    df = conn.execute("""
        SELECT 
            Date, Code, Close,
            AVG(Close) OVER (PARTITION BY Code ORDER BY Date ROWS BETWEEN 74 PRECEDING AND CURRENT ROW) as SMA75,
            AVG(Volume) OVER (PARTITION BY Code ORDER BY Date ROWS BETWEEN 24 PRECEDING AND CURRENT ROW) as Vol25,
            Volume,
            LEAD(Close, 10) OVER (PARTITION BY Code ORDER BY Date) as FutureClose
        FROM daily_quotes
    """).df()
    conn.close()

    if df.empty: return

    df['Bias75'] = (df['Close'] - df['SMA75']) / df['SMA75']
    df['VolRatio'] = df['Volume'] / df['Vol25']
    df['Target'] = (df['FutureClose'] / df['Close'] > 1.05).astype(int)
    
    train_df = df.dropna(subset=['Target', 'Bias75', 'VolRatio']).tail(100000)
    features = ['Bias75', 'VolRatio']
    
    if len(train_df) < 200:
        print("データ蓄積待ちです。")
        return

    model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    model.fit(train_df[features], train_df['Target'])
    
    latest_date = df['Date'].max()
    latest_data = df[df['Date'] == latest_date].dropna(subset=features)
    
    if not latest_data.empty:
        latest_data['AI_Score'] = model.predict_proba(latest_data[features])[:, 1]
        result = latest_data.sort_values('AI_Score', ascending=False).head(20)
        print(f"\n🚀 === AI上昇期待銘柄ランキング ({latest_date.strftime('%Y-%m-%d')}) ===")
        print(result[['Code', 'Close', 'AI_Score']].to_string(index=False))

if __name__ == "__main__":
    if not API_KEY:
        print("❌ API_KEY が設定されていません。")
    else:
        sync_database()
        run_analysis()
