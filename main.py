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
from dotenv import load_dotenv

load_dotenv()

# --- 設定 ---
DB_NAME = "stock_analytics.duckdb"
# BASE_URL を V2 の最新エンドポイントに合わせます
BASE_URL = "https://api.jquants.com/v2" 
API_KEY = os.getenv("JQ_API_KEY")

# =========================================================
# 1. 認証 (J-Quants V2 最新仕様)
# =========================================================
def get_id_token():
    # V2の正しいエンドポイントは /token/generate ではなく /idtoken/get です
    # また、APIキーはヘッダーではなく、クエリパラメータ "mailaddress" にセットします
    # ※V2のAPIキーは実質的にメールアドレスの代わりとして機能します
    url = f"{BASE_URL}/idtoken/get"
    params = {"mailaddress": API_KEY}
    
    res = requests.post(url, params=params)
    
    if res.status_code != 200:
        print(f"認証エラー: {res.status_code}")
        print(f"レスポンス内容: {res.text}")
        res.raise_for_status()
        
    return res.json().get("idToken")

# =========================================================
# 2. データベース蓄積・更新ロジック
# =========================================================
def sync_database():
    conn = duckdb.connect(DB_NAME)
    # テーブル作成
    conn.execute("""
        CREATE TABLE IF NOT EXISTS daily_quotes (
            Date TIMESTAMP, Code VARCHAR, Open DOUBLE, High DOUBLE, 
            Low DOUBLE, Close DOUBLE, Volume DOUBLE
        )
    """)
    
    # 銘柄リスト取得
    token = get_id_token()
    headers = {"Authorization": f"Bearer {token}"}
    m_res = requests.get(f"{BASE_URL}/listed/info", headers=headers)
    all_codes = [item['Code'] for item in m_res.json()['info']]

    # A. yfinanceによる過去分補完 (未取得の銘柄のみ)
    existing_codes = conn.execute("SELECT DISTINCT Code FROM daily_quotes").df()['Code'].tolist()
    remaining_codes = [c for c in all_codes if c not in existing_codes]

    if remaining_codes:
        print(f"新規・未取得の {len(remaining_codes)} 銘柄を yfinance から取得します...")
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
                time.sleep(1.5)
            except Exception as e:
                print(f"Error in batch {batch[0]}: {e}")
                continue

    # B. J-Quants V2による全銘柄の最新日差分更新
    print("J-Quants V2 から全銘柄の最新データを補充中...")
    try:
        r = requests.get(f"{BASE_URL}/prices/daily_quotes/all", headers=headers, params={"format": "csv"})
        r.raise_for_status()
        new_df = pd.read_csv(io.StringIO(r.text))
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
        print(f"J-Quants 更新失敗 (差分なしの可能性): {e}")

    conn.close()

# =========================================================
# 3. 分析・AI推論ロジック (これ単体で完結)
# =========================================================
def run_analysis():
    conn = duckdb.connect(DB_NAME)
    print("DuckDB内で長期指標を計算中...")
    
    # 75日移動平均を含む特徴量をSQLで一括計算
    df = conn.execute("""
        SELECT 
            Date, Code, Close,
            AVG(Close) OVER (PARTITION BY Code ORDER BY Date ROWS BETWEEN 74 PRECEDING AND CURRENT ROW) as SMA75,
            AVG(Close) OVER (PARTITION BY Code ORDER BY Date ROWS BETWEEN 24 PRECEDING AND CURRENT ROW) as SMA25,
            AVG(Volume) OVER (PARTITION BY Code ORDER BY Date ROWS BETWEEN 24 PRECEDING AND CURRENT ROW) as Vol25,
            Volume,
            LEAD(Close, 10) OVER (PARTITION BY Code ORDER BY Date) as FutureClose
        FROM daily_quotes
    """).df()
    conn.close()

    # 特徴量エンジニアリング
    df['Bias75'] = (df['Close'] - df['SMA75']) / df['SMA75']
    df['VolRatio'] = df['Volume'] / df['Vol25']
    df['Target'] = (df['FutureClose'] / df['Close'] > 1.05).astype(int) # 10日で5%以上上昇
    
    # 学習 (データが十分にある行のみ)
    train_df = df.dropna(subset=['Target', 'SMA75', 'VolRatio']).tail(200000)
    features = ['Bias75', 'VolRatio', 'SMA25']
    
    print(f"AI学習開始 (サンプル数: {len(train_df)})...")
    model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    model.fit(train_df[features], train_df['Target'])
    
    # 最新日の予測
    latest_date = df['Date'].max()
    latest_data = df[df['Date'] == latest_date].dropna(subset=features)
    
    if not latest_data.empty:
        latest_data['AI_Score'] = model.predict_proba(latest_data[features])[:, 1]
        result = latest_data.sort_values('AI_Score', ascending=False).head(20)
        
        print(f"\n=== AI上昇期待銘柄ランキング ({latest_date.strftime('%Y-%m-%d')}) ===")
        print(result[['Code', 'Close', 'AI_Score']].to_string(index=False))
    else:
        print("推論対象の最新データが見つかりませんでした。")

# =========================================================
# メイン実行
# =========================================================
if __name__ == "__main__":
    sync_database() # 蓄積・更新
    run_analysis()  # 分析・予測
