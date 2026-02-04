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
# 1. 基本設定と認証ロジック (J-Quants V2 仕様)
# =========================================================
DB_NAME = "stock_analytics.duckdb"    # 保存先データベース名
BASE_URL = "https://api.jquants.com/v2"
API_KEY = os.getenv("JQ_API_KEY")      # GitHub SecretsからAPIキーを取得
PASSWORD = os.getenv("JQ_PASSWORD")    # GitHub Secretsからパスワードを取得

def get_id_token():
    """
    J-Quants V2の『IDトークン』を発行する関数。
    APIキーをメールアドレスとして扱い、パスワードとセットでJSON形式で送信します。
    """
    url = f"{BASE_URL}/token/generate"
    payload = {
        "mailaddress": API_KEY,
        "password": PASSWORD
    }
    
    # POSTリクエストで認証情報を送信
    res = requests.post(url, json=payload)
    
    if res.status_code != 200:
        print(f"【認証エラー】ステータスコード: {res.status_code}")
        print(f"詳細: {res.text}")
        res.raise_for_status()
        
    return res.json().get("idToken")

# =========================================================
# 2. データベース蓄積・同期ロジック
# =========================================================
def sync_database():
    print("--- ステップ1: データの同期を開始します ---")
    conn = duckdb.connect(DB_NAME)
    
    # 株価データを格納するテーブルがなければ作成
    conn.execute("""
        CREATE TABLE IF NOT EXISTS daily_quotes (
            Date TIMESTAMP, Code VARCHAR, Open DOUBLE, High DOUBLE, 
            Low DOUBLE, Close DOUBLE, Volume DOUBLE
        )
    """)
    
    # 認証トークンを取得してヘッダーにセット
    token = get_id_token()
    headers = {"Authorization": f"Bearer {token}"}

    # A. 銘柄一覧の取得
    # J-Quantsから現在上場している全銘柄のリストを取得します
    m_res = requests.get(f"{BASE_URL}/listed/info", headers=headers)
    all_codes = [item['Code'] for item in m_res.json().get('info', [])]

    # B. yfinanceによる過去データの補完 (初回実行時や新規銘柄用)
    # すでにDBにある銘柄はスキップし、未取得の銘柄のみ過去3年分をダウンロードします
    existing_codes = conn.execute("SELECT DISTINCT Code FROM daily_quotes").df()['Code'].tolist()
    remaining_codes = [c for c in all_codes if c not in existing_codes]

    if remaining_codes:
        print(f">> 未取得の {len(remaining_codes)} 銘柄を yfinance から補充します...")
        start_date = (datetime.now() - timedelta(days=3*365)).strftime('%Y-%m-%d')
        
        # サーバー負荷を抑えるため、50銘柄ずつのバッチ処理を行います
        batch_size = 50
        for i in range(0, len(remaining_codes), batch_size):
            batch = remaining_codes[i:i+batch_size]
            yf_codes = [f"{c}.T" for c in batch] # 東証銘柄用に末尾に.Tを付与
            try:
                # データを一括ダウンロード
                raw_data = yf.download(yf_codes, start=start_date, group_by='ticker', threads=False, timeout=30)
                batch_dfs = []
                for ticker in yf_codes:
                    if ticker in raw_data and not raw_data[ticker].dropna().empty:
                        df_t = raw_data[ticker].copy().reset_index()
                        df_t['Code'] = ticker.replace('.T', '')
                        # DBの形式に合わせてカラムを並び替え
                        df_t = df_t[['Date', 'Code', 'Open', 'High', 'Low', 'Close', 'Volume']]
                        batch_dfs.append(df_t)
                
                if batch_dfs:
                    combined = pd.concat(batch_dfs)
                    conn.register("tmp_batch", combined)
                    conn.execute("INSERT INTO daily_quotes SELECT * FROM tmp_batch")
                    conn.unregister("tmp_batch")
                
                print(f"  進捗: {i + len(batch)} 銘柄完了")
                time.sleep(1.0) # 連続アクセス防止のウェイト
            except Exception as e:
                print(f"  バッチ取得中にエラー (スキップします): {e}")
                continue

    # C. J-Quants V2による最新日の差分更新
    # yfinanceは反映が遅れることがあるため、最新の公式株価を反映させます
    print(">> J-Quants V2 から最新の株価データを同期しています...")
    try:
        today_str = datetime.now().strftime('%Y-%m-%d')
        # 特定の日付の全銘柄株価を一括取得
        r = requests.get(f"{BASE_URL}/prices/daily_quotes", headers=headers, params={"date": today_str})
        if r.status_code == 200:
            data = r.json().get("daily_quotes", [])
            if data:
                new_df = pd.DataFrame(data)
                new_df["Date"] = pd.to_datetime(new_df["Date"])
                conn.register("new_data", new_df)
                # 既存データと重複しない（日付とコードが一致しない）場合のみ挿入
                conn.execute("""
                    INSERT INTO daily_quotes 
                    SELECT Date, Code, Open, High, Low, Close, Volume FROM new_data
                    WHERE NOT EXISTS (
                        SELECT 1 FROM daily_quotes 
                        WHERE daily_quotes.Code = new_data.Code AND daily_quotes.Date = new_data.Date
                    )
                """)
                conn.unregister("new_data")
                print(f"   [{today_str}] のデータを更新しました。")
    except Exception as e:
        print(f"   J-Quants 最新データ更新スキップ: {e}")

    conn.close()

# =========================================================
# 3. AI分析・期待銘柄ランキング出力
# =========================================================
def run_analysis():
    print("--- ステップ2: AI分析を開始します ---")
    conn = duckdb.connect(DB_NAME)
    
    # 蓄積されたデータから指標（移動平均など）を計算
    # SMA75: 75日移動平均, SMA25: 25日移動平均, VolRatio: 出来高の変化率
    # FutureClose: 10日後の終値（これを予測の正解にする）
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

    if df.empty:
        print("分析対象のデータがありません。")
        return

    # 特徴量の作成
    df['Bias75'] = (df['Close'] - df['SMA75']) / df['SMA75']  # 75日乖離率
    df['VolRatio'] = df['Volume'] / df['Vol25']              # 直近25日比の出来高
    # 正解ラベル: 10日後に価格が5%以上上昇したなら「1」、そうでなければ「0」
    df['Target'] = (df['FutureClose'] / df['Close'] > 1.05).astype(int)
    
    # 学習データの選別（欠損値を除外し、直近10万件を使用）
    train_df = df.dropna(subset=['Target', 'Bias75', 'VolRatio']).tail(100000)
    features = ['Bias75', 'VolRatio', 'SMA25']
    
    if len(train_df) < 500:
        print("学習用のデータが十分に蓄積されていません。")
        return

    # ランダムフォレストによる学習
    print(f">> AIモデルの学習中 (データ件数: {len(train_df)})...")
    model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    model.fit(train_df[features], train_df['Target'])
    
    # 直近の市場データを使って予測
    latest_date = df['Date'].max()
    latest_data = df[df['Date'] == latest_date].dropna(subset=features)
    
    if not latest_data.empty:
        # 上昇確率（AI_Score）を算出
        latest_data['AI_Score'] = model.predict_proba(latest_data[features])[:, 1]
        # スコアが高い順に上位20銘柄を抽出
        result = latest_data.sort_values('AI_Score', ascending=False).head(20)
        
        print(f"\n🚀 === AI上昇期待銘柄ランキング ({latest_date.strftime('%Y-%m-%d')}) ===")
        print(result[['Code', 'Close', 'AI_Score']].to_string(index=False))
    else:
        print("最新の市場データに基づいた予測ができませんでした。")

# =========================================================
# 実行メイン処理
# =========================================================
if __name__ == "__main__":
    # 認証情報のチェック
    if not API_KEY or not PASSWORD:
        print("❌ エラー: GitHubのSecretsに JQ_API_KEY または JQ_PASSWORD が登録されていません。")
    else:
        sync_database() # ステップ1: データの蓄積・更新
        run_analysis()  # ステップ2: AIによる分析
