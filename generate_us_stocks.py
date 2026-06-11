import pandas as pd
import os
import requests
import logging
import io
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_nasdaq_data(url):
    """HTTPリクエストを使用してデータを取得する（リトライ機能付き）"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36'
    }
    for attempt in range(3):
        try:
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            return pd.read_csv(io.StringIO(response.text), sep="|")
        except Exception as e:
            logger.warning(f"取得試行 {attempt + 1} 失敗: {e}")
            if attempt == 2:
                raise e
            time.sleep(5)

def generate_us_stocks_csv():
    """NASDAQ FTPディレクトリから全銘柄リストを取得してCSV保存する"""
    logger.info("米国市場（NASDAQ/NYSE/AMEX）全銘柄リストを取得中...")
    
    try:
        # 1. NASDAQ上場銘柄の取得
        nasdaq_url = "https://ftp.nasdaqtrader.com/SymbolDirectory/nasdaqlisted.txt"
        df_nasdaq = fetch_nasdaq_data(nasdaq_url)
        # メタデータ行（File Creation Timeで始まる行）を除外
        df_nasdaq = df_nasdaq[~df_nasdaq['Symbol'].astype(str).str.startswith('File Creation Time')]
        
        # ETFを除外し、テスト銘柄を除外
        df_nasdaq = df_nasdaq[(df_nasdaq['ETF'] == 'N') & (df_nasdaq['Test Issue'] == 'N')]
        df_nasdaq = df_nasdaq[['Symbol', 'Security Name']].rename(columns={'Symbol': 'Ticker', 'Security Name': 'Name'})
        df_nasdaq['Exchange'] = 'NASDAQ'

        # 2. その他市場（NYSE, AMEX等）の取得
        other_url = "https://ftp.nasdaqtrader.com/SymbolDirectory/otherlisted.txt"
        df_other = fetch_nasdaq_data(other_url)
        # メタデータ行を除外
        df_other = df_other[~df_other['ACT Symbol'].astype(str).str.startswith('File Creation Time')]
        
        # NYSE(N)とAMEX(A)に限定し、かつETFとテスト銘柄を除去
        exchange_map = {'N': 'NYSE', 'A': 'AMEX'}
        df_other = df_other[
            (df_other['Exchange'].isin(exchange_map.keys())) & 
            (df_other['ETF'] == 'N') & 
            (df_other['Test Issue'] == 'N')
        ].copy()
        df_other['Exchange'] = df_other['Exchange'].map(exchange_map)
        df_other = df_other[['ACT Symbol', 'Security Name', 'Exchange']].rename(columns={'ACT Symbol': 'Ticker', 'Security Name': 'Name'})

        # 3. リストの結合
        output_df = pd.concat([df_nasdaq, df_other], ignore_index=True)
        
        # クリーンアップ: 記号を含む銘柄（優先株、ワラント、ユニット等）を除外
        output_df['Ticker'] = output_df['Ticker'].str.strip()
        # 通常の米国株（1-4文字）に限定し、5文字以上のワラントや権利を除外
        output_df = output_df[
            (output_df['Ticker'].str.match(r'^[A-Z]{1,4}$', na=False))
        ]
        
        # 重複削除
        output_df = output_df.drop_duplicates(subset=['Ticker'])

        output_df.to_csv("us_stocks.csv", index=False, encoding='utf-8')
        logger.info(f"完了: us_stocks.csv に {len(output_df)} 銘柄を保存しました。")
    except Exception as e:
        logger.error(f"リスト取得失敗: {e}")

if __name__ == "__main__":
    generate_us_stocks_csv()
