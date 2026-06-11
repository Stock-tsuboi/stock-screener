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
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
    }
    for attempt in range(3):
        try:
            # タイムアウトを15秒に設定し、リトライを早める
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            return pd.read_csv(io.StringIO(response.text), sep="|")
        except Exception as e:
            logger.warning(f"取得試行 {attempt + 1} 失敗: {e}")
            if attempt == 2:
                return None
            time.sleep(5)

def fetch_fallback_list():
    """NASDAQ FTPがタイムアウトする場合の公式APIフォールバック取得"""
    logger.info("NASDAQ API (api.nasdaq.com) からのデータ取得を試みています...")
    url = "https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=0&offset=0&download=true"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
        'Accept': 'application/json, text/plain, */*',
        'Origin': 'https://www.nasdaq.com',
        'Referer': 'https://www.nasdaq.com/'
    }
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()
        rows = data['data']['rows']
        df = pd.DataFrame(rows)
        # カラム名を統一
        return df.rename(columns={'symbol': 'Ticker', 'name': 'Name'})
    except Exception as e:
        logger.error(f"APIフォールバック失敗: {e}")
        return None

def generate_us_stocks_csv():
    """NASDAQ FTPディレクトリから全銘柄リストを取得してCSV保存する"""
    logger.info("米国市場（NASDAQ/NYSE/AMEX）全銘柄リストを取得中...")
    
    try:
        # 手順1: 優先的にFTPサーバーからの取得を試みる
        nasdaq_url = "https://ftp.nasdaqtrader.com/SymbolDirectory/nasdaqlisted.txt"
        other_url = "https://ftp.nasdaqtrader.com/SymbolDirectory/otherlisted.txt"
        
        df_nasdaq_raw = fetch_nasdaq_data(nasdaq_url)
        df_other_raw = fetch_nasdaq_data(other_url)

        if df_nasdaq_raw is not None and df_other_raw is not None:
            # NASDAQ銘柄の加工
            df_nasdaq = df_nasdaq_raw[~df_nasdaq_raw['Symbol'].astype(str).str.startswith('File Creation Time')].copy()
            df_nasdaq = df_nasdaq[(df_nasdaq['ETF'] == 'N') & (df_nasdaq['Test Issue'] == 'N')]
            df_nasdaq = df_nasdaq[['Symbol', 'Security Name']].rename(columns={'Symbol': 'Ticker', 'Security Name': 'Name'})
            df_nasdaq['Exchange'] = 'NASDAQ'

            # その他市場銘柄の加工
            df_other = df_other_raw[~df_other_raw['ACT Symbol'].astype(str).str.startswith('File Creation Time')].copy()
            exchange_map = {'N': 'NYSE', 'A': 'AMEX'}
            df_other = df_other[
                (df_other['Exchange'].isin(exchange_map.keys())) & 
                (df_other['ETF'] == 'N') & 
                (df_other['Test Issue'] == 'N')
            ].copy()
            df_other['Exchange'] = df_other['Exchange'].map(exchange_map)
            df_other = df_other[['ACT Symbol', 'Security Name', 'Exchange']].rename(columns={'ACT Symbol': 'Ticker', 'Security Name': 'Name'})
            
            output_df = pd.concat([df_nasdaq, df_other], ignore_index=True)
        else:
            # 手順2: FTPが失敗した場合はAPIフォールバックを実行
            logger.warning("NASDAQ FTPサーバーがタイムアウトしました。公式APIへ切り替えます。")
            output_df = fetch_fallback_list()
            if output_df is None:
                raise ConnectionError("全ての銘柄リスト取得手段（FTPおよびAPI）が失敗しました。")
        
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
