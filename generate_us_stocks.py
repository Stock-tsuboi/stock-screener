import pandas as pd
import os
import requests
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_us_stocks_csv():
    """NASDAQ/NYSE/AMEXの全銘柄リストをNASDAQ APIから取得してCSV保存する"""
    logger.info("米国市場（NASDAQ/NYSE/AMEX）全銘柄リストを取得中...")
    
    try:
        # NASDAQ Screener API (全銘柄取得用のURL)
        url = "https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=0&offset=0&download=true"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Origin': 'https://www.nasdaq.com',
            'Referer': 'https://www.nasdaq.com/'
        }
        
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        # JSONデータから銘柄一覧を抽出
        rows = data['data']['rows']
        df = pd.DataFrame(rows)
        
        # yfinanceで利用可能な形式に変換 (symbol -> Ticker, name -> Name)
        output_df = df[['symbol', 'name']].copy()
        output_df.columns = ['Ticker', 'Name']
        
        # 記号のクリーンアップ（yfinanceは / や ^ ではなく - を使用することが多い）
        output_df['Ticker'] = output_df['Ticker'].str.strip().str.replace(r'[/^]', '-', regex=True)
        
        # 重複削除
        output_df = output_df.drop_duplicates(subset=['Ticker'])

        output_df.to_csv("us_stocks.csv", index=False, encoding='utf-8')
        logger.info(f"完了: us_stocks.csv に {len(output_df)} 銘柄を保存しました。")
    except Exception as e:
        logger.error(f"リスト取得失敗: {e}")

if __name__ == "__main__":
    generate_us_stocks_csv()
