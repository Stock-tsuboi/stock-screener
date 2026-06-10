import pandas as pd
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_us_stocks_csv():
    """WikipediaからS&P 500の銘柄リストを取得してCSV保存する"""
    logger.info("S&P 500 銘柄リストを取得中...")
    
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url)
        df = tables[0]
        
        output_df = df[['Symbol', 'Security']].copy()
        output_df.columns = ['Ticker', 'Name']
        output_df['Ticker'] = output_df['Ticker'].str.replace('.', '-', regex=False)
        output_df.to_csv("us_stocks.csv", index=False, encoding='utf-8')
        logger.info(f"完了: us_stocks.csv に {len(output_df)} 銘柄を保存しました。")
    except Exception as e:
        logger.error(f"リスト取得失敗: {e}")

if __name__ == "__main__":
    generate_us_stocks_csv()