import pandas as pd
import os
import requests
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_us_stocks_csv():
    """NASDAQ FTPディレクトリから全銘柄リストを取得してCSV保存する"""
    logger.info("米国市場（NASDAQ/NYSE/AMEX）全銘柄リストを取得中...")
    
    try:
        # 1. NASDAQ上場銘柄の取得
        nasdaq_url = "https://ftp.nasdaqtrader.com/SymbolDirectory/nasdaqlisted.txt"
        df_nasdaq = pd.read_csv(nasdaq_url, sep="|")
        df_nasdaq = df_nasdaq.iloc[:-1]  # 最終行のメタデータ（File Creation Time）を削除
        
        # ETFを除外し、TickerとNameを抽出
        df_nasdaq = df_nasdaq[df_nasdaq['ETF'] == 'N']
        df_nasdaq = df_nasdaq[['Symbol', 'Security Name']].rename(columns={'Symbol': 'Ticker', 'Security Name': 'Name'})

        # 2. その他市場（NYSE, AMEX等）の取得
        other_url = "https://ftp.nasdaqtrader.com/SymbolDirectory/otherlisted.txt"
        df_other = pd.read_csv(other_url, sep="|")
        df_other = df_other.iloc[:-1]  # 最終行のメタデータを削除
        
        # ETFを除外し、TickerとNameを抽出 (Otherは ACT Symbol がティッカー)
        df_other = df_other[df_other['ETF'] == 'N']
        df_other = df_other[['ACT Symbol', 'Security Name']].rename(columns={'ACT Symbol': 'Ticker', 'Security Name': 'Name'})

        # 3. リストの結合
        output_df = pd.concat([df_nasdaq, df_other], ignore_index=True)
        
        # クリーンアップ: 記号を含む銘柄（優先株、ワラント、ユニット等）を除外
        output_df['Ticker'] = output_df['Ticker'].str.strip()
        output_df = output_df[output_df['Ticker'].str.match(r'^[A-Z]+$', na=False)]
        
        # 重複削除
        output_df = output_df.drop_duplicates(subset=['Ticker'])

        output_df.to_csv("us_stocks.csv", index=False, encoding='utf-8')
        logger.info(f"完了: us_stocks.csv に {len(output_df)} 銘柄を保存しました。")
    except Exception as e:
        logger.error(f"リスト取得失敗: {e}")

if __name__ == "__main__":
    generate_us_stocks_csv()
