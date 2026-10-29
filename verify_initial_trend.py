import time
import numpy as np
import pandas as pd
import yfinance as yf

INPUT = 'october_verification.csv'
OUTPUT = 'initial_trend_verification.csv'
SUMMARY = 'initial_trend_summary.csv'

def main():
    base = pd.read_csv(INPUT, dtype={'symbol': str})
    base['date'] = pd.to_datetime(base['date']).dt.normalize()
    first = base[(base['extraction_count'] == 1) & (base['status'] == '検証完了')].copy()
    print(f'初回抽出・検証完了: {len(first)}件')
    rows = []
    for symbol, group in first.groupby('symbol', sort=True):
        print(f'取得中: {symbol}', flush=True)
        try:
            prices = yf.download(symbol, start='2026-07-01', end='2026-10-30', auto_adjust=False, progress=False, threads=False)
            if prices.empty:
                raise ValueError('株価なし')
            if isinstance(prices.columns, pd.MultiIndex):
                prices.columns = prices.columns.get_level_values(0)
            prices.index = pd.to_datetime(prices.index).tz_localize(None).normalize()
            prices = prices.sort_index()
            close = pd.to_numeric(prices['Close'], errors='coerce').dropna()
            for _, r in group.iterrows():
                item = r.to_dict()
                # 当日の終値が未確定の通知でも未来情報を混入させないよう、前営業日までを使用。
                hist = close[close.index < r['date']]
                item.update(trend='判定不能', reason='', prior_close=np.nan, ret5_before_pct=np.nan,
                            bias25_before_pct=np.nan, slope10_before_pct=np.nan, slope20_before_pct=np.nan)
                if len(hist) < 30:
                    item['reason'] = f'過去データ不足: {len(hist)}日'
                    rows.append(item)
                    continue
                p = float(hist.iloc[-1]); ma10 = hist.rolling(10).mean(); ma20 = hist.rolling(20).mean(); ma25 = hist.rolling(25).mean()
                ret5 = (p / float(hist.iloc[-6]) - 1)*100
                bias25 = (p / float(ma25.iloc[-1]) - 1)*100
                slope10 = (float(ma10.iloc[-1])/float(ma10.iloc[-6])-1)*100
                slope20 = (float(ma20.iloc[-1])/float(ma20.iloc[-6])-1)*100
                item.update(prior_close=round(p,2),ret5_before_pct=round(ret5,3),bias25_before_pct=round(bias25,3),
                            slope10_before_pct=round(slope10,3),slope20_before_pct=round(slope20,3))
                # 検証用の仮分類。実際のAIのSlope定義とは別物。
                if slope10 < 0 and slope20 < 0 and bias25 < 0:
                    item['trend'] = 'C_下落継続'
                elif slope10 > 0 and slope20 <= 0:
                    item['trend'] = 'A_上昇転換候補'
                elif slope10 >= 0 and slope20 > 0 and (ret5 > 5 or bias25 > 8):
                    item['trend'] = 'D_上昇済み候補'
                elif abs(slope10) <= 0.5 and abs(slope20) <= 0.5:
                    item['trend'] = 'B_横ばい候補'
                else:
                    item['trend'] = 'その他_混合'
                rows.append(item)
        except Exception as exc:
            print(f'取得失敗: {symbol}: {exc}', flush=True)
            for _, r in group.iterrows():
                item = r.to_dict(); item.update(trend='判定不能', reason=str(exc), prior_close=np.nan,ret5_before_pct=np.nan,
                                                bias25_before_pct=np.nan,slope10_before_pct=np.nan,slope20_before_pct=np.nan)
                rows.append(item)
        time.sleep(0.3)
    result = pd.DataFrame(rows).sort_values(['date','symbol'])
    result['date'] = result['date'].dt.strftime('%Y-%m-%d')
    result.to_csv(OUTPUT, index=False, encoding='utf-8-sig')
    valid = result[result['trend'] != '判定不能'].copy()
    summary = valid.groupby('trend', as_index=False).agg(count=('symbol','size'),
                  wins=('return_5d_pct',lambda x:int((x>0).sum())),win_rate_pct=('return_5d_pct',lambda x:round((x>0).mean()*100,2)),
                  avg_return_5d_pct=('return_5d_pct','mean'),avg_max_gain_pct=('max_gain_pct','mean'),avg_max_loss_pct=('max_loss_pct','mean'))
    for col in ['avg_return_5d_pct','avg_max_gain_pct','avg_max_loss_pct']:
        summary[col] = summary[col].round(2)
    summary.to_csv(SUMMARY,index=False,encoding='utf-8-sig')
    print('\n=== 初回抽出・事前トレンド別検証 ===')
    print(summary.to_string(index=False))
    print(f'判定不能: {len(result)-len(valid)}件 / 全{len(result)}件')
    print(f'保存: {OUTPUT}, {SUMMARY}')

if __name__ == '__main__':
    main()
