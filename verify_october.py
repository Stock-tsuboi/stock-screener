import pandas as pd
import yfinance as yf
import time

# 検証設定
START_DATE = "2026-10-01"
END_DATE = "2026-10-31"
PRICE_END = "2026-10-30"
HOLD_DAYS = 5

HISTORY_URL = (
    "https://raw.githubusercontent.com/"
    "Stock-tsuboi/stock-screener/main/"
    "recommendation_history.csv"
)

OUTPUT_FILE = "october_verification.csv"


def main():
    print("=== 10月AI抽出銘柄 実績検証開始 ===")

    history = pd.read_csv(HISTORY_URL)
    history["date"] = pd.to_datetime(history["date"])
    history["entry_price"] = pd.to_numeric(
        history["entry_price"], errors="coerce"
    )

    history = history[
        history["date"].between(START_DATE, END_DATE)
    ].copy()

    history = history.sort_values(
        ["symbol", "date"], kind="stable"
    )

    history["extraction_count"] = (
        history.groupby("symbol").cumcount() + 1
    )

    results = []

    for symbol, group in history.groupby("symbol"):
        print(f"株価取得中: {symbol}")

        try:
            data = yf.download(
                symbol,
                start=START_DATE,
                end=PRICE_END,
                auto_adjust=False,
                progress=False,
                threads=False
            )

            if data.empty:
                raise ValueError("株価データなし")

            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            data.index = pd.to_datetime(data.index)
            data.index = data.index.tz_localize(None).normalize()
            data = data.sort_index()

            for _, row in group.iterrows():
                entry = row["entry_price"]
                date = row["date"]

                record = {
                    "date": date.strftime("%Y-%m-%d"),
                    "symbol": symbol,
                    "entry_price": entry,
                    "extraction_count": row["extraction_count"],
                    "status": "未判定",
                    "days_observed": 0,
                    "return_5d_pct": None,
                    "max_gain_pct": None,
                    "max_loss_pct": None,
                    "final_close": None,
                }

                # 抽出翌営業日から検証する
                future = data[data.index > date].head(HOLD_DAYS)

                record["days_observed"] = len(future)

                if pd.isna(entry) or entry <= 0:
                    record["status"] = "抽出価格不正"

                elif future.empty:
                    record["status"] = "翌営業日データなし"

                else:
                    high = future["High"].max()
                    low = future["Low"].min()
                    close = future["Close"].iloc[-1]

                    record["max_gain_pct"] = round(
                        (high / entry - 1) * 100, 2
                    )

                    record["max_loss_pct"] = round(
                        (low / entry - 1) * 100, 2
                    )

                    record["final_close"] = round(float(close), 2)

                    if len(future) == HOLD_DAYS:
                        record["return_5d_pct"] = round(
                            (close / entry - 1) * 100, 2
                        )
                        record["status"] = "検証完了"
                    else:
                        record["status"] = "5営業日未経過"

                results.append(record)

        except Exception as e:
            print(f"取得失敗: {symbol} / {e}")

            for _, row in group.iterrows():
                results.append({
                    "date": row["date"].strftime("%Y-%m-%d"),
                    "symbol": symbol,
                    "entry_price": row["entry_price"],
                    "extraction_count": row["extraction_count"],
                    "status": "取得失敗",
                    "days_observed": 0,
                    "return_5d_pct": None,
                    "max_gain_pct": None,
                    "max_loss_pct": None,
                    "final_close": None,
                })

        time.sleep(0.3)

    result = pd.DataFrame(results)
    result = result.sort_values(["date", "symbol"])

    result.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    valid = result[result["status"] == "検証完了"]

    print("\n=== 検証結果 ===")
    print(f"抽出記録数: {len(result)}")
    print(f"検証完了: {len(valid)}")
    print(f"未完了・取得失敗: {len(result) - len(valid)}")

    if not valid.empty:
        print(
            f"5営業日後勝率: "
            f"{(valid['return_5d_pct'] > 0).mean() * 100:.2f}%"
        )

        print(
            f"平均騰落率: "
            f"{valid['return_5d_pct'].mean():.2f}%"
        )

        print(
            f"平均最大上昇率: "
            f"{valid['max_gain_pct'].mean():.2f}%"
        )

        print(
            f"平均最大下落率: "
            f"{valid['max_loss_pct'].mean():.2f}%"
        )

    print(f"\n保存完了: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
