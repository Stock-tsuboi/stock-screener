# =========================================================
# Step0　Import
# =========================================================
import duckdb
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

# =========================================================
# Step1　設定
# =========================================================
DB_FILE = "market.db"
MODEL_FILE = "model.pkl"
THRESHOLD_FILE = "best_threshold.txt"

# =========================================================
# Step2　DBからデータ取得
# =========================================================
print("DB接続中...")

con = duckdb.connect(DB_FILE)

query = """
SELECT *
FROM stock_price
ORDER BY code, date
"""

df = con.execute(query).fetchdf()

print(f"✔ 取得件数: {len(df)}")

# =========================================================
# Step3　特徴量生成
# =========================================================
print("特徴量生成中...")

df = df.sort_values(["code", "date"])

# リターン系
df["return_1"] = df.groupby("code")["close"].pct_change(1)
df["return_5"] = df.groupby("code")["close"].pct_change(5)

# 移動平均
df["ma5"] = df.groupby("code")["close"].transform(lambda x: x.rolling(5).mean())
df["ma20"] = df.groupby("code")["close"].transform(lambda x: x.rolling(20).mean())

# ボラティリティ
df["volatility"] = df.groupby("code")["return_1"].transform(lambda x: x.rolling(5).std())

# =========================================================
# Step4　ラベル作成（5日後上昇）
# =========================================================
print("ラベル作成中...")

df["future_return"] = df.groupby("code")["close"].shift(-5) / df["close"] - 1
df["label"] = (df["future_return"] > 0).astype(int)

# =========================================================
# Step5　不要データ削除
# =========================================================
df = df.dropna()

print(f"✔ 学習データ件数: {len(df)}")

# =========================================================
# Step6　特徴量とラベル分離
# =========================================================
features = [
    "return_1",
    "return_5",
    "ma5",
    "ma20",
    "volatility"
]

X = df[features]
y = df["label"]

# =========================================================
# Step7　モデル学習
# =========================================================
print("RandomForest 学習開始...")

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

model.fit(X, y)

print("✔ 学習完了")

# =========================================================
# Step8　閾値最適化
# =========================================================
print("閾値最適化中...")

probs = model.predict_proba(X)[:, 1]

best_th = 0.5
best_score = -1

for th in np.arange(0.50, 0.91, 0.01):
    preds = (probs >= th).astype(int)

    tp = np.sum((preds == 1) & (y == 1))
    fp = np.sum((preds == 1) & (y == 0))
    fn = np.sum((preds == 0) & (y == 1))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    if precision + recall == 0:
        score = 0
    else:
        score = 2 * precision * recall / (precision + recall)

    if score > best_score:
        best_score = score
        best_th = th

print(f"✔ 最適閾値: {best_th:.2f}")

# =========================================================
# Step9　保存
# =========================================================
joblib.dump(model, MODEL_FILE)

with open(THRESHOLD_FILE, "w") as f:
    f.write(str(best_th))

print("✔ model.pkl 保存完了")
print("✔ best_threshold.txt 保存完了")
