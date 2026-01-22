import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)
import joblib

# =========================================================
# 設定
# =========================================================

DATA_FILE = "training_data.csv"
MODEL_FILE = "model.pkl"
THRESHOLD_FILE = "best_threshold.txt"

# =========================================================
# AI閾値の自動調整ロジック
# =========================================================

def find_best_threshold(model, X, y):
    """
    AI確率の最適閾値を自動で決定する関数
    Precision・Recall・F1 のバランスで最適値を選ぶ
    """

    probs = model.predict_proba(X)[:, 1]
    thresholds = np.arange(0.50, 0.91, 0.01)

    best_th = 0.50
    best_f1 = -1

    for th in thresholds:
        preds = (probs >= th).astype(int)

        tp = np.sum((preds == 1) & (y == 1))
        fp = np.sum((preds == 1) & (y == 0))
        fn = np.sum((preds == 0) & (y == 1))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        if f1 > best_f1:
            best_f1 = f1
            best_th = th

    print("\n===== AI閾値 自動調整結果 =====")
    print(f"最適閾値: {best_th:.2f}")
    print(f"F1スコア: {best_f1:.4f}")

    return best_th

# =========================================================
# 学習データ読み込み
# =========================================================

print("学習データ読み込み中...")
df = pd.read_csv(DATA_FILE, low_memory=False)

# 不要な列を削除
drop_cols = ["コード", "銘柄名", "日付", "5日後上昇率"]
df = df.drop(columns=drop_cols, errors="ignore")

# 欠損値除去
df = df.dropna()

# =========================================================
# 特徴量とラベルに分割
# =========================================================

X = df.drop(columns=["ラベル"])
y = df["ラベル"]

# =========================================================
# 学習データとテストデータに分割
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    shuffle=True,
    stratify=y   # ← ラベルの偏りを維持したまま分割
)

# =========================================================
# モデル構築（RandomForest）
# クラス不均衡を補正 → 上昇ラベルの検出率が改善
# 内部並列を無効化 → sklearn の警告が完全に消える
# =========================================================

print("モデル学習中...")

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=1,               # ← 内部Parallelを無効化（警告ゼロ）
    class_weight="balanced" # ← クラス不均衡を補正（最重要）
)

model.fit(X_train, y_train)

# =========================================================
# モデル評価
# =========================================================

print("\n===== モデル評価 =====")

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# =========================================================
# AI閾値の自動調整
# =========================================================

best_th = find_best_threshold(model, X_test, y_test)

with open(THRESHOLD_FILE, "w") as f:
    f.write(str(best_th))

print(f"\n最適AI閾値を保存しました → {THRESHOLD_FILE}")

# =========================================================
# モデル保存
# =========================================================

joblib.dump(model, MODEL_FILE)
print(f"\n学習済みモデルを保存しました → {MODEL_FILE}")
