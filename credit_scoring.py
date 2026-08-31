import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve
)

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR, "dataset", "cs-training.csv"
)

MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULT_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 65)
print("OPTIMIZED CREDIT SCORING MODEL")
print("=" * 65)

df = pd.read_csv(DATA_PATH)

print("\nOriginal Shape:", df.shape)

# Remove index column
if "Unnamed: 0" in df.columns:
    df.drop("Unnamed: 0", axis=1, inplace=True)


# ============================================================
# DATA CLEANING
# ============================================================

# Replace infinite values
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Remove invalid ages
df = df[df["age"] > 0]

# Separate target
TARGET = "SeriousDlqin2yrs"

X = df.drop(TARGET, axis=1)
y = df[TARGET]

print("\nTarget Distribution:")
print(y.value_counts())

print("\nTarget Percentage:")
print(
    (y.value_counts(normalize=True) * 100).round(2)
)


# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Shape:", X_train.shape)
print("Testing Shape :", X_test.shape)


# ============================================================
# IMPUTATION
# ============================================================

imputer = SimpleImputer(strategy="median")

X_train = pd.DataFrame(
    imputer.fit_transform(X_train),
    columns=X_train.columns,
    index=X_train.index
)

X_test = pd.DataFrame(
    imputer.transform(X_test),
    columns=X_test.columns,
    index=X_test.index
)


# ============================================================
# RANDOM FOREST - OPTIMIZED
# ============================================================

print("\nTraining Optimized Random Forest...")

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=16,
    min_samples_split=10,
    min_samples_leaf=3,
    max_features="sqrt",
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Training completed!")


# ============================================================
# PROBABILITY PREDICTION
# ============================================================

y_prob = model.predict_proba(X_test)[:, 1]


# ============================================================
# THRESHOLD OPTIMIZATION
# ============================================================

print("\nSearching for best prediction threshold...")

thresholds = np.arange(0.10, 0.71, 0.01)

threshold_results = []

for threshold in thresholds:

    y_pred_temp = (
        y_prob >= threshold
    ).astype(int)

    precision = precision_score(
        y_test,
        y_pred_temp,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred_temp,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred_temp,
        zero_division=0
    )

    threshold_results.append({
        "Threshold": threshold,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    })


threshold_df = pd.DataFrame(threshold_results)

best_row = threshold_df.loc[
    threshold_df["F1"].idxmax()
]

best_threshold = best_row["Threshold"]

print(
    f"\nBest Threshold: {best_threshold:.2f}"
)

print(
    f"Best Precision: {best_row['Precision']:.4f}"
)

print(
    f"Best Recall: {best_row['Recall']:.4f}"
)

print(
    f"Best F1 Score: {best_row['F1']:.4f}"
)


# ============================================================
# FINAL PREDICTIONS
# ============================================================

y_pred = (
    y_prob >= best_threshold
).astype(int)


# ============================================================
# FINAL METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_prob
)


print("\n" + "=" * 65)
print("FINAL OPTIMIZED RESULTS")
print("=" * 65)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 65)
print("CLASSIFICATION REPORT")
print("=" * 65)

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "No Serious Delinquency",
            "Serious Delinquency"
        ],
        zero_division=0
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=[
        "Predicted Good",
        "Predicted Risk"
    ],
    yticklabels=[
        "Actual Good",
        "Actual Risk"
    ]
)

plt.title("Optimized Credit Scoring - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULT_DIR,
        "optimized_confusion_matrix.png"
    ),
    dpi=300
)

plt.show()


# ============================================================
# ROC CURVE
# ============================================================

fpr, tpr, _ = roc_curve(
    y_test,
    y_prob
)

plt.figure(figsize=(8, 6))

plt.plot(
    fpr,
    tpr,
    label=f"Random Forest (AUC = {roc_auc:.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULT_DIR,
        "optimized_roc_curve.png"
    ),
    dpi=300
)

plt.show()


# ============================================================
# PRECISION-RECALL CURVE
# ============================================================

precision_values, recall_values, _ = precision_recall_curve(
    y_test,
    y_prob
)

plt.figure(figsize=(8, 6))

plt.plot(
    recall_values,
    precision_values
)

plt.xlabel("Recall")
plt.ylabel("Precision")

plt.title(
    "Precision-Recall Curve"
)

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULT_DIR,
        "precision_recall_curve.png"
    ),
    dpi=300
)

plt.show()


# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

plt.figure(figsize=(9, 6))

plt.plot(
    threshold_df["Threshold"],
    threshold_df["Precision"],
    label="Precision"
)

plt.plot(
    threshold_df["Threshold"],
    threshold_df["Recall"],
    label="Recall"
)

plt.plot(
    threshold_df["Threshold"],
    threshold_df["F1"],
    label="F1 Score"
)

plt.axvline(
    best_threshold,
    linestyle="--",
    label=f"Best Threshold = {best_threshold:.2f}"
)

plt.xlabel("Decision Threshold")
plt.ylabel("Score")

plt.title("Threshold Optimization")

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULT_DIR,
        "threshold_optimization.png"
    ),
    dpi=300
)

plt.show()


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

feature_importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": model.feature_importances_
})

feature_importance.sort_values(
    by="Importance",
    ascending=False,
    inplace=True
)

print("\n" + "=" * 65)
print("TOP 10 IMPORTANT FEATURES")
print("=" * 65)

print(
    feature_importance.head(10).to_string(
        index=False
    )
)

feature_importance.to_csv(
    os.path.join(
        RESULT_DIR,
        "feature_importance.csv"
    ),
    index=False
)


# ============================================================
# SAVE MODEL + IMPUTER + THRESHOLD
# ============================================================

model_package = {
    "model": model,
    "imputer": imputer,
    "threshold": best_threshold,
    "features": list(X_train.columns)
}

model_path = os.path.join(
    MODEL_DIR,
    "credit_scoring_optimized.pkl"
)

joblib.dump(
    model_package,
    model_path
)

print("\nOptimized model saved:")
print(model_path)


# ============================================================
# SAVE METRICS
# ============================================================

metrics = pd.DataFrame([{
    "Model": "Optimized Random Forest",
    "Threshold": best_threshold,
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1 Score": f1,
    "ROC-AUC": roc_auc
}])

metrics.to_csv(
    os.path.join(
        RESULT_DIR,
        "final_metrics.csv"
    ),
    index=False
)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 65)
print("TASK 1 OPTIMIZATION COMPLETED")
print("=" * 65)