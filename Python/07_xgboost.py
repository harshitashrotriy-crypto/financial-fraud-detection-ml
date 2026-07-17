import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from xgboost import XGBClassifier

# =====================================
# Load Processed Dataset
# =====================================

file_path = (
    Path(__file__).parent.parent
    / "Data"
    / "Processed"
    / "processed_fraud_data.csv"
)

df = pd.read_csv(file_path)

print("Dataset Loaded Successfully!")
print("=" * 55)

# =====================================
# Use Sample (for faster training)
# =====================================

sample_size = 500000

if len(df) > sample_size:
    df = df.sample(n=sample_size, random_state=42)

print(f"Using Sample Size : {len(df):,}")

# =====================================
# Features & Target
# =====================================
print(df.dtypes)

X = df.drop("isFraud", axis=1)
y = df["isFraud"]

# =====================================
# Train Test Split
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print()
print(f"Train Samples : {len(X_train):,}")
print(f"Test Samples  : {len(X_test):,}")

# =====================================
# Train XGBoost
# =====================================

print("\nTraining XGBoost...")

model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    eval_metric="logloss",
)

model.fit(X_train, y_train)

print("Training Completed!")

# =====================================
# Prediction
# =====================================

y_pred = model.predict(X_test)

# =====================================
# Evaluation
# =====================================

print("\n" + "=" * 55)
print("MODEL PERFORMANCE")
print("=" * 55)

print(f"Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision : {precision_score(y_test, y_pred):.4f}")
print(f"Recall    : {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score  : {f1_score(y_test, y_pred):.4f}")

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report")
print(classification_report(y_test, y_pred))