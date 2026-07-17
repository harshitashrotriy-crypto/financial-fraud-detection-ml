import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# =====================================================
# Load Processed Dataset
# =====================================================

file_path = (
    Path(__file__).parent.parent
    / "Data"
    / "Processed"
    / "processed_fraud_data.csv"
)

df = pd.read_csv(file_path)

print("Dataset Loaded Successfully!")
print("=" * 60)

# =====================================================
# Use Sample for Faster Training
# =====================================================

df = df.sample(
    n=500000,
    random_state=42
)

print(f"Using Sample Size : {len(df):,}")

# =====================================================
# Drop Unnecessary Columns
# =====================================================

df.drop(["nameOrig", "nameDest"], axis=1, inplace=True)

# =====================================================
# Encode Transaction Type
# =====================================================

encoder = LabelEncoder()

df["type"] = encoder.fit_transform(df["type"])

print("\nTransaction Type Encoding:")

for original, encoded in zip(
    encoder.classes_,
    encoder.transform(encoder.classes_)
):
    print(f"{original} --> {encoded}")

# =====================================================
# Features & Target
# =====================================================

X = df.drop("isFraud", axis=1)
y = df["isFraud"]

# =====================================================
# Train-Test Split
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTrain Samples :", len(X_train))
print("Test Samples  :", len(X_test))

# =====================================================
# Build Random Forest Model
# =====================================================

print("\nTraining Random Forest...")

model = RandomForestClassifier(
    n_estimators=50,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Training Completed!")

# =====================================================
# Prediction
# =====================================================

y_pred = model.predict(X_test)

# =====================================================
# Evaluation
# =====================================================

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(f"Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision : {precision_score(y_test, y_pred):.4f}")
print(f"Recall    : {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score  : {f1_score(y_test, y_pred):.4f}")

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report")
print(classification_report(y_test, y_pred))