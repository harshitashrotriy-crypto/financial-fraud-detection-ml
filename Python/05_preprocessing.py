import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ======================================
# Load Processed Dataset
# ======================================

file_path = (
    Path(__file__).parent.parent
    / "Data"
    / "Processed"
    / "processed_fraud_data.csv"
)

df = pd.read_csv(file_path)

print("Processed Dataset Loaded")
print("=" * 60)

# ======================================
# Drop unnecessary columns
# ======================================

df = df.drop(
    columns=["nameOrig", "nameDest"],
    errors="ignore"
)

print("Dropped nameOrig and nameDest")

# ======================================
# Encode transaction type
# ======================================

encoder = LabelEncoder()

df["type"] = encoder.fit_transform(df["type"])

print("\nTransaction Type Encoding:")

for original, encoded in zip(
    encoder.classes_,
    encoder.transform(encoder.classes_)
):
    print(f"{original} --> {encoded}")

# ======================================
# Features and Target
# ======================================

X = df.drop("isFraud", axis=1)

y = df["isFraud"]

# ======================================
# Train Test Split
# ======================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n" + "=" * 60)
print("Train Test Split Completed")
print("=" * 60)

print(f"Training Samples : {len(X_train):,}")
print(f"Testing Samples  : {len(X_test):,}")

print("\nTraining Fraud Distribution")

print(y_train.value_counts())

print("\nTesting Fraud Distribution")

print(y_test.value_counts())

# ======================================
# Save Preprocessed Dataset
# ======================================

# ======================================
# Save Preprocessed Dataset
# ======================================

processed_path = (
    Path(__file__).parent.parent
    / "Data"
    / "Processed"
    / "processed_fraud_data.csv"
)

df.to_csv(processed_path, index=False)

print("\nPreprocessed dataset saved successfully!")
print(processed_path)
