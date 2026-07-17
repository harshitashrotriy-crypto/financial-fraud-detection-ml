import pandas as pd
from pathlib import Path

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

# =====================================
# Load Dataset
# =====================================

file_path = (
    Path(__file__).parent.parent
    / "Data"
    / "Processed"
    / "processed_fraud_data.csv"
)

df = pd.read_csv(file_path)

print("Dataset Loaded Successfully!")

# =====================================
# Sample Data
# =====================================

sample_size = 100000

if len(df) > sample_size:
    df = df.sample(sample_size, random_state=42)

# =====================================
# Features & Target
# =====================================

X = df.drop("isFraud", axis=1)
y = df["isFraud"]

# =====================================
# Train Model
# =====================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

print("\nRunning 5-Fold Cross Validation...")

scores = cross_val_score(
    model,
    X,
    y,
    cv=5,
    scoring="f1"
)

print("\nCross Validation Scores")

for i, score in enumerate(scores, start=1):
    print(f"Fold {i}: {score:.4f}")

print("\nAverage F1 Score:", round(scores.mean(), 4))
print("Standard Deviation:", round(scores.std(), 4))

