import pandas as pd
from pathlib import Path

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

print("=" * 70)
print("FEATURE CORRELATION WITH FRAUD")
print("=" * 70)

# Correlation with target
correlation = (
    df.corr(numeric_only=True)["isFraud"]
      .sort_values(ascending=False)
)

print(correlation)