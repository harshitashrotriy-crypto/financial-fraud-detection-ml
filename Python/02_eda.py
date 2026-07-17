import pandas as pd
from pathlib import Path

# -------------------------------
# Load Dataset
# -------------------------------

file_path = Path(__file__).parent.parent / "Data" / "Raw" / "PS_20174392719_1491204439457_log.csv"

df = pd.read_csv(file_path)

# -------------------------------
# Basic Information
# -------------------------------

print("=" * 80)
print("DATASET INFORMATION")
print("=" * 80)

print("\nShape:")
print(df.shape)

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nSummary Statistics:")
print(df.describe())

print("\nFraud Distribution:")
print(df["isFraud"].value_counts())

print("\nFraud Percentage:")
print((df["isFraud"].value_counts(normalize=True) * 100).round(4))