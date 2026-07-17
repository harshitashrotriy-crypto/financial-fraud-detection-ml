import pandas as pd
from pathlib import Path

# Dataset path
file_path = Path(__file__).parent.parent / "Data" / "Raw" / "PS_20174392719_1491204439457_log.csv"

# Load complete dataset
df = pd.read_csv(file_path)

print("=" * 80)
print("DATASET LOADED SUCCESSFULLY")
print("=" * 80)

print(f"Rows    : {df.shape[0]:,}")
print(f"Columns : {df.shape[1]}")

print("\nColumn Names:")
print(df.columns.tolist())

print("\nFirst 5 Rows:")
print(df.head())