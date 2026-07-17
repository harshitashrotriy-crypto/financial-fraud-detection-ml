import pandas as pd
from pathlib import Path

# -----------------------------
# Load Dataset
# -----------------------------

file_path = Path(__file__).parent.parent / "Data" / "Raw" / "PS_20174392719_1491204439457_log.csv"

df = pd.read_csv(file_path)

print("Dataset Loaded!")
print("="*60)

# ===========================================
# FEATURE 1 : Sender Balance Difference
# ===========================================

df["sender_balance_diff"] = (
    df["oldbalanceOrg"] - df["newbalanceOrig"]
)

# ===========================================
# FEATURE 2 : Receiver Balance Difference
# ===========================================

df["receiver_balance_diff"] = (
    df["newbalanceDest"] - df["oldbalanceDest"]
)

# ===========================================
# FEATURE 3 : Sender Account Emptied
# ===========================================

df["sender_account_emptied"] = (
    df["newbalanceOrig"] == 0
).astype(int)

# ===========================================
# FEATURE 4 : Amount to Balance Ratio
# ===========================================

df["amount_to_balance_ratio"] = (
    df["amount"] / (df["oldbalanceOrg"] + 1)
)

# ===========================================
# FEATURE 5 : High Value Transaction
# ===========================================

df["high_value_transaction"] = (
    df["amount"] > 200000
).astype(int)

# ===========================================
# FEATURE 6 : Transaction Hour
# ===========================================

df["transaction_hour"] = df["step"] % 24

# ===========================================
# FEATURE 7 : Transaction Day
# ===========================================

df["transaction_day"] = (df["step"] // 24) + 1

print(df[[
    "amount",
    "sender_balance_diff",
    "receiver_balance_diff",
    "sender_account_emptied",
    "amount_to_balance_ratio",
    "high_value_transaction",
    "transaction_hour",
    "transaction_day"
]].head())


# ===========================================
# Save Processed Dataset
# ===========================================

processed_path = (
    Path(__file__).parent.parent
    / "Data"
    / "Processed"
    / "processed_fraud_data.csv"
)

df.to_csv(processed_path, index=False)

print("\nProcessed dataset saved successfully!")
print(processed_path)

