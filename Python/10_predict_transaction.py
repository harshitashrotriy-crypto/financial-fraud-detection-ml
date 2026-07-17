import joblib
import pandas as pd
from pathlib import Path

# ======================================
# Load Saved Model
# ======================================

model_path = (
    Path(__file__).parent.parent
    / "Models"
    / "fraud_detection_model.pkl"
)

model = joblib.load(model_path)

print("Model Loaded Successfully!")

# ======================================
# Sample Transaction
# ======================================

transaction = {
    "step": 1,
    "type": 4,                    # TRANSFER
    "amount": 500000,
    "oldbalanceOrg": 500000,
    "newbalanceOrig": 0,
    "oldbalanceDest": 0,
    "newbalanceDest": 500000,
    "isFlaggedFraud": 0,
    "sender_balance_diff": 500000,
    "receiver_balance_diff": 500000,
    "sender_account_emptied": 1,
    "amount_to_balance_ratio": 1.0,
    "high_value_transaction": 1,
    "transaction_hour": 12,
    "transaction_day": 1
}

df = pd.DataFrame([transaction])

# ======================================
# Prediction
# ======================================

prediction = model.predict(df)[0]

probability = model.predict_proba(df)[0][1]

print("\nPrediction Result")
print("=" * 40)

if prediction == 1:
    print("🚨 FRAUD DETECTED")
else:
    print("✅ LEGITIMATE TRANSACTION")

print(f"Fraud Probability : {probability:.2%}")