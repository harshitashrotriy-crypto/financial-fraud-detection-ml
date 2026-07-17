import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# ==========================================
# Load Dataset
# ==========================================

file_path = (
    Path(__file__).parent.parent
    / "Data"
    / "Processed"
    / "processed_fraud_data.csv"
)

df = pd.read_csv(file_path)

# Use sample for faster training
df = df.sample(n=500000, random_state=42)

# Features & Target
X = df.drop("isFraud", axis=1)
y = df["isFraud"]

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Train Random Forest
model = RandomForestClassifier(
    n_estimators=50,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)

print("Training model...")
model.fit(X_train, y_train)

# ==========================================
# Save Model
# ==========================================

models_path = Path(__file__).parent.parent / "Models"
models_path.mkdir(exist_ok=True)

model_file = models_path / "fraud_detection_model.pkl"

joblib.dump(model, model_file)

print("\nModel saved successfully!")
print(model_file)

