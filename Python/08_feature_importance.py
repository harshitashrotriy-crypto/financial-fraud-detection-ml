import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# ===========================================
# Load Dataset
# ===========================================

file_path = (
    Path(__file__).parent.parent
    / "Data"
    / "Processed"
    / "processed_fraud_data.csv"
)

df = pd.read_csv(file_path)

print("Dataset Loaded Successfully!")

# ===========================================
# Use Sample
# ===========================================

df = df.sample(
    n=500000,
    random_state=42
)

# ===========================================
# Encode Type (only if needed)
# ===========================================

if df["type"].dtype == "object":
    encoder = LabelEncoder()
    df["type"] = encoder.fit_transform(df["type"])

# ===========================================
# Features & Target
# ===========================================

X = df.drop("isFraud", axis=1)

y = df["isFraud"]

# ===========================================
# Train-Test Split
# ===========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ===========================================
# Train Random Forest
# ===========================================

model = RandomForestClassifier(
    n_estimators=50,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)

print("Training Model...")

model.fit(X_train, y_train)

print("Training Completed!")

# ===========================================
# Feature Importance
# ===========================================

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 10 Most Important Features\n")

print(importance.head(10))


# ===========================================
# Plot
# ===========================================

# Folder to save figures
figures_path = (
    Path(__file__).parent.parent
    / "Reports"
    / "Figures"
)

figures_path.mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(10,6))

plt.barh(
    importance["Feature"][:10],
    importance["Importance"][:10],
    color="steelblue",
    edgecolor="black"
)

plt.gca().invert_yaxis()

plt.title("Top 10 Feature Importance", fontsize=14, fontweight="bold")
plt.xlabel("Importance")

plt.tight_layout()

# Save graph
plt.savefig(
    figures_path / "feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nFeature importance graph saved successfully!")
print(figures_path / "feature_importance.png")