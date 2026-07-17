import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ======================================
# Model Performance Results
# ======================================

results = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
    "Random Forest": [1.0000, 1.0000, 0.9845, 0.9922],
    "XGBoost": [1.0000, 0.9844, 0.9767, 0.9805]
})

print("=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(results)

# ======================================
# Save Results as CSV
# ======================================

results_path = (
    Path(__file__).parent.parent
    / "Reports"
    / "Results"
)

results_path.mkdir(parents=True, exist_ok=True)

results.to_csv(
    results_path / "model_comparison.csv",
    index=False
)

# ======================================
# Plot Comparison
# ======================================

plt.figure(figsize=(10,6))

x = range(len(results))

bar_width = 0.35

plt.bar(
    [i - bar_width/2 for i in x],
    results["Random Forest"],
    width=bar_width,
    label="Random Forest"
)

plt.bar(
    [i + bar_width/2 for i in x],
    results["XGBoost"],
    width=bar_width,
    label="XGBoost"
)

plt.xticks(x, results["Metric"])

plt.ylim(0.95, 1.01)

plt.ylabel("Score")
plt.title("Random Forest vs XGBoost")

plt.legend()

plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()

plt.savefig(
    results_path / "model_comparison.png",
    dpi=300
)

plt.show()

print("\nModel comparison saved successfully!")
print(results_path)