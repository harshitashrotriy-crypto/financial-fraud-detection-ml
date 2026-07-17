import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.ticker import FuncFormatter

# -------------------------------------------------
# Load Dataset
# -------------------------------------------------

file_path = Path(__file__).parent.parent / "Data" / "Raw" / "PS_20174392719_1491204439457_log.csv"

df = pd.read_csv(file_path)

# Create Reports/Figures folder if it doesn't exist
figures_path = Path(__file__).parent.parent / "Reports" / "Figures"
figures_path.mkdir(parents=True, exist_ok=True)

print("Dataset Loaded Successfully!")

# =================================================
# GRAPH 1 : Transaction Type Distribution
# =================================================

plt.figure(figsize=(9,6))

ax = df["type"].value_counts().plot(
    kind="bar",
    color="steelblue",
    edgecolor="black"
)

plt.title("Transaction Type Distribution", fontsize=16, fontweight="bold")
plt.xlabel("Transaction Type", fontsize=12)
plt.ylabel("Number of Transactions (Millions)", fontsize=12)

# Y-axis in Millions
ax.yaxis.set_major_formatter(
    FuncFormatter(lambda x, pos: f"{x/1e6:.1f}M")
)

# Labels on top of bars
for p in ax.patches:

    height = p.get_height()

    if height >= 1_000_000:
        label = f"{height/1e6:.2f}M"
    elif height >= 1000:
        label = f"{height/1000:.1f}K"
    else:
        label = str(int(height))

    ax.annotate(
        label,
        (p.get_x() + p.get_width()/2, height),
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold"
    )

plt.tight_layout()

plt.savefig(
    figures_path / "transaction_type_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# =================================================
# GRAPH 2 : Fraud vs Non-Fraud
# =================================================

plt.figure(figsize=(6,5))

ax = df["isFraud"].value_counts().plot(
    kind="bar",
    color=["royalblue", "crimson"],
    edgecolor="black"
)

plt.title("Fraud vs Non-Fraud Transactions", fontsize=16, fontweight="bold")
plt.xlabel("")
plt.ylabel("Number of Transactions (Millions)", fontsize=12)

plt.xticks([0,1], ["Not Fraud", "Fraud"], rotation=0)

# Y-axis in Millions
ax.yaxis.set_major_formatter(
    FuncFormatter(lambda x, pos: f"{x/1e6:.1f}M")
)

# Labels on bars
for p in ax.patches:

    height = p.get_height()

    if height >= 1_000_000:
        label = f"{height/1e6:.2f}M"
    elif height >= 1000:
        label = f"{height/1000:.1f}K"
    else:
        label = str(int(height))

    ax.annotate(
        label,
        (p.get_x() + p.get_width()/2, height),
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold"
    )

plt.tight_layout()

plt.savefig(
    figures_path / "fraud_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nGraphs saved successfully in Reports/Figures")