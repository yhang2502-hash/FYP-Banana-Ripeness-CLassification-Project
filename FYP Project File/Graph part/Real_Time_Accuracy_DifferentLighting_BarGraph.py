import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# =========================
# 1. Load CSV
# =========================
# change to your own path if needed
csv_path = r"D:\Y4 FYP file\Code Learning\snapshots\record_result_log.csv"
df = pd.read_csv(csv_path)

# =========================
# 2. Clean percentage columns
# =========================
acc_cols = [
    "method1_recorded_accuracy",
    "method2_recorded_accuracy",
    "method3_recorded_accuracy",
    "best_method_accuracy"
]

for col in acc_cols:
    df[col] = (
        df[col]
        .astype(str)
        .str.replace("%", "", regex=False)
        .str.strip()
    )
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Optional: remove rows with no result
df = df.dropna(subset=[
    "method1_recorded_accuracy",
    "method2_recorded_accuracy",
    "method3_recorded_accuracy"
])

# =========================
# 3. Overall mean real-time accuracy
# =========================
overall_means = [
    df["method1_recorded_accuracy"].mean(),
    df["method2_recorded_accuracy"].mean(),
    df["method3_recorded_accuracy"].mean()
]

method_names = ["Method 1\nRaw", "Method 2\nMasked + CLAHE",
                "Method 3\nIsolated + CLAHE"]

plt.figure(figsize=(8, 5))
bars = plt.bar(method_names, overall_means)
plt.ylabel("Average Real-Time Accuracy (%)")
plt.title("Average Real-Time Accuracy Comparison of Three Methods")
plt.ylim(0, 100)

for bar, val in zip(bars, overall_means):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        val + 1,
        f"{val:.2f}%",
        ha="center",
        va="bottom",
        fontsize=10
    )

plt.tight_layout()
plt.show()

# =========================
# 4. Accuracy under different lighting
# =========================
lighting_df = df[df["lighting_mode"].isin(["Warm", "Cool"])].copy()

lighting_means = lighting_df.groupby("lighting_mode")[
    ["method1_recorded_accuracy", "method2_recorded_accuracy",
        "method3_recorded_accuracy"]
].mean()

# Optional: force Cool then Warm order
lighting_means = lighting_means.reindex(["Cool", "Warm"])

x = np.arange(len(lighting_means.index))
width = 0.24

plt.figure(figsize=(12, 7))

bars1 = plt.bar(
    x - width,
    lighting_means["method1_recorded_accuracy"],
    width,
    label="Method 1"
)
bars2 = plt.bar(
    x,
    lighting_means["method2_recorded_accuracy"],
    width,
    label="Method 2"
)
bars3 = plt.bar(
    x + width,
    lighting_means["method3_recorded_accuracy"],
    width,
    label="Method 3"
)

plt.xticks(x, lighting_means.index, fontsize=16)
plt.yticks(fontsize=14)
plt.ylabel("Average Real-Time Accuracy (%)", fontsize=17)
plt.xlabel("Lighting Mode", fontsize=17)
plt.title("Real-Time Accuracy under Different Lighting Conditions",
          fontsize=22, fontweight="bold")
plt.ylim(0, 103)

plt.legend(fontsize=15)

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        h = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            h + 1,
            f"{h:.2f}%",
            ha="center",
            va="bottom",
            fontsize=15
        )

plt.tight_layout()
plt.show()

# =========================
# 5. Recommended method count
# =========================
recommend_counts = df["recommended_method"].value_counts()

plt.figure(figsize=(8, 5))
bars = plt.bar(recommend_counts.index, recommend_counts.values)
plt.ylabel("Number of Times Recommended")
plt.title("Recommended Method Frequency in Real-Time Testing")

for bar, val in zip(bars, recommend_counts.values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        val + 0.2,
        str(val),
        ha="center",
        va="bottom",
        fontsize=10
    )

plt.tight_layout()
plt.show()
