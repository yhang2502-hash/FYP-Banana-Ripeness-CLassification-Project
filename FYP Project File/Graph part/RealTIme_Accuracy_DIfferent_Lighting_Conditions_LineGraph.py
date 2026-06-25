import pandas as pd
import matplotlib.pyplot as plt

# =========================
# 1. Load CSV
# =========================
csv_path = r"D:\Y4 FYP file\Code Learning\snapshots\record_result_log.csv"
df = pd.read_csv(csv_path)

# =========================
# 2. Clean accuracy columns
# =========================
acc_cols = [
    "method1_recorded_accuracy",
    "method2_recorded_accuracy",
    "method3_recorded_accuracy"
]

for col in acc_cols:
    df[col] = (
        df[col]
        .astype(str)
        .str.replace("%", "", regex=False)
        .str.strip()
    )
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=acc_cols).reset_index(drop=True)

# Keep only Warm/Cool if available
df = df[df["lighting_mode"].isin(["Warm", "Cool"])].copy()

# Average by lighting mode
lighting_avg = df.groupby("lighting_mode")[
    acc_cols].mean().reindex(["Warm", "Cool"])

plt.figure(figsize=(8, 5))

plt.plot(
    lighting_avg.index,
    lighting_avg["method1_recorded_accuracy"],
    marker="o",
    linewidth=2,
    label="Method 1: Raw"
)

plt.plot(
    lighting_avg.index,
    lighting_avg["method2_recorded_accuracy"],
    marker="s",
    linewidth=2,
    label="Method 2: Masked + CLAHE"
)

plt.plot(
    lighting_avg.index,
    lighting_avg["method3_recorded_accuracy"],
    marker="^",
    linewidth=2,
    label="Method 3: Isolated + CLAHE"
)

plt.title("Real-Time Accuracy under Different Lighting Conditions")
plt.xlabel("Lighting Mode")
plt.ylabel("Average Accuracy (%)")
plt.ylim(0, 100)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()
