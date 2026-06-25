import os
import pandas as pd
import matplotlib.pyplot as plt

# =========================================================
# USER INPUT PART
# Put FOLDER paths here (not direct results.csv path)
# Each folder should contain results.csv
# =========================================================

method1_runs = {
    50:  r"D:\Y4 FYP file\Code Learning\Model_Banana_Method_50&100epoch\Model1_Raw_50epoch",
    100: r"D:\Y4 FYP file\Code Learning\Model_Banana_Method_50&100epoch\Model1_Raw_100epoch",
    150: r"D:\Y4 FYP file\Code Learning\Model_Banana_Method_150epoch\Model1_Raw_150epoch",
    200: r"D:\Y4 FYP file\Code Learning\Model_Banana_Method_200epoch\Model1_Raw_200epoch",
    250: r"D:\Y4 FYP file\Code Learning\Model_Banana_Method_250epoch\Model1_Raw_250epoch"
}

method2_runs = {
    50:  r"D:\Y4 FYP file\Code Learning\Model_Banana_Method_50&100epoch\Model2_Masked_CLAHE_Fair_50epoch",
    100: r"D:\Y4 FYP file\Code Learning\Model_Banana_Method_50&100epoch\Model2_Masked_CLAHE_Fair_100epoch",
    150: r"D:\Y4 FYP file\Code Learning\Model_Banana_Method_150epoch\Model2_Masked_CLAHE_Fair_150epoch",
    200: r"D:\Y4 FYP file\Code Learning\Model_Banana_Method_200epoch\Model2_Masked_CLAHE_Fair_200epoch",
    250: r"D:\Y4 FYP file\Code Learning\Model_Banana_Method_250epoch\Model2_Masked_CLAHE_Fair_250epoch"
}

method3_runs = {
    50:  r"D:\Y4 FYP file\Code Learning\Model_Banana_Method_50&100epoch\Model3_Isolated_CLAHE_50epoch",
    100: r"D:\Y4 FYP file\Code Learning\Model_Banana_Method_50&100epoch\Model3_Isolated_CLAHE_100epoch",
    150: r"D:\Y4 FYP file\Code Learning\Model_Banana_Method_150epoch\Model3_Isolated_CLAHE_150epoch",
    200: r"D:\Y4 FYP file\Code Learning\Model_Banana_Method_200epoch\Model3_Isolated_CLAHE_200epoch",
    250: r"D:\Y4 FYP file\Code Learning\Model_Banana_Method_250epoch\Model3_Isolated_CLAHE_250epoch"
}

# Optional: save graph image
SAVE_FIGURE = True
SAVE_PATH = r"D:\Y4 FYP file\Code Learning\Model_Banana_Method_EpochGraph\epoch_comparison_3methods.png"

# =========================================================
# HELPER FUNCTIONS
# =========================================================


def find_results_csv(run_folder):
    """
    Try to find results.csv inside the given training folder.
    """
    possible_paths = [
        os.path.join(run_folder, "results.csv"),
        os.path.join(run_folder, "weights", "results.csv")
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return path

    for root, dirs, files in os.walk(run_folder):
        if "results.csv" in files:
            return os.path.join(root, "results.csv")

    return None


def find_accuracy_column(df):
    """
    Automatically detect the Top-1 accuracy column from YOLO classification results.csv
    """
    columns = [c.strip() for c in df.columns]

    preferred_names = [
        "metrics/accuracy_top1",
        "metrics/accuracy_top1(B)",
        "accuracy_top1",
        "top1_acc",
        "val/acc",
        "metrics/accuracy"
    ]

    for name in preferred_names:
        if name in columns:
            return name

    for c in columns:
        c_lower = c.lower()
        if "accuracy" in c_lower and "top1" in c_lower:
            return c

    for c in columns:
        c_lower = c.lower()
        if "acc" in c_lower and "top1" in c_lower:
            return c

    for c in columns:
        c_lower = c.lower()
        if "accuracy" in c_lower and "top5" not in c_lower:
            return c

    return None


def extract_final_accuracy(run_folder):
    """
    Read results.csv and return final Top-1 accuracy (%)
    """
    results_csv = find_results_csv(run_folder)

    if results_csv is None:
        print(f"[WARNING] results.csv not found in: {run_folder}")
        return None

    try:
        df = pd.read_csv(results_csv)
    except Exception as e:
        print(f"[ERROR] Failed to read {results_csv}: {e}")
        return None

    acc_col = find_accuracy_column(df)

    if acc_col is None:
        print(f"[WARNING] Accuracy column not found in: {results_csv}")
        print("Columns found:", list(df.columns))
        return None

    final_acc = df[acc_col].iloc[-1]

    if final_acc <= 1.0:
        final_acc *= 100.0

    return float(final_acc)


def collect_method_accuracies(method_runs, method_name="Method"):
    """
    Extract accuracy for all epoch runs of one method
    """
    epochs = []
    accuracies = []

    print(f"\n===== Reading {method_name} =====")

    for epoch in sorted(method_runs.keys()):
        folder = method_runs[epoch]
        acc = extract_final_accuracy(folder)

        if acc is not None:
            epochs.append(epoch)
            accuracies.append(acc)
            print(f"{method_name} | Epoch {epoch}: {acc:.2f}%")
        else:
            print(f"{method_name} | Epoch {epoch}: FAILED")

    return epochs, accuracies


def add_smart_labels(ax, epochs, values, color, series_name):
    """
    Add labels with smarter placement to reduce collision
    """
    for i, (x, y) in enumerate(zip(epochs, values)):
        dx = 0
        dy = 10

        if series_name == "Method 1":
            dx = 0
            dy = 8

        elif series_name == "Method 2":
            dx = 0
            dy = 12

        elif series_name == "Method 3":
            dx = 0
            dy = 8

            # Special fix for the red point at epoch 200
            if x == 200:
                dy = -18   # move label below point
            elif x == 100:
                dy = 6
            elif x == 150:
                dy = 4
            elif x == 250:
                dy = 4

        ax.annotate(
            f"{y:.2f}",
            (x, y),
            textcoords="offset points",
            xytext=(dx, dy),
            ha='center',
            color=color,
            fontsize=9
        )


# =========================================================
# MAIN
# =========================================================

epochs1, acc1 = collect_method_accuracies(method1_runs, "Method 1 (Raw)")
epochs2, acc2 = collect_method_accuracies(
    method2_runs, "Method 2 (Masked + CLAHE)")
epochs3, acc3 = collect_method_accuracies(
    method3_runs, "Method 3 (Isolated + CLAHE)")

# Print summary table
print("\n===== SUMMARY TABLE =====")
all_epochs = sorted(set(list(method1_runs.keys()) +
                    list(method2_runs.keys()) + list(method3_runs.keys())))

summary_rows = []
for ep in all_epochs:
    m1 = acc1[epochs1.index(ep)] if ep in epochs1 else None
    m2 = acc2[epochs2.index(ep)] if ep in epochs2 else None
    m3 = acc3[epochs3.index(ep)] if ep in epochs3 else None
    summary_rows.append([ep, m1, m2, m3])

summary_df = pd.DataFrame(summary_rows, columns=[
    "Epoch",
    "Method 1 Accuracy (%)",
    "Method 2 Accuracy (%)",
    "Method 3 Accuracy (%)"
])

print(summary_df.to_string(index=False))

# =========================================================
# PLOT GRAPH
# =========================================================
fig, ax = plt.subplots(figsize=(13, 6.5))

ax.plot(epochs1, acc1, marker='o', linewidth=2.5,
        color='blue', label='Method 1 (Raw)')
ax.plot(epochs2, acc2, marker='s', linewidth=2.5,
        color='green', label='Method 2 (Masked + CLAHE)')
ax.plot(epochs3, acc3, marker='^', linewidth=2.5,
        color='red', label='Method 3 (Isolated + CLAHE)')

# Smart labels
add_smart_labels(ax, epochs1, acc1, 'blue', "Method 1")
add_smart_labels(ax, epochs2, acc2, 'green', "Method 2")
add_smart_labels(ax, epochs3, acc3, 'red', "Method 3")

ax.set_title("Comparison of Top-1 Accuracy Across Different Epochs (50 to 250)",
             fontsize=18, fontweight='bold', pad=12)
ax.set_xlabel("Training Epoch", fontsize=17)
ax.set_ylabel("Top-1 Accuracy (%)", fontsize=17)
ax.set_xticks(all_epochs)
ax.tick_params(axis='both', labelsize=14)

# Give more margin
y_all = acc1 + acc2 + acc3
ax.set_ylim(min(y_all) - 0.5, max(y_all) + 0.6)

ax.grid(True, linestyle='--', alpha=0.5)

# Legend outside but closer to graph
ax.legend(
    loc='center left',
    bbox_to_anchor=(1.01, 0.5),
    fontsize=11,
    frameon=True,
    borderaxespad=0.8
)

# Manually control subplot position:
# [left, bottom, width, height]
# Increase left a bit and keep right space for legend
fig.subplots_adjust(left=0.10, right=0.72, top=0.88, bottom=0.14)

if SAVE_FIGURE:
    plt.savefig(SAVE_PATH, dpi=300, bbox_inches='tight')
    print(f"\nGraph saved to: {SAVE_PATH}")

plt.show()
