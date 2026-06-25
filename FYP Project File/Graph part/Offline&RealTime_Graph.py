import matplotlib.pyplot as plt
import numpy as np

# =========================
# Data
# =========================
methods = [
    "Method 1\nRaw",
    "Method 2\nMasked + CLAHE",
    "Method 3\nIsolated + CLAHE"
]

offline_accuracy = [91.416309, 91.988555, 90.844063]
realtime_accuracy = [77.99, 39.61, 94.62]

x = np.arange(len(methods))
width = 0.28   # smaller bar width

# =========================
# Plot
# =========================
plt.figure(figsize=(13, 7))

bars1 = plt.bar(
    x - width/2,
    offline_accuracy,
    width,
    label="Offline Accuracy"
)

bars2 = plt.bar(
    x + width/2,
    realtime_accuracy,
    width,
    label="Real-Time Accuracy"
)

# =========================
# Value labels
# =========================
for bar in bars1:
    h = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        h + 0.8,
        f"{h:.2f}%",
        ha="center",
        va="bottom",
        fontsize=15
    )

for bar in bars2:
    h = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        h + 0.8,
        f"{h:.2f}%",
        ha="center",
        va="bottom",
        fontsize=15
    )

# =========================
# Axis and title formatting
# =========================
plt.xticks(x, methods, fontsize=15)
plt.ylabel("Accuracy (%)", fontsize=20)
plt.xlabel("Method", fontsize=16, labelpad=20)

plt.title(
    "Comparison of Offline and Real-Time Accuracy for Three Methods",
    fontsize=20,
    fontweight="bold"
)

plt.ylim(0, 115)
plt.yticks(fontsize=14)

plt.grid(axis="y", linestyle="--", alpha=0.4)

# Legend outside the graph
plt.legend(
    fontsize=15,
    loc="upper left"
)

plt.tight_layout()
plt.show()
