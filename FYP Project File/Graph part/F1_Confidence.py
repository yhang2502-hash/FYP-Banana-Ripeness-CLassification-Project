import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO

# =========================================================
# USER SETTINGS
# =========================================================

CLASS_NAMES = ["Class A", "Class B", "Class C", "Class D"]
NUM_CLASSES = len(CLASS_NAMES)
IMG_SIZE = 224
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

MODELS = {
    "Method 1 Raw": {
        "model_path": r"D:\Y4 FYP file\Code Learning\Model_Banana_Method_50&100epoch\Model1_Raw_100epoch\weights\best.pt",
        "test_dir": r"D:\Y4 FYP file\Dataset\FYP_Data_Final\Method1_Raw\test",
    },
    "Method 2 Masked + CLAHE": {
        "model_path": r"D:\Y4 FYP file\Code Learning\Model_Banana_Method_50&100epoch\Model2_Masked_CLAHE_Fair_100epoch\weights\best.pt",
        "test_dir": r"D:\Y4 FYP file\Dataset\FYP_Data_Final\Method2_Masked_CLAHE_Fair\test",
    },
    "Method 3 Isolated + CLAHE": {
        "model_path": r"D:\Y4 FYP file\Code Learning\Model_Banana_Method_50&100epoch\Model3_Isolated_CLAHE_New2_100epoch\weights\best.pt",
        "test_dir": r"D:\Y4 FYP file\Dataset\FYP_Data_Final\Method3_Isolated_CLAHE_New2\test",
    }
}

THRESHOLDS = np.linspace(0.0, 1.0, 201)

# =========================================================
# HELPER FUNCTIONS
# =========================================================


def get_image_files(folder_path: Path):
    return sorted([p for p in folder_path.rglob("*") if p.suffix.lower() in IMAGE_EXTS])


def collect_test_samples(test_root: Path, class_names):
    image_paths = []
    y_true = []

    for class_idx, class_name in enumerate(class_names):
        class_folder = test_root / class_name
        if not class_folder.exists():
            print(f"[WARNING] Missing class folder: {class_folder}")
            continue

        for img_path in get_image_files(class_folder):
            image_paths.append(img_path)
            y_true.append(class_idx)

    return image_paths, np.array(y_true)


def predict_probs(model, image_path):
    results = model.predict(source=str(image_path),
                            imgsz=IMG_SIZE, verbose=False)

    if not results or results[0].probs is None:
        return None

    probs = results[0].probs.data.cpu().numpy()
    return probs


def compute_method_f1_confidence_curve(model_path, test_dir):
    model = YOLO(model_path)
    test_root = Path(test_dir)

    image_paths, y_true = collect_test_samples(test_root, CLASS_NAMES)

    all_probs = []
    valid_true = []

    for img_path, true_label in zip(image_paths, y_true):
        probs = predict_probs(model, img_path)
        if probs is None:
            continue
        all_probs.append(probs)
        valid_true.append(true_label)

    all_probs = np.array(all_probs)
    valid_true = np.array(valid_true)

    all_class_f1 = []

    for class_idx in range(NUM_CLASSES):
        class_scores = all_probs[:, class_idx]
        y_true_binary = (valid_true == class_idx).astype(int)

        f1_scores = []

        for thr in THRESHOLDS:
            y_pred_binary = (class_scores >= thr).astype(int)

            tp = np.sum((y_pred_binary == 1) & (y_true_binary == 1))
            fp = np.sum((y_pred_binary == 1) & (y_true_binary == 0))
            fn = np.sum((y_pred_binary == 0) & (y_true_binary == 1))

            if tp + fp == 0:
                precision = 1.0
            else:
                precision = tp / (tp + fp)

            if tp + fn == 0:
                recall = 1.0
            else:
                recall = tp / (tp + fn)

            if precision + recall == 0:
                f1 = 0.0
            else:
                f1 = 2 * precision * recall / (precision + recall)

            f1_scores.append(f1)

        all_class_f1.append(f1_scores)

    all_class_f1 = np.array(all_class_f1)
    mean_f1 = all_class_f1.mean(axis=0)

    return mean_f1

# =========================================================
# MAIN PLOT
# =========================================================


plt.figure(figsize=(12, 8))

for method_name, info in MODELS.items():
    mean_f1 = compute_method_f1_confidence_curve(
        info["model_path"],
        info["test_dir"]
    )

    best_idx = np.argmax(mean_f1)
    best_thr = THRESHOLDS[best_idx]
    best_f1 = mean_f1[best_idx]

    plt.plot(
        THRESHOLDS,
        mean_f1,
        linewidth=3,
        label=f"{method_name} ({best_f1:.2f} at {best_thr:.3f})"
    )

plt.title("F1-Confidence Curve", fontsize=18)
plt.xlabel("Confidence", fontsize=14)
plt.ylabel("F1-Score", fontsize=14)
plt.xlim(0, 1)
plt.ylim(0, 1.01)
plt.grid(True, linestyle="--", alpha=0.7)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig("f1_confidence_curve_3methods_only.png", dpi=300)
plt.show()
