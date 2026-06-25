import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# =========================================================
# USER SETTINGS
# =========================================================

# Class names in the correct order
CLASS_NAMES = ["Class A", "Class B", "Class C", "Class D"]

# Image size used during training
IMG_SIZE = 224

# Supported image file extensions
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ---------- MODEL PATHS ----------
MODELS = {
    "Method 1 Raw": r"D:\Y4 FYP file\Code Learning\Model_Banana_Method_50&100epoch\Model1_Raw_100epoch\weights\best.pt",
    "Method 2 Masked + CLAHE": r"D:\Y4 FYP file\Code Learning\Model_Banana_Method_50&100epoch\Model2_Masked_CLAHE_Fair_100epoch\weights\best.pt",
    "Method 3 Isolated + CLAHE": r"D:\Y4 FYP file\Code Learning\Model_Banana_Method_50&100epoch\Model3_Isolated_CLAHE_New2_100epoch\weights\best.pt",
}

# ---------- TEST DATASET PATHS ----------
TEST_DIRS = {
    "Method 1 Raw": r"D:\Y4 FYP file\Dataset\FYP_Data_Final\Method1_Raw\test",
    "Method 2 Masked + CLAHE": r"D:\Y4 FYP file\Dataset\FYP_Data_Final\Method2_Masked_CLAHE_Fair\test",
    "Method 3 Isolated + CLAHE": r"D:\Y4 FYP file\Dataset\FYP_Data_Final\Method3_Isolated_CLAHE_New2\test",
}

# Output folder for reports
OUTPUT_DIR = Path("evaluation_reports")
OUTPUT_DIR.mkdir(exist_ok=True)

# =========================================================
# HELPER FUNCTIONS
# =========================================================


def get_image_files(folder_path: Path):
    return sorted(
        [p for p in folder_path.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
    )


def collect_test_samples(test_root: Path, class_names: list[str]):
    """
    Collect all test image paths and corresponding true labels.
    Assumes folder structure:
    test_root/class_name/image.jpg
    """
    image_paths = []
    y_true = []

    for class_idx, class_name in enumerate(class_names):
        class_folder = test_root / class_name
        if not class_folder.exists():
            print(f"[WARNING] Class folder not found: {class_folder}")
            continue

        files = get_image_files(class_folder)
        for file_path in files:
            image_paths.append(file_path)
            y_true.append(class_idx)

    return image_paths, y_true


def predict_one_image(model: YOLO, image_path: Path, img_size: int):
    """
    Predict one image and return predicted class index.
    """
    results = model.predict(
        source=str(image_path),
        imgsz=img_size,
        verbose=False
    )

    if not results or results[0].probs is None:
        return None

    pred_idx = int(results[0].probs.top1)
    return pred_idx


def evaluate_model(model_name: str, model_path: str, test_dir: str, class_names: list[str]):
    """
    Evaluate one trained YOLO classification model on a folder-based test set.
    Returns summary metrics, classification report dataframe, confusion matrix dataframe,
    and per-image predictions dataframe.
    """
    print(f"\n{'='*70}")
    print(f"Evaluating: {model_name}")
    print(f"Model path: {model_path}")
    print(f"Test dir  : {test_dir}")
    print(f"{'='*70}")

    model = YOLO(model_path)
    test_root = Path(test_dir)

    image_paths, y_true = collect_test_samples(test_root, class_names)

    if len(image_paths) == 0:
        raise ValueError(f"No test images found in: {test_root}")

    y_pred = []
    records = []

    for img_path, true_idx in zip(image_paths, y_true):
        pred_idx = predict_one_image(model, img_path, IMG_SIZE)

        if pred_idx is None:
            # Mark failed prediction as -1
            pred_idx = -1
            pred_name = "unknown"
        else:
            pred_name = class_names[pred_idx] if pred_idx < len(
                class_names) else str(pred_idx)

        y_pred.append(pred_idx)

        records.append({
            "image_path": str(img_path),
            "true_label_index": true_idx,
            "true_label_name": class_names[true_idx],
            "pred_label_index": pred_idx,
            "pred_label_name": pred_name,
            "correct": int(pred_idx == true_idx),
        })

    # Filter out failed predictions if any
    valid_mask = [pred != -1 for pred in y_pred]
    y_true_valid = [t for t, keep in zip(y_true, valid_mask) if keep]
    y_pred_valid = [p for p, keep in zip(y_pred, valid_mask) if keep]

    if len(y_true_valid) == 0:
        raise ValueError("All predictions failed. Cannot compute metrics.")

    # Overall metrics
    accuracy = accuracy_score(y_true_valid, y_pred_valid)
    precision_macro = precision_score(
        y_true_valid, y_pred_valid, average="macro", zero_division=0)
    recall_macro = recall_score(
        y_true_valid, y_pred_valid, average="macro", zero_division=0)
    f1_macro = f1_score(y_true_valid, y_pred_valid,
                        average="macro", zero_division=0)

    precision_weighted = precision_score(
        y_true_valid, y_pred_valid, average="weighted", zero_division=0)
    recall_weighted = recall_score(
        y_true_valid, y_pred_valid, average="weighted", zero_division=0)
    f1_weighted = f1_score(y_true_valid, y_pred_valid,
                           average="weighted", zero_division=0)

    # Detailed classification report
    report_dict = classification_report(
        y_true_valid,
        y_pred_valid,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    report_df = pd.DataFrame(report_dict).transpose()

    # Confusion matrix
    cm = confusion_matrix(y_true_valid, y_pred_valid,
                          labels=list(range(len(class_names))))
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

    # Per-image predictions
    pred_df = pd.DataFrame(records)

    # Summary row
    summary = {
        "Method": model_name,
        "Accuracy": accuracy,
        "Precision_macro": precision_macro,
        "Recall_macro": recall_macro,
        "F1_macro": f1_macro,
        "Precision_weighted": precision_weighted,
        "Recall_weighted": recall_weighted,
        "F1_weighted": f1_weighted,
        "Num_test_images": len(y_true_valid),
    }

    # Print summary
    print("\nOverall Metrics")
    print(f"Accuracy           : {accuracy:.4f}")
    print(f"Precision (macro)  : {precision_macro:.4f}")
    print(f"Recall (macro)     : {recall_macro:.4f}")
    print(f"F1-score (macro)   : {f1_macro:.4f}")
    print(f"Precision (weighted): {precision_weighted:.4f}")
    print(f"Recall (weighted)   : {recall_weighted:.4f}")
    print(f"F1-score (weighted) : {f1_weighted:.4f}")

    print("\nClassification Report")
    print(report_df)

    print("\nConfusion Matrix")
    print(cm_df)

    return summary, report_df, cm_df, pred_df


def plot_confusion_matrix(cm_df, model_name, safe_name):
    """
    Plot and save confusion matrix heatmap using matplotlib.
    """
    plt.figure(figsize=(10, 8))
    im = plt.imshow(cm_df, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix: {model_name}')
    plt.colorbar(im)

    tick_marks = np.arange(len(cm_df.columns))
    plt.xticks(tick_marks, cm_df.columns, rotation=45)
    plt.yticks(tick_marks, cm_df.index)

    # Add numeric annotations
    thresh = cm_df.values.max() / 2.
    for i in range(cm_df.shape[0]):
        for j in range(cm_df.shape[1]):
            plt.text(j, i, format(cm_df.values[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm_df.values[i, j] > thresh else "black")

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{safe_name}_confusion_matrix.png")
    plt.show()
    plt.close()


def plot_method_metrics(summary, safe_name):
    """
    Plot and save bar chart for a single method's metrics.
    """
    metrics = {
        'Accuracy': summary['Accuracy'],
        'Precision (Macro)': summary['Precision_macro'],
        'Recall (Macro)': summary['Recall_macro'],
        'F1 (Macro)': summary['F1_macro']
    }

    names = list(metrics.keys())
    values = list(metrics.values())

    plt.figure(figsize=(8, 6))
    bars = plt.bar(names, values, color=[
                   '#4285F4', '#EA4335', '#FBBC05', '#34A853'])
    plt.ylim(0, 1.05)
    plt.title(f"Evaluation Metrics: {summary['Method']}")
    plt.ylabel('Score')

    # Add labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01,
                 f'{yval:.4f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{safe_name}_metrics.png")
    plt.show()
    plt.close()


def plot_comparison(summary_df):
    """
    Plot comparison of all methods using pure matplotlib.
    """
    metrics = ['Accuracy', 'Precision_macro', 'Recall_macro', 'F1_macro']
    methods = summary_df['Method'].tolist()

    n_metrics = len(metrics)
    n_methods = len(methods)

    fig, ax = plt.subplots(figsize=(12, 7))

    # Setting the width and positions of bars
    width = 0.8 / n_methods
    x = np.arange(n_metrics)

    for i, method in enumerate(methods):
        scores = [summary_df.loc[summary_df['Method']
                                 == method, m].values[0] for m in metrics]
        ax.bar(x + i * width - (n_methods-1) *
               width/2, scores, width, label=method)

    ax.set_ylabel('Score', fontsize=14)
    ax.set_title('Comparison of Training Methods',
                 fontsize=18, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_macro', '') for m in metrics], fontsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_ylim(0, 1.1)
    ax.legend(loc='upper left')
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "methods_comparison.png")
    plt.show()
    plt.close()

# =========================================================
# MAIN EVALUATION
# =========================================================


def main():
    all_summaries = []

    for method_name in MODELS.keys():
        model_path = MODELS[method_name]
        test_dir = TEST_DIRS[method_name]

        summary, report_df, cm_df, pred_df = evaluate_model(
            method_name,
            model_path,
            test_dir,
            CLASS_NAMES
        )

        all_summaries.append(summary)

        safe_name = method_name.replace(" ", "_").replace(
            ":", "").replace("+", "plus")

        # Save detailed outputs
        report_df.to_csv(OUTPUT_DIR / f"{safe_name}_classification_report.csv")
        cm_df.to_csv(OUTPUT_DIR / f"{safe_name}_confusion_matrix.csv")
        pred_df.to_csv(
            OUTPUT_DIR / f"{safe_name}_predictions.csv", index=False)

        # Generate graphs for each method
        print(f"Generating graphs for {method_name}...")
        plot_confusion_matrix(cm_df, method_name, safe_name)
        plot_method_metrics(summary, safe_name)

    # Save comparison summary
    summary_df = pd.DataFrame(all_summaries)
    summary_df.to_csv(OUTPUT_DIR / "model_comparison_summary.csv", index=False)

    # Generate comparison graph
    print("Generating comparison graph...")
    plot_comparison(summary_df)

    print("\n" + "="*70)
    print("FINAL COMPARISON SUMMARY")
    print("="*70)
    print(summary_df)

    # Optional: prettier percentage view for report
    summary_percent = summary_df.copy()
    for col in [
        "Accuracy",
        "Precision_macro",
        "Recall_macro",
        "F1_macro",
        "Precision_weighted",
        "Recall_weighted",
        "F1_weighted",
    ]:
        summary_percent[col] = summary_percent[col] * 100

    print("\nSummary in Percentage (%)")
    print(summary_percent)


if __name__ == "__main__":
    main()
