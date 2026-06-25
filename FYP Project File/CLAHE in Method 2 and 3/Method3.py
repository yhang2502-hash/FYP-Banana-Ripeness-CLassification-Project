from rembg import remove, new_session
import cv2 as cv
import os
import numpy as np

# =========================================================
# METHOD 3 FINAL (OFFLINE DATASET GENERATION)
# =========================================================

session = new_session("silueta")

# ---------- SETTINGS ----------
CLAHE_CLIP = 1.5
CLAHE_GRID = (8, 8)
KERNEL_SIZE = 5
PADDING = 10   # optional extra padding for crop if you want later
SAVE_WITH_BLACK_BACKGROUND = True   # keep full image size with black background


def preprocess_method3_final(img_path):
    img = cv.imread(img_path)
    if img is None:
        return None

    # STEP 1: Gentle CLAHE on L channel
    lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    l, a, b = cv.split(lab)

    clahe = cv.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)
    l_clahe = clahe.apply(l)

    enhanced = cv.cvtColor(cv.merge((l_clahe, a, b)), cv.COLOR_LAB2BGR)

    # STEP 2: AI segmentation
    output = remove(enhanced, session=session)

    if len(output.shape) == 3 and output.shape[2] == 4:
        rgba_planes = cv.split(output)
        mask_ai = rgba_planes[3]
    else:
        gray = cv.cvtColor(output, cv.COLOR_BGR2GRAY)
        _, mask_ai = cv.threshold(gray, 1, 255, cv.THRESH_BINARY)

    # STEP 3: Morphological cleanup
    kernel = np.ones((KERNEL_SIZE, KERNEL_SIZE), np.uint8)
    mask_clean = cv.morphologyEx(mask_ai, cv.MORPH_OPEN, kernel)
    mask_clean = cv.morphologyEx(mask_clean, cv.MORPH_CLOSE, kernel)

    # STEP 4: Keep largest contour only
    contours, _ = cv.findContours(
        mask_clean, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    final_mask = np.zeros_like(mask_clean)

    if contours:
        largest_contour = max(contours, key=cv.contourArea)
        cv.drawContours(
            final_mask, [largest_contour], -1, 255, thickness=cv.FILLED)
    else:
        final_mask = mask_clean

    # STEP 5: Apply final mask
    final_img = cv.bitwise_and(enhanced, enhanced, mask=final_mask)

    return final_img


# =========================================================
# PATH CONFIGURATION
# =========================================================

input_folder = r"D:\Y4 FYP file\Dataset\BananaRipeness-main\Banana Images\Real Dataset"
output_folder = r"D:\Y4 FYP file\Dataset\Enhanced_Dataset_Method3_Final"

if not os.path.exists(input_folder):
    print("ERROR: Input path not found!")
else:
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
                img_path = os.path.join(root, file)
                rel_path = os.path.relpath(root, input_folder)
                save_dir = os.path.join(output_folder, rel_path)
                save_path = os.path.join(save_dir, file)

                processed = preprocess_method3_final(img_path)

                if processed is not None:
                    os.makedirs(save_dir, exist_ok=True)
                    cv.imwrite(save_path, processed)
                    print(f"Saved: {save_path}")

print("DONE! Offline Method 3 dataset generation finished.")
