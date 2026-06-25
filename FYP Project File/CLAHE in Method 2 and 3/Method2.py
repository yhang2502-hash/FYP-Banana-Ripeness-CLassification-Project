import cv2 as cv
import numpy as np
import os


def preprocess_method2_offline(img_path):
    # 1. Load image
    img = cv.imread(img_path)
    if img is None:
        return None

    # 2. Convert to HSV
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    # 3. Same general banana color ranges as real-time Method 2
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([90, 255, 255])

    lower_yellow = np.array([15, 30, 30])
    upper_yellow = np.array([45, 255, 255])

    lower_brown_dark = np.array([0, 10, 0])
    upper_brown_dark = np.array([30, 255, 200])

    # 4. Create masks
    mask_green = cv.inRange(hsv, lower_green, upper_green)
    mask_yellow = cv.inRange(hsv, lower_yellow, upper_yellow)
    mask_brown_dark = cv.inRange(hsv, lower_brown_dark, upper_brown_dark)

    # 5. Combine masks
    mask = cv.bitwise_or(mask_green, mask_yellow)
    mask = cv.bitwise_or(mask, mask_brown_dark)

    # 6. Morphological cleaning
    kernel = np.ones((5, 5), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

    # 7. Apply mask
    masked_img = cv.bitwise_and(img, img, mask=mask)

    # 8. Apply CLAHE on masked image
    lab = cv.cvtColor(masked_img, cv.COLOR_BGR2LAB)
    l, a, b = cv.split(lab)

    clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)

    enhanced_lab = cv.merge((l_clahe, a, b))
    enhanced_img = cv.cvtColor(enhanced_lab, cv.COLOR_LAB2BGR)

    # 9. Put enhanced result only on masked banana region
    result = img.copy()
    result[mask > 0] = enhanced_img[mask > 0]

    return result


# --- PATH CONFIGURATION ---
input_folder = r"D:\Y4 FYP file\Dataset\BananaRipeness-main\Banana Images\Real Dataset"
output_folder = r"D:\Y4 FYP file\Dataset\Enhanced_Dataset_Method2_Fair"


if not os.path.exists(input_folder):
    print(f"❌ ERROR: The path does not exist: {input_folder}")
    print("👉 Check your folder path again.")
else:
    print("✅ Input folder found. Start processing...")

    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(".png"):
                img_path = os.path.join(root, file)

                rel_path = os.path.relpath(root, input_folder)
                save_dir = os.path.join(output_folder, rel_path)
                save_path = os.path.join(save_dir, file)

                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                processed = preprocess_method2_offline(img_path)

                if processed is not None:
                    cv.imwrite(save_path, processed)
                    print(f"✅ Success: {file}")

print("🎉 DONE! Fair Method 2 dataset is ready.")
