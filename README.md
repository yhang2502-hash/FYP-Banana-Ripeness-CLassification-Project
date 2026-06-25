# Vision-Based Banana Ripeness Classification Using YOLOv8

This repository contains my Final Year Project (FYP), **Vision-Based Banana Ripeness Classification Using YOLOv8**. The project focuses on classifying banana ripeness levels using computer vision, image preprocessing, deep learning, and real-time camera testing.

The system classifies banana images into four ripeness classes:

* **Class A:** Green
* **Class B:** Partially Ripe
* **Class C:** Ripe
* **Class D:** Overripe

---

## Project Demo Video

[Watch FYP Project Demo Video at page 12 & 13](https://canva.link/sjgpwx7bi7v2vue)

The demo video shows the real-time camera setup, GUI interface, and banana ripeness prediction results.

## Project Overview

Banana ripeness classification is important in agriculture, food quality control, and smart farming applications. Manual ripeness checking can be subjective because it depends on human observation, lighting conditions, and experience.

This project uses **YOLOv8 image classification** together with image preprocessing techniques to classify banana ripeness more accurately. The project also includes a real-time camera testing system with a graphical user interface (GUI).

---

## Objectives

The objectives of this project are:

1. To classify banana ripeness levels using YOLOv8 classification.
2. To compare different image preprocessing methods.
3. To evaluate model performance using offline testing and real-time camera testing.
4. To develop a real-time GUI system for banana ripeness prediction.
5. To identify the most suitable preprocessing method for practical real-time use.

---

## Technologies Used

* Python
* YOLOv8
* OpenCV
* Tkinter
* NumPy
* Pandas
* Matplotlib
* PIL / Pillow
* rembg
* Webcam
* Ring light

---

## Ripeness Classes

| Class   | Ripeness Level |
| ------- | -------------- |
| Class A | Green          |
| Class B | Partially Ripe |
| Class C | Ripe           |
| Class D | Overripe       |

---

## Methodology

Three different methods were tested and compared in this project.

### Method 1: Raw Image

The original banana images were used directly without additional preprocessing.

### Method 2: Masked Image + CLAHE

A masking technique was applied to extract the banana region, followed by **CLAHE** enhancement to improve contrast and image quality.

CLAHE stands for **Contrast Limited Adaptive Histogram Equalization**. It helps improve image contrast, especially when lighting conditions are uneven.

### Method 3: Isolated Banana Image + CLAHE

The banana object was isolated from the background, followed by CLAHE enhancement. This method helps reduce background interference and improves real-time camera testing performance.

---

## Model Training

The YOLOv8 classification model was trained using banana ripeness images divided into training, validation, and testing datasets.

The dataset was organized into four classes:

```text
Class A - Green
Class B - Partially Ripe
Class C - Ripe
Class D - Overripe
```

The model was trained and evaluated using accuracy, prediction results, and comparison graphs.

---

## Real-Time Camera Testing

A real-time testing system was developed using Python and OpenCV. The system uses a webcam to capture banana images and display prediction results through a GUI.

The GUI shows:

* Actual class
* Predicted class
* Accuracy / confidence
* Majority prediction
* Stability result
* Visual highlight when prediction is correct

The real-time setup included:

* Webcam
* Ring light
* Laptop
* Banana sample
* Python GUI system

---

## Results Summary

The project compared offline testing results and real-time camera testing results.

### Offline Testing

Method 2 achieved the highest offline accuracy because the masked image with CLAHE helped enhance image features.

### Real-Time Testing

Method 3 performed better during real-time camera testing because isolating the banana from the background made the system more robust under different lighting and background conditions.

### Overall Conclusion

* **Best offline method:** Method 2 - Masked Image + CLAHE
* **Best real-time method:** Method 3 - Isolated Banana Image + CLAHE

Method 3 is recommended for practical real-time use because it is more stable when tested with a webcam.

---

## Repository Structure

```text
FYP-Banana-Ripeness-Classification-Project/
│
├── CLAHE in Method 2 and 3/
│   ├── Method2.py
│   └── Method3.py
│
├── Dataset/
│   ├── Method1_Raw/
│   ├── Method2_Masked_CLAHE/
│   └── Method3_Isolated_CLAHE/
│
├── Graph part/
│   └── Result graphs and comparison outputs
│
├── Model_Banana_epoch_train/
│   ├── Model_Train_50epoch/
│   └── Model_Train_100epoch/
│
├── Real TIme Camera Set Up Code/
│   └── Real-time camera GUI code
│
├── YOLOv8 dataset_training/
│   └── YOLOv8 training-related files
│
├── curve_reports/
│   └── Training curve reports
│
├── evaluation_reports/
│   └── Model evaluation reports
│
├── FYP Presentation Slide.pdf
│
└── README.md
```

---

## Dataset Information

The banana ripeness dataset used in this project is based on publicly available banana ripeness image datasets.

Original dataset reference:

* GitHub: https://github.com/luischuquim/BananaRipeness

The dataset was used for academic and Final Year Project purposes.

Only relevant project files and sample/reference images are included in this repository. The dataset source should be credited properly if reused.

---

## My Contribution

My main contributions in this project include:

* Implementing YOLOv8-based banana ripeness classification.
* Applying image preprocessing techniques for comparison.
* Developing Method 2 using Masked Image + CLAHE.
* Developing Method 3 using Isolated Banana Image + CLAHE.
* Building a real-time camera testing system.
* Creating a GUI to display prediction results.
* Comparing offline testing and real-time testing performance.
* Analyzing the strengths and weaknesses of each method.

---

## How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/yhang2502-hash/FYP-Banana-Ripeness-Classification-Project.git
```

### 2. Install required libraries

```bash
pip install -r requirements.txt
```

### 3. Run the training code

Open the YOLOv8 training code and run it using Python.

```bash
python train_yolov8.py
```

### 4. Run the real-time camera GUI

```bash
python real_time_camera_gui.py
```

Note: File names may need to be adjusted based on the actual file names in this repository.

---

## Requirements

The main Python libraries used in this project include:

```text
ultralytics
opencv-python
numpy
matplotlib
pandas
Pillow
rembg
```

---

## Limitations

Some limitations of this project include:

* The model performance may be affected by lighting conditions.
* Banana position and camera angle can influence prediction accuracy.
* Dataset size and image variety may affect model generalization.
* Real-time testing results may differ from offline testing results.
* More testing is needed for different banana types, backgrounds, and lighting conditions.

---

## Future Improvements

Future improvements may include:

* Increasing the dataset size with more banana images.
* Testing under more real-world lighting conditions.
* Improving background removal and segmentation.
* Deploying the model into an embedded or edge AI device.
* Adding object detection before classification.
* Developing a more user-friendly interface.
* Testing the system in a real agriculture or food sorting environment.

---

## Author

**Yu Hang Chok**

Final Year Project
Mechatronics Engineering
Tunku Abdul Rahman University of Management and Technology

---

## Disclaimer

This project was developed for academic and learning purposes. The dataset source is credited, and the preprocessing methods, model training, real-time GUI, and analysis were implemented as part of the author's Final Year Project.
