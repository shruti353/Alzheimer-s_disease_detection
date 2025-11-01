# Alzheimer’s Disease Classification using MRI Scans

**Overview**
This project focuses on automated Alzheimer’s disease classification using MRI brain scans. The model leverages transfer learning with EfficientNetB0 and a custom CNN head to classify MRI images into four cognitive states:


1. No Impairment
2. Very Mild Impairment
3. Mild Impairment
4. Moderate Impairment


By utilizing deep learning and fine-tuning techniques, this system achieves high accuracy (>95%), enabling faster and more reliable clinical insights for early diagnosis.

# Objective
To develop a robust and explainable CNN-based model capable of detecting and categorizing different stages of Alzheimer’s Disease from MRI images.

# Dataset


**Source:** Kaggle Alzheimer’s MRI Dataset

**Structure:**
├── train/
│   ├── Mild Impairment/
│   ├── Moderate Impairment/
│   ├── No Impairment/
│   └── Very Mild Impairment/
├── test/
│   ├── Mild Impairment/
│   ├── Moderate Impairment/
│   ├── No Impairment/
│   └── Very Mild Impairment/

Image size: 224 × 224 pixels
Total samples: ~19,000 MRI images



# Methodology

# 1. Data Preprocessing
Loaded using image_dataset_from_directory()
Applied augmentation: random flips, rotations, and zooms
Normalized pixel values to [0, 1]
Used prefetching for efficient GPU pipeline

# 2. Model Architecture

**Base Model:** EfficientNetB0 (pretrained on ImageNet)
Frozen for Phase 1 training
Fine-tuned top 100 layers during Phase 2

**Custom Head**
GlobalAveragePooling2D()
Dropout(0.4)
Dense(num_classes, activation='softmax')

# 3. Training Strategy


**Phase 1: Train top layers (frozen base)
**

**Phase 2: Unfreeze top 100 layers for fine-tuning**


**Callbacks:**
EarlyStopping
ReduceLROnPlateau
ModelCheckpoint

# Results
MetricScore Accuracy:95.9%, Validation Loss: 0.12 ,Recall / Precision: 0.94 , across all classes Model Size 45.6 MB, ArchitectureEfficientNetB0 + Dense Head
Confusion Matrix
(Add confusion matrix image here — e.g. confusion_matrix.png)
Classification Report
(Add classification_report image or table here)

# Technologies Used
TensorFlow / Keras
Python
NumPy, Pandas, Matplotlib, Seaborn
Scikit-learn
EfficientNetB0 (Transfer Learning)
ONNX for model export

# Model Export

Saved in multiple formats for deployment:
.keras
.h5
.onnx (for interoperability with PyTorch / OpenVINO / ONNX Runtime)





# Future Work

Integrate with a web-based MRI analysis dashboard
Deploy on cloud (Streamlit / Flask + TensorFlow Serving)
Extend to 3D MRI volumes using 3D CNNs
Add Grad-CAM visualization for model explainability



# Author
Ms. Shruti Thakkar
MSC Data Science
Alliance University, Bangalore 
