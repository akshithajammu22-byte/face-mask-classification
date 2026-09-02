# 😷 Face Mask Detection & Image Classification

[![PyTorch](https://img.shields.io/badge/PyTorch-2.14-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org/)
[![MobileNetV2](https://img.shields.io/badge/Model-MobileNetV2-brightgreen.svg)](https://pytorch.org/vision/stable/models/mobilenetv2.html)
[![Accuracy](https://img.shields.io/badge/Test%20Accuracy-99.09%25-blue.svg)](#-model-performance)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A high-performance deep learning image classification system built with **PyTorch** and **MobileNetV2 Transfer Learning** to detect whether a person is wearing a face mask (**WithMask**) or not (**WithoutMask**).

---

## 📌 Project Overview

- **Dataset**: Face Mask Dataset with 11,792 images (10,000 Train, 800 Validation, 992 Test)
- **Architecture**: Pre-trained MobileNetV2 feature extractor with custom classification head
- **Training**: Ultra-fast convergence with cross-entropy loss and Adam optimizer
- **Inference**: Lightning-fast (<15ms per image on CPU)
- **Interactive UI**: Embedded Jupyter Notebook File Upload Widget for real-time inference on custom external images

---

## 📊 Model Performance

Evaluated on the unseen **992 test images**:

| Metric | Score |
| :--- | :---: |
| **Test Accuracy** | **99.09%** (983 / 992) |
| **Validation Accuracy** | **99.25%** |
| **Precision (WithMask)** | **98.97%** |
| **Precision (WithoutMask)** | **99.21%** |
| **Recall (WithMask)** | **99.17%** |
| **Recall (WithoutMask)** | **99.02%** |
| **Macro F1-Score** | **99.09%** |

### Confusion Matrix

```
                      Predicted: WithMask    Predicted: WithoutMask
Actual: WithMask              479                      4
Actual: WithoutMask             5                    504
```

---

## 🚀 Quick Start

### 1. Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/akshithajammu22-byte/face-mask-classification.git
cd face-mask-classification
pip install torch torchvision matplotlib scikit-learn pillow ipywidgets jupyter
```

### 2. Interactive Jupyter Notebook

Launch the notebook to inspect training curves, confusion matrix, and upload custom images:

```bash
jupyter notebook face_mask_classification.ipynb
```

Scroll to **Section 8** in the notebook to use the **📁 Upload Image** widget and classify any photo from your computer in real-time.

### 3. Command-Line Inference

Classify any individual image file using `predict.py`:

```bash
python predict.py --image "path/to/your_photo.jpg"
```

**Example Output:**
```
=============================================
        FACE MASK CLASSIFICATION RESULT      
=============================================
 Image File   : sample.jpg
 Prediction   : [MASK DETECTED] -> WithMask
 Confidence   : 98.99%
---------------------------------------------
 Class Probabilities:
  - WithMask    :  98.99% |########################
  - WithoutMask :   1.01% |
=============================================
```

### 4. Train the Model

To train the model from scratch:

```bash
python train.py
```

---

## 📁 Repository Structure

```
├── face_mask_classification.ipynb   # Complete, executed Jupyter Notebook with interactive widget
├── train.py                         # Model training and test evaluation script
├── predict.py                       # CLI inference script for external images
├── generate_notebook.py             # Automated notebook generator
├── face_mask_mobilenetv2.pth        # Saved model weights checkpoint
└── README.md                        # Documentation & results
```

---

## 👤 Author

**Akshitha**
- GitHub: [@akshithajammu22-byte](https://github.com/akshithajammu22-byte)
- Email: akshithajammu22@gmail.com
