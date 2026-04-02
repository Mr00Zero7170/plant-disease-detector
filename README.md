# 🌿 Plant Disease Detector

> Deep learning model that detects tomato leaf diseases with ~92% accuracy — built with TensorFlow/Keras and trained on the PlantVillage dataset.

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12+-orange?style=flat-square&logo=tensorflow)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Dataset](https://img.shields.io/badge/Dataset-PlantVillage-brightgreen?style=flat-square)

---

## 🔍 What it does

This project uses a **custom Convolutional Neural Network (CNN)** to classify tomato leaf images into one of three categories:

| Class | Description |
|---|---|
| ✅ `healthy` | No disease detected |
| 🟡 `early_blight` | Fungal infection — early stage (*Alternaria solani*) |
| 🔴 `late_blight` | Fungal infection — advanced stage (*Phytophthora infestans*) |

Early and accurate detection of plant diseases can significantly reduce crop losses — this model brings that capability to a lightweight, trainable pipeline anyone can run locally.

---

## 🏗️ Model Architecture
```
Input (128 × 128 × 3)
  ↓
Conv2D(32) → BatchNorm → Conv2D(32) → MaxPool → Dropout(0.25)
  ↓
Conv2D(64) → BatchNorm → Conv2D(64) → MaxPool → Dropout(0.25)
  ↓
Conv2D(128) → BatchNorm → Conv2D(128) → MaxPool → Dropout(0.25)
  ↓
GlobalAveragePooling
  ↓
Dense(256) → BatchNorm → Dropout(0.5)
  ↓
Dense(3, softmax) → Prediction
```

Key techniques:
- **Batch Normalization** — stabilizes and speeds up training
- **Dropout regularization** — prevents overfitting
- **Data Augmentation** — rotation, zoom, flips, shifts
- **Early Stopping** — stops when validation accuracy plateaus
- **Learning Rate Scheduling** — reduces LR when loss stagnates

---

## 📁 Project Structure
```
plant-disease-detector/
├── prepare_dataset.py     # downloads & splits dataset via kagglehub
├── train.py               # builds and trains the CNN
├── evaluate.py            # confusion matrix + classification report
├── predict.py             # classify a single leaf image
├── requirements.txt       # all dependencies
└── dataset/               # created by prepare_dataset.py
    ├── train/
    │   ├── healthy/
    │   ├── early_blight/
    │   └── late_blight/
    └── test/
        ├── healthy/
        ├── early_blight/
        └── late_blight/
```

---

## 🚀 Quickstart

### 1. Clone the repo
```bash
git clone https://github.com/Mr00Zero7170/plant-disease-detector.git
cd plant-disease-detector
```

### 2. Set up environment (Python 3.11 required)
```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Download & prepare dataset
```bash
python prepare_dataset.py
```
> Automatically downloads PlantVillage from Kaggle via `kagglehub`. Requires a free [Kaggle account](https://www.kaggle.com).

### 4. Train
```bash
python train.py
```

### 5. Evaluate
```bash
python evaluate.py
```

### 6. Predict on a new image
```bash
python predict.py path/to/leaf.jpg
```

**Example output:**
```
──────────────────────────────────────
  Image      : tomato_leaf.jpg
  Prediction : EARLY_BLIGHT
  Confidence : 94.3%
──────────────────────────────────────
  All probabilities:
    early_blight     94.3%  ████████████████████
    healthy           4.1%  ████
    late_blight       1.6%  ███
──────────────────────────────────────
```

---

## 📂 Dataset

[PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease) — 54,000+ images of healthy and diseased plant leaves across 38 classes.

> Hughes, D.P. & Salathé, M. (2015). *An open access repository of images on plant health to enable the development of mobile disease diagnostics.* [arXiv:1511.08060](https://arxiv.org/abs/1511.08060)

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

<p align="center">Built by <a href="https://github.com/Mr00Zero7170">Mr00Zero7170</a></p>
