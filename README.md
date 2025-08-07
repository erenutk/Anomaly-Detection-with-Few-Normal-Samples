# 🍃 Anomaly Detection with Few Normal Samples

This project focuses on detecting anomalies in plant leaf images using very limited normal data (only 100 samples). It combines classical ML models (OCSVM, Isolation Forest with HOG features) and deep learning (Convolutional Autoencoder). A user-friendly Gradio GUI is also included for real-time testing.

---

## 📁 Project Structure

```
anomaly-detection-few-normal-samples/
├── data/
│   ├── raw/               # Original images (train/test)
│   └── processed/         # Numpy arrays (.npy) used for training
├── models/                # Trained models (.pkl, .keras)
├── scripts/
│   ├── data_preprocessing.py  # Image loading, augmentation, and .npy creation
│   ├── model_training.py      # Model training (AE, OCSVM, IF)
│   └── gui.py                 # Gradio GUI for anomaly testing
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Preprocess and augment training data

```bash
python scripts/data_preprocessing.py
```

### 3. Train models (CAE, OCSVM, Isolation Forest)

```bash
python scripts/model_training.py
```

### 4. Run the GUI for testing

```bash
python scripts/gui.py
```

---

## 🤖 Models Used

- ✅ **Convolutional Autoencoder (CAE)**: Learns to reconstruct normal samples. Anomaly = High reconstruction error.
- ✅ **One-Class SVM (OCSVM)** + **HOG** features
- ✅ **Isolation Forest (IF)** + **HOG** features

Thresholds and preprocessing steps are handled within the GUI automatically.

---

## 🧪 Dataset

- 100 grayscale plant leaf images, size 64x64.
- Stored under `data/raw/`.
- Augmentation includes flipping, rotation, brightness, etc.
- Final training/testing data saved as `.npy` under `data/processed/`.

---

## 🎛️ Gradio Interface

The GUI allows you to:

- Upload any leaf image
- Choose model: CAE, OCSVM+HOG, or IF+HOG
- Get anomaly prediction in real time
- For CAE, visualize:
  - Reconstructed image
  - Error heatmap

No need for technical knowledge to use the GUI.

---





