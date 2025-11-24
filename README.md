# SECNN vs. CNN Model Comparison

## Overview

This project implements and compares **SECNN (Self-Expressive CNN)** with a **traditional CNN model**.  
By training and testing both models under the same dataset and conditions, the project aims to observe key performance differences between the two architectures.

---

## Purpose

- Compare and analyze **functional differences between SECNN and CNN**, including:
  - Accuracy and loss convergence speed
  - Robustness to noise or imbalanced data
  - Model complexity and inference time
- Provide a reproducible experiment framework for adding new models or datasets in the future.

---

## Environment

- OS: Windows / Linux / macOS
- Python: `3.x`
- PyTorch
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `scikit-learn`
  - `tqdm`
  - `opencv-python`



SECNN-vs.-CNN-Model-Comparison/
│
├─ model/
│   ├─ SE.py                     # SE model
│   ├─ SE_CNN.py                 # SECNN model definition
│   ├─ CNN.py                    # Baseline CNN model definition
│   └─ __init__.py               # model module init
│
├─ train.py                      # Training script
├─ test.py                       # Evaluation script
├─ utils.py                      # functions
└─ README.md                     



## Training Models
python train.py --model cnn --epochs 20
python train.py --model se --epochs 20
