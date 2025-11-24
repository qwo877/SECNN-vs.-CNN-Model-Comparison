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
- `Python: 3.x`
- `PyTorch`
- `torch`
- `torchvision`
- `numpy`
- `matplotlib`
- `scikit-learn`
- `tqdm`

```
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
```


## Training Models
python train.py --model cnn --epochs 20  
python train.py --model se --epochs 20


## Result
in CNN  
<img width="640" height="480" alt="loss_curve cnn" src="https://github.com/user-attachments/assets/27cdfec3-3442-4043-b78a-9764ac539616" />  
<img width="640" height="480" alt="acc_curve cnn" src="https://github.com/user-attachments/assets/aa2cc82d-8548-45a5-b220-7da33c87928c" />  

in SECNN  
<img width="640" height="480" alt="loss_curve" src="https://github.com/user-attachments/assets/cfb5da17-fc33-4488-b51b-8a7cb82ef849" />  
<img width="640" height="480" alt="acc_curve" src="https://github.com/user-attachments/assets/3e3968a8-9578-4b62-a76e-e8c3a8f1ae43" />  

CM:  
<img width="691" height="609" alt="confusion_matrix" src="https://github.com/user-attachments/assets/b3e6f489-97d5-4d11-9e01-02569e97e07c" />


