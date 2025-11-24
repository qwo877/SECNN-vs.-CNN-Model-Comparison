import torch
import matplotlib.pyplot as plt
from typing import List, Dict
import numpy as np
from sklearn.metrics import confusion_matrix
import itertools
import os

def accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    #計算 batch 的 accuracy
    preds = outputs.argmax(dim=1)
    correct = (preds == labels).sum().item()
    return correct / labels.size(0)

def curves(history: Dict[str, List[float]], save_dir: str):
    #畫出 loss 與 accuracy 曲線
    os.makedirs(save_dir, exist_ok=True)

    # Loss
    plt.figure()
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss Curve")
    plt.savefig(os.path.join(save_dir, "loss_curve.png"))
    plt.close()

    # Accuracy
    plt.figure()
    plt.plot(history["train_acc"], label="train_acc")
    plt.plot(history["val_acc"], label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy Curve")
    plt.savefig(os.path.join(save_dir, "acc_curve.png"))
    plt.close()

def plot_cm(y_true, y_pred, classes: List[str], save_path: str,normalize: bool = False):
    #混淆矩陣圖片並儲存
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha="right")
    plt.yticks(tick_marks, classes)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
