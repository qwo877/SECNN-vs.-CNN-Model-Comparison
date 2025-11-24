# test.py
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from model import CNN, SECNN
from utils import plot_cm
import argparse
import os

def get_test_loader(batch_size: int = 64):
    transform_test = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465),
                    (0.2470, 0.2435, 0.2616)),
    ])

    test_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=2)
    return test_loader, test_dataset.classes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default="C:\\Users\\user0409\\AI\\AI專案\\SE CNN\\checkpoints\\best_se.pth",
                        help="模型權重檔路徑")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--out_dir", type=str, default="./results")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用裝置：{device}")

    # 載入 checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    model_type = ckpt.get("model_type", "se")
    num_classes = ckpt.get("num_classes", 10)

    if model_type == "cnn":
        model = CNN(num_classes=num_classes)
    else:
        model = SECNN(num_classes=num_classes)

    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    test_loader, class_names = get_test_loader(batch_size=args.batch_size)

    y_true = []
    y_pred = []
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(1)

            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    print(f"Test Accuracy: {acc:.4f}")

    # 畫混淆矩陣
    os.makedirs(args.out_dir, exist_ok=True)
    cm_path = os.path.join(args.out_dir, "confusion_matrix.png")
    plot_cm(y_true, y_pred, class_names, save_path=cm_path,
                             normalize=True)
    print(f"混淆矩陣已儲存到: {cm_path}")

if __name__ == "__main__":
    main()
