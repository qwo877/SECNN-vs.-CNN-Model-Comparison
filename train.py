import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split
from model import CNN, SECNN
from utils import curves
from tqdm import tqdm
import argparse
import os

def get_dataloaders(batch_size: int = 64, val_ratio: float = 0.1):
    #載入 CIFAR-10，切 train/val
    transform_train = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomCrop(32, padding=4),
        T.RandomRotation(10),        # 新增
        T.ColorJitter(0.2, 0.2),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465),
                    (0.2470, 0.2435, 0.2616)),
    ])

    transform_test = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465),
                    (0.2470, 0.2435, 0.2616)),
    ])

    dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )

    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # val 用較穩定的 transform
    val_dataset.dataset.transform = transform_test

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=2)

    return train_loader, val_loader

def validate(model, loader, device, loss_fn):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = loss_fn(outputs, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (outputs.argmax(1) == labels).sum().item()
            total_samples += batch_size

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="cnn",
                        choices=["cnn", "se"], help="選擇模型：cnn 或 se")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用：{device}")

    # 建立模型
    if args.model == "cnn":
        model = CNN(num_classes=10)
    else:
        model = SECNN(num_classes=10)

    model = model.to(device)

    # Data
    train_loader, val_loader = get_dataloaders(batch_size=args.batch_size)

    # Loss, Optimizer
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.SGD(model.parameters(), 
                     lr=0.01,
                     momentum=0.9,
                     weight_decay=5e-4,  # L2正則化
                     nesterov=True)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    os.makedirs(args.save_dir, exist_ok=True)
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        total_samples = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            running_correct += (outputs.argmax(1) == labels).sum().item()
            total_samples += batch_size

            train_loss = running_loss / total_samples
            train_acc = running_correct / total_samples
            pbar.set_postfix({
                "train_loss": f"{train_loss:.4f}",
                "train_acc": f"{train_acc:.4f}"
            })

        # epoch 結束後做 validation
        val_loss, val_acc = validate(model, val_loader, device, loss_fn)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"[Epoch {epoch}] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # 儲存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(
                args.save_dir,
                f"best_{args.model}.pth"
            )
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_type": args.model,
                "num_classes": 10,
            }, save_path)
            print(f"儲存最佳模型到: {save_path}")

    # 畫曲線
    curves(history, save_dir=args.save_dir)
    print("完成")

if __name__ == "__main__":
    main()
