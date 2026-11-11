
import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

BASE = Path(__file__).resolve().parent
DATA = BASE.parents[2] / "work" / "datasets"
CLASSES = ("飞机", "汽车", "鸟", "猫", "鹿", "狗", "青蛙", "马", "船", "卡车")


# ==================== 核心 1：网络 ====================
def build_model() -> nn.Module:
    class CNN(nn.Module):
        def __init__(self):                                   
            super().__init__()                                
            self.features = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, padding=1),   #卷积层，输入通道数为3，输出通道数为16，卷积核大小为3x3，填充为1
                nn.ReLU(),                                    #ReLU
                nn.MaxPool2d(2),                              #最大池化层，池化窗口大小为2x2
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
            self.head = nn.Sequential(
                nn.Flatten(),                                  #展平
                nn.Linear(32 * 8 * 8, 64),                     #全连接层，输入特征数为32*8*8，输出特征数为64
                nn.ReLU(),
                nn.Linear(64, 10),
            )

        def forward(self, images):                              #前向传播函数，输入为图像张量，输出为未归一化的分数（logits）
            features = self.features(images)                    
            logits = self.head(features)                        #返回未归一化的分数（logits），形状为 (B, 10)，B 是批量大小
            return logits                                       
    return CNN()

# ==================== 核心 2：损失 ====================
def calculate_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    classification_loss = nn.CrossEntropyLoss()      #创建交叉熵损失函数
    loss=classification_loss(logits, labels)         #计算交叉熵损失
    return loss                                      #返回损失

# ==================== 核心 3：参数更新 ====================
def update_parameters(loss: torch.Tensor, optimizer: torch.optim.Optimizer) -> None:
    optimizer.zero_grad()  # 清除旧梯度
    loss.backward()        # 反向传播计算梯度
    optimizer.step()       # 更新参数


# ==================== 以下是已连接的外围接口 ====================
def check_output(logits, labels):
    if not isinstance(logits, torch.Tensor) or logits.shape != (len(labels), 10):
        raise ValueError("网络输出必须是形状 (B, 10) 的张量")
    loss = calculate_loss(logits, labels)
    if not isinstance(loss, torch.Tensor) or loss.ndim != 0:
        raise ValueError("损失必须是标量张量；请勿返回 loss.item()")
    if not torch.isfinite(loss).item():
        raise ValueError("损失出现 NaN 或无穷大")
    return loss


def check_interfaces(model, device):
    """只检查前向计算和损失接口，不改动模型权重。"""
    model.eval()
    with torch.no_grad():
        images = torch.rand(4, 3, 32, 32, device=device)
        labels = torch.tensor([0, 3, 5, 9], device=device)
        logits = model(images)
        loss = check_output(logits, labels)
    print(f"接口通过：图片 {tuple(images.shape)} → 分数 {tuple(logits.shape)}")
    print(f"损失形状 {tuple(loss.shape)}，数值 {loss.item():.4f}")
    print("此检查未运行核心 3；参数更新在训练阶段执行。")


def make_loaders(batch_size, download):
    transform = transforms.ToTensor()
    try:
        full_train = datasets.CIFAR10(DATA, train=True, transform=transform, download=download)
        test_data = datasets.CIFAR10(DATA, train=False, transform=transform, download=download)
    except RuntimeError as error:
        raise RuntimeError(
            "CIFAR-10 缓存尚未完整就绪。等待 lesson02 下载完成，"
            "或运行本文件时加 --download。"
        ) from error
    # 验证集取自原训练集，正式测试集保留到训练完成后使用。
    train_data, val_data = random_split(
        full_train, [45000, 5000], generator=torch.Generator().manual_seed(42)
    )
    common = dict(batch_size=batch_size, num_workers=0)
    return (
        DataLoader(train_data, shuffle=True, **common),
        DataLoader(val_data, shuffle=False, **common),
        DataLoader(test_data, shuffle=False, **common),
    )


def run_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = total_correct = total_count = 0
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = check_output(logits, labels)
            if training:
                update_parameters(loss, optimizer)
            count = len(labels)
            total_loss += loss.item() * count
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_count += count
    return total_loss / total_count, total_correct / total_count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        parser.error("epochs、batch-size 和 lr 必须大于 0")
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("运行设备：", device)
    try:
        model = build_model().to(device)
        check_interfaces(model, device)
        if args.check:
            return
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        train_loader, val_loader, test_loader = make_loaders(args.batch_size, args.download)
        save_path = BASE / "cifar10-best.pt"
        best_accuracy = -1.0
        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = run_epoch(model, train_loader, device, optimizer)
            val_loss, val_acc = run_epoch(model, val_loader, device)
            print(
                f"第 {epoch}/{args.epochs} 轮 | 训练损失 {train_loss:.4f} "
                f"准确率 {train_acc:.1%} | 验证损失 {val_loss:.4f} 准确率 {val_acc:.1%}"
            )
            if val_acc > best_accuracy:
                best_accuracy = val_acc
                torch.save({"model": model.state_dict(), "classes": CLASSES}, save_path)
        checkpoint = torch.load(save_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model"])
        test_loss, test_acc = run_epoch(model, test_loader, device)
        print(f"最佳验证模型的测试结果：损失 {test_loss:.4f}，准确率 {test_acc:.1%}")
        print("已保存：", save_path)
    except NotImplementedError as error:
        print(f"待填写：{error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
