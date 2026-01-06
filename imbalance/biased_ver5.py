import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
from model import SimpleCNN

# 1. 데이터 로드
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

# 2. 롱테일 분포 (Long-tail Distribution)
# 지수적으로 감소
class_ratios = {
    0: 1.0,      # 100%
    1: 0.5,      # 50%
    2: 0.25,     # 25%
    3: 0.125,    # 12.5%
    4: 0.0625,   # 6.25%
    5: 0.03125,  # 3.125%
    6: 0.03125,  # 3.125%
    7: 0.03125,  # 3.125%
    8: 0.03125,  # 3.125%
    9: 0.03125,  # 3.125%
}

# 클래스별로 샘플링하여 불균형 데이터셋 생성
indices = []
targets = np.array(train_dataset.targets)

for class_idx in range(10):
    class_indices = np.where(targets == class_idx)[0]
    n_samples = int(len(class_indices) * class_ratios[class_idx])
    # 최소 10개는 보장
    n_samples = max(n_samples, 10)
    selected = np.random.choice(class_indices, n_samples, replace=False)
    indices.extend(selected)

# 불균형 데이터셋 생성
biased_train_dataset = Subset(train_dataset, indices)

# 클래스별 샘플 수 출력
print("=== 편향된 학습 데이터 분포 (롱테일 분포) ===")
biased_targets = [train_dataset.targets[i] for i in indices]
for class_idx in range(10):
    count = biased_targets.count(class_idx)
    print(f"Class {class_idx}: {count} samples ({class_ratios[class_idx]*100:.2f}%)")
print(f"Total: {len(biased_targets)} samples\n")

train_loader = DataLoader(biased_train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# 3. 학습
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f'Epoch {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.4f}')

def test():
    model.eval()
    correct = 0
    class_correct = [0] * 10
    class_total = [0] * 10

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()

            # 클래스별 정확도 계산
            for i in range(len(target)):
                label = target[i].item()
                class_total[label] += 1
                if pred[i] == label:
                    class_correct[label] += 1

    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest Accuracy: {accuracy:.2f}%')

    # 클래스별 정확도 출력
    print("=== 클래스별 테스트 정확도 ===")
    for i in range(10):
        if class_total[i] > 0:
            class_acc = 100. * class_correct[i] / class_total[i]
            print(f'Class {i}: {class_acc:.2f}% ({class_correct[i]}/{class_total[i]})')
    print()

    return accuracy

# 학습 실행
for epoch in range(1, 6):  # 5 epochs
    train(epoch)
    test()

# 모델 저장
os.makedirs('models', exist_ok=True)
torch.save(model.state_dict(), 'models/mnist_biased_ver5.pth')
print("Model saved to models/mnist_biased_ver5.pth")
