import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import SimpleCNN

# 1. 데이터 로드
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# 2. 학습
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
torch.save(model.state_dict(), 'models/mnist_baseline.pth')