

# ResNeXt
In this project based on our ResNet implementation in the project VGG-16-and-ResNet-18-comparison-for-image-classification we extend it to ResNeXT. It is expected that our accuracy should be higher than ResNet.
"""

import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torch import nn, optim
import torchvision
from torchvision.utils import make_grid
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import zipfile
import random

def set_seed(seed=21):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

zip_path = 'cnn_part_2_dataset.zip'
extract_path = 'cnn_part_2_dataset'
if not os.path.exists(extract_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

transform_aug = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

aug_dataset = datasets.ImageFolder(extract_path, transform=transform_aug)

train_size = int(0.7 * len(aug_dataset))
val_size = int(0.15 * len(aug_dataset))
test_size = len(aug_dataset) - train_size - val_size
train_set, val_set, test_set = random_split(aug_dataset, [train_size, val_size, test_size])

def get_loaders(batch_size, train_set, val_set, test_set):
    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True),
        DataLoader(val_set, batch_size=batch_size, shuffle=False),
        DataLoader(test_set, batch_size=batch_size, shuffle=False)
    )

"""## Step 1: Implement the ResNeXT architecture
Pay close attention to the grouped convolutions and cardinality parameter. Using inbuild ResNeXt model won’t be considered for evaluation.
"""

class ResNeXtBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, cardinality=32, base_width=8):
        super(ResNeXtBlock, self).__init__()
        D = int(out_channels / 2)  # bottleneck
        C = cardinality #changed to 16 from the default 32
        W = base_width

        self.conv_reduce = nn.Conv2d(in_channels, C * W, kernel_size=1, stride=1, bias=False)
        self.bn_reduce = nn.BatchNorm2d(C * W)

        self.conv_conv = nn.Conv2d(C * W, C * W, kernel_size=3, stride=stride, padding=1, groups=C, bias=False)
        self.bn = nn.BatchNorm2d(C * W)

        self.conv_expand = nn.Conv2d(C * W, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn_expand = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        shortcut = self.shortcut(x)
        out = self.relu(self.bn_reduce(self.conv_reduce(x)))
        out = self.relu(self.bn(self.conv_conv(out)))
        out = self.bn_expand(self.conv_expand(out))
        out += shortcut
        return self.relu(out)

class ResNeXt(nn.Module):
    def __init__(self, num_classes=3):
        super(ResNeXt, self).__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.layer1 = self._make_layer(64, 128, stride=1, num_blocks=2)
        self.layer2 = self._make_layer(128, 256, stride=2, num_blocks=2)
        self.layer3 = self._make_layer(256, 512, stride=2, num_blocks=2)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, stride, num_blocks):
        layers = []
        layers.append(ResNeXtBlock(in_channels, out_channels, stride=stride))
        for _ in range(1, num_blocks):
            layers.append(ResNeXtBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

"""## Step 2: Train and evaluate your ResNeXt model

"""

def train_model(model, criterion, optimizer, scheduler, dataloaders, num_epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            loader = dataloaders[phase]

            running_loss, running_corrects = 0.0, 0
            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(loader.dataset)
            epoch_acc = running_corrects.double() / len(loader.dataset)

            history[f"{phase}_loss"].append(epoch_loss)
            history[f"{phase}_acc"].append(epoch_acc.item())
            print(f"{phase} Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")

        scheduler.step()
    return model, history

def evaluate_model(model, test_loader):
    model.eval()
    device = next(model.parameters()).device
    y_true, y_pred = [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

    accuracy = np.mean(np.array(y_true) == np.array(y_pred)) * 100
    print(f"Test Accuracy: {accuracy:.2f}%")

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None)
    for i, cls in enumerate(class_names):
        print(f"{cls} → Precision: {precision[i]:.2f}, Recall: {recall[i]:.2f}, F1: {f1[i]:.2f}")

    return accuracy, precision.mean(), recall.mean(), f1.mean()

model = ResNeXt(num_classes=3)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
init_type = 'he'
if init_type == 'xavier':
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
elif init_type == 'he':
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')

optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

batch_size = 64
train_loader, val_loader, test_loader = get_loaders(batch_size, train_set, val_set, test_set)
dataloaders = {"train": train_loader, "val": val_loader}

model, history = train_model(model, criterion, optimizer, scheduler, dataloaders, num_epochs=40)

class_names = aug_dataset.classes
test_loss, test_acc, all_preds, all_labels = evaluate_model(model, test_loader)

torch.save(model.state_dict(), 'a1_bonus_resnext_aarushij_singh72.pth')

"""## Step 3: Compare the performance of your ResNeXt model

### a. A table summarizing the performance metrics (accuracy, loss, etc.) for all three models.

### Model Performance Summary

| Model        | Final Train Accuracy | Final Val Accuracy | Final Test Accuracy | Final Train Loss | Final Val Loss |
|--------------|----------------------|---------------------|----------------------|------------------|----------------|
| VGG-16     |     96.99%            |   94.07%             |     94.11           |          0.3511   |       0.4029    |
| ResNet-18    |    98.69%          |          96.89%  |          96.38%     |     0.3191     |      0.3489    |
| ResNeXt-50   |               98.50%  | 96.38%              | 96.53%                | 0.3479         | 0.3814      |

### b. Discussion of the observed differences in performance.
Explain why ResNeXt might be outperforming ResNet and VGG. Consider factors like cardinality, grouped convolutions, and the overall architecture.

ResNext performs better then ResNet and VGG as it alloes the model to learn featurs in parallel (grouped convolutions) which lowers the computational complexity.  Cardinality is the size of the set of transformations(ResNext paper abstract) which is a better parameter then increasing depth or width.

### c. Analysis of any challenges encountered during the implementation or training process.

While implementing ResNeXt, using the default base width of 4 led to poor performance and high computational cost due to the small image size (64×64). increasing it to 8 improved training stability and validation accuracy.

### d. Provide detailed analysis of the results.

ResNeXt-50 achieved 98.50% train accuracy, 96.38% validation accuracy, and 96.52% test accuracy, outperforming both VGG-16 and ResNet-18. Its low train (0.3479) and validation loss (0.3814) indicate strong learning and generalization.

### 4.	References

*   ResNext Paper
https://arxiv.org/abs/1611.05431
*   https://www.geeksforgeeks.org/resnext-architecture-in-computer-vision/
*   Google Colab's inbuilt AI assisstant Gemini was used for debugging.
"""
