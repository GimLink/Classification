from custom_dataset import metal_dataset
from torch.utils.data import DataLoader
import torch.optim as optim
import torch
import torch.nn as nn
from torchvision import models
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import copy
import matplotlib.pyplot as plt
import os
import tqdm
from timm.loss import LabelSmoothingCrossEntropy
import pandas as pd
import sys

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_transform = A.Compose([
        A.SmallestMaxSize(max_size=160),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.09,
                           rotate_limit=25, p=1),
        A.Resize(width=224, height=224),
        A.RandomBrightnessContrast(p=0.5),
        A.RandomFog(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    val_transform = A.Compose([
        A.SmallestMaxSize(max_size = 256),
        A.Resize(height = 224, width = 224),
        A.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)),
        ToTensorV2()

    ])


    train_dataset = metal_dataset('Python/0116/data_pad/train', transform = train_transform)
    val_dataset = metal_dataset('Python/0116/data_pad/val', transform = val_transform)
    
    train_loader = DataLoader(train_dataset, batchsize = 128, shuffle = True, num_workers = 2, pin_memory = True)
    val_loader = DataLoader(val_dataset, batchsize = 128, shuffle = True, num_workers = 2, pin_memory = True)

    net = models.resnet50(pretrained = True)
    net.fc = nn.Linear(in_features=2048, out_features=10)
    net.to(device)

    loss_function = LabelSmoothingCrossEntropy()
    optimizer = torch.optim.AdamW(net.parameters(), lr=0.0001)
    epochs = 10

    best_val_acc = 0.0
    save_path = "best.pt"
    dfForAccuracy = pd.DataFrame(index=list(range(epochs)),
                                    columns=["Epoch", "TrainAccuracy", "TrainLoss", 'ValAccuracy', 'ValLoss'])
    if os.path.exists(save_path):
        best_val_acc = max(pd.read_csv("./modelAccuracy.csv")["Accuracy"].tolist())

    for epoch in range(epochs):
        train_running_loss = 0
        val_running_loss = 0
        val_acc = 0
        train_acc = 0

        net.train()
        train_bar = tqdm(train_loader, file=sys.stdout, colour='green')
        for step, data in enumerate(train_bar):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)

            loss = loss_function(outputs, labels)

            optimizer.zero_grad()
            train_acc += (torch.argmax(outputs, dim=1) == labels).sum().item()
            loss.backward()
            optimizer.step()
            train_running_loss += loss.item()

            train_bar.desc = f"train epoch [{epoch + 1}/{epochs}], loss >> {loss.data:.3f}"

        net.eval()
        with torch.no_grad():
            valid_bar = tqdm(val_loader, file=sys.stdout, colour='red')
            for data in valid_bar:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)

                loss = loss_function(outputs, labels)
                val_running_loss += loss

                val_acc += (torch.argmax(outputs, dim=1) == labels).sum().item()

        val_accuracy = val_acc / len(val_dataset)
        train_accuracy = train_acc / len(train_dataset)
        train_loss = train_running_loss / len(train_loader)
        val_loss = val_running_loss / len(val_loader)

        dfForAccuracy.loc[epoch, "Epoch"] = epoch + 1
        dfForAccuracy.loc[epoch, 'TrainAccuracy'] = round(train_accuracy, 3)
        dfForAccuracy.loc[epoch, 'TrainLoss'] = round(train_loss, 5)
        dfForAccuracy.loc[epoch, 'ValAccuracy'] = round(val_accuracy, 3)
        dfForAccuracy.loc[epoch, 'ValLoss'] = round(val_loss, 5)
        print(f"epoch [{epoch + 1}/{epochs}] train_loss {train_loss:.3f}"
                f"train acc : {train_accuracy:.3f} val_acc : {val_accuracy:.3f}"
                )

        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            torch.save(net.state_dict(), save_path)

        if epoch == epochs - 1:
            dfForAccuracy.to_csv("./modelAccuracy.csv", index=False)

    # torch.save(net.state_dict(), "./last.pt")

    

if __name__ == '__main__':
    main()