from custom_dataset import custom_dataset
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
os.environ['KMP_DUPLICATE_LIB_OK']='True'
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
        A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=1),
        A.RandomBrightnessContrast(p=0.5),
        A.HorizontalFlip(p=0.8),
        A.VerticalFlip(p=0.7),
        A.RandomShadow(p=1),
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

    test_transform = A.Compose([
        A.SmallestMaxSize(max_size = 256),
        A.Resize(height = 224, width = 224),
        A.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)),
        ToTensorV2()

    ])

    train_dataset = custom_dataset('Python/0106/data/train', transform = train_transform)
    val_dataset = custom_dataset('Python/0106/data/val', transform = val_transform)
    # test_dataset = custom_dataset('Python/0106/data/test', transform = test_transform)

    def visualize_augmentations(dataset, idx = 0, samples = 20, cols = 5):
        dataset = copy.deepcopy(dataset)
        dataset.transform = A.Compose([t for t in dataset.transform if not isinstance(t, (A.Normalize, ToTensorV2))])
        rows = samples // cols
        figure, ax = plt.subplots(nrows = rows, ncols = cols, figsize = (12, 6))
        for i in range(samples):
            image, _ = dataset[idx]
            ax.ravel()[i].imshow(image)
            ax.ravel()[i].set_axis_off()
        plt.tight_layout()
        plt.show()

    # visualize_augmentations(val_dataset)

    train_loader = DataLoader(train_dataset, batch_size = 128, shuffle = True, num_workers = 2, pin_memory = True)
    val_loader = DataLoader(val_dataset, batch_size = 128, shuffle = True, num_workers = 2, pin_memory = True)
    # test_loader = DataLoader(test_dataset, batch_size = 1, shuffle = False, num_workers = 2, pin_memory = True)
    

    net = models.swin_t(weights='IMAGENET1K_V1')
    net.head = nn.Linear(in_features=768, out_features=3)
    net.to(device)
    print(net)

    # net = models.resnet50(pretrained = True)
    # net.fc = nn.Linear(in_features=2048, out_features=450)
    # net.to(device)

    # net = models.efficientnet_b4(pretrained = True)
    # net.classifier[1] = nn.Linear(in_features=1792, out_features= 450)
    # net.to(device)

    # net = models.mobilenet_v3_large(pretrained=True)
    # net.classifier[3] = nn.Linear(in_features = 1280, out_features=450)
    # net.to(device)


    #### 4 epoch, optim loss
    loss_function = LabelSmoothingCrossEntropy()
    optimizer = torch.optim.AdamW(net.parameters(), lr=0.0001)
    epochs = 10

    best_val_acc = 0.0

    train_steps = len(train_loader)
    valid_steps = len(val_loader)
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

    torch.save(net.state_dict(), "./last.pt")

    

if __name__ == '__main__':
    main()