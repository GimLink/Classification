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
import tqdm
from timm.loss import LabelSmoothingCrossEntropy
import pandas as pd
import sys
from utils import *

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

    

    train_dataset = custom_dataset('Python/0109/dataset/train', transform = train_transform)
    val_dataset = custom_dataset('Python/0109/dataset/valid', transform = val_transform)
    test_dataset = custom_dataset('Python/0109/dataset/test', transform = test_transform)

    train_loader = DataLoader(train_dataset, batch_size= 64, num_worker = 2, pin_memory = True)
    val_loader = DataLoader(val_dataset, batch_size= 64, num_worker = 2, pin_memory = True)
    test_loader = DataLoader(test_dataset, batch_size= 1, shuffle=False)

    model = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_patch16_224', pretrained=True)
    model.head = nn.Linear(in_features= 192, out_features= 100)
    model.to(device)
    epochs = 10

    criterion = LabelSmoothingCrossEntropy()
    optimizer = optim.AdamW(model.parameters(), lr = 0.001)

    # lr scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 30, gamma = 0.1)





    if __name__ == '__main__':
        train(model, train_loader, val_loader, criterion, optimizer,scheduler, device)

    