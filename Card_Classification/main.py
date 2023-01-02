import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from custom_dataset import custom_dataset
from timm.loss import LabelSmoothingCrossEntropy
from adamp import AdamP
from utils import *

# pip install adamp

# augmentation
def main(opt):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_transform = A.Compose([
        A.SmallestMaxSize(max_size=160),
        A.RandomToneCurve(),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15,
                   b_shift_limit=15, p=0.7),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2)),
        ToTensorV2()
    ])

    val_transform = A.Compose([
        A.SmallestMaxSize(max_size=160),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2)),
        ToTensorV2()
    ])

    # dataset
    train_data = custom_dataset(file_path=opt.train_path, transform=train_transform)
    val_data = custom_dataset(file_path=opt.val_path, transform=val_transform)
    test_data = custom_dataset(file_path=opt.test_path, transform = val_transform)

    # dataloader
    train_loader = DataLoader(
        train_data, batch_size=opt.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=opt.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1, shuffle = False)

    # model call
    net = models.__dict__['resnet50'](pretrained=True)
    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, 53)
    net.to(device)
    # print(net)

    # loss
    criterion = LabelSmoothingCrossEntropy()
    criterion = criterion.to(device)
    # optimizer
    optimizer = AdamP(net.parameters(), lr = opt.lr, weight_decay = 1e-2)

    # scheduler
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = [60, 80])

    # model save
    save_dir = opt.save_dir
    os.makedirs(save_dir, exist_ok=True)

    # train(num_epoch, model, train_loader, val_loader, criterion, optimizer, scheduler, save_dir, device):
    train_flg = opt.train_flag
    if train_flg == True:
        train(opt.epoch, net, train_loader, val_loader, criterion, optimizer, scheduler, save_dir, device )
    else:
        # test_species(test_loader, device)
        test_show(test_loader, device)

    # train
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-path', type=str,
                        default='Python/1230/archive/train', help='train data path')
    parser.add_argument(
        '--val-path', type=str, default='Python/1230/archive/valid', help='val data path')
    parser.add_argument('--train-flag', type = bool, default = False, help = 'train or test mode flag')
    parser.add_argument('--test-path', type = str, default = 'Python/1230/archive/test', help = 'test path')
    parser.add_argument('--batch-size', type=int,
                        default= 128, help='batch size')
    parser.add_argument('--epoch', type=int,
                        default=100, help='epoch number')
    parser.add_argument('--lr', type=float,
                        default=0.001, help='learning rate')
    parser.add_argument('--save-dir', type=str, default = 'Python/1230/weights', help = 'weight pt save dir')

    opt = parser.parse_args()

    return opt

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
