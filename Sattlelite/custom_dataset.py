import torch
import torch.nn as nn
import numpy as np
import glob
import os
from PIL import Image
from torch.utils.data import Dataset

class custom_dataset(Dataset):
    def __init__(self,file_path, transform=None):
        self.file_paths = glob.glob(os.path.join(file_path, '*', '*.png'))
        self.transform = transform
        self.label_dict = {'cloudy' : 0, 'desert' : 1, 'green_area' : 2, 'water' : 3 }
        # init 에서 이미지 오픈 처리하는 방법
        # self.image_list = []
        # for image_path in self.file_paths:
        #     self.image_list.append(image_path)

    def __getitem__(self, index):
        # image = self.image_list[index]
        file_path = self.file_paths[index]
        label_temp = file_path.split('/')[4]
        label = self.label_dict[label_temp]
        image = Image.open(file_path)

        if self.transform is not None:
            image = self.transform(image)
        return image, label
    def __len__(self):
        return len(self.file_paths)

# # if __name__ == '__main__':
# test = custom_dataset('Python/0102/dataset/train')
# for i in test:
#     pass



