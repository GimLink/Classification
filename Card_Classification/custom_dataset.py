import glob
import os
import torch
from torch.utils.data import Dataset
# from albumentations.pytorch import ToTensorV2
# import albumentations as A
import cv2
import numpy as np

# 시드 고정하는 법
# random_seed =44444
# torch.manual_seed(random_seed)
# torch.cuda.manual_seed(random_seed)
# torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# np.random.seed(random_seed)
# random.seed(random_seed)


class custom_dataset(Dataset):
    def __init__(self, file_path, transform = None):
        self.file_path = glob.glob(os.path.join(file_path, '*', '*.jpg'))
        self.class_name = os.listdir(file_path)
        self.class_name.sort()
        self.transform = transform
        self.file_path.sort()
        self.labels = []
        for path in self.file_path:
            self.labels.append(self.class_name.index(path.split('/')[4])) # 'Python/1230/dataset/train'
        self.labels = np.array(self.labels)
        print(self.labels)
        

    def __getitem__(self, index):
        image_path = self.file_path[index]
        image = cv2.imread(image_path)
        label = int(self.labels[index])
        if self.transform is not None:
            image = self.transform(image=image)['image']
        
        return image, label

    def __len__(self):
        return len(self.file_path)


test = custom_dataset('Python/1230/archive/train', transform = None)

for i in test:
    pass