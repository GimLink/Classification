import os
import glob
import cv2
from torch.utils.data import Dataset

class custom_dataset(Dataset):
    def __init__(self, file_path, transform=None):
        self.image_path = glob.glob(os.path.join(file_path, '*', '*.jpg'))
        self.transform = transform

        self.label_dict = {}
        for i, category in enumerate(os.listdir('Python/0109/dataset/train/')):
            self.label_dict[category] = int(i)

        self.image_list = []
        for i in self.image_path:
            image = cv2.imread(i)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.image_list.append(image)

        self.label_list = []
        for i in self.image_path:
            label = self.label_dict[i.split('/')[4]]
            self.label_list.append(label)

        
    def __getitem__(self, index):
        image = self.image_list[index]
        label = self.label_list[index]

        if self.transform is not None:
            image = self.transform(image = image)['image']

        return image, label
    def __len__(self):
        return len(self.image_path)

test = custom_dataset('Python/0109/dataset/train')
for i in test:
    pass