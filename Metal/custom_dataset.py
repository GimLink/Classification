import os
import glob
import json
import cv2
from torch.utils.data import Dataset

class metal_dataset(Dataset):
    def __init__(self, file_path, transform = None):
        self.file_path = glob.glob(os.path.join(file_path, '*', '*.jpg'))
        self.transform = transform

        self.label_dict = {}
        for i, category in enumerate(os.listdir('Python/0116/data_pad/train')):
            self.label_dict[category] = int(i)
        
        self.image_list = []
        for i in self.file_path:
            image = cv2.imread(i)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.image_list.append(image)

        self.label_list = []
        for i in self.file_path:
            label_temp = i.split('/')[4]
            label = self.label_dict[label_temp]
            self.label_list.append(label)

    def __getitem__(self, index):
        image = self.image_list[index]
        label = self.label_list[index]
        
        if self.transform is not None:
            image = self.transform(image = image)['image']
        
        print(image, label)
        return image, label
        
    def __len__(self):
        return len(self.file_path)

if __name__ == '__main__':
    test = metal_dataset('Python/0116/data_pad/test')
    for i in test:
        pass



