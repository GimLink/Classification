import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import glob
import cv2
from torch.utils.data import Dataset

class custom_dataset(Dataset):
    def __init__(self, file_path, transform=None):
        self.image_path = glob.glob(os.path.join(file_path, '*', '*.png'))
        self.transform = transform

        self.label_dict = {}
        for i, category in enumerate(os.listdir('Python/0106/data/train')):
            self.label_dict[category] = int(i)

        self.image_list = []
        for i in self.image_path:
            image = cv2.imread(i)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.image_list.append(image)

        self.label_list = []
        for i in self.image_path:
            label_temp = i.split('/')[4]
            label = self.label_dict[label_temp]
            self.label_list.append(label)



    def __getitem__(self, index):
        image = self.image_list[index]
        label = self.label_list[index]

        if self.transform is not None:
            image = self.transform(image=image)['image']
        print(image, label)

        return image, label

    def __len__(self):
        return len(self.image_path)

if __name__ == '__main__':
    test = custom_dataset('Python/0106/data/train')
    for i in test:
        pass

