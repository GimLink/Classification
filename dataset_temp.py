from torch.utils.data import Dataset
import cv2
import os
import glob

label_dict = {'dekopon' : 0, 'grapefruit' : 1, 'kanpei' : 2, 'orange' : 3}

class custom_dataset(Dataset):
    def __init__(self, image_file_path, transform=None):
        self.image_file_paths = glob.glob(os.path.join(image_file_path, '*', '*.png'))
        self.transform = transform
        # print(self.image_file_path)

    def __getitem__(self, index):
        image_path = self.image_file_paths[index]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label_temp = image_path.split('/')
        # Python/1219/dataset/image/train/xxx
        label_temp = label_temp[5]
        label = label_dict[label_temp]

        if self.transform is not None:
            image = self.transform(image=image)['image']
        image = image.float()
        return image, label

    def __len__(self):
        return len(self.image_file_paths)


if __name__ == '__main__':
    test = custom_dataset('Python/1219/dataset/image/train', transform = None)
    for i in test:
        print(i)
        break