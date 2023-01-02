import os
import glob
import cv2
import shutil
from sklearn.model_selection import train_test_split
import argparse

cloudy_path = glob.glob(os.path.join('Python/0102/data/cloudy', '*.jpg'))
desert_path = glob.glob(os.path.join('Python/0102/data/desert', '*.jpg'))
green_path = glob.glob(os.path.join('Python/0102/data/green_area', '*.jpg'))
water_path = glob.glob(os.path.join('Python/0102/data/water', '*.jpg'))

# print(len(cloudy_path), len(desert_path), len(green_path), len(water_path))

cloudy_train, cloudy_val = train_test_split(cloudy_path, test_size= 0.2, random_state= 666)
cloudy_val, cloudy_test = train_test_split(cloudy_val, test_size= 0.5, random_state= 666)

desert_train, desert_val = train_test_split(desert_path, test_size= 0.2, random_state= 666)
desert_val, desert_test = train_test_split(desert_val, test_size= 0.5, random_state= 666)

green_train, green_val = train_test_split(green_path, test_size= 0.2, random_state= 666)
green_val, green_test = train_test_split(green_val, test_size= 0.5, random_state= 666)

water_train, water_val = train_test_split(water_path, test_size= 0.2, random_state = 666)
water_val, water_test = train_test_split(water_val, test_size= 0.5, random_state= 666)

def data_save(data, mode):
    for i in data:
        # move or image save
        image = cv2.imread(i)
        folder_name = i.split('/')[3]
        folder_path = f'Python/0102/dataset/{mode}/{folder_name}'
        image_name = os.path.basename(i)
        image_name = image_name.replace('.jpg', '')
        os.makedirs(folder_path, exist_ok = True)
        cv2.imwrite(os.path.join(folder_path, image_name + '.png'), image)
        # shutil.move(i, f'Python/0102/dataset/train/cloudy/{folder_name}')


data_save(water_test, mode = 'test')