import os
import glob
import cv2
import random
import shutil


def test_val_split(file_path):
    all_category = os.listdir(file_path)
    for category in all_category:
        os.makedirs(f'Python/0106/data/train/{category}', exist_ok=True)
        all_image = os.listdir(file_path + f'/{category}/')
        for image in random.sample(all_image, int(0.9 * len(all_image))):
            shutil.move(file_path + f'/{category}/{image}', f'Python/0106/data/train/{category}/')
    for category in all_category:
        os.makedirs(f'Python/0106/data/val/{category}', exist_ok=True)
        all_image = os.listdir(file_path + f'/{category}/')
        for image in all_image:
            shutil.move(file_path + f'/{category}/{image}', f'Python/0106/data/val/{category}/')

def test_split(file_path):
    all_category = os.listdir(file_path)
    for category in all_category:
        os.makedirs(f'Python/0106/data/test/{category}', exist_ok= True)
        all_image = os.listdir(file_path + f'/{category}/')
        for image in random.sample(all_image, int(0.1 * len(all_image))):
            shutil.move(file_path + f'/{category}/{image}', f'Python/0106/data/test/{category}/')
            
def seperate(file_path):
    all_category = os.listdir(file_path)
    for category in all_category:
        os.makedirs(f'Python/0106/dataset/{category}', exist_ok= True)
        all_image = os.listdir(file_path + f'/{category}/')
        # for image in random.sample(all_image, int(0.5 * len(all_image))):
        for image in all_image:
            shutil.move(file_path + f'{category}/{image}', f'Python/0106/dataset/{category}/')


def jpg_to_png(file_path):
    for category in os.listdir(file_path):
        i = 1
        for image in os.listdir(file_path + category + '/'):
            os.rename(file_path + category + '/' + image, file_path + category + '/' + category + str(i) + '.png')
            i += 1





if __name__ == '__main__':
    file_path = 'Python/0106/archive'
    # test_val_split(file_path)
    seperate('Python/0106/data/test/')
    # jpg_to_png(file_path)
    # test_split(file_path)

