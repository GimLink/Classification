from sklearn.model_selection import train_test_split
import natsort
import os
import glob
import cv2
import shutil

# 데이터 나누기

orange_data_path = 'Python/1219/dataset/image/orange'
grapefruit_data_path = 'Python/1219/dataset/image/grapefruit'
kanpei_data_path = 'Python/1219/dataset/image/kanpei'
dekopon_data_path = 'Python/1219/dataset/image/dekopon'

orange_full_path = natsort.natsorted(glob.glob(os.path.join(f'{orange_data_path}/*.png')))
grapefruit_full_path = natsort.natsorted(glob.glob(os.path.join(f'{grapefruit_data_path}/*.png')))
kanpei_full_path = natsort.natsorted(glob.glob(os.path.join(f'{kanpei_data_path}/*.png')))
dekopon_full_path = natsort.natsorted(glob.glob(os.path.join(f'{dekopon_data_path}/*.png')))

orange_train_data, orange_val_data = train_test_split(orange_full_path, test_size = 0.1, random_state=666)

grapefruit_train_data, grapefruit_val_data = train_test_split(grapefruit_full_path, test_size = 0.1, random_state=666)

kanpei_train_data, kanpei_val_data = train_test_split(kanpei_full_path, test_size = 0.1, random_state=666)

dekopon_train_data, dekopon_val_data = train_test_split(dekopon_full_path, test_size = 0.1, random_state=666)

flog = True

for i in orange_train_data:
    img = cv2.imread(i)
    os.makedirs('Python/1219/dataset/image/train/orange/', exist_ok=True)
    file_name = os.path.basename(i)
    shutil.move(i,f'Python/1219/dataset/image/train/orange/{file_name}')

for i in orange_val_data:
    img = cv2.imread(i)
    os.makedirs('Python/1219/dataset/image/val/orange/', exist_ok=True)
    file_name = os.path.basename(i)
    shutil.move(i,f'Python/1219/dataset/image/val/orange/{file_name}')

for i in grapefruit_train_data:
    img = cv2.imread(i)
    os.makedirs('Python/1219/dataset/image/train/grapefruit/', exist_ok=True)
    file_name = os.path.basename(i)
    shutil.move(i,f'Python/1219/dataset/image/train/grapefruit/{file_name}')

for i in grapefruit_val_data:
    img = cv2.imread(i)
    os.makedirs('Python/1219/dataset/image/val/grapefruit/', exist_ok=True)
    file_name = os.path.basename(i)
    shutil.move(i,f'Python/1219/dataset/image/val/grapefruit/{file_name}')

for i in kanpei_train_data:
    img = cv2.imread(i)
    os.makedirs('Python/1219/dataset/image/train/kanpei/', exist_ok=True)
    file_name = os.path.basename(i)
    shutil.move(i,f'Python/1219/dataset/image/train/kanpei/{file_name}')

for i in kanpei_val_data:
    img = cv2.imread(i)
    os.makedirs('Python/1219/dataset/image/val/kanpei/', exist_ok=True)
    file_name = os.path.basename(i)
    shutil.move(i,f'Python/1219/dataset/image/val/kanpei/{file_name}')

for i in dekopon_train_data:
    img = cv2.imread(i)
    os.makedirs('Python/1219/dataset/image/train/dekopon/', exist_ok=True)
    file_name = os.path.basename(i)
    shutil.move(i,f'Python/1219/dataset/image/train/dekopon/{file_name}')

for i in dekopon_val_data:
    img = cv2.imread(i)
    os.makedirs('Python/1219/dataset/image/val/dekopon/', exist_ok=True)
    file_name = os.path.basename(i)
    shutil.move(i,f'Python/1219/dataset/image/val/dekopon/{file_name}')

