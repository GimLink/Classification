import json
import os
import glob
import cv2
from PIL import Image

json_path = 'Python/0116/metal/anno/annotation.json'

with open(json_path, 'r') as j:
        metal_data = json.load(j)


path_dict = {}
def seperate(file_path, save_root_img):
    # 파일 이름 분리
    all_file_path = glob.glob(os.path.join(file_path, '*.jpg'))
    for i in all_file_path:
        path_dict[i.split('/')[4]] = i


    for filename in metal_data.keys():
        
    
        annos = metal_data[filename]['anno']
        img_path = path_dict[filename]
        img = cv2.imread(img_path)
        for idx, anno in enumerate(annos):
            label = anno['label']
            save_path = os.path.join(save_root_img, label)
            os.makedirs(save_path, exist_ok=True)



            filename_new = filename.split('.')[0] + f'_{idx}.jpg'
            save_path = os.path.join(save_path, filename_new)

    

            bbox = anno['bbox']
            img_cropped = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]



            cv2.imwrite(save_path, img_cropped)

def expand2square(pil_img, background_color) :
    width, height = pil_img.size
    if width == height :
        return pil_img
    elif width > height :
        result = Image.new(pil_img.mode, (width, width), background_color)
        # image add (추가 이미지, 붙일 위치 (가로, 세로))
        result.paste(pil_img, (0, (width-height) // 2))
        return result
    else :
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img,((height-width) // 2, 0))
        return result

def padding(file_path):
    all_file_path = glob.glob(os.path.join(file_path,'*', '*.jpg'))
    for i in all_file_path:
        pil_img = Image.open(i)
        new_image = expand2square(pil_img, 0).resize((256,256))
        new_image.save(i, quality = 100)
# img = Image.open('Python/1213/motorcycle_7.jpg')
# img_new = expand2square(img, (0, 0, 0)).resize((256, 256))
# img_new.save('Python/1213/new_motorcycle_7.png', quality = 100)

seperate('Python/0116/metal/images', 'Python/0116/dataset')
# padding('Python/0116/dataset')