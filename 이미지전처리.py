import os
import glob
import argparse
from PIL import Image

# /dataset/image/폴더명/리사이즈 됨 이미지 저장

def expand2square(img, background_color) :
    width_temp, height_temp = img.size
    if width_temp == height_temp :
        return img
    elif width_temp > height_temp :
        result = Image.new(img.mode, (width_temp, width_temp), background_color)
        # image add (추가 이미지, 붙일 위치 (가로, 세로))
        result.paste(img, (0, (width_temp-height_temp) // 2))
        return result
    else :
        result = Image.new(img.mode, (height_temp, height_temp), background_color)
        result.paste(img,((height_temp-width_temp) // 2, 0))
        return result

def image_procesing(orange_data, grapefruit_data, kanpei_data, dekopon_data):
    orange = orange_data
    grapefruit = grapefruit_data
    kanpei = kanpei_data
    dekopon = dekopon_data

    for i in orange:
        # 이미지 읽고 가로 세로 expand2square에 던지기
        file_name = i.split('/')
        # Python/1219/image/Orange/0.jpg
        file_name = file_name[4]
        file_name = file_name.replace('.jpg', '.png')
        orange_img = Image.open(i)
        orange_img_resize = expand2square(orange_img, (0, 0, 0)).resize((400, 400))
        os.makedirs('Python/1219/dataset/image/orange', exist_ok=True)
        orange_img_resize.save(f'Python/1219/dataset/image/orange/{file_name}')

    for i in grapefruit:
        # 이미지 읽고 가로 세로 expand2square에 던지기
        file_name = i.split('/')
        file_name = file_name[4]
        file_name = file_name.replace('.jpg', '.png')
        grapefruit_img = Image.open(i)
        grapefruit_img_resize = expand2square(grapefruit_img, (0, 0, 0)).resize((400, 400))
        os.makedirs('Python/1219/dataset/image/grapefruit', exist_ok=True)
        grapefruit_img_resize.save(f'Python/1219/dataset/image/grapefruit/{file_name}')

    for i in kanpei:
        # 이미지 읽고 가로 세로 expand2square에 던지기
        file_name = i.split('/')
        file_name = file_name[4]
        file_name = file_name.replace('.jpg', '.png')
        kanpei_img = Image.open(i)
        kanpei_img_resize = expand2square(kanpei_img, (0, 0, 0)).resize((400, 400))
        os.makedirs('Python/1219/dataset/image/kanpei', exist_ok=True)
        kanpei_img_resize.save(f'Python/1219/dataset/image/kanpei/{file_name}')

    for i in dekopon:
        # 이미지 읽고 가로 세로 expand2square에 던지기
        file_name = i.split('/')
        file_name = file_name[4]
        file_name = file_name.replace('.jpg', '.png')
        dekopon_img = Image.open(i)
        dekopon_img_resize = expand2square(dekopon_img, (0, 0, 0)).resize((400, 400))
        os.makedirs('Python/1219/dataset/image/dekopon', exist_ok=True)
        dekopon_img_resize.save(f'Python/1219/dataset/image/dekopon/{file_name}')


def image_file_check(image_path):

    # 각 폴더별 데이터 양 체크
    image_path = opt.image_folder_path

    # 오렌지
    orange_data = glob.glob(os.path.join(image_path, 'orange', '*.jpg'))
    print('오렌지 데이터 개수 >> ', len(orange_data))
    # 자몽
    grapefruit_data = glob.glob(os.path.join(image_path, 'grapefruit', '*.jpg'))
    print('자몽 데이터 개수 >> ', len(grapefruit_data))
    # 레드향
    kanpei_data = glob.glob(os.path.join(image_path, 'kanpei', '*.jpg'))
    print('레드향 데이터 개수 >> ', len(kanpei_data))
    # 한라봉
    dekopon_data = glob.glob(os.path.join(image_path, 'dekopon', '*.jpg'))
    print('한라봉 데이터 개수 >> ', len(dekopon_data))

    return orange_data, grapefruit_data, kanpei_data, dekopon_data



def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-folder-path', type=str, default='Python/1219/image')
    opt = parser.parse_args()

    return opt

if __name__ == "__main__":
    opt = parse_opt()
    orange_data, grapefruit_data, kanpei_data, dekopon_data = image_file_check(opt)
    image_procesing(orange_data, grapefruit_data, kanpei_data, dekopon_data)
