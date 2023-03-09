import cv2
from torchvision import models
from torchvision import transforms
import torch.nn as nn
import torch
from PIL import Image

webcam = cv2.VideoCapture(0)
webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# model
net = models.swin_t(weights='IMAGENET1K_V1')
net.head = nn.Linear(in_features=768, out_features=3)


# 학습시킨 모델 로드
models_loader_path = 'Python/0106//best.pt'
net.load_state_dict(torch.load(models_loader_path, map_location= 'cpu'))
net.to('cpu')

val_transforms = transforms.Compose([
    transforms.Resize((224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
def preprocessing(image):
    image = Image.fromarray(image)
    image = val_transforms(image)
    image = image.float()
    image = image.to('cpu')
    image = image.unsqueeze(0)

    return image


if not webcam.isOpened():
    print('Ther is no cam')
    exit()

while webcam.isOpened():
    status, frame = webcam.read()
    net.eval()
    with torch.no_grad():

        frame = cv2.flip(frame, 1) # 좌우대칭

        if status :
            image = preprocessing(frame)
            output = net(image)
            _, argmax = torch.max(output, 1)
            print('output', argmax)
            cv2.imshow('test', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
webcam.release()
cv2.destroyAllWindows()
