import time
import copy
import tqdm
import sys
import torch
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=20):
    total = 0
    best_loss = 9999
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    

    for epoch in range(num_epochs):

        print(f'Epoch {epoch} / {num_epochs}')
        print('--' * 10)

        for i, (image, label) in enumerate(train_loader):
            image, label = image.to(device), label.to(device)
            output = model(image)
            loss = criterion(output, label)
            scheduler.step()

            optimizer.zero_grad()
            _, argmax = torch.max(output, 1) # 라벨 중에서 가장 점수가 높은(추정한) 라벨을 가져옴
            acc = (label == argmax).float().mean() # 추정한(점수가 가장 높은) 값과 실제 라벨이 같은 경우 평균값 구하기
            loss.backward()
            optimizer.step()

            total += label.size(0)

            if (i + 1) & 10 ==0:
                print('Epoch [{}/{}] Loss {:.4f} Acc {:.2f}'.format(epoch + 1, num_epochs, loss.item(), acc.item() * 100 ))
        avrg_loss, val_acc = validation(epoch, model, val_loader, criterion, device)
        if avrg_loss < best_loss:
            best_loss = avrg_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            save_model(model, save_dir = './')

    time_elapsed = time.time()  - since
    print("Training complete in {:.0f}m {:.0f}s".format(
        time_elapsed // 60, time_elapsed % 6
    ))
    model.load_state_dict(best_model_wts)



def validation(epoch, model, val_loader, criterion, device):
    print('Validation # {} Start'.format(epoch+1))

    model.eval()
    with torch.no_grad():
        total = 0
        correct = 0
        total_loss = 0
        cnt = 0
        batch_loss = 0

        for i, (image, label) in enumerate(val_loader):
            image, label = image.to(device), label.to(device)
            output = model(output)
            loss = criterion(output, label)
            batch_loss += loss.item()

            total += image.size(0)
            _, argmax = torch.max(output, 1)
            correct += (label == argmax).sum().item()
            total_loss += loss.item()
            cnt += 1

    avrg_loss = total_loss / cnt
    val_acc = (correct / total * 100)
    print('Validation # {} Acc : {:.2f}% Average Loss : {:.4f}'.format(epoch +1, val_acc, avrg_loss))
    
    return avrg_loss, val_acc

def save_model(model, save_dir, file_name = 'best.pt'):
    output_path = os.path.join(save_dir, file_name)
    torch.save(model.state_dict(), output_path)
