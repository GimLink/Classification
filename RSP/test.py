import torch

def acc_function(correct, total):
    acc = correct / total * 100
    return acc

def test(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_loader):
            image, label = images.to(device), labels.to(device)
            output = model(image)
            _, argmax = torch.max(output, 1)
            total += image.size(0)
            correct += (labels == argmax).sum().item()
        acc = acc_function(correct, total)
        print(f'acc >> {acc}%')