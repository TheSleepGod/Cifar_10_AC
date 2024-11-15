import torch
from torch.utils.tensorboard.summary import image
import torchvision
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
torch.cuda.empty_cache()
myWriter = SummaryWriter('./tensorboard/log/')

myTransforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

train_dataset = torchvision.datasets.CIFAR10(root='./cifar-10-python/', train=True, download=True,
                                             transform=myTransforms)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)

test_dataset = torchvision.datasets.CIFAR10(root='./cifar-10-python/', train=False, download=True,
                                            transform=myTransforms)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=0)
torch.cuda.empty_cache()
myModel = torchvision.models.resnet50(pretrained=True)
channel = myModel.fc.in_features
myModel.fc = nn.Linear(channel, 10)

myDevice = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
myModel = myModel.to(myDevice)

learning_rate = 0.001
myOptimizer = optim.SGD(myModel.parameters(), lr=learning_rate, momentum=0.9)
myLoss = torch.nn.CrossEntropyLoss()

for _epoch in range(10):
    training_loss = 0.0
    for _step, input_data in enumerate(train_loader):
        image, label = input_data[0].to(myDevice), input_data[1].to(myDevice)  # GPU加速
        predict_label = myModel.forward(image)

        loss = myLoss(predict_label, label)

        myWriter.add_scalar('training loss', loss, global_step=_epoch * len(train_loader) + _step)

        myOptimizer.zero_grad()
        loss.backward()
        myOptimizer.step()

        training_loss = training_loss + loss.item()
        if _step % 10 == 0:
            print('[iteration - %3d] training loss: %.3f' % (_epoch * len(train_loader) + _step, training_loss / 10))
            training_loss = 0.0
            print()
    correct = 0
    total = 0
    torch.save(myModel,
               'D:\\cifar_10_AC\\Cifar10-Adversarial-Competition\\test_bench\\checkpoints\\Resnet50_1.pkl')
    myModel.eval()
    for images, labels in test_loader:
        images = images.to(myDevice)
        labels = labels.to(myDevice)
        outputs = myModel(images)
        numbers, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        predicted = torch.tensor(predicted)
        labels = torch.tensor(labels)
        correct += (predicted == labels).sum().item()

    print('Testing Accuracy : %.3f %%' % (100 * correct / total))
    myWriter.add_scalar('test_Accuracy', 100 * correct / total)
