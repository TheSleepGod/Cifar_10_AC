import os

import torch
import torch.nn as nn
import torch.nn.functional as f
import matplotlib.pyplot as plt
from .base_bench import BaseBench


class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()

        self.conv1 = self._conv_block(3, 64)
        self.conv2 = self._conv_block(64, 64)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = self._conv_block(64, 128)
        self.conv4 = self._conv_block(128, 128)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = self._conv_block(128, 256)
        self.conv6 = self._conv_block(256, 256)
        self.conv7 = self._conv_block(256, 256)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv8 = self._conv_block(256, 512)
        self.conv9 = self._conv_block(512, 512)
        self.conv10 = self._conv_block(512, 512)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv11 = self._conv_block(512, 512)
        self.conv12 = self._conv_block(512, 512)
        self.conv13 = self._conv_block(512, 512)
        self.maxpool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512 * 1 * 1, 512)  # Adjust input size based on image dimensions
        self.fc2 = nn.Linear(512, num_classes)

        self.dropout = nn.Dropout(0.5)
        self.batchnorm = nn.BatchNorm1d(512)

    def _conv_block(self, in_channels, out_channels):
        """ Helper function to create a convolutional block with Conv -> BN -> ReLU. """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Convolutional and Max Pooling layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.maxpool1(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.maxpool2(x)

        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.maxpool3(x)

        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.maxpool4(x)

        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.maxpool5(x)

        # Flatten the tensor before passing to fully connected layers
        x = self.flatten(x)

        # Fully connected layers
        x = self.fc1(x)
        x = self.batchnorm(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return f.softmax(x, dim=1)


class CnnBench(BaseBench):
    def __init__(self, n_cls, name, cuda=True):
        self.name = name
        self.net = CNN(n_cls)
        if cuda:
            self.net = self.net.cuda()
        self.cuda = cuda

    def model(self):
        return self.net

    def train(self, data, iteration=200, lr=0.1, save_root='./checkpoints/'):
        optim = torch.optim.SGD(self.net.parameters(), lr=lr, momentum=0.9)
        self.net.eval()
        train_loss = []  # 记录batch训练过程中的loss变化
        for epoch in range(iteration):
            self.net.train()
            total_loss = 0
            for step, (imgs, labels) in enumerate(data):
                if self.cuda:
                    imgs = imgs.cuda()
                    labels = labels.cuda()
                # import pdb; pdb.set_trace();
                optim.zero_grad()
                predictions = self.net(imgs)
                predictions = f.softmax(predictions, dim=1)
                # print("predictions:{}, labels:{}".format(predictions, labels))
                # print(predictions)
                loss = f.cross_entropy(predictions, labels)
                # print(predictions, labels)
                # print("step: {}, loss: {}".format(step, loss))
                loss.backward()
                optim.step()
                total_loss += loss.item()
                # train_loss.append(loss.item())  # 记录最终训练误差      
                # print("step: {}, loss: {}".format(step, total_loss))
                if step % 1000 == 999:  # 每2000次打印一次
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, step + 1, total_loss / 2000))
                    total_loss = 0.0
            print("epoch: {}, loss: {}".format(epoch, total_loss / len(data)))
            train_loss.append(total_loss / len(data))  # 记录最终训练误差
        plt.plot(range(len(train_loss)), train_loss)
        plt.xlabel("Epoch")
        plt.ylabel("loss")
        plt.title("Train loss")
        plt.show()
        filepath = os.path.join(save_root, self.name)
        if not os.path.isdir(filepath):
            os.mkdir(filepath)
        if iteration % 1 == 0:
            self.save(os.path.join(filepath, 'epoch{}.pth'.format(iteration)))

    def test(self, data):
        self.net.eval()
        predictions = []
        labels = []
        for img, label in data:
            if self.cuda:
                img = img.cuda()
                label = label.cuda()
            pred = self.net(img)
            predictions.append(torch.argmax(pred))
            labels.append(torch.argmax(label))
        # confusion_matrix = F.confusion_matrix(predictions, labels)
        acc = 0
        for i in range(len(predictions)):
            if predictions[i] == labels[i]:
                acc += 1
        acc /= len(predictions)
        # print('Accuracy on test set:%d %%' % (acc))
        return predictions, acc

    def save(self, path):
        print(path)
        torch.save(self.net.state_dict(), path)

    def load(self, path):
        self.net.load_state_dict(torch.load(path))
