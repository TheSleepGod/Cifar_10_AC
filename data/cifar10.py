import os
import pickle
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# 类别名
LABEL_NAMES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# DATA_BATCHES = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
DATA_BATCHES = ['data_batch_1']
TEST_BATCHES = ['test_batch']


class Cifar10(Dataset):
    def __init__(self, root_path: str, train: bool, transform=None):
        super(Cifar10, self).__init__()
        self.transforms = transforms.Compose([transforms.ToTensor()])
        self.paths = []
        self.labels = []

        for index, label_name in enumerate(LABEL_NAMES):
            print(f"================={label_name}================")
            if train:
                base_path = os.path.join(root_path, 'cifar-10-output', 'train', label_name)
            else:
                base_path = os.path.join(root_path, 'cifar-10-output', 'test', label_name)
            for pic_name in os.listdir(base_path):
                print(f"pic_name：{pic_name} category: , {index}")
                self.paths.append(os.path.join(base_path, pic_name))
                self.labels.append(index)
        self.N_CLS = 10

    def __getitem__(self, index):
        path = self.paths[index]
        # print(path)
        img_ = Image.open(path).convert('RGB')
        if self.transforms is not None:
            img_ = self.transforms(img_)
        label_ = self.N_CLS * [0]
        label_[self.labels[index]] = 1
        label_ = torch.Tensor(label_)
        return img_, label_

    def __len__(self):
        return len(self.paths)


def binary2img(root_path, batch, save_cls):
    with open(os.path.join(root_path, 'cifar-10-batches-py', batch), 'rb') as f:
        img_dict = pickle.load(f, encoding='bytes')
    classification = {
        "0": "airplane",
        "1": "automobile",
        "2": "bird",
        "3": "cat",
        "4": "deer",
        "5": "dog",
        "6": "frog",
        "7": "horse",
        "8": "ship",
        "9": "truck",
    }
    for i in range(0, 10000):
        img_ = np.reshape(img_dict[b'data'][i], (3, 32, 32))
        img_ = np.transpose(img_, (1, 2, 0))
        label_ = str(img_dict[b'labels'][i])
        save_path = os.path.join(root_path, 'cifar-10-output', save_cls, str(classification.get(label_)))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        pic_name = save_path + f'\\%d.png' % i
        img_ = img_.transpose(1, 2, 0)
        img_ = Image.fromarray(img_, 'RGB')
        img_.save(pic_name)


if __name__ == '__main__':
    root = 'D:\cifar_10_AC\Cifar10-Adversarial-Competition\cifar-10-python'
    binary2img(root, 'data_batch_1', 'train')
    dataset = Cifar10(root, train=True)
    print(dataset.paths[0], dataset.labels[0])
    data = DataLoader(dataset, batch_size=1)

    for epoch in range(1):
        print("Epoch {}".format(epoch))
        for step, (img, label) in enumerate(data):
            print("step {}: {}".format(step, img.size()))
