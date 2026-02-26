import torch
from torchvision.datasets import CIFAR10
from PIL import Image
import numpy as np

class CustomCIFAR10(CIFAR10):
    def __init__(self, class_num,max_num,imb_ratio,**kwds):
        super().__init__(**kwds)
        n_per_class = make_imb_data(max_num, class_num, imb_ratio, 'long')
        train_idx = train_split(self.targets, n_per_class)
        if train_idx is not None:
            self.data = self.data[train_idx, :, :, :]
            self.targets = np.array(self.targets)[train_idx]
        self.idxsPerClass = [np.where(np.array(self.targets) == idx)[0] for idx in range(10)]
        self.idxsNumPerClass = [len(idxs) for idxs in self.idxsPerClass]
    def __getitem__(self, idx):
        img = self.data[idx]
        img = Image.fromarray(img).convert('RGB')
        imgs = self.transform(img)
        target=self.targets[idx]
        return imgs,target,idx
def train_split(labels, n_per_class):
    labels = np.array(labels)
    train_idxs = []
    length = len(n_per_class)
    for i in range(length):
        idxs = np.where(labels == i)[0]
        train_idxs.extend(idxs[:n_per_class[i]])
    return train_idxs


def make_imb_data(max_num, class_num, gamma, imb):
    if imb == 'long':
        mu = np.power(1 / gamma, 1 / (class_num - 1))
        class_num_list = []
        for i in range(class_num):
            if i == (class_num - 1):
                class_num_list.append(int(max_num / gamma))
            else:
                class_num_list.append(int(max_num * np.power(mu, i)))
        print(class_num_list)
    if imb == 'step':
        class_num_list = []
        for i in range(class_num):
            if i < int((class_num) / 2):
                class_num_list.append(int(max_num))
            else:
                class_num_list.append(int(max_num / gamma))
        print(class_num_list)
    return list(class_num_list)

