
import torch
import numpy as np
import os

from PIL import Image


from torch.utils.data import Dataset



from torchvision.datasets import ImageFolder

class CustomTinyImagenetLT(ImageFolder):
    def __init__(self, class_num,max_num,imb_ratio,**kwds):
        super().__init__(**kwds)
        targets = np.asarray([s[1] for s in self.samples])
        num_per_class = make_imb_data(max_num,class_num,imb_ratio,'long')
        idxs = train_split(targets,num_per_class)
        self.samples = [self.samples[idx] for idx in idxs]
        self.img_num = len(self.samples)
        print(self.img_num)

    def __getitem__(self, idx):
        path,target = self.samples[idx]
        img = self.loader(path)
        # img = Image.fromarray(img).convert('RGB')
        images = self.transform(img)
        # target = self.targets[idx]
        return images, target, idx


class CustomImageNetLT(Dataset):
    num_classes=1000
    def __init__(self, root, txt, transform=None):
        self.img_path = []
        self.labels = []
        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.labels.append(int(line.split()[1]))
        self.targets = self.labels
        self.class_data=[[] for i in range(self.num_classes)]
        for i in range(len(self.labels)):
            y=self.labels[i]
            self.class_data[y].append(i)

        self.cls_num_list=[len(self.class_data[i]) for i in range(self.num_classes)]
        print(self.cls_num_list)
        self.transform = transform
        print(len(self.labels))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        path1 = self.img_path[index]
        label = self.labels[index]
        with open(path1, 'rb') as f:
            img = Image.open(f).convert('RGB')
        images = self.transform(img)
        # target = self.targets[idx]
        return images, label, index


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

