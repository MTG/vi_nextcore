import torch
from torch.utils.data import Dataset
import numpy as np
import deepdish as dd
from utils.dataset_utils import cs_augment
from scipy import interpolate


class ClassificationFixedSize(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, data, labels, h=23, w=1200, masked=False, data_aug = 0):
        self.data = data
        self.labels = np.array(labels)

        self.seed = 42
        self.h = h
        self.w = w
        self.masked = masked
        self.data_aug = data_aug

        unique_labels = np.unique(self.labels)
        self.label_dict = {}
        for i in range(len(unique_labels)):
            self.label_dict[unique_labels[i]] = i

    def __getitem__(self, index):
        if torch.cuda.is_available():
            device = 'cuda:0'
        else:
            device = 'cpu'

        self.seed += 1

        item = self.data[index]
        label = self.labels[index]

        if self.data_aug == 1:
            item = cs_augment(item)
        if item.shape[2] >= self.w:
            p_index = [i for i in range(0, item.shape[2] - self.w + 1)]
            if len(p_index) != 0:
                start = np.random.choice(p_index)
                item = item[:, :, start:start + self.w]
            mask = torch.tensor(0, device=device).float()
        else:
            item = torch.cat((item, torch.zeros([1, self.h, self.w - item.shape[2]])), 2)
            mask = torch.tensor(self.w - item.shape[2], device=device).float()

        if not self.masked:
            mask = torch.tensor(-1, device=device)

        if torch.cuda.is_available():
            item = item.cuda()

        return item, mask, torch.tensor(self.label_dict[label], device=device)

    def __len__(self):
        return len(self.data)
