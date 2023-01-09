import torch
from torch.utils.data import Dataset
import numpy as np
import deepdish as dd


class SingleFullSize(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, data, labels, idx=None, h=23, w=1200, single_files=0):
        if idx is not None:
            self.data = [data[i] for i in idx]
            self.labels = np.array([labels[i] for i in idx])
        else:
            self.data = data
            self.labels = np.array(labels)

        self.h = h
        self.w = w
        self.single_files = single_files

        self.labels_set = list(self.labels)
        self.label_to_indices = {label: np.where(self.labels == label)[0]
                                 for label in self.labels_set}

    def __getitem__(self, index):
        if torch.cuda.is_available():
            device = 'cuda:0'
        else:
            device = 'cpu'

        if self.single_files == 0:
            item1 = self.data[index]
        else:
            item1 = self.load_tensor(self.data[index])

        if torch.cuda.is_available():
            item1 = item1.cuda()

        return item1.float(), torch.tensor(-1, device=device)

    def __len__(self):
        return len(self.data)

    def load_tensor(self, path):
        item_ = dd.io.load(path)

        (r, c) = np.array(item_['crema'].T).shape

        item = torch.reshape(torch.cat((torch.Tensor(item_['crema'].T),
                                        torch.Tensor(item_['crema'].T[:11])), 0), (1, 23, c))

        return item
