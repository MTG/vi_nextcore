import torch
from torch.utils.data import Dataset
import numpy as np
import deepdish as dd
from scipy import interpolate


class SingleFixedSize(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, data, labels, idx=None, h=23, w=1200, masked=False, single_files=False, stretch=0):
        if idx is not None:
            self.data = [data[i] for i in idx]
            self.labels = np.array([labels[i] for i in idx])
        else:
            self.data = data
            self.labels = np.array(labels)

        self.seed = 42
        self.h = h
        self.w = w
        self.masked = masked
        self.single_files = single_files
        self.stretch = stretch

        self.labels_set = list(self.labels)
        self.label_to_indices = {label: np.where(self.labels == label)[0]
                                 for label in self.labels_set}

    def __getitem__(self, index):
        if torch.cuda.is_available():
            device = 'cuda:0'
        else:
            device = 'cpu'
        self.seed += 1

        if self.single_files:
            item = self.load_tensor(self.data[index])
        else:
            item = self.data[index]

        if self.stretch == 0:
            if item.shape[2] >= self.w:
                p_index = [i for i in range(0, item.shape[2] - self.w + 1)]
                if len(p_index) != 0:
                    start = np.random.choice(p_index)
                    item_f = item[:, :, start:start+self.w]
                mask = torch.tensor(0, device=device).float()
            else:
                item_f = torch.cat((item, torch.zeros([1, self.h, self.w - item.shape[2]])), 2)
                mask = torch.tensor(self.w - item.shape[2], device=device).float()
        else:
            func = interpolate.interp1d(np.arange(item.shape[2]), item.numpy(), kind='nearest',
                                        fill_value='extrapolate')
            item_f = torch.from_numpy(func(np.linspace(0, item.shape[2] - 1, self.w)))
            mask = torch.tensor(0, device=device).float()

        if torch.cuda.is_available():
            item_f = item_f.cuda()

        if not self.masked:
            mask = torch.tensor(-1, device=device)

        return item_f.float(), mask

    def __len__(self):
        return len(self.data)

    @staticmethod
    def load_tensor(path):
        item_ = dd.io.load(path)

        (r, c) = np.array(item_['crema']).shape

        item = torch.reshape(torch.cat((torch.Tensor(item_['crema']),
                                        torch.Tensor(item_['crema'][:11])), 0), (1, 23, c))

        return item