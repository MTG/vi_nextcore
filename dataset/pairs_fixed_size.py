import torch
from torch.utils.data import Dataset
import numpy as np
import deepdish as dd
from scipy import interpolate


class PairsFixedSize(Dataset):
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
            self.device = 'cuda:0'
        else:
            self.device = 'cpu'

        self.seed += 1

        if self.single_files:
            item1 = self.load_tensor(self.data[index])
        else:
            item1 = self.data[index]
        label1 = self.labels[index]

        item1_f, mask1 = self.handle_dim(item1)

        if torch.cuda.is_available():
            item1_f = item1_f.cuda()

        # selecting genuine pair
        np.random.seed(self.seed)

        pair_index = index
        while pair_index == index:
            pair_index = np.random.choice(self.label_to_indices[label1])

        if self.single_files:
            item2 = self.load_tensor(self.data[pair_index])
        else:
            item2 = self.data[pair_index]

        item2_f, mask2 = self.handle_dim(item2)

        if torch.cuda.is_available():
            item2_f = item2_f.cuda()

        # selecting impostor pair

        pair_label = np.random.choice([item for item in self.labels_set if item not in list([label1])])
        pair_index = np.random.choice(self.label_to_indices[pair_label])

        if self.single_files:
            item3 = self.load_tensor(self.data[pair_index])
        else:
            item3 = self.data[pair_index]

        item3_f, mask3 = self.handle_dim(item3)

        if torch.cuda.is_available():
            item3_f = item3_f.cuda()

        if not self.masked:
            mask1 = torch.tensor(-1, device=self.device)
            mask2 = torch.tensor(-1, device=self.device)
            mask3 = torch.tensor(-1, device=self.device)

        return (item1_f.float(), item2_f.float(), item3_f.float()), (mask1, mask2, mask3), \
               (torch.tensor(0, device=self.device).float(), torch.tensor(1, device=self.device).float())

    def __len__(self):
        return len(self.data)

    def handle_dim(self, item):
        if self.stretch == 0:
            if item.shape[2] >= self.w:
                p_index = [i for i in range(0, item.shape[2] - self.w + 1)]
                if len(p_index) != 0:
                    start = np.random.choice(p_index)
                    item_f = item[:, :, start:start + self.w]
                mask = torch.tensor(0, device=self.device).float()
            else:
                item_f = torch.cat((item, torch.zeros([1, self.h, self.w - item.shape[2]])), 2)
                mask = torch.tensor(self.w - item.shape[2], device=self.device).float()
        else:
            func = interpolate.interp1d(np.arange(item.shape[2]), item.numpy(), kind='nearest', fill_value='extrapolate')
            item_f = torch.from_numpy(func(np.linspace(0, item.shape[2]-1, self.w)))
            mask = torch.tensor(0, device=self.device).float()

        return item_f, mask


    @staticmethod
    def load_tensor(path):
        item_ = dd.io.load(path)

        (r, c) = np.array(item_['crema']).shape

        item = torch.reshape(torch.cat((torch.Tensor(item_['crema']),
                                        torch.Tensor(item_['crema'][:11])), 0), (1, 23, c))

        return item
