import torch
from torch.utils.data import Dataset
import numpy as np
import deepdish as dd
from utils.dataset_utils import cs_augment
from scipy import interpolate


class TripletMiningFixedSize(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, data, labels, idx=None, h=23, w=1200, masked=0, single_files=False, stretch=0, i_per_c=4,
                 data_aug=0, mul_len=0, rand_length=0, uni_dist=0):
        if idx is not None:
            self.data = [data[i] for i in idx]
            self.labels = np.array([labels[i] for i in idx])
        else:
            self.data = data
            self.labels = np.array(labels)

        self.seed = 42
        self.h = h
        self.w = w
        if masked == 0:
            self.masked = False
        else:
            self.masked = True
        self.single_files = single_files
        self.stretch = stretch
        self.i_per_c = i_per_c
        self.data_aug = data_aug
        self.mul_len = mul_len
        self.rand_length = rand_length

        self.labels_set = list(self.labels)
        self.label_to_indices = {label: np.where(self.labels == label)[0]
                                 for label in self.labels_set}

        self.clique_list = []
        self.clique_weights = []
        for label in self.label_to_indices.keys():
            if uni_dist == 0:
                self.clique_list.append(label)
            else:
                if self.label_to_indices[label].size < 6:
                    self.clique_list.extend([label] * 1)
                elif self.label_to_indices[label].size < 10:
                    self.clique_list.extend([label] * 2)
                elif self.label_to_indices[label].size < 14:
                    self.clique_list.extend([label] * 3)
                else:
                    self.clique_list.extend([label] * 4)

    def __getitem__(self, index):
        if torch.cuda.is_available():
            device = 'cuda:0'
        else:
            device = 'cpu'

        self.seed += 1

        label = self.clique_list[index]

        if self.i_per_c == 4:
            # TODO: if both are shorter than 2 mins, use data augmentation
            if self.label_to_indices[label].size == 2:
                idx1, idx2 = np.random.choice(self.label_to_indices[label], 2, replace=False)
                # TODO: implement load single files
                item1, item2 = self.data[idx1], self.data[idx2]
                item3, item4 = self.data[idx1], self.data[idx2]
            elif self.label_to_indices[label].size == 3:
                idx1, idx2, idx3 = np.random.choice(self.label_to_indices[label], 3, replace=False)
                idx4 = np.random.choice(self.label_to_indices[label], 1, replace=False)[0]
                item1, item2, item3, item4 = self.data[idx1], self.data[idx2], self.data[idx3], self.data[idx4]
            else:
                idx1, idx2, idx3, idx4 = np.random.choice(self.label_to_indices[label], 4, replace=False)
                item1, item2, item3, item4 = self.data[idx1], self.data[idx2], self.data[idx3], self.data[idx4]
            if self.data_aug == 2:
                items_i = [item1, item2, item3, item4,
                           cs_augment(item1), cs_augment(item2), cs_augment(item3), cs_augment(item4)]
            else:
                items_i = [item1, item2, item3, item4]
        else:
            idx1, idx2 = np.random.choice(self.label_to_indices[label], 2, replace=False)
            item1, item2 = self.data[idx1], self.data[idx2]
            items_i = [item1, item2]

        items = {}
        masks = {}
        if self.mul_len == 0:
            lengths = [self.w]
        else:
            lengths = [1200, 1800, 2400]

        for length in lengths:
            items[length] = []
            masks[length] = []
            for item in items_i:
                if self.rand_length == 1:
                    temp_length = torch.randint(low=1000, high=self.w, size=[1]).item()
                else:
                    temp_length = length
                if self.data_aug == 1:
                    item = cs_augment(item)
                if self.stretch == 0:
                    if item.shape[2] >= temp_length:
                        p_index = [i for i in range(0, item.shape[2] - temp_length + 1)]
                        if len(p_index) != 0:
                            start = np.random.choice(p_index)
                            temp_item = item[:, :, start:start + temp_length]
                            if self.rand_length == 1:
                                items[length].append(torch.cat((temp_item, torch.zeros([1, self.h, length - temp_item.shape[2]])), 2))
                                masks[length].append(torch.tensor(length - temp_item.shape[2]).float())
                            else:
                                items[length].append(temp_item)
                                masks[length].append(torch.tensor(0).float())
                    else:
                        items[length].append(torch.cat((item, torch.zeros([1, self.h, length - item.shape[2]])), 2))
                        masks[length].append(torch.tensor(length - item.shape[2]).float())
                else:
                    func = interpolate.interp1d(np.arange(item.shape[2]), item.numpy(), kind='nearest',
                                                fill_value='extrapolate')
                    items[length].append(torch.from_numpy(func(np.linspace(0, item.shape[2]-1, length))))
                    masks[length].append(torch.tensor(0).float())

        if not self.masked:
            for length in lengths:
                for i in range(len(masks[length])):
                    masks[length][i] = torch.tensor(-1)
        """
        if torch.cuda.is_available():
            for length in lengths:
                for i in range(len(items[length])):
                    items[length][i] = items[length][i].cuda()
        """

        if self.data_aug == 2:
            return torch.stack(items[:4], 0), torch.stack(items[4:])
        else:
            if self.mul_len == 0:
                return torch.stack(items[lengths[0]], 0), torch.stack(masks[lengths[0]], 0), label
            else:
                return torch.stack(items[lengths[0]], 0), \
                       torch.stack(items[lengths[1]], 0), \
                       torch.stack(items[lengths[2]], 0), \
                       torch.stack(masks[lengths[0]], 0), \
                       torch.stack(masks[lengths[1]], 0), \
                       torch.stack(masks[lengths[2]], 0)

    def __len__(self):
        return len(self.clique_list)

    @staticmethod
    def load_tensor(path):
        item_ = dd.io.load(path)

        (r, c) = np.array(item_['crema']).shape

        item = torch.reshape(torch.cat((torch.Tensor(item_['crema']),
                                        torch.Tensor(item_['crema'][:11])), 0), (1, 23, c))

        return item
