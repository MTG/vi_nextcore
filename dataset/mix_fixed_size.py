import torch
from torch.utils.data import Dataset
import numpy as np
import deepdish as dd
from utils.dataset_utils import cs_augment
from scipy import interpolate


class MixFixedSize(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, data, labels, idx=None, h=23, w=1200, masked=False, single_files=False, stretch=0, i_per_c=4,
                 data_aug = 0):
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
        self.i_per_c = i_per_c
        self.data_aug = data_aug

        self.labels_set = list(self.labels)
        self.label_to_indices = {label: np.where(self.labels == label)[0]
                                 for label in self.labels_set}

        self.clique_list = []
        self.clique_weights = []
        for label in self.label_to_indices.keys():
            self.clique_list.append(label)
            if self.label_to_indices[label].size < 6:
                self.clique_weights.append(1)
            elif self.label_to_indices[label].size < 10:
                self.clique_weights.append(2)
            else:
                self.clique_weights.append(3)

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

        label = self.clique_list[index]

        if self.i_per_c == 4:
            # TODO: if both are shorter than 2 mins, use data augmentation
            if self.label_to_indices[label].size == 2:
                idx1, idx2 = np.random.choice(self.label_to_indices[label], 2, replace=False)
                # TODO: implement load single files
                item1, item2 = self.data[idx1], self.data[idx2]
                if item1.shape[2] <= 1000:
                    item3 = cs_augment(item2.clone(), p_pitch=1.1, p_stretch=0, p_warp=0)
                    item4 = cs_augment(item2.clone(), p_pitch=1.1, p_stretch=0, p_warp=0)
                elif item2.shape[2] <= 1000:
                    item3 = cs_augment(item1.clone(), p_pitch=1.1, p_stretch=0, p_warp=0)
                    item4 = cs_augment(item1.clone(), p_pitch=1.1, p_stretch=0, p_warp=0)
                else:
                    item3 = cs_augment(item1.clone(), p_pitch=1.1, p_stretch=0, p_warp=0)
                    item4 = cs_augment(item2.clone(), p_pitch=1.1, p_stretch=0, p_warp=0)
            elif self.label_to_indices[label].size == 3:
                idx1, idx2, idx3 = np.random.choice(self.label_to_indices[label], 3, replace=False)
                item1, item2, item3 = self.data[idx1], self.data[idx2], self.data[idx3]
                if item1.shape[2] <= 1000:
                    if item2.shape[2] <= 1000:
                        item4 = cs_augment(item3.clone(), p_pitch=1.1, p_stretch=0, p_warp=0)
                    else:
                        item4 = cs_augment(item2.clone(), p_pitch=1.1, p_stretch=0, p_warp=0)
                else:
                    item4 = cs_augment(item1.clone(), p_pitch=1.1, p_stretch=0, p_warp=0)
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

        items = []
        masks = []

        for item in items_i:
            if self.data_aug == 1:
                item = cs_augment(item)
            if self.stretch == 0:
                if item.shape[2] >= self.w:
                    p_index = [i for i in range(0, item.shape[2] - self.w + 1)]
                    if len(p_index) != 0:
                        start = np.random.choice(p_index)
                        items.append(item[:, :, start:start + self.w])
                    masks.append(torch.tensor(0, device=device).float())
                else:
                    items.append(torch.cat((item, torch.zeros([1, self.h, self.w - item.shape[2]])), 2))
                    masks.append(torch.tensor(self.w - item.shape[2], device=device).float())
            else:
                func = interpolate.interp1d(np.arange(item.shape[2]), item.numpy(), kind='nearest',
                                            fill_value='extrapolate')
                items.append(torch.from_numpy(func(np.linspace(0, item.shape[2]-1, self.w))))
                masks.append(torch.tensor(0, device=device).float())

        if not self.masked:
            for i in range(len(masks)):
                masks[i] = torch.tensor(-1, device=device)

        labels = []
        for _ in range(len(items)):
            labels.append(torch.tensor(self.label_dict[label], device=device))

        if torch.cuda.is_available():
            for i in range(len(items)):
                items[i] = items[i].cuda()

        if self.data_aug == 2:
            return torch.stack(items[:4], 0), torch.stack(items[4:])
        else:
            return torch.stack(items, 0), torch.stack(masks, 0), torch.stack(labels, 0)

    def __len__(self):
        return len(self.clique_list)

    @staticmethod
    def load_tensor(path):
        item_ = dd.io.load(path)

        (r, c) = np.array(item_['crema']).shape

        item = torch.reshape(torch.cat((torch.Tensor(item_['crema']),
                                        torch.Tensor(item_['crema'][:11])), 0), (1, 23, c))

        return item
