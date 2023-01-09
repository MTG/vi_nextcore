import torch
from torch.utils.data import Dataset
import numpy as np


class PairsFullSize(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, data, labels, idx=None):
        if idx is not None:
            self.data = [data[i] for i in idx]
            self.labels = np.array([labels[i] for i in idx])
        else:
            self.data = data
            self.labels = np.array(labels)

        self.seed = 42

        self.labels_set = list(self.labels)
        self.label_to_indices = {label: np.where(self.labels == label)[0]
                                 for label in self.labels_set}

    def __getitem__(self, index):
        if torch.cuda.is_available():
            device = 'cuda:0'
        else:
            device = 'cpu'

        self.seed += 1

        item1, label1 = self.data[index], self.labels[index]

        if torch.cuda.is_available():
            item1 = item1.cuda()

        # selecting genuine pair
        np.random.seed(self.seed)

        pair_index = index
        while pair_index == index:
            pair_index = np.random.choice(self.label_to_indices[label1])

        item2 = self.data[pair_index]

        if torch.cuda.is_available():
            item2 = item2.cuda()

        # selecting impostor pair

        pair_label = np.random.choice([item for item in self.labels_set if item not in list([label1])])
        pair_index = np.random.choice(self.label_to_indices[pair_label])

        item3 = self.data[pair_index]

        if torch.cuda.is_available():
            item3 = item3.cuda()

        return (item1.float(), item2.float(), item3.float()), (torch.tensor(0, device=device).float(),
                                                               torch.tensor(1, device=device).float())

    def __len__(self):
        return len(self.data)
