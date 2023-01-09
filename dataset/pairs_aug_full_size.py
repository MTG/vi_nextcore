import torch
from torch.utils.data import Dataset
import numpy as np

from utils.dataset_utils import cs_augment


class PairsAugFullSize(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, data, labels, idx=None, h=23, w=1200, noise=True):
        if idx is not None:
            self.data = [data[i] for i in idx]
            self.labels = np.array([labels[i] for i in idx])
        else:
            self.data = data
            self.labels = np.array(labels)

        self.seed = 42
        self.h = h
        self.w = w
        self.noise = noise

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

        #if torch.cuda.is_available():
        #    item1 = item1.cuda()

        # selecting genuine pair
        np.random.seed(self.seed)

        aug_w = torch.randint(low=150, high=200, size=(1,))

        p_index = [i for i in range(0, item1.shape[2] - aug_w + 1)]
        start = np.random.choice(p_index)
        chroma = item1[:, :, start:start + aug_w].cpu().numpy()
        aug_chroma = cs_augment(chroma)

        if self.noise:
            item2 = torch.cat((aug_chroma, torch.rand([1, self.h, 600])), 2)
            item2 = torch.cat((torch.rand([1, self.h, 600]), item2), 2)
        else:
            item2 = torch.cat((aug_chroma, torch.zeros([1, self.h, 600])), 2)
            item2 = torch.cat((torch.zeros([1, self.h, 600]), item2), 2)

        #if torch.cuda.is_available():
        #    item2 = item2.cuda()

        return (item1.float(), item2.float()), (torch.tensor(0, device=device).float())

    def __len__(self):
        return len(self.data)
