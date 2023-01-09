import json

import torch
import numpy as np
from scipy import interpolate
from torch.nn.utils.rnn import pad_sequence
import deepdish as dd


def import_dataset_from_json(filename):
    file = open(filename)
    dataset_dict = json.load(file)
    file.close()

    data = []

    for key in dataset_dict.keys():
        (r, c) = np.array(dataset_dict[key]['hpcp']).shape
        data.append(torch.reshape(torch.cat((torch.Tensor(dataset_dict[key]['hpcp']),
                                  torch.Tensor(dataset_dict[key]['hpcp'][:11])), 0), (1, 23, c)))

    labels = [dataset_dict[key]['label'] for key in dataset_dict.keys()]

    return data, labels


def import_dataset_from_pt(filename, chunks=1):
    if chunks > 1:
        for i in range(1, chunks+1):
            dataset_dict = torch.load('{}_{}.pt'.format(filename, i))
            if i == 1:
                data = dataset_dict['data']
                labels = dataset_dict['labels']
            else:
                data.extend(dataset_dict['data'])
                labels.extend(dataset_dict['labels'])
    else:
        dataset_dict = torch.load('{}.pt'.format(filename))
        data = dataset_dict['data']
        labels = dataset_dict['labels']

    return data, labels


def import_dataset_from_h5(filename, chunks=1, cocat=0, benchmark=0):
    data = []
    labels = []
    if chunks > 1:
        for i in range(1, chunks+1):
            dataset_dict = dd.io.load('{}_{}.h5'.format(filename, i))

            for key in dataset_dict.keys():
                (r, c) = np.array(dataset_dict[key]['hpcp']).shape
                if benchmark == 0:
                    if c >= 1000 and dataset_dict[key]['label'] != 'W_11527':
                        if cocat != 7:
                            data.append(torch.reshape(torch.cat((torch.Tensor(dataset_dict[key]['hpcp']),
                                                      torch.Tensor(dataset_dict[key]['hpcp'][:11])), 0), (1, 23, c)))
                        else:
                            data.append(torch.reshape(torch.Tensor(dataset_dict[key]['hpcp']), (1, 12, c)))
                        labels.append(dataset_dict[key]['label'])
                else:
                    if cocat != 7:
                        data.append(torch.reshape(torch.cat((torch.Tensor(dataset_dict[key]['hpcp']),
                                                             torch.Tensor(dataset_dict[key]['hpcp'][:11])), 0),
                                                  (1, 23, c)))
                    else:
                        data.append(torch.reshape(torch.Tensor(dataset_dict[key]['hpcp']), (1, 12, c)))
                    labels.append(dataset_dict[key]['label'])
    else:
        dataset_dict = dd.io.load(filename)

        for key in dataset_dict.keys():
            (r, c) = np.array(dataset_dict[key]['hpcp']).shape
            if benchmark == 0:
                if c >= 1000 and dataset_dict[key]['label'] != 'W_11527':
                    if cocat != 7:
                        data.append(torch.reshape(torch.cat((torch.Tensor(dataset_dict[key]['hpcp']),
                                                             torch.Tensor(dataset_dict[key]['hpcp'][:11])), 0), (1, 23, c)))
                    else:
                        data.append(torch.reshape(torch.Tensor(dataset_dict[key]['hpcp']), (1, 12, c)))
                    labels.append(dataset_dict[key]['label'])
            else:
                if cocat != 7:
                    data.append(torch.reshape(torch.cat((torch.Tensor(dataset_dict[key]['hpcp']),
                                                         torch.Tensor(dataset_dict[key]['hpcp'][:11])), 0), (1, 23, c)))
                else:
                    data.append(torch.reshape(torch.Tensor(dataset_dict[key]['hpcp']), (1, 12, c)))
                labels.append(dataset_dict[key]['label'])

    return data, labels


def import_large_dataset_from_json(filename):
    file = open(filename)
    dataset_dict = json.load(file)
    file.close()

    data = []

    for key in dataset_dict.keys():
        data.append(dataset_dict[key]['path'])

    labels = [dataset_dict[key]['label'] for key in dataset_dict.keys()]

    return data, labels


def cs_augment(chroma, p_pitch=1, p_stretch=0.3, p_warp=0.3):
    chroma = chroma.cpu().detach().numpy()
    # pitch transposition
    if torch.rand(1) <= p_pitch:
        shift_amount = 0
        while shift_amount == 0:
            shift_amount = torch.randint(low=1, high=12, size=(1,))
        chroma_aug = np.roll(chroma, shift_amount, axis=1)
    else:
        chroma_aug = chroma

    _, h, w = chroma_aug.shape
    times = np.arange(0, w)

    func = interpolate.interp1d(times, chroma_aug, kind='nearest', fill_value='extrapolate')

    # time stretch
    if torch.rand(1) < p_stretch:
        p = torch.rand(1)
        if p <= 0.5:
            #times_aug = np.arange(0, w, (1 - p))
            times_aug = np.linspace(0, w - 1, w * int(torch.clamp((1 - p), min=0.7, max=1)[0].numpy()))
        else:
            #times_aug = np.arange(0, w, (2 * p))
            times_aug = np.linspace(0, w - 1, w * int(torch.clamp(2 * p, min=1, max=1.5)[0].numpy()))
        chroma_aug = func(times_aug)
    else:
        times_aug = times
        chroma_aug = func(times_aug)

    # time warping
    if torch.rand(1) < p_warp:
        p = torch.rand(1)

        if p < 0.3:  # silence
            silence_idxs = np.random.choice([False, True], size=times_aug.size, p=[.9, .1])
            chroma_aug[:, :, silence_idxs] = np.zeros((h, 1))

        elif p < 0.7:  # duplicate
            duplicate_idxs = np.random.choice([False, True], size=times_aug.size, p=[.85, .15])
            times_aug = np.sort(np.concatenate((times_aug, times_aug[duplicate_idxs])))
            chroma_aug = func(times_aug)

        else:  # remove
            remaining_idxs = np.random.choice([False, True], size=times_aug.size, p=[.1, .9])
            times_aug = times_aug[remaining_idxs]
            chroma_aug = func(times_aug)

    return torch.from_numpy(chroma_aug)

"""
def cs_augment_torch(chroma, p_pitch=1, p_stretch=0.3, p_warp=0.3):
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    # pitch transposition
    if torch.rand(1) <= p_pitch:
        shift_amount = 0
        while shift_amount == 0:
            shift_amount = torch.randint(low=1, high=12, size=(1,), device=device)
        chroma_aug = torch.cat((chroma[:, -shift_amount:], chroma[:, :-shift_amount]), dim=1)
    else:
        chroma_aug = chroma

    w = chroma_aug.size(2)
    times = torch.arange(0, w, device=device)

    #func = interpolate.interp1d(times, chroma_aug, kind='nearest', fill_value='extrapolate')

    # time stretch
    if torch.rand(1) < p_stretch:
        p = torch.rand(1)
        if p <= 0.5:
            #times_aug = np.arange(0, w, (1 - p))
            times_aug = torch.linspace(0, w - 1, w * torch.clamp((1 - p), min=0.7, max=1))
        else:
            #times_aug = np.arange(0, w, (2 * p))
            times_aug = torch.linspace(0, w - 1, w * torch.clamp(2 * p, min=1, max=1.5))
        chroma_aug = func(times_aug)
    else:
        times_aug = times
        chroma_aug = func(times_aug)

    # time warping
    if torch.rand(1) < p_warp:
        p = torch.rand(1)

        if p < 0.3:  # silence
            silence_idxs = np.random.choice([False, True], size=times_aug.size, p=[.9, .1])
            chroma_aug[:, :, silence_idxs] = np.zeros((h, 1))

        elif p < 0.7:  # duplicate
            duplicate_idxs = np.random.choice([False, True], size=times_aug.size, p=[.85, .15])
            times_aug = np.sort(np.concatenate((times_aug, times_aug[duplicate_idxs])))
            chroma_aug = func(times_aug)

        else:  # remove
            remaining_idxs = np.random.choice([False, True], size=times_aug.size, p=[.1, .9])
            times_aug = times_aug[remaining_idxs]
            chroma_aug = func(times_aug)

    return torch.from_numpy(chroma_aug)
"""

def custom_collate(batch):
    item1 = pad_sequence([item[0][0].permute(2, 0, 1) for item in batch], batch_first=True).permute(0, 2, 3, 1)
    item2 = pad_sequence([item[0][1].permute(2, 0, 1) for item in batch], batch_first=True).permute(0, 2, 3, 1)
    clabel = torch.Tensor([item[1] for item in batch])

    return (item1, item2), clabel


def triplet_mining_collate(batch):
    items = [item[0] for item in batch]
    masks = [item[1] for item in batch]
    labels = [item[2] for item in batch]

    return torch.cat(items, 0), torch.cat(masks, 0), labels

def mul_len_triplet_collate(batch):
    items1 = [item[0] for item in batch]
    items2 = [item[1] for item in batch]
    items3 = [item[2] for item in batch]
    masks1 = [item[3] for item in batch]
    masks2 = [item[4] for item in batch]
    masks3 = [item[5] for item in batch]

    return torch.cat(items1, 0), torch.cat(items2, 0), torch.cat(items3, 0), \
           torch.cat(masks1, 0), torch.cat(masks2, 0), torch.cat(masks3, 0)


def mix_collate(batch):
    items = [item[0] for item in batch]
    masks = [item[1] for item in batch]
    labels = [item[2] for item in batch]

    return torch.cat(items, 0), torch.cat(masks, 0), torch.cat(labels, 0)
