import torch
import deepdish as dd
import numpy as np


def main(filename, chunks=1, benchmark=0, cocat=0):
    data = []
    labels = []
    new_dict = {}

    dataset_dict = dd.io.load('{}'.format(filename))

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

    new_dict['data'] = data
    new_dict['labels'] = labels

    torch.save(new_dict, '{}.pt'.format(filename.split('.')[-2]))


if __name__:
    main('data/tr_subset_crema_1.h5', benchmark=0)
    main('data/tr_subset_crema_2.h5', benchmark=0)
    main('data/tr_subset_crema_3.h5', benchmark=0)
    main('data/tr_subset_crema_4.h5', benchmark=0)
    main('data/val_subset_crema.h5', benchmark=0)
    main('data/benchmark_crema.h5', benchmark=1)
