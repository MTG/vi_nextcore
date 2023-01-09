import numpy as np
import torch
from scipy.spatial.distance import squareform
import json


def MAP(dist_matrix, labels):
    """
        Calculating Mean Average Precision given a symmetric pairwise distance matrix
        and labels of the rows/columns

        Parameters
        ----------
        dist_matrix : numpy.ndarray
            Symmetric distance matrix that contains pairwise distances
        labels : numpy.ndarray
            Labels of all the samples in the dataset

        Returns
        -------
        mAP : float
            Mean average precision obtained from the distance matrix

    """

    # number of samples in the dataset
    num_of_samples = len(labels)

    tuple_dtype = np.dtype([('f1', np.float), ('f2', np.unicode_, 32)])

    # initializing a matrix to store tuples of pairwise distances and labels of the reference samples
    tuple_matrix = np.ndarray(shape=(num_of_samples, num_of_samples), dtype=tuple_dtype)

    # filling the tuple_matrix with distance values and labels
    for i in range(num_of_samples):
        for j in range(num_of_samples):
            tuple_matrix[i][j] = (dist_matrix[i][j], labels[j])

    # initializing mAP
    mAP = 0

    # calculating average precision for each row of the distance matrix
    for i in range(num_of_samples):
        # obtaining the current row
        row = tuple_matrix[i]

        # label of the current query
        label = labels[i]

        # sorting the row with respect to distance values
        row.sort(order='f1')

        # initializing true positive count
        tp = 0

        # initializing precision value
        prec = 0

        # counting number of instances that has the same label as the query
        label_count = 0

        for j in range(1, num_of_samples):
            # checking whether the reference sample has the same label as the query
            if row[j][1] == label:
                # incrementing the number of true positives
                tp += 1

                # updating the precision value
                prec += tp / j

                # incrementing the number of samples with the same label as the query
                label_count += 1

        # updating  mAP
        mAP += prec / label_count

    # updating mAP
    mAP = mAP / num_of_samples

    return mAP


def getEvalStatistics(matrix, cliques='data/cliques.json', topsidx = [1, 10, 100, 1000], matrix_type=1):
    """
    Compute MR, MRR, MAP, Median Rank, and Top X using
    a particular similarity measure
    Parameters
    ----------
    similarity_type: string
        The similarity measure to use
    """
    from itertools import chain
    with open(cliques) as f:
        ci = json.load(f)
    D = matrix.cpu().numpy()
    N = D.shape[0]
    ## Step 1: Re-sort indices of D so that
    ## cover cliques are contiguous
    cliques = [list(ci[s]) for s in ci]
    Ks = np.array([len(c) for c in cliques]) # Length of each clique
    # Sort cliques in descending order of number
    idx = np.argsort(-Ks)
    Ks = Ks[idx]
    cliques = [cliques[i] for i in idx]
    # Unroll array of cliques and put distance matrix in
    # contiguous order
    idx = np.array(list(chain(*cliques)), dtype=int)
    D = D[idx, :]
    D = D[:, idx]

    ## Step 2: Compute MR, MRR, MAP, and Median Rank
    #Fill diagonal with -infinity to exclude song from comparison with self
    np.fill_diagonal(D, -np.inf)
    idx = np.argsort(-D, 1) #Sort row by row in descending order of score
    ranks = np.nan*np.ones(N)
    startidx = 0
    kidx = 0
    allMap = np.nan*np.ones(N)
    for i in range(N):
        if i >= startidx + Ks[kidx]:
            startidx += Ks[kidx]
            kidx += 1
            if Ks[kidx] < 2:
                # We're done once we get to a clique with less than 2
                # since cliques are sorted in descending order
                break
        iranks = []
        for k in range(N):
            diff = idx[i, k] - startidx
            if diff >= 0 and diff < Ks[kidx]:
                iranks.append(k+1)
        iranks = iranks[0:-1] #Exclude the song itself, which comes last
        if len(iranks) == 0:
            print("Recalling 0 songs for clique of size %i at song index %i"%(Ks[kidx], i))
            break
        #For MR, MRR, and MDR, use first song in clique
        ranks[i] = iranks[0]
        #For MAP, use all ranks
        P = np.array([float(j)/float(r) for (j, r) in \
                        zip(range(1, Ks[kidx]), iranks)])
        allMap[i] = np.mean(P)
    MAP = np.nanmean(allMap)
    return MAP


def average_precision(ypred, k=None, eps=1e-10, reduce_mean=True, benchmark=0):
    if benchmark == 0:
        ytrue = 'data/ytrue_red.pt'
    elif benchmark == 1:
        ytrue = 'data/ytrue_benchmark.pt'
    else:
        ytrue = 'data/ytrue_ytc.pt'
        '''
        #ytrue = 'data/ytrue_ytc_reftest.pt'
        import pandas as pd
        i1 = pd.read_csv('ytc_test.txt', header=None, index_col=None).values.flatten().tolist()
        i2 = pd.read_csv('ytc_ref.txt', header=None, index_col=None).values.flatten().tolist()
        i1 = [item - 1 for item in i1]
        i2 = [item - 1 for item in i2]
        ypred = ypred[i1][:, i2]
        '''
    ytrue = torch.load(ytrue).float()
    # ytrue = ytrue[i1][:, i2]
    if k is None:
        k = ypred.size(1)
    _, spred = torch.topk(ypred, k, dim=1)
    found = torch.gather(ytrue, 1, spred)

    temp = torch.arange(k).float() * 1e-6
    _, sel = torch.topk(found - temp, 1, dim=1)
    mrr = torch.mean(1/(sel+1).float())
    mr = torch.mean((sel+1).float())
    top1 = torch.sum(found[:, 0])
    top10 = torch.sum(found[:, :10])

    pos = torch.arange(1, spred.size(1)+1).unsqueeze(0).to(ypred.device)
    prec = torch.cumsum(found, 1)/pos.float()
    mask = (found > 0).float()
    ap = torch.sum(prec*mask, 1)/(torch.sum(ytrue, 1)+eps)
    ap = ap[torch.sum(ytrue, 1) > 0]
    print('mAP: {:.3f}'.format(ap.mean().item()))
    print('MRR: {:.3f}'.format(mrr.item()))
    print('MR: {:.3f}'.format(mr.item()))
    print('Top1: {}'.format(top1.item()))
    print('Top10: {}'.format(top10.item()))
    if reduce_mean:
        return ap.mean()
    return ap


def square_dist_tensor(dist_tensor):
    return torch.from_numpy(squareform(dist_tensor.cpu().detach().numpy()))


def calculate_flatten_size(input_size,
                           conv1_kernel,
                           conv2_kernel,
                           conv3_kernel,
                           pool1_kernel,
                           pool2_kernel,
                           pool3_kernel):
    (i_c, i_h, i_w) = input_size
    (c1_h, c1_w) = conv1_kernel
    if conv2_kernel is not None:
        (c2_h, c2_w) = conv2_kernel
    else:
        (c2_h, c2_w) = (1, 1)
    if conv3_kernel is not None:
        (c3_h, c3_w) = conv3_kernel
    else:
        (c3_h, c3_w) = (1, 1)
    if pool1_kernel is not None:
        (p1_h, p1_w) = pool1_kernel
    else:
        (p1_h, p1_w) = (1, 1)
    if pool2_kernel is not None:
        (p2_h, p2_w) = pool2_kernel
    else:
        (p2_h, p2_w) = (1, 1)
    if pool3_kernel is not None:
        (p3_h, p3_w) = pool3_kernel
    else:
        (p3_h, p3_w) = (1, 1)

    f_h = int(int(int(int(int(int(i_h + 1 - c1_h) / p1_h) + 1 - c2_h) / p2_h) + 1 - c3_h) / p3_h)
    f_w = int(int(int(int(int(int(i_w + 1 - c1_w) / p1_w) + 1 - c2_w) / p2_w) + 1 - c3_w) / p3_w)

    flatten_size = i_c * f_h * f_w

    return flatten_size


def convert_square_to_condensed(idxs1, idxs2, row_size):
    for i in range(idxs1.shape[0]):
        if idxs1[i] < idxs2[i]:
            temp = idxs1[i].clone()
            idxs1[i] = idxs2[i].clone()
            idxs2[i] = temp

    return row_size * idxs2 - idxs2 * (idxs2 + 1) / 2 + idxs1 - 1 - idxs2
