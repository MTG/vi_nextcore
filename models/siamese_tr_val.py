import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import pdist as scipdist

from utils.siamese_utils import square_dist_tensor
from utils.siamese_utils import convert_square_to_condensed


def contrastive_loss(res_1, res_2, clabel, margin=1):
    dist = torch.pow(F.pairwise_distance(res_1, res_2), 2)
    return torch.mean((1 - clabel) * dist + clabel * F.relu(margin - dist))


def contrastive_loss_gamma(res_1, res_2, clabel, model, margin=1):
    dist = torch.pow(F.pairwise_distance(res_1, res_2), 2)
    loss = torch.mean((1 - clabel) * dist + clabel * F.relu(margin - dist * torch.sigmoid(model.gamma)))
    genuine = torch.mean(((1 - clabel) * dist)[:int(dist.size()[0] / 2)])
    impostor = torch.mean((clabel * F.relu(margin - dist * torch.sigmoid(model.gamma)))[int(dist.size()[0] / 2):])
    return loss, genuine, impostor


def triplet_loss(res_1, res_2, res_3, size_average=True, margin=1, loss_t=0):
    if loss_t == 0:
        dist_g = torch.pow(F.pairwise_distance(res_1, res_2), 2)
        dist_i = torch.pow(F.pairwise_distance(res_1, res_3), 2)
    else:
        dist_g = 1 - F.cosine_similarity(res_1, res_2)
        dist_i = 1 - F.cosine_similarity(res_1, res_3)
    loss = F.relu(dist_g + (margin - dist_i))
    genuine = dist_g.mean() if size_average else dist_g.sum()
    impostor = (margin - dist_i).mean() \
        if size_average else (margin - dist_i).sum()
    return loss.mean() if size_average else loss.sum(), genuine, impostor


def triplet_loss_gamma(res_1, res_2, res_3, model, size_average=True, margin=1):
    dist_g = torch.pow(F.pairwise_distance(res_1, res_2), 2)
    dist_i = torch.pow(F.pairwise_distance(res_1, res_3), 2)
    loss = F.relu(dist_g + (margin - dist_i * torch.sigmoid(model.gamma)))
    genuine = dist_g.mean() if size_average else dist_g.sum()
    impostor = (margin - dist_i * torch.sigmoid(model.gamma)).mean() \
        if size_average else (margin - dist_i * torch.sigmoid(model.gamma)).sum()
    return loss.mean() if size_average else loss.sum(), genuine, impostor


def triplet_loss_gamma_alpha(res_1, res_2, res_3, model, size_average=True):
    dist_g = torch.pow(F.pairwise_distance(res_1, res_2), 2)
    dist_i = torch.pow(F.pairwise_distance(res_1, res_3), 2)
    margin = torch.exp(model.alpha) + 0.5
    loss = F.relu(dist_g + (margin - dist_i * torch.sigmoid(model.gamma)))
    genuine = dist_g.mean() if size_average else dist_g.sum()
    impostor = (margin - dist_i * torch.sigmoid(model.gamma)).mean() \
        if size_average else (margin - dist_i * torch.sigmoid(model.gamma)).sum()
    return loss.mean() if size_average else loss.sum(), genuine, impostor


def lossless_triplet_gamma(res_1, res_2, res_3, model, margin=256):
    dist_g = torch.pow(res_1 - res_2, 2).sum(1)
    dist_i = torch.pow(res_1 - res_3, 2).sum(1)
    log_dist_g = -torch.log(-torch.div(dist_g, 256) + 1 + 1e-8)
    log_dist_i = -torch.log(-torch.div((margin - dist_i * torch.sigmoid(model.gamma)), 256) + 1 + 1e-8)
    loss = log_dist_g + log_dist_i
    return loss.mean(), log_dist_g.mean(), log_dist_i.mean()


def pairwise_distance_matrix(x, y=None, eps=1e-12):
    x_norm = x.pow(2).sum(1).view(-1, 1)
    if y is not None:
        y_norm = y.pow(2).sum(1).view(1, -1)
    else:
        y = x
        y_norm = x_norm.view(1, -1)

    dist = x_norm+y_norm-2*torch.mm(x, y.t().contiguous())
    return torch.clamp(dist, eps, np.inf)


def triplet_loss_mining(res_1, siamese_model, size_average=True, margin=1.5, batch_hard=0, loss_t=0, allpos=1, norm_dist=0, labels=None):
    if labels is not None:
        aux = {}
        ilabels = []
        for l in labels:
            if l not in aux:
                aux[l] = len(aux)
            ilabels+=[aux[l]]*4

        ilabels = torch.Tensor(ilabels).view(-1, 1)
        mask_diag = (1 - torch.eye(res_1.size(0))).long()
        if torch.cuda.is_available():
            ilabels = ilabels.cuda()
            mask_diag = mask_diag.cuda()
        temp_mask = (pairwise_distance_matrix(ilabels) < 0.5).long()
        mask_pos = mask_diag * temp_mask
        mask_neg = mask_diag * (1 - mask_pos)
    if loss_t == 0:
        #dist_all = torch.pow(torch.pdist(res_1), 2)
        dist_all = pairwise_distance_matrix(res_1)
        if norm_dist == 1:
            dist_all /= siamese_model.fin_emb_size
    else:
        dist_all = torch.Tensor([1 - F.cosine_similarity(res_1[i], res_1[j], dim=0) for i in range(res_1.size(0)) for j in range(i+1, res_1.size(0))])
        dist_all.requires_grad = True

    if batch_hard == 0:
        dist_g, dist_i = triplet_mining_semihard(dist_all, mask_pos, mask_neg)
    elif batch_hard == 1:
        dist_g, dist_i = triplet_mining_hard(dist_all, mask_pos, mask_neg)
    else:
        dist_g, dist_i = triplet_mining_random(dist_all, mask_pos, mask_neg)
    loss = F.relu(dist_g + (margin - dist_i))
    genuine = dist_g.mean() if size_average else dist_g.sum()
    impostor = (margin - dist_i).mean() \
        if size_average else (margin - dist_i).sum()
    return loss.mean() if size_average else loss.sum(), genuine, impostor


def triplet_loss_mining_aug(res_1, res_2, size_average=True, margin=1, batch_hard=0):
    dist_all = torch.pow(torch.pdist(res_1), 2)
    if batch_hard == 0:
        g_idx, i_idx = triplet_mining_random(res_1.shape[0])
    else:
        g_idx, i_idx = triplet_mining_hard(dist_all, res_1.shape[0])
    dist_g, dist_i = dist_all[g_idx], dist_all[i_idx]
    loss = F.relu(dist_g + (margin - dist_i)) + torch.pow(F.pairwise_distance(res_1, res_2), 2)
    genuine = dist_g.mean() if size_average else dist_g.sum()
    impostor = (margin - dist_i).mean() \
        if size_average else (margin - dist_i).sum()
    return loss.mean() if size_average else loss.sum(), genuine, impostor


def triplet_mining_hard(dist_all, mask_pos, mask_neg):
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    _, sel_pos = torch.max(dist_all * mask_pos.float(), 1)
    dists_pos = torch.gather(dist_all, 1, sel_pos.view(-1, 1))
    mask_neg = torch.where(mask_neg == 0, torch.tensor(float('inf'), device=device), torch.tensor(1., device=device))
    _, sel_neg = torch.min(dist_all + mask_neg.float(), 1)
    dists_neg = torch.gather(dist_all, 1, sel_neg.view(-1, 1))

    return dists_pos, dists_neg


def triplet_mining_random(dist_all, mask_pos, mask_neg):
    _, sel_pos = torch.max(mask_pos.float() + torch.rand_like(dist_all), 1)
    dists_pos = torch.gather(dist_all, 1, sel_pos.view(-1, 1))
    _, sel_neg = torch.max(mask_neg.float() + torch.rand_like(dist_all), 1)
    dists_neg = torch.gather(dist_all, 1, sel_neg.view(-1, 1))

    return dists_pos, dists_neg


def triplet_mining_semihard(dist_all, mask_pos, mask_neg):
    _, sel_pos = torch.max(mask_pos.float() + torch.rand_like(dist_all), 1)
    dists_pos = torch.gather(dist_all, 1, sel_pos.view(-1, 1))
    _, sel_neg = torch.max((mask_neg + mask_neg * (dist_all < dists_pos.expand_as(dist_all)).long()).float() + torch.rand_like(dist_all), 1)
    dists_neg = torch.gather(dist_all, 1, sel_neg.view(-1, 1))

    return dists_pos, dists_neg


def train(siamese_model, optimizer, train_loader, margin):
    siamese_model.train()
    loss_log = []
    genuine_loss_log = []
    impostor_loss_log = []

    for batch_idx, ((item1, item2, item3), (clabel1, clabel2)) in enumerate(train_loader):

        batch_1 = torch.cat((item1, item1), 0)
        batch_2 = torch.cat((item2, item3), 0)
        clabel = torch.cat((clabel1, clabel2), 0)

        # output of the first batch
        res_1 = siamese_model(batch_1)

        # output of the second batch
        res_2 = siamese_model(batch_2)

        loss, genuine_loss, impostor_loss = contrastive_loss_gamma(res_1, res_2, clabel, siamese_model, margin)

        # setting gradients of the optimizer to zero
        optimizer.zero_grad()

        # calculating gradients with backpropagation
        loss.backward()

        # updating the weights
        optimizer.step()

        loss_log.append(loss.cpu().item())
        genuine_loss_log.append(genuine_loss.cpu().item())
        impostor_loss_log.append(impostor_loss.cpu().item())

    train_loss = np.mean(np.array(loss_log))
    genuine_loss = np.mean(np.array(genuine_loss_log))
    impostor_loss = np.mean(np.array(impostor_loss_log))

    return train_loss, genuine_loss, impostor_loss


def validate(siamese_model, val_loader, margin):
    with torch.no_grad():
        siamese_model.eval()
        loss_log = []
        genuine_loss_log = []
        impostor_loss_log = []

        for batch_idx, ((item1, item2, item3), (clabel1, clabel2)) in enumerate(val_loader):
            batch_1 = torch.cat((item1, item1), 0)
            batch_2 = torch.cat((item2, item3), 0)
            clabel = torch.cat((clabel1, clabel2), 0)

            # output of the first batch
            res_1 = siamese_model(batch_1)

            # output of the second batch
            res_2 = siamese_model(batch_2)

            loss, genuine_loss, impostor_loss = contrastive_loss_gamma(res_1, res_2, clabel, siamese_model, margin)

            loss_log.append(loss.cpu().item())
            genuine_loss_log.append(genuine_loss.cpu().item())
            impostor_loss_log.append(impostor_loss.cpu().item())

        val_loss = np.mean(np.array(loss_log))
        genuine_loss = np.mean(np.array(genuine_loss_log))
        impostor_loss = np.mean(np.array(impostor_loss_log))

    return val_loss, genuine_loss, impostor_loss


def train_classification(siamese_model, optimizer, train_loader):
    siamese_model.train()
    loss_log = []

    criterion = nn.CrossEntropyLoss()

    for batch_idx, (items, masks, labels) in enumerate(train_loader):
        # output of the first batch

        if masks[0].sum() < 0:
            res_1, _ = siamese_model(items, None)
        else:
            res_1, _ = siamese_model(items, masks)

        # loss, genuine_loss, impostor_loss = triplet_loss_gamma(res_1, res_2, res_3, siamese_model, margin)
        loss = criterion(res_1, labels)

        # setting gradients of the optimizer to zero
        optimizer.zero_grad()

        # calculating gradients with backpropagation
        loss.backward()

        # updating the weights
        optimizer.step()

        loss_log.append(loss.cpu().item())

    train_loss = np.mean(np.array(loss_log))

    return train_loss


def validate_classification(siamese_model, val_loader):
    with torch.no_grad():
        siamese_model.eval()
        loss_log = []

        criterion = nn.CrossEntropyLoss()

        for batch_idx, (items, masks, labels) in enumerate(val_loader):
            # output of the first batch
            if masks[0].sum() < 0:
                res_1, _ = siamese_model(items, None)
            else:
                res_1, _ = siamese_model(items, masks)

            loss = criterion(res_1, labels)

            loss_log.append(loss.cpu().item())

        val_loss = np.mean(np.array(loss_log))

        return val_loss


def train_mix(siamese_model, optimizer, train_loader, epoch, two_mining=0, changel=0):
    siamese_model.train()
    loss_log = []

    criterion = nn.CrossEntropyLoss()

    for batch_idx, (items, masks, labels) in enumerate(train_loader):
        # output of the first batch

        if masks[0].sum() < 0:
            res_1, res_2 = siamese_model(items, None)
        else:
            res_1, res_2 = siamese_model(items, masks)

        if two_mining != 0:
            if epoch < two_mining:
                batch_hard = 0
            else:
                batch_hard = 1
        else:
            batch_hard = 0

        if changel == 0:
            loss1, _, _ = triplet_loss_mining(res_2, margin=1.5, batch_hard=batch_hard, loss_t=0)
            loss2 = criterion(res_1, labels)
            loss = loss1 + loss2
        else:
            if epoch < 30:
                loss, _, _ = triplet_loss_mining(res_2, margin=1.5, batch_hard=batch_hard, loss_t=0)
            else:
                loss = criterion(res_1, labels)

        # setting gradients of the optimizer to zero
        optimizer.zero_grad()

        # calculating gradients with backpropagation
        loss.backward()

        # updating the weights
        optimizer.step()

        loss_log.append(loss.cpu().item())

    train_loss = np.mean(np.array(loss_log))

    return train_loss


def train_triplet(siamese_model, optimizer, train_loader, margin, writer, epoch):
    siamese_model.train()
    loss_log = []
    genuine_loss_log = []
    impostor_loss_log = []

    log_step = np.ceil(len(train_loader)/50)*epoch+1
    log_counter = 0

    for batch_idx, ((item1, item2, item3), (mask1, mask2, mask3), (clabel1, clabel2)) in enumerate(train_loader):
        # output of the first batch
        if mask1.sum() < 0:
            res_1 = siamese_model(item1, None)
            res_2 = siamese_model(item2, None)
            res_3 = siamese_model(item3, None)
        else:
            res_1 = siamese_model(item1, mask1)
            res_2 = siamese_model(item2, mask2)
            res_3 = siamese_model(item3, mask3)

        # loss, genuine_loss, impostor_loss = triplet_loss_gamma(res_1, res_2, res_3, siamese_model, margin)
        loss, genuine_loss, impostor_loss = triplet_loss(res_1, res_2, res_3, margin=margin)

        # setting gradients of the optimizer to zero
        optimizer.zero_grad()

        # calculating gradients with backpropagation
        loss.backward()

        # updating the weights
        optimizer.step()

        loss_log.append(loss.cpu().item())
        genuine_loss_log.append(genuine_loss.cpu().item())
        impostor_loss_log.append(impostor_loss.cpu().item())

        if batch_idx % 50 == 0:
            writer.add_scalar('training_loss', loss.cpu().item(), global_step=log_step+log_counter)
            writer.add_scalar('training_genuine_loss', genuine_loss.cpu().item(), global_step=log_step+log_counter)
            writer.add_scalar('training_impostor_loss', impostor_loss.cpu().item(), global_step=log_step+log_counter)
            for tag, value in siamese_model.named_parameters():
                if value.data.cpu().numpy().size == 1:
                    writer.add_scalar(tag, value.data.cpu().numpy(), global_step=log_step+log_counter)
                else:
                    writer.add_histogram(tag, value.data.cpu().numpy(), global_step=log_step+log_counter)
                    writer.add_histogram('{}_grad'.format(tag), value.grad.data.cpu().numpy(), global_step=log_step+log_counter)
            log_counter += 1

    train_loss = np.mean(np.array(loss_log))
    genuine_loss = np.mean(np.array(genuine_loss_log))
    impostor_loss = np.mean(np.array(impostor_loss_log))

    return train_loss, genuine_loss, impostor_loss


def train_triplet_mining(siamese_model, optimizer, train_loader, margin, writer, epoch, two_mining, loss_t, all_pos, threem, mask_pos=None, mask_neg=None, norm_dist=0, ms=3):
    siamese_model.train()
    loss_log = []

    for batch_idx, batch in enumerate(train_loader):
        if len(batch) == 3:
            items_list = [batch[0]]
            masks_list = [batch[1]]
            labels = batch[2]
        else:
            items_list = [batch[0], batch[1], batch[2]]
            masks_list = [batch[3], batch[4], batch[5]]

        for i in range(len(items_list)):
            items = items_list[i]
            masks = masks_list[i]
            # output of the first batch
            if torch.cuda.is_available():
                masks = masks.cuda()
                items = items.cuda()

            if masks[0].sum() < 0:
                res_1 = siamese_model(items, None)
            else:
                res_1 = siamese_model(items, masks)

            if ms == 0:
                batch_hard = 2
            elif ms == 1:
                batch_hard = 0
            elif ms == 2:
                batch_hard = 1
            else:
                if two_mining != 0:
                    if epoch < two_mining:
                        batch_hard = 0
                    else:
                        batch_hard = 1
                else:
                    batch_hard = 0

            # loss, genuine_loss, impostor_loss = triplet_loss_gamma(res_1, res_2, res_3, siamese_model, margin)
            loss, _, _ = triplet_loss_mining(res_1, siamese_model, margin=margin, batch_hard=batch_hard, loss_t=loss_t, allpos=all_pos,
                                             norm_dist=norm_dist, labels=labels)

            # setting gradients of the optimizer to zero
            optimizer.zero_grad()

            # calculating gradients with backpropagation
            loss.backward()

            # updating the weights
            optimizer.step()

            loss_log.append(loss.cpu().item())

    train_loss = np.mean(np.array(loss_log))

    return train_loss, 0, 0


def train_triplet_mining_aug(siamese_model, optimizer, train_loader, margin, writer, epoch):
    siamese_model.train()
    loss_log = []
    genuine_loss_log = []
    impostor_loss_log = []

    log_step = np.ceil(len(train_loader)/25)*epoch+1
    log_counter = 0

    for batch_idx, (items, items_aug) in enumerate(train_loader):
        # output of the first batch
        res_1 = siamese_model(items, None)

        res_2 = siamese_model(items_aug, None)

        # loss, genuine_loss, impostor_loss = triplet_loss_gamma(res_1, res_2, res_3, siamese_model, margin)
        loss, genuine_loss, impostor_loss = triplet_loss_mining_aug(res_1, res_2, margin=margin)

        # setting gradients of the optimizer to zero
        optimizer.zero_grad()

        # calculating gradients with backpropagation
        loss.backward()

        # updating the weights
        optimizer.step()

        loss_log.append(loss.cpu().item())
        genuine_loss_log.append(genuine_loss.cpu().item())
        impostor_loss_log.append(impostor_loss.cpu().item())

        if batch_idx % 25 == 0:
            writer.add_scalar('training_loss', loss.cpu().item(), global_step=log_step+log_counter)
            for tag, value in siamese_model.named_parameters():
                if len(tag.split('.')) == 3:
                    tag1, tag2 = tag.split('.')[-2:]
                else:
                    tag1, tag2 = tag.split('.')
                if value.data.cpu().numpy().size == 1:
                    writer.add_scalar('{}/{}'.format(tag1, tag2), value.data.cpu().numpy(), global_step=log_step+log_counter)
                else:
                    writer.add_histogram('{}/{}'.format(tag1, tag2), value.data.cpu().numpy(), global_step=log_step+log_counter)
                    writer.add_histogram('{}/{}_grad'.format(tag1, tag2), value.grad.data.cpu().numpy(), global_step=log_step+log_counter)
                    writer.add_scalar('{}/{}_gradnorm'.format(tag1, tag2), value.grad.norm(2).data.cpu().numpy(), global_step=log_step+log_counter)
                    writer.add_scalar('{}/{}_gradmean'.format(tag1, tag2), value.grad.mean().data.cpu().numpy(), global_step=log_step+log_counter)
                    writer.add_scalar('{}/{}_gradstd'.format(tag1, tag2), value.grad.std().data.cpu().numpy(), global_step=log_step+log_counter)
            log_counter += 1

    train_loss = np.mean(np.array(loss_log))
    genuine_loss = np.mean(np.array(genuine_loss_log))
    impostor_loss = np.mean(np.array(impostor_loss_log))

    return train_loss, genuine_loss, impostor_loss


def validate_triplet_mining(siamese_model, val_loader, margin, writer, epoch, loss_t=0, cla=0, mask_pos=None, mask_neg=None, norm_dist=0):
    with torch.no_grad():
        siamese_model.eval()
        loss_log = []
        genuine_loss_log = []
        impostor_loss_log = []

        log_step = np.ceil(len(val_loader)/25)*epoch+1
        log_counter = 0

        for batch_idx, (items, masks, labels) in enumerate(val_loader):
            # output of the first batch

            if torch.cuda.is_available():
                masks = masks.cuda()
                items = items.cuda()

            if masks[0].sum() < 0:
                if cla == 0:
                    res_1 = siamese_model(items, None)
                else:
                    _, res_1 = siamese_model(items, None)
            else:
                if cla == 0:
                    res_1 = siamese_model(items, masks)
                else:
                    _, res_1 = siamese_model(items, masks)

            # loss, genuine_loss, impostor_loss = triplet_loss_gamma(res_1, res_2, res_3, siamese_model, margin)
            loss, genuine_loss, impostor_loss = triplet_loss_mining(res_1, siamese_model, margin=margin, batch_hard=2, loss_t=loss_t,
                                                                    norm_dist=norm_dist, labels=labels)

            loss_log.append(loss.cpu().item())
            genuine_loss_log.append(genuine_loss.cpu().item())
            impostor_loss_log.append(impostor_loss.cpu().item())

        val_loss = np.mean(np.array(loss_log))
        genuine_loss = np.mean(np.array(genuine_loss_log))
        impostor_loss = np.mean(np.array(impostor_loss_log))

        return val_loss, genuine_loss, impostor_loss


def validate_triplet(siamese_model, val_loader, margin, writer, epoch, loss_t, cla=0):
    with torch.no_grad():
        siamese_model.eval()
        loss_log = []
        genuine_loss_log = []
        impostor_loss_log = []

        log_step = np.ceil(len(val_loader) / 25) * epoch + 1
        log_counter = 0

        for batch_idx, ((item1, item2, item3), (mask1, mask2, mask3), (clabel1, clabel2)) in enumerate(val_loader):

            if mask1.sum() < 0:
                if cla == 0:
                    res_1 = siamese_model(item1, None)
                    res_2 = siamese_model(item2, None)
                    res_3 = siamese_model(item3, None)
                else:
                    _, res_1 = siamese_model(item1, None)
                    _, res_2 = siamese_model(item2, None)
                    _, res_3 = siamese_model(item3, None)
            else:
                if cla == 0:
                    res_1 = siamese_model(item1, mask1)
                    res_2 = siamese_model(item2, mask2)
                    res_3 = siamese_model(item3, mask3)
                else:
                    _, res_1 = siamese_model(item1, mask1)
                    _, res_2 = siamese_model(item2, mask2)
                    _, res_3 = siamese_model(item3, mask3)

            loss, genuine_loss, impostor_loss = triplet_loss(res_1, res_2, res_3, margin=margin, loss_t=loss_t)

            loss_log.append(loss.cpu().item())
            genuine_loss_log.append(genuine_loss.cpu().item())
            impostor_loss_log.append(impostor_loss.cpu().item())

            """
            if batch_idx % 25 == 0:
                writer.add_scalar('validation_loss', loss.cpu().item(), global_step=log_step+log_counter)
                log_counter += 1
            """

        val_loss = np.mean(np.array(loss_log))
        genuine_loss = np.mean(np.array(genuine_loss_log))
        impostor_loss = np.mean(np.array(impostor_loss_log))

    return val_loss, genuine_loss, impostor_loss


def train_lossless_t(siamese_model, optimizer, train_loader, margin):
    siamese_model.train()
    loss_log = []
    genuine_loss_log = []
    impostor_loss_log = []

    for batch_idx, ((item1, item2, item3), (clabel1, clabel2)) in enumerate(train_loader):

        # output of the first batch
        res_1 = siamese_model(item1)

        # output of the second batch
        res_2 = siamese_model(item2)

        res_3 = siamese_model(item3)

        # loss, genuine_loss, impostor_loss = triplet_loss_gamma(res_1, res_2, res_3, siamese_model, margin)
        loss, genuine_loss, impostor_loss = lossless_triplet_gamma(res_1, res_2, res_3, siamese_model, margin=margin)

        # setting gradients of the optimizer to zero
        optimizer.zero_grad()

        # calculating gradients with backpropagation
        loss.backward()

        # updating the weights
        optimizer.step()

        loss_log.append(loss.cpu().item())
        genuine_loss_log.append(genuine_loss.cpu().item())
        impostor_loss_log.append(impostor_loss.cpu().item())

    train_loss = np.mean(np.array(loss_log))
    genuine_loss = np.mean(np.array(genuine_loss_log))
    impostor_loss = np.mean(np.array(impostor_loss_log))

    return train_loss, genuine_loss, impostor_loss


def validate_lossless_t(siamese_model, val_loader, margin):
    with torch.no_grad():
        siamese_model.eval()
        loss_log = []
        genuine_loss_log = []
        impostor_loss_log = []

        for batch_idx, ((item1, item2, item3), (clabel1, clabel2)) in enumerate(val_loader):
            # output of the first batch
            res_1 = siamese_model(item1)

            # output of the second batch
            res_2 = siamese_model(item2)

            res_3 = siamese_model(item3)

            loss, genuine_loss, impostor_loss = lossless_triplet_gamma(res_1, res_2, res_3, siamese_model, margin=margin)

            loss_log.append(loss.cpu().item())
            genuine_loss_log.append(genuine_loss.cpu().item())
            impostor_loss_log.append(impostor_loss.cpu().item())

        val_loss = np.mean(np.array(loss_log))
        genuine_loss = np.mean(np.array(genuine_loss_log))
        impostor_loss = np.mean(np.array(impostor_loss_log))

    return val_loss, genuine_loss, impostor_loss


def train_aug(siamese_model, optimizer, train_loader):
    siamese_model.train()
    loss_log = []
    genuine_loss_log = []
    impostor_loss_log = []

    for batch_idx, ((item1, item2), (clabel)) in enumerate(train_loader):

        if torch.cuda.is_available():
            item1 = item1.cuda()
            item2 = item2.cuda()
            clabel = clabel.cuda()

        # output of the first batch
        res_1 = siamese_model(item1)

        # output of the second batch
        res_2 = siamese_model(item2)

        loss, genuine_loss, impostor_loss = contrastive_loss_gamma(res_1, res_2, clabel, siamese_model)

        # setting gradients of the optimizer to zero
        optimizer.zero_grad()

        # calculating gradients with backpropagation
        loss.backward()

        # updating the weights
        optimizer.step()

        loss_log.append(loss.cpu().item())
        genuine_loss_log.append(genuine_loss.cpu().item())
        impostor_loss_log.append(impostor_loss.cpu().item())

    train_loss = np.mean(np.array(loss_log))
    genuine_loss = np.mean(np.array(genuine_loss_log))
    impostor_loss = np.mean(np.array(impostor_loss_log))

    return train_loss, genuine_loss, impostor_loss


def validate_aug(siamese_model, val_loader):
    with torch.no_grad():
        siamese_model.eval()
        loss_log = []
        genuine_loss_log = []
        impostor_loss_log = []

        for batch_idx, ((item1, item2), (clabel)) in enumerate(val_loader):

            if torch.cuda.is_available():
                item1 = item1.cuda()
                item2 = item2.cuda()
                clabel = clabel.cuda()

            # output of the first batch
            res_1 = siamese_model(item1)

            # output of the second batch
            res_2 = siamese_model(item2)

            loss, genuine_loss, impostor_loss = contrastive_loss_gamma(res_1, res_2, clabel, siamese_model)

            loss_log.append(loss.cpu().item())
            genuine_loss_log.append(genuine_loss.cpu().item())
            impostor_loss_log.append(impostor_loss.cpu().item())

        val_loss = np.mean(np.array(loss_log))
        genuine_loss = np.mean(np.array(genuine_loss_log))
        impostor_loss = np.mean(np.array(impostor_loss_log))

    return val_loss, genuine_loss, impostor_loss


def test(siamese_model, test_loader, loss_t=0, cla=0):
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    # deactivating gradient tracking for testing
    with torch.no_grad():
        # setting the model to evaluation mode
        siamese_model.eval()

        # tensor for storing all the pairwise distances
        embed_all = torch.tensor([], device=device)

        for batch_idx, (batch_1, mask1) in enumerate(test_loader):

            if torch.cuda.is_available():
                mask1 = mask1.cuda()
                batch_1 = batch_1.cuda()
            # output of the first batch
            if mask1.sum() < 0:
                if cla == 0:
                    res_1 = siamese_model(batch_1, None)
                else:
                    _, res_1 = siamese_model(batch_1, None)
            else:
                if cla == 0:
                    res_1 = siamese_model(batch_1, mask1)
                else:
                    _, res_1 = siamese_model(batch_1, mask1)

            embed_all = torch.cat((embed_all, res_1))

        if loss_t == 0:
            #dist_all = torch.norm(embed_all[:, None] - embed_all, dim=2, p=2)
            dist_all = F.pdist(embed_all)
        else:
            dist_all = torch.from_numpy(scipdist(embed_all.cpu().detach().numpy(), metric='cosine'))

    return dist_all


def find_lr(model, optimizer, train_loader, margin=1, init_value=1e-8, final_value=10., beta=0.7):
    """
    taken from: https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html
    """
    import math

    num = len(train_loader)-1
    mult = (final_value / init_value) ** (1/num)
    lr = init_value
    optimizer.param_groups[0]['lr'] = lr
    avg_loss = 0.
    best_loss = 0.
    batch_num = 0
    losses = []
    log_lrs = []

    for batch_idx, ((item1, item2, item3), (clabel1, clabel2)) in enumerate(train_loader):

        batch_1 = torch.cat((item1, item1), 0)
        batch_2 = torch.cat((item2, item3), 0)
        clabel = torch.cat((clabel1, clabel2), 0)

        # output of the first batch
        res_1 = model(batch_1)

        # output of the second batch
        res_2 = model(batch_2)

        loss, genuine_loss, impostor_loss = contrastive_loss_gamma(res_1, res_2, clabel, model, margin=margin)

        batch_num += 1

        # Compute the smoothed loss
        avg_loss = beta * avg_loss + (1-beta) * loss.cpu().item()
        smoothed_loss = avg_loss / (1 - beta**batch_num)
        # Stop if the loss is exploding
        if batch_num > 1 and smoothed_loss > 4 * best_loss:
            return log_lrs, losses
        # Record the best loss
        if smoothed_loss < best_loss or batch_num == 1:
            best_loss = smoothed_loss
        # Store the values
        losses.append(smoothed_loss)
        log_lrs.append(math.log10(lr))
        # Do the SGD step
        loss.backward()
        optimizer.step()
        # Update the lr for the next step
        lr *= mult
        optimizer.param_groups[0]['lr'] = lr

    return log_lrs, losses


def find_lr_triplet(model, optimizer, train_loader, margin=1, init_value=1e-8, final_value=1., beta=0.98):
    """
    taken and modified from: https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html
    """
    import math

    num = len(train_loader)-1
    mult = (final_value / init_value) ** (1/num)
    lr = init_value
    optimizer.param_groups[0]['lr'] = lr
    avg_loss = 0.
    best_loss = 0.
    batch_num = 0
    losses = []
    log_lrs = []

    for batch_idx, ((item1, item2, item3), (mask1, mask2, mask3), (clabel1, clabel2)) in enumerate(train_loader):

        # output of the first batch
        if mask1.sum() < 0:
            res_1 = model(item1, None)
            res_2 = model(item2, None)
            res_3 = model(item3, None)
        else:
            res_1 = model(item1, mask1)
            res_2 = model(item2, mask2)
            res_3 = model(item3, mask3)

        # loss, genuine_loss, impostor_loss = triplet_loss_gamma(res_1, res_2, res_3, siamese_model, margin)
        loss, genuine_loss, impostor_loss = triplet_loss(res_1, res_2, res_3, margin=margin)

        batch_num += 1

        # Compute the smoothed loss
        avg_loss = beta * avg_loss + (1-beta) * loss.cpu().item()
        smoothed_loss = avg_loss / (1 - beta**batch_num)
        # Stop if the loss is exploding
        if batch_num > 1 and smoothed_loss > 4 * best_loss:
            return log_lrs, losses
        # Record the best loss
        if smoothed_loss < best_loss or batch_num == 1:
            best_loss = smoothed_loss
        # Store the values
        losses.append(smoothed_loss)
        log_lrs.append(math.log10(lr))
        # Do the SGD step
        loss.backward()
        optimizer.step()
        # Update the lr for the next step
        lr *= mult
        optimizer.param_groups[0]['lr'] = lr

    return log_lrs, losses


def find_lr_lossless_t(model, optimizer, train_loader, margin=1, init_value=1e-8, final_value=10., beta=0.7):
    """
    taken and modified from: https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html
    """
    import math

    num = len(train_loader)-1
    mult = (final_value / init_value) ** (1/num)
    lr = init_value
    optimizer.param_groups[0]['lr'] = lr
    avg_loss = 0.
    best_loss = 0.
    batch_num = 0
    losses = []
    log_lrs = []

    for batch_idx, ((item1, item2, item3), (clabel1, clabel2)) in enumerate(train_loader):

        # output of the first batch
        res_1 = model(item1)

        # output of the second batch
        res_2 = model(item2)

        res_3 = model(item3)

        # loss, genuine_loss, impostor_loss = triplet_loss_gamma(res_1, res_2, res_3, siamese_model, margin)
        loss, genuine_loss, impostor_loss = lossless_triplet_gamma(res_1, res_2, res_3, model, margin=margin)

        batch_num += 1

        # Compute the smoothed loss
        avg_loss = beta * avg_loss + (1-beta) * loss.cpu().item()
        smoothed_loss = avg_loss / (1 - beta**batch_num)
        # Stop if the loss is exploding
        if batch_num > 1 and smoothed_loss > 4 * best_loss:
            return log_lrs, losses
        # Record the best loss
        if smoothed_loss < best_loss or batch_num == 1:
            best_loss = smoothed_loss
        # Store the values
        losses.append(smoothed_loss)
        log_lrs.append(math.log10(lr))
        # Do the SGD step
        loss.backward()
        optimizer.step()
        # Update the lr for the next step
        lr *= mult
        optimizer.param_groups[0]['lr'] = lr

    return log_lrs, losses
