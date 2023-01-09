import torch.optim as optim
import torch
import os

from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
import argparse
import json
import numpy as np
import time
import math

from models.Conv2 import Conv2
from models.ConvX import ConvX
from models.ConvConcat import ConvConcat
from models.ConvBaseline import ConvBaseline
from models.ConvRes import ConvRes
from models.ConvResLarge import ConvResLarge
from models.ConvResLarge2 import ConvResLarge2
from models.ConvResNonM import ConvResNonM
from models.ConvResNonK import ConvResNonK
from models.ConvBaseNonM import ConvBaseNonM
from models.ConvBaselineJP import ConvBaselineJP
from models.ConvBaselineJP5L import ConvBaselineJP5L
from models.ConvBaselineJPD import ConvBaselineJPD
from models.ConvBaselineJP5LIN import ConvBaselineJP5LIN
from models.ConvBaselineJP5LINMP import ConvBaselineJP5LINMP
from models.ConvBaseline5L import ConvBaseline5L
from models.ConvBaseline5LD import ConvBaseline5LD
from models.ConvBaselineJP5LD import ConvBaselineJP5LD
from models.ConvBaseline6L import ConvBaseline6L
from models.ConvBaselineJP6L import ConvBaselineJP6L
from models.ConvBaselineAJP5LD import ConvBaselineAJP5LD
from models.ConvBaselineAJP5LND import ConvBaselineAJP5LND
from models.ConvBaselineAJP5LNDS import ConvBaselineAJP5LNDS
from models.ConvBaselineJP5LSD import ConvBaselineJP5LSD
from models.ConvX import make_layers
from models.Conv5 import Conv5
from dataset.pairs_fixed_size import PairsFixedSize
from dataset.triplet_mining_fixed_size import TripletMiningFixedSize
from dataset.classification_fixed_size import ClassificationFixedSize
from dataset.mix_fixed_size import MixFixedSize
from dataset.pairs_aug_full_size import PairsAugFullSize
from dataset.pairs_aug_fixed_size import PairsAugFixedSize
from dataset.single_fixed_size import SingleFixedSize
from dataset.single_full_size import SingleFullSize
from models.siamese_tr_val import *
from utils.siamese_utils import MAP
from utils.siamese_utils import square_dist_tensor
from utils.siamese_utils import average_precision
from utils.siamese_utils import getEvalStatistics
from utils.dataset_utils import import_dataset_from_json
from utils.dataset_utils import import_dataset_from_h5
from utils.dataset_utils import import_large_dataset_from_json
from utils.dataset_utils import import_dataset_from_pt
from utils.dataset_utils import custom_collate
from utils.dataset_utils import triplet_mining_collate
from utils.dataset_utils import mix_collate
from utils.dataset_utils import mul_len_triplet_collate

# TODO: standardize validation, save loss plots as image


def main(main_params,
         config_name,
         shortname,
         notes,
         savename,
         rootdir,
         conv_cfg,
         margin,
         sel_loss,
         sigmoid,
         flr,
         tm,
         pl,
         o,
         tmb,
         cocat,
         st,
         es,
         pltr,
         dr,
         lin1,
         lin2,
         bn,
         da,
         summ,
         ap,
         val_mining,
         twom,
         lt,
         cla,
         cluster,
         changel,
         mullen,
         fullen,
         allpos,
         threem,
         bigt,
         mulprelu,
         mulautop,
         masking,
         norm_dist,
         lr,
         nw,
         rl,
         chunks,
         ud,
         ms
         ):

    summary = dict()
    summary['main_params'] = main_params

    print('{}_{}'.format(savename, config_name))

    #torch.set_num_threads(1)

    if main_params['seed'] is not None:
        np.random.seed(main_params['seed'])
        torch.manual_seed(main_params['seed'])
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.cuda.manual_seed(main_params['seed'])

    # SELECTING THE MODEL

    if cocat == 0:
        siamese_model = ConvBaseline(bn=bn, lin1=lin1, lin2=lin2, sum_method=summ, autopool_p=ap, cla=cla,
                                     use_sigmoid=sigmoid, mulprelu=mulprelu, mulautop=mulautop)
    elif cocat == 1:
        siamese_model = ConvConcat()
    elif cocat == 2:
        siamese_model = ConvRes()
    elif cocat == 3:
        siamese_model = ConvResLarge(sum_method=summ, autopool_p=ap, cla=cla)
    elif cocat == 4:
        siamese_model = ConvResLarge2(cla=cla)
    elif cocat == 5:
        siamese_model = ConvResNonM(sum_method=summ, autopool_p=ap, cla=cla)
    elif cocat == 6:
        siamese_model = ConvResNonK(cla=cla)
    elif cocat == 7:
        siamese_model = ConvBaseNonM(bn=bn, lin1=lin1, lin2=lin2, sum_method=summ, autopool_p=ap, cla=cla)
    elif cocat == 8:
        siamese_model = ConvBaselineJP(bn=bn, lin1=lin1, lin2=lin2, sum_method=summ, autopool_p=ap, cla=cla)
    elif cocat == 9:
        siamese_model = ConvBaselineJPD(bn=bn, lin1=lin1, lin2=lin2, sum_method=summ, autopool_p=ap, cla=cla)
    elif cocat == 10:
        siamese_model = ConvBaselineJP5L(bn=bn, lin1=lin1, lin2=lin2, sum_method=summ, autopool_p=ap, cla=cla,
                                         use_sigmoid=sigmoid)
    elif cocat == 11:
        siamese_model = ConvBaselineJP5LIN(bn=bn, lin1=lin1, lin2=lin2, sum_method=summ, autopool_p=ap, cla=cla,
                                           use_sigmoid=sigmoid)
    elif cocat == 12:
        siamese_model = ConvBaselineJP5LINMP(bn=bn, lin1=lin1, lin2=lin2, sum_method=summ, autopool_p=ap, cla=cla,
                                             use_sigmoid=sigmoid)
    elif cocat == 13:
        siamese_model = ConvBaseline5L(bn=bn, lin1=lin1, lin2=lin2, sum_method=summ, autopool_p=ap, cla=cla,
                                       use_sigmoid=sigmoid)
    elif cocat == 14:
        siamese_model = ConvBaseline5LD(bn=bn, lin1=lin1, lin2=lin2, sum_method=summ, autopool_p=ap, cla=cla,
                                        use_sigmoid=sigmoid)
    elif cocat == 15:
        siamese_model = ConvBaselineJP5LD(bn=bn, lin1=lin1, lin2=lin2, sum_method=summ, autopool_p=ap, cla=cla,
                                          use_sigmoid=sigmoid)
    elif cocat == 16:
        siamese_model = ConvBaseline6L(bn=bn, lin1=lin1, lin2=lin2, sum_method=summ, autopool_p=ap, cla=cla,
                                       use_sigmoid=sigmoid)
    elif cocat == 17:
        siamese_model = ConvBaselineJP6L(bn=bn, lin1=lin1, lin2=lin2, sum_method=summ, autopool_p=ap, cla=cla,
                                         use_sigmoid=sigmoid)
    elif cocat == 18:
        siamese_model = ConvBaselineAJP5LD(bn=bn, lin1=lin1, lin2=lin2, sum_method=summ, autopool_p=ap, cla=cla,
                                           use_sigmoid=sigmoid)
    elif cocat == 19:
        siamese_model = ConvBaselineAJP5LND(bn=bn, lin1=lin1, lin2=lin2, sum_method=summ, autopool_p=ap, cla=cla,
                                            use_sigmoid=sigmoid)
    elif cocat == 20:
        siamese_model = ConvBaselineJP5LSD(bn=bn, lin1=lin1, lin2=lin2, sum_method=summ, autopool_p=ap, cla=cla,
                                           use_sigmoid=sigmoid)
    elif cocat == 21:
        siamese_model = ConvBaselineAJP5LNDS(bn=bn, lin1=lin1, lin2=lin2, sum_method=summ, autopool_p=ap, cla=cla,
                                             use_sigmoid=sigmoid)

    # siamese_model = Conv2()

    if torch.cuda.is_available():
        siamese_model.cuda()

    # SELECTING THE OPTIMIZER

    if o == 0:
        optimizer = optim.SGD(siamese_model.parameters(),
                              lr=lr,
                              weight_decay=main_params['weight_decay'],
                              momentum=main_params['momentum'])
    else:
        optimizer = optim.Adam(siamese_model.parameters(), lr=1e-4)

    batch_size = main_params['batch_size']

    num_of_epochs = main_params['num_of_epochs']

    # INITIALIZING LOSS TRACKERS

    train_loss_log = []
    val_loss_log = []
    val_map_log = []
    test_map_log = []

    train_data, train_labels = import_dataset_from_pt('{}'.format(main_params['train_path']), chunks=chunks)
    #train_data, train_labels = import_dataset_from_h5('{}{}'.format(rootdir, main_params['train_path']), 4, cocat=cocat)
    #train_data, train_labels = import_dataset_from_json('{}{}'.format(rootdir, main_params['train_path']))
    print('Train data has been loaded!')

    val_data, val_labels = import_dataset_from_pt('{}'.format(main_params['test_path']), chunks=1)
    #val_data, val_labels = import_dataset_from_h5('{}{}'.format(rootdir, main_params['test_path']), cocat=cocat)
    #test_data, test_labels = import_dataset_from_json('{}{}'.format(rootdir, main_params['test_path']))
    print('Validation data has been loaded!')

    if bigt == 1:
        train_data.extend(val_data)
        train_labels.extend(val_labels)

    if cla == 1:
        unique_labels = np.unique(train_labels)
        class_labels = []

    if cocat == 7:
        h = 12
    else:
        h = 23

    if cla == 0:
        if tm == 1:
            train_set = TripletMiningFixedSize(train_data, train_labels, idx=None, h=h, w=pltr, stretch=st, data_aug=da,
                                               mul_len=mullen, masked=masking, rand_length=rl, uni_dist=ud)
            # sampler = WeightedRandomSampler(train_set.clique_weights, len(train_set), replacement=False)
            collate_func = triplet_mining_collate if mullen == 0 else mul_len_triplet_collate
            train_loader = DataLoader(train_set, batch_size=tmb, shuffle=True,
                                      collate_fn=collate_func, drop_last=True, num_workers=nw)
        else:
            train_set = PairsFixedSize(train_data, train_labels, idx=None, w=pl)
            train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    elif cla == 1:
        train_set = ClassificationFixedSize(train_data, train_labels, w=pltr, data_aug=da)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    else:
        train_set = MixFixedSize(train_data, train_labels, idx=None, w=pltr, stretch=st, data_aug=da)
        # sampler = WeightedRandomSampler(train_set.clique_weights, len(train_set), replacement=False)
        train_loader = DataLoader(train_set, batch_size=tmb, shuffle=True,
                                  collate_fn=mix_collate)

    if bigt == 0:
        if val_mining == 1:
            val_set = TripletMiningFixedSize(val_data, val_labels, idx=None, h=h, w=pl, stretch=st, data_aug=0,
                                             masked=masking)
            # val_sampler = WeightedRandomSampler(val_set.clique_weights, len(val_set), replacement=False)
            val_loader = DataLoader(val_set, batch_size=tmb, shuffle=True,
                                      collate_fn=triplet_mining_collate, drop_last=True, num_workers=nw)
        else:
            val_set = PairsFixedSize(val_data, val_labels, idx=None, w=pl, stretch=st)
            val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

    if fullen == 0:
        val_map_set = SingleFixedSize(val_data, val_labels, idx=None, h=h, w=pl, stretch=st)
        val_map_loader = DataLoader(val_map_set, batch_size=batch_size, shuffle=False)
    else:
        val_map_set = SingleFullSize(val_data, val_labels, idx=None)
        val_map_loader = DataLoader(val_map_set, batch_size=1, shuffle=False)

    if bigt == 1:
        test_data, test_labels = import_dataset_from_h5('{}{}'.format(rootdir, 'data/benchmark_crema.h5'), benchmark=1)
        # test_data, test_labels = import_large_dataset_from_json('{}{}'.format(rootdir, 'data/large_benchmark.json'))
        test_map_set = SingleFullSize(test_data, test_labels, idx=None, single_files=0)

        test_map_loader = DataLoader(test_map_set, batch_size=1, shuffle=False)

    start_time = time.monotonic()

    early_s_pat = main_params['early_s_pat']

    if early_s_pat != 0:
        patience = early_s_pat
        min_val_loss = 100
    if es == 1:
        min_val_loss *= -1

    '''
    find lr 
    log_lrs, losses = find_lr(siamese_model, optimizer, train_loader)
    lrs = dict()
        lrs['log_lrs'] = log_lrs
        lrs['losses'] = losses
        with open('find_lr.json', 'w') as log:
            json.dump(lrs, log, indent='\t')
    '''
    lr = main_params['lr']
    if flr == 1:
        # Find Learning Rate
        start = time.monotonic()
        if sel_loss == 0:
            log_lrs, losses = find_lr(siamese_model, optimizer, train_loader, margin=margin)
        elif sel_loss == 1:
            log_lrs, losses = find_lr_triplet(siamese_model, optimizer, train_loader, margin=margin)
        else:
            log_lrs, losses = find_lr_lossless_t(siamese_model, optimizer, train_loader, margin=margin)
        print('find lr {}'.format(time.monotonic()-start))
        # Reinitializing the model and the optimizer

        siamese_model = ConvX(make_layers(conv_cfg), use_sigmoid=sigmoid)
        # siamese_model = Conv2()

        if torch.cuda.is_available():
            siamese_model.cuda()

        # SELECTING THE OPTIMIZER

        lr = math.pow(10, log_lrs[np.argmin(losses[10:])])

        optimizer = optim.SGD(siamese_model.parameters(),
                              lr=lr,
                              weight_decay=main_params['weight_decay'],
                              momentum=main_params['momentum'])

    # SELECTING THE LR SCHEDULER

    if main_params['lr_decay'] == 1:
        if main_params['lrsch'] == 0:
            pass
        elif main_params['lrsch'] == 1:
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                T_max=50,
                                                                eta_min=0.00001)
        elif main_params['lrsch'] == 2:
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer,
                                                     step_size=main_params['patience'],
                                                     gamma=main_params['factor'])

        elif main_params['lrsch'] == 3:
            if es == 0:
                lrs_mode = 'min'
            else:
                lrs_mode = 'max'
            lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                mode=lrs_mode,
                                                                factor=main_params['factor'],
                                                                patience=main_params['patience'],
                                                                threshold=main_params['threshold'],
                                                                threshold_mode='abs',
                                                                cooldown=main_params['cooldown'],
                                                                min_lr=main_params['min_lr'],
                                                                verbose=True)

        elif main_params['lrsch'] == 4:
            if config_name == 'config_file_34':
                milestones = [100, 125]
            else:
                milestones = [80, 100]
            lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                          milestones=milestones,
                                                          gamma=main_params['factor'])

    if cluster == 0:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir='./tensorboard_1309/{}'.format(savename))
    else:
        writer = None

    best_mAP = 0

    mask_pos = None
    mask_neg = None
    
    for epoch in range(num_of_epochs):
        last_epoch = epoch

        if cla == 0:
            if sel_loss == 0:
                train_loss, _, _ = train(siamese_model, optimizer, train_loader, margin)
                val_loss, _, _ = validate(siamese_model, val_loader, margin)
            elif sel_loss == 1:
                start = time.monotonic()
                if tm == 1:
                    if da == 2:
                        train_loss, _, _ = train_triplet_mining_aug(siamese_model, optimizer,
                                                                                             train_loader,
                                                                                             margin, writer, epoch)
                    else:
                        train_loss, _, _ = train_triplet_mining(siamese_model, optimizer, train_loader,
                                                                margin, writer, epoch, two_mining=twom, loss_t=lt,
                                                                all_pos=allpos, threem=threem, mask_pos=mask_pos,
                                                                mask_neg=mask_neg, norm_dist=norm_dist, ms=ms)
                else:
                    train_loss, _, _ = train_triplet(siamese_model, optimizer, train_loader,
                                                     margin, writer, epoch)
                print('training epoch {} time {}'.format(epoch, time.monotonic()-start))
                if bigt == 0:
                    start = time.monotonic()
                    if val_mining == 1:
                        val_loss, _, _ = validate_triplet_mining(siamese_model, val_loader, margin, writer, epoch,
                                                                 loss_t=lt, mask_pos=mask_pos, mask_neg=mask_neg,
                                                                 norm_dist=norm_dist)
                    else:
                        val_loss, _, _ = validate_triplet(siamese_model, val_loader, margin, writer, epoch, loss_t=lt)
                    print('validation epoch {} time {}'.format(epoch, time.monotonic()-start))
            else:
                train_loss, _, _ = train_lossless_t(siamese_model, optimizer, train_loader,
                                                                              margin)
                val_loss, _, _ = validate_lossless_t(siamese_model, val_loader, margin)
        elif cla == 1:
            start = time.monotonic()
            train_loss = train_classification(siamese_model, optimizer, train_loader)
            print('training epoch {} time {}'.format(epoch, time.monotonic() - start))
            start = time.monotonic()
            if val_mining == 1:
                val_loss, _, _ = validate_triplet_mining(siamese_model, val_loader, margin, writer, epoch, loss_t=lt, cla=cla)
            else:
                val_loss, _, _ = validate_triplet(siamese_model, val_loader, margin, writer, epoch, loss_t=lt, cla=cla)
            print('validation epoch {} time {}'.format(epoch, time.monotonic() - start))
        else:
            start = time.monotonic()
            train_loss = train_mix(siamese_model, optimizer, train_loader, epoch=epoch, two_mining=twom, changel=changel)
            print('training epoch {} time {}'.format(epoch, time.monotonic() - start))
            start = time.monotonic()
            if val_mining == 1:
                val_loss, _, _ = validate_triplet_mining(siamese_model, val_loader, margin, writer, epoch, loss_t=lt,
                                                         cla=cla)
            else:
                val_loss, _, _ = validate_triplet(siamese_model, val_loader, margin, writer, epoch, loss_t=lt, cla=cla)
            print('validation epoch {} time {}'.format(epoch, time.monotonic() - start))
        '''
        if temp_lr == optimizer.param_groups[0]['lr']:
            if val_loss >= min_val_loss:
                patience -= 1
            else:
                min_val_loss = val_loss
                patience = 5

            if patience == 0:
                break
        else:
            patience = 5
            min_val_loss = 100
        '''

        train_loss_log.append(train_loss)
        if bigt == 0:
            val_loss_log.append(val_loss)

        start = time.monotonic()
        dist_map_tensor = test(siamese_model, val_map_loader, loss_t=lt, cla=cla)
        print('dist_map_tensor time {}'.format(time.monotonic()-start))

        #start = time.monotonic()
        dist_map_matrix = square_dist_tensor(dist_map_tensor)
        #print('square form {}'.format(time.monotonic()-start))

        #start = time.monotonic()
        #val_map_score = MAP(dist_map_matrix, val_map_set.labels)
        #print(val_map_score)
        start = time.monotonic()
        # val_map_score1 = getEvalStatistics(1/(dist_map_matrix.clone()+1e-15))
        val_map_score = average_precision(-1 * dist_map_matrix.float().clone() + torch.diag(torch.ones(len(val_data)) * float('-inf')))
        print('val map calculation {}'.format(time.monotonic()-start))
        val_map_log.append(val_map_score.item())

        if val_map_score.item() >= best_mAP:
            best_mAP = val_map_score.item()
            if main_params['save_model'] == 1:
                if not os.path.exists('{}saved_models_0210/'.format(rootdir)):
                    os.mkdir('{}saved_models_0210/'.format(rootdir))
                torch.save(siamese_model.state_dict(),
                           '{}saved_models_0210/model_{}_{}_{}.pt'.format(rootdir, shortname, savename, config_name))

        if bigt == 1:
            start = time.monotonic()
            dist_map_tensor = test(siamese_model, test_map_loader)
            dist_map_matrix = square_dist_tensor(dist_map_tensor)
            test_map_score = average_precision(
                -1 * dist_map_matrix.float().clone() + torch.diag(torch.ones(len(test_data)) * float('-inf')), benchmark=1)
            print('test map calculation {}'.format(time.monotonic() - start))
            test_map_log.append(test_map_score.item())

        if cluster == 0:
            writer.add_scalar('mAP/joan', val_map_score, global_step=epoch)
            # writer.add_scalar('mAP/chris', val_map_score1, global_step=epoch)
            writer.add_scalar('global/training_loss_global', train_loss, global_step=epoch)
            if bigt == 0:
                writer.add_scalar('global/validation_loss_global', val_loss, global_step=epoch)
            writer.add_scalar('global/tr_val_diff', val_loss - train_loss, global_step=epoch)
        else:
            print('training_loss: {}'.format(train_loss))
            if bigt == 0:
                print('val_loss: {}'.format(val_loss))
            print('mAP: {}'.format(val_map_score))

        if main_params['lr_decay'] == 1:
            if main_params['lrsch'] == 3:
                if epoch >= 40:
                    if es == 0:
                        lr_scheduler.step(val_loss)
                    else:
                        lr_scheduler.step(val_map_score.item())
            else:
                lr_scheduler.step()

        if early_s_pat != 0:
            if es == 0:
                if val_loss >= min_val_loss + 0.005:
                    patience -= 1
                else:
                    min_val_loss = val_loss
                    patience = early_s_pat
            else:
                if val_map_score.item() <= min_val_loss:
                    patience -= 1
                else:
                    min_val_loss = val_map_score.item()
                    patience = early_s_pat

            if patience == 0:
                break

        summary['train_loss_log'] = train_loss_log
        if bigt == 0:
            summary['val_loss_log'] = val_loss_log
        summary['val_map_log'] = val_map_log
        if bigt == 1:
            summary['test_map_log'] = test_map_log

        if main_params['save_summary'] == 1:
            if not os.path.exists('{}result_summaries_0210/'.format(rootdir)):
                os.mkdir('{}result_summaries_0210/'.format(rootdir))

            with open('{}result_summaries_0210/summary_{}_{}_{}.json'.format(rootdir, shortname, savename, config_name),
                      'w') as log:
                json.dump(summary, log, indent='\t')

        #if main_params['lr_decay'] == 1:
        #    if lr*1e-4 > optimizer.param_groups[0]['lr']:
        #        break

    end_time = time.monotonic()

    summary['notes'] = notes
    """
    if early_s_pat != 0:
        summary['train_loss_fin'] = train_loss_log[-(early_s_pat - patience + 1)]
        summary['genuine_loss_tr_fin'] = genuine_loss_tr_log[-(early_s_pat - patience + 1)]
        summary['impostor_loss_tr_fin'] = impostor_loss_tr_log[-(early_s_pat - patience + 1)]
        summary['val_loss_fin'] = val_loss_log[-(early_s_pat - patience + 1)]
        summary['genuine_loss_fin'] = genuine_loss_log[-(early_s_pat - patience + 1)]
        summary['impostor_loss_fin'] = impostor_loss_log[-(early_s_pat - patience + 1)]
        summary['val_map_fin'] = val_map_log[-(early_s_pat - patience + 1)]
    else:
        summary['train_loss_fin'] = train_loss_log[-1]
        summary['genuine_loss_tr_fin'] = genuine_loss_tr_log[-1]
        summary['impostor_loss_tr_fin'] = impostor_loss_tr_log[-1]
        summary['val_loss_fin'] = val_loss_log[-1]
        summary['genuine_loss_fin'] = genuine_loss_log[-1]
        summary['impostor_loss_fin'] = impostor_loss_log[-1]
        summary['val_map_fin'] = val_map_log[-1]
    """
    summary['last_epoch'] = last_epoch
    summary['training_time'] = end_time - start_time
    # summary['gamma'] = siamese_model.gamma.item()
    summary['margin'] = margin
    # summary['corr_coef'] = np.corrcoef(np.array(val_loss_log[:-1]), np.array(val_map_log[:-1]))[0, 1]
    summary['lr'] = lr
    summary['sel_loss'] = sel_loss
    summary['sigmoid'] = sigmoid

    summary['train_loss_log'] = train_loss_log
    if bigt == 0:
        summary['val_loss_log'] = val_loss_log
    summary['val_map_log'] = val_map_log
    if bigt == 1:
        summary['test_map_log'] = test_map_log

    if cluster == 0:
        writer.close()

    if main_params['save_summary'] == 1:
        if not os.path.exists('{}result_summaries_0210/'.format(rootdir)):
            os.mkdir('{}result_summaries_0210/'.format(rootdir))

        with open('{}result_summaries_0210/summary_{}_{}_{}.json'.format(rootdir, shortname, savename, config_name), 'w') as log:
            json.dump(summary, log, indent='\t')

    if main_params['save_model'] == 1:
        if not os.path.exists('{}saved_models_0210/'.format(rootdir)):
            os.mkdir('{}saved_models_0210/'.format(rootdir))
        torch.save(siamese_model.state_dict(), '{}saved_models_0210/model_{}_{}_{}.pt'.format(rootdir, shortname, savename, config_name))


def evaluate(rootdir,
             shortname,
             savename,
             config_name,
             conv_cfg,
             cocat,
             pl,
             st,
             bn,
             lin1,
             lin2,
             summ,
             ap,
             sigmoid,
             mulprelu,
             mulautop,
             cla,
             te,
             dataset
             ):

    print('{}_{}'.format(savename, config_name))
    if dataset == 1:
        dataset_name = 'data/benchmark_crema.h5'
    else:
        dataset_name = 'data/ytc_crema.h5'

    if cocat == 0:
        siamese_model = ConvBaseline(bn=bn, lin1=lin1, lin2=lin2, sum_method=summ, autopool_p=ap, cla=cla,
                                     use_sigmoid=sigmoid, mulprelu=mulprelu, mulautop=mulautop)
    elif cocat == 1:
        siamese_model = ConvConcat()
    elif cocat == 2:
        siamese_model = ConvRes()
    elif cocat == 3:
        siamese_model = ConvResLarge(sum_method=summ, autopool_p=ap, cla=cla)
    elif cocat == 4:
        siamese_model = ConvResLarge2(cla=cla)
    elif cocat == 5:
        siamese_model = ConvResNonM(sum_method=summ, autopool_p=ap, cla=cla)
    elif cocat == 6:
        siamese_model = ConvResNonK(cla=cla)
    elif cocat == 7:
        siamese_model = ConvBaseNonM(bn=bn, lin1=lin1, lin2=lin2, sum_method=summ, autopool_p=ap, cla=cla)
    elif cocat == 8:
        siamese_model = ConvBaselineJP(bn=bn, lin1=lin1, lin2=lin2, sum_method=summ, autopool_p=ap, cla=cla)
    elif cocat == 9:
        siamese_model = ConvBaselineJPD(bn=bn, lin1=lin1, lin2=lin2, sum_method=summ, autopool_p=ap, cla=cla)
    elif cocat == 10:
        siamese_model = ConvBaselineJP5L(bn=bn, lin1=lin1, lin2=lin2, sum_method=summ, autopool_p=ap, cla=cla,
                                         use_sigmoid=sigmoid)
    elif cocat == 11:
        siamese_model = ConvBaselineJP5LIN(bn=bn, lin1=lin1, lin2=lin2, sum_method=summ, autopool_p=ap, cla=cla,
                                           use_sigmoid=sigmoid)
    elif cocat == 12:
        siamese_model = ConvBaselineJP5LINMP(bn=bn, lin1=lin1, lin2=lin2, sum_method=summ, autopool_p=ap, cla=cla,
                                             use_sigmoid=sigmoid)
    elif cocat == 13:
        siamese_model = ConvBaseline5L(bn=bn, lin1=lin1, lin2=lin2, sum_method=summ, autopool_p=ap, cla=cla,
                                       use_sigmoid=sigmoid)
    elif cocat == 14:
        siamese_model = ConvBaseline5LD(bn=bn, lin1=lin1, lin2=lin2, sum_method=summ, autopool_p=ap, cla=cla,
                                        use_sigmoid=sigmoid)
    elif cocat == 15:
        siamese_model = ConvBaselineJP5LD(bn=bn, lin1=lin1, lin2=lin2, sum_method=summ, autopool_p=ap, cla=cla,
                                          use_sigmoid=sigmoid)
    elif cocat == 16:
        siamese_model = ConvBaseline6L(bn=bn, lin1=lin1, lin2=lin2, sum_method=summ, autopool_p=ap, cla=cla,
                                       use_sigmoid=sigmoid)
    elif cocat == 17:
        siamese_model = ConvBaselineJP6L(bn=bn, lin1=lin1, lin2=lin2, sum_method=summ, autopool_p=ap, cla=cla,
                                         use_sigmoid=sigmoid)
    elif cocat == 18:
        siamese_model = ConvBaselineAJP5LD(bn=bn, lin1=lin1, lin2=lin2, sum_method=summ, autopool_p=ap, cla=cla,
                                           use_sigmoid=sigmoid)
    elif cocat == 19:
        siamese_model = ConvBaselineAJP5LND(bn=bn, lin1=lin1, lin2=lin2, sum_method=summ, autopool_p=ap, cla=cla,
                                            use_sigmoid=sigmoid)
    elif cocat == 20:
        siamese_model = ConvBaselineJP5LSD(bn=bn, lin1=lin1, lin2=lin2, sum_method=summ, autopool_p=ap, cla=cla,
                                           use_sigmoid=sigmoid)
    model_name = '{}saved_models_0210/model_{}_{}_{}'.format(rootdir, shortname, savename, config_name)
    if te != 0:
        model_name = '{}_e{}.pt'.format(model_name, te)
    else:
        model_name = '{}.pt'.format(model_name)
    siamese_model.load_state_dict(torch.load(model_name, map_location='cpu'))
    siamese_model.eval()

    if torch.cuda.is_available():
        siamese_model.cuda()

    test_data, test_labels = import_dataset_from_h5('{}{}'.format(rootdir, dataset_name), benchmark=1)

    #test_data, test_labels = import_large_dataset_from_json('{}{}'.format(rootdir, 'data/large_benchmark.json'))
    val_map_set = SingleFullSize(test_data, test_labels, idx=None, single_files=0)

    val_map_loader = DataLoader(val_map_set, batch_size=1, shuffle=False)
    start = time.monotonic()
    dist_map_tensor = test(siamese_model, val_map_loader)
    print('{:.3f} secs'.format(time.monotonic() - start))
    dist_map_matrix = square_dist_tensor(dist_map_tensor)

    val_map_score = average_precision(
        -1 * dist_map_matrix.clone() + torch.diag(torch.ones(len(test_data)) * float('-inf')), benchmark=dataset)
    #print('mAP score: {}'.format(val_map_score.item()))


if __name__:
    parser = argparse.ArgumentParser(description='Training and testing with Siamese Network')
    parser.add_argument('-p',
                        '--params',
                        type=str,
                        default='config_file_35',
                        help='Name of the config file. Must be in config folder')
    parser.add_argument('-s',
                        '--shortname',
                        type=str,
                        default='abs_framework',
                        help='Short name of the experiments for saving purposes')
    parser.add_argument('-n',
                        '--notes',
                        type=str,
                        default='',
                        help='short note for saving purposes')
    parser.add_argument('-r',
                        '--rootdir',
                        type=str,
                        default='',
                        help='rootdir for colab experiments')
    parser.add_argument('-sn',
                        '--savename',
                        type=str,
                        default='hpc_tmr32_vmr32_70e_mix_base_fullen_em4096_bh30_semihard_daug_autop0_margin_1h_6',
                        help='short description for saving purposes')
    parser.add_argument('-cc',
                        '--convcfg',
                        type=int,
                        default=6,
                        help='conv config')
    parser.add_argument('-m',
                        '--margin',
                        type=float,
                        default=1.0,
                        help='margin for loss')
    parser.add_argument('-l',
                        '--sel_loss',
                        type=int,
                        choices=(0, 1, 2),
                        default=1,
                        help='select loss: contrastive pair 0, triplet 1, lossless triplet 2')
    parser.add_argument('-sg',
                        '--sigmoid',
                        type=int,
                        choices=(0, 1, 2, 3),
                        default=3,
                        help='1 for using sigmoid on the final embedding, 0 for else')
    parser.add_argument('-flr',
                        '--findlr',
                        type=int,
                        choices=(0, 1),
                        default=0,
                        help='1 for using find learning rate, 0 for else')
    parser.add_argument('-tm',
                        '--tripletmining',
                        type=int,
                        choices=(0, 1),
                        default=1,
                        help='1 for using triplet mining, 0 for else')
    parser.add_argument('-pl',
                        '--patchlen',
                        type=int,
                        default=1800,
                        help='size of the input len in w dim')
    parser.add_argument('-o',
                        '--optim',
                        type=int,
                        choices=(0, 1),
                        default=0,
                        help='0 for sgd, 1 for adam')
    parser.add_argument('-tmb',
                        '--tm_batch',
                        type=int,
                        default=16,
                        help='number of cliques per batch for triplet mining')
    parser.add_argument('-cat',
                        '--conv_concat',
                        type=int,
                        default=10,
                        help='0 for ConvX, 1 for ConvConcat, 2 for ConvRes')
    parser.add_argument('-st',
                        '--stretch',
                        type=int,
                        choices=(0, 1),
                        default=0,
                        help='0 for zeropadding for fixed size, 1 for stretching')
    parser.add_argument('-es',
                        '--earlystop',
                        type=int,
                        choices=(0, 1),
                        default=0,
                        help='0 for val_loss for early_stop, 1 for mAP')
    parser.add_argument('-pltr',
                        '--patchlentriplet',
                        type=int,
                        default=1800,
                        help='size of the input len in w dim for triplets')
    parser.add_argument('-dr',
                        '--dropout',
                        type=float,
                        default=0,
                        help='dropout value')
    parser.add_argument('-lin1',
                        '--linear1',
                        type=int,
                        default=512,
                        help='size of first linear layer')
    parser.add_argument('-lin2',
                        '--linear2',
                        type=int,
                        default=0,
                        help='size of second linear layer, 0 for not using')
    parser.add_argument('-bn',
                        '--batchnorm',
                        type=int,
                        choices=(0, 1),
                        default=0,
                        help='0 for no batch norm, 1 using it')
    parser.add_argument('-da',
                        '--dataaug',
                        type=int,
                        choices=(0, 1, 2),
                        default=1,
                        help='0 for no data aug, 1 using it, 2 for adding term to loss')
    parser.add_argument('-test',
                        '--test',
                        type=int,
                        choices=(0, 1, 2),
                        default=0,
                        help='0 for training, 1 for test on da-tacos, 2 for test on ytc')
    parser.add_argument('-summ',
                        '--summethod',
                        type=int,
                        choices=(0, 1, 2),
                        default=2,
                        help='0 for max-pool, 1 for mean-pool, 2 for autopool')
    parser.add_argument('-ap',
                        '--autopoolp',
                        type=float,
                        default=0,
                        help='default value for autopool parameter')
    parser.add_argument('-vm',
                        '--valmining',
                        type=int,
                        choices=(0, 1),
                        default=1,
                        help='0 for random val triplets, 1 for rand mining')
    parser.add_argument('-twom',
                        '--twomining',
                        type=int,
                        default=50,
                        help='0 for just rand mining, int for which epoch for switching to batch hard')
    parser.add_argument('-lt',
                        '--loss_t',
                        type=int,
                        choices=(0, 1),
                        default=0,
                        help='0 for euclidean, 1 for cosine')
    parser.add_argument('-cla',
                        '--classification',
                        type=int,
                        choices=(0, 1, 2),
                        default=0,
                        help='0 for triplet, 1 for classification, 2 for mix')
    parser.add_argument('-cluster',
                        '--cluster',
                        type=int,
                        choices=(0, 1),
                        default=0,
                        help='0 for non cluster, 1 for cluster')
    parser.add_argument('-changel',
                        '--changel',
                        type=int,
                        choices=(0, 1),
                        default=0,
                        help='0 for normal, 1 for changing loss after 30 epochs')
    parser.add_argument('-mullen',
                        '--mullen',
                        type=int,
                        choices=(0, 1),
                        default=0,
                        help='0 for 1 length, 1 for 3 length training')
    parser.add_argument('-fullen',
                        '--fullen',
                        type=int,
                        choices=(0, 1),
                        default=1,
                        help='0 for fixed length val map, 1 for full length')
    parser.add_argument('-allpos',
                        '--allpos',
                        type=int,
                        choices=(0, 1),
                        default=0,
                        help='0 for using random pos, 1 for all positive pairs')
    parser.add_argument('-threem',
                        '--threem',
                        type=int,
                        choices=(0, 1),
                        default=0,
                        help='0 for not three m, 1 for using it')
    parser.add_argument('-bigt',
                        '--bigt',
                        type=int,
                        choices=(0, 1),
                        default=0,
                        help='0 for separate train and val sets, 1 for combining it')
    parser.add_argument('-mulp',
                        '--mulprelu',
                        type=int,
                        choices=(0, 1),
                        default=0,
                        help='0 for 1 parameter, 1 for multiple')
    parser.add_argument('-mula',
                        '--mulautop',
                        type=int,
                        choices=(0, 1),
                        default=0,
                        help='0 for 1 parameter, 1 for multiple')
    parser.add_argument('-mask',
                        '--mask',
                        type=int,
                        choices=(0, 1),
                        default=0,
                        help='0 for not using masking, 1 for using it')
    parser.add_argument('-nd',
                        '--norm_dist',
                        type=int,
                        choices=(0, 1),
                        default=1,
                        help='0 for not normal, 1 for normalizing the distance')
    parser.add_argument('-lr',
                        '--learning_rate',
                        type=float,
                        default=1e-1,
                        help='initial learning rate')
    parser.add_argument('-nw',
                        '--n_workers',
                        type=int,
                        default=0,
                        help='number of workers')
    parser.add_argument('-rl',
                        '--rand_length',
                        type=int,
                        default=0,
                        help='0 for fixed length training, 1 for random length')
    parser.add_argument('-ch',
                        '--chunks',
                        type=int,
                        default=6,
                        help='number of chunks for training set')
    parser.add_argument('-ud',
                        '--uni_dist',
                        type=int,
                        default=0,
                        help='0 for uni_distribution of cliques, 1 for weighted')
    parser.add_argument('-ms',
                        '--mining_strategy',
                        type=int,
                        default=3,
                        help='0 for only random, 1 for only semi-hard, 2 for only hard, 3 for semi-hard to hard')
    parser.add_argument('-te',
                        '--test_epoch',
                        type=int,
                        default=0,
                        help='0 for without suffix, int for epoch number')

    args = parser.parse_args()

    with open('{}config/{}_{}.json'.format(args.rootdir, args.shortname, args.params)) as f:
        params = json.load(f)

    for key1 in params.keys():
        for key2 in params[key1].keys():
            if type(params[key1][key2]) == list:
                params[key1][key2] = tuple(params[key1][key2])
    """
    conv_cfg = {
        1: [(128, (12, 50)), 'P', (128, (1, 5)), (128, (1, 5)), (128, (1, 5))],
        2: [(128, (12, 50)), 'P', (128, (1, 5)), 'MP', (128, (1, 5)), 'MP', (128, (1, 5))],
        3: [(128, (12, 50)), 'P', (128, (1, 5)), (96, (1, 5)), (64, (1, 5))],
        4: [(128, (12, 50)), 'P', (128, (1, 5)), 'MP', (96, (1, 5)), 'MP', (64, (1, 5))],
        5: [(128, (12, 50)), 'P', (128, (1, 5)), (96, (1, 5), (1, 20)), (64, (1, 5))],
        6: [(128, (12, 50)), 'P', (128, (1, 5)), 'MP', (96, (1, 5), (1, 20)), 'MP', (64, (1, 5))],
        7: [(128, (12, 50)), 'P', (128, (1, 5)), (128, (1, 5), (1, 20)), (128, (1, 5))],
        8: [(128, (12, 50)), 'P', (128, (1, 5)), 'MP', (128, (1, 5), (1, 20)), 'MP', (128, (1, 5))],
    }
    """
    conv_cfg = {
        1: [(128, (12, 50)), 'P', (128, (1, 5))],
        2: [(128, (12, 30)), 'P', (128, (1, 7))],
        3: [(128, (12, 50)), 'P', (128, (1, 5)), (128, (1, 5)), (128, (1, 5))],
        4: [(128, (12, 50)), 'P', (128, (1, 5)), 'MP', (128, (1, 5)), 'MP', (128, (1, 5))],
        5: [(128, (12, 50)), 'P', (128, (1, 5)), (128, (1, 5), (1, 20)), (128, (1, 5))],
        6: [(256, (12, 50)), 'P', (256, (1, 5)), (256, (1, 5), (1, 20)), (256, (1, 5))],
        7: [(128, (12, 50)), 'P', (128, (1, 5)), (128, (1, 5), (1, 26)), (128, (1, 5))],
        8: [(128, (12, 50)), 'P', (128, (1, 5)), (128, (1, 5)), (128, (1, 5), (1, 28)), (128, (1, 5))],
        9: [(128, (12, 50)), 'P', (128, (1, 5)), (128, (1, 5), (1, 20)), (128, (1, 5)), (128, (1, 3), (1, 68))],
        10: [(128, (12, 50)), 'P', (128, (1, 5)), (128, (1, 5), (1, 20)), (128, (1, 5)), (128, (1, 3), (1, 68)),
             (128, (1, 3))],
        11: [(512, (12, 50)), 'P', (512, (1, 5)), (512, (1, 5), (1, 20)), (512, (1, 5))],
        12: [(128, (12, 50)), 'P', (128, (1, 5)), 'BN', (128, (1, 5), (1, 20)), 'BN', (128, (1, 5)), 'BN'],
        13: [(128, (12, 50)), 'P', (128, (1, 5)), 'MP', (128, (1, 5), (1, 20)), 'MP', (128, (1, 5))],
        14: [(256, (12, 50)), 'P', (256, (1, 5)), 'MP', (256, (1, 5), (1, 20)), 'MP', (256, (1, 5))],
    }

    lr_arg = '{}'.format(args.learning_rate).replace('.', '-')
    margin_arg = '{}'.format(args.margin).replace('.', '-')

    savename = 'em{}_tmb{}_lr{}_twom{}_m{}_cat{}_ch{}_ms{}_nw{}'.format(args.linear1, args.tm_batch, lr_arg,
                                                                        args.twomining, margin_arg, args.conv_concat,
                                                                        args.chunks, args.mining_strategy,
                                                                        args.n_workers)

    baseline_models = [0, 13]

    if args.sigmoid != 1:
        savename = '{}_sg{}'.format(savename, args.sigmoid)
    if args.conv_concat in baseline_models:
        savename = '{}_summ{}'.format(savename, args.summethod)
    if args.rand_length == 1:
        savename = '{}_rl_w{}'.format(savename, args.patchlentriplet)
    if args.mullen == 1:
        savename = '{}_mullen'.format(savename)
    if args.uni_dist == 1:
        savename = '{}_noud'.format(savename)
    if args.norm_dist == 1:
        savename = '{}_nd'.format(savename)
    if args.cluster == 1:
        savename = '{}_cluster'.format(savename)

    if args.test == 0:
        main(main_params=params['main_params'],
             config_name=args.params,
             shortname=args.shortname,
             notes=args.notes,
             savename=savename,
             rootdir=args.rootdir,
             conv_cfg=conv_cfg[args.convcfg],
             margin=args.margin,
             sel_loss=args.sel_loss,
             sigmoid=args.sigmoid,
             flr=args.findlr,
             tm=args.tripletmining,
             pl=args.patchlen,
             o=args.optim,
             tmb=args.tm_batch,
             cocat=args.conv_concat,
             st=args.stretch,
             es=args.earlystop,
             pltr=args.patchlentriplet,
             dr=args.dropout,
             lin1=args.linear1,
             lin2=args.linear2,
             bn=args.batchnorm,
             da=args.dataaug,
             summ=args.summethod,
             ap=args.autopoolp,
             val_mining=args.valmining,
             twom=args.twomining,
             lt=args.loss_t,
             cla=args.classification,
             cluster=args.cluster,
             changel=args.changel,
             mullen=args.mullen,
             fullen=args.fullen,
             allpos=args.allpos,
             threem=args.threem,
             bigt=args.bigt,
             mulprelu=args.mulprelu,
             mulautop=args.mulautop,
             masking=args.mask,
             norm_dist=args.norm_dist,
             lr=args.learning_rate,
             nw=args.n_workers,
             rl=args.rand_length,
             chunks=args.chunks,
             ud=args.uni_dist,
             ms=args.mining_strategy
             )
    else:
        evaluate(rootdir=args.rootdir,
                 shortname=args.shortname,
                 savename=savename,
                 config_name=args.params,
                 conv_cfg=conv_cfg[args.convcfg],
                 pl=args.patchlen,
                 st=args.stretch,
                 bn=args.batchnorm,
                 lin1=args.linear1,
                 lin2=args.linear2,
                 summ=args.summethod,
                 ap=args.autopoolp,
                 cocat=args.conv_concat,
                 sigmoid=args.sigmoid,
                 mulprelu=args.mulprelu,
                 mulautop=args.mulautop,
                 cla=args.classification,
                 te=args.test_epoch,
                 dataset=args.test
                 )

