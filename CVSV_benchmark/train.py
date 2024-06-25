import torch
import argparse
import random
import numpy as np
import os
import time
from sklearn.manifold import TSNE
from configs.defaults import get_cfg_defaults
from dataset.dataset import load_dataset
from utils.logger import setup_logger
from models.model import CAT
from models.clip_model2 import baseline as UTSV
from utils.preprocess import frames_preprocess
from torch.utils.tensorboard import SummaryWriter
from utils.loss import compute_cls_loss, compute_seq_loss
import utils
from utils.metrics import compute_WDR, pred_dist
from tqdm import tqdm
from sklearn.metrics import auc, roc_curve
import torch.nn.functional as F
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

def train():
    MAX_SEQ_LENGTH = 6
    
    if cfg.MODEL.BACKBONE == 'cat':
        model = CAT(num_class=cfg.DATASET.NUM_CLASS,
                    num_clip=cfg.DATASET.NUM_CLIP,
                    dim_embedding=cfg.MODEL.DIM_EMBEDDING,
                    pretrain=cfg.MODEL.PRETRAIN,
                    dropout=cfg.TRAIN.DROPOUT,
                    use_TE=cfg.MODEL.TRANSFORMER,
                    use_SeqAlign=cfg.MODEL.ALIGNMENT,
                    freeze_backbone=cfg.TRAIN.FREEZE_BACKBONE).to(device)
    elif cfg.MODEL.BACKBONE == 'utsv':
        model = UTSV(model_log=False,
                    num_class=cfg.DATASET.NUM_CLASS,
                    num_clip=cfg.DATASET.NUM_CLIP,
                    dim_embedding=cfg.MODEL.DIM_EMBEDDING,
                    pretrain=cfg.MODEL.PRETRAIN,
                    dropout=cfg.TRAIN.DROPOUT,
                    feature_encoder='clip',
                    use_TE=cfg.MODEL.TRANSFORMER,
                    use_text=args.use_txt,
                    use_gumbel=args.use_gumbel,
                    use_SeqAlign=cfg.MODEL.ALIGNMENT,
                    freeze_backbone=cfg.TRAIN.FREEZE_BACKBONE).to(device)
    
    
    for name, param in model.named_parameters():
        print(name, param.nelement())
    logger.info('Model have {} paramerters in total'.format(sum(x.numel() for x in model.parameters())))
    logger.info('Model have {} paramerters to train'.format(sum(x.numel() for x in model.parameters() if x.requires_grad)))
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.TRAIN.MAX_EPOCH, eta_min=cfg.TRAIN.LR * 0.01)

    # Load checkpoint
    start_epoch = 0
    if args.load_path and os.path.isfile(args.load_path):
        checkpoint = torch.load(args.load_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        logger.info('-> Loaded checkpoint %s (epoch: %d)' % (args.load_path, start_epoch))

    # Mulitple gpu
    if torch.cuda.device_count() > 1 and torch.cuda.is_available():
        logger.info('Let us use %d GPUs' % torch.cuda.device_count())
        model = torch.nn.DataParallel(model)

    model.train()

    # Create Tensorboard writer
    writer = SummaryWriter(os.path.join(cfg.TRAIN.SAVE_PATH, 'tensorboard'))

    # Create checkpoint dir
    if cfg.TRAIN.SAVE_PATH:
        checkpoint_dir = os.path.join(cfg.TRAIN.SAVE_PATH, 'save_models')
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)


    # Start training
    start_time = time.time()
    for epoch in range(start_epoch, cfg.TRAIN.MAX_EPOCH):
        loss_per_epoch = 0
        num_true_pred = 0
        num_samples = 0
        model.train()
        for iter, sample in enumerate(train_loader):
            if args.pair:
                frames1 = sample['clips1'][0].to(device)    # [B, 3, 16, 224, 224]
                frames2 = sample['clips2'][0].to(device)
                labels1 = sample['labels1'].to(device)
                labels2 = sample['labels2'].to(device)

                # language data
                if args.use_txt:
                    label_token1 = sample['label_token1'].to(device)
                    label_token_phrase1 = sample['label_token_phrase1'][0].to(device)
                    label_phrase_num1 = sample['label_token_phrase1'][1].to(device)
                    label_token2 = sample['label_token2'].to(device)
                    label_token_phrase2 = sample['label_token_phrase2'][0].to(device)
                    label_phrase_num2 = sample['label_token_phrase2'][1].to(device)
                    label_neg_token1 = sample['label_neg_token1'].to(device)
                    label_neg_token2 = sample['label_neg_token2'].to(device)
                
                if args.use_txt:
                    pred1, text_feature1, seq_features1, embed1, logit_scale1_1, all_text_feature1, text_feature_phrase1, logit_scale1_2 = model(frames1,
                                                                                                            label_token1, label_token_phrase1, label_neg_token1)
                    pred2, text_feature2, seq_features2, embed2, logit_scale2_1, all_text_feature2, text_feature_phrase2, logit_scale2_2 = model(frames2,
                                                                                                            label_token2, label_token_phrase2, label_neg_token2)
                else:
                    pred1, seq_features1 = model(frames1)
                    pred2, seq_features2 = model(frames2)
                # compute losses
                if args.use_txt:
                    if args.use_neg_txt:
                        loss_info1 = utils.loss.compute_info_loss_neg(embed1, text_feature1, all_text_feature1, logit_scale1_1,)
                        loss_info2 = utils.loss.compute_info_loss_neg(embed2, text_feature2, all_text_feature2, logit_scale2_1,)
                    else:
                        loss_info1 = utils.loss.compute_info_loss(embed1, text_feature1, labels1, logit_scale1_1)
                        loss_info2 = utils.loss.compute_info_loss(embed2, text_feature2, labels2, logit_scale2_1)
                    loss_info = loss_info1 + loss_info2
                else:
                    loss_info = torch.zeros([1]).to(device)

                if args.use_gumbel:
                    loss_gumbel1 = utils.loss.compute_gumbel_loss(seq_features1, text_feature_phrase1, label_phrase_num1, logit_scale1_2, max_seq_length=MAX_SEQ_LENGTH, gt_type=args.gt_type,labels1=sample['labels1_raw'])
                    loss_gumbel2 = utils.loss.compute_gumbel_loss(seq_features2, text_feature_phrase2, label_phrase_num2, logit_scale1_2, max_seq_length=MAX_SEQ_LENGTH, gt_type=args.gt_type,labels1=sample['labels2_raw'])
                    loss_gumbel = loss_gumbel1 + loss_gumbel2
                else:
                    loss_gumbel = torch.zeros([1]).to(device)
                    
                if cfg.TRAIN.WEAK_SUPERVISION:
                    loss_cls = torch.zeros([1]).to(device)
                else:
                    loss_cls = utils.loss.compute_cls_loss(pred1, labels1) + utils.loss.compute_cls_loss(pred2, labels2)
                loss_seq = utils.loss.compute_seq_loss(seq_features1, seq_features2)
            
            else:
                frames1 = sample['clips1'][0].to(device)    # [B, 3, 16, 224, 224]
                labels1 = sample['labels1'].to(device)
                
                # language data
                label_token1 = sample['label_token1'].to(device)
                label_token_phrase1 = sample['label_token_phrase1'][0].to(device)
                label_phrase_num1 = sample['label_token_phrase1'][1].to(device)
                label_neg_token1 = sample['label_neg_token1'].to(device)
                
                if args.use_txt:
                    pred1, text_feature1, seq_features1, embed1, logit_scale1_1, all_text_feature1, text_feature_phrase1, logit_scale1_2 = model(frames1,
                                                                                                            label_token1, label_token_phrase1, label_neg_token1)
                else:
                    pred1, seq_features1 = model(frames1)
                
                # compute losses
                if args.use_txt:
                    if args.use_neg_txt:
                        loss_info1 = utils.loss.compute_info_loss_neg(embed1, text_feature1, all_text_feature1, logit_scale1_1,)
                    else:
                        loss_info1 = utils.loss.compute_info_loss(embed1, text_feature1, labels1, logit_scale1_1)
                    loss_info = loss_info1
                else:
                    loss_info = torch.zeros([1]).to(device)

                if args.use_gumbel:
                    loss_gumbel1 = utils.loss.compute_gumbel_loss(seq_features1, text_feature_phrase1, label_phrase_num1, logit_scale1_2,max_seq_length=MAX_SEQ_LENGTH, gt_type=args.gt_type,labels1=sample['labels1_raw'])
                    loss_gumbel = loss_gumbel1
                else:
                    loss_gumbel = torch.zeros([1]).to(device)
                    
                if cfg.TRAIN.WEAK_SUPERVISION:
                    loss_cls = torch.zeros([1]).to(device)
                else:
                    loss_cls = utils.loss.compute_cls_loss(pred1, labels1)
                loss_seq = torch.zeros([1]).to(device) 
            
            loss = loss_cls + cfg.MODEL.SEQ_LOSS_COEF * loss_seq + cfg.MODEL.INFO_LOSS_COEF * loss_info \
                           + cfg.MODEL.GUMBEL_LOSS_COEF * loss_gumbel
            # loss = cfg.MODEL.INFO_LOSS_COEF * loss_info

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
                        
            # assert 1==2
            if (iter + 1) % 10 == 0:
                logger.info( 'Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Loss_cls: {:.4f}, Loss_seq: {:.4f}, Loss_info: {:.4f}, Loss_gumbel: {:.4f}'.format(epoch + 1, cfg.TRAIN.MAX_EPOCH, iter + 1, len(train_loader), 
                                                                                                                                          loss.item(), loss_cls.item(),(cfg.MODEL.SEQ_LOSS_COEF * loss_seq).item(),
                                                                                                                                          (cfg.MODEL.INFO_LOSS_COEF * loss_info).item(), (cfg.MODEL.GUMBEL_LOSS_COEF * loss_gumbel).item()))

            loss_per_epoch += loss.item()
            num_true_pred += torch.sum(torch.argmax(pred1, dim=-1) == labels1) + torch.sum(torch.argmax(pred2, dim=-1) == labels2)
            num_samples += pred1.shape[0] + pred2.shape[0]
            # Update weights
            

        # Log training statistics
        loss_per_epoch /= (iter + 1)
        # accuracy = num_true_pred / (cfg.DATASET.NUM_SAMPLE * 2)
        accuracy = num_true_pred / num_samples
        logger.info('Epoch [{}/{}], LR: {:.6f}, Accuracy: {:.4f}, Loss: {:.4f}'
                    .format(epoch + 1, cfg.TRAIN.MAX_EPOCH, optimizer.param_groups[0]['lr'], accuracy, loss_per_epoch))
        
        if epoch ==0 or (epoch + 1) % 10 == 0:
            iter_test = tqdm(test_loader, leave=False)
            # auc, wdr, intra_view_auc, inter_view_auc, ego2ego_view_auc, exo2exo_view_auc = eval_per_epoch(model, iter_test, eval_cfg, args, epoch=epoch)
            results = eval_per_epoch(model, iter_test, eval_cfg, args, epoch=epoch)
            auc, wdr, intra_view_auc, inter_view_auc, ego2ego_view_auc, exo2exo_view_auc = results['auc'], results['wdr'], results['intra_view_auc'], results['inter_view_auc'], results['ego2ego_view_auc'], results['exo2exo_view_auc']
            r1_retrieval, r5_retrieval, ego_r1_retrieval, ego_r5_retrieval, exo_r1_retrieval, exo_r5_retrieval = results['r1_retrieval'], results['r5_retrieval'], results['ego_r1_retrieval'], results['ego_r5_retrieval'], results['exo_r1_retrieval'], results['exo_r5_retrieval']
            mAP, ego_mAP, exo_mAP = results['mAP'], results['ego_mAP'], results['exo_mAP']
            exo2ego_r1_retrieval , ego2exo_r1_retrieval = results['exo2ego_r1_retrieval'], results['ego2exo_r1_retrieval']
            exo2ego_r5_retrieval , ego2exo_r5_retrieval = results['exo2ego_r5_retrieval'], results['ego2exo_r5_retrieval']
            exo2ego_mAP, ego2exo_mAP = results['exo2ego_mAP'], results['ego2exo_mAP']
            # logger.info('Epoch [{}/{}], AUC is {:.4f}, wdr is {:.4f}, intra_view_auc is {:.4f}, inter_view_auc is {:.4f}, ego2ego_view_auc is {:.4f}, exo2exo_view_auc is {:.4f}.'.format(epoch + 1, cfg.TRAIN.MAX_EPOCH, auc, wdr, intra_view_auc, inter_view_auc, ego2ego_view_auc, exo2exo_view_auc))
            
            logger.info('Epoch [%d/%d], AUC is %.4f, intra_view_auc is %.4f, inter_view_auc is %.4f, ego2ego_view_auc is %.4f, exo2exo_view_auc is %.4f' % (epoch + 1, cfg.TRAIN.MAX_EPOCH, auc, intra_view_auc, inter_view_auc, ego2ego_view_auc, exo2exo_view_auc))
            logger.info('Epoch [%d/%d], R1_retrieval is %.4f, R1_retrieval_ego is %.4f, R1_retrieval_exo is %.4f, R1_retrieval_exo2ego is %.4f, R1_retrieval_ego2exo is %.4f' % (epoch + 1, cfg.TRAIN.MAX_EPOCH, r1_retrieval, ego_r1_retrieval, exo_r1_retrieval, exo2ego_r1_retrieval , ego2exo_r1_retrieval))
            logger.info('Epoch [%d/%d], R5_retrieval is %.4f, R5_retrieval_ego is %.4f, R5_retrieval_exo is %.4f, R5_retrieval_exo2ego is %.4f, R5_retrieval_ego2exo is %.4f' % (epoch + 1, cfg.TRAIN.MAX_EPOCH, r5_retrieval, ego_r5_retrieval, exo_r5_retrieval, exo2ego_r5_retrieval , ego2exo_r5_retrieval))
            logger.info('Epoch [%d/%d], mAP is %.4f, mAP_ego is %.4f, mAP_exo is %.4f, mAP_exo2ego is %.4f, mAP_ego2exo is %.4f' % (epoch + 1, cfg.TRAIN.MAX_EPOCH, mAP, ego_mAP, exo_mAP, exo2ego_mAP, ego2exo_mAP))
            
            writer.add_scalar('AUC/test', auc, epoch)
            writer.add_scalar('WDR/test', wdr, epoch)
            writer.add_scalar('AUC/intra_view/test', intra_view_auc, epoch)
            writer.add_scalar('AUC/inter_view/test', inter_view_auc, epoch)
            writer.add_scalar('AUC/ego2ego/test', ego2ego_view_auc, epoch)
            writer.add_scalar('AUC/exo2exo/test', exo2exo_view_auc, epoch)
            writer.add_scalar('R1_Retrieval/ego2ego/test', ego_r1_retrieval, epoch)
            writer.add_scalar('R1_Retrieval/exo2exo/test', exo_r1_retrieval, epoch)
            writer.add_scalar('R1_Retrieval/ego2exo/test', ego2exo_r1_retrieval, epoch)
            writer.add_scalar('R1_Retrieval/exo2ego/test', exo2ego_r1_retrieval, epoch)
            writer.add_scalar('R1_Retrieval/all_view/test', r1_retrieval, epoch)
            writer.add_scalar('R5_Retrieval/ego2ego/test', ego_r5_retrieval, epoch)
            writer.add_scalar('R5_Retrieval/exo2exo/test', exo_r5_retrieval, epoch)
            writer.add_scalar('R5_Retrieval/ego2exo/test', ego2exo_r5_retrieval, epoch)
            writer.add_scalar('R5_Retrieval/exo2ego/test', exo2ego_r5_retrieval, epoch)
            writer.add_scalar('R5_Retrieval/all_view/test', r5_retrieval, epoch)
            writer.add_scalar('mAP/ego2ego/test', ego_mAP, epoch)
            writer.add_scalar('mAP/exo2exo/test', exo_mAP, epoch)
            writer.add_scalar('mAP/ego2exo/test', ego2exo_mAP, epoch)
            writer.add_scalar('mAP/exo2ego/test', exo2ego_mAP, epoch)
            writer.add_scalar('mAP/all_view/test', mAP, epoch)


        LR = optimizer.param_groups[0]['lr']
        writer.add_scalar('LR', LR, epoch)
        writer.add_scalar('Accuracy', accuracy, epoch)
        writer.add_scalar('Loss_per_epoch', loss_per_epoch, epoch)
        

        # Save model every X epochs
        if (epoch + 1) % cfg.MODEL.SAVE_EPOCHS == 0:
            save_dict = {'epoch': epoch + 1,  # after training one epoch, the start_epoch should be epoch+1
                         'optimizer_state_dict': optimizer.state_dict(),
                         'loss': loss.item(),
                         }
            try:  # with nn.DataParallel() the net is added as a submodule of DataParallel
                save_dict['model_state_dict'] = model.module.state_dict()
            except:
                save_dict['model_state_dict'] = model.state_dict()

            save_name = 'epoch_' + str(epoch + 1) + '.tar'
            torch.save(save_dict, os.path.join(checkpoint_dir, save_name))
            logger.info('Save ' + os.path.join(checkpoint_dir, save_name) + ' done!')

        # Learning rate decay
        scheduler.step()

    end_time = time.time()
    duration = end_time - start_time

    hour = duration // 3600
    min = (duration % 3600) // 60
    sec = duration % 60

    logger.info('Training cost %dh%dm%ds' % (hour, min, sec))

def retreival_metrics(feature_dict, label_dict):
    feature_list = torch.cat([f.reshape(1, -1) for k, f in feature_dict.items()], dim=0).cuda()    # N, d
    n = feature_list.shape[0]
    feature_list = feature_list.unsqueeze(1).repeat(1, n, 1)        # N, N, d
    sim_matrix = torch.sum((feature_list - feature_list.transpose(0, 1)) ** 2, dim=-1)
    label_list = torch.cat([l.reshape(1, -1) for k ,l in label_dict.items()], dim=0).cuda()        # N, 1
    label_list = label_list.repeat(1, n)        # N, N
    label_matrix = (label_list == label_list.transpose(0,1)).int()  # N, N
    # Retreival Metrics
    sim_matrix.fill_diagonal_(float('inf')) # 将对角线的值设置为无穷
    sim_matrix = -1 * sim_matrix    # 因为之前算的是距离，因此要取反。取反后越大表示相似度越高
    label_matrix.fill_diagonal_(0) # 不考虑自己跟自己的关系
    
    # == Rank 5 检索 ==
    _, top_sim_idx = torch.topk(sim_matrix, k=5, dim=1) # top_sim_idx: N, 5     # 获取每一行top 5元素的索引
    rank_5_retrieval = torch.mean(((torch.sum(label_matrix.float()[torch.arange(n).unsqueeze(1), top_sim_idx], dim=1) / 5) > 0).float())    # 根据top 1索引，对label矩阵的每一行进行求和，然后取平均值
    # == Rank 1 检索 ==
    _, top_sim_idx = torch.topk(sim_matrix, k=1, dim=1) # top_sim_idx: N, 1     # 获取每一行top 1元素的索引
    rank_1_retrieval = torch.mean(torch.sum(label_matrix.float()[torch.arange(n).unsqueeze(1), top_sim_idx], dim=1))    # 根据top 1索引，对label矩阵的每一行进行求和，然后取平均值
    # == mAP == 
    sorted_indices = torch.argsort(sim_matrix, dim=1)   # 从小到大排序的索引
    reverse_sorted_indices = torch.flip(sorted_indices, dims=[1])   # 从大到小排序的索引
    sorted_label_matrix = torch.gather(label_matrix, dim=1, index=reverse_sorted_indices)
    cumulative_sum_matrix = torch.cumsum(sorted_label_matrix, dim=1)    # 计算前项和
    cumulative_sum_matrix = cumulative_sum_matrix * sorted_label_matrix # 过滤非匹配位置的值
    weight_matrix = 1.0 / torch.arange(1, n+1, dtype=torch.float).unsqueeze(0) # 1* n
    weight_matrix = weight_matrix.cuda()
    APs = torch.matmul(cumulative_sum_matrix.float(), weight_matrix.T).reshape(-1) / torch.sum(label_matrix.float(), dim=1)     # mAP
    nan_indices = torch.isnan(APs)
    # 从一维张量中去除NaN值
    mAP = torch.mean(APs[~nan_indices])

    return rank_1_retrieval, rank_5_retrieval, mAP, sim_matrix, label_matrix

def cross_view_retreival_metrics(ego_feature_dict, ego_label_dict, exo_feature_dict, exo_label_dict):
    ego_feature_list = torch.cat([f.reshape(1, -1) for k, f in ego_feature_dict.items()], dim=0).cuda()    # n, d
    exo_feature_list = torch.cat([f.reshape(1, -1) for k, f in exo_feature_dict.items()], dim=0).cuda()    # m, d
    n = ego_feature_list.shape[0]
    m = exo_feature_list.shape[0]
    ego_feature_list = ego_feature_list.unsqueeze(1).repeat(1, m, 1)        # N, M, d
    exo_feature_list = exo_feature_list.unsqueeze(1).repeat(1, n, 1)        # M, N, d
    sim_matrix = torch.sum((ego_feature_list - exo_feature_list.transpose(0, 1)) ** 2, dim=-1)  # N, M, D
    
    ego_label_list = torch.cat([l.reshape(1, -1) for k ,l in ego_label_dict.items()], dim=0).cuda()        # N, 1
    exo_label_list = torch.cat([l.reshape(1, -1) for k ,l in exo_label_dict.items()], dim=0).cuda()        # M, 1
    ego_label_list = ego_label_list.repeat(1, m)        # N, M
    exo_label_list = exo_label_list.repeat(1, n)        # M, N
    label_matrix = (ego_label_list == exo_label_list.transpose(0,1)).int()  # N，M
    # Retreival Metrics
    sim_matrix = -1 * sim_matrix    # 因为之前算的是距离，因此要取反。取反后越大表示相似度越高
    sim_matrix_T = sim_matrix.transpose(0,1)
    label_matrix_T = label_matrix.transpose(0,1)
    # == Rank 5 检索 ==
    _, top_sim_idx = torch.topk(sim_matrix, k=5, dim=1) # top_sim_idx: N, 5     # 获取每一行top 5元素的索引
    rank_5_retrieval = torch.mean(((torch.sum(label_matrix.float()[torch.arange(n).unsqueeze(1), top_sim_idx], dim=1) / 5) > 0).float())    # 根据top 5索引，对label矩阵的每一行进行求和，然后取平均值
    _, top_sim_idx_T = torch.topk(sim_matrix_T, k=5, dim=1) # top_sim_idx: M, 5     # 获取每一行top 5元素的索引
    rank_5_retrieval_T = torch.mean(((torch.sum(label_matrix_T.float()[torch.arange(m).unsqueeze(1), top_sim_idx_T], dim=1) / 5) > 0).float())    # 根据top 5索引，对label矩阵的每一行进行求和，然后取平均值
    # == Rank 1 检索 ==
    _, top_sim_idx = torch.topk(sim_matrix, k=1, dim=1) # top_sim_idx: N, 1     # 获取每一行top 1元素的索引
    rank_1_retrieval = torch.mean(torch.sum(label_matrix.float()[torch.arange(n).unsqueeze(1), top_sim_idx], dim=1))    # 根据top 1索引，对label矩阵的每一行进行求和，然后取平均值
    _, top_sim_idx_T = torch.topk(sim_matrix_T, k=1, dim=1) # top_sim_idx: M, 5     # 获取每一行top 5元素的索引
    rank_1_retrieval_T = torch.mean(torch.sum(label_matrix_T.float()[torch.arange(m).unsqueeze(1), top_sim_idx_T], dim=1))
    # == mAP == 
    sorted_indices = torch.argsort(sim_matrix, dim=1)   # 从小到大排序的索引
    reverse_sorted_indices = torch.flip(sorted_indices, dims=[1])   # 从大到小排序的索引
    sorted_label_matrix = torch.gather(label_matrix, dim=1, index=reverse_sorted_indices)
    cumulative_sum_matrix = torch.cumsum(sorted_label_matrix, dim=1)    # 计算前项和
    cumulative_sum_matrix = cumulative_sum_matrix * sorted_label_matrix # 过滤非匹配位置的值
    weight_matrix = 1.0 / torch.arange(1, m+1, dtype=torch.float).unsqueeze(0) # 1* n
    weight_matrix = weight_matrix.cuda()
    APs = torch.matmul(cumulative_sum_matrix.float(), weight_matrix.T).reshape(-1) / torch.sum(label_matrix.float(), dim=1)     # mAP
    nan_indices = torch.isnan(APs)
    # 从一维张量中去除NaN值
    mAP = torch.mean(APs[~nan_indices])
    
    sorted_indices_T = torch.argsort(sim_matrix_T, dim=1)   # 从小到大排序的索引
    reverse_sorted_indices_T = torch.flip(sorted_indices_T, dims=[1])   # 从大到小排序的索引
    sorted_label_matrix_T = torch.gather(label_matrix_T, dim=1, index=reverse_sorted_indices_T)
    cumulative_sum_matrix_T = torch.cumsum(sorted_label_matrix_T, dim=1)    # 计算前项和
    cumulative_sum_matrix_T = cumulative_sum_matrix_T * sorted_label_matrix_T # 过滤非匹配位置的值
    weight_matrix_T = 1.0 / torch.arange(1, n+1, dtype=torch.float).unsqueeze(0) # 1* n
    weight_matrix_T = weight_matrix_T.cuda()
    APs_T = torch.matmul(cumulative_sum_matrix_T.float(), weight_matrix_T.T).reshape(-1) / torch.sum(label_matrix_T.float(), dim=1)     # mAP
    nan_indices_T = torch.isnan(APs_T)
    # 从一维张量中去除NaN值
    mAP_T = torch.mean(APs_T[~nan_indices_T])
    metrics = {'rank_1_retrieval_ego2exo':rank_1_retrieval, 'rank_5_retrieval_ego2exo':rank_5_retrieval, 'mAP_ego2exo':mAP, 'sim_matrix_ego2exo':sim_matrix, 'label_matrix_ego2exo':label_matrix, 
               'rank_1_retrieval_exo2ego':rank_1_retrieval_T, 'rank_5_retrieval_exo2ego':rank_5_retrieval_T, 'mAP_exo2ego':mAP_T, 'sim_matrix_exo2ego':sim_matrix_T, 'label_matrix_exo2ego':label_matrix_T}
    return metrics


def eval_per_epoch(model, val_loader, eval_cfg,args, epoch=-1):
    model.eval()
    with torch.no_grad():
        labels, preds, labels1_all, labels2_all = None, None, None, None
        views_raw1 = []
        views_raw2 = []
        labels_raw1 = []
        labels_raw2 = []
        # label_list = torch.Tensor([])
        # feature_list = torch.Tensor([]).cuda()
        feature_dict = {}
        label_dict = {}
        ego_feature_dict = {}
        ego_label_dict = {}
        exo_feature_dict = {}
        exo_label_dict = {}
        for iter, sample in enumerate(tqdm(val_loader)):
            frames1_list = sample['clips1']
            frames2_list = sample['clips2']
            assert len(frames1_list) == len(frames2_list), 'frames1_list:{},frames2_list{}'.format(
                len(frames1_list), len(frames2_list))

            labels1 = sample['labels1']
            labels2 = sample['labels2']
            clips_name1 = sample['clips1_name']
            # clips_name2  = sample['clips2_name']
            label = torch.tensor(np.array(labels1) == np.array(labels2)).to(device)

            view1 = sample['view1']
            view2 = sample['view2']
            
            views_raw1.extend(sample['view1_raw'])
            views_raw2.extend(sample['view2_raw'])
            labels_raw1.extend(sample['label1_raw'])
            labels_raw2.extend(sample['label2_raw'])
            
            embeds1_list = torch.Tensor([]).to(device)
            embeds2_list = torch.Tensor([]).to(device)

            for i in range(len(frames1_list)):
                frames1 = frames1_list[i].to(device)
                frames2 = frames2_list[i].to(device)
                embeds1 = model(frames1, embed=True)
                embeds2 = model(frames2, embed=True)

                embeds1_list = torch.cat([embeds1_list, embeds1.unsqueeze(0)])
                embeds2_list = torch.cat([embeds2_list, embeds2.unsqueeze(0)])

            embeds1_avg = torch.mean(embeds1_list, dim=0)
            embeds2_avg = torch.mean(embeds2_list, dim=0)

            # print(args.dist)

            pred = pred_dist(args.dist, embeds1_avg, embeds2_avg)

            # add all data to list
            # feature_list = torch.cat([feature_list, embeds1_avg], dim=0)
            # label_list = torch.cat([label_list, labels1.reshape(-1, 1)], dim=0)
            
            # Liym: register features
            for n, clip_name in enumerate(clips_name1):
                normed_embeds1_avg = F.normalize(embeds1_avg, p=2, dim=1)
                feature_dict[clip_name] = normed_embeds1_avg[n]
                label_dict[clip_name] = labels1.reshape(-1,1)[n]
                if sample['view1_raw'][n] in ['go_left', 'go_right', 'gopro']:
                    ego_feature_dict[clip_name] = normed_embeds1_avg[n]
                    ego_label_dict[clip_name] = labels1.reshape(-1,1)[n]
                else:
                    exo_feature_dict[clip_name] = normed_embeds1_avg[n]
                    exo_label_dict[clip_name] = labels1.reshape(-1,1)[n]
                
            
            intra_view_label = torch.Tensor([label[i] for i in range(pred.shape[0]) if (view1[i]==view2[i])]).cuda()       # 视角内的label
            inter_view_label = torch.Tensor([label[i] for i in range(pred.shape[0]) if (view1[i]!=view2[i])]).cuda()       # 视角间的label
            ego2ego_view_label = torch.Tensor([label[i] for i in range(pred.shape[0]) if ((view1[i]==view2[i]) and view1[i])]).cuda()       # ego2ego的label
            exo2exo_view_label = torch.Tensor([label[i] for i in range(pred.shape[0]) if ((view1[i]==view2[i]) and (not view1[i]))]).cuda()       # exo2exo的label
            intra_view_pred = torch.Tensor([pred[i] for i in range(pred.shape[0]) if (view1[i]==view2[i])]).cuda()       # 视角内的pred
            inter_view_pred = torch.Tensor([pred[i] for i in range(pred.shape[0]) if (view1[i]!=view2[i])]).cuda()       # 视角间的pred
            ego2ego_view_pred = torch.Tensor([pred[i] for i in range(pred.shape[0]) if ((view1[i]==view2[i]) and view1[i])]).cuda()       # ego2ego的pred
            exo2exo_view_pred = torch.Tensor([pred[i] for i in range(pred.shape[0]) if ((view1[i]==view2[i]) and (not view1[i]))]).cuda()       # exo2exo的pred
            
            if iter == 0:
                preds = pred
                labels = label
                intra_view_preds = intra_view_pred
                inter_view_preds = inter_view_pred
                ego2ego_view_preds = ego2ego_view_pred
                exo2exo_view_preds = exo2exo_view_pred
                intra_view_labels = intra_view_label
                inter_view_labels = inter_view_label
                ego2ego_view_labels = ego2ego_view_label
                exo2exo_view_labels = exo2exo_view_label
                labels1_all = labels1
                labels2_all = labels2
            else:
                preds = torch.cat([preds, pred])
                labels = torch.cat([labels, label])
                intra_view_preds = torch.cat([intra_view_preds, intra_view_pred])
                inter_view_preds = torch.cat([inter_view_preds, inter_view_pred])
                ego2ego_view_preds = torch.cat([ego2ego_view_preds, ego2ego_view_pred])
                exo2exo_view_preds = torch.cat([exo2exo_view_preds, exo2exo_view_pred])
                intra_view_labels = torch.cat([intra_view_labels, intra_view_label])
                inter_view_labels = torch.cat([inter_view_labels, inter_view_label])
                ego2ego_view_labels = torch.cat([ego2ego_view_labels, ego2ego_view_label])
                exo2exo_view_labels = torch.cat([exo2exo_view_labels, exo2exo_view_label])
                labels1_all += labels1
                labels2_all += labels2
    
    
    rank_1_retrieval, rank_5_retrieval, m_ap, sim_matrix, label_matrix = retreival_metrics(feature_dict=feature_dict, label_dict=label_dict)
    ego_rank_1_retrieval, ego_rank_5_retrieval, ego_m_ap, ego_sim_matrix, ego_label_matrix = retreival_metrics(feature_dict=ego_feature_dict, label_dict=ego_label_dict)
    exo_rank_1_retrieval, exo_rank_5_retrieval, exo_m_ap, exo_sim_matrix, exo_label_matrix = retreival_metrics(feature_dict=exo_feature_dict, label_dict=exo_label_dict)
    results = cross_view_retreival_metrics(ego_feature_dict, ego_label_dict, exo_feature_dict, exo_label_dict)
    rank_1_retrieval_ego2exo, rank_5_retrieval_ego2exo, mAP_ego2exo, sim_matrix_ego2exo, label_matrix_ego2exo = results['rank_1_retrieval_ego2exo'], results['rank_5_retrieval_ego2exo'], results['mAP_ego2exo'], results['sim_matrix_ego2exo'], results['label_matrix_ego2exo']
    rank_1_retrieval_exo2ego, rank_5_retrieval_exo2ego, mAP_exo2ego, sim_matrix_exo2ego, label_matrix_exo2ego = results['rank_1_retrieval_exo2ego'], results['rank_5_retrieval_exo2ego'], results['mAP_exo2ego'], results['sim_matrix_exo2ego'], results['label_matrix_exo2ego']
    
    fpr, tpr, thresholds = roc_curve(labels.cpu().detach().numpy(), preds.cpu().detach().numpy(), pos_label=False)
    intra_view_fpr, intra_view_tpr, intra_view_thresholds = roc_curve(intra_view_labels.cpu().detach().numpy(), intra_view_preds.cpu().detach().numpy(), pos_label=False)
    inter_view_fpr, inter_view_tpr, inter_view_thresholds = roc_curve(inter_view_labels.cpu().detach().numpy(), inter_view_preds.cpu().detach().numpy(), pos_label=False)
    ego2ego_view_fpr, ego2ego_view_tpr, ego2ego_view_thresholds = roc_curve(ego2ego_view_labels.cpu().detach().numpy(), ego2ego_view_preds.cpu().detach().numpy(), pos_label=False)
    exo2exo_view_fpr, exo2exo_view_tpr, exo2exo_view_thresholds = roc_curve(exo2exo_view_labels.cpu().detach().numpy(), exo2exo_view_preds.cpu().detach().numpy(), pos_label=False)
    
    auc_value = auc(fpr, tpr)
    intra_view_auc_value = auc(intra_view_fpr, intra_view_tpr)
    inter_view_auc_value = auc(inter_view_fpr, inter_view_tpr)
    ego2ego_view_auc_value = auc(ego2ego_view_fpr, ego2ego_view_tpr)
    exo2exo_view_auc_value = auc(exo2exo_view_fpr, exo2exo_view_tpr)
    # wdr_value = compute_WDR(preds, labels1_all, labels2_all, eval_cfg.DATASET.NAME, cfg.DATASET.TXT_PATH.replace('train_pairs.txt', 'label_bank_coarse.json'))
    wdr_value = 0
    # del labels, preds, labels1_all, labels2_all
    
    results_file.writerow(['Epoch {}'.format(epoch)])
    results_file.writerow(labels.cpu().detach().numpy().tolist())
    results_file.writerow(preds.cpu().detach().numpy().tolist())
    results_file.writerow(views_raw1)
    results_file.writerow(views_raw2)
    results_file.writerow(labels_raw1)
    results_file.writerow(labels_raw2)
    
    result_list = [[preds.cpu().detach().numpy().tolist()[i], labels.cpu().detach().numpy().tolist()[i]] for i in range(len(preds))]
    result_list.sort(key=lambda x: x[0])
    # 设置图的宽度和高度
    plt.figure(figsize=(100, 25))  # 8单位宽，4单位高
    # 创建渐变色映射
    # cmap = plt.get_cmap('coolwarm')
    # 创建柱状图，每个柱子一个颜色
    for i in range(len(preds)):
        # color = cmap(color_positions[i])
        plt.bar(i, result_list[i][0], color='red' if result_list[i][1] == 1 else 'blue')
    # 添加标题和轴标签
    plt.title('title', fontsize=30)
    plt.xlabel('instance', fontsize=30)
    plt.ylabel('score', fontsize=30)
    # 修改横坐标和纵坐标的刻度标签大小
    plt.xticks([])  # 设置横坐标刻度标签的大小为12
    plt.yticks(fontsize=30)  # 设置纵坐标刻度标签的大小为12
    # 保存图像为PNG文件
    plt.savefig(os.path.join(cfg.TRAIN.SAVE_PATH, 'logs','score2label_epoch{}.png'.format(epoch)))
    # plt.savefig('score_label.png') 
    plt.close()

    torch.save(sim_matrix.detach().cpu(), os.path.join(cfg.TRAIN.SAVE_PATH, 'logs','sim_matrix_epoch{}.pth'.format(epoch)))
    torch.save(label_matrix.detach().cpu(), os.path.join(cfg.TRAIN.SAVE_PATH, 'logs','label_matrix_epoch{}.pth'.format(epoch)))
    torch.save(ego_sim_matrix.detach().cpu(), os.path.join(cfg.TRAIN.SAVE_PATH, 'logs','ego_sim_matrix_epoch{}.pth'.format(epoch)))
    torch.save(ego_label_matrix.detach().cpu(), os.path.join(cfg.TRAIN.SAVE_PATH, 'logs','ego_label_matrix_epoch{}.pth'.format(epoch)))
    torch.save(exo_sim_matrix.detach().cpu(), os.path.join(cfg.TRAIN.SAVE_PATH, 'logs','exo_sim_matrix_epoch{}.pth'.format(epoch)))
    torch.save(exo_label_matrix.detach().cpu(), os.path.join(cfg.TRAIN.SAVE_PATH, 'logs','exo_label_matrix_epoch{}.pth'.format(epoch)))
    # my_tsne(feature_list.detach().cpu().numpy(),  label_list.detach().cpu().numpy(), epoch=epoch)
    results = {
        'auc': auc_value, 
        'wdr': wdr_value, 
        'intra_view_auc' : intra_view_auc_value, 
        'inter_view_auc' : inter_view_auc_value, 
        'ego2ego_view_auc': ego2ego_view_auc_value,  
        'exo2exo_view_auc': exo2exo_view_auc_value,
        'r5_retrieval': rank_5_retrieval,
        'r1_retrieval': rank_1_retrieval,
        'mAP': m_ap,
        'ego_r5_retrieval': ego_rank_5_retrieval,
        'ego_r1_retrieval': ego_rank_1_retrieval,
        'ego_mAP': ego_m_ap,
        'exo_r5_retrieval': exo_rank_5_retrieval,
        'exo_r1_retrieval': exo_rank_1_retrieval,
        'exo_mAP': exo_m_ap,
        'ego2exo_r1_retrieval':rank_1_retrieval_ego2exo, 
        'ego2exo_r5_retrieval':rank_5_retrieval_ego2exo, 
        'ego2exo_mAP':mAP_ego2exo, 
        'exo2ego_r1_retrieval':rank_1_retrieval_exo2ego, 
        'exo2ego_r5_retrieval':rank_5_retrieval_exo2ego, 
        'exo2ego_mAP':mAP_exo2ego
    }

    return results

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', default='configs/train_resnet_config.yml', help='config file path')
    parser.add_argument('--save_path', default=None, help='path to save models and log')
    parser.add_argument('--load_path', default=None, help='path to load the model')
    parser.add_argument('--log_name', default='train_log', help='log name')
    parser.add_argument('--debug', default=False, action='store_true', help='log name')
    parser.add_argument('--pair', default=False, action='store_true', help='use pair data')
    
    parser.add_argument('--use_txt', default=False, action='store_true', help='use text loss')
    parser.add_argument('--use_neg_txt', default=False, action='store_true', help='use text loss')
    parser.add_argument('--use_gumbel', default=False, action='store_true', help='use text loss')
    
    parser.add_argument('--gt_type', type=str, default='sort', help='the type of gumbel loss gt type')
    parser.add_argument('--dist', type=str, default='NormL2', help='the distance of inference final scores ')
    # parser.add_argument('--weak_sup', default=Falsse, action='store_true', help='log name')


    args = parser.parse_args()
    return args


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_val_config():
    eval_cfg = get_cfg_defaults()

    eval_cfg.merge_from_file(args.config.replace('train', 'test'))
    return eval_cfg

if __name__ == "__main__":

    args = parse_args()
    cfg = get_cfg_defaults()
    print(cfg)
    if args.config:
        print(args.config)
        cfg.merge_from_file(args.config)
    eval_cfg = get_val_config()

    local_time = time.localtime(time.time())[0:5]
    time_str = ""
    for t in local_time:
        time_str += str(t) + '_'
    
    if args.debug:
        cfg.TRAIN.SAVE_PATH  += '_debug'
    else:
        cfg.TRAIN.SAVE_PATH += '_' + time_str
        
    setup_seed(cfg.TRAIN.SEED)
    use_cuda = cfg.TRAIN.USE_CUDA and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    # print(cfg)
    logger_path = os.path.join(cfg.TRAIN.SAVE_PATH, 'logs')
    logger = setup_logger('SV', logger_path, args.log_name, 0)
    logger.info('Running with config:\n{}\n'.format(cfg))

    if cfg.MODEL.BACKBONE == 'cat':
        is_clip=False
    else:
        is_clip=True
    train_loader, train_dataset = load_dataset(cfg, args.pair, is_clip=is_clip)
    test_loader, test_dataset = load_dataset(eval_cfg, args.pair, is_clip=is_clip)
    # test results file:
    csv_file =  open(os.path.join(cfg.TRAIN.SAVE_PATH, 'logs', 'test_results.csv'),  mode='w', newline='')
    results_file = csv.writer(csv_file)
    
    train()