import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from utils.utils_distributed import all_gather_concat


def compute_cls_loss(pred, labels, use_cosface=False):
    if use_cosface:
        # CosFace Loss
        s = 30.0
        m = 0.4
        cos_value = torch.diagonal(pred.transpose(0, 1)[labels])
        numerator = s * (cos_value - m)
        excl = torch.cat([torch.cat((pred[i, :y], pred[i, y + 1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(s * excl), dim=1)
        L = numerator - torch.log(denominator)
        loss = -torch.mean(L)
    else:
        # Softmax Loss

        criterion = CrossEntropyLoss().cuda()
        loss = criterion(pred, labels)

    return loss


def generate_sent_masks(batch_size, max_seq_length, source_lengths):
    masks = torch.zeros(batch_size, max_seq_length, dtype=torch.float)
    for e_id, src_len in enumerate(source_lengths.int()):
        masks[e_id, :src_len] = 1
    return masks

def viterbi(y, A, B, Pi=None):
    """
    Return the MAP estimate of state trajectory of Hidden Markov Model.

    Parameters
    ----------
    y : array (T,)
        Observation state sequence. int dtype.
    A : array (K, K)
        State transition matrix. See HiddenMarkovModel.state_transition  for
        details.
    B : array (K, M)
        Emission matrix. See HiddenMarkovModel.emission for details.
    Pi: optional, (K,)
        Initial state probabilities: Pi[i] is the probability x[0] == i. If
        None, uniform initial distribution is assumed (Pi[:] == 1/K).

    Returns
    -------
    x : array (T,)
        Maximum a posteriori probability estimate of hidden state trajectory,
        conditioned on observation sequence y under the model parameters A, B,
        Pi.
    T1: array (K, T)
        the probability of the most likely path so far
    T2: array (K, T)
        the x_j-1 of the most likely path so far
    """
    # Cardinality of the state space
    K = A.shape[0]
    # Initialize the priors with default (uniform dist) if not given by caller
    Pi = Pi if Pi is not None else np.full(K, 1 / K)
    T = len(y)
    T1 = np.empty((K, T), 'd')
    T2 = np.empty((K, T), 'B')

    # Initilaize the tracking tables from first observation
    T1[:, 0] = Pi * B[:, y[0]]
    T2[:, 0] = 0

    # Iterate throught the observations updating the tracking tables
    for i in range(1, T):
        T1[:, i] = np.max(T1[:, i - 1] * A.T * B[np.newaxis, :, y[i]].T, 1)
        T2[:, i] = np.argmax(T1[:, i - 1] * A.T, 1)

    # Build the output, optimal model trajectory
    x = np.empty(T, 'B')
    x[-1] = np.argmax(T1[:, T - 1])
    for i in reversed(range(1, T)):
        x[i - 1] = T2[x[i], i]

    return x, T1, T2

def get_viterbi_gt(sim_matrix):
    # from L -> R ==>> T -> D
    sim_matrix = sim_matrix.t()
    transition_probability = np.triu(np.full((len(sim_matrix), len(sim_matrix)), 1 / len(sim_matrix)))
    emission_probability = sim_matrix.detach().cpu().numpy()
    obervation_list = np.array(range(sim_matrix.shape[1]))
    import torch.distributed as dist
    # if dist.get_rank() ==0:
    #     print(sim_matrix.shape)
    #     print(obervation_list.shape)
    #     print(transition_probability.shape)
    #     print(emission_probability.shape)
    x, _, _ = viterbi(obervation_list, transition_probability, emission_probability)
    return x

def generate_gumbel_viterbi_gt(sim_matrices = None):
    # sim_matrix [b,t,n]
    b,t,n = sim_matrices.shape
    gt = []
    for sim_matrix in sim_matrices:
        x = get_viterbi_gt(sim_matrix)
        gt.append(torch.from_numpy(x).unsqueeze(dim=0).long().to(sim_matrices.device))
    gt = torch.cat(gt, dim=0)
    return gt

def generate_gumbel_sort_gt(preds_label=None):
    gt_label = torch.max(preds_label, dim=-1)[1]  # [b,t,n](one hot) -> [b,t](index)
    gt_label = torch.sort(gt_label, dim=-1)[0]  # [b,t](index) ->sort-> [b,t](ground truth)
    return gt_label


def generate_gumbel_split_gt(sim_matrices=None, sim_type=None, unpadding_text_length=None):
    """

    :param sim_matrices:
    :param bs_w: padding num of text, use for image-text matrix

    :return: if h<w:
                return gt, new_sim
            else:
                return gt
    """
    b, h, w = sim_matrices.shape
    gts,sims = [],[]

    for index, sim_matrix in enumerate(sim_matrices):
        unpadding_length = unpadding_text_length[index].item()
        if sim_type =='image':
            gt, sim = _generate_gumbel_split_gt_per_batch_image(sim_matrix, unpadding_w=unpadding_length)
        elif sim_type =='text':
            gt, sim = _generate_gumbel_split_gt_per_batch_text(sim_matrix, unpadding_h=unpadding_length)
        else:
            raise ValueError('wrong sim type: ',sim_type)
        gts.append(gt.unsqueeze(dim=0))
        sims.append(sim.unsqueeze(dim=0))
    gts = torch.cat(gts, dim=0)
    sims = torch.cat(sims, dim=0)
    return gts.long(), sims


def _generate_gumbel_split_gt_per_batch_text(sim_matrix, unpadding_h=None):
    gt, sim = [], []
    _, w = sim_matrix.shape
    h = unpadding_h
    sim = torch.zeros([h, h]).to(sim_matrix.device)
    sim_padding = torch.zeros(sim_matrix.shape)
    gt_padding = torch.zeros(sim_matrix.shape[0])
    if h > w:
        for i in range(h):
            index = i * w / h
            gt.append(round(index))
        gt = torch.tensor(gt)
        sim_padding = sim_matrix
    elif h == w:
        gt = torch.arange(h)
        sim_padding = sim_matrix
    else:
        for i in range(h):
            clip = w / h
            for j in range(h):
                s_t = clip * j
                e_t = clip * (j + 1)
                sim[i][j] = sim_matrix[i][int(s_t):int(e_t)].mean()
        gt = torch.arange(h)
        sim_padding[:h, :h] = sim
    gt_padding[:h] = gt
    return gt_padding.to(sim_matrix.device), sim_padding.to(sim_matrix.device)

def _generate_gumbel_split_gt_per_batch_image(sim_matrix, unpadding_w=None):
    """
    :param sim_matrix: [h, per_w + padding] ->[h,20]
    :param per_w: for remove padding, just use for image-text matrix
    :return:
    """
    h, _ = sim_matrix.shape
    w = unpadding_w
    sim = torch.zeros([h, h]).to(sim_matrix.device)
    sims_padding = torch.zeros(sim_matrix.shape)
    if h > w:
        gt = []
        for i in range(h):
            index = i * w / h
            gt.append(round(index))
        gt = torch.tensor(gt)
        sim_padding = sim_matrix
    elif h == w:
        gt = torch.arange(h)
        sim_padding = sim_matrix
    else:
        for i in range(h):
            clip = w / h
            for j in range(h):
                s_t = clip * j
                e_t = clip * (j + 1)
                sim[i][j] = sim_matrix[i][int(s_t):int(e_t)].mean()
        gt = torch.arange(h)
        sims_padding[:, :h] = sim
    return gt.to(sim_matrix.device), sims_padding.to(sim_matrix.device)


def generate_gumbel_gt(sim_matrices=None, preds_label=None, sim_type=None,text_label_num=None, gt_type=None):
    if gt_type == 'sort':
        return generate_gumbel_sort_gt(preds_label=preds_label)
    elif gt_type == 'viterbi':
        return generate_gumbel_viterbi_gt(sim_matrices=sim_matrices)
    elif gt_type == 'split':
        return generate_gumbel_split_gt(sim_matrices=sim_matrices, sim_type=sim_type, unpadding_text_length=text_label_num)
    else:
        raise ValueError('unknow gt_type:', gt_type)

def visualize_gumbel(sim_matrices,text_label_num, image=True,labels1=None):
    import matplotlib.pyplot as plt
    import time
    import torch.distributed as dist
    import os

    preds_image_label = F.gumbel_softmax(sim_matrices.unsqueeze(dim=0), dim=-1, tau=1/0.07, hard=True)
    gt_image_label = generate_gumbel_gt(sim_matrices=sim_matrices.unsqueeze(dim=0), preds_label=preds_image_label, gt_type='viterbi')
    loss = F.cross_entropy(preds_image_label.permute(0, 2, 1), gt_image_label, reduction='mean')
    if loss < 3:
        image_path = r'/public/home/dongsx/svip/vis_image4/'
        text_path = r'/public/home/dongsx/svip/vis_image4_text/'
        # os.mkdir(image_path)
        plt.cla()
        t = time.time()
        filename = str(loss.item()) + '_' + str(t) + str(dist.get_rank())

        # plt.colorbar()
        if image:
            H,W = 16, text_label_num.item()
        else:
            H,W = text_label_num.item(), 16,
        plt.imshow(sim_matrices[:H, :W].cpu().float().detach().numpy())
        plt.yticks(np.arange(0, H, 1))
        plt.xticks(np.arange(0, W, 1))
        plt.savefig(os.path.join(image_path, filename + '.jpg'))
        f = open((os.path.join(text_path, filename) + '.txt'), 'w')
        f.write(labels1)
        f.write('\n')

        f.write(str(preds_image_label.squeeze().cpu().detach()))
        f.write('\n')
        f.write(str(torch.argmax(preds_image_label,dim=-1).squeeze().cpu().detach()))
        f.write('\n')
        f.write(str(gt_image_label.squeeze().cpu().detach()))
        f.close()

def compute_gumbel_loss(image_features, text_feature_phrase, text_label_num, logit_scale=1 / 0.07, max_seq_length=20,gt_type='sort',labels1=None):
    """

    :param image_features: [b,t,d]
    :param text_feature_phrase:[b,n,d]
    :param text_label_num:[l1,l2,l3,l4,...]
    :param logit_scale:temperature
    :return:
    """
    logit_scale = torch.mean(logit_scale)
    
    # image_mask = None
    text_mask = generate_sent_masks(image_features.shape[0], max_seq_length, text_label_num).to(image_features.device)
    image_features = image_features / (image_features.norm(dim=-1, keepdim=True) + 1e-6)
    text_features_phrase = text_feature_phrase / (text_feature_phrase.norm(dim=-1, keepdim=True) + 1e-6)
    logit_per_image = image_features@text_features_phrase.permute(0, 2, 1)
    logit_per_text = text_features_phrase@image_features.permute(0, 2, 1)
    # visualize_gumbel(logit_per_image[0], text_label_num[0], image=True,labels1=labels1[0])
    # image loss

    if gt_type == 'split':
        gt_image_label, logit_per_image = generate_gumbel_gt(sim_matrices=logit_per_image,sim_type='image', text_label_num=text_label_num,gt_type=gt_type)
        # -> [b,t,n] -> [b,t,n](one hot)
        preds_image_label = F.gumbel_softmax(logit_per_image, dim=-1, tau=logit_scale, hard=True)
    else:
        # -> [b,t,n] -> [b,t,n](one hot)
        preds_image_label = F.gumbel_softmax(logit_per_image, dim=-1, tau=logit_scale, hard=True)
        gt_image_label = generate_gumbel_gt(sim_matrices=logit_per_image, preds_label=preds_image_label, gt_type=gt_type)

    image_loss = F.cross_entropy(preds_image_label.permute(0, 2, 1), gt_image_label, reduction='mean')  # CE([b,t,n](one hot), [b,t](ground truth))

    # text loss
    if gt_type == 'split':
        gt_text_label, logit_per_text = generate_gumbel_gt(sim_matrices=logit_per_text, sim_type='text', text_label_num=text_label_num, gt_type=gt_type)
        preds_text_label = F.gumbel_softmax(logit_per_text, dim=-1, tau=logit_scale, hard=True)  # -> [b,n,t]
    else:
        preds_text_label = F.gumbel_softmax(logit_per_text, dim=-1, tau=logit_scale, hard=True)  # -> [b,n,t]
        gt_text_label = generate_gumbel_gt(sim_matrices=logit_per_text, preds_label=preds_text_label,gt_type=gt_type)

    text_loss = F.cross_entropy(preds_text_label.permute(0, 2, 1), gt_text_label, reduction='none')
    text_loss_masked = text_loss * text_mask

    loss = image_loss + text_loss_masked.mean()
    return loss

def compute_info_loss(image_features, text_features, labels=None, logit_scale=1 / 0.07, ):
    logit_scale = torch.mean(logit_scale)
    # print(sum(image_features))
    image_features = image_features / (image_features.norm(dim=-1, keepdim=True) + 1e-6)
    text_features = text_features / (text_features.norm(dim=-1, keepdim=True) + 1e-6)
    
    logits_per_image = logit_scale * image_features @ text_features.t()  # img -> text [b,t,l]
    logits_per_text = logit_scale * text_features @ image_features.t()  # text -> img  [b,l,t]
    
    labels = torch.arange(len(logits_per_image)).to(logits_per_image.device)
    # print(sum(logits_per_image))

    loss_image = F.cross_entropy(logits_per_image, labels)
    loss_text = F.cross_entropy(logits_per_text, labels)

    loss = loss_image + loss_text
    return loss


def compute_info_loss_ddp(image_features, text_features, labels=None, logit_scale=1 / 0.07, ):
    image_features = all_gather_concat(image_features)
    text_features = all_gather_concat(text_features)

    image_features = image_features / (image_features.norm(dim=-1, keepdim=True) + 1e-6)
    text_features = text_features / (text_features.norm(dim=-1, keepdim=True) + 1e-6)
    logits_per_image = logit_scale * image_features @ text_features.t()  # img -> text [b,t,l]
    logits_per_text = logit_scale * text_features @ image_features.t()  # text -> img  [b,l,t]

    labels = torch.arange(len(logits_per_image)).to(logits_per_image.device)

    loss_image = F.cross_entropy(logits_per_image, labels)
    loss_text = F.cross_entropy(logits_per_text, labels)

    loss = loss_image + loss_text

    return loss


def compute_info_loss_mask(image_features, text_features, labels=None, logit_scale=1 / 0.07, ):

    image_features = image_features / (image_features.norm(dim=-1, keepdim=True) + 1e-6)
    text_features = text_features / (text_features.norm(dim=-1, keepdim=True) + 1e-6)
    logits_per_image = logit_scale * image_features @ text_features.t()  # img -> text [b,t,l]
    logits_per_text = logit_scale * text_features @ image_features.t()  # text -> img  [b,l,t]
    label_mask = (labels.unsqueeze(dim=-2) == labels.unsqueeze(dim=-1)).int()

    def NCE_loss(sample, mask):
        # pos_samples = sample.
        pos_samples = (sample.exp()*mask).sum(dim=-1)
        all_samples = sample.exp().sum(dim=-1)
        NCE_loss = torch.log(pos_samples / all_samples)
        return NCE_loss

    loss_image = NCE_loss(logits_per_image, label_mask).mean()
    loss_text = NCE_loss(logits_per_text, label_mask).mean()

    loss = -(loss_image + loss_text)

    return loss




def compute_info_loss_mask_ddp(image_features, text_features, labels=None, logit_scale=1 / 0.07, ):
    import torch.distributed as dist
    if dist.get_rank() == 1:
        print(image_features.shape)
        print(text_features.shape)
        print(labels.shape)
    image_features = all_gather_concat(image_features)
    text_features = all_gather_concat(text_features)
    labels = all_gather_concat(labels)
    if dist.get_rank() == 1:
        print(image_features.shape)
        print(text_features.shape)
        print(labels.shape)

    image_features = image_features / (image_features.norm(dim=-1, keepdim=True) + 1e-6)
    text_features = text_features / (text_features.norm(dim=-1, keepdim=True) + 1e-6)

    logits_per_image = logit_scale * image_features @ text_features.t()  # img -> text [b,t,l]
    logits_per_text = logit_scale * text_features @ image_features.t()  # text -> img  [b,l,t]

    label_mask = (labels.unsqueeze(dim=-2) == labels.unsqueeze(dim=-1)).int()

    def NCE_loss(sample, mask):
        pos_samples = (sample.exp()*mask).sum(dim=-1)
        all_samples = sample.exp().sum(dim=-1)
        NCE_loss = torch.log(pos_samples / all_samples)
        return NCE_loss

    loss_image = NCE_loss(logits_per_image, label_mask).mean()
    loss_text = NCE_loss(logits_per_text, label_mask).mean()

    loss = - (loss_image + loss_text)

    return loss



def compute_info_loss_neg(image_features, pos_text_features, all_text_features, logit_scale=1 / 0.07, ):
    image_features = image_features / (image_features.norm(dim=-1, keepdim=True) + 1e-6)
    pos_text_features = pos_text_features / (pos_text_features.norm(dim=-1, keepdim=True) + 1e-6)
    all_text_features = all_text_features / (all_text_features.norm(dim=-1, keepdim=True) + 1e-6)

    image_features = image_features.unsqueeze(dim=1)
    pos_text_features = pos_text_features.unsqueeze(dim=1)

    logits_per_image_pos = logit_scale * image_features @ pos_text_features.permute(0, 2, 1)  # img -> text [b,1,1]
    logits_per_text_pos = logit_scale * pos_text_features @ image_features.permute(0, 2, 1)  # text -> img  [b,1,1]
    logits_per_image_all = logit_scale * image_features @ all_text_features.permute(0, 2, 1)  # img -> text [b,1,n]
    logits_per_text_all = logit_scale * all_text_features @ image_features.permute(0, 2, 1)  # text -> img [b,1,n]

    def NCE_loss(pos_samples, all_samples):
        # pos_samples = sample.
        pos_samples = pos_samples.exp().squeeze(dim=-1)
        all_samples = all_samples.exp().sum(dim=-1)

        NCE_loss = torch.log(pos_samples / all_samples)

        return NCE_loss

    loss_image = NCE_loss(logits_per_image_pos, logits_per_image_all)
    loss_text = NCE_loss(logits_per_text_pos, logits_per_text_all.permute(0, 2, 1))

    loss = - (loss_image + loss_text).mean()
    return loss

def compute_seq_loss(seq1, seq2, temperature=1):
    # min loss: 13.5392

    if seq1 == None or seq2 == None:
        return 0

    seq1 = F.normalize(seq1, 2, dim=-1)
    seq2 = F.normalize(seq2, 2, dim=-1)

    bs, length, _ = seq1.size()

    corr = torch.bmm(seq1, seq2.transpose(1, 2))
    corr /= temperature
    corr1 = nn.Softmax(dim=1)(corr)  # Softmax across columns
    corr2 = nn.Softmax(dim=2)(corr)  # Softmax across rows
    corr = (corr1 + corr2) / 2

    sims = torch.diagonal(corr, dim1=1, dim2=2)

    loss = torch.sum(torch.tensor(1) - sims) / bs

    return loss


# ------------------------------------------------------------------
# -------------------------------------------------------------------
# def align_matrix(video_embed, lang_embed, Mask, sim='cos', t-emp=1):
#     """
#     to get alignment matrix by cosine_similaritys
#     video_embed: tensor [B,T,512]
#     lang_embed: list [B*[L,512]]
#     """
#
#     # contrastive_logits =torch.cosine_similarity(video_embed, lang_embed, dim=-1)
#     batch_size = video_embed.shape[0]
#     matrices = []
#     for i in range(batch_size):
#         mask = Mask[i, :, :]  # [T,512]
#         # lang = lang_embed[i].to(video_embed.get_device())
#         lang = lang_embed[i]
#         video_feature_norm = (video_embed[i, :, :] / video_embed[i, :, :].norm(dim=-1, keepdim=True)) * mask
#
#         # video_feature_norm = (video_embed[i] - video_embed[i].mean(dim=-1, keepdim=True)) / video_embed[i].std(dim=-1,keepdim=True) * mask
#
#         text_feature_norm = lang / lang.norm(dim=-1, keepdim=True)
#         # text_feature_norm = (lang - lang.mean(dim=-1, keepdim=True)) / lang.std(dim=-1, keepdim=True)
#         if sim == 'cos':
#             cosine_similaritys = torch.einsum("td,ld->tl", video_feature_norm, text_feature_norm).div(
#                 temp).T  # temperature
#         else:
#             cosine_similaritys = torch.einsum("td,ld->tl", video_feature_norm, text_feature_norm).T  # [text_len, video_len]
#
#         matrices.append(cosine_similaritys)
#
#     return matrices


def align_matrix2(image_features, text_features, Mask=None, sim='cos', logit_scale=1):
    # matrices = align_matrix(image_features, text_features, Mask, sim='cos', temp=1)
    if Mask is not None:
        image_features = image_features / (image_features.norm(dim=-1, keepdim=True) + 1e-8) * Mask
    else:
        image_features = image_features / (image_features.norm(dim=-1, keepdim=True) + 1e-8)
    # text_features = torch.stack(text_features, dim=0)
    text_features = text_features / (text_features.norm(dim=-1, keepdim=True) + 1e-8)
    # logits_per_image = logit_scale * torch.einsum('btd,bdl->btl',)
    logits_per_image = logit_scale * image_features @ text_features.permute(0, 2, 1)  # img -> text [b,t,l]
    logits_per_text = logit_scale * text_features @ image_features.permute(0, 2, 1)  # text -> img  [b,l,t]

    corr1 = nn.Softmax(dim=-1)(logits_per_image)  # Softmax across columns
    corr2 = nn.Softmax(dim=-1)(logits_per_text)  # Softmax across rows
    corr = (corr1 + corr2) / 2

    sims = torch.diagonal(corr, dim1=1, dim2=2)

    loss = torch.sum(torch.tensor(1) - sims) / 1

    return logits_per_image, logits_per_text, loss


def stamp2label(matrices, time_stamps):
    """

    :param matrices: [3,256,3]
    :param time_stamps: [3,3,2]
    :return:
    """
    time_stamps = np.array(time_stamps)

    b, t, l = matrices.size()
    labels = torch.zeros([b, t])
    for batch, stamps in enumerate(time_stamps):
        for index, (start, end) in enumerate(stamps):
            labels[batch][int(start):int(end + 1)] = index + 1

    # labels = torch.arange(l)
    return labels


def get_pos_samples(matrix_row, stamp):
    # caculate each row
    # matrix tensor[1 ,T] stamp [1,2]
    start = int(stamp[0])
    end = int(stamp[1])
    pos = torch.sum(torch.exp(matrix_row[start:end + 1]), dim=-1)
    # print(pos)
    F.log_softmax()
    return pos


def get_neg_samples(matrix_row, stamp):
    start = int(stamp[0])
    end = int(stamp[1])
    neg = torch.sum(torch.exp(matrix_row[0:start]), dim=-1) + torch.sum(torch.exp(matrix_row[end + 1:-1]), dim=-1)
    # print(neg)
    return neg


def contrastive_loss(matrices, time_stamps):
    # matrices [B,L,T] time_stamps [B, L, 2]
    B = len(matrices)
    loss = []
    for mat, stamps in zip(matrices, time_stamps):
        # [L,T] [L,2]
        info_nce = []
        for i, stamp in enumerate(stamps):
            pos_samples = get_pos_samples(mat[i, :], stamp)
            neg_samples = get_neg_samples(mat[i, :], stamp)

            entropy = torch.log(torch.div(pos_samples, pos_samples + neg_samples))
            # print(entropy)
            info_nce.append(entropy)
        info_nce = torch.sum(torch.stack(info_nce, dim=0)).view(1)
        # print(info_nce)
        loss.append(info_nce)
    loss = -torch.sum(torch.stack(loss)).view(1)
    loss = loss / B

    return loss


def contrastive_loss2(matrices, time_stamps):
    # matrices [B,L,T] time_stamps [B, L, 2]
    B = len(matrices)
    loss = []
    for mat, stamps in zip(matrices, time_stamps):
        # [L,T] [L,2]
        info_nce = []
        for i, stamp in enumerate(stamps):
            pos_samples = get_pos_samples(mat[i], stamp)
            neg_samples = get_neg_samples(mat[i], stamp)

            entropy = torch.log(torch.div(pos_samples, pos_samples + neg_samples))
            # print(entropy)
            info_nce.append(entropy)
        info_nce = torch.sum(torch.stack(info_nce, dim=0)).view(1)
        # print(info_nce)
        loss.append(info_nce)
    loss = -torch.sum(torch.stack(loss)).view(1)
    loss = loss / B

    return loss


# -------------------------------------------------------------------
# -------------------------------------------------------------------


if __name__ == '__main__':
    # seq1 = torch.reshape(torch.tensor([x for x in range(16)]),[2,8])
    seq1 = torch.rand([4, 512])
    seq2 = torch.rand([4,16, 512])
    seq3 = torch.rand([4, 20, 512])
    # seq3 = torch.tensor([4,5,6,7])
    labels1 = torch.empty(4).random_(20)
    while True:
        loss1 = compute_gumbel_loss(seq2, seq3,torch.tensor([10,12,13,11]))
        print(loss1)
        # _ = compute_gumbel_loss(seq1, seq2, labels1, logit_scale=1 / 0.07,max_seq_length=20)
