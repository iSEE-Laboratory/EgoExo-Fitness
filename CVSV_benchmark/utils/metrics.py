
import os

import numpy as np
import torch
import torch.nn.functional as F


def compute_WDR(preds, labels1, labels2, DATASET, label_bank_path):
    # compute weighted distance ratio
    #        weighted dist / # unmatched pairs
    # WDR = ---------------------------------
    #             dist / # matched pairs
    import json
    def read_json(file_path):
        with open(file_path, 'r') as f:
            data = json.loads(f.read())
        return data

    def compute_edit_dist(seq1, seq2):
        """
        计算字符串 seq1 和 seq1 的编辑距离
        :param seq1
        :param seq2
        :return:
        """
        matrix = [[i + j for j in range(len(seq2) + 1)] for i in range(len(seq1) + 1)]
        for i in range(1, len(seq1) + 1):
            for j in range(1, len(seq2) + 1):
                if (seq1[i - 1] == seq2[j - 1]):
                    d = 0
                else:
                    d = 2
                matrix[i][j] = min(matrix[i - 1][j] + 1, matrix[i][j - 1] + 1, matrix[i - 1][j - 1] + d)
        return matrix[len(seq1)][len(seq2)]

    # Load steps info for the corresponding dataset
    # label_bank_path = os.path.join('annotations', DATASET, 'label_bank.json')
    label_bank = read_json(label_bank_path)
    # label_bank = read_json('Datasets/COIN-SV/label_bank.json')
    # label_bank = read_json('Datasets/Diving48-SV/label_bank.json')
    # label_bank = read_json('Datasets/CSV/label_bank.json')

    # Calcualte wdr
    # import ipdb
    # ipdb.set_trace()
    labels = torch.tensor(np.array(labels1) == np.array(labels2))
    m_dists = preds[labels]
    um_dists = []
    for i in range(labels.size(0)):
        label = labels[i]
        if not label:
            # unmatched pair
            # NormL2 dist / edit distance
            um_dists.append(preds[i] / compute_edit_dist(label_bank[labels1[i]], label_bank[labels2[i]]))
    # ipdb.set_trace()
    if len(um_dists) == 0:
        # um_dists.append(0)
        return 0
    return torch.tensor(um_dists).mean() / m_dists.mean()


def pred_dist(dist, embeds1_avg, embeds2_avg):
    if dist == 'L1':
        # L1 distance
        pred = torch.sum(torch.abs(embeds1_avg - embeds2_avg), dim=1)
    elif dist == 'L2':
        # L2 distance
        pred = torch.sum((embeds1_avg - embeds2_avg) ** 2, dim=1)
    elif dist == 'NormL2':
        # L2 distance between normalized embeddings
        pred = torch.sum((F.normalize(embeds1_avg, p=2, dim=1) - F.normalize(embeds2_avg, p=2, dim=1)) ** 2,
                         dim=1)
    elif dist == 'cos':
        # Cosine similarity
        pred = torch.cosine_similarity(embeds1_avg, embeds2_avg, dim=1)
    else:
        raise ValueError('need dist')

    return pred



