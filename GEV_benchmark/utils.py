import os
import os.path as osp
from torchvideotransforms import video_transforms, volume_transforms
import shutil
import sys
import torch.nn.functional as F
import torch.nn as nn

def get_video_trans():
    train_trans = video_transforms.Compose([
        video_transforms.RandomHorizontalFlip(),
        video_transforms.Resize((455,256)),
        video_transforms.RandomCrop(224),
        volume_transforms.ClipToTensor(),
        video_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_trans = video_transforms.Compose([
        video_transforms.Resize((455,256)),
        video_transforms.CenterCrop(224),
        volume_transforms.ClipToTensor(),
        video_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_trans, test_trans


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    # _, pred = output.topk(maxk, 1, True, True)
    # pred = pred.t()
    pred = output
    # print(pred.shape)
    # print(target.shape)
    correct = pred.eq(target.view(-1,1).expand_as(pred))    # N,1
    # print(pred)
    # print(target)
    # print(correct)
    # assert 1==2
    res = []
    for k in topk:
        correct_k = correct[:,:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def per_class_accuracy(target, output, class_count_dict, acc_count_dict, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    
    # _, pred = output.reshape(-1, 1).topk(maxk, 1, True, True)
    # pred = pred.t()
    pred = output
    correct = pred.eq(target.view(-1,1).expand_as(pred))    # N,1
    for i, gt in enumerate(target.reshape(-1)):
        class_count_dict[int(gt.item())] += 1
        if correct[i].item():
            acc_count_dict[int(gt.item())] += 1

def per_action_accuracy(output, target, action_class_count_dict, action_acc_count_dict, total_action_id, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    pred = output
    correct = pred.eq(target.view(-1,1).expand_as(pred))    # N,1
    for i, gt in enumerate(target.reshape(-1)):
        action = int(total_action_id[i].item())
        action_class_count_dict[action][int(gt.item())] += 1
        if correct[i].item():
            action_acc_count_dict[action][int(gt.item())] += 1
    return

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return

def back_up(save_root)-> None:
    """
        back up a snapshot of current main codes
    """
    back_up_dir = osp.join(save_root,"backups")
    if not osp. exists (back_up_dir) :
        os.makedirs (back_up_dir)
    back_up_list = [
        "datasets",
        "models",
        "utils.py",
        "opt.py",
        "main.py",
        "runner.py"]
    back_up_list = [osp.join(osp.dirname(osp.abspath(__file__)), x) for x in back_up_list]
    back_up_list = list(filter(lambda x: osp.exists(x), back_up_list))
    for f in back_up_list:
        filename = osp.split(f) [-1]
        if osp.isdir(f):
            shutil.copytree(f, osp.join(back_up_dir, filename), symlinks=True, dirs_exist_ok=True)
        else:
            shutil.copy(f, back_up_dir, follow_symlinks=False)
        print(f"{filename} back-up finished")

    # 获取命令行参数列表
    command_args = sys.argv
    command = 'python'
    for arg in command_args:
        command += ' {}'.format(arg)
    run_f = open(osp.join(back_up_dir, 'run.sh'), 'w')
    run_f.write(command)
    run_f.close()

################### Loss functions

def weighted_bce_loss(y_hat, y, loss_weight, action_index,mode='mean'):
    # Compute bce loss for each item in the batch
    bce_loss = nn.BCELoss(reduction='none')(y_hat, y)

    # Fetch the weights for each item in the batch based on action_index
    # Assuming action_index is 1-based and needs to be converted to 0-based for indexing
    weights = loss_weight[action_index]

    # Apply weights
    weighted_loss = bce_loss * weights

    # Sum over all dimensions except the batch dimension
    weighted_loss = weighted_loss.sum(dim=1)

    if mode == 'mean':
        # Average the weighted loss across the batch
        loss = weighted_loss.mean()
    elif mode == 'sum':
        # Sum the weighted loss across the batch
        loss = weighted_loss.sum()
    else:
        raise ValueError('mode must be either mean or sum')
    return loss

def weighted_mse_loss(y_hat, y, mse_weight, action_index ,mode='mean'):
    # Compute MSE loss for each item in the batch
    mse_loss = nn.MSELoss(reduction='none')(y_hat, y)

    # Fetch the weights for each item in the batch based on action_index
    # Assuming action_index is 1-based and needs to be converted to 0-based for indexing
    weights = mse_weight[action_index]

    # Apply weights
    weighted_loss = mse_loss * weights

    # Sum over all dimensions except the batch dimension
    weighted_loss = weighted_loss.sum(dim=1)

    if mode == 'mean':
        # Average the weighted loss across the batch
        loss = weighted_loss.mean()
    elif mode == 'sum':
        # Sum the weighted loss across the batch
        loss = weighted_loss.sum()
    else:
        raise ValueError('mode must be either mean or sum')

    return loss
    
def set_loss_weight(action_count_dict, loss_weight):
    for idx, action_count in enumerate(action_count_dict):
        for action in action_count:
            # Avoid division by zero for actions with 0 count
            if action_count[action] > 0:
                # Inverse frequency weighting
                loss_weight[idx, action] = 1.0 / action_count[action]
            else:
                # Assign a default weight for actions with 0 count, if needed
                loss_weight[idx, action] = 0  # or some other default value

    # Normalize each row
    row_sums = loss_weight.sum(dim=1, keepdim=True)
    # Avoid division by zero for rows that sum to 0
    row_sums[row_sums == 0] = 1
    loss_weight = loss_weight / row_sums
    return loss_weight



def calculate_f1_for_label(tp, total, fp):
    fn = total - tp  # False negatives are the total count minus true positives
    if tp == 0:
        return 0  # Avoid division by zero
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0