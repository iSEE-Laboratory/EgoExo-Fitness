
import argparse


parser = argparse.ArgumentParser(description="PyTorch implementation of Action Recognition of FLAG3D")


# model settings
parser.add_argument('--backbone_arch', default='i3d', type=str,
                    help='backbone architecture')
parser.add_argument('--video_enc_arch', default=None, type=str,
                    help='video encoder architecture')
parser.add_argument('--fuser_arch', default=None, type=str,
                    help='modal fuser architecture')
parser.add_argument('--fix_backbone', action='store_true', default=False, 
                    help='fix backbone during training')
parser.add_argument('--classifier_in_dim', default=512, type=int,  
                    help='the input of classifier input dimension')

# dataset
parser.add_argument('--view', default='all', type=str,
                    help='view')
parser.add_argument('--vid_enc', default='i3d', type=str,
                    help='video encoder')
parser.add_argument('--lan_enc', default='roberta', type=str,
                    help='language encoder')
parser.add_argument('--dataset_split', default=0, type=int,
                    help='random seed')
parser.add_argument('--split_mode', default=0, type=int,
                    help='select split mode [0, 1] for v1 and v2')
parser.add_argument('--list_file_train', default='./annotations/train_split_0.txt', type=str,
                    help='path of training list file')
parser.add_argument('--list_file_test', default='./annotations/test_split_0.txt', type=str,
                    help='path of testing list file')
parser.add_argument('--list_file_open', default='./annotations/open_split_0.txt', type=str,
                    help='path of opening list file')
parser.add_argument('--sample_ego', action='store_true', default=False, 
                    help='sample egocentric videos')
parser.add_argument('--sample_rate', default=1.0, type=float,
                    help='sample rate for egocentric videos')
parser.add_argument('--open_set_action', default='0', type=str,
                    help='path of opening list file')
parser.add_argument('--test_actions', default='0', type=str,
                    help='test_action_list')
# training settings
parser.add_argument('--num_epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--bs_train', default=32, type=int,
                    help='batch size ')
parser.add_argument('--bs_test', default=32, type=int,
                    help='batch size ')
parser.add_argument('--workers', default=4, type=int,
                    help='workers')
parser.add_argument('--base_lr', default=1e-6, type=float,
                    help='base LR')
parser.add_argument('--lr_factor', default=0.1, type=float,
                    help='LR factor')
parser.add_argument('--nce_factor', default=0.3, type=float,
                    help='InfoNCE factor')
parser.add_argument('--weight_decay', default=0.0, type=float,
                    help='weight decay')
parser.add_argument('--lr_decay', action='store_true', default=False, 
                    help='lr_decay')
parser.add_argument('--num_frame_per_seg', default=1, type=int,
                    help='number of frames per segment')
parser.add_argument('--loss_type', default='mse', type=str, choices=['mse', 'weighted_mse', 'bce', 'weighted_bce'],
                    help='loss function')
parser.add_argument('--loss_weight', default='none', type=str, choices=['none','fixed', 'adaptive'],
                    help='whether the weight of loss is fixed or adpative')


# path
parser.add_argument('--pretrained_backbone', default='/home/yuanming/Code/pretrained_models/i3d_model_rgb.pth', type=str,
                    help='path of pretrained backbone')
parser.add_argument('--label_path', default='./annotations/', type=str,
                    help='path of label')
parser.add_argument('--data_path', default='/mnt/Datasets/Ego-iSEE/st_ed_sync_videos_frame', type=str,
                    help='path of data ')
parser.add_argument('--ckpt_path', default='', type=str,
                    help='path of checkpoint')
parser.add_argument('--cfg_path', default='', type=str,
                    help='path of config')             


# exp
parser.add_argument('--gpu', default='0,1', type=str,
                    help='gpus')
parser.add_argument('--exp_name', default='debug', type=str,
                    help='exp name')
parser.add_argument('--seed', default=2048, type=int,
                    help='random seed')
parser.add_argument('--running_mode', default='train', type=str,
                    help='running mode ')
parser.add_argument('--test_on_training_set', action='store_true', default=False, 
                    help='test model on training set')
parser.add_argument('--return_attn', action='store_true', default=False, 
                    help='return the attention weight')
