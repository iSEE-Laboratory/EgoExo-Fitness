import torch
from torch.utils import data
from torchvision import transforms as tf
import logging
import os
import numpy as np
from PIL import Image
import cv2
import random
import json as js
import clip
from dataset.label import LABELS
from torchvideotransforms import video_transforms, volume_transforms

def get_video_trans(is_clip=False):
    # (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
    if not is_clip:
        logger.info('Using ImageNet normalization')
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
    else:
        logger.info('Using CLIP normalization')
        train_trans = video_transforms.Compose([
            video_transforms.RandomHorizontalFlip(),
            video_transforms.Resize((455,256)),
            video_transforms.RandomCrop(224),
            volume_transforms.ClipToTensor(),
            video_transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
        ])
        test_trans = video_transforms.Compose([
            video_transforms.Resize((455,256)),
            video_transforms.CenterCrop(224),
            volume_transforms.ClipToTensor(),
            video_transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
        ])
    return train_trans, test_trans

logger = logging.getLogger('Sequence Verification')



class VerificationDataset(data.Dataset):

    def __init__(self,
                 mode='train',
                 dataset_name='CSV',
                 txt_path=None,
                 normalization=None,
                 num_clip=16,
                 augment=True,
                 num_sample=600, transform=None,
                 random_seed = 0,
                 pair_data=True):

        assert mode in ['train', 'test', 'val'], 'Dataset mode is expected to be train, test, or val. But get %s instead.' % mode
        self.mode = mode
        self.dataset_name = dataset_name
        self.normalization = normalization
        self.num_clip = num_clip
        self.augment = augment
        if augment:
            self.aug_flip = True
            self.aug_crop = True
            self.aug_color = True
            self.aug_rot = True
        self.num_sample = num_sample  # num of pairs randomly selected from all training pairs
        self.pair_data = True
        self.txt_path = txt_path
        if not pair_data:
            self.txt_path = self.txt_path.replace('train_pairs.txt', 'train.txt')
        self.data_list = [line.strip() for line in open(txt_path, 'r').readlines()]
        
        
        
        if num_sample > len(self.data_list):
            logger.info('Randomly selecting [%d] samples from [%d] %s set' % (len(self.data_list), len(self.data_list), self.mode))
            self.data_list = self.data_list
        else:
            random.seed(random_seed)
            random.shuffle(self.data_list)
            logger.info('Randomly selecting [%d] samples from [%d] %s set' % (num_sample, len(self.data_list), self.mode))
            self.data_list = self.data_list[:num_sample]

        # logger.info('Randomly selecting [%d] samples from [%d] testing samples' % (len(self.data_list), len(self.data_list)))
        self.transform = transform
        
        # logger.info('Successfully construct dataset with [%s] mode and [%d] samples randomly selected from [%d] samples' % (mode, len(self), len(self.data_list)))
        logger.info('Successfully construct dataset')
        self.views = ['gopro', 'go_left', 'go_right', 'logit_left', 'logit_right', 'kinect']

    def __getitem__(self, index):
        data_path = self.data_list[index]
        data_path_split = data_path.strip().split(' ')

        if self.pair_data or self.mode != 'train':
            sample = {
                'data': data_path,
                'clips1_name': '{}_{}_{}'.format(data_path_split[0], int(data_path_split[2]), int(data_path_split[3])),
                'clips1': self.sample_clips(data_path_split[0], int(data_path_split[2]), int(data_path_split[3]), True)[0],
                'idx1': self.sample_clips(data_path_split[0], int(data_path_split[2]), int(data_path_split[3]), True)[1],
                'labels1': LABELS[self.dataset_name][self.mode].index(data_path_split[1]),
                'label_token1': self._load_label_token(data_path_split[1]),
                'label_token_phrase1': self._load_label_token_phrase(data_path_split[1]),
                'label_neg_token1': self._load_neg_label_token(data_path_split[1]),
                'view1': (self.views.index(data_path_split[0].split('/')[-2]) < 3),
                'view1_raw': data_path_split[0].split('/')[-2],
                'label1_raw': data_path_split[1],
                
                'clips2_name': '{}_{}_{}'.format(data_path_split[4], int(data_path_split[6]), int(data_path_split[7])),
                'clips2': self.sample_clips(data_path_split[4], int(data_path_split[6]), int(data_path_split[7]), True)[0],
                'idx2': self.sample_clips(data_path_split[4], int(data_path_split[6]), int(data_path_split[7]), True)[1],
                'labels2': LABELS[self.dataset_name][self.mode].index(data_path_split[5]),
                'label_token2': self._load_label_token(data_path_split[5]),
                'label_token_phrase2': self._load_label_token_phrase(data_path_split[5]),
                'label_neg_token2': self._load_neg_label_token(data_path_split[5]),
                'view2': (self.views.index(data_path_split[4].split('/')[-2]) < 3),       # 当前的版本是只考虑ego和exo
                'view2_raw': data_path_split[4].split('/')[-2],
                'label2_raw': data_path_split[5]
            }
        else:
            sample = {
                # 'index': index,
                'data': data_path_split[0],
                'clips1': self.sample_clips(data_path_split[0], int(data_path_split[2]), int(data_path_split[3]), True)[0],
                'idx1': self.sample_clips(data_path_split[0], int(data_path_split[2]), int(data_path_split[3]), True)[1],
                'labels1': LABELS[self.dataset_name][self.mode].index(data_path_split[1]) if self.mode == 'train' else data_path_split[1],
                'label_token1': self._load_label_token(data_path_split[1]),
                'label_token_phrase1': self._load_label_token_phrase(data_path_split[1]),
                'label_neg_token1': self._load_neg_label_token(data_path_split[1]),
                'view1': (self.views.index(data_path_split[0].split('/')[-2]) < 3),
                'labels1_raw': data_path_split[1],
            }
        return sample


    def __len__(self):
        # if self.mode == 'train':
        #     return self.num_sample
        # else:
        #     return len(self.data_list)
        return len(self.data_list)

    def get_one_instance(self, index):
        data_path = self.data_list[index]
        data_path_split = data_path.strip().split(' ')
        print(data_path_split)
        if self.pair_data or self.mode != 'train':
            sample = {
                'data': data_path,
                
                'clips1': self.sample_clips(data_path_split[0], int(data_path_split[2]), int(data_path_split[3]), True)[0],
                'idx1': self.sample_clips(data_path_split[0], int(data_path_split[2]), int(data_path_split[3]), True)[1],
                'labels1': LABELS[self.dataset_name][self.mode].index(data_path_split[1]) if self.mode == 'train' else data_path_split[1],
                'label_token1': self._load_label_token(data_path_split[1]),
                'label_token_phrase1': self._load_label_token_phrase(data_path_split[1]),
                'label_neg_token1': self._load_neg_label_token(data_path_split[1]),
                'view1': (self.views.index(data_path_split[0].split('/')[-2]) < 3),
                'labels1_raw': data_path_split[1],
                
                'clips2': self.sample_clips(data_path_split[4], int(data_path_split[6]), int(data_path_split[7]), True)[0],
                'idx2': self.sample_clips(data_path_split[4], int(data_path_split[6]), int(data_path_split[7]), True)[1],
                'labels2': LABELS[self.dataset_name][self.mode].index(data_path_split[5]) if self.mode == 'train' else data_path_split[5],
                'label_token2': self._load_label_token(data_path_split[3]),
                'label_token_phrase2': self._load_label_token_phrase(data_path_split[3]),
                'label_neg_token2': self._load_neg_label_token(data_path_split[3]),
                'view2': (self.views.index(data_path_split[4].split('/')[-2]) < 3),       # 当前的版本是只考虑ego和exo
                'labels2_raw': data_path_split[3]
            }
        else:
            sample = {
                # 'index': index,
                'data': data_path_split[0],
                'clips1': self.sample_clips(data_path_split[0], int(data_path_split[2]), int(data_path_split[3]), True)[0],
                'idx1': self.sample_clips(data_path_split[0], int(data_path_split[2]), int(data_path_split[3]), True)[1],
                'labels1': LABELS[self.dataset_name][self.mode].index(data_path_split[1]) if self.mode == 'train' else data_path_split[1],
                'label_token1': self._load_label_token(data_path_split[1]),
                'label_token_phrase1': self._load_label_token_phrase(data_path_split[1]),
                'label_neg_token1': self._load_neg_label_token(data_path_split[1]),
                'view1': (self.views.index(data_path_split[0].split('/')[-2]) < 3),
                'labels1_raw': data_path_split[1],
            }
        return sample
    
    def sample_clips(self, video_dir_path, st_frame, ed_frame, return_idx=False):
        # all_frames = os.listdir(video_dir_path)
        # all_frames = [x for x in all_frames if '_' not in x]
        
        if st_frame >= ed_frame:
            print(video_dir_path, st_frame, ed_frame)
        # Evenly divide a video into [self.num_clip] segments
        segments = np.linspace(st_frame, ed_frame-1, self.num_clip + 1, dtype=int)
        # segments = np.linspace(0, len(all_frames) - 2, self.num_clip + 1, dtype=int)
        # print(video_dir_path, st_frame, ed_frame)
        sampled_clips = []
        num_sampled_per_segment = 1 if self.mode == 'train' else 3
        index_list = []
        for i in range(num_sampled_per_segment):
            sampled_frames = []
            for j in range(self.num_clip):
                if self.mode == 'train':
                    frame_index = np.random.randint(segments[j], segments[j + 1])
                else:
                    frame_index = segments[j] + int((segments[j + 1] - segments[j]) / 4) * (i + 1)
                sampled_frames.append(self.sample_one_frame(video_dir_path, frame_index))
                index_list.append(frame_index)
            sampled_clips.append(self.transform(sampled_frames))
            # sampled_clips = self.transform(sampled_frames)    # [3, 16, 224, 224]
            # print(index_list)
        # assert 1==2
        if return_idx:
            return sampled_clips, index_list
        return sampled_clips


    def sample_one_frame(self, data_path, frame_index):
        image_tmpl='frame_{:010d}.jpg'
        # frame_path = os.path.join(data_path, str(frame_index + 1) + '.jpg')
        frame_path = os.path.join(data_path, image_tmpl.format(frame_index)).strip()

        try:
            frame = cv2.imread(frame_path)
            frame = Image.fromarray(frame[:, :, [2, 1, 0]])     # Convert RGB to BGR and transform to PIL.Image
            return frame
        except:
            logger.info('Wrong image path %s' % frame_path)
            exit(-1)

    def _load_label_token_phrase(self, label_number):
        if self.mode != 'train':
            return -1
        js_path = self.txt_path.replace('train_pairs.txt', 'label_bank_coarse.json')
        if self.dataset_name == 'CSV':
            seq_length = 20
        elif self.dataset_name =='COIN-SV':
            seq_length = 25
        elif self.dataset_name =="DIVING48-SV":
            seq_length = 4
        else:
            seq_length = 6
        out = torch.zeros(seq_length, 77)
        label_file = js.load(open(js_path))
        label_str = label_file[label_number]
        # label_str = ','.join(label_str)
        label_token = clip.tokenize(label_str, truncate=True)
        out[:len(label_str), :] = label_token
        return [out.int(), len(label_str)]

    def _load_label_token(self, label_number):
        if self.mode != 'train':
            return -1

        js_path = self.txt_path.replace('train_pairs.txt', 'label_bank_coarse.json')

        if self.dataset_name == 'CSV' and self.pool_sent:
            label_file = js.load(open(js_path))
            label_list = label_file[label_number]
            if len(label_list) <= 13:
                label_str1 = ', '.join(label_list)
                label_str2 = ' '
            else:
                label_str1 = ', '.join(label_list[:13])
                label_str2 = ', '.join(label_list[13:])

            label_token = clip.tokenize([label_str1, label_str2], truncate=True)
            return label_token

        label_file = js.load(open(js_path))
        label_str = label_file[label_number]
        label_str = ', '.join(label_str)
        
        label_token = clip.tokenize(label_str, truncate=True)
        return label_token.squeeze(dim=0)

    def _load_neg_label_token(self, label_number):
        if self.mode != 'train':
            return -1
        if self.dataset_name != 'CSV':
            return -1
        js_path = self.txt_path.replace('train_pairs.txt', 'label_bank_coarse.json')
        label_file = js.load(open(js_path))
        neg_label_number = self.neg_label_dic[label_number[:-2]]
        neg_label_tokens = []
        for neg_label in neg_label_number:
            label_str = label_file[neg_label]
            label_str = ','.join(label_str)
            label_token = clip.tokenize(label_str, truncate=True)
            label_token.squeeze(dim=0)
            neg_label_tokens.append(label_token)
        neg_label_tokens = torch.cat(neg_label_tokens, dim=0)
        return neg_label_tokens

    def _init_neg_label_dic(self, ):
        neg_label_dic = {}
        if self.dataset_name == 'CSV':
            for index in range(15):
                neg_label_number = [str(index + 1) + '.' + str(i + 1) for i in range(5)]
                neg_label_dic[str(index + 1)] = neg_label_number
        else:
            pass
        return neg_label_dic


class RandomSampler(data.Sampler):
    # randomly sample [len(self.dataset)] items from [len(self.data_list))] items

    def __init__(self, dataset, txt_path, shuffle=False):
        self.dataset = dataset
        self.data_list = [line.strip() for line in open(txt_path, 'r').readlines()]
        self.shuffle = shuffle

    def __iter__(self):

        tmp = random.sample(range(len(self.data_list)), len(self.dataset))
        if not self.shuffle:
            tmp.sort()

        # print(tmp)
        return iter(tmp)

    def __len__(self):
        return len(self.dataset)




def load_dataset(cfg, pair_data, is_clip=False):
    train_trans, test_trans = get_video_trans(is_clip=is_clip)

    ImageNet_normalization = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    if cfg.DATASET.MODE == 'train':
        transform = train_trans
        shuffle = True
    else: 
        transform = test_trans
        shuffle = False
    dataset = VerificationDataset(mode=cfg.DATASET.MODE,
                                  dataset_name=cfg.DATASET.NAME,
                                  txt_path=cfg.DATASET.TXT_PATH,
                                  normalization=ImageNet_normalization,
                                  num_clip=cfg.DATASET.NUM_CLIP,
                                  augment=cfg.DATASET.AUGMENT,
                                  num_sample=cfg.DATASET.NUM_SAMPLE,
                                  transform = transform,
                                  pair_data = pair_data
                                  )

    # sampler = RandomSampler(dataset, cfg.DATASET.TXT_PATH, cfg.DATASET.SHUFFLE)     # 注意这个sampler的用法

    loaders = data.DataLoader(dataset=dataset,
                              batch_size=cfg.TRAIN.BATCH_SIZE,
                              shuffle=shuffle,
                            #   sampler=sampler,
                              drop_last=False,
                              num_workers=cfg.DATASET.NUM_WORKERS,
                              pin_memory=True)

    
    return loaders, dataset




if __name__ == "__main__":

    import sys
    sys.path.append('/public/home/qianych/code/SVIP-Sequence-VerIfication-for-Procedures-in-Videos')
    from configs.defaults import get_cfg_defaults

    cfg = get_cfg_defaults()
    cfg.merge_from_file('configs/train_resnet_config.yml')

    train_loader = load_dataset(cfg)

    for iter, sample in enumerate(train_loader):
        print(sample.keys())
        print(sample['clips1'][0].size())
        print(sample['labels1'])
        break


