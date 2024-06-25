import json as js
import logging
import os
import random

import clip
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils import data
from torchvision import transforms as tf
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

from Data.label import LABELS
from Data.sampler_ddp import DistributedSampler
from utils.data_prefetcher import data_prefetcher

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
logger = logging.getLogger('Sequence Verification')


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


class VerificationDataset(data.Dataset):

    def __init__(self,
                 mode='train',
                 dataset_name='CSV',
                 txt_path=None,
                 normalization=None,
                 num_clip=16,
                 augment=False,
                 random_sample=True,
                 image_size=224,
                 shuffle=False,
                 pair_data=True,
                 data_aug=True,
                 pool_sent=False,
                 feature_encoder='resnet',
                 num_sample=600):

        assert mode in ['train', 'test',
                        'val'], 'Dataset mode is expected to be train, test, or val. But get %s instead.' % mode
        self.mode = mode
        self.dataset_name = dataset_name
        self.normalization = normalization
        self.num_clip = num_clip
        self.augment = augment
        self.shuffle = shuffle
        self.pair_data = pair_data
        self.data_aug = data_aug
        self.pool_sent = pool_sent
        self.random_sample = random_sample
        self.feature_encoder = feature_encoder
        self.image_H, self.image_W = self._set_image_rs(self.feature_encoder)
        self.txt_path = txt_path
        if not self.pair_data and self.mode != 'test' and self.mode != 'val':
            txt_path = txt_path.replace('train_pairs.txt', 'train.txt')
        self.neg_label_dic = self._init_neg_label_dic()
        self.random_sample = random_sample
        self.frame_root = '/mnt/Datasets/CSV/raw_data/frames'  # TODO：修改frame路径
        if augment:
            self.aug_flip = True
            self.aug_crop = True
            self.aug_color = True
            self.aug_rot = True
        self.num_sample = num_sample  # num of pairs randomly selected from all training pairs
        self.data_list = [line.strip() for line in open(txt_path, 'r').readlines()]
        logger.info(
            'Successfully construct dataset with [%s] mode and [%d] samples randomly selected from [%d] samples' % (
                mode, len(self), len(self.data_list)))

    def __getitem__(self, index):

        data_path = self.data_list[index]
        if self.pair_data or self.mode != 'train':
            data_path_split = data_path.strip().split(' ')
            sample = {
                # 'index': index,
                'Data': data_path,
                'clips1': self.sample_clips(data_path_split[0]),
                'clips2': self.sample_clips(data_path_split[2]),
                'labels1': LABELS[self.dataset_name][self.mode].index(data_path_split[1]) if self.mode == 'train' else
                data_path_split[1],
                'label_token1': self._load_label_token(data_path_split[1]),
                'label_token_phrase1': self._load_label_token_phrase(data_path_split[1]),
                'label_neg_token1': self._load_neg_label_token(data_path_split[1]),
                'labels2': LABELS[self.dataset_name][self.mode].index(data_path_split[3]) if self.mode == 'train' else
                data_path_split[3],
                'label_token2': self._load_label_token(data_path_split[3]),
                'label_token_phrase2': self._load_label_token_phrase(data_path_split[3]),
                'label_neg_token2': self._load_neg_label_token(data_path_split[3]),
                'labels1_raw': data_path_split[1],
                'labels2_raw': data_path_split[3],
            }
            return sample
        else:
            data_path_split = data_path.strip().split(' ')
            # print(data_path_split)
            # assert 1==2
            sample = {
                # 'index': index,
                'Data': data_path_split[0],
                'clips1': self.sample_clips(data_path_split[0]),
                'labels1': LABELS[self.dataset_name][self.mode].index(data_path_split[1]) if self.mode == 'train' else
                data_path_split[1],
                'label_token1': self._load_label_token(data_path_split[1]),
                'label_token_phrase1': self._load_label_token_phrase(data_path_split[1]),
                'label_neg_token1': self._load_neg_label_token(data_path_split[1]),
                'labels1_raw': data_path_split[1],
            }
            return sample

    def __len__(self):
        if self.mode == 'train':
            return self.num_sample
        else:
            return len(self.data_list)
        # return len(self.data_list)

    @staticmethod
    def _set_image_rs(net):
        if net == 'resnet':
            h, w = 180, 320
        else:
            h, w = 224, 224
        return h, w

    def _load_label_token_phrase(self, label_number):
        if self.mode != 'train':
            return -1
        js_path = self.txt_path.replace('train_pairs.txt', 'label_bank.json')
        if self.dataset_name == 'CSV':
            seq_length = 20
        elif self.dataset_name =='COIN-SV':
            seq_length = 25
        elif self.dataset_name =="DIVING48-SV":
            seq_length = 4
        else:
            raise ValueError('wrong dataset_name')
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

        js_path = self.txt_path.replace('train_pairs.txt', 'label_bank.json')

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
        label_str = ','.join(label_str)
        label_token = clip.tokenize(label_str, truncate=True)
        return label_token.squeeze(dim=0)

    def _load_neg_label_token(self, label_number):
        if self.mode != 'train':
            return -1
        if self.dataset_name != 'CSV':
            return -1
        js_path = self.txt_path.replace('train_pairs.txt', 'label_bank.json')
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
        #     # raise ValueError('just code for CSV')
        #     print('just code for CSV')
        return neg_label_dic

    def sample_clips(self, video_dir_path):
        frame_dir_path = os.path.join(self.frame_root, video_dir_path)
        all_frames = os.listdir(frame_dir_path)

        # all_frames = [x for x in all_frames if '_' not in x]

        # Evenly divide a video into [self.num_clip] segments
        segments = np.linspace(0, len(all_frames) - 2, self.num_clip + 1, dtype=int)

        sampled_clips = []
        num_sampled_per_segment = 1 if self.mode == 'train' else 3
        # num_sampled_per_segment =  3

        for i in range(num_sampled_per_segment):
            sampled_frames = []

            for j in range(self.num_clip):

                if self.mode == 'train':
                    if self.random_sample:
                        frame_index = np.random.randint(segments[j], segments[j + 1])
                    else:
                        # uniform
                        frame_index = segments[j]
                else:
                    frame_index = segments[j] + int((segments[j + 1] - segments[j]) / 4) * (i + 1)

                sampled_frames.append(self.sample_one_frame(frame_dir_path, frame_index))
            if self.feature_encoder != 'resnet' and not self.data_aug:
                sampled_clips.append(self.preprocess_clip(sampled_frames))
            else:
                sampled_clips.append(self.preprocess(sampled_frames))
            # sampled_clips.append(self.preprocess(sampled_frames))
        return sampled_clips

    @staticmethod
    def sample_one_frame(data_path, frame_index, use_cv2=False):
        image_tmpl='frame_{:010d}.jpg'
        # frame_path = os.path.join(data_path, str(frame_index + 1) + '.jpg')
        frame_path = os.path.join(data_path, image_tmpl.format(frame_index+1))

        try:
            if use_cv2:
                frame = cv2.imread(frame_path)
                frame = Image.fromarray(frame[:, :, [2, 1, 0]])  # Convert RGB to BGR and transform to PIL.Image
            else:
                frame = Image.open(frame_path)
            return frame
        except:
            logger.info('Wrong image path %s' % frame_path)
            exit(-1)

    @staticmethod
    def preprocess_clip(frames):
        transforms = _transform(224)
        frames = torch.cat([transforms(frame).unsqueeze(-1) for frame in frames], dim=-1)
        return frames

    def preprocess(self, frames, apply_normalization=True):
        # Apply augmentation and normalization on a clip of frames
        # Data augmentation on the frames
        transforms = []
        if self.augment:
            # Flip
            if np.random.random() > 0.5 and self.aug_flip:
                transforms.append(tf.RandomHorizontalFlip(1))

            # Random crop
            if np.random.random() > 0.5 and self.aug_crop:
                transforms.append(tf.RandomResizedCrop((180, 320), (0.7, 1.0)))
                if self.feature_encoder == 'clip':
                    transforms.append(tf.Resize(self.image_H, interpolation=BICUBIC))
                    transforms.append(tf.CenterCrop((self.image_H, self.image_W)))
            else:
                transforms.append(tf.Resize(self.image_H, interpolation=BICUBIC))
                transforms.append(tf.CenterCrop((self.image_H, self.image_W)))

            # Color augmentation
            if np.random.random() > 0.5 and self.aug_color:
                transforms.append(tf.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.5))

            # # Rotation
            # if np.random.random() > 0.5 and self.aug_rot:
            #     transforms.append(tf.RandomRotation(30))
        else:
            transforms.append(tf.Resize(self.image_H, interpolation=BICUBIC))
            transforms.append(tf.CenterCrop((self.image_H, self.image_W)))

        # PIL image to tensor
        transforms.append(tf.ToTensor())

        # Normalization
        if self.normalization is not None and apply_normalization:
            transforms.append(tf.Normalize(self.normalization[0], self.normalization[1]))

        transforms = tf.Compose(transforms)
        frames = torch.cat([transforms(frame).unsqueeze(-1) for frame in frames], dim=-1)

        return frames


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

        return iter(tmp)

    def __len__(self):
        return len(self.dataset)


def load_dataset_ddp(cfg, args, drop_last=True):
    ImageNet_normalization = ([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])

    dataset = VerificationDataset(mode=cfg.DATASET.MODE,
                                  dataset_name=cfg.DATASET.NAME,
                                  txt_path=cfg.DATASET.TXT_PATH,
                                  normalization=ImageNet_normalization,
                                  num_clip=cfg.DATASET.NUM_CLIP,
                                  augment=cfg.DATASET.AUGMENT,
                                  pair_data=args.pair,
                                  data_aug=args.data_aug,
                                  pool_sent=args.pool_sent,
                                  feature_encoder=args.backbone,
                                  random_sample=cfg.DATASET.RANDOM_SAMPLE,
                                  num_sample=cfg.DATASET.NUM_SAMPLE)
    if not args.pair and cfg.DATASET.MODE != 'test' and cfg.DATASET.MODE != 'val':
        txt_path = cfg.DATASET.TXT_PATH.replace('train_pairs.txt', 'train.txt')
        train_sampler = DistributedSampler(dataset=dataset,
                                           txt_path=txt_path,
                                           shuffle=cfg.DATASET.SHUFFLE,
                                           )
    else:
        train_sampler = DistributedSampler(dataset=dataset,
                                           txt_path=cfg.DATASET.TXT_PATH,
                                           shuffle=cfg.DATASET.SHUFFLE,
                                           )
    # train_sampler = torch.utils.data.DistributedSampler(dataset=dataset, shuffle=cfg.DATASET.SHUFFLE)
    loaders = torch.utils.data.DataLoader(dataset=dataset,
                                          batch_size=cfg.TRAIN.BATCH_SIZE,
                                          shuffle=False,
                                          sampler=train_sampler,
                                          drop_last=drop_last,
                                          num_workers=cfg.DATASET.NUM_WORKERS,
                                          pin_memory=False)

    if args.pre_load:
        prefetcher = data_prefetcher(loaders)

        return prefetcher
    else:
        return loaders


def load_dataset(cfg, pre_load=False, pair_data=True, backbone='resnet'):
    ImageNet_normalization = ([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])

    dataset = VerificationDataset(mode='train',
                                  dataset_name=cfg.DATASET.NAME,
                                  txt_path=cfg.DATASET.TXT_PATH,
                                  normalization=ImageNet_normalization,
                                  num_clip=cfg.DATASET.NUM_CLIP,
                                  augment=cfg.DATASET.AUGMENT,
                                  pair_data=pair_data,
                                  feature_encoder=backbone,
                                  random_sample=cfg.DATASET.RANDOM_SAMPLE,
                                  num_sample=cfg.DATASET.NUM_SAMPLE)

    sampler = RandomSampler(dataset, cfg.DATASET.TXT_PATH, cfg.DATASET.SHUFFLE)

    loaders = torch.utils.data.DataLoader(dataset=dataset,
                                          batch_size=cfg.TRAIN.BATCH_SIZE * torch.cuda.device_count() if torch.cuda.is_available() else cfg.TRAIN.BATCH_SIZE,
                                          shuffle=False,
                                          sampler=sampler,
                                          drop_last=False,
                                          num_workers=cfg.DATASET.NUM_WORKERS * torch.cuda.device_count() if torch.cuda.is_available() else 0,
                                          pin_memory=False)

    if pre_load:
        prefetcher = data_prefetcher(loaders)

        return prefetcher
    else:
        return loaders
    # return loaders


#
if __name__ == "__main__":
    # #
    # #     import sys
    # #     sys.path.append('/public/home/dongsx/svip')
    from configs.defaults import get_cfg_defaults

    ImageNet_normalization = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    cfg = get_cfg_defaults()
    test_cfg = get_cfg_defaults()
    cfg.merge_from_file('/public/home/dongsx/svip/configs/train_config.yml')
    test_cfg.merge_from_file('/public/home/dongsx/svip/configs/eval_resnet_config.yml')

    dataset = VerificationDataset(mode=cfg.DATASET.MODE,
                                  dataset_name=cfg.DATASET.NAME,
                                  txt_path=cfg.DATASET.TXT_PATH,
                                  normalization=ImageNet_normalization,
                                  num_clip=cfg.DATASET.NUM_CLIP,
                                  augment=cfg.DATASET.AUGMENT,
                                  pair_data=True,
                                  feature_encoder='clip',
                                  random_sample=cfg.DATASET.RANDOM_SAMPLE,
                                  num_sample=cfg.DATASET.NUM_SAMPLE)

    _ = dataset[1]
    DataLoader = load_dataset(cfg, )
    for sample in DataLoader:
        print(sample)
        print(sample['labels1'])
        label_token1 = sample['label_token1']
        label_neg_token1 = sample['label_neg_token1']
        print(label_token1.unsqueeze(dim=1) == label_neg_token1)
    #     print(sample['labels2'])
