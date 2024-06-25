import torch.nn as nn
import torch
from .I3D import I3D
import logging

class Video_backbone(nn.Module):
    def __init__(self, backbone_arch, num_class):
        super(Video_backbone, self).__init__()
        if backbone_arch == 'i3d':
            print('Using I3D backbone')
            self.backbone_arch = 'i3d'
            self.backbone = I3D(num_classes=num_class, modality='rgb', dropout_prob=0.5)
        
    def load_pretrain(self, ckpt_path):
        if self.backbone_arch == 'i3d':
            try:
                self.backbone.load_state_dict(torch.load(ckpt_path))
                print('loading ckpt done')
            except:
                assert 1==2
                pass
    
    def forward(self, video):
        """
            video: [B, 3, T, H, W]
        """
        if self.backbone_arch == 'i3d':
            # print(video.shape)
            bs, c, t, h, w = video.shape
            # 这里需要保证video的T是16的倍数
            packed_video = torch.cat([video[:, :, i:i+16].unsqueeze(0) for i in range(0, t, 16)], dim=0) # [num_clip, B, 3, 16, H, W]
            # print(packed_video.shape)
            num_clip = packed_video.shape[0]
            # print(packed_video.shape)
            packed_video = packed_video.reshape(-1, c, 16, h, w)
            # print(packed_video.shape)
            # assert 1==2
            
            feature = self.backbone(packed_video)  # [B, 1024, 1, 1, 1]
            _, feature_dim, feature_t, feature_h, feature_w = feature.shape
            
            feature = feature.reshape(num_clip, -1, feature_dim, feature_t, feature_h, feature_w)
            feature = feature.transpose(0, 1)   # [b, num_clip, 1024]
            # print(feature.shape)
            # feature = feature.mean(1)
            feature = feature.reshape(bs, num_clip, -1)
        return feature