import math
from typing import Callable

import clip
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
# from torchsummary import summary

from utils.builder import Builder


class baseline(nn.Module):
    def __init__(self,
                 model_log=False,
                 num_class=20,
                 num_clip=16,
                 dim_embedding=128,
                 feature_encoder='resnet',
                 pretrain=None,
                 dropout=0,
                 concat_sim=False,
                 device='cpu',
                 # temperature=13.544,
                 temperature=13.544,
                 use_TE=False,
                 use_text=False,
                 use_gumbel=False,
                 use_SeqAlign=False,
                 freeze_backbone=True):
        """

        :param num_class: the number of total steps. Such as 37 in coin-sv.
        :param num_clip:
        :param dim_embedding:
        :param input_mode:
            0: (default) input as SVIP, which is 16 frames sampled from raw video. [bs*num_clip,c:=3,h,w]
            1: (for s3dg) input 16 segments and everyone has 16 frames. [bs*num_clip, 16,c:=3,h,w]
        :param pretrain: load pretrain model
        :param dropout:
        :param use_TE:
        :param use_SeqAlign:
        :param freeze_backbone:
        """
        super().__init__()
        self.model_log = model_log

        self.num_clip = num_clip
        self.use_TE = use_TE
        self.scale = 4
        self.use_SeqAlign = use_SeqAlign
        self.freeze_backbone = freeze_backbone
        # print('freeze_backbone:', freeze_backbone)
        self.feature_encoder = feature_encoder
        # self.dim_embedding = dim_embedding

        self.pretrain = pretrain
        self.use_text = use_text
        self.use_gumbel = use_gumbel

        # self.sim_dist = sim_dist
        self.concat_sim = concat_sim
        self.conv_channels = 64 if self.concat_sim else 64
        # self.add_sim = add_sims
        self.head = 8

        self.temperature = temperature
        self.dim_embedding = dim_embedding if not self.use_text else 512
        self.backbone, module_builder = self._init_backbone()

        # self.clip_model, _ = clip.load("ViT-B/32", device='cpu', jit=False)

        self.seq_features_extractor = module_builder.build_seq_features_extractor()

        self.TE = module_builder.build_transformer_encoder()

        self.embed_head = module_builder.build_embed_head()
        self.vit_dim = 1024
        self.ffn3 = nn.Linear(1024, self.vit_dim)
        self.ffn_seq = nn.Linear(512, self.vit_dim)
        self.ffn_gumbel = nn.Linear(self.vit_dim, 512)
        self.dropout = nn.Dropout(dropout)
        self.cls_fc = nn.Linear(self.dim_embedding, num_class)
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1 / 0.07)))
        self.logit_scale2 = nn.Parameter(torch.log(torch.tensor(1 / 0.07)))
        self.initialize_weights()

    def _init_backbone(self):
        module_builder = Builder(num_clip=self.num_clip, pretrain=self.pretrain, use_TE=self.use_TE,
                                 use_gumbel=self.use_gumbel, dim_embedding=self.dim_embedding)
        clip_model, _ = clip.load("ViT-B/32", device='cpu', jit=False)
        return clip_model, module_builder

    def initialize_weights(self):
        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        # torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self._init_ffn(self.ffn3)
        self._init_ffn(self.ffn_seq)
        self._init_ffn(self.ffn_gumbel)
        # self._init_ffn(self.cls_fc)

    @staticmethod
    def _init_ffn(m):
        torch.nn.init.xavier_uniform_(m.weight)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)

    @staticmethod
    def _init_conv(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            nn.init.xavier_uniform_(m.weight)
            # torch.nn.init.kaiming_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def train(self, mode=True, ):
        """
        Override the default train() to freeze the backbone
        :return:
        """
        super().train(mode)
        print_log = False
        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        else:
            for param in self.backbone.parameters():
                param.requires_grad = True


    def forward(self, x, text_token=None, text_token_phrase=None, neg_text_token=None, embed=False, retrieval=False):


        text_feature = None
        all_text_feature = None
        text_feature_phrase = None
        
        if not embed and self.use_text:
            # liym: 这里的文本编码器是无梯度计算的
            # txt_feature [b,77]
            with torch.no_grad():
                if len(text_token.shape) == 3:
                    text_token = rearrange(text_token, 'b n d -> (b n) d', n=2)
                    text_feature = self.backbone.encode_text(text_token)
                    text_feature = rearrange(text_feature, '(b n) d -> b n d', n=2)
                    text_feature = text_feature.mean(dim=1)
                elif len(text_token.shape) == 2:
                    text_feature = self.backbone.encode_text(text_token)
                else:
                    raise ValueError('text_token shape:', text_token.shape)

                _, n, _ = text_token_phrase.shape
                text_token_phrase = rearrange(text_token_phrase, 'b n d -> (b n) d')
                text_feature_phrase = self.backbone.encode_text(
                    text_token_phrase)  # [b,10,77] -> [b,20,77] -> [b,20,512] ->(mask) [b,20,512] -> loss.backward()
                text_feature_phrase = rearrange(text_feature_phrase, '(b n) d -> b n d', n=n)

        logit_scale = self.logit_scale.exp()
        logit_scale2 = self.logit_scale2.exp()

        if self.feature_encoder == 'clip':
            # print(x.shape)  # b, 3, 16. 224. 224.
            
            bs, c, self.num_clip, h, w = x.size()
            x = x.permute(0, 2, 1, 3, 4)
            x = x.reshape(-1, c, h, w)  # [B*num_clip, 3, 224, 224]
            
            x = self.backbone.encode_image(x)  # [bs * num_clip, 512]
            
            x = self.ffn_seq(x)  # -> [b*t,vit_dim]
            x = rearrange(x, '(b t) d -> b t d', t=self.num_clip)
            
            x = F.relu(self.ffn3(x))
            
            x = self.TE(x, embedded=True)  # ->[bs, num_clip, 1024]
            if self.use_gumbel:
                x = self.ffn_gumbel(x)
            

            if self.use_SeqAlign:
                # seq_features: [b,num_clip,1024]
                seq_features = x
            else:
                seq_features = None
            # x = F.relu(x)

            x = self.embed_head(x)

            if embed:
                return x
            else:
                embed_x = x
                # x = F.relu(x)
            x = self.dropout(x)

            x = self.cls_fc(x)
            
            if not self.use_text:

                return x, seq_features
            
            return x, text_feature, seq_features, embed_x, logit_scale, all_text_feature, text_feature_phrase, logit_scale2


        # ------------------------------------------------------------------------------------------------------



if __name__ == "__main__":
    import torch
    input = torch.rand(4 * 16, 3, 224, 224)
    txt_token = torch.randint(1, 5000, [4, 77])
    model = baseline(use_TE=True, model_log=False, use_sim=True, sim_dist='l2', use_text=True, sim_pos='first',
                     use_SeqAlign=True,
                     feature_encoder='clip', freeze_backbone=False, concat_sim=True, add_sim=False, num_clip=16)
    model.train()
    # summary(model, (16, 3, 224, 224), depth=3)
    _, _, _ = model(input, txt_token)
    # sims = Similarity_matrix(num_heads=8, model_dim=512)
    # x = torch.rand(16,16,512)
    # sim = sims(x,x,x)
    # print(sim.shape)
