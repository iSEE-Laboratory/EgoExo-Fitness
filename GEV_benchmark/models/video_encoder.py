import torch.nn as nn
import torch
# from transformer import *   # 如果单独运行改文件需要使用这个
from .transformer import *
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence
from yacs.config import CfgNode as CN
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

# 创建一个配置节点_C
_C = CN()
# _C.hidden_dim = 128
# _C.tgt_dim = 768
# _C.src_dim = 1024
# _C.num_head = 4
# _C.dim_ffm = 64
# _C.dropout = 0.4
# _C.num_layer = 2

default_cfg = _C.clone()


class Video_Encoder(nn.Module):
    def __init__(self, arch=None, cfg=None, v_max_length=111):
        super(Video_Encoder, self).__init__()
        if cfg is None:
            cfg = default_cfg
        self.arch = arch
        if self.arch == 'Tr_Enc':
            self.norm = LayerNorm(cfg.hidden_dim, eps=1e-5)
            self.mapper = nn.Linear(cfg.src_dim, cfg.hidden_dim)
            decoder_layer = TransformerEncoderLayer(d_model=cfg.hidden_dim, nhead=cfg.num_head,
                                                    dim_feedforward=cfg.dim_ffm, dropout=cfg.dropout,
                                                    activation='relu', layer_norm_eps=1e-5, batch_first=True)
            decoder_norm = LayerNorm(cfg.hidden_dim, eps=1e-5)
            self.encoder = TransformerEncoder(decoder_layer, cfg.num_layer, decoder_norm)
            
            self.pos_embed = nn.Parameter(
                torch.zeros(1, v_max_length, cfg.hidden_dim))  # remember to take pos_embed[1:] for tiling over time
            trunc_normal_(self.pos_embed, std=.02)
            
        elif self.arch == 'LSTM_Enc':
            self.lstm = nn.LSTM(cfg.src_dim, cfg.hidden_dim, cfg.num_layer, batch_first=False)
        print('vid_enc architecture: ', self.arch)

    def forward(self, src, src_key_padding_mask=None, v_len=111):
        """
            src: video features
            t: text features
        """
        bs = src.size(0)
        if self.arch == None:
            return src
        elif self.arch == 'Tr_Enc':
            src = self.mapper(src)
            # src = self.norm(src)
            src = src + self.pos_embed.repeat(bs, 1, 1)
            
            output = self.encoder(src, src_key_padding_mask=src_key_padding_mask)
            return output
        elif self.arch == 'LSTM_Enc':
            # print(v_len)
            t_ = v_len
            bs, max_length, _ = src.shape
            x = pack_sequence([src[i, :t, :] for i,t in enumerate(t_)], enforce_sorted=False)
            
            output, states = self.lstm(x)
            unpacked_output, lengths = pad_packed_sequence(output, batch_first=True)
            # print(unpacked_output.shape)

            pad_mask = torch.zeros([bs, max_length-max(lengths), unpacked_output.shape[-1]]).cuda()
            unpacked_output = torch.cat([unpacked_output, pad_mask], dim=1)
            
            # print(unpacked_output.shape)
            # assert 1==2
            assert torch.all(lengths.reshape(-1) == t_.cpu().reshape(-1))
            return unpacked_output
