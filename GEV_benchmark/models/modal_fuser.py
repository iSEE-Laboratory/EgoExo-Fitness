import torch.nn as nn
import torch
# from transformer import *   # 如果单独运行改文件需要使用这个
from .transformer import *  


from yacs.config import CfgNode as CN

# 创建一个配置节点_C
_C = CN()
# _C.hidden_dim = 128
# _C.tgt_dim = 768
# _C.memory_dim = 128
# _C.num_head = 4
# _C.dim_ffm = 64
# _C.dropout = 0.4
# _C.num_layer = 2

default_cfg = _C.clone()
        
class Modal_Fuser(nn.Module):
    def __init__(self, arch=None, cfg=None, return_attn=False):
        super(Modal_Fuser, self).__init__()
        if cfg is None:
            cfg = default_cfg
            
        self.arch = arch
        self.return_attn = return_attn
        
        if self.arch == 'Tr_Dec':
            self.norm1 = LayerNorm(cfg.hidden_dim, eps=1e-5)
            self.norm2 = LayerNorm(cfg.hidden_dim, eps=1e-5)
            self.tgt_mapper = nn.Linear(cfg.tgt_dim, cfg.hidden_dim)
            self.memory_mapper = nn.Linear(cfg.memory_dim, cfg.hidden_dim)
            decoder_layer = TransformerDecoderLayer(d_model=cfg.hidden_dim, nhead=cfg.num_head, dim_feedforward=cfg.dim_ffm, dropout=cfg.dropout,
                                                    activation='relu', layer_norm_eps=1e-5, batch_first=True, return_attn=return_attn)
            decoder_norm = LayerNorm(cfg.hidden_dim, eps=1e-5)
            self.fuser = TransformerDecoder(decoder_layer, cfg.num_layer, decoder_norm, return_attn=return_attn)
        print('modal_fuser architecture: ', self.arch)

        
    def forward(self, v, t=None, v_mask=None, t_mask=None, v_len=111, t_len=165):
        """
            v: video features
            t: text features
        """
        if self.arch == None:
            return t
        if self.arch == 'Pool_Concat':
            v_mask_ = (~v_mask).int().unsqueeze(-1).expand(v.shape)
            t_ = v_len
            v = torch.sum(v*v_mask_, dim=1) # B, L, hidden_dim -> B, hidden_dim
            v = v / t_.unsqueeze(-1).expand(v.shape)
            v = v.unsqueeze(1)
            v = v.repeat(1, t.shape[1], 1)

            return torch.cat([v, t], dim=-1)
        if self.arch == 'Tr_Dec':
            t = self.tgt_mapper(t)
            v = self.memory_mapper(v)
            if  self.return_attn:
                output, attns_list = self.fuser(t, v, memory_key_padding_mask=v_mask, tgt_key_padding_mask=t_mask)
            else:
                output = self.fuser(t, v, memory_key_padding_mask=v_mask, tgt_key_padding_mask=t_mask)
            
            v_mask_ = (~v_mask).int().unsqueeze(-1).expand(v.shape)
            t_ = v_len
            v = torch.sum(v*v_mask_, dim=1) # B, L, hidden_dim -> B, hidden_dim
            v = v / t_.unsqueeze(-1).expand(v.shape)
            v = v.unsqueeze(1)
            v = v.repeat(1, t.shape[1], 1)
            
            if self.return_attn:
                return torch.cat([v, output], dim=-1), attns_list[0]
            return torch.cat([v, output], dim=-1)

if __name__ == '__main__':

    _C = CN()
    _C.hidden_dim = 128
    _C.num_head = 4
    _C.dim_ffm = 32
    _C.dropout = 0.0
    _C.num_layer = 2
    cfg = _C.clone()

    fuser = Modal_Fuser('Tr_Dec', cfg)

    vid_feature = torch.randn(2, 5, 128)
    txt_feature = torch.randn(2, 9, 128)
    output = fuser(vid_feature, txt_feature)
    print(output.shape)