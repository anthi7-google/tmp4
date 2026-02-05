import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils import weight_norm

#Focal Channel Recovery Mixer

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        self.C = configs.enc_in-1 #channels
        self.D = configs.cnn_hidden

        self.emb_kernel = configs.xt_emb_kernel
        self.in_embed = nn.Conv1d(self.C, self.D, self.emb_kernel, padding='same')

        self.Focal_layers = nn.ModuleList([
            FocalBlock(configs)
            for _ in range(configs.cen_num_layers)
        ])
     
        self.Linear = nn.Linear(self.D, 1)

    def forward(self, x):
        #print("x:size = ", x.shape)
        B, T, C = x.shape

        x = x.permute(0,2,1)          # [B, C, T]
        x = self.in_embed(x)          # [B, D, T]

        for layer in self.Focal_layers:
            x = layer(x)                 # [B, D, T]
        x = x.permute(0,2,1)          # [B, T, D]
        x = self.Linear(x)            # [B, T, 1]

        return x


class FocalBlock(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.C = configs.enc_in-1 #channels
        self.D = configs.cnn_hidden

        ch_in = self.D
        ch_out = self.D 

        self.kernel = configs.cnn_kernel

        #Channel-wise Temporal Filter
        self.ctf = nn.Conv1d(ch_in, ch_out, self.kernel, padding='same', groups=ch_in)
        self.bn = nn.BatchNorm1d(ch_out)

        a1_init= 0.9
        a2_init= 0.1
        self.a1 = nn.Parameter(torch.tensor([a1_init], dtype=torch.float32))
        self.a2 = nn.Parameter(torch.tensor([a2_init], dtype=torch.float32))

        lf = 10
        self.a1.register_hook(lambda grad: grad * lf)
        self.a2.register_hook(lambda grad: grad * lf)

        self.fcm1 = FocalChannelMixer(ch_out, 2*ch_out, kernel_size = 3)
        self.fcm2 = FocalChannelMixer(2*ch_out, ch_out, kernel_size = 3)

        self.act = nn.GELU()#nn.ReLU()

        drop = configs.xt_dropout
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        a1 = self.a1
        a2 = self.a2
        a1 = a1 + (a1.clamp(0.01, 0.99) - a1).detach()
        a2 = a2 + (a2.clamp(0.01, 0.99) - a2).detach()
        res = x       # [B, C, T]
        x=self.ctf(x)   # [B, C, T]
        x=self.bn(x)   # [B, C, T]
        x=self.fcm1(x,a1)  # [B, 2C, T]
        x=self.act(x)  # [B, 2C, T]
        x=self.fcm2(x,a2)  # [B, C, T]
        x=self.drop(x) # [B, C, T]
        x=x+res

        return x


class FocalChannelMixer(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size=3, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, padding='same', **kwargs)

        center = kernel_size // 2
        dist = torch.tensor([abs(i - center) for i in range(kernel_size)], dtype=torch.float32)
        # (1, 1, kernel_size) 형태로 뷰 변경
        self.register_buffer('dist_map', dist.view(1, 1, -1))

    def forward(self, x, retention_rate):
        focal_decay = retention_rate ** self.dist_map

        focused_weight = self.weight * focal_decay
        
        return F.conv1d(x, focused_weight, self.bias, self.stride, 
                        self.padding, self.dilation, self.groups)
