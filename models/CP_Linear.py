import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Model(nn.Module):
    """
    Decomposition-Linear
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        self.channels = configs.enc_in - 1

        self.Linear = nn.Linear(self.channels * self.seq_len, self.pred_len)

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])
        
        x = self.Linear(x)

        x = x.unsqueeze(-1)
        
        return x  # [Batch, Output length, Channel]
