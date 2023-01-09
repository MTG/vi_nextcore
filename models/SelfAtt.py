import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAtt(nn.Module):

    def __init__(self, d_model=256):

        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, 8)
        self.layer_n = nn.LayerNorm(d_model)
        self.ff_1 = nn.Linear(d_model, 1024)
        self.ff_2 = nn.Linear(1024, d_model)

    def forward(self, x):

        att, w = self.attn(x, x, x)
        x = self.layer_n(att + x)
        x = self.layer_n(self.ff_2(F.relu(self.ff_1(x))) + x)

        return x, torch.max(torch.sum(w, dim=1), dim=-1).indices[0]
