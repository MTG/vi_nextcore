import torch
import torch.nn as nn
import torch.nn.functional as F

from models import PositionalEncoding
from models import SelfAtt


class SelfAtt2(nn.Module):

    def __init__(self, d_model=256, num_of_layers=1):

        super().__init__()

        self.prelu1 = nn.PReLU(init=0.01)

        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=d_model,
                               kernel_size=(12, 50),
                               bias=True)
        torch.nn.init.xavier_normal_(self.conv1.weight,
                                     gain=nn.init.calculate_gain('leaky_relu'))

        self.pos_emb = PositionalEncoding(d_model, 0)
        self.tr1 = SelfAtt(256)
        self.num_of_layers = num_of_layers
        if self.num_of_layers == 2:
            self.tr2 = SelfAtt(256)

        self.gamma = torch.nn.Parameter(torch.tensor(0.6))

        self.ff_3 = nn.Linear(d_model, 256)

    def forward(self, data):
        x = self.conv1(data)
        x = self.prelu1(x)

        x = F.max_pool2d(x, kernel_size=(12, 1))

        x = x.squeeze().permute(0, 2, 1)

        x = self.pos_emb(x)

        x = x.permute(1, 0, 2)
        x, idx = self.tr1(x)
        if self.num_of_layers == 2:
            x = self.tr2(x)

        x = x.permute(1, 2, 0)
        #x = torch.mean(x, dim=-1)'
        x = x[:, :, idx]
        x = self.ff_3(x)

        return x