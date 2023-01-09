import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class ConvConcat(nn.Module):
    """
        The class for a siamese network. The network contains:
            1 - 3 convolutional layers with ReLU non-linearization
            2 - 2 max-pooling layers in between first and second,
                and second and third convolutional layers
            3 - 2 fully connected layers

    """

    def __init__(self):
        """
            Initializing the network
        """
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=128,
                               kernel_size=(12, 49),
                               padding=(0, int(49 / 2)),
                               bias=True)
        nn.init.kaiming_normal_(self.conv1.weight, a=0.01, mode='fan_in', nonlinearity='leaky_relu')
        self.prelu1 = nn.PReLU(init=0.01)

        self.conv2 = nn.Conv2d(in_channels=1,
                               out_channels=128,
                               kernel_size=(12, 65),
                               padding=(0, int(65 / 2)),
                               bias=True)
        nn.init.kaiming_normal_(self.conv2.weight, a=0.01, mode='fan_in', nonlinearity='leaky_relu')
        self.prelu2 = nn.PReLU(init=0.01)

        self.conv3 = nn.Conv2d(in_channels=1,
                               out_channels=128,
                               kernel_size=(12, 85),
                               padding=(0, int(85 / 2)),
                               bias=True)
        nn.init.kaiming_normal_(self.conv3.weight, a=0.01, mode='fan_in', nonlinearity='leaky_relu')
        self.prelu3 = nn.PReLU(init=0.01)

        self.conv4 = nn.Conv2d(in_channels=128*3,
                               out_channels=256,
                               kernel_size=(1, 5),
                               bias=True)
        nn.init.kaiming_normal_(self.conv4.weight, a=0.01, mode='fan_in', nonlinearity='leaky_relu')
        self.prelu4 = nn.PReLU(init=0.01)

        self.conv5 = nn.Conv2d(in_channels=256,
                               out_channels=256,
                               kernel_size=(1, 5),
                               dilation=(1, 20),
                               bias=True)
        nn.init.kaiming_normal_(self.conv5.weight, a=0.01, mode='fan_in', nonlinearity='leaky_relu')
        self.prelu5 = nn.PReLU(init=0.01)

        self.conv6 = nn.Conv2d(in_channels=256,
                               out_channels=256,
                               kernel_size=(1, 5),
                               bias=True)
        nn.init.kaiming_normal_(self.conv6.weight, a=0.01, mode='fan_in', nonlinearity='leaky_relu')
        self.prelu6 = nn.PReLU(init=0.01)

        self.lin1_in = 256

        self.lin1 = nn.Linear(in_features=256, out_features=256)
        self.fin_emb_size = 256

    def forward(self, data, mask_idx):
        """
            Defining a forward pass of the network

            Parameters
            ----------
            data : torch.Tensor
                Input tensor for the network

            Returns
            -------
            x : torch.Tensor
                Output tensor from the network

        """
        # passing the data through first convolutional layer
        # and applying non-linearization with ReLU

        x1 = self.prelu1(self.conv1(data))
        x2 = self.prelu2(self.conv2(data))
        x3 = self.prelu3(self.conv3(data))

        x = torch.cat((x1, x2, x3), 1)

        x = F.max_pool2d(x, kernel_size=(12, 1))

        x = self.prelu4(self.conv4(x))
        x = self.prelu5(self.conv5(x))
        x = self.prelu6(self.conv6(x))

        if mask_idx is not None:
            batch, maxlen = x.size(0), x.size(3)
            mask_idx = maxlen - mask_idx
            if torch.cuda.is_available():
                mask = (torch.arange(maxlen).float().cuda()[None, :] < mask_idx[:, None]).float()
            else:
                mask = (torch.arange(maxlen).float()[None, :] < mask_idx[:, None]).float()
            mask[mask == 0] = float('-inf')
            mask[mask == 1] = 0.0
            x = (x.permute(1, 2, 0, 3) + mask).permute(2, 0, 1, 3)

        x = torch.max(x, dim=3).values.reshape(x.shape[0], x.shape[1], x.shape[2], 1)

        x = x.view(-1, self.lin1_in)
        x = self.lin1(x)

        x = torch.sigmoid(x)

        return x
