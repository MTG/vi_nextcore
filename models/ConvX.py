import torch
import torch.nn as nn
from collections import OrderedDict


class ConvX(nn.Module):
    """
        The class for a siamese network. The network contains:
            1 - 3 convolutional layers with ReLU non-linearization
            2 - 2 max-pooling layers in between first and second,
                and second and third convolutional layers
            3 - 2 fully connected layers

    """

    def __init__(self, layers, use_sigmoid):
        """
            Initializing the network
        """
        super().__init__()

        self.layers = layers[0]

        self.lin1_in = layers[1]

        self.lin1 = nn.Linear(in_features=self.lin1_in, out_features=256)
        self.fin_emb_size = 256
        self.use_sigmoid = use_sigmoid

        self.initialize_weights()

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

        x = self.layers(data)

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

        if self.use_sigmoid == 1:
            x = torch.sigmoid(x)

        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.01, mode='fan_in', nonlinearity='leaky_relu')


def make_layers(cfg, dropout=0):
    '''

    Taken from https://github.com/adobe/antialiased-cnns/

    '''
    layers = []
    in_channels = 1
    conv_c = 1
    prelu_c = 1
    maxpool_c = 1
    bn_c = 1
    for v in cfg:
        if v == 'P':
            layers += [('KeyPool', nn.MaxPool2d(kernel_size=(12, 1)))]
        elif v == 'MP':
            layers += [('MaxPool{}'.format(maxpool_c), nn.MaxPool2d(kernel_size=(1, 2)))]
            maxpool_c += 1
        elif v == 'BN':
            if isinstance(layers[-1], nn.PReLU):
                layers.pop(-1)
                prelu_c -= 1
            bn = nn.BatchNorm2d(num_features=in_channels)
            prelu = nn.PReLU(init=0.01)
            layers += [('BatchNorm{}'. format(bn_c), bn), ('PReLU{}'.format(prelu_c), prelu)]
            prelu_c = 1
            bn_c = 1
        else:
            if len(v) == 2:
                out_channels, kernel_size = v
                dilation = 1
            if len(v) == 3:
                out_channels, kernel_size, dilation = v
            conv = nn.Conv2d(in_channels=in_channels,
                             out_channels=out_channels,
                             kernel_size=kernel_size,
                             dilation=dilation,
                             bias=True)
            prelu = nn.PReLU(init=0.01)
            layers += [('Conv{}_{}'.format(conv_c, out_channels), conv), ('PReLU{}'.format(prelu_c), prelu)]
            if dropout != 0 and conv_c != 1:
                layers += [('Dropout', nn.Dropout(dropout))]
            conv_c += 1
            prelu_c += 1
            in_channels = out_channels
    return nn.Sequential(OrderedDict(layers)), in_channels

