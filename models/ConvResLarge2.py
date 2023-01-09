import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class ConvResLarge2(nn.Module):
    """
        The class for a siamese network. The network contains:
            1 - 3 convolutional layers with ReLU non-linearization
            2 - 2 max-pooling layers in between first and second,
                and second and third convolutional layers
            3 - 2 fully connected layers

    """

    def __init__(self, sum_method=0, autopool_p=1, cla=0):
        """
            Initializing the network
        """
        super().__init__()

        self.input_conv = nn.Conv2d(in_channels=1,
                                    out_channels=256,
                                    kernel_size=(12, 50),
                                    bias=False)
        nn.init.kaiming_normal_(self.input_conv.weight, a=0.01, mode='fan_out', nonlinearity='leaky_relu')
        self.prelu1 = nn.PReLU(init=0.01)
        self.bn1 = nn.BatchNorm2d(256)
        nn.init.constant_(self.bn1.weight, 1)
        nn.init.constant_(self.bn1.bias, 0)

        self.block1 = BasicBlock()
        self.block2 = BasicBlock()
        self.block3 = BasicBlock()

        self.conv_dil1 = nn.Conv2d(in_channels=256,
                               out_channels=256,
                               kernel_size=(1, 5),
                               padding=(0, 10),
                               dilation=(1, 5),
                               bias=False)
        nn.init.kaiming_normal_(self.conv_dil1.weight, a=0.01, mode='fan_out', nonlinearity='leaky_relu')
        self.prelu2 = nn.PReLU(init=0.01)
        self.bn2 = nn.BatchNorm2d(256)
        nn.init.constant_(self.bn2.weight, 1)
        nn.init.constant_(self.bn2.bias, 0)

        self.block4 = BasicBlock()
        self.block5 = BasicBlock()
        self.block6 = BasicBlock()

        self.conv_dil2 = nn.Conv2d(in_channels=256,
                                   out_channels=256,
                                   kernel_size=(1, 5),
                                   padding=(0, 10),
                                   dilation=(1, 5),
                                   bias=False)
        nn.init.kaiming_normal_(self.conv_dil2.weight, a=0.01, mode='fan_out', nonlinearity='leaky_relu')
        self.prelu3 = nn.PReLU(init=0.01)
        self.bn3 = nn.BatchNorm2d(256)
        nn.init.constant_(self.bn3.weight, 1)
        nn.init.constant_(self.bn3.bias, 0)

        self.block7 = BasicBlock()
        self.block8 = BasicBlock()
        self.block9 = BasicBlock()

        self.conv_dil3 = nn.Conv2d(in_channels=256,
                                   out_channels=256,
                                   kernel_size=(1, 5),
                                   padding=(0, 10),
                                   dilation=(1, 5),
                                   bias=False)
        nn.init.kaiming_normal_(self.conv_dil3.weight, a=0.01, mode='fan_out', nonlinearity='leaky_relu')
        self.prelu4 = nn.PReLU(init=0.01)
        self.bn4 = nn.BatchNorm2d(256)
        nn.init.constant_(self.bn4.weight, 1)
        nn.init.constant_(self.bn4.bias, 0)

        self.block10 = BasicBlock()
        self.block11 = BasicBlock()
        self.block12 = BasicBlock()

        self.block13 = BasicBlock()
        self.block14 = BasicBlock()
        self.block15 = BasicBlock()

        self.block16 = BasicBlock()
        self.block17 = BasicBlock()
        self.block18 = BasicBlock()

        self.block19 = BasicBlock()
        self.block20 = BasicBlock()
        self.block21 = BasicBlock()

        self.bn5 = nn.BatchNorm2d(256)
        nn.init.constant_(self.bn5.weight, 1)
        nn.init.constant_(self.bn5.bias, 0)

        self.lin1_in = 256

        self.lin1 = nn.Linear(in_features=256, out_features=256)
        self.fin_emb_size = 256

        self.autopool_p = nn.Parameter(torch.tensor(float(autopool_p)))
        self.sum_method = sum_method
        self.cla = cla

        if self.cla == 1:
            self.lin2 = nn.Linear(in_features=256, out_features=14500)


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

        x = self.prelu1(self.bn1(self.input_conv(data)))

        x = F.max_pool2d(x, kernel_size=(12, 1))

        x = self.block3(self.block2(self.block1(x)))

        x = self.prelu2(self.bn2(self.conv_dil1(x)))

        x = F.max_pool2d(x, kernel_size=(1, 2))

        x = self.block6(self.block5(self.block4(x)))

        x = self.prelu3(self.bn3(self.conv_dil2(x)))

        x = F.max_pool2d(x, kernel_size=(1, 2))

        x = self.block9(self.block8(self.block7(x)))

        x = self.prelu4(self.bn4(self.conv_dil3(x)))

        x = self.block12(self.block11(self.block10(x)))

        x = self.block15(self.block14(self.block13(x)))

        x = self.block18(self.block17(self.block16(x)))

        x = self.block21(self.block20(self.block19(x)))

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

        if self.sum_method == 0:
            x = torch.max(x, dim=3, keepdim=True).values
        if self.sum_method == 1:
            x = torch.mean(x, dim=3, keepdim=True)
        else:
            x = self.autopool(x)

        x = self.bn5(x)

        x = x.view(-1, self.lin1_in)
        x = self.lin1(x)

        emb = torch.sigmoid(x)

        if self.cla == 1:
            x = self.lin2(F.relu(emb))
            return x, emb
        else:
            return emb

    def autopool(self, data):
        x = data * self.autopool_p
        max_values = torch.max(x, dim=3, keepdim=True).values
        softmax = torch.exp(x - max_values)
        weights = softmax / torch.sum(softmax, dim=3, keepdim=True)
        values = torch.sum(data * weights, dim=3, keepdim=True)

        return values


class BasicBlock(nn.Module):
    def __init__(self):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=256,
                               out_channels=256,
                               kernel_size=(1, 5),
                               padding=(0, int(5/2)),
                               bias=False)
        nn.init.kaiming_normal_(self.conv1.weight, a=0.01, mode='fan_out', nonlinearity='leaky_relu')
        self.prelu1 = nn.PReLU(init=0.01)

        self.bn1 = nn.BatchNorm2d(256)
        nn.init.constant_(self.bn1.weight, 1)
        nn.init.constant_(self.bn1.bias, 0)

        self.conv2 = nn.Conv2d(in_channels=256,
                               out_channels=256,
                               kernel_size=(1, 5),
                               padding=(0, int(5/2)),
                               bias=False)
        nn.init.kaiming_normal_(self.conv2.weight, a=0.01, mode='fan_out', nonlinearity='leaky_relu')
        self.prelu2 = nn.PReLU(init=0.01)
        self.bn2 = nn.BatchNorm2d(256)
        nn.init.constant_(self.bn2.weight, 0)
        nn.init.constant_(self.bn2.bias, 0)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.prelu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.prelu2(out)

        return out

