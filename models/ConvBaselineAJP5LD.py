import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBaselineAJP5LD(nn.Module):
    """
        The class for a siamese network. The network contains:
            1 - 3 convolutional layers with ReLU non-linearization
            2 - 2 max-pooling layers in between first and second,
                and second and third convolutional layers
            3 - 2 fully connected layers

    """

    def __init__(self, bn=0, lin1=256, lin2=0, sum_method=0, autopool_p=0, cla=0, use_sigmoid=1, mulprelu=0, mulautop=0):
        """
            Initializing the network
        """
        super().__init__()

        self.do_bn = bn

        if self.do_bn == 1:
            do_bias = False
            self.bn1 = nn.BatchNorm2d(256)
            self.bn2 = nn.BatchNorm2d(256)
            self.bn3 = nn.BatchNorm2d(256)
            self.bn4 = nn.BatchNorm2d(256)
        else:
            do_bias = True

        if mulprelu == 0:
            self.prelu1 = nn.PReLU(init=0.01)
            self.prelu2 = nn.PReLU(init=0.01)
            self.prelu3 = nn.PReLU(init=0.01)
            self.prelu4 = nn.PReLU(init=0.01)
            self.prelu5 = nn.PReLU(init=0.01)
        else:
            self.prelu1 = nn.PReLU(num_parameters=256, init=0.01)
            self.prelu2 = nn.PReLU(num_parameters=256, init=0.01)
            self.prelu3 = nn.PReLU(num_parameters=256, init=0.01)
            self.prelu4 = nn.PReLU(num_parameters=256, init=0.01)
            self.prelu5 = nn.PReLU(num_parameters=256, init=0.01)

        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=256,
                               kernel_size=(12, 50),
                               bias=do_bias)
        nn.init.kaiming_normal_(self.conv1.weight, a=0.01, mode='fan_in', nonlinearity='leaky_relu')

        self.key_pool = nn.MaxPool2d(kernel_size=(12, 1))

        self.conv2 = nn.Conv2d(in_channels=256,
                               out_channels=256,
                               kernel_size=(1, 5),
                               bias=do_bias)
        nn.init.kaiming_normal_(self.conv2.weight, a=0.01, mode='fan_in', nonlinearity='leaky_relu')

        self.conv3 = nn.Conv2d(in_channels=256,
                               out_channels=256,
                               kernel_size=(1, 5),
                               dilation=(1, 13),
                               bias=do_bias)
        nn.init.kaiming_normal_(self.conv3.weight, a=0.01, mode='fan_in', nonlinearity='leaky_relu')

        self.conv4 = nn.Conv2d(in_channels=256,
                               out_channels=256,
                               kernel_size=(1, 5),
                               bias=do_bias)
        nn.init.kaiming_normal_(self.conv4.weight, a=0.01, mode='fan_in', nonlinearity='leaky_relu')

        self.conv5 = nn.Conv2d(in_channels=256,
                               out_channels=512,
                               kernel_size=(1, 5),
                               dilation=(1, 20),
                               bias=do_bias)
        nn.init.kaiming_normal_(self.conv5.weight, a=0.01, mode='fan_in', nonlinearity='leaky_relu')

        self.lin1_in = 256

        self.fin_emb_size = lin1

        self.cla = cla

        if self.cla != 0:
            self.lin2 = nn.Linear(in_features=256, out_features=14500)

        self.mulautop = mulautop

        if mulautop == 0:
            self.autopool_p = nn.Parameter(torch.tensor(0.).float())
        else:
            self.autopool_p = nn.Parameter(torch.zeros(256))
        self.sum_method = 3
        self.use_sigmoid = use_sigmoid

        lin_bias = True
        if self.use_sigmoid == 3:
            self.lin_bn = nn.BatchNorm1d(lin1, affine=False)
            lin_bias = False

        self.lin1 = nn.Linear(in_features=256, out_features=lin1, bias=lin_bias)

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

        x = self.conv1(data)
        if self.do_bn == 1:
            x = self.prelu1(self.bn1(x))
        else:
            x = self.prelu1(x)

        x = self.key_pool(x)

        x = self.conv2(x)
        if self.do_bn == 1:
            x = self.prelu2(self.bn2(x))
        else:
            x = self.prelu2(x)

        x = self.conv3(x)
        if self.do_bn == 1:
            x = self.prelu3(self.bn3(x))
        else:
            x = self.prelu3(x)

        x = self.conv4(x)
        if self.do_bn == 1:
            x = self.prelu4(self.bn4(x))
        else:
            x = self.prelu4(x)

        x = self.prelu5(self.conv5(x))

        """
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
        """
        if self.sum_method == 0:
            x = torch.max(x, dim=3, keepdim=True).values
        elif self.sum_method == 1:
            x = torch.mean(x, dim=3, keepdim=True)
        elif self.sum_method == 2:
            x = self.autopool(x, mask_idx)
        else:
            x_temp = x[:, :256] * self.autopool_p
            max_values = torch.max(x_temp, dim=3, keepdim=True).values
            softmax = torch.exp(x_temp - max_values)
            weights = softmax / torch.sum(softmax, dim=3, keepdim=True)
            x = torch.sum(x[:, 256:] * weights, dim=3, keepdim=True)
            #x = torch.sum(x[:, :256] * torch.nn.functional.softmax(x[:, 256:], dim=3), dim=3, keepdim=True)

        x = x.view(-1, self.lin1_in)
        x = self.lin1(x)

        if self.use_sigmoid == 1:
            emb = torch.sigmoid(x)
        elif self.use_sigmoid == 2:
            emb = torch.tanh(x)
        elif self.use_sigmoid == 3:
            emb = self.lin_bn(x)
        else:
            emb = x

        if self.cla != 0:
            x = self.lin2(emb)
            return x, emb
        else:
            return emb

    def autopool(self, data, mask_idx):
        if self.mulautop == 0:
            x = data * self.autopool_p
        else:
            x = (data.permute(0, 2, 3, 1) * self.autopool_p).permute(0, 3, 1, 2)
        max_values = torch.max(x, dim=3, keepdim=True).values
        softmax = torch.exp(x - max_values)
        weights = softmax / torch.sum(softmax, dim=3, keepdim=True)
        if mask_idx is not None:
            batch, maxlen = x.size(0), x.size(3)
            mask_idx = maxlen - mask_idx
            if torch.cuda.is_available():
                mask = (torch.arange(maxlen).float().cuda()[None, :] < mask_idx[:, None]).float()
            else:
                mask = (torch.arange(maxlen).float()[None, :] < mask_idx[:, None]).float()
            masked_weights = (weights.permute(1, 2, 0, 3) * mask).permute(2, 0, 1, 3)
            weights = masked_weights / masked_weights.sum(3, keepdim=True)
        values = torch.sum(data * weights, dim=3, keepdim=True)

        return values
