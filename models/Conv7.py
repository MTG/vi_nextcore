import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv7(nn.Module):
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

        self.prelu1 = nn.PReLU(init=0.01)
        self.prelu2 = nn.PReLU(init=0.01)
        self.prelu3 = nn.PReLU(init=0.01)
        self.prelu4 = nn.PReLU(init=0.01)
        self.prelu5 = nn.PReLU(init=0.01)
        self.prelu6 = nn.PReLU(init=0.01)
        self.prelu7 = nn.PReLU(init=0.01)

        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=256,
                               kernel_size=(12, 150),
                               bias=True)
        torch.nn.init.xavier_normal_(self.conv1.weight,
                                     gain=nn.init.calculate_gain('leaky_relu'))

        self.conv2 = nn.Conv2d(in_channels=256,
                               out_channels=256,
                               kernel_size=(1, 3),
                               bias=True)
        torch.nn.init.xavier_normal_(self.conv2.weight, gain=nn.init.calculate_gain('leaky_relu'))

        self.conv3 = nn.Conv2d(in_channels=256,
                               out_channels=256,
                               kernel_size=(1, 3),
                               padding=(0, int(3 / 2)),
                               bias=True)
        torch.nn.init.xavier_normal_(self.conv3.weight,
                                     gain=nn.init.calculate_gain('leaky_relu'))

        self.conv4 = nn.Conv2d(in_channels=256,
                               out_channels=256,
                               kernel_size=(1, 3),
                               padding=(0, int(3 / 2)),
                               bias=True)
        torch.nn.init.xavier_normal_(self.conv4.weight,
                                     gain=nn.init.calculate_gain('leaky_relu'))

        self.conv5 = nn.Conv2d(in_channels=256,
                               out_channels=256,
                               kernel_size=(1, 3),
                               padding=(0, int(3 / 2)),
                               bias=True)
        torch.nn.init.xavier_normal_(self.conv5.weight,
                                     gain=nn.init.calculate_gain('leaky_relu'))

        self.conv6 = nn.Conv2d(in_channels=256,
                               out_channels=256,
                               kernel_size=(1, 3),
                               padding=(0, int(3 / 2)),
                               bias=True)
        torch.nn.init.xavier_normal_(self.conv6.weight,
                                     gain=nn.init.calculate_gain('leaky_relu'))

        self.conv7 = nn.Conv2d(in_channels=256,
                               out_channels=256,
                               kernel_size=(1, 3),
                               padding=(0, int(3 / 2)),
                               bias=True)
        torch.nn.init.xavier_normal_(self.conv7.weight,
                                     gain=nn.init.calculate_gain('leaky_relu'))

        self.lin1_in = 256

        self.lin1 = nn.Linear(in_features=self.lin1_in, out_features=self.lin1_out)
        self.fin_emb_size = self.lin1_out

        if self.lin2_out is not None:
            self.lin2 = nn.Linear(in_features=self.lin1_out, out_features=self.lin2_out)
            self.fin_emb_size = self.lin2_out

        self.gamma = torch.nn.Parameter(torch.tensor(0.6))

    def forward(self, data):
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
        x = self.prelu1(x)

        x = F.max_pool2d(x, kernel_size=(12, 1))

        x = self.conv2(x)
        x = self.prelu2(x)

        x = F.max_pool2d(x, kernel_size=(1, 3))

        x = self.conv3(x)
        x = self.prelu3(x)

        x = F.max_pool2d(x, kernel_size=(1, 3))

        x = self.conv4(x)
        x = self.prelu4(x)

        x = F.max_pool2d(x, kernel_size=(1, 3))

        x = self.conv5(x)
        x = self.prelu5(x)

        x = F.max_pool2d(x, kernel_size=(1, 3))

        x = self.conv6(x)
        x = self.prelu6(x)

        x = F.max_pool2d(x, kernel_size=(1, 3))

        x = self.conv7(x)
        x = self.prelu7(x)

        x = torch.max(x, dim=3).values.reshape(x.shape[0], x.shape[1], x.shape[2], 1)

        x = x.view(-1, self.lin1_in)
        x = self.lin1(x)

        return x