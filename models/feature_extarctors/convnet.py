from torch import nn

from models.feature_extarctors.base import NoFlatteningBackbone


def conv_block(in_channels, out_channels):
    bn = nn.BatchNorm2d(out_channels)
    nn.init.uniform_(bn.weight)  # for pytorch 1.2 or later
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        bn,
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
    )


class ConvNet256(NoFlatteningBackbone):

    def __init__(self, with_drop=False, pooling=False):
        super().__init__()

        self.pooling = pooling

        self.drop_layer = with_drop

        self.hidden = 64
        self.layer1 = conv_block(3, self.hidden)
        self.layer2 = conv_block(self.hidden, int(1.5 * self.hidden))
        self.layer3 = conv_block(int(1.5 * self.hidden), 2 * self.hidden)
        self.layer4 = conv_block(2 * self.hidden, 4 * self.hidden)
        self.pooling_layer = nn.AvgPool2d(6)

        # self.weight = nn.Linear(4 * self.hidden, 64)
        # nn.init.xavier_uniform_(self.weight.weight)
        #
        # self.conv1_ls = nn.Conv2d(in_channels=4 * self.hidden, out_channels=1, kernel_size=3)
        # self.bn1_ls = nn.BatchNorm2d(1, eps=2e-5)
        # self.relu = nn.ReLU(inplace=True)
        # self.fc1_ls = nn.Linear(16, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.pooling:
            x = self.pooling_layer(x)
        # print(x.size())
        return x

    def output_featmap_size(self):
        if self.pooling:
            return 1
        return 6

    def output_features(self):
        return 256


class ConvNet64(NoFlatteningBackbone):

    def __init__(self, with_drop=False, x_dim=3, hid_dim=64, z_dim=64, pooling=False):
        super().__init__()

        self.pooling = pooling

        self.drop_layer = with_drop

        self.layer1 = conv_block(x_dim, hid_dim)
        self.layer2 = conv_block(hid_dim, hid_dim)
        self.layer3 = conv_block(hid_dim, z_dim)
        self.layer4 = conv_block(z_dim, z_dim)
        self.pooling_layer = nn.AvgPool2d(6)

    def forward(self, x):
        x = self.layer1(x)  # 42 x 42
        x = self.layer2(x)  # 21 x 21
        x = self.layer3(x)  # 11 x 11
        x = self.layer4(x)  # 6 x 6
        if self.pooling:
            x = self.pooling_layer(x)

        # print(x.size())
        return x

    def output_featmap_size(self):
        if self.pooling:
            return 1
        return 6

    def output_features(self):
        return 64
