from torch import nn

from models.feature_extarctors.base import NoFlatteningBackbone

ceil = True
inp = False


class ResNetBlock(nn.Module):
    def __init__(self, inplanes, planes):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=inplanes, out_channels=planes, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(planes, eps=2e-5)
        self.conv2 = nn.Conv2d(in_channels=planes, out_channels=planes, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(planes, eps=2e-5)
        self.conv3 = nn.Conv2d(in_channels=planes, out_channels=planes, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(planes, eps=2e-5)

        self.convr = nn.Conv2d(in_channels=inplanes, out_channels=planes, kernel_size=3, padding=1)
        self.bnr = nn.BatchNorm2d(planes, eps=2e-5)

        self.relu = nn.ReLU(inplace=inp)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=ceil)

    def forward(self, x):
        identity = self.convr(x)
        identity = self.bnr(identity)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        x += identity
        x = self.relu(x)
        x = self.maxpool(x)
        return x


class ResNet12(NoFlatteningBackbone):
    def __init__(self, drop_ratio=0.1, with_drop=True):
        super(ResNet12, self).__init__()

        self.drop_layers = with_drop
        self.inplanes = 3
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(drop_ratio, inplace=inp)
        self.layer1 = self._make_layer(ResNetBlock, 64)
        self.layer2 = self._make_layer(ResNetBlock, 128)
        self.layer3 = self._make_layer(ResNetBlock, 256)
        self.layer4 = self._make_layer(ResNetBlock, 512)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='conv2d')
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes):
        layers = [block(self.inplanes, planes)]
        self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.dropout(x)

        x = self.layer2(x)
        x = self.dropout(x)

        x = self.layer3(x)
        x = self.dropout(x)

        x = self.layer4(x)
        x = self.dropout(x)

        # if self.drop_layers:
        #     return [x4, x3]
        # else:
        #     return [x4]

        # print(x.shape)

        return x

    def output_featmap_size(self):
        return 6

    def output_features(self):
        return 512
