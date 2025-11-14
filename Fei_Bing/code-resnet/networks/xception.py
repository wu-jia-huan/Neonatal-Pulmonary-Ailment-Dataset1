import torch
import torch.nn as nn
# from torchinfo import summary

class SeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1):
        super(SeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding,
                                   dilation, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0,
                                   dilation=1, groups=1, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, num_conv, strides=1, start_with_relu=True, channel_change=True):
        super(Block, self).__init__()
        if out_channels != in_channels or strides != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=strides, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Sequential()

        layers = []
        channels = in_channels

        if channel_change:
            layers.append(nn.ReLU(inplace=True))
            layers.append(SeparableConv(in_channels, out_channels, 3, stride=1, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            channels = out_channels

        for i in range(num_conv - 1):
            layers.append(nn.ReLU(inplace=True))
            layers.append(SeparableConv(channels, channels, 3, stride=1, padding=1))
            layers.append(nn.BatchNorm2d(channels))

        if not channel_change:
            layers.append(nn.ReLU(inplace=True))
            layers.append(SeparableConv(in_channels, out_channels, 3, stride=1, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))

        if start_with_relu:
            self.layers = nn.Sequential(*layers)
        else:
            self.layers = nn.Sequential(*layers[1:])

        if strides != 1:
            self.layers.add_module('maxpool', nn.MaxPool2d(3, strides, 1))

    def forward(self, x):
        return self.layers(x) + self.shortcut(x)

class Xception(nn.Module):
    def __init__(self, num_classes=1000):
        super(Xception, self).__init__()
        self.num_classes = num_classes

        # Entry flow
        self.entry_flow = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            Block(64, 128, 2, 2, start_with_relu=False, channel_change=True),
            Block(128, 256, 2, 2, start_with_relu=True, channel_change=True),
            Block(256, 728, 2, 2, start_with_relu=True, channel_change=True),
        )

        # Middle flow
        self.middle_flow = nn.Sequential(
            *[Block(728, 728, 3, 1, start_with_relu=True, channel_change=True) for _ in range(8)]
        )

        # Exit flow
        self.exit_flow = nn.Sequential(
            Block(728, 1024, 2, 2, start_with_relu=True, channel_change=False),
            SeparableConv(1024, 1536, 3, 1, 1),
            nn.BatchNorm2d(1536),
            nn.ReLU(inplace=True),

            SeparableConv(1536, 2048, 3, 1, 1),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.fc = nn.Linear(2048, num_classes)
        self._initialize_weights()

    def forward(self, x):
        x = self.entry_flow(x)
        x = self.middle_flow(x)
        x = self.exit_flow(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

