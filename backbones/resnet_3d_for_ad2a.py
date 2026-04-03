import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

class ChannelPool(nn.Module):

    def forward(self, x):
        return torch.cat(
            (torch.max(x, 1)[0].unsqueeze(1),
             torch.mean(x, 1).unsqueeze(1)),
            dim=1
        )


class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv3d(
            in_planes, out_planes,
            kernel_size=kernel_size, stride=stride, padding=padding,
            dilation=dilation, groups=groups, bias=bias
        )
        self.bn = nn.BatchNorm3d(out_planes, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class SpatialGate(nn.Module):

    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 3
        self.compress = ChannelPool()
        self.spatial = BasicConv(
            2, 1, kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            relu=False
        )

    def forward(self, x):
        x_compress = self.compress(x)          
        x_out = self.spatial(x_compress)      
        scale = torch.sigmoid(x_out)           
        self.attention = scale               
        return x * self.attention               


# ---------------------------------------------------------------------------
# 3D ResNet Backbone (AD2A variant)
# ---------------------------------------------------------------------------

class ResNet3D(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 n_input_channels=1,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0):
        super().__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        self.conv1 = nn.Conv3d(
            n_input_channels,
            self.in_planes,
            kernel_size=(conv1_t_size, 7, 7),
            stride=(conv1_t_stride, 2, 2),
            padding=(conv1_t_size // 2, 3, 3),
            bias=False
        )
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0], shortcut_type)

        # --- AD2A-Specific: Spatial Attention Gate ---
        self.spatial_gate = SpatialGate()
        # ---------------------------------------------

        self.layer2 = self._make_layer(block, block_inplanes[1], layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(block, block_inplanes[2], layers[2], shortcut_type, stride=2)
        self.layer4 = self._make_layer(block, block_inplanes[3], layers[3], shortcut_type, stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.activation = nn.Identity()

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(
            out.size(0), planes - out.size(1),
            out.size(2), out.size(3), out.size(4)
        )
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()
        out = torch.cat([out.data, zero_pads], dim=1)
        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    self._downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride
                )
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion)
                )

        layers = []
        layers.append(
            block(in_planes=self.in_planes, planes=planes,
                  stride=stride, downsample=downsample)
        )
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)

        # --- AD2A-Specific: Apply spatial attention ---
        x = self.spatial_gate(x)
        # -----------------------------------------------------------

        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        return x.view(x.size(0), -1)

