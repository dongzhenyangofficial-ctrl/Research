import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from functools import partial
import math


def conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3D(
        in_channels=in_planes,
        out_channels=out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias_attr=False
    )


class Bottleneck(nn.Layer):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.stride = stride

        self.conv1 = nn.Conv3D(in_planes, planes, kernel_size=1, bias_attr=False)
        self.bn1 = nn.BatchNorm3D(planes)
        self.conv2 = nn.Conv3D(planes, planes, kernel_size=3, stride=stride, padding=1, bias_attr=False)
        self.bn2 = nn.BatchNorm3D(planes)
        self.conv3 = nn.Conv3D(planes, planes * self.expansion, kernel_size=1, bias_attr=False)
        self.bn3 = nn.BatchNorm3D(planes * self.expansion)
        self.relu = nn.ReLU()

        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Layer):
    def __init__(self,
                 block,
                 layers,
                 sample_size,
                 sample_duration,
                 shortcut_type='B',
                 num_classes=8):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv3D(3, 64, kernel_size=7, stride=(1, 2, 2), padding=(3, 3, 3), bias_attr=False)
        self.bn1 = nn.BatchNorm3D(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool3D(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], shortcut_type, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], shortcut_type, stride=2)
        last_duration = int(math.ceil(sample_duration / 16))
        last_size = int(math.ceil(sample_size / 32))
        self.avgpool = nn.AvgPool3D((last_duration, last_size, last_size), stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.sublayers():
            if isinstance(m, nn.Conv3D):
                nn.initializer.KaimingNormal()(m.weight)
            elif isinstance(m, nn.BatchNorm3D):
                nn.initializer.Constant(1.0)(m.weight)
                nn.initializer.Constant(0.0)(m.bias)

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        # downsample case
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                assert True, 'Not implemented!'
            else:
                downsample = nn.Sequential(
                    nn.Conv3D(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride, bias_attr=False),
                    nn.BatchNorm3D(planes * block.expansion),
                )
        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample))
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
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet101(n_classes, sample_duration, sample_size):
    """Constructs a 3D ResNet-101 model."""
    model = ResNet(block=Bottleneck, layers=[3, 4, 23, 3], shortcut_type='B', num_classes=n_classes,
                   sample_duration=sample_duration, sample_size=sample_size)
    return model


def pretrained_resnet101(snippet_duration: int,
                         sample_size: int,
                         n_classes=8,
                         ft_begin_index=5,
                         pretrained_resnet101_path="/home/aistudio/VAANet/weights/resnet-101-kinetics.pdparams"):
    n_finetune_classes = 700
    model = resnet101(n_classes, snippet_duration, sample_size)
    # model = model.cuda()
    print('Loading pretrained 3D ResNet-101 {}'.format(pretrained_resnet101_path))
    pretrain = paddle.load(pretrained_resnet101_path)
    # ---------------------------------------------------------------- #
    model.fc = nn.Linear(model.fc.weight.shape[0], n_finetune_classes)
    # model.fc = model.fc.cuda()
    # ---------------------------------------------------------------- #
    # new_state_dict = {}
    # old_state_dict = pretrain['state_dict']
    # for name in old_state_dict:
    #     new_name = name[7:]
    #     new_state_dict[new_name] = old_state_dict[name]
    model.set_state_dict(pretrain['model']['state_dict'])
    # ---------------------------------------------------------------- #
    model.fc = nn.Linear(model.fc.weight.shape[0], n_classes)
    # model.fc = model.fc.cuda()
    parameters = get_fine_tuning_parameters(model, ft_begin_index)
    return model, parameters


def get_fine_tuning_parameters(model, ft_begin_index):
    if ft_begin_index == 0:
        return model.parameters()

    ft_module_names = []
    for i in range(ft_begin_index, 5):
        ft_module_names.append('layer{}'.format(i))
    ft_module_names.append('fc')

    parameters = []
    for k, v in model.named_parameters():
        for ft_module in ft_module_names:  # fc
            if ft_module in k:
                parameters.append({'params': v, 'lr': 2e-4})
                break
        else:
            parameters.append({'params': v, 'lr': 0.0})
    return parameters
