import os
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo


__all__ = ['ResNet', 'resnet18', 'Modified_Model', 'Modified_Model_npic', 'Modified_Model_npic_SPGAN', 'Modified_Model_npic_feature', 'Model_lstm', 'Fine_Tuned_Model', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

res18_model_name = r'resnet18-5c106cde.pth'

model_urls = {
    'resnet18': '/nfs/home/yue/ICCV/code/resnet_pretrained/resnet18-5c106cde.pth',
    'resnet34': '/nfs/home/yue/ICCV/code/resnet_pretrained/resnet34-333f7ec4.pth',
    'resnet50': '/nfs/home/yue/ICCV/code/resnet_pretrained/resnet50-19c8e357.pth',
    'resnet101': '/nfs/home/yue/ICCV/code/resnet_pretrained/resnet101-5d3b4d8f.pth',
    'resnet152': '/nfs/home/yue/ICCV/code/resnet_pretrained/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

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


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        # self.softmax = nn.functional.softmax()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

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
        # x = self.softmax(x)

        return x

class ResNet_Removed(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
        super(ResNet_Removed, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

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

        return x

class ResNet_Feature(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
        super(ResNet_Feature, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        feature_1 = x
        x = self.layer4(x)
        feature_2 = x

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return x, feature_1, feature_2

class Classifier_Model(nn.Module):
    def __init__(self, block, num_classes=3):
        super(Classifier_Model, self).__init__()
        self.fc_1 = nn.Sequential(
            nn.Linear(512 * block.expansion, 64),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.fc_2 = nn.Sequential(
            nn.Linear(64, num_classes),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.fc_1(x)
        x = self.fc_2(x)

        return x

class FC_3(nn.Module):
    def __init__(self, block, num_classes=3):
        super(FC_3, self).__init__()
        self.fc_3 = nn.Sequential(
            nn.Linear(512 * block.expansion, 64),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        x = self.fc_3(x)
    
        return x

class FC_1_lstm(nn.Module):
    def __init__(self, input_classes=3, num_classes=3):
        super(FC_1_lstm, self).__init__()
        self.fc_1 = nn.Sequential(
            nn.Linear(512, 128), # lstm: 512, Micro_3: 1024
            # nn.Sigmoid(),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.fc_1(x)
    
        return x

class FC_1(nn.Module):
    def __init__(self, input_classes=3, num_classes=3):
        super(FC_1, self).__init__()
        self.fc_1 = nn.Sequential(
            nn.Linear(1024, 128), # lstm: 512, Micro_3: 1024
            # nn.Sigmoid(),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.fc_1(x)
    
        return x

class FC_1_origin(nn.Module):
    def __init__(self, input_classes=3, num_classes=3):
        super(FC_1_origin, self).__init__()
        self.fc_1 = nn.Sequential(
            # nn.Linear(512, 128), # lstm: 512, Micro_3: 1024
            # # nn.Sigmoid(),
            # nn.ReLU(),
            nn.Linear(64 * input_classes, num_classes)
        )

    def forward(self, x):
        x = self.fc_1(x)
    
        return x

class Fine_Tuning_Model(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
        super(Fine_Tuning_Model, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.classifier = nn.Sequential(     
            nn.Linear(512 * block.expansion, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 7)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

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
        # x = self.fc(x)
        x = self.classifier(x)

        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict( torch.load(model_urls['resnet18']))
    return model

    # model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    # if pretrained:
    #     torch_home = os.path.expanduser(os.getenv('TORCH_HOME', '~/.torch'))
    #     model_dir = os.getenv('TORCH_MODEL_ZOO', os.path.join(torch_home, 'models'))
    #     model_path = os.path.join(model_dir, res18_model_name)
    #     model_param = torch.load(model_path)
    #     for name, value in model_param.items():
    #         if name.startswith('fc'):
    #             continue
    #         else:
    #             if isinstance(value, torch.Tensor):
    #                 model.state_dict()[name].copy_(value)
    #             else:
    #                 model.state_dict()[name].copy_(value.data)
    return model

def Modified_Model(num_classes=3, **kwargs):

    model_resnet = ResNet_Removed(BasicBlock, [2, 2, 2, 2], **kwargs)
    model_classifier = Classifier_Model(BasicBlock, num_classes=num_classes)

    return model_resnet, model_classifier

def Modified_Model_npic(num_pic=3, num_classes=3, **kwargs):

    model_resnet = []
    model_FC = []
    for i in range(num_pic):
        model_resnet.append(ResNet_Removed(BasicBlock, [2, 2, 2, 2], **kwargs))  #model_resnet.append(ResNet_Removed(BasicBlock, [2, 2, 2, 2], **kwargs))
        model_FC.append(FC_3(BasicBlock))

    model_classifier = FC_1_origin(input_classes=num_pic, num_classes=num_classes)

    return model_resnet, model_FC, model_classifier

def Modified_Model_npic_SPGAN(num_pic=6, num_classes=3, **kwargs):

    model_resnet = []
    model_FC = []
    for i in range(num_pic):
        model_resnet.append(ResNet_Removed(BasicBlock, [2, 2, 2, 2], **kwargs))  #model_resnet.append(ResNet_Removed(BasicBlock, [2, 2, 2, 2], **kwargs))
        model_FC.append(FC_3(BasicBlock))

    model_classifier = FC_1_origin(input_classes=4, num_classes=num_classes)

    return model_resnet, model_FC, model_classifier

def Modified_Model_npic_feature(num_pic=3, num_classes=3, **kwargs):

    model_resnet = []
    model_FC = []
    for i in range(num_pic):
        model_resnet.append(ResNet_Feature(BasicBlock, [2, 2, 2, 2], **kwargs))  #model_resnet.append(ResNet_Removed(BasicBlock, [2, 2, 2, 2], **kwargs))
        model_FC.append(FC_3(BasicBlock))

    model_classifier = FC_1_origin(input_classes=num_pic, num_classes=num_classes)

    return model_resnet, model_FC, model_classifier

def Model_lstm(batch_size=32, num_classes=3, **kwargs):

    model_resnet = ResNet_Removed(BasicBlock, [2, 2, 2, 2], **kwargs)

    model_lstm = nn.LSTM(input_size=512, hidden_size=512)

    model_classifier = FC_1_lstm(input_classes=3, num_classes=3)

    return model_resnet, model_lstm, model_classifier

def Fine_Tuned_Model(**kwargs):
    resnet = resnet18(pretrained=True)
    model = Fine_Tuning_Model(BasicBlock, [2, 2, 2, 2], **kwargs)
    pretrained_dict = resnet.state_dict()
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model

def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict( torch.load(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict( torch.load(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict( torch.load(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict( torch.load(model_urls['resnet152']))
    return model