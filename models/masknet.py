from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

class MnistNet(nn.Module):
    def __init__(self,input_size, num_classes):
        super(MnistNet, self).__init__()
        #self.inNorm = nn.BatchNorm2d(3)
        self.fc1 = MaskLinear(input_size, 2000)
        self.fc2 = MaskLinear(2000, 2000)
        self.fc3 = nn.Linear(2000, num_classes)
        self.drp1 = nn.Dropout(p=.5)

    def forward(self, x):
        x = self.drp1(F.relu(self.fc1(x)))
        x = self.drp1(F.relu(self.fc2(x)))


        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

class CifarNet(nn.Module):
    def __init__(self,input_size, num_classes):
        super(CifarNet, self).__init__()
        #self.inNorm = nn.BatchNorm2d(3)
        #self.conv1 = MaskConv2d(3, input_size, 3)
        self.conv1 = MaskConv2d(3, input_size, 3, stride=1, bias=False)
        self.conv2 = MaskConv2d(32, 32, 3, stride=1, bias=False)
        self.conv3 = MaskConv2d(32, 64, 3, 1, bias=False)
        self.conv4 = MaskConv2d(64, 64, 3, 1, bias=False)
        self.fc1 = MaskLinear(5*5*64, 512)
        self.fc2 = nn.Linear(512, num_classes)   #for last layer use Linear only
        self.drp1 = nn.Dropout(p=.25)
        self.drp2 = nn.Dropout(p=.5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.drp1(F.max_pool2d(x, 2, 2))

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.drp1(F.max_pool2d(x, 2, 2))

        x = x.view(-1, 5*5*64)
        x = self.drp2(F.relu(self.fc1(x)))
        x = self.fc2(x)
        #x = self.fc3(x)
        return F.log_softmax(x, dim=1)

###orginal lidar model from infocom; most succesfull
class FlashNet(nn.Module):
    print('**************using orginal lidar model from infocom***********')
    def __init__(self,modality, num_classes, shrink=1):
        super(FlashNet, self).__init__()
        self.shrink=shrink

        dropProb1 = 0.3
        dropProb2 = 0.2
        channel = 32
        # self.conv1 = nn.Conv2d(90, channel, kernel_size=(3, 3), bias=False)
        self.conv1 = MaskConv2d(45, channel, kernel_size=(3, 3), bias=False)
        self.conv2 = MaskConv2d(channel, channel, kernel_size=(3, 3), bias=False)
        self.conv3 = MaskConv2d(channel, channel, kernel_size=(3, 3), bias=False)
        self.conv4 = MaskConv2d(channel, channel, kernel_size=(3, 3), bias=False)
        self.conv5 = MaskConv2d(channel, channel, kernel_size=(3, 3), bias=False)
        self.conv6 = MaskConv2d(channel, channel, kernel_size=(3, 3), bias=False)
        self.conv7 = MaskConv2d(channel, channel, kernel_size=(3, 3), bias=False)
        self.conv8 = MaskConv2d(channel, channel, kernel_size=(3, 3), bias=False)
        self.conv9 = MaskConv2d(channel, channel, kernel_size=(3, 3), bias=False)

        self.pool1 = nn.MaxPool2d((2,2))
        self.pool2 = nn.MaxPool2d((1, 2))

        # self.hidden1 = nn.Linear(320, 1024)  #orginal
        # self.hidden1 = nn.Linear(2560, 1024)    #with zero padding
        self.hidden1 = MaskLinear(1280, 1024)    #with zero padding image/2
        # self.hidden2 = MaskLinear(1024, 512)  # we are not using this
        self.hidden3 = MaskLinear(1024, 256)
        self.out = nn.Linear(256, 64)  # 128
        #######################
        self.drop1 = nn.Dropout(dropProb1)
        self.drop2 = nn.Dropout(dropProb2)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
    def forward(self, x):
        # FOR CNN BASED IMPLEMENTATION
        x = F.pad(x, (1, 1, 1, 1))
        a = x = self.relu(self.conv1(x))
        x = F.pad(x, (1, 1, 1, 1))
        x = self.relu(self.conv2(x))
        x = F.pad(x, (1, 1, 1, 1))
        x = self.relu(self.conv3(x))
        x = torch.add(x, a)
        x = self.pool1(x)
        b = x = self.drop1(x)

        x = F.pad(x, (1, 1, 1, 1))
        x = self.relu(self.conv4(x))
        x = F.pad(x, (1, 1, 1, 1))
        x = self.relu(self.conv5(x))
        x = torch.add(x, b)
        x = self.pool1(x)
        c = x = self.drop1(x)

        x = F.pad(x, (1, 1, 1, 1))
        x = self.relu(self.conv6(x))
        x = F.pad(x, (1, 1, 1, 1))
        x = self.relu(self.conv7(x))
        x = torch.add(x, c)
        x = self.pool2(x)
        d = x = self.drop1(x)

        x = F.pad(x, (1, 1, 1, 1))
        x = self.relu(self.conv8(x))
        x = F.pad(x, (1, 1, 1, 1))
        x = self.relu(self.conv9(x))
        x = torch.add(x, d)
        # print('xshape',x.shape)
        #########
        x = x.view(x.size(0), -1)

        # print("shape", x.shape)
        x = self.relu(self.hidden1(x))
        x = self.drop2(x)

        x = self.relu(self.hidden3(x))
        x = self.drop2(x)
        x = self.out(x)  # no softmax: CrossEntropyLoss()
        return x


# ###added batool: old
# class FlashNet(nn.Module):
#     def __init__(self,modality, num_classes, shrink=1):
#         super(FlashNet, self).__init__()
#         dropProb1 = 0.3
#         dropProb2 = 0.2
#         channel = 32
#         self.conv1 = MaskConv2d(90, channel, kernel_size=(3, 3), bias=False)
#         self.conv2 = MaskConv2d(channel, channel, kernel_size=(3, 3), bias=False)
#         self.conv3 = MaskConv2d(channel, channel, kernel_size=(3, 3), bias=False)
#         self.conv4 = MaskConv2d(channel, channel, kernel_size=(3, 3), bias=False)
#         self.conv5 = MaskConv2d(channel, channel, kernel_size=(3, 3), bias=False)
#         self.conv6 = MaskConv2d(channel, channel, kernel_size=(3, 3), bias=False)
#         self.conv7 = MaskConv2d(channel, channel, kernel_size=(3, 3), bias=False)
#         self.conv8 = MaskConv2d(channel, channel, kernel_size=(3, 3), bias=False)
#         self.conv9 = MaskConv2d(channel, channel, kernel_size=(3, 3), bias=False)

#         self.pool1 = nn.MaxPool2d((2,2))
#         self.pool2 = nn.MaxPool2d((1, 2))

#         # self.hidden1 = nn.Linear(320, 1024)  #orginal
#         self.hidden1 = MaskLinear(2560, 1024)    #with zero padding
#         # self.hidden2 = nn.Linear(1024, 512)
#         self.hidden3 = MaskLinear(1024, 256)
#         self.out = nn.Linear(256, 64)  # 128
#         #######################
#         self.drop1 = nn.Dropout(dropProb1)
#         self.drop2 = nn.Dropout(dropProb2)
#         self.relu = nn.ReLU()
#         self.tanh = nn.Tanh()
#         self.softmax = nn.Softmax()
#     def forward(self, x):
#         # FOR CNN BASED IMPLEMENTATION
#         # print('g1',x.shape)
#         x = F.pad(x, (1, 1, 1, 1))
#         a = x = self.relu(self.conv1(x))
#         x = F.pad(x, (1, 1, 1, 1))
#         # print('g3',a.shape)
#         x = self.relu(self.conv2(x))
#         x = F.pad(x, (1, 1, 1, 1))
#         x = self.relu(self.conv3(x))
#         # print("Shapes: ", x.shape, a.shape)
#         # print('g5',a.shape)

#         x = torch.add(x, a)
#         x = self.pool1(x)
#         b = x = self.drop1(x)
#         x = F.pad(x, (1, 1, 1, 1))

#         x = self.relu(self.conv4(x))
#         x = F.pad(x, (1, 1, 1, 1))

#         x = self.relu(self.conv5(x))
#         x = torch.add(x, b)
#         x = self.pool1(x)
#         c = x = self.drop1(x)

#         x = F.pad(x, (1, 1, 1, 1))
#         x = self.relu(self.conv6(x))
#         x = F.pad(x, (1, 1, 1, 1))
#         x = self.relu(self.conv7(x))
#         x = torch.add(x, c)
#         x = self.pool2(x)
#         d = x = self.drop1(x)

#         x = F.pad(x, (1, 1, 1, 1))
#         x = self.relu(self.conv8(x))
#         x = F.pad(x, (1, 1, 1, 1))
#         x = self.relu(self.conv9(x))
#         x = torch.add(x, d)


#         #########
#         x = x.view(x.size(0), -1)

#         # print("shape", x.shape)
#         x = self.relu(self.hidden1(x))
#         x = self.drop2(x)

#         x = self.relu(self.hidden3(x))
#         x = self.drop2(x)
#         # x = self.softmax(self.out(x))
#         x = self.out(x)  # no softmax: CrossEntropyLoss()
#         return x




class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = MaskConv1d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        #self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = MaskConv1d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        #self.bn2 = nn.BatchNorm1d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                MaskConv1d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                #nn.BatchNorm1d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = MaskConv1d(in_planes, planes, kernel_size=1, bias=False)
        #self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = MaskConv1d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        #self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = MaskConv1d(planes, self.expansion*planes, kernel_size=1, bias=False)
        #self.bn3 = nn.BatchNorm1d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                MaskConv1d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                #nn.BatchNorm1d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = self.conv3(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, slice_size, num_classes):
    #def __init__(self, block, num_blocks, num_classes):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = MaskConv1d(2, 64, kernel_size=7, stride=2, padding=1, bias=False)
        #self.bn1 = nn.BatchNorm1d(64)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        #self.linear = nn.Linear(512*block.expansion, num_classes)
        ## Needs discussion ##
        if slice_size == 512:
            self.linear = nn.Linear(512*block.expansion*4, num_classes)
        else:
            self.linear = nn.Linear(512*block.expansion, num_classes)
        ## Needs discussion ##

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        #out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool1d(out, 7)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18_1d(slice_size, num_classes):
    return ResNet(BasicBlock, [2,2,2,2], slice_size, num_classes)

def ResNet34_1d(slice_size, num_classes):
    return ResNet(BasicBlock, [3,4,6,3], slice_size, num_classes)

def ResNet50_1d(slice_size, num_classes):
    print(1)
    return ResNet(Bottleneck, [3,4,6,3], slice_size, num_classes)
def ResNet101_1d(slice_size, num_classes):
    return ResNet(Bottleneck, [3,4,23,3], slice_size,  num_classes)

def ResNet152_1d(num_classes):
    return ResNet(Bottleneck, [3,8,36,3], slice_size, num_classes)

class MaskLinear(nn.Linear):
    def __init__(self, in_features, out_features):
        super(MaskLinear, self).__init__(in_features, out_features, True)
        self.w_mask = Parameter(torch.ones(out_features, in_features))
        #self.b_mask = Parameter(torch.ones(out_features))

    def forward(self, input):
        return F.linear(input, self.weight*self.w_mask, self.bias)


class MaskConv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super(MaskConv1d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)
        self.w_mask = Parameter(torch.ones(self.weight.shape))
        #self.b_mask = Parameter(torch.ones(self.bias.shape))

    def _conv_forward(self, input, weight):
        return F.conv1d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input):
        return self._conv_forward(input, self.weight * self.w_mask)

class MaskConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super(MaskConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)
        #super(MaskConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
        #         padding, dilation, groups, bias, padding_mode)
        self.w_mask = Parameter(torch.ones(self.weight.shape))
        #self.b_mask = Parameter(torch.ones(self.bias.shape))

    def _conv_forward(self, input, weight):
        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input):
        return self._conv_forward(input, self.weight * self.w_mask)
