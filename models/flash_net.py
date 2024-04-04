from __future__ import print_function,division
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
torch.manual_seed(1)
# torch.set_deterministic(True)
torch.backends.cudnn.deterministic = True
###orginal lidar model from infocom
class FlashNet(nn.Module):
    print('**************using orginal lidar model from infocom***********')
    def __init__(self,modality, num_classes, shrink=1):
        super(FlashNet, self).__init__()
        self.shrink=shrink

        dropProb1 = 0.3
        dropProb2 = 0.2
        channel = 32
        # self.conv1 = nn.Conv2d(90, channel, kernel_size=(3, 3), bias=False)
        self.conv1 = nn.Conv2d(45, channel, kernel_size=(3, 3), bias=False)
        self.conv2 = nn.Conv2d(channel, channel, kernel_size=(3, 3), bias=False)
        self.conv3 = nn.Conv2d(channel, channel, kernel_size=(3, 3), bias=False)
        self.conv4 = nn.Conv2d(channel, channel, kernel_size=(3, 3), bias=False)
        self.conv5 = nn.Conv2d(channel, channel, kernel_size=(3, 3), bias=False)
        self.conv6 = nn.Conv2d(channel, channel, kernel_size=(3, 3), bias=False)
        self.conv7 = nn.Conv2d(channel, channel, kernel_size=(3, 3), bias=False)
        self.conv8 = nn.Conv2d(channel, channel, kernel_size=(3, 3), bias=False)
        self.conv9 = nn.Conv2d(channel, channel, kernel_size=(3, 3), bias=False)

        self.pool1 = nn.MaxPool2d((2,2))
        self.pool2 = nn.MaxPool2d((1, 2))

        # self.hidden1 = nn.Linear(320, 1024)  #orginal
        # self.hidden1 = nn.Linear(2560, 1024)    #with zero padding
        self.hidden1 = nn.Linear(1280, 1024)    #with zero padding image/2
        # self.hidden2 = nn.Linear(1024, 512)  # we are not using this
        self.hidden3 = nn.Linear(1024, 256)
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



# ###orginal image model from infocom on (45,90,3)
# class FlashNet(nn.Module):
#     print('**************using orginal img model from infocom for (45,90,3)[img_inf]***********')
#     def __init__(self,modality, num_classes, shrink=1):
#         super(FlashNet, self).__init__()
#         self.shrink=shrink

#         channel = 32
#         # self.conv1 = nn.Conv2d(90, channel, kernel_size=(3, 3), bias=False)
#         self.conv1 = nn.Conv2d(45, channel, kernel_size=(7, 7), bias=False)
#         self.conv2 = nn.Conv2d(channel, channel, kernel_size=(3, 3), bias=False)
#         self.conv3 = nn.Conv2d(channel, channel, kernel_size=(3, 3), bias=False)
#         self.conv4 = nn.Conv2d(channel, channel, kernel_size=(3, 3), bias=False)
#         self.conv5 = nn.Conv2d(channel, channel, kernel_size=(3, 3), bias=False)
#         self.conv6 = nn.Conv2d(channel, channel, kernel_size=(3, 3), bias=False)

#         self.pool1 = nn.MaxPool2d((2,2))
#         self.pool2 = nn.MaxPool2d((3, 3), padding=1)

#         self.hidden1 = nn.Linear(1792, 512)  #orginal
#         # self.hidden1 = nn.Linear(2560, 1024)    #with zero padding
#         # self.hidden1 = nn.Linear(1248, 512)    #with zero padding image/2
#         self.hidden2 = nn.Linear(512, 256)
#         # self.hidden3 = nn.Linear(512, 256)
#         self.out = nn.Linear(256, 64)  # 128
#         #######################
#         self.drop = nn.Dropout(0.25)
#         self.relu = nn.ReLU()
#         self.tanh = nn.Tanh()
#         self.softmax = nn.Softmax()

#     def forward(self, x):
#         # FOR CNN BASED IMPLEMENTATION
#         x = F.pad(x, (3, 3, 3, 3))
#         x = self.relu(self.conv1(x))
#         x = F.pad(x, (1, 1, 1, 1))
#         b = x = self.relu(self.conv2(x))
#         x = F.pad(x, (1, 1, 1, 1))
#         x = self.relu(self.conv3(x))
#         x = F.pad(x, (1, 1, 1, 1))
#         x = self.relu(self.conv4(x))
#         x = torch.add(x, b)
#         x = self.pool1(x)
#         c = x = self.drop(x)

#         x = F.pad(x, (1, 1, 1, 1))
#         x = self.relu(self.conv5(x))
#         x = F.pad(x, (1, 1, 1, 1))
#         x = self.relu(self.conv6(x))
#         x = torch.add(x, c)
#         x = self.pool2(x)
#         x = self.drop(x)
#         # print('xshape',x.shape)
#         x = x.view(x.size(0), -1)

#         # print("shape", x.shape)
#         x = self.relu(self.hidden1(x))
#         x = self.drop(x)

#         x = self.relu(self.hidden2(x))
#         x = self.drop(x)

#         x = self.out(x)  # no softmax: CrossEntropyLoss()
#         return x







# ##custom1 model: this is similar to lidar original model
# class FlashNet(nn.Module):
#     print('**************using custom1 model from infocom***********')
#     def __init__(self,modality, num_classes, shrink=1):
#         super(FlashNet, self).__init__()
#         self.shrink=shrink

#         dropProb1 = 0.3
#         dropProb2 = 0.2
#         channel = 32
#         # self.conv1 = nn.Conv2d(90, channel, kernel_size=(3, 3), bias=False)
#         self.conv1 = nn.Conv2d(45, channel, kernel_size=(3, 3), bias=False)
#         self.conv2 = nn.Conv2d(channel, channel, kernel_size=(3, 3), bias=False)
#         self.conv3 = nn.Conv2d(channel, channel, kernel_size=(3, 3), bias=False)
#         self.conv4 = nn.Conv2d(channel, channel, kernel_size=(3, 3), bias=False)
#         self.conv5 = nn.Conv2d(channel, channel, kernel_size=(3, 3), bias=False)
#         self.conv6 = nn.Conv2d(channel, channel, kernel_size=(3, 3), bias=False)
#         self.conv7 = nn.Conv2d(channel, channel, kernel_size=(3, 3), bias=False)
#         self.conv8 = nn.Conv2d(channel, channel, kernel_size=(3, 3), bias=False)
#         self.conv9 = nn.Conv2d(channel, channel, kernel_size=(3, 3), bias=False)

#         self.pool1 = nn.MaxPool2d((2,2))
#         self.pool2 = nn.MaxPool2d((1, 2))

#         # self.hidden1 = nn.Linear(320, 1024)  #orginal
#         # self.hidden1 = nn.Linear(2560, 1024)    #with zero padding this was correct
#         self.hidden1 = nn.Linear(1280, 1024)    #with zero padding this was 45 not 90

#         # self.hidden2 = nn.Linear(1024, 512)
#         self.hidden3 = nn.Linear(1024, 256)
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

#         # print('xshape',x.shape)
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









###7,3 is ok so don't need this
# ###orginal image model from infocom on (45,90,3)
# class FlashNet(nn.Module):
#     print('**************using orginal img model from infocom for (45,90,3)[img_inf_31]***********')
#     def __init__(self,modality, num_classes, shrink=1):
#         super(FlashNet, self).__init__()
#         self.shrink=shrink

#         channel = 32
#         # self.conv1 = nn.Conv2d(90, channel, kernel_size=(3, 3), bias=False)
#         self.conv1 = nn.Conv2d(45, channel, kernel_size=(3, 3), bias=False)
#         self.conv2 = nn.Conv2d(channel, channel, kernel_size=(1, 1), bias=False)
#         self.conv3 = nn.Conv2d(channel, channel, kernel_size=(1, 1), bias=False)
#         self.conv4 = nn.Conv2d(channel, channel, kernel_size=(1, 1), bias=False)
#         self.conv5 = nn.Conv2d(channel, channel, kernel_size=(1, 1), bias=False)
#         self.conv6 = nn.Conv2d(channel, channel, kernel_size=(1, 1), bias=False)

#         self.pool1 = nn.MaxPool2d((2,2))
#         self.pool2 = nn.MaxPool2d((3, 3), padding=1)

#         self.hidden1 = nn.Linear(1792, 512)  #data is/2 and filters as well=>the same as 7,3
#         # self.hidden1 = nn.Linear(2560, 1024)    #with zero padding
#         # self.hidden1 = nn.Linear(1248, 512)    #with zero padding image/2
#         self.hidden2 = nn.Linear(512, 256)
#         # self.hidden3 = nn.Linear(512, 256)
#         self.out = nn.Linear(256, 64)  # 128
#         #######################
#         self.drop = nn.Dropout(0.25)
#         self.relu = nn.ReLU()
#         self.tanh = nn.Tanh()
#         self.softmax = nn.Softmax()

#     def forward(self, x):
#         # FOR CNN BASED IMPLEMENTATION
#         x = F.pad(x, (1, 1, 1, 1))
#         x = self.relu(self.conv1(x))
#         # x = F.pad(x, (1, 1, 1, 1))
#         b = x = self.relu(self.conv2(x))
#         # x = F.pad(x, (1, 1, 1, 1))
#         x = self.relu(self.conv3(x))
#         # x = F.pad(x, (1, 1, 1, 1))
#         x = self.relu(self.conv4(x))
#         x = torch.add(x, b)
#         x = self.pool1(x)
#         c = x = self.drop(x)

#         # x = F.pad(x, (1, 1, 1, 1))
#         x = self.relu(self.conv5(x))
#         # x = F.pad(x, (1, 1, 1, 1))
#         x = self.relu(self.conv6(x))
#         x = torch.add(x, c)
#         x = self.pool2(x)
#         x = self.drop(x)
#         # print('xshape',x.shape)
#         x = x.view(x.size(0), -1)

#         # print("shape", x.shape)
#         x = self.relu(self.hidden1(x))
#         x = self.drop(x)

#         x = self.relu(self.hidden2(x))
#         x = self.drop(x)

#         x = self.out(x)  # no softmax: CrossEntropyLoss()
#         return x









###for justifying acc of image from infocom
# ###orginal image model from infocom on [90,180,3]
# class FlashNet(nn.Module):
#     print('**************using orginal img model from infocom (90,180,3)***********')
#     def __init__(self,modality, num_classes, shrink=1):
#         super(FlashNet, self).__init__()
#         self.shrink=shrink

#         channel = 32
#         # self.conv1 = nn.Conv2d(90, channel, kernel_size=(3, 3), bias=False)
#         self.conv1 = nn.Conv2d(90, channel, kernel_size=(7, 7), bias=False)
#         self.conv2 = nn.Conv2d(channel, channel, kernel_size=(3, 3), bias=False)
#         self.conv3 = nn.Conv2d(channel, channel, kernel_size=(3, 3), bias=False)
#         self.conv4 = nn.Conv2d(channel, channel, kernel_size=(3, 3), bias=False)
#         self.conv5 = nn.Conv2d(channel, channel, kernel_size=(3, 3), bias=False)
#         self.conv6 = nn.Conv2d(channel, channel, kernel_size=(3, 3), bias=False)

#         self.pool1 = nn.MaxPool2d((2,2))
#         self.pool2 = nn.MaxPool2d((3, 3), padding=1)

#         self.hidden1 = nn.Linear(864, 512)  #orginal
#         # self.hidden1 = nn.Linear(2560, 1024)    #with zero padding
#         # self.hidden1 = nn.Linear(1248, 512)    #with zero padding image/2
#         self.hidden2 = nn.Linear(512, 256)
#         # self.hidden3 = nn.Linear(512, 256)
#         self.out = nn.Linear(256, 64)  # 128
#         #######################
#         self.drop = nn.Dropout(0.25)
#         self.relu = nn.ReLU()
#         self.tanh = nn.Tanh()
#         self.softmax = nn.Softmax()

#     def forward(self, x):
#         # FOR CNN BASED IMPLEMENTATION
#         x = F.pad(x, (3, 3, 3, 3))
#         x = self.relu(self.conv1(x))
#         x = F.pad(x, (1, 1, 1, 1))
#         b = x = self.relu(self.conv2(x))
#         x = F.pad(x, (1, 1, 1, 1))
#         x = self.relu(self.conv3(x))
#         x = F.pad(x, (1, 1, 1, 1))
#         x = self.relu(self.conv4(x))
#         x = torch.add(x, b)
#         x = self.pool1(x)
#         c = x = self.drop(x)

#         x = F.pad(x, (1, 1, 1, 1))
#         x = self.relu(self.conv5(x))
#         x = F.pad(x, (1, 1, 1, 1))
#         x = self.relu(self.conv6(x))
#         x = torch.add(x, c)
#         x = self.pool2(x)
#         x = self.drop(x)
#         # print('xshape',x.shape)
#         x = x.view(x.size(0), -1)

#         # print("shape", x.shape)
#         x = self.relu(self.hidden1(x))
#         x = self.drop(x)

#         x = self.relu(self.hidden2(x))
#         x = self.drop(x)

#         x = self.out(x)  # no softmax: CrossEntropyLoss()
#         return x























#######################################
#######################################
#######################################


# ###orginal image model from infocom
# class FlashNet(nn.Module):
#     print('**************using orginal img model from infocom (3,1)***********')
#     def __init__(self,modality, num_classes, shrink=1):
#         super(FlashNet, self).__init__()
#         self.shrink=shrink

#         channel = 32
#         # self.conv1 = nn.Conv2d(90, channel, kernel_size=(3, 3), bias=False)
#         self.conv1 = nn.Conv2d(45, channel, kernel_size=(3, 3), bias=False)
#         self.conv2 = nn.Conv2d(channel, channel, kernel_size=(1, 1), bias=False)
#         self.conv3 = nn.Conv2d(channel, channel, kernel_size=(1, 1), bias=False)
#         self.conv4 = nn.Conv2d(channel, channel, kernel_size=(1, 1), bias=False)
#         self.conv5 = nn.Conv2d(channel, channel, kernel_size=(1, 1), bias=False)
#         self.conv6 = nn.Conv2d(channel, channel, kernel_size=(1, 1), bias=False)

#         self.pool1 = nn.MaxPool2d((2,2))
#         self.pool2 = nn.MaxPool2d((3, 3), padding=1)

#         # self.hidden1 = nn.Linear(320, 1024)  #orginal
#         # self.hidden1 = nn.Linear(2560, 1024)    #with zero padding
#         self.hidden1 = nn.Linear(1248, 512)    #with zero padding image/2
#         self.hidden2 = nn.Linear(512, 256)
#         self.hidden3 = nn.Linear(512, 256)
#         self.out = nn.Linear(256, 64)  # 128
#         #######################
#         self.drop = nn.Dropout(0.25)
#         self.relu = nn.ReLU()
#         self.tanh = nn.Tanh()
#         self.softmax = nn.Softmax()

#     def forward(self, x):
#         # FOR CNN BASED IMPLEMENTATION
#         # x = F.pad(x, (1, 1, 1, 1))
#         x = self.relu(self.conv1(x))
#         # x = F.pad(x, (1, 1, 1, 1))
#         b = x = self.relu(self.conv2(x))
#         # x = F.pad(x, (1, 1, 1, 1))
#         x = self.relu(self.conv3(x))
#         # x = F.pad(x, (1, 1, 1, 1))
#         x = self.relu(self.conv4(x))
#         x = torch.add(x, b)
#         x = self.pool1(x)
#         c = x = self.drop(x)

#         # x = F.pad(x, (1, 1, 1, 1))
#         x = self.relu(self.conv5(x))
#         # x = F.pad(x, (1, 1, 1, 1))
#         x = self.relu(self.conv6(x))
#         x = torch.add(x, c)
#         x = self.pool2(x)
#         x = self.drop(x)
#         # print('xshape',x.shape)
#         x = x.view(x.size(0), -1)

#         # print("shape", x.shape)
#         x = self.relu(self.hidden1(x))
#         x = self.drop(x)

#         x = self.relu(self.hidden2(x))
#         x = self.drop(x)

#         x = self.out(x)  # no softmax: CrossEntropyLoss()
#         return x














# class FlashNet(nn.Module):
#     def __init__(self,modality, num_classes, shrink=1):
#         super(FlashNet, self).__init__()
#         self.shrink=shrink
#         # if modality =='gps':
#         #     self.conv1 = nn.Conv2d(2, int(20*shrink), 1, 1, bias=False)
#         #     self.fc2 = nn.Linear(64, 64)  #num_classes   #30720

#         # if modality =='img':
#         #     self.conv1 = nn.Conv2d(90, int(20*shrink), 1, 1, bias=False)
#         #     self.fc2 = nn.Linear(30720, 64)  #num_classes   #30720

#         # if modality =='lidar':
#         #     self.conv1 = nn.Conv2d(20, int(20*shrink), 1, 1, bias=False)

#         # self.conv1_img = nn.Conv2d(90, int(20*shrink), 2, 1, bias=False)
#         # self.conv1_lid = nn.Conv2d(20, int(20*shrink), 2, 1, bias=False)

#         self.conv1 = nn.Conv2d(90, int(20*shrink), 20, 20, bias=False)
#         self.conv2 = nn.Conv2d(int(20*shrink), int(32*shrink), 1, 1, bias=False)
#         self.conv3 = nn.Conv2d(int(32*shrink), int(64*shrink), 1, 1, bias=False)
#         self.conv4 = nn.Conv2d(int(64*shrink), int(64*shrink), 1, 1, bias=False)
#         # self.fc1 = nn.Linear(5*5*int(64*shrink), 512)
#         self.fc2 = nn.Linear(512, 512)  #num_classes   #30720
#         self.fc3 = nn.Linear(512, 256)  #num_classes   #30720
#         self.fc4 = nn.Linear(256, 64)  #num_classes   #30720

#         self.drp1 = nn.Dropout(p=.25)
#         self.drp2 = nn.Dropout(p=.5)

#     def forward(self,x):
#         # print('x1',x.shape)
#         x = F.relu(self.conv1(x))
#         # print('x2',x.shape)
#         x = F.relu(self.conv2(x))
#         # print('x3',x.shape)
#         x = self.drp1(x)
#         # print('x4',x.shape)

#         x = F.relu(self.conv3(x))
#         # print('x5',x.shape)
#         x = F.relu(self.conv4(x))
#         # print('x6',x.shape)
#         x = self.drp1(x)
#         # print('x7',x.shape)
#         x = x.view(x.size(0), -1)
#         # print('x8',x.shape)
#         x = self.drp2(x)
#         # print('x9',x.shape)
#         x = self.fc2(x)
#         # print('x10',x.shape)
#         x = self.fc3(x)
#         x = self.drp2(x)
#         x = self.fc4(x)
#         output = F.log_softmax(x, dim=1)
#         # print('output',output.shape)
#         return output


# class DenseConv2d(torch.nn.Conv2d):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, use_bias=True,
#                  padding_mode='zeros', mask: torch.FloatTensor = None, use_mask=True):
#         super(DenseConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding,
#                                           dilation, groups, use_bias, padding_mode)
#     def conv2d_forward(self, input, weight):
#         if self.padding_mode == 'circular':
#             expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
#                                 (self.padding[0] + 1) // 2, self.padding[0] // 2)
#             return F.conv2d(F.pad(input, expanded_padding, mode='circular'),
#                             weight, self.bias, self.stride,
#                             _pair(0), self.dilation, self.groups)
#         else:
#             return F.conv2d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


# class ResNet(torch.nn.Module):
#     def __init__(self, module):
#         super().__init__()
#         self.module = module

#     def forward(self, inputs):
#         return self.module(inputs) + inputs



##runnin seperate
# class FlashNet(nn.Module):
#     def __init__(self,modality, num_classes, shrink=1):
#         super(FlashNet, self).__init__()
#         self.shrink=shrink

#         self.conv1 = DenseConv2d(20, 32, kernel_size=3, padding=1)
#         self.conv2 = DenseConv2d(32, 32, kernel_size=3, padding=1)

#         self.fc1 = nn.Linear(1280, 1024)  #num_classes   #30720
#         self.fc2 = nn.Linear(1024, 512)  #num_classes   #30720
#         self.fc3 = nn.Linear(512, 256)  #num_classes   #30720
#         self.fc4 = nn.Linear(256, 64)  #num_classes   #30720
#         self.maxp = nn.MaxPool2d(kernel_size=2)

#         self.drp1 = nn.Dropout(p=0.3)
#         self.drp2 = nn.Dropout(p=0.2)





#     def forward(self,x):
#         # print('x1',x.shape)
#         x1 = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x1))
#         x = F.relu(self.conv2(x))
#         x_r1 = x1+x
#         x = self.maxp(x_r1)
#         x = self.drp1(x)


#         x2 = F.relu(self.conv2(x))
#         x = F.relu(self.conv2(x2))
#         x_r2 = x2+x
#         x = self.maxp(x_r2)
#         x = self.drp1(x)


#         x3 = F.relu(self.conv2(x))
#         x = F.relu(self.conv2(x3))
#         x_r3 = x3+x
#         x = self.maxp(x_r3)
#         x = self.drp1(x)


#         x4 = F.relu(self.conv2(x))
#         x = F.relu(self.conv2(x4))
#         x_r4 = x4+x

#         x = x.view(x.size(0), -1)
#         # print('flar shape',x.shape)
#         x = self.fc1(x)
#         x = self.drp2(x)
#         x = self.fc2(x)
#         x = self.drp2(x)
#         x = self.fc3(x)
#         x = self.drp2(x)
#         x = self.fc4(x)
#         x = self.drp2(x)
#         output = F.log_softmax(x, dim=1)
#         # print('output',output.shape)
#         return output


        ## runing
    # def forward(self,x):
    #     # print('x1',x.shape)
    #     x1 = F.relu(self.conv1(x))
    #     x = F.relu(self.conv2(x1))
    #     x = F.relu(self.conv2(x))
    #     x_r1 = x1+x
    #     x = self.maxp(x_r1)
    #     x = self.drp1(x)


    #     x2 = F.relu(self.conv2(x))
    #     x = F.relu(self.conv2(x2))
    #     x_r2 = x2+x
    #     x = self.maxp(x_r2)
    #     x = self.drp1(x)


    #     x3 = F.relu(self.conv2(x))
    #     x = F.relu(self.conv2(x3))
    #     x_r3 = x3+x
    #     x = self.maxp(x_r3)
    #     x = self.drp1(x)


    #     x4 = F.relu(self.conv2(x))
    #     x = F.relu(self.conv2(x4))
    #     x_r4 = x4+x

    #     x = x.view(x.size(0), -1)
    #     # print('flar shape',x.shape)
    #     x = self.fc1(x)
    #     x = self.drp2(x)
    #     x = self.fc2(x)
    #     x = self.drp2(x)
    #     x = self.fc3(x)
    #     x = self.drp2(x)
    #     x = self.fc4(x)
    #     x = self.drp2(x)
    #     output = F.log_softmax(x, dim=1)
    #     # print('output',output.shape)
    #     return output





    #     classifier = nn.Sequential(nn.Linear(832, 2048),  #800, 3200
    #                                 nn.ReLU(inplace=True),
    #                                 # nn.BatchNorm1d(2048),

    #                                 nn.Linear(2048, 1024),
    #                                 nn.ReLU(inplace=True),
    #                                 # nn.BatchNorm1d(1024),

    #                                 nn.Linear(1024, 512),
    #                                 nn.ReLU(inplace=True),
    #                                 # nn.BatchNorm1d(512),

    #                                 nn.Linear(512, 256),
    #                                 nn.ReLU(inplace=True),
    #                                 # nn.BatchNorm1d(256),

    #                                 nn.Linear(256, 128),
    #                                 nn.ReLU(inplace=True),
    #                                 # nn.BatchNorm1d(128),

    #                                 nn.Linear(128, 64)
    #                                 # ,nn.Softmax(dim=1)
    #                                 )
    # def _make_feature_layers_lidar(self):
    #     layers = []
    #     in_channels = 90
    #     print('********Model is intialized*************')
    #     return nn.Sequential(
    #         DenseConv2d(in_channels, 32, kernel_size=3, padding=1),nn.ReLU(inplace=True),
    #         ResNet(
    #             nn.Sequential(
    #                 DenseConv2d(32, 32, kernel_size=3, padding=1),nn.ReLU(inplace=True),
    #                 DenseConv2d(32, 32, kernel_size=3, padding=1),nn.ReLU(inplace=True)
    #                 )),
    #         nn.MaxPool2d(kernel_size=2),
    #         nn.Dropout(p=0.3),

    #         ResNet(
    #             nn.Sequential(
    #                 DenseConv2d(32, 32, kernel_size=3, padding=1),nn.ReLU(inplace=True),
    #                 DenseConv2d(32, 32, kernel_size=3, padding=1),nn.ReLU(inplace=True)
    #                 )),
    #         nn.MaxPool2d(kernel_size=2),
    #         nn.Dropout(p=0.3),

    #         ResNet(
    #             nn.Sequential(
    #                 DenseConv2d(32, 32, kernel_size=3, padding=1),nn.ReLU(inplace=True),
    #                 DenseConv2d(32, 32, kernel_size=3, padding=1),nn.ReLU(inplace=True)
    #                 )),
    #         nn.MaxPool2d((1, 2)),
    #         nn.Dropout(p=0.3),

    #         ResNet(
    #             nn.Sequential(
    #                 DenseConv2d(32, 32, kernel_size=3, padding=1),nn.ReLU(inplace=True),
    #                 DenseConv2d(32, 32, kernel_size=3, padding=1),nn.ReLU(inplace=True),
    #                 )),
    #         nn.Flatten(),
    #         nn.Linear(832, 2048),  #800, 3200
    #         nn.ReLU(inplace=True),
    #         # nn.BatchNorm1d(2048),

    #         nn.Linear(2048, 1024),
    #         nn.ReLU(inplace=True),
    #         # nn.BatchNorm1d(1024),

    #         nn.Linear(1024, 512),
    #         nn.ReLU(inplace=True),
    #         # nn.BatchNorm1d(512),

    #         nn.Linear(512, 256),
    #         nn.ReLU(inplace=True),
    #         # nn.BatchNorm1d(256),

    #         nn.Linear(256, 128),
    #         nn.ReLU(inplace=True),
    #         # nn.BatchNorm1d(128),

    #         nn.Linear(128, 64)
    #                 )


    # def forward(self, inputs1):
    #     # print('check point',inputs.shape)
    #     outputs_lidar = self.features_lidar(inputs1)
    #     # print('outputs_lidar features',outputs_lidar.shape)
    #     outputs = outputs_lidar.view(outputs_lidar.size(0), -1)
    #     return outputs




# model1
# class FlashNet(nn.Module):
#     def __init__(self,modality, num_classes, shrink=1):
#         super(FlashNet, self).__init__()
#         self.shrink=shrink

#         self.conv1 = DenseConv2d(90, 32, kernel_size=3, padding=1)
#         self.conv2 = DenseConv2d(32, 32, kernel_size=3, padding=1)

#         self.fc2 = nn.Linear(1280, 512)  #num_classes   #30720
#         self.fc3 = nn.Linear(512, 256)  #num_classes   #30720
#         self.fc4 = nn.Linear(256, 64)  #num_classes   #30720
#         self.maxp = nn.MaxPool2d(kernel_size=2)

#         self.drp1 = nn.Dropout(p=0.3)
#         self.drp2 = nn.Dropout(p=0.2)

#     def forward(self,x):
#         # print('x1',x.shape)
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = F.relu(self.conv2(x))
#         x = self.maxp(x)
#         x = self.drp1(x)

#         x = F.relu(self.conv2(x))
#         x = F.relu(self.conv2(x))
#         x = self.maxp(x)
#         x = self.drp1(x)

#         x = F.relu(self.conv2(x))
#         x = F.relu(self.conv2(x))
#         x = self.maxp(x)
#         x = self.drp1(x)


#         x = x.view(x.size(0), -1)
#         # print('flar shape',x.shape)
#         x = self.fc2(x)
#         x = self.drp2(x)
#         x = self.fc3(x)
#         x = self.drp2(x)
#         x = self.fc4(x)
#         x = self.drp2(x)
#         output = F.log_softmax(x, dim=1)
#         # print('output',output.shape)
#         return output
