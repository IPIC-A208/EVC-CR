import torch
import torch.nn as nn
# from base_block import *
from utils import calculate_variables
from tensorboardX import SummaryWriter

class ConvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=1, padding=1, bias=True, activation='lrelu', norm=None, dilation=1):
        super(ConvBlock, self).__init__()
        self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=bias)
        self.norm = norm
        if self.norm =='batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)
        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()
    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.conv(x))
        else:
            out = self.conv(x)
        if self.activation is not None:
            return self.act(out)
        else:
            return out

############################################################MultiFrame  HRNet #############################################################
class CRNN(torch.nn.Module):
    def __init__(self, input_channel, base_channel, neighbor_frames=2, use_norm_at_begin=False, use_norm_in_ru=False, use_norm_at_end=False):
        super(CRNN, self).__init__()
        self.head = ConvBlock(input_size=1, output_size=16)
        self.fusion = torch.nn.ModuleList()
        for _ in range(neighbor_frames):
            self.fusion.append(ConvBlock(input_size=2, output_size=16))
        self.RP_module = RP_module()
        self.output = ConvBlock(input_size=neighbor_frames*16, output_size=1)

    def forward(self, x, neighbor):
        '''
            x: [B, C, W, H]
            neighbor: [B, N, C, W, H] ,N denote the number of neighbor
        '''
        ### initial feature extraction
        feature = self.head(x)
        feature_fusion = []
        
        for i in range(neighbor.shape[1]):
            feature_fusion.append(self.fusion[i](torch.cat((x, neighbor[:,i,:,:,:]), 1)))

        ### recursive
        feature_cat = []
        for i in range(neighbor.shape[1]):
            res = feature - feature_fusion[i]
            res = self.RP_module(res)
            feature = feature + res
            feature_cat.append(feature)
        
        out = torch.cat(feature_cat, 1)
        out = self.output(out)
        return out
    
class RP_module(torch.nn.Module):
    def __init__(self):
        super(RP_module, self).__init__()
        # upsample use bilinear
        self.conv1 = ConvBlock(input_size=16, output_size=16,kernel_size=3, stride=1, padding=1)
        self.conv2 = ConvBlock(input_size=32, output_size=16,kernel_size=1, stride=1, padding=0)
        self.conv3 = ConvBlock(input_size=16, output_size=16,kernel_size=3, stride=1, padding=1)
        self.conv4 = ConvBlock(input_size=48, output_size=16,kernel_size=1, stride=1, padding=0)
        self.conv5 = ConvBlock(input_size=16, output_size=16,kernel_size=3, stride=1, padding=1)
        self.conv6 = ConvBlock(input_size=32, output_size=16,kernel_size=1, stride=1, padding=0)

        self.downsample2 = torch.nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.downsample4 = torch.nn.AvgPool2d(kernel_size=(2, 2), stride=4)
        self.upsample2 = torch.nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsample4 = torch.nn.Upsample(scale_factor=4, mode='bilinear')

    def forward(self,x):
        x_b1_1 = self.conv1(x)
        x_b2_1 = self.conv1(self.downsample2(x))

        x_b1_2 = self.conv2(torch.cat([x_b1_1,self.upsample2(x_b2_1)],1))
        x_b2_2 = self.conv2(torch.cat([self.downsample2(x_b1_1), x_b2_1],1))
        x_b3_2 = self.conv2(torch.cat([self.downsample4(x_b1_1), self.downsample2(x_b2_1)],1))

        x_b1_3 = self.conv3(x_b1_2)
        x_b2_3 = self.conv3(x_b2_2)
        x_b3_3 = self.conv3(x_b3_2)
        
        x_b1_4 = self.conv4(torch.cat([x_b1_3, self.upsample2(x_b2_3), self.upsample4(x_b3_3)],1 ))
        x_b2_4 = self.conv4(torch.cat([self.downsample2(x_b1_3), x_b2_3, self.upsample2(x_b3_3)],1 ))

        x_b1_5 = self.conv5(x_b1_4)
        x_b2_5 = self.conv5(x_b2_4)

        x_out = self.conv6(torch.cat([x_b1_5, self.upsample2(x_b2_5)],1 ))

        return x_out

