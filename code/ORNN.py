import torch


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


class ORNN(torch.nn.Module):
    def __init__(self, block=4, neighbor_frames=2, channel=16):
        super(ORNN, self).__init__()
        self.head = ConvBlock(input_size=1, output_size=channel)
        self.fusion = torch.nn.ModuleList()
        for _ in range(neighbor_frames):
            self.fusion.append(ConvBlock(input_size=2, output_size=channel))
        recursive = [ConvBlock(input_size=channel, output_size=channel) for _ in range(block)]
        self.recursive = torch.nn.Sequential(*recursive)
        self.output = ConvBlock(input_size=neighbor_frames*channel, output_size=1)

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
            res = self.recursive(res)
            feature = feature + res
            feature_cat.append(feature)
        
        out = torch.cat(feature_cat, 1)
        out = self.output(out)
        return out


