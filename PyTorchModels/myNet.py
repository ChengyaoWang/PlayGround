import torch 
import torch.nn as nn 

class ResidualBlock(nn.Module):

    def __init__(self, in_channel, out_channel, FirstConv_Stride = 1, downsample = None):
        super(ResidualBlock, self).__init__()
        self.conv_1 = nn.Conv2d(in_channel,
                                out_channel,
                                kernel_size = 3,
                                stride = FirstConv_Stride,
                                padding = 1, 
                                bias = False)
        self.bn_1   = nn.BatchNorm2d(out_channel)
        self.relu_1 = nn.ReLU(inplace = True)
        self.conv_2 = nn.Conv2d(out_channel,
                                out_channel,
                                kernel_size = 3, 
                                stride = 1, 
                                padding = 1, 
                                bias = False)
        self.bn_2   = nn.BatchNorm2d(out_channel)
        self.relu_2 = nn.ReLU(inplace = True)
        self.downsample = downsample

    def forward(self, X):
        identity = X
        X = self.bn_1(self.conv_1(X))
        X = self.relu_1(X)
        X = self.bn_2(self.conv_2(X))
        if self.downsample is not None:
            identity = self.downsample(identity)
        return X + identity


class myNet(nn.Module):
    model_name = 'MyModel'
    def __init__(self, ResidualDepth, DepthShrink = 1.0, num_of_classes = 10):
        super(myNet, self).__init__()

        self.block = ResidualBlock

        self.in_channel_buff = 16 // 2
        # Define the Layers
        # If the Input Image is not large enough for such intense dimension reduction
        self.conv_1    = nn.Conv2d(3, 16 // 2, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn_1      = nn.BatchNorm2d(16 // 2)
        self.relu_1    = nn.ReLU(inplace = True)
        self.highway_12 = self._highway_fetch(16 // 2, 16 // 2, 1)
        self.highway_13 = self._highway_fetch(16 // 2, 32 // 2, 2)
        self.highway_14 = self._highway_fetch(16 // 2, 64 // 2, 4)

        self.conv2_x   = self._layer_gen(16 // 2, 16 // 2, ResidualDepth)
        self.relu_2    = nn.ReLU(inplace = True)
        self.highway_23 = self._highway_fetch(16//2, 32//2, 2)
        self.highway_24 = self._highway_fetch(16//2, 64//2, 4)


        self.conv3_x   = self._layer_gen(16//2, 32//2, ResidualDepth, stride = 2)
        self.relu_3    = nn.ReLU(inplace = True)
        self.highway_34 = self._highway_fetch(32//2, 64//2, 2)

        self.conv4_x   = self._layer_gen(32//2, 64//2, ResidualDepth, stride = 2)
        self.relu_4    = nn.ReLU(inplace = True)
        self.avgPool   = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(64//2, num_of_classes)


    def _layer_gen(self, in_channel, out_channel, repeat, stride = 1):
        # If it's not conv2_x
        if stride != 1:
            downsample = nn.Sequential(nn.Conv2d(self.in_channel_buff,
                                                 out_channel,
                                                 kernel_size = 1,
                                                 stride = 2),
                                       nn.BatchNorm2d(out_channel))
        else:
            downsample = None
        layers = [self.block(self.in_channel_buff, out_channel, FirstConv_Stride = stride,
                             downsample = downsample)]
        self.in_channel_buff = out_channel
        for _ in range(1, repeat):
            layers.append(self.block(self.in_channel_buff, out_channel, FirstConv_Stride = 1,
                                     downsample = None))
        return nn.Sequential(* layers)


    def _highway_fetch(self, in_channel, out_channel, stride = 1):
        return nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size = 1, stride = stride),
                             nn.BatchNorm2d(out_channel))



    def _forward_impl(self, x):
        x  = self.relu_1(self.bn_1(self.conv_1(x)))
        x2 = self.relu_2(self.conv2_x(x)  + self.highway_12(x))
        x3 = self.relu_3(self.conv3_x(x2) + self.highway_13(x) + self.highway_23(x2))
        x4 = self.relu_4(self.conv4_x(x3) + self.highway_14(x) + self.highway_24(x2) + self.highway_34(x3))
        x4 = self.avgPool(x4)
        x4 = torch.flatten(x4, 1)
        x4 = self.fc(x4)
        return x4
    
    def forward(self, x):
        return self._forward_impl(x)