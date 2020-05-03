import torch
import torch.nn as nn 

ResNet_Structure = {'ResNet18v1': [2, 2, 2, 2],
                    'ResNet34v1': [3, 4, 6, 3],
                    'ResNet50v1': [3, 4, 6, 3],
                    'ResNet101v1':[3, 4, 23, 3],
                    'ResNet152v1':[3, 8, 36, 3]}

# Function to Fetch ResNet
def Fetch_ResNet(model_name):
    if model_name in ResNet_Structure:
        return ResNetv1(model_name)
    else:
        depth = int(model_name[6:-2]) - 2
        assert (depth % 6 == 0), "Wrong Depth for ResNet_FOR_CIFAR"
        return ResNet_FOR_CIFAR(depth // 6)
        
# This is the basic block for ResNet 18 / 34
class ResNetv1_BasicBlock(nn.Module):
    expansion = 1
    def __init__( self, in_channels,
                        out_channels,
                        FirstConv_Stride = 1,
                        downsample = None, 
                        norm_layer = None):
        super(ResNetv1_BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d()
        # Self.conv1 performs downsampling if asked
        self.conv1 = self.conv3x3(in_channels, out_channels, stride = FirstConv_Stride)
        self.bn1   = norm_layer(out_channels)
        self.relu1 = nn.ReLU(inplace = True)
        self.conv2 = self.conv3x3(out_channels, out_channels, stride = 1)
        self.bn2   = norm_layer(out_channels)
        self.relu2 = nn.ReLU(inplace = True)
        self.downsample = downsample


    def conv3x3(self, in_channels, out_channels, stride = 1, padding = 1):
        return nn.Conv2d(in_channels,
                         out_channels, 
                         kernel_size = 3, 
                         stride = stride, 
                         padding = padding, 
                         bias = False)

    def conv1x1(self, in_channels, out_channels, stride = 1):
        return nn.Conv2d(in_channels,
                         out_channels, 
                         kernel_size = 1, 
                         stride = stride, 
                         bias = False)

    def forward(self, x):
        identity = x
        x = self.bn1(self.conv1(x))
        x = self.relu1(x)
        x = self.bn2(self.conv2(x))
        if self.downsample is not None:
            identity = self.downsample(identity)
        x = self.relu2(x + identity)
        return x
        
# This is the basic block for ResNet 50 / 101 / 152
class ResNetv1_BottleNeck(nn.Module):
    expansion = 4
    def __init__( self, in_channels,
                        out_channels,
                        FirstConv_Stride = 1,
                        downsample = None, 
                        norm_layer = None):
        super(ResNetv1_BottleNeck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # We perform downsampling in self.conv2
        self.conv1 = self.conv1x1(in_channels, out_channels)
        self.bn1   = norm_layer(out_channels)
        self.relu1 = nn.ReLU(inplace = True)
        self.conv2 = self.conv3x3(out_channels, out_channels, stride = FirstConv_Stride)
        self.bn2   = norm_layer(out_channels)
        self.relu2 = nn.ReLU(inplace = True)
        self.conv3 = self.conv1x1(out_channels, out_channels * self.expansion)
        self.bn3   = norm_layer(out_channels * self.expansion)
        self.relu3 = nn.ReLU(inplace = True)
        self.downsample = downsample

    def conv3x3(self, in_channels, out_channels, stride = 1, padding = 1):
        return nn.Conv2d(in_channels,
                         out_channels, 
                         kernel_size = 3, 
                         stride = stride, 
                         padding = padding, 
                         bias = False)

    def conv1x1(self, in_channels, out_channels, stride = 1):
        return nn.Conv2d(in_channels,
                         out_channels, 
                         kernel_size = 1, 
                         stride = stride, 
                         bias = False)
    
    def forward(self, x):
        identity = x

        x = self.bn1(self.conv1(x))
        x = self.relu1(x)
        x = self.bn2(self.conv2(x))
        x = self.relu2(x)
        x = self.bn3(self.conv3(x))
        if self.downsample is not None:
            identity = self.downsample(identity)
        x = self.relu3(x + identity)

        return x

class ResNetv1(nn.Module):
    def __init__(self,
                 model_name,
                 Small_Input = True,
                 num_classes = 10,
                 zero_init_residual = False,
                 norm_layer = None):
        super(ResNetv1, self).__init__()
        self.model_name = model_name
        self.in_channel_buff = 64
        if norm_layer is None:
            self.norm_layer = nn.BatchNorm2d
        if self.model_name in ['ResNet18v1', 'ResNet34v1']:
            self.block = ResNetv1_BasicBlock
        elif self.model_name in ['ResNet50v1', 'ResNet101v1', 'ResNet153v2']:
            self.block = ResNetv1_BottleNeck
        # Define the Layers
        # If the Input Image is not large enough for such intense dimension reduction
        if Small_Input:
            self.conv1 = nn.Conv2d(3, 64, kernel_size = 7, stride = 1, padding = 3, bias = False)
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.bn1   = self.norm_layer(64)
        self.relu  = nn.ReLU(inplace = True)
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.conv2_x = self._layer_gen(64,  64, ResNet_Structure[self.model_name][0])
        self.conv3_x = self._layer_gen(64, 128, ResNet_Structure[self.model_name][1], stride = 2)
        self.conv4_x = self._layer_gen(128,256, ResNet_Structure[self.model_name][2], stride = 2)
        self.conv5_x = self._layer_gen(256,512, ResNet_Structure[self.model_name][3], stride = 2)
        self.avgPool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * self.block.expansion, num_classes)
        # Initialzie the Network
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode = 'fan_out', nonlinearity = 'relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        # This zero-init applies to the last BN in each residual branch
        # This improves the mode by 0.2 % ~ 0.3 % according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, ResNetv1_BottleNeck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, ResNetv1_BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
            
    def _layer_gen(self, in_channel, out_channel, layerList, stride = 1):
        # If it's not conv2_x
        if stride != 1:
            downsample = nn.Sequential(nn.Conv2d(self.in_channel_buff,
                                                 out_channel * self.block.expansion,
                                                 kernel_size = 1,
                                                 stride = 2),
                                       self.norm_layer(out_channel * self.block.expansion))
        else:
            downsample = None
        layers = []
        layers.append(self.block(self.in_channel_buff, out_channel, FirstConv_Stride = stride,
                                 downsample = downsample, norm_layer = self.norm_layer))
        self.in_channel_buff = out_channel * self.block.expansion
        for _ in range(1, layerList):
            layers.append(self.block(self.in_channel_buff, out_channel, FirstConv_Stride = 1,
                                     downsample = None, norm_layer = self.norm_layer))
        return nn.Sequential(* layers)

    def _forward_impl(self, x):
        x = self.bn1(self.conv1(x))
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)

        x = self.avgPool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
    
    def forward(self, x):
        return self._forward_impl(x)


class ResNet_FOR_CIFAR(nn.Module):

    def __init__(self,
                 ResBlock_Repetition,
                 num_classes = 10,
                 zero_init_residual = False,
                 norm_layer = None):
        super(ResNet_FOR_CIFAR, self).__init__()
        self.model_name = 'ResNet' + str(6 * ResBlock_Repetition + 2) + 'v1_FOR_CIFAR'
        if norm_layer is None:
            self.norm_layer = nn.BatchNorm2d
        self.block = ResNetv1_BasicBlock
        self.in_channel_buff = 16
        # Define the Layers
        # If the Input Image is not large enough for such intense dimension reduction
        self.conv1 = nn.Conv2d(3, 16, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn1   = self.norm_layer(16)
        self.relu  = nn.ReLU(inplace = True)
        self.conv2_x = self._layer_gen(16,  16, ResBlock_Repetition)
        self.conv3_x = self._layer_gen(16,  32, ResBlock_Repetition, stride = 2)
        self.conv4_x = self._layer_gen(32,  64, ResBlock_Repetition, stride = 2)
        self.avgPool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * self.block.expansion, num_classes)
        # Initialzie the Network
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode = 'fan_out', nonlinearity = 'relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        # This zero-init applies to the last BN in each residual branch
        # This improves the mode by 0.2 % ~ 0.3 % according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, ResNetv1_BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
            
    def _layer_gen(self, in_channel, out_channel, layerList, stride = 1):
        # If it's not conv2_x
        if stride != 1:
            downsample = nn.Sequential(nn.Conv2d(self.in_channel_buff,
                                                 out_channel * self.block.expansion,
                                                 kernel_size = 1,
                                                 stride = 2),
                                       self.norm_layer(out_channel * self.block.expansion))
        else:
            downsample = None
        layers = []
        layers.append(self.block(self.in_channel_buff, out_channel, FirstConv_Stride = stride,
                                 downsample = downsample, norm_layer = self.norm_layer))
        self.in_channel_buff = out_channel * self.block.expansion
        for _ in range(1, layerList):
            layers.append(self.block(self.in_channel_buff, out_channel, FirstConv_Stride = 1,
                                     downsample = None, norm_layer = self.norm_layer))
        return nn.Sequential(* layers)

    
    def _forward_impl(self, x):
        x = self.bn1(self.conv1(x))
        x = self.relu(x)

        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)

        x = self.avgPool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
    
    def forward(self, x):
        return self._forward_impl(x)