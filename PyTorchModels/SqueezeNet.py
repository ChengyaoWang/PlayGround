import torch 
import torch.nn as nn 

class Fire_Basic(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Fire_Basic, self).__init__()
        self.conv_1 = nn.Conv2d(in_channel, out_channel // 8, kernel_size = 1, stride = 1, padding = 0)
        self.relu_1 = nn.ReLU(inplace = True)
        self.conv_21 = nn.Conv2d(out_channel // 8, out_channel // 2, kernel_size = 1, stride = 1, padding = 0)
        self.conv_23 = nn.Conv2d(out_channel // 8, out_channel // 2, kernel_size = 3, stride = 1, padding = 1)
        self.relu_2 = nn.ReLU(inplace = True)
    
    def forward(self, X):
        X = self.relu_1(self.conv_1(X))
        X = torch.cat([self.conv_21(X), self.conv_23(X)], 1)
        X = self.relu_2(X)
        return X

class Fire_SimpleBypass(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Fire_SimpleBypass, self).__init__()
        self.conv_1 = nn.Conv2d(in_channel, out_channel // 8, kernel_size = 1, stride = 1, padding = 0)
        self.relu_1 = nn.ReLU(inplace = True)
        self.conv_21 = nn.Conv2d(out_channel // 8, out_channel // 2, kernel_size = 1, stride = 1, padding = 0)
        self.conv_23 = nn.Conv2d(out_channel // 8, out_channel // 2, kernel_size = 3, stride = 1, padding = 1)
        self.relu_2 = nn.ReLU(inplace = True)
    
    def forward(self, X):
        identity = X
        X = self.relu_1(self.conv_1(X))
        X = torch.cat([self.conv_21(X), self.conv_23(X)], 1)
        X = self.relu_2(X + identity)
        return X 

class Fire_ComplexBypass(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Fire_ComplexBypass, self).__init__()
        self.bypass = nn.Conv2d(in_channel, out_channel, kernel_size = 1, stride = 1, padding = 0)
        self.conv_1 = nn.Conv2d(in_channel, out_channel // 8, kernel_size = 1, stride = 1, padding = 0)
        self.relu_1 = nn.ReLU(inplace = True)
        self.conv_21 = nn.Conv2d(out_channel // 8, out_channel // 2, kernel_size = 1, stride = 1, padding = 0)
        self.conv_23 = nn.Conv2d(out_channel // 8, out_channel // 2, kernel_size = 3, stride = 1, padding = 1)
        self.relu_2 = nn.ReLU(inplace = True)
    
    def forward(self, X):
        identity = X
        X = self.relu_1(self.conv_1(X))
        X = torch.cat([self.conv_21(X), self.conv_23(X)], 1)
        X = self.relu_2(X + self.bypass(X))
        return X 


'''
    We are defining Variants Seperately for Time efficiency
'''
class SqueezeNet(nn.Module):

    def __init__(self, variant = 'Basic', num_of_classes = 10, SPARSIFY = False):
        super(SqueezeNet, self).__init__()
        if variant == 'Basic':
            self.variant = 0x00
        elif variant == 'SimpleBypass':
            self.variant = 0x01
        elif variant == 'ComplexBypass':
            self.variant = 0x03
        else:
            raise ValueError('Current Variant not supported')
        self.model_name = 'SqueezeNet_' + variant
        '''
            Model Body
        '''
        self.conv_1    = nn.Conv2d(3, 96, kernel_size = 7, stride = 2, padding = 3)
        self.relu_1    = nn.ReLU(inplace = True)
        self.maxpool_1 = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 0, ceil_mode = True)
        self.fire_2    = Fire_ComplexBypass(96, 128)  if self.variant & 0x02 else Fire_Basic(96, 128)
        self.fire_3    = Fire_SimpleBypass(128, 128)  if self.variant & 0x01 else Fire_Basic(128, 128)
        self.fire_4    = Fire_ComplexBypass(128, 256) if self.variant & 0x02 else Fire_Basic(128, 256)
        self.maxpool_2 = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 0, ceil_mode = True)
        self.fire_5    = Fire_SimpleBypass(256, 256)  if self.variant & 0x01 else Fire_Basic(256, 256)
        self.fire_6    = Fire_ComplexBypass(256, 384) if self.variant & 0x02 else Fire_Basic(256, 384)
        self.fire_7    = Fire_SimpleBypass(384, 384)  if self.variant & 0x01 else Fire_Basic(384, 384)
        self.fire_8    = Fire_ComplexBypass(384, 512) if self.variant & 0x02 else Fire_Basic(384, 512)
        self.maxpool_3 = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 0, ceil_mode = True)
        self.fire_9    = Fire_SimpleBypass(512, 512)  if self.variant & 0x01 else Fire_Basic(512, 512)
        self.dropout   = nn.Dropout(p = 0.5)
        self.conv_10   = nn.Conv2d(512, num_of_classes, kernel_size = 1, stride = 1, padding = 0)
        self.relu_2    = nn.ReLU(inplace = True)
        self.avgpool   = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, X):
        X = self.relu_1(self.conv_1(X))
        X = self.maxpool_1(X)
        X = self.fire_2(X)
        X = self.fire_3(X)
        X = self.fire_4(X)
        X = self.maxpool_2(X)
        X = self.fire_5(X)
        X = self.fire_6(X)
        X = self.fire_7(X)
        X = self.fire_8(X)
        X = self.maxpool_3(X)
        X = self.fire_9(X)
        # X = self.dropout(X)
        X = self.relu_2(self.conv_10(X))
        X = self.avgpool(X)
        return torch.flatten(X, 1)