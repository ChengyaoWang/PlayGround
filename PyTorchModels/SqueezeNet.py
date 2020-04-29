import torch 
import torch.nn as nn 

POOL = ['Basic',  'Bypass_Simple', 'Bypass_Complex']

# Function to Fetch SqueezeNet
def Fetch_SqueezeNet(model_name):
    if model_name not in POOL:
        raise ValueError('Current Variant not supported')
    if model_name == 'Basic':
        return SqueezeNet_Basic()
    elif model_name == 'Bypass_Simple':
        return SqueezeNet_BypassSimple()
    elif model_name == 'Bypass_Complex':
        return SqueezeNet_BypassComplex()


class fireModule(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(fireModule, self).__init__()
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

'''
    We are defining Variants Seperately for Time efficiency
'''
class SqueezeNet_Basic(nn.Module):
    model_name = 'SqueezeNet_Basic'
    def __init__(self, num_of_classes = 10, SPARSIFY = False, INIT = True):
        super(SqueezeNet_Basic, self).__init__()
        self.conv_1    = nn.Conv2d(3, 96, kernel_size = 7, stride = 2, padding = 3)
        self.relu_1    = nn.ReLU(inplace = True)
        self.maxpool_1 = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 0, ceil_mode = True)
        self.fire_2    = fireModule(96, 128)
        self.fire_3    = fireModule(128, 128)
        self.fire_4    = fireModule(128, 256)
        self.maxpool_2 = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 0, ceil_mode = True)
        self.fire_5    = fireModule(256, 256)
        self.fire_6    = fireModule(256, 384)
        self.fire_7    = fireModule(384, 384)
        self.fire_8    = fireModule(384, 512)
        self.maxpool_3 = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 0, ceil_mode = True)
        self.fire_9    = fireModule(512, 512)
        self.dropout   = nn.Dropout(p = 0.5)
        self.conv_10   = nn.Conv2d(512, num_of_classes, kernel_size = 1, stride = 1, padding = 0)
        self.relu_2    = nn.ReLU(inplace = True)
        self.avgpool   = nn.AdaptiveAvgPool2d((1, 1))

        if INIT:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, X):
        X = self.relu_1(self.conv_1(X))
        X = self.maxpool_1(X)
        X = self.fire_2(X)# + self.shortcut_2(X)
        X = self.fire_3(X)# + X
        X = self.fire_4(X)# + self.shortcut_4(X)
        X = self.maxpool_2(X)
        X = self.fire_5(X)# + X
        X = self.fire_6(X)# + self.shortcut_6(X)
        X = self.fire_7(X)# + X
        X = self.fire_8(X)# + self.shortcut_8(X)
        X = self.maxpool_3(X)
        X = self.fire_9(X)# + X
        X = self.dropout(X)
        X = self.relu_2(self.conv_10(X))
        X = self.avgpool(X)
        X = torch.flatten(X, 1)
        return X

class SqueezeNet_BypassSimple(nn.Module):
    model_name = 'SqueezeNet_BypassSimple'
    def __init__(self, num_of_classes = 10, SPARSIFY = False, INIT = True):
        super(SqueezeNet_Basic, self).__init__()
        self.conv_1    = nn.Conv2d(3, 96, kernel_size = 7, stride = 2, padding = 3)
        self.relu_1    = nn.ReLU(inplace = True)
        self.maxpool_1 = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 0)
        self.fire_2    = fireModule(96, 128)
        self.fire_3    = fireModule(128, 128)
        self.fire_4    = fireModule(128, 256)
        self.maxpool_2 = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 0)
        self.fire_5    = fireModule(256, 256)
        self.fire_6    = fireModule(256, 384)
        self.fire_7    = fireModule(384, 384)
        self.fire_8    = fireModule(384, 512)
        self.maxpool_3 = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 0)
        self.fire_9    = fireModule(512, 512)
        self.dropout   = nn.Dropout(0.5, inplace = True)
        self.conv_10   = nn.Conv2d(512, num_of_classes, kernel_size = 1, stride = 1, padding = 0)
        self.relu_2    = nn.ReLU(inplace = True)
        self.avgpool   = nn.AdaptiveAvgPool2d((1, 1))

        if INIT:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, X):
        X = self.relu_1(self.conv_1(X))
        X = self.maxpool_1(X)
        X = self.fire_2(X)# + self.shortcut_2(X)
        X = self.fire_3(X) + X
        X = self.fire_4(X)# + self.shortcut_4(X)
        X = self.maxpool_2(X)
        X = self.fire_5(X) + X
        X = self.fire_6(X)# + self.shortcut_6(X)
        X = self.fire_7(X) + X
        X = self.fire_8(X)# + self.shortcut_8(X)
        X = self.maxpool_3(X)
        X = self.fire_9(X) + X
        X = self.dropout(X)
        X = self.relu_2(self.conv_10(X))
        X = self.avgpool(X)
        X = torch.flatten(X, 1)
        return X

class SqueezeNet_BypassComplex(nn.Module):
    model_name = 'SqueezeNet_BypassComplex'
    def __init__(self, num_of_classes = 10, SPARSIFY = False, INIT = True):
        super(SqueezeNet_Basic, self).__init__()
        self.conv_1    = nn.Conv2d(3, 96, kernel_size = 7, stride = 2, padding = 3)
        self.relu_1    = nn.ReLU(inplace = True)
        self.maxpool_1 = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 0)
        self.fire_2    = fireModule(96, 128)
        self.shortcut_2 = nn.Conv2d(96, 128, kernel_size = 1, stride = 1, padding = 0)
        self.fire_3    = fireModule(128, 128)
        self.fire_4    = fireModule(128, 256)
        self.shortcut_4 = nn.Conv2d(128, 256, kernel_size = 1, stride = 1, padding = 0)
        self.maxpool_2 = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 0)
        self.fire_5    = fireModule(256, 256)
        self.fire_6    = fireModule(256, 384)
        self.shortcut_6 = nn.Conv2d(256, 384, kernel_size = 1, stride = 1, padding = 0)
        self.fire_7    = fireModule(384, 384)
        self.fire_8    = fireModule(384, 512)
        self.shortcut_8 = nn.Conv2d(384, 512, kernel_size = 1, stride = 1, padding = 0)
        self.maxpool_3 = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 0)
        self.fire_9    = fireModule(512, 512)
        self.dropout   = nn.Dropout(0.5, inplace = True)
        self.conv_10   = nn.Conv2d(512, num_of_classes, kernel_size = 1, stride = 1, padding = 0)
        self.relu_2    = nn.ReLU(inplace = True)
        self.avgpool   = nn.AdaptiveAvgPool2d((1, 1))

        if INIT:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)


    def forward(self, X):
        X = self.relu_1(self.conv_1(X))
        X = self.maxpool_1(X)
        X = self.fire_2(X) + self.shortcut_2(X)
        X = self.fire_3(X) + X
        X = self.fire_4(X) + self.shortcut_4(X)
        X = self.maxpool_2(X)
        X = self.fire_5(X) + X
        X = self.fire_6(X) + self.shortcut_6(X)
        X = self.fire_7(X) + X
        X = self.fire_8(X) + self.shortcut_8(X)
        X = self.maxpool_3(X)
        X = self.fire_9(X) + X
        X = self.dropout(X)
        X = self.relu_2(self.conv_10(X))
        X = self.avgpool(X)
        X = torch.flatten(X, 1)
        return X


