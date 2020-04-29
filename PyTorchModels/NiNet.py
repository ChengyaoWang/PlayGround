import torch
import torch.nn as nn 



class NetworkInNetwork(nn.Module):
    model_name = 'NiNet'
    def __init__(self):
        super(NetworkInNetwork, self).__init__()

        self.mlpconv_1 = self._MLPCONV_fetch([3, 192, 160, 96], IF_FIRST = True)
        self.maxpool_1 = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 0)
        self.mlpconv_2 = self._MLPCONV_fetch([96, 192, 192, 192])
        self.maxpool_2 = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 0)
        self.mlpconv_3 = self._MLPCONV_fetch([192, 192, 192, 10])
        self.avgpool_3 = nn.AdaptiveAvgPool2d((1, 1))

    def _MLPCONV_fetch(self, width: list, IF_FIRST = False):
        layers = []
        if not IF_FIRST:
            layers.append(nn.Dropout(0.5))
        dim0, dim1, dim2, dim3 = width[0], width[1], width[2], width[3]
        layers += [nn.Conv2d(dim0, dim1, kernel_size = 5, stride = 1, padding = 2), nn.ReLU(inplace = True),
                   nn.Conv2d(dim1, dim2, kernel_size = 1, stride = 1, padding = 0), nn.ReLU(inplace = True),
                   nn.Conv2d(dim2, dim3, kernel_size = 1, stride = 1, padding = 0), nn.ReLU(inplace = True)]

        return nn.Sequential( * layers)

    def forward(self, X):
        X = self.maxpool_1(self.mlpconv_1(X))
        X = self.maxpool_2(self.mlpconv_2(X))
        X = self.avgpool_3(self.mlpconv_3(X))
        X = X.view(-1, 10)
        return X