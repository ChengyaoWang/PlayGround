import torch
import torch.nn as nn 
import torch.nn.functional as F

# Padding Values are slightly different from original paper
# self.conv1 has stride 4 -> 2 to fit the size of Cifar 10
class AlexNet(nn.Module):
    model_name = 'AlexNet'
    def __init__(self, num_of_classes = 10):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size = 11, stride = 2, padding = 4)
        self.maxpool1 = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 0)
        self.conv2 = nn.Conv2d(96, 256, kernel_size = 5, stride = 1, padding = 2)
        self.maxpool2 = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 0)
        self.conv3 = nn.Conv2d(256, 384, kernel_size = 3, stride = 1, padding = 1)
        self.conv4 = nn.Conv2d(384, 384, kernel_size = 3, stride = 1, padding = 1)
        self.conv5 = nn.Conv2d(384, 256, kernel_size = 3, stride = 1, padding = 1)
        self.maxpool3 = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 0)
        self.avgPool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(nn.Dropout(0.5),
                                        nn.Linear(256 * 6 * 6, 4096),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(0.5),
                                        nn.Linear(4096, 4096),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(4096, num_of_classes))

    def forward(self, x):
        x = self.maxpool1(F.relu(self.conv1(x)))
        x = self.maxpool2(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.maxpool3(F.relu(self.conv5(x)))
        # Reshape
        x = self.avgPool(x)
        x = x.view(-1, 256 * 6 * 6)
        x = self.classifier(x)
        return x