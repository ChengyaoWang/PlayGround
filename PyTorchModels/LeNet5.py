import torch
import torch.nn as nn 
import torch.nn.functional as F

class LeNet5(nn.Module):
    model_name = 'LeNet5'
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 3,
                               out_channels = 6,
                               kernel_size = 5,
                               stride = 1,
                               padding = 0)
        self.maxpool1 = nn.MaxPool2d(kernel_size = 2,
                                     stride = 2,
                                     padding = 0)
        self.conv2 = nn.Conv2d(in_channels = 6,
                               out_channels = 16,
                               kernel_size = 5,
                               stride = 1,
                               padding = 0)
        self.maxpool2 = nn.MaxPool2d(kernel_size = 2,
                                     stride = 2,
                                     padding = 0)
        self.fc1 = nn.Linear(in_features = 400,
                             out_features = 120,
                             bias = True)
        self.fc2 = nn.Linear(in_features = 120,
                             out_features = 84,
                             bias = True)
        self.fc3 = nn.Linear(in_features = 84,
                             out_features = 10,
                             bias = True)

    def flatten_feature_num(self, x):
        # Except the Batch Size
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def forward(self, x):
        x = self.maxpool1(F.relu(self.conv1(x)))
        x = self.maxpool2(F.relu(self.conv2(x)))
        # Reshape
        x = x.view(-1, self.flatten_feature_num(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x