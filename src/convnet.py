import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_1 = nn.Conv2d(1, 32, (3, 3))
        self.conv_2 = nn.Conv2d(32, 64, (3, 3))
        self.lin = nn.Linear(1600, 10)

    def forward(self, x):
        x = F.relu(self.conv_1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv_2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.shape[0], -1)
        return F.softmax(self.lin(x), dim=0)
