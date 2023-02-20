import torch
import torch.nn as nn
import torch.nn.functional as F

class TFCifar(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.conv2 = nn.Conv2d(64, 64, 5)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        self.lin1 = nn.Linear(1600, 384)
        self.lin2 = nn.Linear(384, 192)
        self.lin3 = nn.Linear(192, 10, bias=False)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
        x = torch.flatten(x, 1)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = self.lin3(x)
        return x