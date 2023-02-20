import torch
import torch.nn as nn
import torch.nn.functional as F

class DNet(nn.Module):
    def __init__(self, nb_labels=10, dropout=0.1):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 3)

        self.pool = nn.MaxPool2d(2, 2)

        self.dropout = nn.Dropout(p=dropout)

        self.lin1 = nn.Linear(64*2*2, 512)
        self.lin2 = nn.Linear(512, 10, bias=False)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = F.relu(self.lin1(x))
        x = self.dropout(x)
        x = self.lin2(x)
        return x
