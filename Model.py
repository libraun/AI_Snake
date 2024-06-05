import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        self.linear = nn.Linear(8,32)
        self.linear2 = nn.Linear(32,1)
    def forward(self):
        F.relu()