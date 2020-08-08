import torch
import torch.nn as nn
import torch.nn.functional as F

from my_args import get_parser

# =============================================================================
parser = get_parser()
opts = parser.parse_args()
# # =============================================================================

class FCNet(nn.Module):
    def __init__(self):
        super(FCNet, self).__init__()
        self.fc1 = nn.Linear(3*28*28, 128)
        self.fc2 = nn.Linear(128,       5)

    def forward(self, x):
        x = x.view(-1, 3*28*28) # flatten
        x = torch.sigmoid(self.fc1(x))
        # x = F.softmax(self.fc2(x), dim=0)
        x = self.fc2(x)
        return x
