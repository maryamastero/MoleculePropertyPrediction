import torch
from torch_geometric.loader import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from pretrain_model import MaskedAtomIdentification
from utils import *
import matplotlib.pyplot as plt
from zincdataset import *
import os
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import pickle
from torch import nn

from sklearn.metrics import mean_squared_error, r2_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PretrainedGINForPropertyPrediction(nn.Module):
    def __init__(self, pretrained_gin, hidden_channels, out_channels):
        super(PretrainedGINForPropertyPrediction, self).__init__()
        self.gin = pretrained_gin  # Use pretrained GIN layers
        self.bn = nn.BatchNorm1d(hidden_channels)
        self.fc1 = nn.Linear(hidden_channels * 2, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        x = self.gin(x, edge_index)
        x = self.bn(x)
        global_representation = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = F.relu(self.fc1(global_representation))
        x = self.fc2(x)
        return x

