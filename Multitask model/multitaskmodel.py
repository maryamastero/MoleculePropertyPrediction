import torch
from gin import GIN
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool, global_max_pool

class MultiTaskModel(nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_layers ,num_classes, out_channels):
        super(MultiTaskModel, self).__init__()
        self.gin = GIN(num_node_features, hidden_channels, num_layers)
        self.node_classification_head = nn.Linear(hidden_channels, num_classes)
        self.property_prediction_head = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, x, edge_index, batch):
        shared_rep = self.gin(x, edge_index)
        node_embedding = F.relu(shared_rep)
        node_classification_out = self.node_classification_head(node_embedding)

        global_representation = torch.cat([global_mean_pool(shared_rep, batch), global_max_pool(shared_rep, batch)], dim=1)
        property_prediction_out = self.property_prediction_head(global_representation)

        return node_classification_out, property_prediction_out
