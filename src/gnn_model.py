import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from src.graph_factory import build_hydrological_graph, extract_graph_matrices

def prepare_pytorch_geometric_data():
    csv_file_path = "data/raw/india_groundwater_2025.csv"
    G = build_hydrological_graph(csv_file_path, radius_km=100.0)
    _, edge_index = extract_graph_matrices(G)
    
    node_features = [
        [15.4, 1.2, 65.0],
        [22.1, 0.8, 60.0],
        [18.9, 2.1, 45.0],
        [45.2, 3.4, 20.0]
    ]
    x = torch.tensor(node_features, dtype=torch.float)
    targets = [16.1, 23.4, 19.5, 46.8]
    y = torch.tensor(targets, dtype=torch.float).unsqueeze(1)
    
    return Data(x=x, edge_index=edge_index, y=y)

class HydrologySTGNN(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super(HydrologySTGNN, self).__init__()
        self.spatial_conv1 = GCNConv(in_channels, hidden_channels)
        self.spatial_conv2 = GCNConv(hidden_channels, hidden_channels)
        self.relu = nn.ReLU()
        self.regression_head = nn.Linear(hidden_channels, out_channels)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        out = self.spatial_conv1(x, edge_index)
        out = self.relu(out)
        out = self.spatial_conv2(out, edge_index)
        out = self.relu(out)
        return self.regression_head(out)