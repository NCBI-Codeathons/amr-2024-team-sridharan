import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GCNConv, GATConv, to_hetero
from torch.nn import Linear
import pickle as pkl
from generate_graph import pickle_path

# Load the graph data
with open(pickle_path, 'rb') as f:
    data = pkl.load(f)


class HeteroGraphModel(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()

        # First layer: SAGEConv for neighborhood aggregation
        self.conv1 = SAGEConv((-1, -1), hidden_channels)  # Use tuple (-1, -1) for SAGEConv
        
        # Second layer: GATConv, explicitly set add_self_loops=False
        self.conv2 = GATConv((-1, -1), hidden_channels, heads=4, concat=False, add_self_loops=False)

        # Linear layer for final embeddings
        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        # Apply first SAGEConv layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        
        # Apply second GATConv layer
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        # Final linear layer
        x = self.lin(x)
        return x


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # Linear layer for predicting the existence of edges
        self.lin = torch.nn.Linear(2 * in_channels, 1)  # Concatenate node embeddings

    def forward(self, z_protein, z_drug_class, edge_label_index):
        # Concatenate the source and target node embeddings
        src_emb = z_protein[edge_label_index[0]]  # Source node embeddings (protein)
        tgt_emb = z_drug_class[edge_label_index[1]]  # Target node embeddings (drug class)
        
        # Concatenate the source and target embeddings
        edge_features = torch.cat([src_emb, tgt_emb], dim=-1)

        # Apply linear layer to predict edge existence
        return torch.sigmoid(self.lin(edge_features)).squeeze()

class HeteroLinkPredictionModel(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        # Initialize the heterogeneous graph model (GCN)
        self.gnn = HeteroGraphModel(hidden_channels, out_channels)

        # Convert to heterogeneous model for multiple node types
        self.gnn = to_hetero(self.gnn, metadata=data.metadata())

        # Initialize the link predictor
        self.link_predictor = LinkPredictor(out_channels)

    def forward(self, data):
        # Get node embeddings from the GNN
        z_dict = self.gnn(data.x_dict, data.edge_index_dict)

        # Use the link predictor to predict the existence of edges
        pred = self.link_predictor(
            z_dict['protein'], z_dict['class'],  # Use embeddings for 'protein' and 'class'
            data['protein', 'interacts_with', 'class'].edge_label_index
        )
        return pred
