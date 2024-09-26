import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv, to_hetero
from torch.nn import Linear
import pickle as pkl
from generate_graph import pickle_path


'''
loading the graph data
'''

with open(pickle_path, 'rb') as f:
    data = pkl.load(f)


class HeteroGraphModel(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_heads=4):
        super().__init__()

        # GraphSAGE for the first layer
        self.conv1 = SAGEConv((-1, -1), hidden_channels)

        # GATConv for the second layer, with add_self_loops set to False
        self.conv2 = GATConv((-1, -1), hidden_channels, heads=num_heads, concat=False, add_self_loops=False)

        # Final linear layer
        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        # First layer: GraphSAGE
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        
        # Second layer: GATConv
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        # Final linear layer
        x = self.lin(x)
        return x

class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.lin = torch.nn.Linear(in_channels, 1)

    def forward(self, z_protein, z_class, edge_label_index):
        # Extract the embeddings for source and target nodes
        src_emb = z_protein[edge_label_index[0]]  
        tgt_emb = z_class[edge_label_index[1]]    

        # Element-wise product of source and target node embeddings
        edge_features = src_emb * tgt_emb

        # Predict edge probability using a sigmoid layer
        return torch.sigmoid(self.lin(edge_features)).squeeze()

class HeteroLinkPredictionModel(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_heads=4):
        super().__init__()
        # Initialize the heterogeneous graph model (GraphSAGE + GATConv)
        self.gnn = HeteroGraphModel(hidden_channels, out_channels, num_heads)

        # Convert the homogeneous GNN into a heterogeneous GNN
        self.gnn = to_hetero(self.gnn, metadata=data.metadata())

        # Initialize the link predictor
        self.link_predictor = LinkPredictor(out_channels)

    def forward(self, data):
        # Get node embeddings from the GNN
        z_dict = self.gnn(data.x_dict, data.edge_index_dict)

        # Use the link predictor to predict the existence of edges
        pred = self.link_predictor(
            z_dict['protein'], z_dict['class'],
            data['protein', 'interacts_with', 'drug_class'].edge_label_index
        )
        return pred
