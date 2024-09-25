from torch_geometric.loader import LinkNeighborLoader
import torch_geometric.transforms as T
import pickle as pkl
from generate_graph import pickle_path
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Loading the graph data
with open(pickle_path, 'rb') as f:
    data = pkl.load(f)
    
data = data.to(device)

print(f"Node types: {data.node_types}")
print(f"Edge types: {data.edge_types}")
print(f"Number of protein nodes: {data['protein'].num_nodes}")
print(f"Number of drug class nodes: {data['class'].num_nodes}")
print(f"Edge index shape: {data['protein', 'interacts_with', 'class'].edge_index.shape}")

transform = T.RandomLinkSplit(
    num_val=0.1,
    num_test=0.1,
    is_undirected=True,
    neg_sampling_ratio=2.0, 
    edge_types=('protein', 'interacts_with', 'class'),
    rev_edge_types=('class', 'interacted_by', 'protein'),  # Reverse edges
)

train_data, val_data, test_data = transform(data)

print("Edge splitting completed!")
print(f"Number of training edges: {train_data['protein', 'interacts_with', 'class'].edge_label_index.shape[1]}")
print(f"Number of validation edges: {val_data['protein', 'interacts_with', 'class'].edge_label_index.shape[1]}")
print(f"Number of test edges: {test_data['protein', 'interacts_with', 'class'].edge_label_index.shape[1]}")

# Define seed edges for training and move to device
edge_label_index = train_data['protein', 'interacts_with', 'class'].edge_label_index.to(device)
edge_label = train_data['protein', 'interacts_with', 'class'].edge_label.to(device)
print(f"Training edge label index shape: {edge_label_index.shape}")
print(f"Training edge labels shape: {edge_label.shape}")

# Link loader for training
train_loader = LinkNeighborLoader(
    data=train_data, 
    num_neighbors=[20, 10],  
    neg_sampling_ratio=2.0,  
    edge_label_index=(('protein', 'interacts_with', 'class'), edge_label_index),
    edge_label=edge_label,  
    batch_size=128,  
    shuffle=True,
)

# Link loader for validation
val_edge_label_index = val_data['protein', 'interacts_with', 'class'].edge_label_index.to(device)
val_edge_label = val_data['protein', 'interacts_with', 'class'].edge_label.to(device)

val_loader = LinkNeighborLoader(
    data=val_data,
    num_neighbors=[20, 10],
    neg_sampling_ratio=2.0,
    edge_label_index=(('protein', 'interacts_with', 'class'), val_edge_label_index),
    edge_label=val_edge_label,
    batch_size=128,
    shuffle=False,
)

# Link loader for testing
test_edge_label_index = test_data['protein', 'interacts_with', 'class'].edge_label_index.to(device)
test_edge_label = test_data['protein', 'interacts_with', 'class'].edge_label.to(device)

test_loader = LinkNeighborLoader(
    data=test_data,
    num_neighbors=[20, 10],
    neg_sampling_ratio=2.0,
    edge_label_index=(('protein', 'interacts_with', 'class'), test_edge_label_index),
    edge_label=test_edge_label,
    batch_size=128,
    shuffle=False,
)

print("Data loaders created with tensors moved to device!")
