from torch_geometric.loader import LinkNeighborLoader
import torch_geometric.transforms as T
import pickle as pkl
from generate_graph import pickle_path


'''
loading the graph data
'''

with open(pickle_path, 'rb') as f:
    data = pkl.load(f)


'''
testing if the data loaded correctly
'''
print(f"Node types: {data.node_types}")
print(f"Edge types: {data.edge_types}")
print(f"Number of protein nodes: {data['protein'].num_nodes}")
print(f"Number of drug class nodes: {data['class'].num_nodes}")
print(f"Edge index shape: {data['protein', 'interacts_with', 'class'].edge_index.shape}")


'''
splitting the edges into tran test and validation
'''
transform = T.RandomLinkSplit(
    num_val=0.1,
    num_test=0.1,
    is_undirected=True,
    neg_sampling_ratio=2.0, 
    edge_types=('protein', 'interacts_with', 'drug_class'),
    rev_edge_types=('drug_class', 'interacted_by', 'protein'),  # Reverse edges
)

train_data, val_data, test_data = transform(data)

print("Edge splitting completed!")
print(f"Number of training edges: {train_data['protein', 'interacts_with', 'drug_class'].edge_label_index.shape[1]}")
print(f"Number of validation edges: {val_data['protein', 'interacts_with', 'drug_class'].edge_label_index.shape[1]}")
print(f"Number of test edges: {test_data['protein', 'interacts_with', 'drug_class'].edge_label_index.shape[1]}")

# Define seed edges for training
edge_label_index = train_data['protein', 'interacts_with', 'drug_class'].edge_label_index
edge_label = train_data['protein', 'interacts_with', 'drug_class'].edge_label
print(f"Training edge label index shape: {edge_label_index.shape}")
print(f"Training edge labels shape: {edge_label.shape}")

