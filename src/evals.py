import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim import Adam
from model import HeteroLinkPredictionModel
from loaders import train_loader, val_loader, test_loader
import pickle as pkl
from generate_graph import pickle_path
from loaders import train_loader, val_loader, test_loaders, train_data, test_data, val_data
from torch_geometric.loader import LinkNeighborLoader
from sklearn.metrics import roc_auc_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

hidden_channels = 64
out_channels = 32
num_heads = 4

'''
loading the graph data
'''


def load_model(model_path, hidden_channels, out_channels):
    # Initialize the model
    model = HeteroLinkPredictionModel(hidden_channels, out_channels)
    model = model.to(device)

    # Load the model checkpoint
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

    # Load the trained model
model_path = '/shared_venv/model_checkpoint_10_0.001.pth'
model = load_model(model_path, hidden_channels, out_channels)

with open(pickle_path, 'rb') as f:
    data = pkl.load(f)

# Define the validation seed edges:
edge_label_index = val_data['protein', 'interacts_with', 'drug_class'].edge_label_index
edge_label = val_data['drug_class', 'interacts_with', 'protein'].edge_label
val_loader = LinkNeighborLoader(
    data=val_data,
    num_neighbors=[20, 10],
    edge_label_index=(('protein', 'interacts_with', 'drug_class'), edge_label_index),
    edge_label=edge_label,
    batch_size=3 * 128,
    shuffle=False,
)
sampled_data = next(iter(val_loader))

preds = []
ground_truths = []
for sampled_data in tqdm.tqdm(val_loader):
    with torch.no_grad():
        sampled_data.to(device)
        preds.append(model(sampled_data))
        ground_truths.append(sampled_data['protein', 'interacts_with', 'drug_class'].edge_label)
pred = torch.cat(preds, dim=0).cpu().numpy()
ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()
auc = roc_auc_score(ground_truth, pred)
print()
print(f"Validation AUC: {auc:.4f}")