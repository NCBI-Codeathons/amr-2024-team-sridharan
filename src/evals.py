import torch
from tqdm import tqdm
from model import HeteroLinkPredictionModel
import pickle as pkl
from generate_graph import pickle_path
from loaders import val_loader
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_recall_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm

# Assuming model is already trained and val_loader is defined


# Set device to CUDA if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running on device: {device}")

hidden_channels = 64
out_channels = 32

# Function to load the trained model
def load_model(model_path, hidden_channels, out_channels):
    model = HeteroLinkPredictionModel(hidden_channels, out_channels)
    model = model.to(device)

    # Load the model checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

# Load the trained model
model_path = '/shared_venv/model_checkpoint_50_0.0001.pth'
model = load_model(model_path, hidden_channels, out_channels)


def evaluate_edge_prediction(model, loader, device='cuda'):
    """
    Evaluate edge prediction performance for a trained model.
    
    Arguments:
    - model: The trained model.
    - loader: DataLoader for validation or test data.
    
    Returns:
    - A dictionary of evaluation metrics (AUC-ROC, AUPR, etc.)
    """
    model.eval()
    preds, ground_truths = [], []
    
    # Iterate over validation or test set
    for batch in tqdm(loader, desc="Evaluating"):
        batch = batch.to(device)
        
        with torch.no_grad():
            # Forward pass
            pred = model(batch)
            pred = torch.sigmoid(pred)  # Get probabilities
            
            # Ground truth for edge labels
            ground_truth = batch['protein', 'interacts_with', 'drug_class'].edge_label
        
        preds.append(pred.cpu().numpy())
        ground_truths.append(ground_truth.cpu().numpy())
    
    # Concatenate all predictions and ground truth
    preds = np.concatenate(preds)
    ground_truths = np.concatenate(ground_truths)
    
    # Metrics calculation
    results = {}
    
    # ROC-AUC
    results['AUC-ROC'] = roc_auc_score(ground_truths, preds)
    
    # Precision-Recall Curve & AUPR
    precision, recall, _ = precision_recall_curve(ground_truths, preds)
    results['AUPR'] = auc(recall, precision)
    
    # Binarize predictions for confusion matrix
    pred_binary = (preds >= 0.5).astype(int)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(ground_truths, pred_binary).ravel()
    results['confusion_matrix'] = {'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn}
    
    return results


def plot_metrics(results):
    """
    Plot AUPR and Confusion Matrix.
    
    Arguments:
    - results: The dictionary of evaluation metrics returned by evaluate_edge_prediction
    """
    # Confusion Matrix plot
    cm = np.array([[results['confusion_matrix']['TN'], results['confusion_matrix']['FP']], 
                   [results['confusion_matrix']['FN'], results['confusion_matrix']['TP']]])
    
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig("confusion_matrix.png")
    plt.close()
''' # AUPR Plot (Precision-Recall Curve)
    plt.plot(precision, recall, marker='.')
    plt.title(f"AUPR: {results['AUPR']:.4f}")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.savefig('aupr_plot.png')
    plt.close()
'''

# Load your model and DataLoader (example)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#model = HeteroLinkPredictionModel(hidden_channels=64, out_channels=32).to(device)
# Load model checkpoint, etc.

# Evaluate on validation/test set
results = evaluate_edge_prediction(model, val_loader, device)

# Plot the evaluation metrics
plot_metrics(results)

# Print the metrics
print(f"AUC-ROC: {results['AUC-ROC']:.4f}")
print(f"AUPR: {results['AUPR']:.4f}")
print(f"Confusion Matrix: {results['confusion_matrix']}")
