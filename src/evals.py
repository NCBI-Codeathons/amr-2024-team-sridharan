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

# Set device to CUDA if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running on device: {device}")

hidden_channels = 64
out_channels = 32
batch_size = 512

# Function to load the trained model
def load_model(model_path, hidden_channels, out_channels):
    model = HeteroLinkPredictionModel(hidden_channels, out_channels)
    model = model.to(device)

    # Load the model checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

# Function to calculate confusion matrix elements
def calculate_confusion_matrix(y_true, y_pred, threshold=0.5):
    y_pred_bin = (y_pred >= threshold).astype(int)  # Binarize predictions based on threshold
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_bin).ravel()
    return tn, fp, fn, tp

# Function to calculate AUPR and F1-Score
def calculate_aupr(y_true, y_pred):
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    aupr = auc(recall, precision)
    return aupr

# Function to plot confusion matrix
def plot_confusion_matrix(cm, class_label, filename):
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"Confusion Matrix for Class {class_label}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(filename)
    plt.close()

# Load the trained model
model_path = '/shared_venv/model_checkpoint_50_0.001.pth'
model = load_model(model_path, hidden_channels, out_channels)
model = model.to(device)

# Load the graph data
with open(pickle_path, 'rb') as f:
    data = pkl.load(f)

model.eval()
preds = []
ground_truths = []

pbar = tqdm(total=len(val_loader), desc="Evaluating")

with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)

            # Forward pass
            pred = model(batch)
            ground_truth = batch['protein', 'interacts_with', 'drug_class'].edge_label
            pred = torch.sigmoid(pred) >= 0.5

            preds.append(pred)
            ground_truths.append(batch['protein', 'interacts_with', 'drug_class'].edge_label)

# Concatenate predictions and ground truths
pred = torch.cat(preds, dim=0).cpu().numpy()
ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()

# Calculate AUC score
auc_score = roc_auc_score(ground_truth, pred)
print(f"\nValidation AUC: {auc_score:.4f}")

# Calculate confusion matrix and AUPR
tn, fp, fn, tp = calculate_confusion_matrix(ground_truth, pred)
print(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

# Calculate AUPR
aupr = calculate_aupr(ground_truth, pred)
print(f"Validation AUPR: {aupr:.4f}")

# Plot and save confusion matrix
cm = np.array([[tn, fp], [fn, tp]])
plot_confusion_matrix(cm, class_label="All", filename="confusion_matrix_all_classes.png")

# Save AUPR and Confusion Matrix plot
with open('evaluation_metrics.txt', 'w') as f:
    f.write(f"AUC: {auc_score:.4f}\n")
    f.write(f"AUPR: {aupr:.4f}\n")
    f.write(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}\n")
