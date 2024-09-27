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

# Function to calculate AUPR
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

# Function to calculate AUC and AUPR for each class
def evaluate_per_class(y_true, y_pred, num_classes):
    auc_scores = []
    aupr_scores = []
    confusion_matrices = []

    for i in range(num_classes):
        y_true_class = y_true[:, i]
        y_pred_class = y_pred[:, i]

        # Calculate AUC
        auc_score = roc_auc_score(y_true_class, y_pred_class)
        auc_scores.append(auc_score)

        # Calculate AUPR
        aupr_score = calculate_aupr(y_true_class, y_pred_class)
        aupr_scores.append(aupr_score)

        # Calculate confusion matrix
        tn, fp, fn, tp = calculate_confusion_matrix(y_true_class, y_pred_class)
        confusion_matrices.append([[tn, fp], [fn, tp]])

    return auc_scores, aupr_scores, confusion_matrices

# Load the trained model
model_path = '/shared_venv/model_checkpoint_50_0.0001.pth'
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
        pred = torch.sigmoid(model(batch))  # Get probabilities
        preds.append(pred)
        ground_truths.append(batch['protein', 'interacts_with', 'drug_class'].edge_label)

# Concatenate predictions and ground truths
pred = torch.cat(preds, dim=0).cpu().numpy()
ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()

# Calculate per-class AUC, AUPR, and confusion matrices
num_classes = ground_truth.shape[1]  # Assuming multi-label data with one-hot encoding
auc_scores, aupr_scores, confusion_matrices = evaluate_per_class(ground_truth, pred, num_classes)

# Save AUC, AUPR, and Confusion Matrix values to a text file
with open('evaluation_metrics.txt', 'w') as f:
    for i in range(num_classes):
        f.write(f"Class {i}: AUC = {auc_scores[i]:.4f}, AUPR = {aupr_scores[i]:.4f}\n")
        f.write(f"Confusion Matrix: TN={confusion_matrices[i][0][0]}, FP={confusion_matrices[i][0][1]}, FN={confusion_matrices[i][1][0]}, TP={confusion_matrices[i][1][1]}\n\n")

# Plot confusion matrices and save AUPR curves
for i in range(num_classes):
    # Confusion matrix plot
    cm = np.array(confusion_matrices[i])
    plot_confusion_matrix(cm, class_label=f"Class {i}", filename=f"confusion_matrix_class_{i}.png")

# Save summary AUPR plot
plt.figure(figsize=(10, 6))
plt.bar(range(num_classes), aupr_scores)
plt.title('AUPR Scores per Class')
plt.xlabel('Class')
plt.ylabel('AUPR')
plt.savefig('aupr_scores_per_class.png')
plt.close()

print("Evaluation completed, plots saved.")
