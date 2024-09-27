import torch
import torch.nn.functional as F
from sklearn.metrics import precision_recall_curve, auc, f1_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from model import HeteroLinkPredictionModel
from loaders import test_loader

# Set parameters for hidden channels, output channels, and device
hidden_channels = 64
out_channels = 32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the model from checkpoint
def load_model(model_path, hidden_channels, out_channels):
    model = HeteroLinkPredictionModel(hidden_channels, out_channels)
    model = model.to(device)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

# Evaluate the model on the test dataset
def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []

    pbar = tqdm(total=len(loader), desc="Evaluating")

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            # Forward pass
            pred = model(batch)
            ground_truth = batch['protein', 'interacts_with', 'drug_class'].edge_label

            # Store predictions and ground truth
            all_preds.append(torch.sigmoid(pred).cpu().numpy())
            all_labels.append(ground_truth.cpu().numpy())

            pbar.update(1)

    pbar.close()

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    return all_preds, all_labels

# Calculate AUPR and Fmax for each label
def calculate_metrics(y_true, y_pred, threshold=0.5):
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    aupr = auc(recall, precision)

    y_pred_bin = (y_pred >= threshold).astype(int)
    f1 = f1_score(y_true, y_pred_bin)

    return aupr, f1

# Calculate confusion matrix elements
def calculate_confusion_matrix_elements(y_true, y_pred, threshold=0.5):
    y_pred_bin = (y_pred >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_bin).ravel()
    return tp, fp, tn, fn

# Plot and save violin plots
def plot_violin(metrics, metric_name, filename):
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=metrics)
    plt.title(f'{metric_name} for each drug class')
    plt.xlabel('Drug Class')
    plt.ylabel(metric_name)
    plt.savefig(filename)
    plt.close()

# Plot confusion matrix
def plot_confusion_matrix(tp, fp, tn, fn, class_label, filename):
    cm = np.array([[tn, fp], [fn, tp]])
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix for Drug Class {class_label}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(filename)
    plt.close()

if __name__ == "__main__":
    # Load the trained model
    model_path = '/shared_venv/model_checkpoint_10_0.001.pth'
    model = load_model(model_path, hidden_channels, out_channels)

    # Evaluate the model on the test set
    y_preds, y_true = evaluate(model, test_loader)

    # Initialize lists to store AUPR, Fmax, and confusion matrix elements for each drug class
    auprs, fmax_scores = [], []
    tps, fps, tns, fns = [], [], [], []

    num_classes = y_true.shape[1]  # Number of drug classes

    for i in range(num_classes):
        # For each class (drug), get the true and predicted values
        y_true_class = y_true[:, i]
        y_pred_class = y_preds[:, i]

        # Calculate TP, FP, TN, FN
        tp, fp, tn, fn = calculate_confusion_matrix_elements(y_true_class, y_pred_class)
        tps.append(tp)
        fps.append(fp)
        tns.append(tn)
        fns.append(fn)

        # Calculate AUPR and Fmax for this class
        aupr, fmax = calculate_metrics(y_true_class, y_pred_class)
        auprs.append(aupr)
        fmax_scores.append(fmax)

        # Save Confusion Matrix plot
        plot_confusion_matrix(tp, fp, tn, fn, class_label=i, filename=f'confusion_matrix_class_{i}.png')

    # Plot and save violin plots for AUPR and Fmax scores
    plot_violin(auprs, metric_name="AUPR", filename="aupr_violin_plot.png")
    plot_violin(fmax_scores, metric_name="Fmax", filename="fmax_violin_plot.png")

    print(f'Evaluation completed. Plots saved.')
