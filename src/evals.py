import torch
import torch.nn.functional as F
from sklearn.metrics import precision_recall_curve, auc, confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from model import HeteroLinkPredictionModel
from loaders import test_loader

hidden_channels = 64
out_channels = 32
num_heads = 4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(model_path, hidden_channels, out_channels):
    # Initialize the model
    model = HeteroLinkPredictionModel(hidden_channels, out_channels)
    model = model.to(device)

    # Load the model checkpoint
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

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
            
            print(f"Shape of y_pred: {all_preds.shape()}")
            print(f"Shape of y_true:{all_labels.shape()}")
            pbar.update(1)

    pbar.close()

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    return all_preds, all_labels

def calculate_confusion_matrix_elements(y_true, y_pred, threshold=0.5):
    y_pred_bin = (y_pred >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_bin).ravel()
    return tp, fp, tn, fn

def calculate_metrics(y_true, y_pred):
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    aupr = auc(recall, precision)
    
    f1_scores = 2 * (precision * recall) / (precision + recall)
    f1_scores = np.nan_to_num(f1_scores)  # Replace NaN with 0
    fmax = np.max(f1_scores)
    
    return aupr, fmax

def plot_metrics(metrics, labels, metric_name="AUPR"):
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=metrics)
    plt.title(f'{metric_name} for each drug class')
    plt.xlabel('Drug Class')
    plt.ylabel(metric_name)
    plt.show()

if __name__ == "__main__":
    # Load the trained model
    model_path = '/shared_venv/model_checkpoint_10_0.001.pth'
    model = load_model(model_path, hidden_channels, out_channels)

    # Evaluate the model on the test set
    y_preds, y_true = evaluate(model, test_loader)
    print(f"Shape of y_pred: {y_preds}")
    print(f"Shape of y_true:{y_true}")
    print(f"Shape of y_pred: {y_preds[0].shape()}")
    print(f"Shape of y_true:{y_true[0].shape()}")

    # Initialize lists to store AUPR and Fmax for each drug class
    auprs, fmax_scores = [], []
    
    # Confusion matrix elements for each label
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

    # Plot Confusion Matrix for the first class as an example
    print(f'TP: {tps[0]}, FP: {fps[0]}, TN: {tns[0]}, FN: {fns[0]}')

    # Plot Violin plots for AUPR and Fmax scores
    plot_metrics(auprs, labels=list(range(num_classes)), metric_name="AUPR")
    plot_metrics(fmax_scores, labels=list(range(num_classes)), metric_name="Fmax")
