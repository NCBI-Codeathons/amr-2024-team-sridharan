import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim import Adam
from model import HeteroLinkPredictionModel
from loaders import train_loader, val_loader, test_loader
import pickle as pkl
from generate_graph import pickle_path


hidden_channels = 64
out_channels = 32
num_heads = 4

'''
loading the graph data
'''

with open(pickle_path, 'rb') as f:
    data = pkl.load(f)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(model, loader, optimizer):
    '''
    Training fucntion
    '''
    model.train()
    total_loss = 0
    total_examples = 0
    
    #To show progress
    pbar = tqdm(total=len(loader), desc="Training")
    
    for batch in loader:
        optimizer.zero_grad()

        # Move batch to device
        batch = batch.to(device)

        # Forward pass
        pred = model(batch)

        # Ground truth for edge labels
        ground_truth = batch['protein', 'interacts_with', 'drug_class'].edge_label

        # Compute binary cross-entropy loss
        loss = F.binary_cross_entropy_with_logits(pred, ground_truth)

        # Backpropagation
        loss.backward()
        optimizer.step()

        # Track total loss and number of examples
        total_loss += loss.item() * ground_truth.size(0)
        total_examples += ground_truth.size(0)

        pbar.update(1)

    pbar.close()
    
    return total_loss / total_examples


def validate(model, loader):
    '''
    validation
    '''
    model.eval()
    total_loss = 0
    total_examples = 0
    
    pbar = tqdm(total=len(loader), desc="Validating")

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            # Forward pass
            pred = model(batch)
            ground_truth = batch['protein', 'interacts_with', 'drug_class'].edge_label

            # Compute loss
            loss = F.binary_cross_entropy_with_logits(pred, ground_truth)

            # Track total loss and number of examples
            total_loss += loss.item() * ground_truth.size(0)
            total_examples += ground_truth.size(0)

            pbar.update(1)

    pbar.close()
    
    return total_loss / total_examples


def test(model, loader):
    '''
    Test function (can also be used for evaluation)

    '''
    model.eval()
    total_correct = 0
    total_examples = 0
    
    pbar = tqdm(total=len(loader), desc="Testing")

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            # Forward pass
            pred = model(batch)
            ground_truth = batch['protein', 'interacts_with', 'drug_class'].edge_label

            # Apply sigmoid to convert logits to probabilities, then threshold at 0.5
            pred = torch.sigmoid(pred) >= 0.5
            total_correct += pred.eq(ground_truth).sum().item()
            total_examples += ground_truth.size(0)

            pbar.update(1)

    pbar.close()
    
    return total_correct / total_examples


epochs=10
lr=0.001

model = HeteroLinkPredictionModel(hidden_channels, out_channels)
model = model.to(device)

optimizer = Adam(model.parameters(), lr=lr)

train_losses, val_losses = [], []
for epoch in range(1, epochs + 1):
    print(f'Epoch {epoch}/{epochs}')
    
    # Train
    train_loss = train(model, train_loader, optimizer)
    train_losses.append(train_loss)
    print(f'Train Loss: {train_loss:.4f}')
    
    # Validate
    val_loss = validate(model, val_loader)
    val_losses.append(val_loss)
    print(f'Validation Loss: {val_loss:.4f}')

# Test
test_acc = test(model, test_loader)
print(f'Test Accuracy: {test_acc * 100:.2f}%')

torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict' : optimizer.state_dict(),
        'train_losses' : train_losses,
        'val_losses' : val_losses
    }, f'model_checkpoint_{epoch}_{lr}.pth')