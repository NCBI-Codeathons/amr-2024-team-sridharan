"""
This file contains functions to generate sequence embeddings from a Facebook/ESM-2 model 

"""

import torch, json
import polars as pl

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def generate_embeddings(
        sequences, # list of tuples of the form ('header','sequence')
        model,
        alphabet,
        embedding_layer=12, # The final layer of the embedding model
        device=device, # will automatically use CUDA if its available else will fall back on cpu
):

    batch_converter = alphabet.get_batch_converter()
    # Generating data batches
    batch_labels, batch_strs, batch_tokens = batch_converter(sequences)
    with torch.no_grad():
        batch_tokens = batch_tokens.to(device)
        results = model(batch_tokens, repr_layers=[embedding_layer], return_contacts=True)
    token_representations = results["representations"][embedding_layer]

    # Generating sequence representating via averaging token representations
    sequence_representations = []
    for i, (_, seq) in enumerate(sequences):
        sequence_representations.append(token_representations[i, 1 : len(seq) + 1].mean(0))
    
    del batch_tokens, token_representations
    torch.cuda.empty_cache()
    return sequence_representations # Embedding dim is 1280


if __name__=="__main__":
    data = [
    ("protein1", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"),
    ("protein2", "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"),
    ("protein1", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"),
    ("protein2", "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"),
    ("protein1", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"),
    ("protein2", "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"),
    ("protein1", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"),
    ("protein2", "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"),
    ("protein1", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"),
    ("protein2", "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"),
    ("protein1", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"),
    ("protein2", "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"),
    ("protein1", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"),
    ("protein2", "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"),
    ] # lots of duplicates to test the code
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t12_35M_UR50D")
    model.eval()
    model.to(device)
    print(generate_embeddings(data,model, alphabet,12,'cuda')) 