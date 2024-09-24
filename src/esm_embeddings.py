"""
This file contains functions to generate sequence embeddings from a Facebook/ESM-2 model 

"""

import torch, json
import polars as pl
from gs_to_dict import parse_fasta_from_gcs

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def generate_embeddings(
        sequences, # list of tuples of the form ('header','sequence')
        device=device, # will automatically use CUDA if its available else will fall back on cpu
):

    model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")
    batch_converter = alphabet.get_batch_converter()
    model.eval()
    model.to(device)

    # Generating data batches
    batch_labels, batch_strs, batch_tokens = batch_converter(sequences)
    with torch.no_grad():
        batch_tokens = batch_tokens.to(device)
        results = model(batch_tokens, repr_layers=[33], return_contacts=True)
    token_representations = results["representations"][33]

    # Generating sequence representating via averaging token representations
    sequence_representations = []
    for i, (_, seq) in enumerate(sequences):
        sequence_representations.append(token_representations[i, 1 : len(seq) + 1].mean(0))
    
    return [(batch_labels[i],rep) for i,rep in enumerate(sequence_representations)] # Embedding dim is 1280


if __name__=="__main__":
    data = [
    ("protein1", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"),
    ("protein2", "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"),
    ]

    print(generate_embeddings(data,device)) 