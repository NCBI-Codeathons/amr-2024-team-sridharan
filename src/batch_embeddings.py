"""
This script loads in one of the Facebook ESM models to genereate sequence embeddings. This is done for each gzipped fasta file in a given fasta folder. 

It can also check to see if there are any alrady generated embedding pickles in the embedding folder and then skip the relevant fasta files in those embeddings

"""

import torch,os, gzip,pickle, argparse

batch_size = 5
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Parsing command line arguments
parser = argparse.ArgumentParser(description='Get arguments')
parser.add_argument('-s', '--start', type=int, default=0, help='Start index (default: 0)')
parser.add_argument('-e', '--end', type=int, default=1000, help='End index (default: 1000)')
parser.add_argument('-f', '--fastafol', type=str, default="temp/fastas/", help='Folder of fasta files to generate embeddings for. Expect them to be fa.gz format')
parser.add_argument('-o', '--outputdir', type=str, default="temp/embeddings/", help='Output directory for embeddings')

args = parser.parse_args()

tempfolder = args.fastafol
embeddingsdir = args.outputdir
fastafolder = os.listdir(tempfolder)
startindex = int(args.start)
endindex = int(args.end) if int(args.end) < len(fastafolder) else len(fastafolder)

# Load in the embedding model 
# (Change this to something larger if you have the VRAM to spare)
model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t12_35M_UR50D")
model.eval()
model.to(device)

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
    token_representations = results["representations"][embedding_layer].cpu().detach()

    # Generating sequence representating via averaging token representations
    sequence_representations = []
    for i, (_, seq) in enumerate(sequences):
        sequence_representations.append((batch_labels[i],token_representations[i, 1 : len(seq) + 1].mean(0)))
    
    del batch_tokens, token_representations, results
    torch.cuda.empty_cache()
    return sequence_representations # Embedding dim is 420


embeddings = []
# main loop
for i in range(startindex,endindex):
    file = fastafolder[i]

    # Create a BytesIO buffer for the gzip file and extract the content
    with gzip.open(os.path.join(tempfolder,file),'rt') as handle:
        data = ''.join(handle.read().split('\n')[1:])

    try:
        temp = generate_embeddings([(file,data)],model,alphabet)
        embeddings.append( temp )

    # Some proteins have amino acids that the model doesn't recognize. In this case we warn the user and skip that protein
    except KeyError:
        print(f"protein {file} had an unrecognized AA")

    # Every 100 batches we dump the currently generated embeddings
    # This is more to prevent the script from eating all your RAM than anything else tbh
    if i%(batch_size*100)==0:
        with open(f"temp/embeddings/{i}.pickle", 'wb') as handle:
            pickle.dump(embeddings,handle)
        embeddings = []

with open(f"temp/embeddings/{i}.pickle", 'wb') as handle:
    pickle.dump(embeddings,handle)
    
