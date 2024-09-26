"""
This script loads in one of the Facebook ESM models to genereate sequence embeddings. This is done for each gzipped fasta file in a given fasta folder. 

It can also check to see if there are any alrady generated embedding pickles in the embedding folder and then skip the relevant fasta files in those embeddings

"""

import torch,os, gzip,pickle, argparse
from esm_embeddings import generate_embeddings

tempfolder = "temp/fastas/"
embeddingsdir = "temp/embeddings/"
fastafolder = os.listdir(tempfolder)
batch_size = 5

embeddings = []
# Parsing command line arguments
parser = argparse.ArgumentParser(description='Get start and end indices')
parser.add_argument('-s', '--start', type=int, default=0, help='Start index (default: 0)')
parser.add_argument('-e', '--end', type=int, default=1000, help='End index (default: 1000)')
args = parser.parse_args()

startindex = int(args.start)
endindex = int(args.end) if int(args.end) < len(fastafolder) else len(fastafolder)

# Load in the embedding model 
# (Change this to something larger if you have the VRAM to spare)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t12_35M_UR50D")
model.eval()
model.to(device)



# main loop
for i in range(startindex,endindex):
    file = fastafolder[i]

    # Create a BytesIO buffer for the gzip file and extract the content
    with gzip.open(os.path.join(tempfolder,file),'rt') as handle:
        data = ''.join(handle.read().split('\n')[1:])
    i += 1    

    try:
        temp = generate_embeddings([(file,data)],model,alphabet)
        embeddings.append( temp )

    except KeyError:
        print(f"protein {file} had an unrecognized AA")
    # Some proteins have amino acids that the model doesn't recognize. In this case we warn the user and skip that protein
    

    # Every 100 batches we dump the currently generated embeddings
    # This is more to prevent the script from eating all your RAM than anything else tbh
    if i%(batch_size*100)==0:
        with open(f"temp/embeddings/{i}.pickle", 'wb') as handle:
            pickle.dump(embeddings,handle)
        embeddings = []

with open(f"temp/embeddings/{i}.pickle", 'wb') as handle:
    pickle.dump(embeddings,handle)
    
