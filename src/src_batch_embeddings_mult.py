"""
This script loads in one of the Facebook ESM models to genereate sequence embeddings. This is done for each gzipped fasta file in a given fasta folder. 

It can also check to see if there are any alrady generated embedding pickles in the embedding folder and then skip the relevant fasta files in those embeddings

"""

import torch,os, gzip,pickle
from esm_embeddings import generate_embeddings

tempfolder = "temp/fastas/"
embeddingsdir = "temp/embeddings/"
fastafolder = os.listdir(tempfolder)
batch_size = 5

embeddings = []

# Load in the embedding model 
# (Change this to something larger if you have the VRAM to spare)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t12_35M_UR50D")
model.eval()
model.to(device)

# Here we build a list of all the previously processed files from the pickles in embeddingdir
last_embedding = os.listdir(embeddingsdir)
i=len(fastafolder)-1
if len(last_embedding)>0:
    previous = []
    for filename in last_embedding:
        with open(os.path.join(embeddingsdir,filename),'rb') as handle:
            data = pickle.load(handle)
        previous += [i[0] for i in data]
else:
    previous=[]

# main loop
print(f"found {len(previous)} embeddings generated. proceesing...")
while i<len(fastafolder):
    file = fastafolder[i]

    if file in previous:
        i -= 1
        continue

    # Create a BytesIO buffer for the gzip file and extract the content
    with gzip.open(os.path.join(tempfolder,file),'rt') as handle:
        data = ''.join(handle.read().split('\n')[1:])
    i -= 1    

    try:
        temp = generate_embeddings([(file,data)],model,alphabet)
        embeddings.append( temp )
        print(temp)
    except KeyError:
        print(f"protein {file} had an unrecognized AA")
    # Some proteins have amino acids that the model doesn't recognize. In this case we warn the user and skip that protein
    

    # Every 100 batches we dump the currently generated embeddings
    # This is more to prevent the script from eating all your RAM than anything else tbh
    if i%(batch_size*100)==0:
        with open(f"temp/embeddings/{i}.pickle", 'wb') as handle:
            pickle.dump(embeddings,handle)
        embeddings = []
    

