import torch,os, gzip,pickle

from esm_embeddings import generate_embeddings

tempfolder = "temp/fastas/"
fastafolder = os.listdir(tempfolder)
batch_size = 100

sequences = []
embeddings = []

for i,file in enumerate(fastafolder):
    # Create a BytesIO buffer for the gzip file and extract the content
    with gzip.open(os.path.join(tempfolder,file),'rt') as handle:
        data = ''.join(handle.read().split('\n')[1:])
    sequences.append((file,data))
    
    if i%batch_size==0:
        embeddings += generate_embeddings(sequences)
        sequences = []
    
    if i%(batch_size*1000)==0:
        with open(f"temp/fastas/embeddings/{i},pickle", 'wb') as handle:
            pickle.dump(embeddings,handle)
        embeddings = []
