import torch,os, gzip,pickle

from esm_embeddings import generate_embeddings

tempfolder = "temp/fastas/"
embeddingsdir = "temp/embeddings/"
fastafolder = os.listdir(tempfolder)
batch_size = 100

sequences = []
embeddings = []

last_embedding = os.listdir(embeddingsdir)
if len(last_embedding)==0:
    i=0
else:
    i = int(last_embedding[-1].split('.')[0])

while i<len(fastafolder):
    file = fastafolder[i]

    # Create a BytesIO buffer for the gzip file and extract the content
    with gzip.open(os.path.join(tempfolder,file),'rt') as handle:
        data = ''.join(handle.read().split('\n')[1:])
    sequences.append((file,data))

    i += 1    

    if i%batch_size==0:
        embeddings += generate_embeddings(sequences)
        sequences = []

    if i%(batch_size*1000)==0:
        with open(f"temp/fastas/embeddings/{i}.pickle", 'wb') as handle:
            pickle.dump(embeddings,handle)
        embeddings = []
    

