import torch,os, gzip,pickle

from esm_embeddings import generate_embeddings

tempfolder = "temp/fastas/"
embeddingsdir = "temp/embeddings/"
fastafolder = os.listdir(tempfolder)
batch_size = 100

embeddings = []

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")
model.eval()
model.to(device)

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
    i += 1    

    embeddings += generate_embeddings([(file,data)],model,alphabet)

    if i%(batch_size*100)==0:
        with open(f"temp/embeddings/{file}.pickle", 'wb') as handle:
            pickle.dump(embeddings,handle)
        embeddings = []
    

