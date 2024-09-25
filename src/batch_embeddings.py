import torch,os, gzip,pickle

from esm_embeddings import generate_embeddings

tempfolder = "temp/fastas/"
embeddingsdir = "temp/embeddings/"
fastafolder = os.listdir(tempfolder)
batch_size = 100

embeddings = []

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t12_35M_UR50D")
model.eval()
model.to(device)

last_embedding = os.listdir(embeddingsdir)
i=0
if len(last_embedding)>0:
    previous = []
    for filename in last_embedding:
        with open(os.path.join(embeddingsdir,filename),'rb') as handle:
            data = pickle.load(handle)
        previous += [i[0] for i in data]

while i<len(fastafolder):
    file = fastafolder[i]

    if file in previous:
        i += 1
        continue

    # Create a BytesIO buffer for the gzip file and extract the content
    with gzip.open(os.path.join(tempfolder,file),'rt') as handle:
        data = ''.join(handle.read().split('\n')[1:])
    i += 1    

    try:
        embeddings += generate_embeddings([(file,data)],model,alphabet)
    except KeyError:
        print(f"protein {file} had an unrecognized AA")

    if i%(batch_size*100)==0:
        with open(f"temp/embeddings/{i}.pickle", 'wb') as handle:
            pickle.dump(embeddings,handle)
        embeddings = []
    

