import torch,os, gzip

from esm_embeddings import generate_embeddings

fastafolder = os.listdir("temp/fastas/")

for file in fastafolder:
    # Create a BytesIO buffer for the gzip file and extract the content
    with gzip.open(os.path.join(fastafolder,file),'rt') as handle:
        data = handle.read()
    
    print(data); quit()
