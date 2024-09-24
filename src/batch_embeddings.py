import torch,os, gzip

from esm_embeddings import generate_embeddings

tempfolder = "temp/fastas/"
fastafolder = os.listdir(tempfolder)

for file in fastafolder:
    # Create a BytesIO buffer for the gzip file and extract the content
    with gzip.open(os.path.join(tempfolder,file),'rt') as handle:
        data = handle.read()
    
    print(data); quit()
