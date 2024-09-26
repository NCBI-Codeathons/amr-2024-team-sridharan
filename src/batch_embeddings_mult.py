import torch, os, gzip, pickle
from esm_embeddings import generate_embeddings
from concurrent.futures import ThreadPoolExecutor, as_completed

tempfolder = "temp/fastas/"
embeddingsdir = "temp/embeddings/"
fastafolder = os.listdir(tempfolder)
batch_size = 5
max_workers = 5  # Number of concurrent workers

embeddings = []

# Load in the embedding model
# (Change this to something larger if you have the VRAM to spare)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t12_35M_UR50D")
model.eval()
model.to(device)

# Here we build a list of all the previously processed files from the pickles in embeddingdir
last_embedding = os.listdir(embeddingsdir)
previous = []
if len(last_embedding) > 0:
    for filename in last_embedding:
        with open(os.path.join(embeddingsdir, filename), 'rb') as handle:
            data = pickle.load(handle)
        previous += [i[0] for i in data]

# Function to process individual fasta file
def process_fasta(file):
    if file in previous:
        return None  # Skip if already processed

    # Create a BytesIO buffer for the gzip file and extract the content
    with gzip.open(os.path.join(tempfolder, file), 'rt') as handle:
        data = ''.join(handle.read().split('\n')[1:])
    
    try:
        temp = generate_embeddings([(file, data)], model, alphabet)
        torch.cuda.empty_cache()
        return temp
    except KeyError:
        print(f"protein {file} had an unrecognized AA")
        return None

# Main loop to process files concurrently
print(f"found {len(previous)} embeddings generated. processing...")

with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = {executor.submit(process_fasta, file): file for file in fastafolder}
    processed = 0
    
    for future in as_completed(futures):
        result = future.result()
        if result:
            embeddings.append(result)
        processed += 1
        
        # Every 100 batches we dump the currently generated embeddings
        if processed % (batch_size * 20) == 0:  # Adjusted to match batch size
            with open(f"temp/embeddings/{processed}.pickle", 'wb') as handle:
                pickle.dump(embeddings, handle)
            embeddings = []

# Save any remaining embeddings
if embeddings:
    with open(f"temp/embeddings/final_{processed}.pickle", 'wb') as handle:
        pickle.dump(embeddings, handle)
