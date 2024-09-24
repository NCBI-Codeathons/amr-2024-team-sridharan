import argparse, os
import polars as pl
from google.cloud import storage
from esm_embeddings import generate_embeddings
from gs_to_dict import parse_fasta_from_gcs

parser = argparse.ArgumentParser(description='Get start and end indices')
parser.add_argument('-s', '--start', type=int, default=0, help='Start index (default: 0)')
parser.add_argument('-e', '--end', type=int, default=1000, help='End index (default: 1000)')
args = parser.parse_args()

startindex = int(args.start)
endindex = int(args.end)

dataset_path = "/shared_venv/data_from_bigquery.csv"
data = pl.read_csv(dataset_path, separator=',')
# Initialize Google Cloud Storage client
client = storage.Client()

if endindex > data.shape[0]: endindex = data.shape[0]
data = data[startindex:endindex][['protein_acc','protein_url']].to_numpy()

output_path=f'temp/fastas'

for i,(acc,url) in enumerate(data):
    output_path_item = f"{output_path}/{acc}.fa.gz"

    if os.path.exists(output_path_item): continue

    bucket_name, *file_path = url[5:].split("/")
    file_path = "/".join(file_path)
    # Get the bucket and the blob (file) from GCS
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_path)

    # Download the file content as bytes
    file_data = blob.download_as_bytes()

    # save downloaded file
    with open(output_path_item,'wb') as file:
        file.write(file_data)

