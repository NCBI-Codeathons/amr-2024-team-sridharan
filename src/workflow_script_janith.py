"""
Script to batch download fasta files from google cloud storage
The arguments specify which indices in the data file the script should run for

This script is meant to be run in parallel to allow for faster downloads

"""

import argparse, os
import polars as pl
from google.cloud import storage
from esm_embeddings import generate_embeddings

# Parsing command line arguments
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

# Check to see if the given index is larger than the data file and 
# if it is only going up to the end of the file
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

