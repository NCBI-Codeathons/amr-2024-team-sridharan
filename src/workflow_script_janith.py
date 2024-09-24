import json, argparse
import polars as pl
from esm_embeddings import generate_embeddings, protein_url2fasta_json
from gs_to_dict import parse_fasta_from_gcs

parser = argparse.ArgumentParser(description='Get start and end indices')
parser.add_argument('-s', '--start', type=int, default=0, help='Start index (default: 0)')
parser.add_argument('-e', '--end', type=int, default=1000, help='End index (default: 1000)')
args = parser.parse_args()

startindex = int(args.start)
endindex = int(args.end)

dataset_path = "/shared_venv/data_from_bigquery.csv"
data = pl.read_csv(dataset_path, separator=',')
if endindex > data.shape[0]: endindex = data.shape[0]

fastadict = protein_url2fasta_json(data, 'temp/fastas/indexes',startindex=startindex, endindex=endindex)

embeddings = [proteindict.items() for key,proteindict in fastadict.items()]

embeddings = dict(generate_embeddings(embeddings))

with open(f'temp/embeddings/indexes_{startindex}-{endindex}.json','w') as file:
    json.dump(embeddings,file)
