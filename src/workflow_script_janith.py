import json
import polars as pl
from esm_embeddings import generate_embeddings, protein_url2fasta_json
from gs_to_dict import parse_fasta_from_gcs

dataset_path = "data_from_bigquery.csv"

fastadict = protein_url2fasta_json(dataset_path, 'fastas.json')

embeddings = [proteindict.items() for key,proteindict in fastadict.items()]

embeddings = dict(generate_embeddings(embeddings))

with open('embeddings.json','w') as file:
    json.dump(embeddings,file)