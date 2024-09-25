from torch_geometric.data import HeteroData
import torch
from processing_csv import create_protein_drug_class_tsv, create_edge_index_from_tsv, create_eye_matrix_from_tsv
import pickle as pkl
'''
Inputs:

Protein embeddings : Shape [num_proteins, embedding_dim]
Drug class embeddings : Shape [num_drug_classes, embedding_dim]
Edge Indices: [2, num_edges] #This we have to create a function that makes COO matrices, takes the 

'''


'''
===============Dummy Data=====================
'''
num_proteins = 12
embedding_dim =480
num_drug_classes = 10

#protein_embeddings = torch.randn(num_proteins, embedding_dim)
#drug_class_embeddings = torch.randn(num_drug_classes, embedding_dim) #binary matrix, multihot encoding

embedding_pkl = "/shared_venv/embeddings.pkl"
with open(embedding_pkl, 'rb') as f:
    protein_embeddings = pkl.load(f)
    print(protein_embeddings.shape)

'''
===============Dummy Data=====================
'''


input_csv = "/shared_venv/data_from_bigquery.csv"
output_tsv = "hetero_protein_graph.tsv"


create_protein_drug_class_tsv(input_csv, output_tsv)
edge_index = create_edge_index_from_tsv(output_tsv)
drug_class_embeddings,_ = create_eye_matrix_from_tsv(output_tsv)


data = HeteroData()

data['protein'].x = protein_embeddings  # Shape [num_proteins, embedding_dim]

data['class'].x = drug_class_embeddings  # Shape [num_drug_classes, embedding_dim]

data['protein', 'interacts_with', 'class'].edge_index = edge_index  # Shape [2, num_edges]


'''
pickling the graph data
'''

pickle_path = "hetero_graph_data.pkl"

with open(pickle_path, 'wb') as f:
    pkl.dump(data, f)

print(f'HeteroData object successfully saved to {pickle_path}')

print(data['protein'].x)
print(data['class'].x)
print(data['protein', 'interacts_with', 'class'].edge_index)
print(data)