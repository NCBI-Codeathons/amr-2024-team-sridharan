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
#num_proteins = 12
embedding_dim =480
num_drug_classes = 10

#protein_embeddings = torch.randn(num_proteins, embedding_dim) #(N, 280)
#drug_class_embeddings = torch.randn(num_drug_classes, embedding_dim) #binary matrix, multihot encoding

embedding_pkl = "/shared_venv/all_embeddings.pickle"
with open(embedding_pkl, 'rb') as f:
    protein_embeddings = pkl.load(f)
    protein_embeddings = torch.from_numpy(protein_embeddings)
    print(f"Loaded the protein embeddings of shape : {protein_embeddings.shape} \n")

'''
===============Dummy Data=====================
'''

input_csv = "/shared_venv/filtered_data.csv" #This is the csv file downlaoded from big query
output_tsv = "/shared_venv/hetero_protein_graph.tsv"

create_protein_drug_class_tsv(input_csv, output_tsv)
edge_index = create_edge_index_from_tsv(output_tsv)

#protein_embeddings = torch.randn(edge_index.shape[1], 280) # Creating dummy embeddings detelt this later(N, 280)

drug_class_embeddings,_ = create_eye_matrix_from_tsv(output_tsv)


data = HeteroData()

data['protein'].x = protein_embeddings  # Shape [num_proteins, embedding_dim]

data['drug_class'].x = drug_class_embeddings  # Shape [num_drug_classes, embedding_dim]

data['protein', 'interacts_with', 'drug_class'].edge_index = edge_index  # Shape [2, num_edges]

# Add reverse edges manually by swapping the source and target indices
reverse_edge_index = torch.stack([edge_index[1], edge_index[0]], dim=0)

# Adding the reverse edge type to the HeteroData object
data['drug_class', 'interacted_by', 'protein'].edge_index = reverse_edge_index

'''
pickling the graph data
'''

pickle_path = "/shared_venv/hetero_graph_data.pkl"

with open(pickle_path, 'wb') as f:
    pkl.dump(data, f)

print(f'HeteroData object successfully saved to {pickle_path}')

print(data['protein'].x)
print(data['drug_class'].x)
print(data['protein', 'interacts_with', 'drug_class'].edge_index)
print(data)