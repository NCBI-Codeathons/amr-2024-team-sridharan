from torch_geometric.data import HeteroData
import torch
from processing_csv import create_protein_drug_class_tsv, create_edge_index_from_tsv, create_eye_matrix_from_tsv

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
embedding_dim =1028
num_drug_classes = 10

protein_embeddings = torch.randn(num_proteins, embedding_dim)
#drug_class_embeddings = torch.randn(num_drug_classes, embedding_dim) #binary matrix, multihot encoding

'''
===============Dummy Data=====================
'''


input_csv = ""
output_tsv = ""

create_protein_drug_class_tsv(input_csv, output_tsv)
edge_index = create_edge_index_from_tsv(output_tsv)
drug_class_embeddings,_ = create_eye_matrix_from_tsv(output_tsv)


data = HeteroData()

data['protein'].x = protein_embeddings  # Shape [num_proteins, embedding_dim]

data['class'].x = drug_class_embeddings  # Shape [num_drug_classes, embedding_dim]

data['protein', 'interacts_with', 'class'].edge_index = edge_index  # Shape [2, num_edges]


print(data['protein'].x)
print(data['class'].x)
print(data['protein', 'interacts_with', 'class'].edge_index)
print(data)