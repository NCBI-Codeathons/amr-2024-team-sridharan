from torch_geometric.data import HeteroData
import torch 

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
drug_class_embeddings = torch.randn(num_drug_classes, embedding_dim) #binary matrix
edge_index = torch.tensor([[0, 1, 2, 3],  # Protein indices
                           [0, 2, 1, 3]])

'''
===============Dummy Data=====================
'''

data = HeteroData()

data['protein'].x = protein_embeddings  # Shape [num_proteins, embedding_dim]

data['drug_class'].x = drug_class_embeddings  # Shape [num_drug_classes, embedding_dim]

data['protein', 'interacts_with', 'drug_class'].edge_index = edge_index  # Shape [2, num_edges]


print(data['protein'].x)
print(data['drug_class'].x)
print(data['protein', 'interacts_with', 'drug_class'].edge_index)
print(data)