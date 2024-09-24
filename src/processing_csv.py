import pandas as pd
import torch


def create_protein_drug_class_tsv(input_csv, output_tsv):
    '''
    
    Input : the CSV file obtained from BigQuery, rows interested : protein_accession, class

    Output: A TSV file that has the ordered list of distinct drug classes in the first row and then each protein and corresponding drug classes it's resistant to
    '''
    df = pd.read_csv(input_csv)
    
    # To ensure that the CSV has 'protein_acc' and 'drug_class' columns
    if 'protein_acc' not in df.columns or 'class' not in df.columns: # Must change the col names here !!!!
        raise ValueError("Input CSV must contain 'protein_acc' and 'drug_class' columns.")
    
    # Sort all the drug classes
    drug_classes = sorted(df['class'].unique())

    protein_drug_map = {}

    for _, row in df.iterrows():
        protein_acc = row['protein_acc']
        drug_class = row['class']     
        if protein_acc not in protein_drug_map:
            protein_drug_map[protein_acc] = {drug: 0 for drug in drug_classes}        
        protein_drug_map[protein_acc][drug_class] = 1

    # Writing
    with open(output_tsv, 'w') as f_out:
        # Write the header (drug classes, tab-separated)
        f_out.write('\t'.join(['protein_acc'] + drug_classes) + '\n')
        
        # Write each protein and labels
        for protein_acc, drug_map in protein_drug_map.items():
            # Create the row: protein accession followed by 0/1 for each drug class
            row = [protein_acc] + [str(drug_map[drug]) for drug in drug_classes]
            f_out.write('\t'.join(row) + '\n')


def create_edge_index_from_tsv(tsv_file):
    '''
    Input: The TSV file generated from the function "create_protein_drug_class_tsv"

    Output: 
    1. The edge indice matrix showing the edged between the proteins and the drug classes. Shape: (2, num_edges)
    2. Eye matrix for the labels, each row is a drug class with "1" at the position corresponding to the drug class. Shape: (num_drug_classes, num_drug_classes)
    
    '''

    df = pd.read_csv(tsv_file, sep='\t')
    
    # Get the list of drug classes (from the header)
    drug_classes = list(df.columns[1:])  # Make sure that the first column is 'protein_acc', rest are drug classes!!
    
    # Create lists to store the protein-to-drug class edges
    protein_indices = []
    drug_class_indices = []
    
    # Iterate over each protein (row in the DataFrame)
    for protein_idx, row in df.iterrows():
        # Loop through each drug class (column in the row)
        for drug_idx, drug_class in enumerate(drug_classes):
            if row[drug_class] == 1:  # If the relationship exists (binary 1)
                protein_indices.append(protein_idx)  # Source node: Protein index
                drug_class_indices.append(drug_idx)  # Target node: Drug class index

    # Convert to PyTorch tensor: Edge index with shape (2, num_edges)
    edge_index = torch.tensor([protein_indices, drug_class_indices], dtype=torch.long)
    
    return edge_index

def create_eye_matrix_from_tsv(tsv_file):
    '''
    This is to create the eye matrix and the row order is used to refer the edges between proteins and drug classes. 
    This is a eye vector of the drug classes.

    Input: TSV file generated from the fucntion "create_protein_drug_class_tsv"

    Ouput : Eye matrix and the drug classes
    
    '''
    df = pd.read_csv(tsv_file, sep='\t')
    
    drug_classes = list(df.columns[1:])  # First column is 'protein_acc', the rest are drug classes

    # Number of distinct drug classes
    num_classes = len(drug_classes)

    # Create an identity matrix of size (num_classes, num_classes)
    eye_matrix = torch.eye(num_classes, dtype=torch.float32)
    
    return eye_matrix, drug_classes



