# GrAMR (WIP)

List of participants and affiliations:
- Ganeshiny Sridharan, University of Colombo  (Team Leader)
- Janith Weeraman, University of Calgary 
- Nimna Alupotha Gamage, University of Colombo
- Nuwan Medawaththa, University of Colombo
- Ruwanthika Premarathne, University of Colombo

## Project Goals

Idenfity novel antibiotic resistances in organisms using graph representation learning

## Approach (WIP)

![Graphical abstract](misc/Workflow_updated.drawio.png)

## Explanation of the workflow

The 'MicroBIGG-E' database was used to obtain AMR-related information, specifically focusing on the ESKAPE species. For each protein sequence, an embedding model (esm2 35M in this case) is used to produces embeddings representing each sequence in a high-dimensional space. This will also integrate positional information learned through the embedding model's pre-training process. These high-dimensional embeddings are used as node-features in the drug-target graph to inform the future predictions.

The protein sequences are labeled according to different AMR classes. These labels are then label encoded, converting the categorical labels into numerical format for use in machine learning models. The AMR classes are represented as a multihot label vector. A bi-partite graph is constructed where one set of nodes represents proteins and the other set represents drug classes. Edges between protein nodes and drug class nodes represent interactions. 

After the bi-partite graph is constructed, the edge prediction task is handled by several layers. After the graph processing, a classifier is used to assign the AMR class labels to the protein sequences based on the predictions. This model attempts to predict the likelihood that a protein belongs to a particular AMR class. In the final step, new links (edges) are predicted between proteins and drug classes, indicating whether a protein is resistant to a specific drug class. These predicted edges form the basis of understanding the AMR protein-drug interactions.

## Results

## Future Work

- Switching out the current embedding model for a larger and more performant model

## NCBI Codeathon Disclaimer
This software was created as part of an NCBI codeathon, a hackathon-style event focused on rapid innovation. While we encourage you to explore and adapt this code, please be aware that NCBI does not provide ongoing support for it.

For general questions about NCBI software and tools, please visit: [NCBI Contact Page](https://www.ncbi.nlm.nih.gov/home/about/contact/)

