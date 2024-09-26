# GrAMR 

List of participants and affiliations:
- Ganeshiny Sridharan, University of Colombo  (Team Leader)
- Janith Weeraman, University of Calgary (Tech Lead)
- Nimna Alupotha Gamage, University of Colombo
- Nuwan Medawaththa, University of Colombo
- Ruwanthika Premarathne, University of Colombo

## Project Goals

Idenfity novel antibiotic resistances in organisms using graph representation learning

## Workflow

![Graphical abstract](misc/Workflow_updated.drawio.png)

## Approach

The study focused on identifying AMR-related novel resistances in ESKAPE species. Initially the MicroBIGG-E database was queried with the above filters to get the protein ids and AMR class for each of the proteins. The protein sequences were then used to generate 420 dimensional sequence embeddings using the ESM-2 (35M) protein embedding model. 

We then construct a bi-partite graph is constructed based on the drug-protein interactions in the database, where the nodes in the graph are of type 'protein' or 'drug class' and edges indicate the presence of AMR activity. Then, the sequence embeddings constructed in the previous step are added to the graph as node features, and one-hot encoded class labels are used as node features for the drug nodes. 

After constructing the sequence-informed drug-target graph, it is used as an input to a Graph Neural Network, which will be trained to predict novel AMR effects. This training will be carried out via a link prediction objective. 

## Required libraries
```
pytorch
pytorch-geometric # with optional dependancies 
polars
networkx
transformers
```

**Optional: only used when pulling data from google cloud**
```
google-cloud-bigquery
google-cloud-storage
```

## Results


## Future Work

- Incorporating a larger subset of the proteins 
- Switching out the current embedding model for a larger variant of the ESM-2 model family
- Fine-tuning the GNN architecture

## NCBI Codeathon Disclaimer
This software was created as part of an NCBI codeathon, a hackathon-style event focused on rapid innovation. While we encourage you to explore and adapt this code, please be aware that NCBI does not provide ongoing support for it.

For general questions about NCBI software and tools, please visit: [NCBI Contact Page](https://www.ncbi.nlm.nih.gov/home/about/contact/)

