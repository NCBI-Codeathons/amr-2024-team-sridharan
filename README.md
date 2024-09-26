# GrAMR 

List of participants and affiliations:
- Ganeshiny Sridharan, University of Colombo  **(Team Lead)**
- Janith Weeraman, University of Calgary **(Tech Lead)**
- Nimna Alupotha Gamage, University of Colombo
- Nuwan Medawaththa, University of Colombo
- Ruwanthika Premarathne, University of Colombo

To define the biological problem in a clear and focused way, the background section should explain why understanding **antimicrobial resistance (AMR)** is critical and how the research problem relates to this larger challenge. Here's a suggestion for the **Background** section:

---

## Background

**Antimicrobial resistance (AMR)** is one of the most significant global health threats. It occurs when microorganisms like bacteria, viruses, fungi, and parasites evolve to resist the effects of antimicrobial drugs, rendering these treatments ineffective. As a result, infections that were once easily treatable can become life-threatening, leading to higher medical costs, longer hospital stays, and increased mortality.

The rise of AMR is particularly concerning in **ESKAPE pathogens** (Enterococcus faecium, Staphylococcus aureus, Klebsiella pneumoniae, Acinetobacter baumannii, Pseudomonas aeruginosa, and Enterobacter species). These pathogens are responsible for a large proportion of hospital-acquired infections and are known for their ability to "escape" the effects of antimicrobial drugs, contributing to the growing problem of multidrug resistance.

A key challenge in combating AMR is identifying **novel resistance mechanisms**. Many proteins that confer resistance have not yet been characterized, and mutations in bacterial genomes can lead to new forms of resistance that are not well understood. Traditional laboratory methods to detect and study these resistance mechanisms are time-consuming and often lag behind the emergence of new resistances.

### The Role of Graph Representation Learning in AMR Research

**Graph representation learning** provides a powerful computational approach to address this challenge. By representing proteins and their interactions with drugs as graphs, we can model complex relationships between these entities. In particular, **Graph Neural Networks (GNNs)** can learn from the structure of the protein-drug interaction networks to predict unknown or emerging AMR mechanisms. 

In this project, **GrAMR** (Graph-based Antimicrobial Resistance) aims to leverage these techniques to identify novel AMR-related resistances. By constructing a graph where nodes represent proteins and drug classes, and edges represent known AMR interactions, we can train a model to predict whether uncharacterized proteins or mutations are likely to exhibit resistance. This has the potential to provide early warnings of new resistances, guiding future drug development and helping prioritize laboratory validation efforts.

## Project Goals

This project aims to idenfity novel antibiotic resistances in ESKAPE organisms using graph representation learning

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
## Project outcomes

The successful implementation  of GrAMR would allow for users to determine possible AMR properties of novel/unkown proteins where only sequence data is available. Furthermore, with relatively minor tweaking it wouldbe possible to determine the effects of certain protein sequence mutations on the likelihood of creating AMR effects in a protein by analyzing model weights and activations. 

## Results


## Future Work

- Incorporating a larger subset of the proteins 
- Switching out the current embedding model for a larger variant of the ESM-2 model family
- Fine-tuning the GNN architecture

## NCBI Codeathon Disclaimer
This software was created as part of an NCBI codeathon, a hackathon-style event focused on rapid innovation. While we encourage you to explore and adapt this code, please be aware that NCBI does not provide ongoing support for it.

For general questions about NCBI software and tools, please visit: [NCBI Contact Page](https://www.ncbi.nlm.nih.gov/home/about/contact/)

