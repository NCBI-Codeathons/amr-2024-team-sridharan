# GrAMR (WIP)

List of participants and affiliations:
- Ganeshiny Sridharan, University of Colombo  (Team Leader)
- Janith Weeraman, University of Calgary 
- Nimna Alupotha Gamage, University of Colombo
- Nuwan Medawaththa, University of Colombo

## Project Goals

Idenfity novel antibiotic resistances in organisms using graph representation learning (WIP)

## Approach (WIP)

![Graphical abstract](misc/Workflow.png)

## Explanation of the workflow
The step-by-step explanation of the workflow is as follows;

- Data preprocessing

The 'MicroBIGG-E' database was used to obtain AMR-related information, specifically focusing on the ESKAPE species. For each protein sequence, a FASTA-like structure is used, consisting of amino acids. These sequences serve as input for further steps.

- Embedding Generation

This step likely uses a pre-trained protein language model (ESM). The Base ESM takes the protein sequences and produces embeddings representing each sequence in a high-dimensional space. The embeddings generated from the ESM model are further fine-tuned using ESM2, creating feature representations with dimensions [N, 1024], where N is the number of sequences and 1024 is the size of the embedding. This embedding captures essential information about protein sequences for AMR prediction.

- AMR Class Labeling and Encoding

The protein sequences are labeled according to different AMR classes. These labels are then label encoded, converting the categorical labels into numerical format for use in machine learning models. The AMR classes are represented as a multihot label vector, which is a binary vector where each entry corresponds to a specific AMR class. 

## Results

## Future Work

## NCBI Codeathon Disclaimer
This software was created as part of an NCBI codeathon, a hackathon-style event focused on rapid innovation. While we encourage you to explore and adapt this code, please be aware that NCBI does not provide ongoing support for it.

For general questions about NCBI software and tools, please visit: [NCBI Contact Page](https://www.ncbi.nlm.nih.gov/home/about/contact/)

