# **DTI-PREDICTION**  
## **Protein-Ligand Interaction Prediction Models**  

This project explores **drug-target interaction (DTI) prediction** using deep learning. We utilize a pipeline inspired by [this repository](https://github.com/Fitnessnlp/DeepEmbedding-DTI/tree/master/dataset), which combines **3-gram embeddings + BiLSTM + attention mechanism** for protein embeddings and **graph neural networks (GNNs)** for ligand embeddings.  

---

## **Overview of Experiments**  

We extend the baseline model (`models/exp1.py`) with two additional variations:  

1. **Baseline Model (`exp1.py`)**  
   - Uses **3-gram embeddings** for proteins.  
   - Employs **BiLSTM + attention** for sequence representation.  
   - GNN for ligand embeddings.  

2. **Transformer-based Model (`exp2.py`)**  
   - Converts **3-gram indices back into protein sequences**.  
   - Feeds sequences into **ProtBERT** ([Hugging Face’s `prot_bert`](https://huggingface.co/Rostlab/prot_bert/tree/main)) for embedding.  
   - **Limitation**: ProtBERT has a fixed **512-token limit**, leading to **truncation** of longer sequences.  

3. **Chunking Model (`exp3.py`)**  
   - Addresses truncation issues by **splitting longer sequences into 512-token chunks**.  
   - Processes these chunks sequentially with ProtBERT.  
   - Aggregates chunked representations for a more complete protein embedding.  

---

## **Dataset**  

We use a **subset of the DUD-E dataset** (`dataset/cleaned_dataset.txt`). This dataset contains protein sequences of varying lengths, with some exceeding the **512-token limit** of ProtBERT.  

---

## **Results**  

- Training, validation, and test metrics for all models are logged in the `results/` directory.  
- To analyze dataset characteristics, use `EDA_DATASETSPLIT.ipynb`.  

---

## **Dependencies**  

All required Python libraries are listed in `prerequisites.txt`. Install them using:  

```bash
pip install -r prerequisites.txt


## Flow
To recreate the experiments these steps have to be followed:
1. Run preprocess_data.py with input the cleaned_dataset.txt this script preprocesses the initial dataset and outputs 7 files
   proteins.pkl
   compounds.pkl
   adjacencies.pkl
   interactions.pkl
   SMILES.txt
   fingerpint_dict.pickle
   word_dict.pickle
2. Then run the jupyter script EDA_DATASETSPLIT.ipynb which outputs some characteristics of our dataset and also creates the preprocessed datasets by combining  ( compounds.pkl,adjacencies.pkl,proteins.pkl, interactions.pkl) and splitting this into Training,Validation and Test set.
3. Now we can run any model from our models directory were exp1.py is the basis 3-gram model , exp2.py is prot_bert with truncation and exp3.py is our model with prot_bert+chunking. InputS are the datasets created from step 2 so train_set.ex3.pt, dev_set_ex3.pt, test_set_ex3.pt alongside with fingerpint_dict.pickle and word_dict_pickle created at step 1.

## Other scripts
1.Running script analyse_dataset_length.py we can visualize the proteins length distribution across the datasets
2.Running script flowchart_reconstruction.py we can create a flowchart that explains the reconstruction algorithm that takes 3-gram indices and turn them back to the full amino acid sequence
3.Running visual_results.py we can visualize the training metrics that are being logged when we train our models.

