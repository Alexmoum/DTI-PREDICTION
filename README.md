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
pip install -r prerequisites.txt

---

## **Experimental Workflow**
To reproduce the experiments, follow these steps: 
### **1. Preprocessing the Data**
Run the following script: preprocess_data.py dataset/cleaned_dataset.txt 
This script preprocesses the  dataset and generates the following files:
- proteins.pkl
- compounds.pkl
- adjacencies.pkl
- interactions.pkl
- SMILES.txt
- fingerpint_dict.pickle
- word_dict.pickle
  
### **2. Exploratory Data Analysis & Dataset Splitting**
Run  **EDA_DATASETSPLIT.ipynb**  to:
- Analyze dataset characteristics
- Split processed data into training, validation, and test sets

### **3. Running Models**
Choose one of the models from the models/ directory and run it with:
- python models/exp1.py  # 3-gram model  
- python models/exp2.py  # ProtBERT with truncation  
- python models/exp3.py  # ProtBERT with chunking
inputs:
- **Training set: train_set_ex3.pt**
- **Validation set: dev_set_ex3.pt**
- **Test set: test_set_ex3.pt**
- **Dictionary files: fingerprint_dict.pickle, word_dict.pickle**


---

## Additional Scripts
1. **analyse_dataset_length.py** &#8594; Visualize the protein sequence length distribution.
2. **flowchart_reconstruction.py** &#8594; Generates a flowchart explaining the reconstruction of 3-gram indices into full amino acid sequences.
3. **visual_results.py** &#8594; Plots training metrics (loss, AUC, etc.) from model training.

