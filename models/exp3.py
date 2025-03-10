# -*- coding: utf-8 -*-

import pickle
import sys
import timeit
from collections import defaultdict
import numpy as np
import random
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import roc_auc_score, precision_score, recall_score
from tabulate import tabulate
import pandas as pd
from transformers import BertTokenizer, BertModel
import re


class CompoundProteinInteractionPrediction(nn.Module):
    def __init__(self, reverse_word_dict, protbert_model_name):
        super(CompoundProteinInteractionPrediction, self).__init__()
        
        # Load ProtBERT model and tokenizer
        self.protbert_tokenizer = BertTokenizer.from_pretrained(protbert_model_name)
        self.protbert_model = BertModel.from_pretrained(protbert_model_name)
        
        # Freeze all layers except for the last few layers of ProtBERT
        for param in self.protbert_model.parameters():
            param.requires_grad = False
        # Unfreeze the last few layers
        for param in self.protbert_model.encoder.layer[-2:].parameters():  # Unfreeze the last 2 layers
            param.requires_grad = True
            
        
        self.reverse_word_dict = reverse_word_dict
        
        
        self.embed_fingerprint = nn.Embedding(n_fingerprint, dim)

        self.W_gnn = nn.ModuleList([nn.Linear(dim, dim).to(device)
                                    for _ in range(layer_gnn)])
        self.bilstm = nn.LSTM(dim, 512, 1, dropout=0, bidirectional=True)

        self.W_attention = nn.Linear(dim, dim)
        self.W_out = nn.ModuleList([nn.Linear(2 * dim, 2 * dim)
                                    for _ in range(layer_output)])
        self.W_interaction = nn.Linear(2 * dim, 2)

    
    def reconstruct_protein_sequence(self, protein_sequence):
        """
        Reconstruct the protein sequence from indices of overlapping 3-grams in the sequence.
        
        protein_sequence: List of indices pointing to 3-grams in reverse_word_dict.
        """
        reconstructed_sequence = []  # List to store the reconstructed sequence
        
        # Loop through the protein sequence, skipping the first and last triplets
        for i, index in enumerate(protein_sequence):
            triplet = self.reverse_word_dict.get(int(index))
    
            if triplet:
               if i == 0:  # Skip the first triplet (do not add to the sequence)
                  continue
               elif i == len(protein_sequence) - 1:  # Skip the last triplet (do not add to the sequence)
                    continue
               elif i == 1:  # For the second triplet, add the whole triplet
                  reconstructed_sequence.extend(list(triplet))
               else:  # For every next triplet, only add the last amino acid
                  reconstructed_sequence.append(triplet[-1])
        
        # Return the reconstructed sequence as a string
        return ' '.join(reconstructed_sequence)

    
    def generate_bert_embeddings(self, protein_sequence):
        """
        Generate embeddings for the protein sequence using ProtBERT.
        The input is a list of indices pointing to 3-grams in reverse_word_dict.
        """
        # Reconstruct the full protein sequence
        reconstructed_sequence = self.reconstruct_protein_sequence(protein_sequence)
        reconstructed_sequence = re.sub(r"[UZOB]", "X", reconstructed_sequence)
        
        # Split the sequence into chunks, reserving space for [CLS] and [SEP]
        max_length = 512  # ProtBERT's maximum input size
        chunk_size = max_length - 2  # Reserve 2 tokens for [CLS] and [SEP]
        chunks = [reconstructed_sequence[i:i + chunk_size] for i in range(0, len(reconstructed_sequence), chunk_size)]
        
        # Process each chunk with ProtBERT
        embeddings = []
        for chunk in chunks:
            # Tokenize the chunk using ProtBERT's tokenizer
            inputs = self.protbert_tokenizer(
                chunk,
                return_tensors="pt",
                padding='max_length',
                truncation=False,
                add_special_tokens=True,
                max_length=max_length,
            ).to(device)
            
            # Pass through ProtBERT
            outputs = self.protbert_model(**inputs)
            
            # Extract the embeddings (shape: [batch_size, seq_length, hidden_size])
            chunk_embeddings = outputs.last_hidden_state  # Shape: [1, 512, 1024]
            
            # Get the attention mask (shape: [1, 512])
            attention_mask = inputs['attention_mask'].squeeze(0).bool()  # Shape: [512]
            
            # Use the attention mask to filter out padding token embeddings
            # Select only the embeddings where attention_mask is True
            chunk_embeddings = chunk_embeddings[0, attention_mask, :]  # Shape: [num_non_padding_tokens, 1024]
            
            embeddings.append(chunk_embeddings)
        
        # Concatenate the embeddings from all chunks
        embeddings = torch.cat(embeddings, dim=0)  # Shape: [total_num_non_padding_tokens, 1024]
        
        return embeddings

    def gnn(self, xs, A, layer):
        for i in range(layer):
            hs = torch.relu(self.W_gnn[i](xs))
            xs = xs + torch.matmul(A, hs)
        return torch.unsqueeze(torch.mean(xs, 0), 0)

    def attention(self, x, xs):
        xs = torch.unsqueeze(xs, 0)  # Adding a batch dimension
        #print(f"xs shape before BiLSTM: {xs.shape}")  # Debugging: print shape
        
       
        bilstms, _ = self.bilstm(xs.to(device))  # Pass through BiLSTM
        bilstms = torch.squeeze(bilstms, 0)  # Remove the extra batch dimension
        
        #print(f"bilstms shape after BiLSTM: {bilstms.shape}")  # Debugging: print shape
    
        h = torch.relu(self.W_attention(x))
        hs = torch.relu(self.W_attention(bilstms))
        weights = torch.tanh(F.linear(h, hs))
        ys = torch.t(weights) * hs
        return torch.unsqueeze(torch.mean(ys, 0), 0)

    def forward(self, inputs):
        fingerprints, adjacency, protein_sequence = inputs
        
        # Ensure all inputs are on the same device
        fingerprints = fingerprints.to(device)
        adjacency = adjacency.to(device)
        
        # Debugging print to check devices
        #print(f"Device of fingerprints: {fingerprints.device}")
        #print(f"Device of adjacency: {adjacency.device}")
    
        fingerprint_vectors = self.embed_fingerprint(fingerprints)
        compound_vector = self.gnn(fingerprint_vectors, adjacency, layer_gnn)
        #print(f"Compound vector device: {compound_vector.device}")
        
        word_vectors = self.generate_bert_embeddings(protein_sequence)
        
        #compound_vector = F.normalize(compound_vector, p=2,dim=-1).to(device) 
        #word_vectors = F.normalize(word_vectors, p=2,dim=-1).to(device)
    
        protein_vector = self.attention(compound_vector, word_vectors)
        #print(f"Protein vector device: {protein_vector.device}")
    
        cat_vector = torch.cat((compound_vector, protein_vector), 1)
        #print(f"Cat vector device (after concat): {cat_vector.device}")
    
        for j in range(layer_output):
            cat_vector = torch.relu(self.W_out[j](cat_vector))
        
        
        interaction = self.W_interaction(cat_vector)
        #print(f"Interaction device: {interaction.device}")
    
        return interaction


    def __call__(self, data, train=True):
        inputs, correct_interaction = data[:-1], data[-1].to(device)
        predicted_interaction = self.forward(inputs)
        #print("_SOS_  :  ",predicted_interaction)

        if train:
            loss = F.cross_entropy(predicted_interaction, correct_interaction)
            return loss
        else:
            correct_labels = correct_interaction.to('cpu').data.numpy()
            ys = F.softmax(predicted_interaction, 1).to('cpu').data.numpy()
            predicted_labels = list(map(lambda x: np.argmax(x), ys))
            predicted_scores = list(map(lambda x: x[1], ys))
            return correct_labels, predicted_labels, predicted_scores


class Trainer(object):
    def __init__(self, model):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

    def train(self, dataset):
        np.random.shuffle(dataset)
        loss_total = 0
        for data in dataset:
                
            loss = self.model(data)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_total += loss.to('cpu').data.numpy()
        
        return loss_total


class Tester(object):
    def __init__(self, model):
        self.model = model

    def test(self, dataset):
        T, Y, S = [], [], []
        for data in dataset:
            (correct_labels, predicted_labels,
             predicted_scores) = self.model(data, train=False)
            T.append(correct_labels)
            Y.append(predicted_labels)
            S.append(predicted_scores)

        AUC = roc_auc_score(T, S)
        precision = precision_score(T, Y, zero_division=0)
        recall = recall_score(T, Y)
        return AUC, precision, recall

    
    def save_AUCs(self, all_metrics_df, filename):
        
        
        with open(filename, 'a') as f:
             f.write(tabulate(all_metrics_df, headers='keys', tablefmt='grid'))
        
        # Print the updated table at the end of each training phase
        print(tabulate(all_metrics_df, headers='keys', tablefmt='grid'))            


    

            
    def save_model(self, model, filename):
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
        }, filename)


def load_tensor(file_name, dtype):
    with open(file_name + '.pkl', 'rb') as f:
        return [dtype(d).to(device) for d in pickle.load(f)]

def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)


def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset


def split_dataset(dataset, ratio):
    n = int(ratio * len(dataset))
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2



if __name__ == "__main__":

    """Hyperparameters."""
    (DATASET, dim, layer_gnn, layer_output,
     lr, lr_decay, decay_interval, weight_decay, iteration,
     setting) = ['dude',1024, 3, 3, 1e-3, 0.5, 10, 1e-6, 100,
                 'dude--dim422--layer_gnn3--layer_output3--lr1e-3--lr_decay0.5--decay_interval10--weight_decay1e-6--iteration100']
    (dim, layer_gnn, layer_output, decay_interval,
     iteration) = map(int, [dim, layer_gnn, layer_output,
                            decay_interval, iteration])
    lr, lr_decay, weight_decay = map(float, [lr, lr_decay, weight_decay])

    """CPU or GPU."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('The code uses GPU...')
    else:
        device = torch.device('cpu')
        print('The code uses CPU!!!')

    """Load preprocessed data."""
    #    dir_input = ('../dataset/' + DATASET + '/input/'
    #                 'radius' + str(radius) + '_ngram' + str(ngram) + '/')
    #dir_input = (
     #   'C:/Users/alexandra/PycharmProjects/GING_DIPLOMA/DeepEmbedding-DTI/dataset/dude/preprocessed_clean_dataset/')
    dir_input = ('/home/alexmoum/')

    fingerprint_dict = load_pickle(dir_input + 'fingerprint_dict.pickle')
    word_dict = load_pickle(dir_input + 'word_dict.pickle')
    reverse_word_dict = {v: k for k, v in word_dict.items()}
    n_fingerprint = len(fingerprint_dict)
    n_word = len(word_dict)

    """Load saved dataset splits for reproducibility."""
    dataset_train = torch.load('/home/alexmoum/train_set_ex3.pt',weights_only=False)#/fold_train.pt' ,weights_only=False)
    
    dataset_dev = torch.load('/home/alexmoum/dev_set_ex3.pt',weights_only=False)#/fold_dev.pt' ,weights_only=False)
 
    dataset_test = torch.load('/home/alexmoum/test_set_ex3.pt',weights_only=False)#/fold_test.pt' ,weights_only=False)#
   
    # file_name = ('/home/alexmoum/DeepEmbedding-DTI/amino_embeddings.pickle')

    

    # Verify the size of each dataset
    print(f"Training set: {len(dataset_train)} samples")
    print(f"Validation set: {len(dataset_dev)} samples")
    print(f"Test set: {len(dataset_test)} samples")

   

    """Output files."""
    #file_AUCs = 'C:/Users/alexandra/PycharmProjects/GING_DIPLOMA/DeepEmbedding-DTI/dataset/dude/bert/AUCs--bert-21' + setting + '.txt'
    #file_model = 'C:/Users/alexandra/PycharmProjects/GING_DIPLOMA/DeepEmbedding-DTI/dataset/dude/bert/-bert-21' + setting + '.pth'
    """Output files."""
    file_AUCs = '/home/alexmoum/bert/AUCs--final' + setting + '.txt'
    
    #file_model = '/home/alexmoum/bert/1' + setting + '.pth'


    # Early Stopping Parameters
    patience = 5  # Number of epochs to wait for improvement
    best_auc = 0  # Initialize best AUC
    patience_counter = 0  # Counter for epochs without improvement

    start = timeit.default_timer()

    best_metrics = []
    all_metrics=[]
    
    # Array of three random seeds
    seed_array = [7,22,11]  # Replace these with your desired seed values

    for seed_idx, seed in enumerate(seed_array):
    
        print(f"Seed {seed_idx + 1}/{len(seed_array)}: Using seed {seed}")
        
        # Define file path for this seed's best model
        file_model = f'/home/alexmoum/bert/best_model_seed{seed}.pth'
        
        
        
        """Set a model."""
        torch.manual_seed(seed)
        model = CompoundProteinInteractionPrediction(reverse_word_dict=reverse_word_dict, protbert_model_name="Rostlab/prot_bert").to(device)
        trainer = Trainer(model)
        tester = Tester(model)
            
       
        # re-initiallise stoping parameters for next seed
        best_auc = 0
        patience_counter = 0
        
        # Train the model on this seed
        for epoch in range(1, iteration + 1):  # Use iteration + 1 to include the last epoch

            if epoch % decay_interval == 0:
                trainer.optimizer.param_groups[0]['lr'] *= lr_decay
            print(f"Epoch is {epoch}")
            loss_train = trainer.train(dataset_train)
            AUC_dev, precision_dev, recall_dev = tester.test(dataset_dev)

            end = timeit.default_timer()
            time = end - start

            AUCs = [seed, epoch, time, loss_train, AUC_dev, precision_dev, recall_dev]
            all_metrics.append(AUCs)
            # Print metrics immediately for each epoch
            print(f"{seed}) Epoch {epoch}/{iteration} - Time: {time:.2f}s, Loss: {loss_train:.4f}, "
            f"AUC (Dev): {AUC_dev:.4f}, Precision (Dev): {precision_dev:.4f}, Recall (Dev): {recall_dev:.4f}")

            # Early Stopping Logic
            if AUC_dev > best_auc:
                best_auc = AUC_dev
                patience_counter = 0  # Reset counter if we have an improvement
                # Optionally save the best model here
                tester.save_model(model, file_model)
            else:
                patience_counter += 1  # Increment patience counter

            if patience_counter >= patience:
                print(f'Early stopping triggered after {epoch} epochs .')
                break  # Stop training if patience is exceeded

       
       
        # After training, load the best model for this seed and evaluate on the test set

        print(f"Evaluating the best model for Seed {seed} on the test set...")

        best_model_checkpoint = torch.load(file_model)

        model.load_state_dict(best_model_checkpoint['model_state_dict'])

        model.to(device)  # Ensure model is on the correct device

    

        # Evaluate the best model on the test set

        AUC_test, precision_test, recall_test = tester.test(dataset_test)
        
        AUCs =[f"Seed {seed}", AUC_test,precision_test,recall_test]
    
        # Log test metrics for this seed

        best_metrics.append(AUCs)
        best_metrics_df=pd.DataFrame(best_metrics, columns=["Seed","AUC_test","precision_test","recall_test"])
        
        
    all_metrics_df=pd.DataFrame(all_metrics, columns=["Seed", "epoch", "time", "loss_train", "AUC_dev", "precision_dev", "recall_dev"])
    tester.save_AUCs(all_metrics_df, file_AUCs)
    tester.save_AUCs(best_metrics_df, file_AUCs )
    print("Best Model Metrics:", best_metrics)# Convert best_metrics to a NumPy array, excluding the first column (the labels)
    
    seed_metrics_array = np.array([metrics[1:] for metrics in best_metrics], dtype=float)
    # Calculate and log final metrics
    mean_metrics = seed_metrics_array.mean(axis=0)
    std_metrics = seed_metrics_array.std(axis=0)
    
    # Create a DataFrame for the final summary
    final_summary_df = pd.DataFrame({
        "Metric": ["Mean AUC", "Mean Precision", "Mean Recall"],
        "Value": [
            f"{mean_metrics[0]:.4f} +/- {std_metrics[0]:.4f}",
            f"{mean_metrics[1]:.4f} +/- {std_metrics[1]:.4f}",
            f"{mean_metrics[2]:.4f} +/- {std_metrics[2]:.4f}"
        ]
    })
    
    # Log and print summary metrics
    tester.save_AUCs(final_summary_df, file_AUCs)
    sys.exit(0)