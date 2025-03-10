# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 21:49:54 2019

@author: chenwei
"""

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


class CompoundProteinInteractionPrediction(nn.Module):
    def __init__(self):
        super(CompoundProteinInteractionPrediction, self).__init__()

        self.embed_fingerprint = nn.Embedding(n_fingerprint, dim)
        self.embed_word = nn.Embedding(n_word, dim)
        self.W_gnn = nn.ModuleList([nn.Linear(dim, dim).to(device)
                                    for _ in range(layer_gnn)])
     
        self.bilstm = nn.LSTM(dim, 5, 1, dropout=0, bidirectional=True)

        self.W_attention = nn.Linear(dim, dim)
        self.W_out = nn.ModuleList([nn.Linear(2 * dim, 2 * dim)
                                    for _ in range(layer_output)])
        self.W_interaction = nn.Linear(2 * dim, 2)

    def gnn(self, xs, A, layer):
        for i in range(layer):
            hs = torch.relu(self.W_gnn[i](xs))
            xs = xs + torch.matmul(A, hs)
        # return torch.unsqueeze(torch.sum(xs, 0), 0)
        return torch.unsqueeze(torch.mean(xs, 0), 0)#this step colapse all node feautre vectors to a single graph wide feature vector

    def attention_cnn(self, x, xs):
        
        # print(xs.size())

        xs = torch.unsqueeze(xs, 0)
        bilstms, _ = self.bilstm(xs.to(device))
        bilstms = torch.squeeze(bilstms, 0)
        
        h = torch.relu(self.W_attention(x))
        hs = torch.relu(self.W_attention(bilstms))
        weights = torch.tanh(F.linear(h, hs))
        ys = torch.t(weights) * hs

        # return torch.unsqueeze(torch.sum(ys, 0), 0)
        return torch.unsqueeze(torch.mean(ys, 0), 0)

    def forward(self, inputs):

        fingerprints, adjacency, words = inputs

        """Compound vector with GNN."""
        fingerprint_vectors = self.embed_fingerprint(fingerprints.to(device))
        compound_vector = self.gnn(fingerprint_vectors, adjacency.to(device),layer_gnn)

        """Protein vector with attention-CNN."""
        word_vectors = self.embed_word(words.to(device))
        #        print(word_vectors.dim())
        #        print(word_vectors.size())
        #        print(len(word_vectors))
        protein_vector = self.attention_cnn(compound_vector,
                                            word_vectors)

        """Concatenate the above two vectors and output the interaction."""
        cat_vector = torch.cat((compound_vector, protein_vector), 1)
        for j in range(layer_output):
            cat_vector = torch.relu(self.W_out[j](cat_vector))
        interaction = self.W_interaction(cat_vector)

        return interaction

    def __call__(self, data, train=True):

        inputs, correct_interaction = data[:-1], data[-1].to(device)
        predicted_interaction = self.forward(inputs)

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
        N = len(dataset)
        # print(N)
        loss_total = 0
        for data in dataset:
            loss = self.model(data)
            self.optimizer.zero_grad() # Clears Previous Gradients
            loss.backward() # Computes the gradients via backpropagation
            self.optimizer.step() # Updates the model's parameters
            loss_total += loss.to('cpu').data.numpy() # Accumulate loss
        return loss_total


class Tester(object):
    def __init__(self, model):
        self.model = model

    def test(self, dataset):
        N = len(dataset)
        # print(N)
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



    def save_model(self, model,
                   filename): 
        torch.save({
            'model_state_dict': model.state_dict(),
            'n_fingerprint': n_fingerprint,
            'n_word': n_word,
            'dim': dim,
            'layer_gnn': layer_gnn,
            'layer_cnn': layer_cnn,
            'layer_output': layer_output,
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'epoch': epoch
        }, filename)


def load_tensor(file_name, dtype):
    with open(file_name + '.pkl', 'rb') as f:
        return [dtype(d).to(device) for d in pickle.load(f)]


def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)



if __name__ == "__main__":

    """Hyperparameters."""
    (DATASET, dim, layer_gnn, layer_output,
     lr, lr_decay, decay_interval, weight_decay, iteration,
     setting) = ['dude',10,3, 1e-3, 0.5, 10, 1e-6, 100,
                 'dude--dim10--layer_gnn3--layer_output3--lr1e-3--lr_decay0.5--decay_interval10--weight_decay1e-6--iteration100']
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

    """Load data dictionaries."""
 
    dir_input = ('/home/alexmoum/')

    fingerprint_dict = load_pickle(dir_input + 'fingerprint_dict.pickle')
    word_dict = load_pickle(dir_input + 'word_dict.pickle')
    
    n_fingerprint = len(fingerprint_dict)
    n_word = len(word_dict)

    """Load saved dataset splits for reproducibility."""
    dataset_train = torch.load('/home/alexmoum/train_set_ex3.pt',weights_only=False)
    dataset_dev = torch.load('/home/alexmoum/dev_set_ex3.pt',weights_only=False)
    dataset_test = torch.load('/home/alexmoum/test_set_ex3.pt',weights_only=False)
 



    """ Verify the size of each dataset """
    print(f"Training set: {len(dataset_train)} samples")
    print(f"Validation set: {len(dataset_dev)} samples")
    print(f"Test set: {len(dataset_test)} samples")

   

    """Output files."""
    #file_AUCs = 'C:/Users/alexandra/PycharmProjects/GING_DIPLOMA/DeepEmbedding-DTI/dataset/dude/bert/AUCs--bert-21' + setting + '.txt'
    #file_model = 'C:/Users/alexandra/PycharmProjects/GING_DIPLOMA/DeepEmbedding-DTI/dataset/dude/bert/-bert-21' + setting + '.pth'
    """Output files."""
    file_AUCs = '/home/alexmoum/bert/AUCs--final' + setting + '.txt'
    


    # Early Stopping Parameters
    patience = 5  # Number of epochs to wait for improvement
    best_auc = 0  # Initialize best AUC
    patience_counter = 0  # Counter for epochs without improvement

    start = timeit.default_timer()

    best_metrics = []
    all_metrics=[]
    
    # Array of three random seeds
    seed_array = [7,22,11]  

    for seed_idx, seed in enumerate(seed_array):
    
        print(f"Seed {seed_idx + 1}/{len(seed_array)}: Using seed {seed}")
        
        # Define file path for this seed's best model
        file_model = f'/home/alexmoum/bert/best_model_seed{seed}.pth'
        
        
        
        """Set a model."""
        torch.manual_seed(seed)
        model = CompoundProteinInteractionPrediction().to(device)
        trainer = Trainer(model)
        tester = Tester(model)
            
       
        # re-initiallise stoping parameters for next seed
        best_auc = 0
        patience_counter = 0
        
        # Train the model on this seed
        for epoch in range(1, iteration + 1):  # Use iteration + 1 to include the last epoch

            if epoch % decay_interval == 0:
                trainer.optimizer.param_groups[0]['lr'] *= lr_decay

            loss_train = trainer.train(dataset_train)
            AUC_dev, precision_dev, recall_dev = tester.test(dataset_dev)

            end = timeit.default_timer()
            time = end - start

            AUCs = [seed, epoch, time, loss_train, AUC_dev, precision_dev, recall_dev]
            all_metrics.append(AUCs)
            
         

            # Early Stopping Logic
            if AUC_dev > best_auc:
                best_auc = AUC_dev
                patience_counter = 0  # Reset counter if we have an improvement
                # Save the best model here
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
    print("Best Model Metrics:", best_metrics)
    
    seed_metrics_array = np.array([metrics[1:] for metrics in best_metrics], dtype=float) # Convert best_metrics to a NumPy array, excluding the first column (the labels)
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
    
    