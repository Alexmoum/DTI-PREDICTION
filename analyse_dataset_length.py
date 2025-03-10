#LENGTHS OF EACH DATASETS


import torch
import matplotlib.pyplot as plt

# Load the datasets from the .pt files
dataset_train = torch.load('/home/alexmoum/train_set_ex3.pt', weights_only=False)
dataset_val = torch.load('/home/alexmoum/dev_set_ex3.pt', weights_only=False)
dataset_test = torch.load('/home/alexmoum/test_set_ex3.pt', weights_only=False)

# Function to extract unique tensor lengths
def get_unique_tensor_lengths(dataset):
    # Extract the third tensor from each sample
    third_tensors = [sample[2] for sample in dataset]
    
    # Remove duplicate tensors
    unique_tensors = []
    for tensor in third_tensors:
        if not any(torch.equal(tensor, existing_tensor) for existing_tensor in unique_tensors):
            unique_tensors.append(tensor)
    
    # Calculate the length of each unique tensor
    tensor_lengths = [len(tensor) for tensor in unique_tensors]
    return tensor_lengths

# Get lengths for each dataset
train_lengths = get_unique_tensor_lengths(dataset_train)
val_lengths = get_unique_tensor_lengths(dataset_val)
test_lengths = get_unique_tensor_lengths(dataset_test)

# Create a figure with three subplots
plt.figure(figsize=(8, 6))

# Plot for training set
plt.subplot(3, 1, 1)
plt.hist(train_lengths, bins=30, color='skyblue', edgecolor='black')
plt.title('Training Set: Distribution of Sequence Lengths',fontsize=15)
plt.xlabel('Length',fontsize=15)
plt.ylabel('Frequency',fontsize=15)
plt.grid(True)

# Plot for validation set
plt.subplot(3, 1, 2)
plt.hist(val_lengths, bins=30, color='lightgreen', edgecolor='black')
plt.title('Validation Set: Distribution of Sequence Lengths',fontsize=15)
plt.xlabel('Length',fontsize=15)
plt.ylabel('Frequency',fontsize=15)
plt.grid(True)

# Plot for test set
plt.subplot(3, 1, 3)
plt.hist(test_lengths, bins=30, color='salmon', edgecolor='black')
plt.title('Test Set: Distribution of Sequence Lengths',fontsize=15)
plt.xlabel('Length',fontsize=15)
plt.ylabel('Frequency',fontsize=15)
plt.grid(True)

# Save the figure to a file
plt.tight_layout()
plt.savefig('/home/alexmoum/sequence_lengths_distribution.png', dpi=300)
plt.show()
