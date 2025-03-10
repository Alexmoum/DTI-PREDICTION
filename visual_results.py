# -*- coding: utf-8 -*-
import pandas as pd
import re
import matplotlib.pyplot as plt

def clean_line(line):
    """Cleans the line by stripping unwanted spaces and checking for special characters."""
    return line.strip()

def load_log_to_dataframe(log_file_path):
    try:
        with open(log_file_path, 'r') as log_file:
            lines = log_file.readlines()
    except FileNotFoundError:
        print(f"Error: The file '{log_file_path}' was not found.")
        return None, None, None
    
    # Initialize lists to collect data for each section
    training_data = []
    metrics_data = []
    summary_data = []
    metrics_started = False
    summary_started = False

    # Iterate through the log lines
    for line in lines:
        line = clean_line(line)
        
        if not line:
            continue
        
        # Skip separator lines like +----+
        if re.match(r'^\+[-\+]+$', line):
            continue

        # Split the line into parts and clean
        parts = [part.strip() for part in line.split('|') if part.strip()]
        
        # Skip lines with no meaningful data after cleaning
        if not parts:
            continue

        # Handle Training Data Section (expected 8 parts per row)
        if len(parts) == 8:  # Training data row format
            training_data.append(parts[1:])  # Skip the extra index column
        
        # Handle Metrics Data Section (lines with "Seed" and expected 5 parts)
        if len(parts) == 5 and 'Seed' in parts[1]:
            metrics_started = True
            metrics_data.append(parts[1:])  # Skip the extra index column
        
        # Handle Summary Section (lines with "Mean" in the first part)
        if len(parts) == 3 and summary_started:
            summary_data.append(parts)  # No need to skip, since it's a 3-column format
        
        # Look for the start of the summary section (based on 'Metric' in the header)
        if len(parts) == 2 and 'Metric' in parts[0] and 'Value' in parts[1]:
            summary_started = True
            continue  # Skip header line and start capturing data

    # Convert training data to DataFrame
    if training_data:
        columns = ['Seed', 'epoch', 'time', 'loss_train', 'AUC_dev', 'Precision_dev', 'Recall_dev']
        training_df = pd.DataFrame(training_data, columns=columns)
    else:
        training_df = pd.DataFrame()

    # Convert metrics data to DataFrame
    if metrics_data:
        metrics_columns = ['Seed', 'AUC_test', 'Precision_test', 'Recall_test']
        metrics_df = pd.DataFrame(metrics_data, columns=metrics_columns)
    else:
        metrics_df = pd.DataFrame()

    # Convert summary data to DataFrame
    if summary_data:
        summary_columns = ['Index', 'Metric', 'Value']  # Modify this to include 'Index' as a column
        summary_df = pd.DataFrame(summary_data, columns=summary_columns)
    else:
        summary_df = pd.DataFrame()

    return training_df, metrics_df, summary_df

# Example usage with your .txt file
log_file_path = '/home/alexmoum/bert/bert.txt'  # Path to your .txt file
training_df, metrics_df, summary_df = load_log_to_dataframe(log_file_path)

# Show dataframes for inspection
print("Training Data:")
print(training_df.head())

print("\nMetrics Data:")
print(metrics_df.head())

print("\nSummary Data:")
print(summary_df.head())


def plot_training_metrics(training_df, output_file='training_metrics_plot.png'):
    # Convert necessary columns to float
    training_df['loss_train']=training_df['loss_train'].astype(float)
    training_df['AUC_dev'] = training_df['AUC_dev'].astype(float)
    training_df['Precision_dev'] = training_df['Precision_dev'].astype(float)
    training_df['Recall_dev'] = training_df['Recall_dev'].astype(float)
    training_df['epoch'] = training_df['epoch'].astype(int)

    plt.figure(figsize=(8, 8))

    # Plot AUC vs Iterations for each seed
    plt.subplot(4, 1, 1)
    for seed in training_df['Seed'].unique():
        seed_data = training_df[training_df['Seed'] == seed]
        plt.plot(seed_data['epoch'], seed_data['AUC_dev'], marker='o', linestyle='-', label=f'Seed {seed}')
    plt.title('AUC Dev vs Iterations for each Seed',fontsize=15)
    
    plt.xticks(fontsize=12)  
    plt.yticks(fontsize=12)  

    plt.xlabel('Epoch',fontsize=15)
    plt.ylabel('AUC Dev',fontsize=15)
    plt.legend()

    # Plot Precision vs Iterations for each seed
    plt.subplot(4, 1, 2)
    for seed in training_df['Seed'].unique():
        seed_data = training_df[training_df['Seed'] == seed]
        plt.plot(seed_data['epoch'], seed_data['Precision_dev'], marker='x', linestyle='-', label=f'Seed {seed}')
    plt.title('Precision Dev vs Epoch for each Seed',fontsize=15)
    
    plt.xticks(fontsize=12)  
    plt.yticks(fontsize=12)  
    
    plt.xlabel('Epoch',fontsize=15)
    plt.ylabel('Precision Dev',fontsize=15)
    plt.legend()

    # Plot Recall vs Iterations for each seed
    plt.subplot(4, 1, 3)
    for seed in training_df['Seed'].unique():
        seed_data = training_df[training_df['Seed'] == seed]
        plt.plot(seed_data['epoch'], seed_data['Recall_dev'], marker='s', linestyle='-', label=f'Seed {seed}')
    plt.title('Recall Dev vs Epoch for each Seed',fontsize=15)
    
    plt.xticks(fontsize=12)  
    plt.yticks(fontsize=12)  

    plt.xlabel('Iteration',fontsize=15)
    plt.ylabel('Recall Dev',fontsize=15)
    plt.legend()
    
    # Plot Loss_Train vs Iteration for each seed
    plt.subplot(4, 1, 4)
    for seed in training_df['Seed'].unique():
        seed_data = training_df[training_df['Seed'] == seed]
        plt.plot(seed_data['epoch'], seed_data['loss_train'], marker='o', linestyle='-', label=f'Seed {seed}')
    plt.title('Loss_train vs Iterations for each Seed',fontsize=15)
    
    plt.xticks(fontsize=12)  
    plt.yticks(fontsize=12)  

    plt.xlabel('Epoch',fontsize=15)
    plt.ylabel('Loss_train',fontsize=15)
    plt.legend()

    plt.tight_layout()

    # Save the plot to a file
    plt.savefig(output_file)
    print(f"Plot saved as {output_file}")

# Call the function to plot
plot_training_metrics(training_df, output_file='/home/alexmoum/training_metrics_plot_exp2.png')
