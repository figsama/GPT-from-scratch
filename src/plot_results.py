import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob

# Set plot style
sns.set_style("whitegrid")
plt.style.use('seaborn')

def plot_metrics(file_path, title, ylabel, output_dir):
    # Read CSV file
    df = pd.read_csv(file_path)
    
    # Create figure
    plt.figure(figsize=(10, 6))
    plt.plot(df['Step'], df['Value'], marker='o', markersize=4, linewidth=2)
    
    # Set title and labels
    plt.title(title, fontsize=14, pad=15)
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    plt.savefig(os.path.join(output_dir, f"{os.path.basename(file_path).replace('.csv', '.png')}"), 
                dpi=300, bbox_inches='tight')
    plt.close()

# Create base output directory
if not os.path.exists('plots'):
    os.makedirs('plots')

# Process files for each experiment group (A, B, C)
for group in ['A', 'B', 'C']:
    # Create group directory
    group_dir = os.path.join('plots', group)
    if not os.path.exists(group_dir):
        os.makedirs(group_dir)
    
    # Get all CSV files for this group
    csv_files = glob.glob(f'results/{group}/*.csv')
    
    for file_path in csv_files:
        # Extract model name and metric type from filename
        filename = os.path.basename(file_path)
        parts = filename.replace('.csv', '').split('_')
        
        model_name = parts[0].upper()
        metric_type = '_'.join(parts[1:])
        
        # Create title based on filename
        if 'perplexity' in metric_type:
            title = f'{model_name} {metric_type.replace("_", " ").title()}'
            ylabel = 'Perplexity'
        else:
            title = f'{model_name} {metric_type.replace("_", " ").title()}'
            ylabel = 'Loss'
            
        plot_metrics(file_path, title, ylabel, group_dir)

print("All plots have been generated successfully!") 