import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_metrics(csv_path='logs/metrics.csv', output_path='logs/metrics_plot.png'):
    if not os.path.exists(csv_path):
        print(f"File {csv_path} not found. Train the model first.")
        return

    df = pd.read_csv(csv_path)
    if len(df) == 0:
        print("CSV is empty.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Losses
    axes[0].plot(df['Epoch'], df['AvgLoss'], marker='o', label='Total Avg Loss', color='blue')
    axes[0].plot(df['Epoch'], df['AvgBCELoss'], marker='x', label='Avg BCE Loss', color='red')
    axes[0].set_title('Training Loss over Epochs')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    if len(df['Epoch']) <= 20:
        axes[0].set_xticks(df['Epoch'])
    axes[0].legend()
    axes[0].grid(True)
    
    # Performance Metrics
    if 'F1_Score' in df.columns:
        axes[1].plot(df['Epoch'], df['Precision'], marker='o', label='Precision')
        axes[1].plot(df['Epoch'], df['Recall'], marker='s', label='Recall')
        axes[1].plot(df['Epoch'], df['F1_Score'], marker='^', label='F1 Score / Dice')
        axes[1].plot(df['Epoch'], df['IoU'], marker='d', label='IoU')
        
        axes[1].set_title('Training Metrics over Epochs')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Score (0 to 1)')
        if len(df['Epoch']) <= 20:
            axes[1].set_xticks(df['Epoch'])
        axes[1].set_ylim(0, 1.05)
        axes[1].legend()
        axes[1].grid(True)
    else:
        axes[1].set_title('Training Metrics over Epochs')
        axes[1].text(0.5, 0.5, 'Metrics columns not found in CSV.\nPlease run training to log new metrics.', 
                     horizontalalignment='center', verticalalignment='center', transform=axes[1].transAxes)
        axes[1].axis('off')
        
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    print(f"Plot saved successfully to {output_path}")

if __name__ == '__main__':
    plot_metrics()
