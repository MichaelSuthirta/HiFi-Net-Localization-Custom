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

    plt.figure(figsize=(10, 5))
    
    # Plot Total Loss
    plt.plot(df['Epoch'], df['AvgLoss'], marker='o', label='Total Avg Loss', color='blue')
    # Plot BCE Loss
    plt.plot(df['Epoch'], df['AvgBCELoss'], marker='x', label='Avg BCE Loss', color='red')

    plt.title('Training Loss Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xticks(df['Epoch'])
    plt.legend()
    plt.grid(True)
    
    plt.savefig(output_path)
    print(f"Plot saved successfully to {output_path}")

if __name__ == '__main__':
    plot_metrics()
