import os
import sys
import shutil
import subprocess
import pandas as pd
import matplotlib.pyplot as plt

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

def run_command(cmd, desc):
    print(f"\n--- Running: {desc} ---")
    print(f"Command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=CURRENT_DIR)
    print(f"--- Finished: {desc} ---\n")

def parse_mixed_csv(csv_path):
    columns = ['Epoch', 'AvgLoss', 'AvgBCELoss', 'Precision', 'Recall', 'F1_Score', 'IoU', 'Dice',
               'ValLoss', 'ValBCELoss', 'ValPrecision', 'ValRecall', 'ValF1_Score', 'ValIoU', 'ValDice']
    data = []
    if not os.path.exists(csv_path):
        return pd.DataFrame(columns=columns)
        
    with open(csv_path, 'r') as f:
        f.readline()
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if parts[0] == 'Epoch':
                continue
            
            row = {col: None for col in columns}
            try:
                row['Epoch'] = float(parts[0])
            except ValueError:
                continue
                
            row['AvgLoss'] = float(parts[1])
            row['AvgBCELoss'] = float(parts[2])
            row['Precision'] = float(parts[3])
            row['Recall'] = float(parts[4])
            row['F1_Score'] = float(parts[5])
            row['IoU'] = float(parts[6])
            row['Dice'] = float(parts[7])
            
            if len(parts) >= 15:
                row['ValLoss'] = float(parts[8])
                row['ValBCELoss'] = float(parts[9])
                row['ValPrecision'] = float(parts[10])
                row['ValRecall'] = float(parts[11])
                row['ValF1_Score'] = float(parts[12])
                row['ValIoU'] = float(parts[13])
                row['ValDice'] = float(parts[14])
            data.append(row)
    return pd.DataFrame(data)

def plot_metrics(csv_path, output_path):
    print(f"Generating training & validation plot: {output_path}...")
    df = parse_mixed_csv(csv_path)
    if len(df) == 0:
        print("Metrics CSV is empty. Skipping plot generation.")
        return

    metrics_pairs = [
        ('AvgLoss', 'ValLoss', 'Total Loss'),
        ('AvgBCELoss', 'ValBCELoss', 'BCE Loss'),
        ('Precision', 'ValPrecision', 'Precision'),
        ('Recall', 'ValRecall', 'Recall'),
        ('F1_Score', 'ValF1_Score', 'F1 Score'),
        ('IoU', 'ValIoU', 'IoU'),
        ('Dice', 'ValDice', 'Dice')
    ]

    fig, axes = plt.subplots(4, 2, figsize=(15, 20))
    axes = axes.flatten()
    
    for i, (train_col, val_col, title) in enumerate(metrics_pairs):
        ax = axes[i]
        ax.plot(df['Epoch'], df[train_col], marker='o', label=f'Train {title}', color='blue')
        
        if val_col in df.columns and not df[val_col].isna().all():
            valid_val_data = df.dropna(subset=[val_col])
            ax.plot(valid_val_data['Epoch'], valid_val_data[val_col], marker='s', label=f'Val {title}', color='red')
            
        ax.set_title(f'Train vs Val: {title}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(title)
        
        if len(df['Epoch']) <= 20:
            ax.set_xticks(df['Epoch'])
            
        if 'Loss' not in title:
            ax.set_ylim(0, 1.05)
            
        ax.legend()
        ax.grid(True)
    
    fig.delaxes(axes[7])
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"Plot successfully saved to: {output_path}")

def aggregate_results(out_dir):
    print(f"Aggregating Sobel results into: {out_dir}...")
    os.makedirs(out_dir, exist_ok=True)
    weights_out = os.path.join(out_dir, "weights")
    logs_out = os.path.join(out_dir, "logs")
    results_out = os.path.join(out_dir, "results")
    
    os.makedirs(weights_out, exist_ok=True)
    os.makedirs(logs_out, exist_ok=True)
    
    # 1. Copy weights
    weights_src_dir = os.path.join(CURRENT_DIR, "weights")
    if os.path.exists(weights_src_dir):
        for model in ["FENet", "SegNet"]:
            for suffix in ["sobel_best.pth", "sobel_latest.pth", "sobel_25.pth", "sobel_50.pth"]:
                filename = f"{model}_{suffix}"
                src = os.path.join(weights_src_dir, filename)
                if os.path.exists(src):
                    shutil.copy(src, os.path.join(weights_out, filename))
                
    # 2. Copy test visualization images
    results_src_dir = os.path.join(CURRENT_DIR, "results")
    if os.path.exists(results_src_dir):
        if os.path.exists(results_out):
            shutil.rmtree(results_out)
        shutil.copytree(results_src_dir, results_out)
        print(f"Copied visual results from {results_src_dir} to {results_out}")

    # 3. Copy logs
    logs_src_dir = os.path.join(CURRENT_DIR, "logs")
    if os.path.exists(logs_src_dir):
        for log_file in ['metrics.csv', 'test_metrics.csv', 'metrics_val_plot.png']:
            src = os.path.join(logs_src_dir, log_file)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(logs_out, log_file))
                
    print("File aggregation completed.")

def generate_summary(out_dir):
    print("Generating best model summary CSV...")
    metrics_path = os.path.join(out_dir, "logs", "metrics.csv")
    test_metrics_path = os.path.join(out_dir, "logs", "test_metrics.csv")
    
    if not os.path.exists(metrics_path) or not os.path.exists(test_metrics_path):
        print(f"Warning: Missing metrics files at {metrics_path} or {test_metrics_path}.")
        return

    df = parse_mixed_csv(metrics_path)
    if df.empty or 'ValF1_Score' not in df.columns:
        print("Train/Val metrics are empty.")
        return
        
    best_row = df.loc[df['ValF1_Score'].idxmax()]
    best_epoch = int(best_row['Epoch'])
    
    test_df = pd.read_csv(test_metrics_path)
    if test_df.empty:
        print("Test metrics are empty.")
        return
    test_row = test_df.iloc[-1]

    summary = {
        'Dataset': ['Train', 'Validation', 'Test'],
        'Precision': [best_row['Precision'], best_row['ValPrecision'], test_row['Precision']],
        'Recall': [best_row['Recall'], best_row['ValRecall'], test_row['Recall']],
        'F1_Score': [best_row['F1_Score'], best_row['ValF1_Score'], test_row['F1_Score']],
        'IoU': [best_row['IoU'], best_row['ValIoU'], test_row['IoU']]
    }
    summary_df = pd.DataFrame(summary)
    summary_path = os.path.join(out_dir, "best_model_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    
    print(f"Summary successfully saved to {summary_path}")
    print(f"\n{'='*45}")
    print(f"   BEST SOBEL MODEL SUMMARY (Epoch {best_epoch})")
    print(f"{'='*45}")
    print(summary_df.to_string(index=False))
    print(f"{'='*45}\n")

if __name__ == "__main__":
    OUT_DIR = os.path.join(CURRENT_DIR, "sobel_results")
    
    # 1. Run testing & evaluation
    run_command(['uv', 'run', 'test_sobel.py'], 'Testing (test_sobel.py)')
    
    # 2. Plot train vs validation metrics
    metrics_csv = os.path.join(CURRENT_DIR, 'logs', 'metrics.csv')
    plot_output = os.path.join(CURRENT_DIR, 'logs', 'metrics_val_plot.png')
    plot_metrics(metrics_csv, plot_output)
    
    # 3. Aggregate files into output directory
    aggregate_results(OUT_DIR)
    
    # 4. Generate best model summary table
    generate_summary(OUT_DIR)
    
    print("\n--- Sobel Pipeline Completed Successfully ---")
