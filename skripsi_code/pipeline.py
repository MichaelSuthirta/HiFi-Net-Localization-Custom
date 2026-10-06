import os
import shutil
import subprocess
import pandas as pd
import re

def run_command(cmd, desc):
    print(f"\n--- Running: {desc} ---")
    print(f"Command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"--- Finished: {desc} ---\n")

def aggregate_results(out_dir):
    print(f"Aggregating results into {out_dir}...")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(f"{out_dir}/weights", exist_ok=True)
    os.makedirs(f"{out_dir}/logs", exist_ok=True)
    
    # 1. Copy weights
    for model in ["FENet", "SegNet"]:
        for suffix in ["best.pth", "25.pth", "50.pth"]:
            filename = f"{model}_{suffix}"
            src = f"weights/{filename}"
            if os.path.exists(src):
                shutil.copy(src, f"{out_dir}/weights/{filename}")
            else:
                print(f"Warning: {src} not found (This might be expected if the epoch wasn't reached).")

    # 2. Copy results directory (test visualizations)
    if os.path.exists('results'):
        if os.path.exists(f"{out_dir}/results"):
            shutil.rmtree(f"{out_dir}/results")
        shutil.copytree('results', f"{out_dir}/results")

    # 3. Copy logs
    for log_file in ['metrics.csv', 'test_metrics.csv', 'metrics_val_plot.png']:
        src = f"logs/{log_file}"
        if os.path.exists(src):
            shutil.copy(src, f"{out_dir}/logs/{log_file}")
            
    print("File aggregation completed.")

def parse_mixed_csv(csv_path):
    columns = ['Epoch', 'AvgLoss', 'AvgBCELoss', 'Precision', 'Recall', 'F1_Score', 'IoU', 'Dice',
               'ValLoss', 'ValBCELoss', 'ValPrecision', 'ValRecall', 'ValF1_Score', 'ValIoU', 'ValDice']
    data = []
    with open(csv_path, 'r') as f:
        f.readline()
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split(',')
            if parts[0] == 'Epoch': continue
            
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

def generate_summary(out_dir):
    print("Generating best model summary CSV...")
    metrics_path = f"{out_dir}/logs/metrics.csv"
    test_metrics_path = f"{out_dir}/logs/test_metrics.csv"
    
    if not os.path.exists(metrics_path) or not os.path.exists(test_metrics_path):
        print("Missing metrics files. Cannot generate summary.")
        return

    # Parse train/val
    df = parse_mixed_csv(metrics_path)
    if df.empty:
        print("Train/Val metrics are empty.")
        return
        
    # Find best epoch based on ValF1_Score
    best_row = df.loc[df['ValF1_Score'].idxmax()]
    best_epoch = best_row['Epoch']
    
    # Parse test
    test_df = pd.read_csv(test_metrics_path)
    if test_df.empty:
        print("Test metrics are empty.")
        return
    test_row = test_df.iloc[-1]

    # Combine
    summary = {
        'Dataset': ['Train', 'Validation', 'Test'],
        'Precision': [best_row['Precision'], best_row['ValPrecision'], test_row['Precision']],
        'Recall': [best_row['Recall'], best_row['ValRecall'], test_row['Recall']],
        'F1_Score': [best_row['F1_Score'], best_row['ValF1_Score'], test_row['F1_Score']],
        'IoU': [best_row['IoU'], best_row['ValIoU'], test_row['IoU']]
    }
    summary_df = pd.DataFrame(summary)
    summary_path = f"{out_dir}/best_model_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    
    print(f"Summary successfully saved to {summary_path}")
    print(f"\n--- Best Epoch summary (Epoch {best_epoch}) ---")
    print(summary_df)

if __name__ == "__main__":
    OUT_DIR = "tuning_results_pos_weight_4"
    
    run_command(['uv', 'run', 'test.py'], 'Testing (test.py)')
    run_command(['uv', 'run', 'plot_metrics_val.py'], 'Plotting (plot_metrics_val.py)')
    
    aggregate_results(OUT_DIR)
    generate_summary(OUT_DIR)
    
    print("\n--- Pipeline Completed Successfully ---")
