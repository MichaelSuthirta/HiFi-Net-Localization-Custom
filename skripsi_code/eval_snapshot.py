import os
import sys
import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import ForgeryDataset
from models.seg_hrnet import HighResolutionNet
from models.seg_hrnet_config import get_cfg_defaults
from models.NLCDetection_loc import NLCDetection

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

def parse_metrics_csv(csv_path):
    columns = ['Epoch', 'AvgLoss', 'AvgBCELoss', 'Precision', 'Recall', 'F1_Score', 'IoU', 'Dice',
               'ValLoss', 'ValBCELoss', 'ValPrecision', 'ValRecall', 'ValF1_Score', 'ValIoU', 'ValDice']
    data = []
    with open(csv_path, 'r') as f:
        header = f.readline()
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if parts[0] == 'Epoch':
                continue
            try:
                row = {
                    'Epoch': float(parts[0]),
                    'AvgLoss': float(parts[1]),
                    'AvgBCELoss': float(parts[2]),
                    'Precision': float(parts[3]),
                    'Recall': float(parts[4]),
                    'F1_Score': float(parts[5]),
                    'IoU': float(parts[6]),
                    'Dice': float(parts[7]),
                }
                if len(parts) >= 15:
                    row['ValLoss'] = float(parts[8])
                    row['ValBCELoss'] = float(parts[9])
                    row['ValPrecision'] = float(parts[10])
                    row['ValRecall'] = float(parts[11])
                    row['ValF1_Score'] = float(parts[12])
                    row['ValIoU'] = float(parts[13])
                    row['ValDice'] = float(parts[14])
                data.append(row)
            except (ValueError, IndexError):
                continue
    return pd.DataFrame(data)

def find_weights_path(folder, epoch):
    candidate_dirs = [
        os.path.join(folder, 'models'),
        os.path.join(folder, 'weights'),
        folder
    ]
    fenet_names = [f'FENet_{epoch}.pth', f'FENet_{epoch}']
    segnet_names = [f'SegNet_{epoch}.pth', f'SegNet_{epoch}']
    
    fenet_path, segnet_path = None, None
    for d in candidate_dirs:
        for f_name in fenet_names:
            p = os.path.join(d, f_name)
            if os.path.exists(p):
                fenet_path = p
                break
        for s_name in segnet_names:
            p = os.path.join(d, s_name)
            if os.path.exists(p):
                segnet_path = p
                break
        if fenet_path and segnet_path:
            break
            
    return fenet_path, segnet_path

def evaluate_test_set(fenet_path, segnet_path, test_dataset, batch_size=8, device=None):
    if device is None:
        device = get_device()
    
    print(f"Loading weights:\n  FENet : {fenet_path}\n  SegNet: {segnet_path}")
    cfg = get_cfg_defaults()
    FENet = HighResolutionNet(cfg).to(device)
    SegNet = NLCDetection().to(device)
    
    FENet.load_state_dict(torch.load(fenet_path, map_location=device), strict=False)
    SegNet.load_state_dict(torch.load(segnet_path, map_location=device), strict=False)
    
    FENet.eval()
    SegNet.eval()
    
    dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    epsilon = 1e-7
    
    print(f"Running test evaluation on {len(test_dataset)} samples using device: {device}...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Test Evaluation"):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            features = FENet(images)
            _, mask_binary = SegNet(features, images)
            
            prob_mask = torch.sigmoid(mask_binary)
            pred_mask = (prob_mask > 0.5).float()
            
            tp = torch.sum((pred_mask == 1) & (masks == 1)).item()
            fp = torch.sum((pred_mask == 1) & (masks == 0)).item()
            fn = torch.sum((pred_mask == 0) & (masks == 1)).item()
            
            total_tp += tp
            total_fp += fp
            total_fn += fn
            
    precision = total_tp / (total_tp + total_fp + epsilon)
    recall = total_tp / (total_tp + total_fn + epsilon)
    f1_score = 2 * total_tp / (2 * total_tp + total_fp + total_fn + epsilon)
    iou = total_tp / (total_tp + total_fp + total_fn + epsilon)
    dice = f1_score
    
    return {
        'Precision': precision,
        'Recall': recall,
        'F1_Score': f1_score,
        'IoU': iou,
        'Dice': dice
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate best model score up to a given epoch and test set snapshot.")
    parser.add_argument('--folder', type=str, default='STGAN7k-noPreproc-epoch100-bs4-lr1e-3-721',
                        help="Target experiment folder name or path")
    parser.add_argument('--max_epoch', type=int, default=50,
                        help="Cutoff epoch to consider for the best model")
    parser.add_argument('--snapshot_epoch', type=int, default=50,
                        help="Snapshot epoch weight to load for test evaluation")
    parser.add_argument('--batch_size', type=int, default=8,
                        help="Batch size for test evaluation")
    parser.add_argument('--output_csv', type=str, default=None,
                        help="Path to save summary CSV (default: <folder>/best_model_summary_epoch{max_epoch}.csv)")
    args = parser.parse_args()

    # Resolve folder path
    folder = args.folder
    if not os.path.exists(folder):
        candidate = os.path.join('logs', folder)
        if os.path.exists(candidate):
            folder = candidate
        else:
            print(f"Error: Directory '{folder}' not found.")
            sys.exit(1)
            
    metrics_path = os.path.join(folder, 'metrics.csv')
    if not os.path.exists(metrics_path):
        print(f"Error: '{metrics_path}' not found.")
        sys.exit(1)
        
    # 1. Find best model up to max_epoch
    df = parse_metrics_csv(metrics_path)
    if df.empty:
        print("Error: metrics.csv contains no valid rows.")
        sys.exit(1)
        
    df_sub = df[df['Epoch'] <= args.max_epoch]
    if df_sub.empty:
        print(f"Error: No records found with Epoch <= {args.max_epoch}.")
        sys.exit(1)
        
    best_idx = df_sub['ValF1_Score'].idxmax()
    best_row = df_sub.loc[best_idx]
    best_epoch = int(best_row['Epoch'])
    
    print(f"\n==================================================")
    print(f"Experiment Folder : {folder}")
    print(f"Evaluation Window : Epoch 1 to {args.max_epoch}")
    print(f"Best Validation F1: {best_row['ValF1_Score']:.4f} at Epoch {best_epoch}")
    print(f"==================================================")
    
    # 2. Find snapshot weights
    fenet_path, segnet_path = find_weights_path(folder, args.snapshot_epoch)
    if not fenet_path or not segnet_path:
        print(f"Error: Snapshot weights for epoch {args.snapshot_epoch} not found in '{folder}'.")
        sys.exit(1)
        
    # 3. Setup Test Dataset
    test_mask_dir = 'datasets/STGAN_7k_split/test/masks'
    test_fake_dir = 'datasets/STGAN_7k_split/test/images'
    test_txt_dir = 'datasets/STGAN_7k_split/test/alllist.txt'
    
    if not (os.path.exists(test_mask_dir) and os.path.exists(test_fake_dir)):
        print(f"Error: Test dataset paths not found at '{test_mask_dir}' / '{test_fake_dir}'.")
        sys.exit(1)
        
    test_dataset = ForgeryDataset(
        mask_dir=test_mask_dir,
        fake_dir=test_fake_dir,
        txt_dir=test_txt_dir if os.path.exists(test_txt_dir) else None,
        crop_size=(256, 256),
        is_train=False
    )
    
    # 4. Evaluate Test Set using Snapshot Weights
    device = get_device()
    test_metrics = evaluate_test_set(
        fenet_path, segnet_path, test_dataset, batch_size=args.batch_size, device=device
    )
    
    # 5. Build and print summary table
    summary_data = {
        'Dataset': [f'Train (Epoch {best_epoch})', f'Validation (Epoch {best_epoch})', f'Test (Snapshot {args.snapshot_epoch})'],
        'Precision': [best_row['Precision'], best_row['ValPrecision'], test_metrics['Precision']],
        'Recall': [best_row['Recall'], best_row['ValRecall'], test_metrics['Recall']],
        'F1_Score': [best_row['F1_Score'], best_row['ValF1_Score'], test_metrics['F1_Score']],
        'IoU': [best_row['IoU'], best_row['ValIoU'], test_metrics['IoU']],
        'Dice': [best_row['Dice'], best_row['ValDice'], test_metrics['Dice']]
    }
    summary_df = pd.DataFrame(summary_data)
    
    print("\n" + "="*70)
    print(f"  MODEL SCORE COMPARISON AT POINT OF EPOCH {args.max_epoch}")
    print("="*70)
    print(summary_df.to_string(index=False))
    print("="*70 + "\n")
    
    # Also print Loss details for Train & Val
    print(f"Best Epoch {best_epoch} Loss Details:")
    print(f"  Train AvgLoss   : {best_row['AvgLoss']:.4f} | Train AvgBCELoss: {best_row['AvgBCELoss']:.4f}")
    print(f"  Val Loss        : {best_row['ValLoss']:.4f} | Val BCELoss    : {best_row['ValBCELoss']:.4f}")
    print("")

    # Save summary CSV
    out_csv = args.output_csv or os.path.join(folder, f'best_model_summary_epoch{args.max_epoch}.csv')
    summary_df.to_csv(out_csv, index=False)
    print(f"Summary CSV saved to: {out_csv}\n")

if __name__ == '__main__':
    main()
