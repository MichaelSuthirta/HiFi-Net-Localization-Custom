import os
import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import ForgeryDataset
from models.seg_hrnet import HighResolutionNet, get_seg_model
from models.seg_hrnet_config import get_cfg_defaults
from models.NLCDetection_loc import NLCDetection
from utils.custom_loss import IsolatingLossFunction

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

def train():
    device = get_device()
    print(f"Using device: {device}")

    # 1. Dataset & DataLoader
    dataset = ForgeryDataset(
        # fake_dir='data-CASIA1/fake',
        # mask_dir='data-CASIA1/mask',
        # txt_dir='data-CASIA1/alllist.txt' if os.path.exists('data-NIST16/alllist.txt') else None

        fake_dir='data-NIST16/probe',
        mask_dir='data-NIST16/mask',
        txt_dir='data-NIST16/alllist.txt' if os.path.exists('data-NIST16/alllist.txt') else None
    )
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0, drop_last=True)

    if len(dataset) == 0:
        print("Dataset is empty. Exiting...")
        return

    # 2. Init Models
    print("Loading models...")
    cfg = get_cfg_defaults()
    FENet = get_seg_model(cfg).to(device)
    SegNet = NLCDetection().to(device)
    
    # 3. Setup Optimizers
    params = list(FENet.parameters()) + list(SegNet.parameters())
    optimizer = torch.optim.Adam(params, lr=1e-3)

    # 4. Setup Losses
    bce_loss_fn = nn.BCELoss()
    ce_loss_fn = nn.CrossEntropyLoss()

    use_isolating_loss = False
    center_path = 'center_loc/radius_center.pth'
    if os.path.exists(center_path):
        print(f"Loading precomputed center and radius from {center_path}")
        ckpt = torch.load(center_path, map_location=device)
        center = ckpt['center'].to(device)
        radius = ckpt['radius'].to(device)
        isolating_loss_fn = IsolatingLossFunction(center, radius)
        use_isolating_loss = True
    else:
        print("Precomputed center/radius not found, skipping Isolating Loss. (Using BCE only).")

    # 5. Training Loop
    epochs = 10
    
    # Initialize metric logging
    os.makedirs('logs', exist_ok=True)
    with open('logs/metrics.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'AvgLoss', 'AvgBCELoss', 'Precision', 'Recall', 'F1_Score', 'IoU', 'Dice'])

    for epoch in range(epochs):
        FENet.train()
        SegNet.train()
        
        running_loss = 0.0
        running_bce = 0.0
        
        total_tp = 0
        total_fp = 0
        total_fn = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)     # GT Mask
            labels = batch['cls'].to(device)     
            
            optimizer.zero_grad()
            
            # Forward pass
            features = FENet(images)
            mask_feat, mask_binary, cls_4, cls_3, cls_2, cls_1 = SegNet(features, images)
            
            # Compute Training Metrics (detached to save memory)
            with torch.no_grad():
                pred_mask = (mask_binary > 0.5).float()
                tp = torch.sum((pred_mask == 1) & (masks == 1)).item()
                fp = torch.sum((pred_mask == 1) & (masks == 0)).item()
                fn = torch.sum((pred_mask == 0) & (masks == 1)).item()
                total_tp += tp
                total_fp += fp
                total_fn += fn
            
            # Compute Losses
            loss_bce = bce_loss_fn(mask_binary, masks)
            
            if use_isolating_loss:
                loss_metric, mani_loss, nat_loss = isolating_loss_fn(mask_feat, masks)
            else:
                loss_metric = torch.tensor(0.0).to(device)
                
            loss_cls = ce_loss_fn(cls_1, labels) 
            
            loss = loss_bce + loss_metric + 1e-4 * loss_cls
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            running_bce += loss_bce.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'bce': f"{loss_bce.item():.4f}"})
            
        avg_loss = running_loss/len(dataloader)
        avg_bce = running_bce/len(dataloader)
        
        epsilon = 1e-7
        precision = total_tp / (total_tp + total_fp + epsilon)
        recall = total_tp / (total_tp + total_fn + epsilon)
        f1_score = 2 * total_tp / (2 * total_tp + total_fp + total_fn + epsilon)
        iou = total_tp / (total_tp + total_fp + total_fn + epsilon)
        dice = f1_score # Dice is mathematically equivalent to F1 score for binary classification
        
        print(f"Epoch {epoch+1} finished. Avg Loss: {avg_loss:.4f} | F1: {f1_score:.4f} | IoU: {iou:.4f}")
        
        # Log to CSV
        with open('logs/metrics.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, avg_loss, avg_bce, precision, recall, f1_score, iou, dice])

    print("Training finished! Saving weights...")
    os.makedirs('weights', exist_ok=True)
    torch.save(FENet.state_dict(), 'weights/FENet_latest.pth')
    torch.save(SegNet.state_dict(), 'weights/SegNet_latest.pth')
    print("Saved to weights/")

if __name__ == '__main__':
    train()
