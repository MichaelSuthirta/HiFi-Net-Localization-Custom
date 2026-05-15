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

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # Flatten inputs and targets
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + self.smooth)/(inputs.sum() + targets.sum() + self.smooth)  
        
        return 1 - dice

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

        mask_dir='datasets/data_split_STGAN+COVERAGE/train/masks',
        fake_dir='datasets/data_split_STGAN+COVERAGE/train/images_compressed',
        txt_dir='datasets/data_split_STGAN+COVERAGE/train/train.txt' if os.path.exists('datasets/data_split_STGAN+COVERAGE/train/train.txt') else None
    )
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2, drop_last=True)

    val_dataset = ForgeryDataset(
        mask_dir='datasets/data_split_STGAN+COVERAGE/val/masks',
        fake_dir='datasets/data_split_STGAN+COVERAGE/val/images_compressed',
        txt_dir='datasets/data_split_STGAN+COVERAGE/val/val.txt' if os.path.exists('datasets/data_split_STGAN+COVERAGE/val/val.txt') else None
    )
    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2, drop_last=False)

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
    optimizer = torch.optim.Adam(params, lr=1e-4)

    # 4. Setup Losses
    bce_loss_fn = nn.BCELoss()
    dice_loss_fn = DiceLoss()
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
    epochs = 50
    start_epoch = 0
    checkpoint_path = 'weights/checkpoint.pth'
    best_val_f1 = 0.0

    # Load checkpoint jika ada
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        FENet.load_state_dict(checkpoint['FENet_state_dict'])
        SegNet.load_state_dict(checkpoint['SegNet_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        if 'best_val_f1' in checkpoint:
            best_val_f1 = checkpoint['best_val_f1']
        print(f"Resuming training from epoch {start_epoch + 1}")
    else:
        print("No checkpoint found. Starting from scratch.")
    
    # Initialize metric logging
    os.makedirs('logs', exist_ok=True)
    if start_epoch == 0:
        with open('logs/metrics.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'AvgLoss', 'AvgBCELoss', 'Precision', 'Recall', 'F1_Score', 'IoU', 'Dice',
                             'ValLoss', 'ValBCELoss', 'ValPrecision', 'ValRecall', 'ValF1_Score', 'ValIoU', 'ValDice'])

    for epoch in range(start_epoch, epochs):
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
            loss_bce_only = bce_loss_fn(mask_binary, masks)
            loss_dice = dice_loss_fn(mask_binary, masks)
            
            # BCE + Dice Loss mengatasi class imbalance
            loss_bce = loss_bce_only + loss_dice
            
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
        
        print(f"Epoch {epoch+1} Train finished. Avg Loss: {avg_loss:.4f} | F1: {f1_score:.4f} | IoU: {iou:.4f}")
        
        # --- Validation Loop ---
        FENet.eval()
        SegNet.eval()
        
        val_running_loss = 0.0
        val_running_bce = 0.0
        
        val_total_tp = 0
        val_total_fp = 0
        val_total_fn = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_dataloader, desc=f"Val Epoch {epoch+1}/{epochs}")
            for batch in val_pbar:
                images = batch['image'].to(device)
                masks = batch['mask'].to(device)     
                labels = batch['cls'].to(device)     
                
                features = FENet(images)
                mask_feat, mask_binary, cls_4, cls_3, cls_2, cls_1 = SegNet(features, images)
                
                pred_mask = (mask_binary > 0.5).float()
                tp = torch.sum((pred_mask == 1) & (masks == 1)).item()
                fp = torch.sum((pred_mask == 1) & (masks == 0)).item()
                fn = torch.sum((pred_mask == 0) & (masks == 1)).item()
                val_total_tp += tp
                val_total_fp += fp
                val_total_fn += fn
                
                loss_bce_only = bce_loss_fn(mask_binary, masks)
                loss_dice = dice_loss_fn(mask_binary, masks)
                loss_bce = loss_bce_only + loss_dice
                
                if use_isolating_loss:
                    loss_metric, mani_loss, nat_loss = isolating_loss_fn(mask_feat, masks)
                else:
                    loss_metric = torch.tensor(0.0).to(device)
                    
                loss_cls = ce_loss_fn(cls_1, labels) 
                loss = loss_bce + loss_metric + 1e-4 * loss_cls
                
                val_running_loss += loss.item()
                val_running_bce += loss_bce.item()
                val_pbar.set_postfix({'val_loss': f"{loss.item():.4f}"})
                
        val_avg_loss = val_running_loss / len(val_dataloader) if len(val_dataloader) > 0 else 0
        val_avg_bce = val_running_bce / len(val_dataloader) if len(val_dataloader) > 0 else 0
        
        val_precision = val_total_tp / (val_total_tp + val_total_fp + epsilon)
        val_recall = val_total_tp / (val_total_tp + val_total_fn + epsilon)
        val_f1_score = 2 * val_total_tp / (2 * val_total_tp + val_total_fp + val_total_fn + epsilon)
        val_iou = val_total_tp / (val_total_tp + val_total_fp + val_total_fn + epsilon)
        val_dice = val_f1_score
        
        print(f"Epoch {epoch+1} Val finished. Avg Loss: {val_avg_loss:.4f} | F1: {val_f1_score:.4f} | IoU: {val_iou:.4f}")

        # Log to CSV
        with open('logs/metrics.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, avg_loss, avg_bce, precision, recall, f1_score, iou, dice,
                             val_avg_loss, val_avg_bce, val_precision, val_recall, val_f1_score, val_iou, val_dice])

        # Save checkpoint tiap akhir epoch
        os.makedirs('weights', exist_ok=True)
        torch.save({
            'epoch': epoch,
            'FENet_state_dict': FENet.state_dict(),
            'SegNet_state_dict': SegNet.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'best_val_f1': best_val_f1
        }, checkpoint_path)
        
        if val_f1_score > best_val_f1:
            best_val_f1 = val_f1_score
            torch.save(FENet.state_dict(), 'weights/FENet_best.pth')
            torch.save(SegNet.state_dict(), 'weights/SegNet_best.pth')
            print(f"*** New Best Model Saved (Val F1: {best_val_f1:.4f}) ***")
            
        print(f"Checkpoint saved at epoch {epoch+1}")
    print("Training finished! Saving weights...")
    os.makedirs('weights', exist_ok=True)
    torch.save(FENet.state_dict(), 'weights/FENet_latest.pth')
    torch.save(SegNet.state_dict(), 'weights/SegNet_latest.pth')
    print("Saved to weights/")

if __name__ == '__main__':
    train()
