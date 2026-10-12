import os
import sys
import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from dataset import ForgeryDataset
from models.seg_hrnet_config import get_cfg_defaults
from models.NLCDetection_loc import NLCDetection
from models.seg_hrnet_sobel import get_seg_model
from utils.custom_loss import IsolatingLossFunction

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)  
        return 1 - dice

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

def train():
    torch.cuda.empty_cache()
    device = get_device()
    print(f"Using device: {device}")

    # 1. Dataset & DataLoader
    import re
    from collections import Counter
    from torch.utils.data import WeightedRandomSampler

    dataset = ForgeryDataset(
        mask_dir=os.path.join(BASE_DIR, 'datasets/STGAN_7k_split/train/masks'),
        fake_dir=os.path.join(BASE_DIR, 'datasets/STGAN_7k_split/train/images'),
        txt_dir=os.path.join(BASE_DIR, 'datasets/STGAN_7k_split/train/alllist.txt'),
        is_train=True
    )

    print("Calculating dataset weights for balancing...")
    dataset_counts = Counter()
    dataset_labels = []
    
    for img_path, _ in dataset.image_files:
        basename = os.path.basename(img_path)
        match = re.match(r'(.+)_\d+\.', basename)
        if match:
            ds_name = match.group(1)
        else:
            ds_name = basename.split('_')[0]
        dataset_labels.append(ds_name)
        dataset_counts[ds_name] += 1
        
    weights = []
    for ds_name in dataset_labels:
        weights.append(1.0 / dataset_counts[ds_name])
        
    weights = torch.DoubleTensor(weights)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    dataloader = DataLoader(dataset, batch_size=4, sampler=sampler, num_workers=2, drop_last=True)

    val_dataset = ForgeryDataset(
        mask_dir=os.path.join(BASE_DIR, 'datasets/STGAN_7k_split/val/masks'),
        fake_dir=os.path.join(BASE_DIR, 'datasets/STGAN_7k_split/val/images'),
        txt_dir=os.path.join(BASE_DIR, 'datasets/STGAN_7k_split/val/alllist.txt')
    )
    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2, drop_last=False)

    if len(dataset) == 0:
        print("Dataset is empty. Exiting...")
        return

    # 2. Init Models (FENet with Sobel Kernel + SegNet Baseline)
    print("Loading models (FENet with Sobel + SegNet NLCDetection)...")
    cfg = get_cfg_defaults()
    FENet = get_seg_model(cfg).to(device)
    SegNet = NLCDetection().to(device)
    
    fenet_init = os.path.join(BASE_DIR, 'weights/FENet_modal_awal.pth')
    segnet_init = os.path.join(BASE_DIR, 'weights/SegNet_modal_awal.pth')
    if os.path.exists(fenet_init) and os.path.exists(segnet_init):
        FENet.load_state_dict(torch.load(fenet_init, map_location=device), strict=False)
        print("FENet weights loaded (Sobel frequency branch).")
        SegNet.load_state_dict(torch.load(segnet_init, map_location=device), strict=False)
        print("SegNet weights loaded.")
    else:
        print("Warning: Modal awal weights not found. Training from scratch.")

    # 3. Setup Optimizers
    params = list(FENet.parameters()) + list(SegNet.parameters())
    optimizer = torch.optim.Adam(params, lr=1e-4, weight_decay=1e-5)

    # 4. Setup Losses
    bce_loss_fn = nn.BCEWithLogitsLoss()
    dice_loss_fn = DiceLoss()

    use_isolating_loss = False
    center_path = os.path.join(BASE_DIR, 'center_loc/radius_center.pth')
    if use_isolating_loss and os.path.exists(center_path):
        print(f"Loading precomputed center and radius from {center_path}")
        ckpt = torch.load(center_path, map_location=device)
        center = ckpt['center'].to(device)
        radius = ckpt['radius'].to(device)
        isolating_loss_fn = IsolatingLossFunction(center, radius)
    else:
        print("Isolating Loss is disabled. Training with BCE + Dice Loss.")

    # 5. Training Loop
    weights_dir = os.path.join(CURRENT_DIR, 'weights')
    logs_dir = os.path.join(CURRENT_DIR, 'logs')
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    epochs = 50
    start_epoch = 0
    checkpoint_path = os.path.join(weights_dir, 'checkpoint_sobel.pth')
    best_val_f1 = 0.0
    accumulation_steps = 1

    # Load checkpoint jika ada
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        FENet.load_state_dict(checkpoint['FENet_state_dict'], strict=False)
        SegNet.load_state_dict(checkpoint['SegNet_state_dict'], strict=False)
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except Exception:
            print("Warning: Could not load optimizer state dict.")
        start_epoch = checkpoint['epoch'] + 1
        if 'best_val_f1' in checkpoint:
            best_val_f1 = checkpoint['best_val_f1']
        print(f"Resuming Sobel training from epoch {start_epoch + 1}")
    else:
        print("No Sobel checkpoint found. Starting fresh from epoch 1.")
    
    metrics_csv = os.path.join(logs_dir, 'metrics.csv')
    if start_epoch == 0:
        with open(metrics_csv, 'w', newline='') as f:
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
        
        optimizer.zero_grad()
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Sobel Epoch {epoch+1}/{epochs}")
        for i, batch in pbar:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device) 
            labels = batch['cls'].to(device)     
            
            # Forward pass
            features = FENet(images)
            mask_feat, mask_binary = SegNet(features, images)
            
            prob_mask = torch.sigmoid(mask_binary)
            
            # Compute Losses
            loss_bce_only = bce_loss_fn(mask_binary, masks)
            loss_dice = dice_loss_fn(prob_mask, masks)
            loss_bce = loss_bce_only + loss_dice
            
            # Compute Training Metrics
            with torch.no_grad():
                pred_mask = (prob_mask > 0.5).float()
                tp = torch.sum((pred_mask == 1) & (masks == 1)).item()
                fp = torch.sum((pred_mask == 1) & (masks == 0)).item()
                fn = torch.sum((pred_mask == 0) & (masks == 1)).item()
                total_tp += tp
                total_fp += fp
                total_fn += fn
            
            if use_isolating_loss:
                loss_metric, mani_loss, nat_loss = isolating_loss_fn(mask_feat, masks)
            else:
                loss_metric = torch.tensor(0.0).to(device)
                
            loss = loss_bce + loss_metric
            loss = loss / accumulation_steps
            loss.backward()
            
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(dataloader):
                optimizer.step()
                optimizer.zero_grad()
            
            running_loss += (loss.item() * accumulation_steps)
            running_bce += loss_bce.item()
            pbar.set_postfix({'loss': f"{(loss.item() * accumulation_steps):.4f}", 'bce': f"{loss_bce.item():.4f}"})
            
        avg_loss = running_loss / len(dataloader)
        avg_bce = running_bce / len(dataloader)
        
        epsilon = 1e-7
        precision = total_tp / (total_tp + total_fp + epsilon)
        recall = total_tp / (total_tp + total_fn + epsilon)
        f1_score = 2 * total_tp / (2 * total_tp + total_fp + total_fn + epsilon)
        iou = total_tp / (total_tp + total_fp + total_fn + epsilon)
        dice = f1_score
        
        print(f"Sobel Epoch {epoch+1} Train. Avg Loss: {avg_loss:.4f} | F1: {f1_score:.4f} | IoU: {iou:.4f}")
        
        # --- Validation Loop ---
        FENet.eval()
        SegNet.eval()
        
        val_running_loss = 0.0
        val_running_bce = 0.0
        
        val_total_tp = 0
        val_total_fp = 0
        val_total_fn = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_dataloader, desc=f"Sobel Val Epoch {epoch+1}/{epochs}")
            for batch in val_pbar:
                images = batch['image'].to(device)
                masks = batch['mask'].to(device)     
                labels = batch['cls'].to(device)     
                
                features = FENet(images)
                mask_feat, mask_binary = SegNet(features, images)
                
                prob_mask = torch.sigmoid(mask_binary)
                pred_mask = (prob_mask > 0.5).float()
                tp = torch.sum((pred_mask == 1) & (masks == 1)).item()
                fp = torch.sum((pred_mask == 1) & (masks == 0)).item()
                fn = torch.sum((pred_mask == 0) & (masks == 1)).item()
                val_total_tp += tp
                val_total_fp += fp
                val_total_fn += fn
                
                loss_bce_only = bce_loss_fn(mask_binary, masks)
                loss_dice = dice_loss_fn(prob_mask, masks)
                loss_bce = loss_bce_only + loss_dice 
                
                if use_isolating_loss:
                    loss_metric, mani_loss, nat_loss = isolating_loss_fn(mask_feat, masks)
                else:
                    loss_metric = torch.tensor(0.0).to(device)
                    
                loss = loss_bce + loss_metric
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
        
        print(f"Sobel Epoch {epoch+1} Val. Avg Loss: {val_avg_loss:.4f} | F1: {val_f1_score:.4f} | IoU: {val_iou:.4f}")

        # Log to CSV
        with open(metrics_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, avg_loss, avg_bce, precision, recall, f1_score, iou, dice,
                             val_avg_loss, val_avg_bce, val_precision, val_recall, val_f1_score, val_iou, val_dice])

        # Save checkpoint
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
            torch.save(FENet.state_dict(), os.path.join(weights_dir, 'FENet_sobel_best.pth'))
            torch.save(SegNet.state_dict(), os.path.join(weights_dir, 'SegNet_sobel_best.pth'))
            print(f"*** New Best Model Saved (Val F1: {best_val_f1:.4f}) ***")
            
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current Learning Rate: {current_lr}")
        
        if (epoch+1) % 25 == 0:
            import shutil
            best_fenet = os.path.join(weights_dir, 'FENet_sobel_best.pth')
            best_segnet = os.path.join(weights_dir, 'SegNet_sobel_best.pth')
            if os.path.exists(best_fenet) and os.path.exists(best_segnet):
                shutil.copy(best_fenet, os.path.join(weights_dir, f'FENet_sobel_{epoch+1}.pth'))
                shutil.copy(best_segnet, os.path.join(weights_dir, f'SegNet_sobel_{epoch+1}.pth'))
                print(f"*** Best Sobel Model up to Epoch {epoch+1} Saved as Snapshot ***")
            else:
                torch.save(FENet.state_dict(), os.path.join(weights_dir, f'FENet_sobel_{epoch+1}.pth'))
                torch.save(SegNet.state_dict(), os.path.join(weights_dir, f'SegNet_sobel_{epoch+1}.pth'))
                print(f"*** Current Sobel Model Saved at Epoch {epoch+1} ***")
        
        print(f"Sobel Checkpoint saved at epoch {epoch+1}")
        
    print("Training finished! Saving latest weights...")
    torch.save(FENet.state_dict(), os.path.join(weights_dir, 'FENet_sobel_latest.pth'))
    torch.save(SegNet.state_dict(), os.path.join(weights_dir, 'SegNet_sobel_latest.pth'))
    print(f"Saved latest to {weights_dir}/")

if __name__ == '__main__':
    train()
