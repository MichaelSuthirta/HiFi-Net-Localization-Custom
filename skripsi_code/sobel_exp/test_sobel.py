import os
import sys
import csv
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
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
from models.seg_hrnet_sobel import HighResolutionNetSobel
from train_sobel import get_device

def evaluate_and_visualize():
    device = get_device()
    print(f"Using device for testing: {device}")

    results_dir = os.path.join(CURRENT_DIR, 'results')
    os.makedirs(results_dir, exist_ok=True)

    dataset = ForgeryDataset(
        mask_dir=os.path.join(BASE_DIR, 'datasets/STGAN_7k_split/test/masks'),
        fake_dir=os.path.join(BASE_DIR, 'datasets/STGAN_7k_split/test/images'),
        txt_dir=os.path.join(BASE_DIR, 'datasets/STGAN_7k_split/test/alllist.txt'),
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    if len(dataset) == 0:
        print("Dataset is empty. Exiting...")
        return

    print("Loading trained Sobel models...")
    cfg = get_cfg_defaults()
    FENet = HighResolutionNetSobel(cfg).to(device)
    SegNet = NLCDetection().to(device)
    
    fenet_candidates = [
        os.path.join(CURRENT_DIR, 'weights', 'FENet_sobel_best.pth'),
        os.path.join(CURRENT_DIR, 'weights', 'FENet_best.pth'),
        os.path.join(BASE_DIR, 'weights', 'FENet_sobel_best.pth'),
    ]
    fenet_weight = next((p for p in fenet_candidates if os.path.exists(p)), None)

    segnet_candidates = [
        os.path.join(CURRENT_DIR, 'weights', 'SegNet_sobel_best.pth'),
        os.path.join(CURRENT_DIR, 'weights', 'SegNet_best.pth'),
        os.path.join(BASE_DIR, 'weights', 'SegNet_sobel_best.pth'),
    ]
    segnet_weight = next((p for p in segnet_candidates if os.path.exists(p)), None)

    if fenet_weight and segnet_weight:
        FENet.load_state_dict(torch.load(fenet_weight, map_location=device), strict=False)
        SegNet.load_state_dict(torch.load(segnet_weight, map_location=device), strict=False)
        print(f"Successfully loaded weights:\n  FENet : {fenet_weight}\n  SegNet: {segnet_weight}")
    else:
        print("Warning: Trained weights not found. Using untrained models.")

    FENet.eval()
    SegNet.eval()

    print(f"Starting evaluation... Saving visualizations to {results_dir}/")
    
    total_tp = 0
    total_fp = 0
    total_fn = 0

    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Sobel Evaluasi & Visualisasi")):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            features = FENet(images)
            mask_feat, mask_binary = SegNet(features, images)
            
            prob_mask = torch.sigmoid(mask_binary)
            pred_mask = (prob_mask > 0.5).float()
            
            tp = torch.sum((pred_mask == 1) & (masks == 1)).item()
            fp = torch.sum((pred_mask == 1) & (masks == 0)).item()
            fn = torch.sum((pred_mask == 0) & (masks == 1)).item()
            
            total_tp += tp
            total_fp += fp
            total_fn += fn

            if i < 100:
                img_disp = images[0].cpu().permute(1, 2, 0).numpy()
                gt_disp = masks[0].cpu().numpy()
                pred_disp = pred_mask[0].cpu().numpy()

                mean = [0.485, 0.456, 0.406]
                std = [0.229, 0.224, 0.225]
                img_disp = (img_disp * std + mean).clip(0, 1)

                overlay = img_disp.copy()
                gt_disp_2d = gt_disp.squeeze() if gt_disp.ndim == 3 else gt_disp
                pred_disp_2d = pred_disp.squeeze() if pred_disp.ndim == 3 else pred_disp
                
                tp_mask = (gt_disp_2d == 1) & (pred_disp_2d == 1)
                fp_mask = (gt_disp_2d == 0) & (pred_disp_2d == 1)
                fn_mask = (gt_disp_2d == 1) & (pred_disp_2d == 0)
                
                overlay[tp_mask] = [0, 1, 0] # Hijau: True Positive
                overlay[fp_mask] = [1, 0, 0] # Merah: False Positive
                overlay[fn_mask] = [0, 0, 1] # Biru: False Negative
                
                alpha = 0.5
                blended_overlay = (1 - alpha) * img_disp + alpha * overlay
                
                fig, axes = plt.subplots(1, 5, figsize=(20, 4))
                axes[0].imshow(img_disp)
                axes[0].set_title('Fake Original Image')
                axes[0].axis('off')

                axes[1].imshow(gt_disp_2d, cmap='gray')
                axes[1].set_title('Ground Truth Mask')
                axes[1].axis('off')

                axes[2].imshow(prob_mask[0].detach().cpu().squeeze().numpy(), cmap='inferno')
                axes[2].set_title('Raw Soft Mask (Heatmap)')
                axes[2].axis('off')

                axes[3].imshow(pred_disp_2d, cmap='gray')
                axes[3].set_title('Thresholded Mask (>0.5)')
                axes[3].axis('off')
                
                axes[4].imshow(blended_overlay)
                axes[4].set_title('Overlay (TP:G, FP:R, FN:B)')
                axes[4].axis('off')
                
                plt.tight_layout()
                save_path = os.path.join(results_dir, f'result_sample_{i+1}.png')
                plt.savefig(save_path)
                plt.close(fig)

    print("Evaluation completed!")
    
    epsilon = 1e-7
    precision = total_tp / (total_tp + total_fp + epsilon)
    recall = total_tp / (total_tp + total_fn + epsilon)
    f1_score = 2 * total_tp / (2 * total_tp + total_fp + total_fn + epsilon)
    iou = total_tp / (total_tp + total_fp + total_fn + epsilon)
    dice = f1_score
    
    print("\n" + "="*30)
    print("   SOBEL EVALUATION METRICS")
    print("="*30)
    print(f"Global Precision: {precision:.4f}")
    print(f"Global Recall   : {recall:.4f}")
    print(f"Global F1-Score : {f1_score:.4f}")
    print(f"Global IoU      : {iou:.4f}")
    print(f"Global Dice     : {dice:.4f}")
    print("="*30 + "\n")

    logs_dir = os.path.join(CURRENT_DIR, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    with open(os.path.join(logs_dir, 'test_metrics.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Precision', 'Recall', 'F1_Score', 'IoU', 'Dice'])
        writer.writerow([precision, recall, f1_score, iou, dice])

if __name__ == '__main__':
    evaluate_and_visualize()
