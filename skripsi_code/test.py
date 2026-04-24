import os
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset import ForgeryDataset
from models.seg_hrnet import HighResolutionNet
from models.seg_hrnet_config import get_cfg_defaults
from models.NLCDetection_loc import NLCDetection
from train import get_device

def evaluate_and_visualize():
    device = get_device()
    print(f"Using device for testing: {device}")

    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)

    # 1. Dataset & DataLoader (We use the same sample folders for demo purposes)
    dataset = ForgeryDataset(
        fake_dir='data-CASIA1/fake',
        mask_dir='data-CASIA1/mask'
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    if len(dataset) == 0:
        print("Dataset is empty. Exiting...")
        return

    # 2. Init & Load Models
    print("Loading trained models...")
    cfg = get_cfg_defaults()
    FENet = HighResolutionNet(cfg).to(device)
    SegNet = NLCDetection().to(device)
    
    if os.path.exists('weights/FENet_latest.pth') and os.path.exists('weights/SegNet_latest.pth'):
        FENet.load_state_dict(torch.load('weights/FENet_latest.pth', map_location=device))
        SegNet.load_state_dict(torch.load('weights/SegNet_latest.pth', map_location=device))
        print("Successfully loaded trained weights.")
    else:
        print("Warning: Trained weights not found. Using untrained models.")

    FENet.eval()
    SegNet.eval()

    print(f"Starting evaluation... Saving visualizations to {results_dir}/")
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Evaluasi & Visualisasi")):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            # Forward pass
            features = FENet(images)
            mask_feat, mask_binary, cls_4, cls_3, cls_2, cls_1 = SegNet(features, images)
            
            # mask_binary is [B, 256, 256] from sigmoid layer. Threshold it > 0.5
            pred_mask = (mask_binary > 0.2).float()

            # Move to CPU for visualization
            img_disp = images[0].cpu().permute(1, 2, 0).numpy()
            gt_disp = masks[0].cpu().numpy()
            pred_disp = pred_mask[0].cpu().numpy()

            # Denormalize image for better display
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            img_disp = (img_disp * std + mean).clip(0, 1)

            # Create side-by-side plot with Soft Mask
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            
            axes[0].imshow(img_disp)
            axes[0].set_title('Fake Original Image')
            axes[0].axis('off')

            axes[1].imshow(gt_disp, cmap='gray')
            axes[1].set_title('Ground Truth Mask')
            axes[1].axis('off')

            # Render the raw probability before thresholding 
            # This helps debug if the model is learning anything or stuck at ~0.3
            axes[2].imshow(mask_binary[0].detach().cpu().numpy(), cmap='inferno')
            axes[2].set_title('Raw Soft Mask (Heatmap)')
            axes[2].axis('off')

            axes[3].imshow(pred_disp, cmap='gray')
            axes[3].set_title('Thresholded Mask (>0.5)')
            axes[3].axis('off')
            
            plt.tight_layout()
            save_path = os.path.join(results_dir, f'result_sample_{i+1}.png')
            plt.savefig(save_path)
            plt.close(fig)

    print("Evaluation completed!")

if __name__ == '__main__':
    evaluate_and_visualize()
