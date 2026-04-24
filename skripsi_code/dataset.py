import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
import numpy as np
import torch

class ForgeryDataset(Dataset):
    def __init__(self, fake_dir, mask_dir, crop_size=(256, 256)):
        super().__init__()
        self.fake_dir = fake_dir
        self.mask_dir = mask_dir
        self.crop_size = crop_size
        
        self.image_files = []
        
        # We assume for each image like 1_10.jpg, there is a mask 1_10_label.jpg
        for file in os.listdir(fake_dir):
            if file.endswith(('.jpg', '.png', '.jpeg')):
                basename = file.split('.')[0]
                # Guess mask name: Some use 1_10_label.jpg or just 1_10.jpg or 1_10.png
                mask_file1 = f"{basename}_label.jpg"
                mask_file2 = f"{basename}_label.png"
                mask_file3 = file # Exactly same
                
                # Check which one exists
                mask_path = None
                for mf in [mask_file1, mask_file2, mask_file3]:
                    if os.path.exists(os.path.join(mask_dir, mf)):
                        mask_path = os.path.join(mask_dir, mf)
                        break
                
                if mask_path is not None:
                    self.image_files.append((os.path.join(fake_dir, file), mask_path))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path, mask_path = self.image_files[idx]
        
        # Load image (RGB)
        image = Image.open(img_path).convert('RGB')
        # Load mask (Grayscale)
        mask = Image.open(mask_path).convert('L')
        
        # Resize to typical model requirement (e.g. 256x256)
        image = image.resize(self.crop_size, Image.Resampling.BILINEAR)
        mask = mask.resize(self.crop_size, Image.Resampling.NEAREST)
        
        # Convert to tensor. Image: [3,H,W] scaled to 0-1.
        image = F.to_tensor(image)
        # Normalize with mean/std usually used in pre-trained models
        image = F.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        # Mask to binary [0.0 or 1.0]
        mask = F.to_tensor(mask)    # [1, H, W]
        mask = (mask > 0.5).float() # Thresholding
        
        # Squeeze mask dimension, usually cross_entropy/bce expects [H, W] or [1,H,W]
        # In author's code, BCE loss operates on masks with shape [b, 256, 256]
        mask = mask.squeeze(0)  # Now [256, 256]
        
        # Level 1 label. Since this is partial forgery, author's codebase maps partials from 1 to 7.
        # We will use 1 (Splice) as a dummy for our focused localization loss.
        # Authentic is 0. But for our data, it's forged.
        fake_cls = torch.tensor(1, dtype=torch.long)
        
        # The Custom loss center_radius_init function of author's expects: image, paths/masks, cls, fcls, scls, ...
        # But we will use a simpler structure if we rewrite the train loop.
        # We will return the dictionary.
        return {
            'image': image, 
            'mask': mask, 
            'cls': fake_cls
        }

if __name__ == '__main__':
    # Test dataloader
    dataset = ForgeryDataset(
        fake_dir='../dataset_small_sample/fake_small_sample',
        mask_dir='../dataset_small_sample/mask_small_sample'
    )
    print(f"Loaded {len(dataset)} samples")
    if len(dataset) > 0:
        sample = dataset[0]
        print("Image shape:", sample['image'].shape)
        print("Mask shape:", sample['mask'].shape)
