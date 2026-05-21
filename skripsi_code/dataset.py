import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
import numpy as np
import torch

class ForgeryDataset(Dataset):
    def __init__(self, fake_dir, mask_dir, txt_dir=None, crop_size=(256, 256), invert_mask=None, is_train=False):
        super().__init__()
        self.fake_dir = fake_dir
        self.mask_dir = mask_dir
        self.crop_size = crop_size
        self.txt_dir = txt_dir
        self.is_train = is_train
        
        if invert_mask is None:
            self.invert_mask = 'NIST16' in fake_dir or 'nist16' in fake_dir.lower()
        else:
            self.invert_mask = invert_mask
        
        self.image_files = []
        
        if self.txt_dir and os.path.exists(self.txt_dir):
            txt_base_path = os.path.dirname(self.txt_dir)
            with open(self.txt_dir, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) >= 2:
                        img_path = os.path.join(txt_base_path, parts[0])
                        mask_path = os.path.join(txt_base_path, parts[1])
                        if os.path.exists(img_path) and os.path.exists(mask_path):
                            self.image_files.append((img_path, mask_path))
        else:
            valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
            for file in os.listdir(fake_dir):
                if file.lower().endswith(valid_extensions):
                    basename, ext = os.path.splitext(file)
                    
                    # Possible mask filenames to check
                    possible_mask_names = [
                        f"{basename}_label.jpg",
                        f"{basename}_label.png",
                        f"{basename}_label.jpeg",
                        f"{basename}_gt.jpg",
                        f"{basename}_gt.png",
                        f"{basename}_gt.jpeg",
                        f"{basename}_mask.png",
                        f"{basename}.jpg",
                        f"{basename}.png",
                        f"{basename}.jpeg",
                        f"mani_{basename}.png",
                        f"mani_{basename}.jpg",
                        file  
                    ]
                    
                    # COVERAGE specific rule: 100t.tif -> 100forged.tif
                    if basename.lower().endswith('t'):
                        possible_mask_names.append(f"{basename[:-1]}forged.tif")
                        possible_mask_names.append(f"{basename[:-1]}forged.png")
                    
                    mask_path = None
                    for mf in possible_mask_names:
                        candidate_path = os.path.join(mask_dir, mf)
                        if os.path.exists(candidate_path):
                            mask_path = candidate_path
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
        
        if self.is_train:
            import random
            import io
            
            # 1. Spatial Augmentations
            if random.random() > 0.5:
                image = F.hflip(image)
                mask = F.hflip(mask)
            if random.random() > 0.5:
                image = F.vflip(image)
                mask = F.vflip(mask)
                
            # 2. Compression Augmentation (Random JPEG Quality)
            if random.random() > 0.5:
                quality = random.randint(50, 100)
                buffer = io.BytesIO()
                image.save(buffer, format='JPEG', quality=quality)
                buffer.seek(0)
                image = Image.open(buffer).convert('RGB')
        
        # Convert to tensor. Image: [3,H,W] scaled to 0-1.
        image = F.to_tensor(image)
        # Normalize with mean/std usually used in pre-trained models
        image = F.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        # Mask to binary [0.0 or 1.0]
        mask = F.to_tensor(mask)    # [1, H, W]
        
        if self.invert_mask:
            mask = 1.0 - mask
            
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
        fake_dir='./Dataset Fix April/STGAN/fake',
        mask_dir='./Dataset Fix April/STGAN/mask'
    )
    print(f"Loaded {len(dataset)} samples")
    if len(dataset) > 0:
        sample = dataset[0]
        print("Image shape:", sample['image'].shape)
        print("Mask shape:", sample['mask'].shape)
