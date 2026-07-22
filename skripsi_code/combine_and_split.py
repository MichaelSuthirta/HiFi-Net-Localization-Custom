import os
import shutil
import random
import uuid
from tqdm import tqdm

def find_pairs_generic(img_dir, mask_dir):
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    pairs = []
    if not os.path.exists(img_dir) or not os.path.exists(mask_dir):
        return pairs
        
    for file in os.listdir(img_dir):
        if file.lower().endswith(valid_extensions):
            basename, ext = os.path.splitext(file)
            
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
                file  
            ]
            
            if basename.lower().endswith('t'):
                possible_mask_names.append(f"{basename[:-1]}forged.tif")
                possible_mask_names.append(f"{basename[:-1]}forged.png")
                possible_mask_names.append(f"{basename[:-1]}paste.tif")
                possible_mask_names.append(f"{basename[:-1]}paste.png")
                
            mask_path = None
            for mf in possible_mask_names:
                candidate = os.path.join(mask_dir, mf)
                if os.path.exists(candidate):
                    mask_path = candidate
                    break
            
            if mask_path is not None:
                pairs.append((os.path.join(img_dir, file), mask_path))
    return pairs

def find_pairs_txt(txt_path):
    pairs = []
    if not os.path.exists(txt_path):
        return pairs
    base_dir = os.path.dirname(txt_path)
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split()
            if len(parts) >= 2:
                img = os.path.join(base_dir, parts[0])
                mask = os.path.join(base_dir, parts[1])
                if os.path.exists(img) and os.path.exists(mask):
                    pairs.append((img, mask))
    return pairs

def main():
    random.seed(42)
    MAX_SAMPLES = 300
    
    datasets = [
        {"name": "CASIA1", "type": "generic", "img": "datasets/data_combined_raw/CASIA/CASIA1/fake", "mask": "datasets/data_combined_raw/CASIA/CASIA1/mask"},
        {"name": "CASIA2", "type": "generic", "img": "datasets/data_combined_raw/CASIA/CASIA2/fake", "mask": "datasets/data_combined_raw/CASIA/CASIA2/mask"},
        {"name": "Coverage", "type": "generic", "img": "datasets/data_combined_raw/Coverage/image", "mask": "datasets/data_combined_raw/Coverage/mask"},
        {"name": "IMD2020", "type": "generic", "img": "datasets/data_combined_raw/IMD2020/fake_img", "mask": "datasets/data_combined_raw/IMD2020/mask"},
        {"name": "Columbia", "type": "generic", "img": "datasets/data_combined_raw/columbia/4cam_splc", "mask": "datasets/data_combined_raw/columbia/mask"},
        {"name": "NIST16", "type": "txt", "txt": "datasets/data_combined_raw/NIST16/alllist.txt"}
    ]
    
    all_pairs = []
    
    for ds in datasets:
        if ds["type"] == "generic":
            pairs = find_pairs_generic(ds["img"], ds["mask"])
        else:
            pairs = find_pairs_txt(ds["txt"])
            
        print(f"{ds['name']}: Found {len(pairs)} pairs")
        
        # Balance
        if len(pairs) > MAX_SAMPLES:
            pairs = random.sample(pairs, MAX_SAMPLES)
            
        print(f"{ds['name']}: Sampled {len(pairs)} pairs")
        all_pairs.extend(pairs)
        
    print(f"\nTotal pairs across all datasets: {len(all_pairs)}")
    
    random.shuffle(all_pairs)
    
    # Split
    total = len(all_pairs)
    train_end = int(total * 0.8)
    val_end = train_end + int(total * 0.1)
    
    splits = {
        'train': all_pairs[:train_end],
        'val': all_pairs[train_end:val_end],
        'test': all_pairs[val_end:]
    }
    
    out_dir = "datasets/data_split_combined"
    
    for split_name, pairs in splits.items():
        print(f"\nProcessing {split_name} ({len(pairs)} pairs)...")
        split_img_dir = os.path.join(out_dir, split_name, 'images')
        split_mask_dir = os.path.join(out_dir, split_name, 'masks')
        
        os.makedirs(split_img_dir, exist_ok=True)
        os.makedirs(split_mask_dir, exist_ok=True)
        
        txt_out_path = os.path.join(out_dir, split_name, 'alllist.txt')
        
        with open(txt_out_path, 'w') as f_txt:
            for src_img, src_mask in tqdm(pairs):
                # Ensure unique filenames in case of collision
                uid = uuid.uuid4().hex[:6]
                
                img_name = f"{uid}_{os.path.basename(src_img)}"
                mask_name = f"{uid}_{os.path.basename(src_mask)}"
                
                dst_img = os.path.join(split_img_dir, img_name)
                dst_mask = os.path.join(split_mask_dir, mask_name)
                
                shutil.copy2(src_img, dst_img)
                
                # Invert NIST16 mask
                if 'NIST16' in src_mask:
                    from PIL import Image, ImageOps
                    mask_img = Image.open(src_mask).convert('L')
                    inverted_mask = ImageOps.invert(mask_img)
                    inverted_mask.save(dst_mask)
                else:
                    shutil.copy2(src_mask, dst_mask)
                
                # Write to alllist.txt (relative paths)
                # We use the names relative to the directory where txt_out_path resides
                f_txt.write(f"images/{img_name} masks/{mask_name}\n")

if __name__ == "__main__":
    main()
