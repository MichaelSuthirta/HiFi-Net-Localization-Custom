import os
import shutil
import random
from tqdm import tqdm

def find_pairs_generic(img_dir, mask_dir):
    pairs = []
    if not os.path.exists(img_dir) or not os.path.exists(mask_dir):
        return pairs
    valid_exts = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
    
    for file in os.listdir(img_dir):
        if any(file.lower().endswith(ext) for ext in valid_exts):
            basename, _ = os.path.splitext(file)
            possible_mask_names = [
                f"{basename}_mask.png",
                f"{basename}_mask.jpg",
                f"{basename}_gt.png",
                f"{basename}_gt.jpg",
                f"{basename}.png",
                f"{basename}.jpg",
                f"{basename}_label.png",
                f"{basename}forged.tif",
                f"{basename.replace('t', '')}forged.tif",
                file
            ]
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

def split_data(pairs, train_ratio=0.8, val_ratio=0.1):
    random.shuffle(pairs)
    n = len(pairs)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return {
        'train': pairs[:n_train],
        'val': pairs[n_train:n_train+n_val],
        'test': pairs[n_train+n_val:]
    }

def main():
    random.seed(42)
    
    # 1. Tentukan sumber dataset
    datasets_info = {
        "CASIA1": {"type": "generic", "img": "datasets/data_combined_raw/CASIA/CASIA1/fake", "mask": "datasets/data_combined_raw/CASIA/CASIA1/mask"},
        "Columbia": {"type": "generic", "img": "datasets/data_combined_raw/columbia/4cam_splc", "mask": "datasets/data_combined_raw/columbia/mask"},
        "CASIA2": {"type": "generic", "img": "datasets/data_combined_raw/CASIA/CASIA2/fake", "mask": "datasets/data_combined_raw/CASIA/CASIA2/mask"},
        "NIST16": {"type": "txt", "txt": "datasets/data_combined_raw/NIST16/alllist.txt"},
        "IMD2020": {"type": "generic", "img": "datasets/data_combined_raw/IMD2020/fake_img", "mask": "datasets/data_combined_raw/IMD2020/mask"},
        "Coverage": {"type": "generic", "img": "datasets/data_combined_raw/Coverage/image", "mask": "datasets/data_combined_raw/Coverage/mask"}
    }
    
    for name, ds in datasets_info.items():
        print(f"\n--- Memproses {name} ---")
        if ds["type"] == "generic":
            pairs = find_pairs_generic(ds["img"], ds["mask"])
        else:
            pairs = find_pairs_txt(ds["txt"])
            
        print(f"Ditemukan {len(pairs)} pasangan gambar. Memulai proses split...")
        splits = split_data(pairs)
        
        base_out_dir = os.path.join("datasets", "individual", name)
        os.makedirs(base_out_dir, exist_ok=True)
        
        for split_name, split_pairs in splits.items():
            split_img_dir = os.path.join(base_out_dir, split_name, "images")
            split_mask_dir = os.path.join(base_out_dir, split_name, "masks")
            os.makedirs(split_img_dir, exist_ok=True)
            os.makedirs(split_mask_dir, exist_ok=True)
            
            txt_out_path = os.path.join(base_out_dir, split_name, "alllist.txt")
            
            with open(txt_out_path, 'w') as f:
                for src_img, src_mask in tqdm(split_pairs, desc=f"Menyalin {split_name}"):
                    # Gunakan hashing agar file yang sama selalu punya nama yang sama
                    import hashlib
                    path_hash = hashlib.md5(src_img.encode('utf-8')).hexdigest()[:8]
                    
                    img_name = f"{path_hash}_{os.path.basename(src_img)}"
                    mask_name = f"{path_hash}_{os.path.basename(src_mask)}"
                    
                    dst_img = os.path.join(split_img_dir, img_name)
                    dst_mask = os.path.join(split_mask_dir, mask_name)
                    
                    # Salin file gambar
                    shutil.copy2(src_img, dst_img)
                    
                    # Salin dan proses mask
                    if name == 'NIST16':
                        from PIL import Image, ImageOps
                        mask_img = Image.open(src_mask).convert('L')
                        inverted_mask = ImageOps.invert(mask_img)
                        inverted_mask.save(dst_mask)
                    else:
                        shutil.copy2(src_mask, dst_mask)
                            
                    # Simpan path relatif ke alllist.txt
                    f.write(f"images/{img_name} masks/{mask_name}\n")

if __name__ == '__main__':
    main()
