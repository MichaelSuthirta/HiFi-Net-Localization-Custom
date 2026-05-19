import os
import shutil
import random
from tqdm import tqdm

def split_dataset(image_dir, mask_dir, output_dir, train_ratio, val_ratio, test_ratio, txt_file=None, seed=42):
    # Pastikan total persentase = 1.0
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Total ratio harus 1.0"
    
    random.seed(seed)
    
    valid_pairs = []
    
    if txt_file and os.path.exists(txt_file):
        print(f"Membaca daftar file dari {txt_file}...")
        txt_base_path = os.path.dirname(txt_file)
        with open(txt_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    img_path = os.path.join(txt_base_path, parts[0])
                    mask_path = os.path.join(txt_base_path, parts[1])
                    if os.path.exists(img_path) and os.path.exists(mask_path):
                        valid_pairs.append((img_path, mask_path))
                    else:
                        print(f"Warning: File tidak ditemukan -> {img_path} atau {mask_path}")
    else:
        print("Mencari pasangan gambar dan mask dari direktori...")
        image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
        
        for img_name in image_files:
            base_name = os.path.splitext(img_name)[0]
            
            # kandidat mask
            mask_candidates = [
                f"{base_name}_label.jpg",
                f"{base_name}_label.png",
                f"{base_name}_label.jpeg",
                f"{base_name}_gt.jpg",
                f"{base_name}_gt.png",
                f"{base_name}_gt.jpeg",
                f"{base_name}.jpg",
                f"{base_name}.png",
                f"{base_name}.jpeg",
                img_name
            ]
            
            found_mask = None
            for cand in mask_candidates:
                if os.path.exists(os.path.join(mask_dir, cand)):
                    found_mask = cand
                    break
                    
            if found_mask:
                valid_pairs.append((os.path.join(image_dir, img_name), os.path.join(mask_dir, found_mask)))
            else:
                print(f"Warning: Mask tidak ditemukan untuk gambar {img_name}")
            
    print(f"Ditemukan {len(valid_pairs)} pasangan gambar & mask yang valid.")
    
    if len(valid_pairs) == 0:
        print("Tidak ada pasangan yang cocok. Silakan cek folder dan penamaan file Anda.")
        return
        
    # Acak urutan data
    random.shuffle(valid_pairs)
    
    # Hitung jumlah untuk setiap split
    total = len(valid_pairs)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    splits = {
        'train': valid_pairs[:train_end],
        'val': valid_pairs[train_end:val_end],
        'test': valid_pairs[val_end:]
    }
    
    # Buat direktori output dan salin file
    for split_name, pairs in splits.items():
        print(f"\nMemproses set '{split_name}' ({len(pairs)} gambar)...")
        
        split_img_dir = os.path.join(output_dir, split_name, 'images')
        split_mask_dir = os.path.join(output_dir, split_name, 'masks')
        
        os.makedirs(split_img_dir, exist_ok=True)
        os.makedirs(split_mask_dir, exist_ok=True)
        
        for src_img, src_mask in tqdm(pairs):
            img_name = os.path.basename(src_img)
            mask_name = os.path.basename(src_mask)
            
            dst_img = os.path.join(split_img_dir, img_name)
            dst_mask = os.path.join(split_mask_dir, mask_name)
            
            shutil.copy2(src_img, dst_img)
            shutil.copy2(src_mask, dst_mask)
            
    print(f"\nSelesai! Dataset berhasil dibagi di dalam folder: '{output_dir}'")
    print(f"Struktur Folder:")
    print(f"- {output_dir}/train/ (images & masks)")
    print(f"- {output_dir}/val/ (images & masks)")
    print(f"- {output_dir}/test/ (images & masks)")

if __name__ == '__main__':

    INPUT_IMAGE_DIR = 'datasets/Dataset STGAN + COVERAGE/May_train/fake'
    INPUT_MASK_DIR = 'datasets/Dataset STGAN + COVERAGE/May_train/mask'
    TXT_FILE_DIR = 'manipulated_data_NIST16/alllist.txt' 

    
    OUTPUT_BASE_DIR = 'data_split_NIST16'
    
    split_dataset(
        image_dir=INPUT_IMAGE_DIR, 
        mask_dir=INPUT_MASK_DIR, 
        output_dir=OUTPUT_BASE_DIR,
        train_ratio=0.3,
        val_ratio=0.6, 
        test_ratio=0.1,
        txt_file=TXT_FILE_DIR
    )
