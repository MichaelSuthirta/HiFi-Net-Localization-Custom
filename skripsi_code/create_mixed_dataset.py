import os
import shutil
import random
from tqdm import tqdm

def collect_split_dataset(base_dir, dataset_name):
    """Collects all image and mask paths from a dataset directory using alllist.txt."""
    data = []
    # Collect from train, val, test splits if they exist
    for split in ['train', 'val', 'test']:
        txt_path = os.path.join(base_dir, dataset_name, split, 'alllist.txt')
        if os.path.exists(txt_path):
            with open(txt_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    parts = line.split()
                    if len(parts) >= 2:
                        img_rel_path = parts[0]
                        mask_rel_path = parts[1]
                        
                        img_path = os.path.join(base_dir, dataset_name, split, img_rel_path)
                        mask_path = os.path.join(base_dir, dataset_name, split, mask_rel_path)
                        
                        if os.path.exists(img_path) and os.path.exists(mask_path):
                            data.append((img_path, mask_path))
    return data

def collect_stgan(base_dir, dataset_name):
    """Collects STGAN_7k files."""
    data = []
    img_dir = os.path.join(base_dir, dataset_name, 'fake')
    mask_dir = os.path.join(base_dir, dataset_name, 'mask')
    if os.path.exists(img_dir) and os.path.exists(mask_dir):
        for img_name in os.listdir(img_dir):
            if img_name.startswith('.'): continue
            img_path = os.path.join(img_dir, img_name)
            # STGAN masks end with _label.jpg
            mask_name = img_name.rsplit('.', 1)[0] + '_label.jpg'
            mask_path = os.path.join(mask_dir, mask_name)
            if os.path.exists(mask_path):
                data.append((img_path, mask_path))
    return data

def collect_faceshifter(base_dir, dataset_name):
    """Collects FaceShifter files."""
    data = []
    img_dir = os.path.join(base_dir, dataset_name, 'fake', 'FaceShifter')
    mask_dir = os.path.join(base_dir, dataset_name, 'mask', 'FaceShifter')
    if os.path.exists(img_dir) and os.path.exists(mask_dir):
        for img_name in os.listdir(img_dir):
            if img_name.startswith('.'): continue
            img_path = os.path.join(img_dir, img_name)
            # FaceShifter masks end with _mask.png
            mask_name = img_name.rsplit('.', 1)[0] + '_mask.png'
            mask_path = os.path.join(mask_dir, mask_name)
            if os.path.exists(mask_path):
                data.append((img_path, mask_path))
    return data

def main():
    base_datasets_dir = 'datasets'
    individual_dir = os.path.join(base_datasets_dir, 'individual')
    output_dir = os.path.join(base_datasets_dir, 'Mixed_Traditional_Dataset')
    
    print("Collecting data paths...")
    all_data = []
    
    # 1. Collect ONLY from Traditional Individual datasets
    individual_datasets = ['CASIA2', 'Columbia', 'Coverage', 'IMD2020', 'NIST16']
    for ds in individual_datasets:
        ds_data = collect_split_dataset(individual_dir, ds)
        all_data.extend(ds_data)
        print(f"{ds}: {len(ds_data)} images")
        
    total_images = len(all_data)
    print(f"\nTotal traditional images collected: {total_images}")
    
    if total_images == 0:
        print("Error: No images found. Check the paths.")
        return

    # Shuffle all data randomly
    random.seed(42)
    random.shuffle(all_data)
    
    # Determine split sizes (80% train, 10% val, 10% test)
    train_size = int(total_images * 0.8)
    remaining = total_images - train_size
    val_size = remaining // 2
    test_size = remaining - val_size
    
    splits = {
        'train': all_data[:train_size],
        'val': all_data[train_size:train_size + val_size],
        'test': all_data[train_size + val_size:]
    }
    
    print(f"Splits -> Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")
    
    # Create directories and copy files
    for split_name, file_pairs in splits.items():
        if len(file_pairs) == 0:
            continue
            
        print(f"\nProcessing {split_name} split...")
        split_img_dir = os.path.join(output_dir, split_name, 'images')
        split_mask_dir = os.path.join(output_dir, split_name, 'masks')
        os.makedirs(split_img_dir, exist_ok=True)
        os.makedirs(split_mask_dir, exist_ok=True)
        
        txt_path = os.path.join(output_dir, split_name, 'alllist.txt')
        
        with open(txt_path, 'w') as f:
            for i, (img_path, mask_path) in enumerate(tqdm(file_pairs)):
                ds_prefix = img_path.split('/')[-4] if 'individual' in img_path else img_path.split('/')[-3]
                ext_img = os.path.splitext(img_path)[1]
                ext_mask = os.path.splitext(mask_path)[1]
                
                new_img_name = f"{ds_prefix}_{i:05d}{ext_img}"
                new_mask_name = f"{ds_prefix}_{i:05d}{ext_mask}"
                
                dest_img_path = os.path.join(split_img_dir, new_img_name)
                dest_mask_path = os.path.join(split_mask_dir, new_mask_name)
                
                shutil.copy2(img_path, dest_img_path)
                shutil.copy2(mask_path, dest_mask_path)
                
                # Write to alllist.txt
                f.write(f"images/{new_img_name} masks/{new_mask_name}\n")

    print("\nDataset preparation completed!")
    print(f"Mixed dataset saved at: {output_dir}")

if __name__ == '__main__':
    main()
