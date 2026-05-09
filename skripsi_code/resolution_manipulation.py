import os
import io
import random
import argparse
import numpy as np
from PIL import Image, ImageFilter
from pathlib import Path
from tqdm import tqdm

def add_social_media_noise(img):
    """Menambahkan sedikit noise dan Gaussian Blur ringan khas pemrosesan server."""
    # Convert to numpy for noise addition
    img_array = np.array(img)
    
    # Add very light Gaussian noise
    noise = np.random.normal(0, 1.5, img_array.shape).astype(np.float32)
    img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    
    # Back to PIL and apply very light blur (simulating resizing/compression softness)
    noisy_img = Image.fromarray(img_array)
    noisy_img = noisy_img.filter(ImageFilter.GaussianBlur(radius=0.3))
    return noisy_img

def manipulate_image(input_path, output_path, scale_factor=0.5, jpeg_quality=60, chain_saves=3):
    """
    Melakukan manipulasi resolusi dan kompresi lossy berantai (Generation Loss).
    Simulasi kompresi sosial media berantai (misal: di-forward berkali-kali di WhatsApp).
    """
    try:
        img = Image.open(input_path)
        if img.mode in ('RGBA', 'P'):
            img = img.convert('RGB')
            
        orig_size = img.size
        
        # 1. Resolusi awal jika ada downscaling
        if scale_factor < 1.0:
            new_size = (int(orig_size[0] * scale_factor), int(orig_size[1] * scale_factor))
            img = img.resize(new_size, Image.Resampling.BICUBIC)
            img = img.resize(orig_size, Image.Resampling.BICUBIC)

        # 2. Chain Saves (Simulasi gambar di-forward / di-save berulang-ulang)
        for i in range(chain_saves):
            buffer = io.BytesIO()
            # Kompresi Chroma subsampling terjadi secara natural saat save JPEG
            img.save(buffer, format='JPEG', quality=jpeg_quality)
            buffer.seek(0)
            img = Image.open(buffer)
            
            # Pada beberapa lompatan (forward), tambahkan noise buatan server platform
            if i % 2 == 1:
                img = add_social_media_noise(img)

        # Simpan final gambar
        img.save(output_path, 'JPEG', quality=jpeg_quality)
        return True
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return False

def process_directory(input_dir, output_dir, scale_factor, jpeg_quality, chain_saves):
    in_path = Path(input_dir)
    out_path = Path(output_dir)
    
    if not in_path.exists():
        print(f"Error: Direktori input '{input_dir}' tidak ditemukan.")
        return
        
    out_path.mkdir(parents=True, exist_ok=True)
    extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    
    image_files = []
    for ext in extensions:
        image_files.extend(list(in_path.rglob(f"*{ext}")))
        image_files.extend(list(in_path.rglob(f"*{ext.upper()}")))
        
    image_files = list(set(image_files))
    if not image_files:
        print(f"Tidak ada gambar ditemukan di dalam {input_dir}")
        return
        
    print(f"Ditemukan {len(image_files)} gambar. Memulai simulasi Generation Loss...")
    print(f"Parameter: Scale={scale_factor}, JPEG Quality={jpeg_quality}, Di-forward/Save={chain_saves} kali")
    
    success_count = 0
    for img_path in tqdm(image_files, desc="Memproses gambar"):
        rel_path = img_path.relative_to(in_path)
        out_file_path = out_path / rel_path
        out_file_path.parent.mkdir(parents=True, exist_ok=True)
        out_file_path = out_file_path.with_suffix('.jpg')
        
        if manipulate_image(img_path, out_file_path, scale_factor, jpeg_quality, chain_saves):
            success_count += 1
            
    print(f"Selesai! Berhasil memproses {success_count}/{len(image_files)} gambar.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulasi Generation Loss (di-forward berkali-kali).")
    parser.add_argument("-i", "--input_dir", type=str, required=True, help="Direktori asal gambar.")
    parser.add_argument("-o", "--output_dir", type=str, default="manipulated_data", help="Direktori tujuan.")
    parser.add_argument("-s", "--scale_factor", type=float, default=0.5, help="Faktor downsampling resolusi.")
    parser.add_argument("-q", "--quality", type=int, default=60, help="Kualitas kompresi JPEG per rantai (1-100).")
    parser.add_argument("-c", "--chain_saves", type=int, default=3, help="Jumlah gambar di-save ulang/di-forward. Semakin banyak, semakin hancur noise khasnya.")
    
    args = parser.parse_args()
    process_directory(args.input_dir, args.output_dir, args.scale_factor, args.quality, args.chain_saves)
