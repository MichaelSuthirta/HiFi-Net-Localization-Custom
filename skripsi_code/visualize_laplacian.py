import os
import sys
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

# Import modul GaussianSmoothing milik Anda
from models.GaussianSmoothing import GaussianSmoothing

def visualize_laplacian(img_path, output_dir, kernel_size=15, sigma=2.0, scale=2):
    # Buat folder output jika belum ada
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Buka Gambar Asli
    try:
        img = Image.open(img_path).convert('RGB')
    except Exception as e:
        print(f"Gagal membuka gambar: {e}")
        return

    print(f"Memproses gambar: {img_path} (Ukuran: {img.size})")
    
    # Konversi ke Tensor [1, 3, H, W]
    transform = transforms.ToTensor()
    x = transform(img).unsqueeze(0)
    
    # 2. Setup Gaussian Smoothing
    smoothing = GaussianSmoothing(channels=3, kernel_size=kernel_size, sigma=sigma, dim=2)
    
    # Beri padding agar ukurannya tidak mengecil
    pad_size = kernel_size // 2
    x_padded = F.pad(x, (pad_size, pad_size, pad_size, pad_size), mode='reflect')
    
    with torch.no_grad():
        # --- A. PROSES BLUR (SMOOTHING) ---
        sm = smoothing(x_padded)
        
        # --- B. PROSES DOWN-UP SAMPLING (Sesuai kode LaPlacianMs) ---
        # Di LaPlacianMs, gambar blur di-downsample lalu di-upsample lagi
        # untuk menghilangkan informasi frekuensi tinggi secara lebih agresif
        h, w = x.shape[2], x.shape[3]
        sm = F.interpolate(sm, scale_factor=1/scale, mode='bilinear', align_corners=False)
        sm = F.interpolate(sm, size=(h, w), mode='bilinear', align_corners=False)
        
        # --- C. PROSES LAPLACIAN (PENGURANGAN) ---
        # diff = x - sm
        diff = x - sm
        
        # --- D. PROSES VISUALISASI ---
        # Karena hasil pengurangan (diff) bisa menghasilkan angka negatif, 
        # kita harus menormalkannya agar bisa dilihat sebagai gambar.
        
        # PENINGKATAN KONTRAS (ENHANCEMENT)
        # Kita kalikan hasil selisihnya dengan faktor pengali agar teksturnya lebih "keluar".
        # Silakan ganti angka 5.0 ini jadi lebih besar jika teksturnya masih kurang jelas.
        amplification_factor = 100.0
        diff_amplified = diff * amplification_factor
        
        # Cara 1: Menggeser nilai ke abu-abu (0 -> 127)
        # Angka negatif jadi gelap, angka 0 jadi abu-abu, positif jadi terang
        diff_gray_centered = diff_amplified + 0.5 
        diff_gray_centered = diff_gray_centered.clamp(0, 1)
        
    to_pil = transforms.ToPILImage()
    
    # Simpan Gambar 1: Blur
    img_blurred = to_pil(sm.squeeze(0).clamp(0, 1))
    path_blur = os.path.join(output_dir, "1_gambar_blur.jpg")
    img_blurred.save(path_blur)
    
    # Simpan Gambar 2: Laplacian (High-Frequency)
    img_laplacian = to_pil(diff_gray_centered.squeeze(0))
    path_laplacian = os.path.join(output_dir, "2_gambar_laplacian.jpg")
    img_laplacian.save(path_laplacian)
    
    print("\nVisualisasi Berhasil! Silakan periksa folder:", output_dir)
    print("1. Gambar Blur (sm) ->", path_blur)
    print("2. Gambar Laplacian (x - sm) ->", path_laplacian)
    print("Catatan: Gambar Laplacian didominasi warna abu-abu. Tepi (edge) dan noise/tekstur akan terlihat lebih terang/gelap.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Cara Penggunaan:")
        print("python visualize_laplacian.py <path_gambar_input>")
        print("Contoh: python visualize_laplacian.py testing_gaussianSmoothing/hasil_blur_sigma_2_4.jpg")
        sys.exit(1)
        
    input_img = sys.argv[1]
    output_dir = "testing_gaussianSmoothing/visualisasi_laplacian"
    
    if not os.path.exists(input_img):
        print(f"\nError: Gambar '{input_img}' tidak ditemukan!\n")
        sys.exit(1)
        
    # Menggunakan kernel dan sigma default LaPlacianMs
    visualize_laplacian(input_img, output_dir, kernel_size=3, sigma=2.0, scale=2)
