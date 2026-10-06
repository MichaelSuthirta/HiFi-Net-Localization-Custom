import os
import sys
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms

# Import modul GaussianSmoothing milik Anda
from models.GaussianSmoothing import GaussianSmoothing

def apply_blur(img_path, output_path, kernel_size=15, sigma=2):
    # 1. Buka Gambar
    try:
        img = Image.open(img_path).convert('RGB')
    except Exception as e:
        print(f"Gagal membuka gambar: {e}")
        return

    print(f"Memproses gambar: {img_path} (Ukuran: {img.size})")
    
    # 2. Konversi gambar ke Tensor PyTorch
    transform = transforms.ToTensor()
    # Tambahkan dimensi batch menjadi [1, 3, H, W]
    img_tensor = transform(img).unsqueeze(0) 
    
    # 3. Siapkan Modul Gaussian Smoothing
    # Gambar RGB memiliki 3 channel
    smoothing = GaussianSmoothing(channels=3, kernel_size=kernel_size, sigma=sigma, dim=2)
    
    # 4. Beri Padding (Agar ukuran gambar output tidak mengecil akibat proses konvolusi)
    pad_size = kernel_size // 2
    # Menggunakan mode 'reflect' agar pinggiran gambar terlihat natural saat di-blur
    img_tensor_padded = F.pad(img_tensor, (pad_size, pad_size, pad_size, pad_size), mode='reflect')
    
    # 5. Aplikasikan Blur
    with torch.no_grad():
        blurred_tensor = smoothing(img_tensor_padded)
        
    # 6. Kembalikan ke format gambar
    # Hapus dimensi batch dan pastikan nilai piksel aman di rentang [0, 1]
    blurred_tensor = blurred_tensor.squeeze(0).clamp(0, 1) 
    
    to_pil = transforms.ToPILImage()
    blurred_img = to_pil(blurred_tensor)
    
    # 7. Simpan Output
    blurred_img.save(output_path)
    print(f"Berhasil! Gambar hasil blur disimpan di: {output_path}")
    print(f"Parameter -> Kernel Size: {kernel_size}, Sigma: {sigma}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("\n=== Cara Penggunaan ===")
        print("python apply_gaussian.py <path_gambar_input> <path_gambar_output> [kernel_size] [sigma]")
        print("Contoh: python apply_gaussian.py gambar_asli.jpg hasil_blur.jpg")
        print("Contoh dengan custom blur: python apply_gaussian.py gambar_asli.jpg hasil_blur.jpg 21 5.0\n")
        sys.exit(1)
        
    input_img = sys.argv[1]
    output_img = sys.argv[2]
    
    # Default: kernel 15, sigma 3.0 (lumayan nge-blur)
    k_size = int(sys.argv[3]) if len(sys.argv) > 3 else 15
    sig = float(sys.argv[4]) if len(sys.argv) > 4 else 3.0
    
    if not os.path.exists(input_img):
        print(f"\nError: Gambar '{input_img}' tidak ditemukan!\n")
        sys.exit(1)
        
    apply_blur(input_img, output_img, kernel_size=k_size, sigma=sig)
