import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

class Sobel(nn.Module):
    def __init__(self, in_channel=64): # Kita pasang channelnya aja di sini, tapi dah default 64 (B, C, H, W => Pakai B, 64, 256, 256)
        super(Sobel, self).__init__()

        # Horizontal and vertical kernels
        self.matrix_h = torch.Tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)# Horizontal
        self.matrix_v = torch.Tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32) # Vertical

        # Form kernels from the matrices
        # View 1, 1, 3, 3 itu buat dimasukin ke conv, isinya outChannel, inChannel/group, kernelHeight, kernelWidth
        # Group buat apa? Ngatur hubungan input dan output channel. For sobel, every channel is processed on its own to detect
        # edges in each features, therefore group will be 64 bcs 64 channels. 64/64 = 1

        # Repeat? Sejumlah channel. Jadi kernel di-copy sejumlah channel
        self.kernel_h = self.matrix_h.view(1, 1, 3, 3).repeat(64, 1, 1, 1)
        self.kernel_v = self.matrix_v.view(1, 1, 3, 3).repeat(64, 1, 1, 1)

        # Register buffer => Utk otomatis bs masukin ke device
        self.register_buffer("kernel_h", self.kernel_h)
        self.register_buffer("kernel_v", self.kernel_v)
        self.in_channels = in_channel

    def forward(self, x): #Panggil ini, x = image
        x_gradient = F.conv2d(x, self.kernel_h, padding=1, groups=self.in_channels) # Horizontal
        y_gradient = F.conv2d(x, self.kernel_v, padding=1, groups=self.in_channels) # Vertical

        gradient = torch.sqrt(x_gradient ** 2 + y_gradient ** 2 + 1e-6)
        # Dikasih constant kecil 1e-6 in case akar 0, krn bs masalah waktu backpropagation
        
        return gradient