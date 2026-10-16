import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '../..'))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from models.seg_hrnet import HighResolutionNet

class Sharpen(nn.Module):
    def __init__(self, in_channel=64): # Kita pasang channelnya aja di sini, tapi dah default 64 (B, C, H, W => Pakai B, 64, 256, 256)
        super(Sharpen, self).__init__()

        # Horizontal and vertical kernels
        matrix = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)

        # Form kernels from the matrices
        # View 1, 1, 3, 3 itu buat dimasukin ke conv, isinya outChannel, inChannel/group, kernelHeight, kernelWidth
        # Group buat apa? Ngatur hubungan input dan output channel. For sobel, every channel is processed on its own to detect
        # edges in each features, therefore group will be 64 bcs 64 channels. 64/64 = 1

        # Repeat? Sejumlah channel. Jadi kernel di-copy sejumlah channel
        kernel_sharpen = matrix.view(1, 1, 3, 3).repeat(in_channel, 1, 1, 1)

        # Register buffer => Utk otomatis bs masukin ke device
        self.register_buffer("kernel_sharpen", kernel_sharpen)
        self.in_channels = in_channel

    def forward(self, x): #Panggil ini, x = image
        gradient = F.conv2d(x, self.kernel_sharpen, padding=1, groups=self.in_channels)
        
        return gradient

class HighResolutionNetSobel(HighResolutionNet):
    def __init__(self, config, **kwargs):
        super(HighResolutionNetSobel, self).__init__(config, **kwargs)
        # Gantikan cabang LaplacianMs dengan Sharpen kernel
        self.sharpen = Sharpen(in_channel=64)

    def forward(self, x):
        # 1. Cabang Frekuensi / Tepi (Sobel)
        x_fre = self.conv1fre(x)
        x_fre = self.bn1fre(x_fre)
        x_fre = self.relu(x_fre)
        
        # Ekstraksi tepi menggunakan Sobel menggantikan Laplacian
        x_fre = self.sharpen(x_fre)
        
        x_fre = self.conv2fre(x_fre)
        x_fre = self.bn2fre(x_fre)
        x_fre = self.relu(x_fre)

        # 2. Cabang RGB Standar
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        # 3. Penggabungan Fitur
        x = self.concat_depth(x, x_fre)
        x = self.conv_1x1_merge(x)
        x = self.layer1(x)

        # 4. HRNet Stages
        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                if i < self.stage2_cfg['NUM_BRANCHES']:
                    x_list.append(self.transition2[i](y_list[i]))
                else:
                    x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                if i < self.stage3_cfg['NUM_BRANCHES']:
                    x_list.append(self.transition3[i](y_list[i]))
                else:
                    x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        x = self.stage4(x_list)
        return x

def get_seg_model(cfg, **kwargs):
    model = HighResolutionNetSobel(cfg, **kwargs)
    model.init_weights(cfg.PRETRAINED)
    return model
