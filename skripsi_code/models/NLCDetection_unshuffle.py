# ------------------------------------------------------------------------------
# Modified NLCDetection with True Space-to-Depth via Pixel Unshuffle / Shuffle
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.seg_hrnet_config import get_cfg_defaults
import math

def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
    return init_fun

class PartialConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.input_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                    stride, padding, dilation, groups, bias)
        self.mask_conv  = nn.Conv2d(in_channels, out_channels, kernel_size,
                                    stride, padding, dilation, groups, False)
        self.input_conv.apply(weights_init('kaiming'))
        torch.nn.init.constant_(self.mask_conv.weight, 1.0)
        for param in self.mask_conv.parameters():
            param.requires_grad = False

    def forward(self, input, mask):
        output = self.input_conv(input)
        if self.input_conv.bias is not None:
            output_bias = self.input_conv.bias.view(1, -1, 1, 1).expand_as(output)
        else:
            output_bias = torch.zeros_like(output)        

        with torch.no_grad():
            output_mask = self.mask_conv(mask)

        no_update_holes = output_mask == 0
        mask_sum = output_mask.masked_fill_(no_update_holes, 1.0)
        output_pre = (output - output_bias) / mask_sum + output_bias
        output = output_pre.masked_fill_(no_update_holes, 0.0)

        new_mask = torch.ones_like(output)
        new_mask = new_mask.masked_fill_(no_update_holes, 0.0)
        
        return output, new_mask

class NonLocalMaskUnshuffle(nn.Module):
    def __init__(self, in_channels, reduce_scale):
        super(NonLocalMaskUnshuffle, self).__init__()

        self.r = reduce_scale  # r = 4

        # input channel number: 64 * 4 * 4 = 1024
        self.ic = in_channels * self.r * self.r
        self.mc = self.ic

        self.g = nn.Conv2d(in_channels=self.ic, out_channels=self.ic,
                           kernel_size=1, stride=1, padding=0)

        self.theta = nn.Conv2d(in_channels=self.ic, out_channels=self.mc,
                               kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(in_channels=self.ic, out_channels=self.mc,
                             kernel_size=1, stride=1, padding=0)
        self.W_s = nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                             kernel_size=1, stride=1, padding=0)

        self.gamma_s = nn.Parameter(torch.ones(1))
        self.conv_1 = nn.Conv2d(in_channels=in_channels, out_channels=18,
                                kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.conv_2 = nn.Conv2d(in_channels=18, out_channels=1, 
                                kernel_size=3, stride=1, padding=1)

        ## Pconv
        self.Pconv_1 = PartialConv(3, 3, kernel_size=3, stride=2)
        self.Pconv_2 = PartialConv(3, 3, kernel_size=3, stride=2)
        self.Pconv_3 = PartialConv(3, 1, kernel_size=3, stride=2)

    def forward(self, x, img):
        b, c, h, w = x.shape  # (b, 64, 256, 256)

        # Space-to-depth via pixel_unshuffle:
        # Collects genuine 4x4 spatial patches into channels -> (b, 1024, 64, 64)
        x1 = F.pixel_unshuffle(x, self.r)

        # g_x: (b, 1024, 4096) -> permute -> (b, 4096, 1024)
        g_x = self.g(x1).view(b, self.ic, -1)
        g_x = g_x.permute(0, 2, 1)

        # theta_x: (b, 4096, 1024)
        theta_x = self.theta(x1).view(b, self.mc, -1)
        theta_x_s = theta_x.permute(0, 2, 1)

        # phi_x: (b, 1024, 4096)
        phi_x = self.phi(x1).view(b, self.mc, -1)
        phi_x_s = phi_x

        # non-local attention: (b, 4096, 4096)
        f_s = torch.matmul(theta_x_s, phi_x_s)
        f_s_div = F.softmax(f_s, dim=-1)

        # get y_s: (b, 4096, 1024) -> permute -> (b, 1024, 4096)
        y_s = torch.matmul(f_s_div, g_x)
        y_s = y_s.permute(0, 2, 1).contiguous()
        
        # Depth-to-space via pixel_shuffle:
        # Reshape to (b, 1024, 64, 64) then restore exact spatial coordinates (b, 64, 256, 256)
        y_s = y_s.view(b, self.ic, h // self.r, w // self.r)
        y_s = F.pixel_shuffle(y_s, self.r)

        # Residual connection
        mask_feat = x + self.gamma_s * self.W_s(y_s)

        mask_feat = self.conv_1(mask_feat)
        mask_binary = mask_feat
        mask_binary = self.relu(mask_binary)
        mask_logits = self.conv_2(mask_binary)
        mask_probs = torch.sigmoid(mask_logits)
        mask_tmp = mask_probs.repeat(1, 3, 1, 1)
        mask_img = img * mask_tmp

        x, new_mask = self.Pconv_1(mask_img, mask_tmp)
        x, new_mask = self.Pconv_2(x, new_mask)
        x, _        = self.Pconv_3(x, new_mask)
        mask_logits = mask_logits.squeeze(dim=1)

        return x, torch.sigmoid(mask_feat), mask_logits

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
        
    def forward(self, x):       
        return x.view(x.size(0), -1)

class Classifer(nn.Module):
    def __init__(self, in_channels, output_channels):
        super(Classifer, self).__init__()
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Flatten()
        )
        self.fc = nn.Linear(in_channels, output_channels, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feat = self.pool(x)
        feat = self.relu(feat)
        cls_res = self.fc(feat)
        return cls_res

class BranchCLS(nn.Module):
    def __init__(self, in_channels, output_channels):
        super(BranchCLS, self).__init__()
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)), Flatten())
        self.fc = nn.Linear(18, output_channels, bias=True)
        self.bn = nn.BatchNorm1d(18, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.branch_cls = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, padding=1, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=18, padding=1, kernel_size=3, stride=1),
            nn.ReLU(inplace=True), 
        )
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        feat = self.branch_cls(x)
        x = self.pool(feat)
        x = self.bn(x)
        cls_res = self.fc(x)
        cls_pro = self.leakyrelu(cls_res)
        zero_vec = -9e15 * torch.ones_like(cls_pro)
        cls_pro = torch.where(cls_pro > 0, cls_pro, zero_vec)
        return cls_res, cls_pro, feat

class FPN_loc(nn.Module):
    def __init__(self, clip_dim=64, multi_feat=None):
        super(FPN_loc, self).__init__()
        feat1_num, feat2_num, feat3_num, feat4_num = multi_feat

        self.smooth_s4 = nn.Sequential(
            nn.Conv2d(feat4_num, clip_dim, kernel_size=(1, 1), stride=(1, 1)),
            nn.Conv2d(clip_dim, clip_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        self.smooth_s3 = nn.Sequential(
            nn.Conv2d(feat3_num, clip_dim, kernel_size=(1, 1), stride=(1, 1)),
            nn.Conv2d(clip_dim, clip_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        self.smooth_s2 = nn.Sequential(
            nn.Conv2d(feat2_num, clip_dim, kernel_size=(1, 1), stride=(1, 1)),
            nn.Conv2d(clip_dim, clip_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        self.smooth_s1 = nn.Sequential(
            nn.Conv2d(feat1_num, clip_dim, kernel_size=(1, 1), stride=(1, 1)),
            nn.Conv2d(clip_dim, clip_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )

        self.fpn1 = nn.Sequential(
            nn.Conv2d(clip_dim, clip_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(clip_dim),
            nn.ReLU()
        )
        self.fpn2 = nn.Sequential(
            nn.Conv2d(clip_dim, clip_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(clip_dim),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
        )
        self.fpn3 = nn.Sequential(
            nn.Conv2d(clip_dim, clip_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(clip_dim),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
        )
        self.fpn4 = nn.Sequential(
            nn.Conv2d(clip_dim, clip_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(clip_dim),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
        )

class NLCDetection(nn.Module):
    def __init__(self):
        super(NLCDetection, self).__init__()
        self.crop_size = (256, 256)
        self.softmax_m = nn.Softmax(dim=1)
        FENet_cfg = get_cfg_defaults()

        feat_dim = 64
        self.getmask = NonLocalMaskUnshuffle(feat_dim, 4)
        self.FPN_LOC = FPN_loc(clip_dim=feat_dim, multi_feat=FENet_cfg['STAGE4']['NUM_CHANNELS'])

    def feature_resize(self, feat):
        s1, s2, s3, s4 = feat
        s1 = F.interpolate(s1, size=self.crop_size, mode='bilinear', align_corners=True)
        s2 = F.interpolate(s2, size=[i // 2 for i in self.crop_size], mode='bilinear', align_corners=True)
        s3 = F.interpolate(s3, size=[i // 4 for i in self.crop_size], mode='bilinear', align_corners=True)
        s4 = F.interpolate(s4, size=[i // 8 for i in self.crop_size], mode='bilinear', align_corners=True)
        return s1, s2, s3, s4

    def forward(self, feat, img):
        s1, s2, s3, s4 = self.feature_resize(feat)
        img = F.interpolate(img, size=self.crop_size, mode='bilinear', align_corners=True)

        feat_4 = self.FPN_LOC.smooth_s4(s4)
        feat_4 = self.FPN_LOC.fpn4(feat_4)   
        feat_3 = self.FPN_LOC.smooth_s3(s3)
        feat_3 = self.FPN_LOC.fpn3(feat_3 + feat_4)   
        feat_2 = self.FPN_LOC.smooth_s2(s2)
        feat_2 = self.FPN_LOC.fpn2(feat_2 + feat_3)   
        feat_1 = self.FPN_LOC.smooth_s1(s1)
        s1 = self.FPN_LOC.fpn1(feat_1 + feat_2)   
        pconv_feat, mask, mask_binary = self.getmask(s1, img)

        mask = mask.squeeze(dim=1)
        return mask, mask_binary
