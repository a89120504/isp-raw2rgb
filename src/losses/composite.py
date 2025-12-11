# -*- coding: utf-8 -*-
"""
Composite Loss Functions for ISP Training
- SSIMLoss: Structural Similarity Loss
- DeltaELoss: CIE Delta E Color Difference Loss
- CompositeLoss: Weighted combination of L1, SSIM, and Delta E
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SSIMLoss(nn.Module):
    """Structural Similarity Index Loss"""
    
    def __init__(self, window_size: int = 11):
        super().__init__()
        self.window_size = window_size
        self.channel = 3
        self.window = self._create_window(window_size, self.channel)
    
    def _gaussian(self, window_size: int, sigma: float) -> torch.Tensor:
        gauss = torch.Tensor([
            math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
            for x in range(window_size)
        ])
        return gauss / gauss.sum()
    
    def _create_window(self, window_size: int, channel: int) -> torch.Tensor:
        _1D_window = self._gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window
    
    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        _, channel, _, _ = img1.size()
        
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self._create_window(self.window_size, channel)
            window = window.to(img1.device).type_as(img1)
            self.window = window
            self.channel = channel
        
        mu1 = F.conv2d(img1, window, padding=self.window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size // 2, groups=channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size // 2, groups=channel) - mu1_mu2
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return 1 - ssim_map.mean()


class DeltaELoss(nn.Module):
    """CIE Delta E (2000) Color Difference Loss"""
    
    def __init__(self):
        super().__init__()
    
    def _rgb_to_xyz(self, image: torch.Tensor) -> torch.Tensor:
        """Convert sRGB to XYZ color space"""
        mask = image > 0.04045
        linear = torch.where(mask, ((image + 0.055) / 1.055) ** 2.4, image / 12.92)
        
        r, g, b = linear[:, 0], linear[:, 1], linear[:, 2]
        x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
        y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
        z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b
        
        return torch.stack([x, y, z], dim=1)
    
    def _xyz_to_lab(self, xyz: torch.Tensor) -> torch.Tensor:
        """Convert XYZ to CIELAB color space"""
        # D65 illuminant reference
        xn, yn, zn = 0.95047, 1.00000, 1.08883
        xyz = xyz / torch.tensor([xn, yn, zn], device=xyz.device).view(1, 3, 1, 1)
        xyz = torch.clamp(xyz, min=1e-8)
        
        mask = xyz > 0.008856
        f = torch.where(mask, torch.pow(xyz, 1/3), 7.787 * xyz + 16/116)
        
        L = 116 * f[:, 1] - 16
        a = 500 * (f[:, 0] - f[:, 1])
        b = 200 * (f[:, 1] - f[:, 2])
        
        return torch.stack([L, a, b], dim=1)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.clamp(pred, 0, 1)
        target = torch.clamp(target, 0, 1)
        
        pred_lab = self._xyz_to_lab(self._rgb_to_xyz(pred))
        target_lab = self._xyz_to_lab(self._rgb_to_xyz(target))
        
        delta_e = torch.sqrt(torch.sum(torch.pow(pred_lab - target_lab, 2), dim=1) + 1e-8)
        return torch.mean(delta_e)


class CompositeLoss(nn.Module):
    """
    Composite Loss combining L1, SSIM, and Delta E
    
    Args:
        l1_weight: Weight for L1 loss
        ssim_weight: Weight for SSIM loss
        delta_e_weight: Weight for Delta E loss
        device: Device to place loss modules on
    """
    
    def __init__(
        self,
        l1_weight: float = 1.0,
        ssim_weight: float = 0.25,
        delta_e_weight: float = 0.15,
        device: str = 'cuda'
    ):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.ssim = SSIMLoss().to(device)
        self.delta_e = DeltaELoss()
        
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.delta_e_weight = delta_e_weight
    
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor,
        use_warmup: bool = False
    ) -> torch.Tensor:
        """
        Compute composite loss
        
        Args:
            pred: Predicted RGB image
            target: Ground truth RGB image
            use_warmup: If True, only use L1 + small SSIM (for warmup epochs)
        """
        l1_loss = self.l1(pred, target)
        ssim_loss = self.ssim(pred, target)
        
        if use_warmup:
            return l1_loss + 0.1 * ssim_loss
        
        de_loss = self.delta_e(pred, target)
        return (self.l1_weight * l1_loss + 
                self.ssim_weight * ssim_loss + 
                self.delta_e_weight * de_loss)
