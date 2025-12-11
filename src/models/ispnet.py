# -*- coding: utf-8 -*-
"""
ISPNet: 3-Level Restormer for RAW to RGB Conversion
Architecture: Encoder-Decoder with skip connections and PixelShuffle upsampling
"""

import torch
import torch.nn as nn
from typing import List

from .blocks import TransformerBlock


class ISPNet(nn.Module):
    """
    ISP Network based on 3-Level Restormer Architecture
    
    Args:
        dim: Base channel dimension (default: 48)
        enc_blocks: Number of transformer blocks in each encoder level
        mid_blocks: Number of transformer blocks in bottleneck
        dec_blocks: Number of transformer blocks in each decoder level
        heads: Number of attention heads at each level
    """
    
    def __init__(
        self,
        dim: int = 48,
        enc_blocks: List[int] = [4, 4, 6],
        mid_blocks: int = 4,
        dec_blocks: List[int] = [4, 4],
        heads: List[int] = [1, 2, 4, 8]
    ):
        super().__init__()
        
        # Level 1 (Full resolution)
        self.intro = nn.Conv2d(4, dim, 3, 1, 1)
        self.enc1 = nn.Sequential(
            *[TransformerBlock(dim, heads[0]) for _ in range(enc_blocks[0])]
        )
        self.down1 = nn.Conv2d(dim, dim * 2, 2, 2)
        
        # Level 2 (1/2 resolution)
        self.enc2 = nn.Sequential(
            *[TransformerBlock(dim * 2, heads[1]) for _ in range(enc_blocks[1])]
        )
        self.down2 = nn.Conv2d(dim * 2, dim * 4, 2, 2)
        
        # Level 3 (1/4 resolution) - Bottleneck
        self.enc3 = nn.Sequential(
            *[TransformerBlock(dim * 4, heads[2]) for _ in range(enc_blocks[2])]
        )
        self.middle = nn.Sequential(
            *[TransformerBlock(dim * 4, heads[2]) for _ in range(mid_blocks)]
        )
        
        # Decoder
        self.up2 = nn.ConvTranspose2d(dim * 4, dim * 2, 2, 2)
        self.up1 = nn.ConvTranspose2d(dim * 2, dim, 2, 2)
        self.dec1 = nn.Sequential(
            *[TransformerBlock(dim, heads[0]) for _ in range(dec_blocks[0])]
        )
        
        # Output (PixelShuffle for 2x upsampling)
        self.out = nn.Conv2d(dim, 12, 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input RAW tensor [B, 4, H, W] (packed Bayer pattern)
            
        Returns:
            RGB tensor [B, 3, H*2, W*2]
        """
        # Encoder
        x = self.intro(x)
        enc1 = self.enc1(x)
        x = self.down1(enc1)
        
        enc2 = self.enc2(x)
        x = self.down2(enc2)
        
        x = self.enc3(x)
        x = self.middle(x)
        
        # Decoder with skip connections
        x = self.up2(x) + enc2
        x = self.up1(x) + enc1
        x = self.dec1(x)
        
        # Output
        x = self.out(x)
        x = self.pixel_shuffle(x)
        
        return torch.sigmoid(x)
