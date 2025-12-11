# -*- coding: utf-8 -*-
"""
Transformer Building Blocks for Restormer
- MDTA: Multi-Dconv Head Transposed Attention
- GDFN: Gated-Dconv Feed-Forward Network
- TransformerBlock: Combined block with gradient checkpointing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class MDTA(nn.Module):
    """Multi-Dconv Head Transposed Attention"""
    
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.scale = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, 1)
        self.qkv_dw = nn.Conv2d(dim * 3, dim * 3, 3, 1, 1, groups=dim * 3)
        self.proj = nn.Conv2d(dim, dim, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        qkv = self.qkv_dw(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        
        q = q.reshape(b, self.num_heads, -1, h * w)
        k = k.reshape(b, self.num_heads, -1, h * w)
        v = v.reshape(b, self.num_heads, -1, h * w)
        
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        out = (attn @ v).reshape(b, -1, h, w)
        return self.proj(out)


class GDFN(nn.Module):
    """Gated-Dconv Feed-Forward Network"""
    
    def __init__(self, dim: int, expansion: float = 2.66):
        super().__init__()
        hidden = int(dim * expansion)
        self.proj_in = nn.Conv2d(dim, hidden * 2, 1)
        self.dw = nn.Conv2d(hidden * 2, hidden * 2, 3, 1, 1, groups=hidden * 2)
        self.proj_out = nn.Conv2d(hidden, dim, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj_in(x)
        x = self.dw(x)
        x1, x2 = x.chunk(2, dim=1)
        return self.proj_out(F.gelu(x1) * x2)


class TransformerBlock(nn.Module):
    """Transformer Block with Gradient Checkpointing"""
    
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MDTA(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = GDFN(dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            return checkpoint(self._forward_impl, x, use_reentrant=False)
        else:
            return self._forward_impl(x)
    
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        
        # Attention
        x_in = x.permute(0, 2, 3, 1)
        x = x + self.attn(self.norm1(x_in).permute(0, 3, 1, 2))
        
        # FFN
        x_in = x.permute(0, 2, 3, 1)
        x = x + self.ffn(self.norm2(x_in).permute(0, 3, 1, 2))
        
        return x
