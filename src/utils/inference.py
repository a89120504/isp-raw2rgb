# -*- coding: utf-8 -*-
"""
Tiled Inference for Memory-Efficient Processing
Handles large images by processing overlapping tiles
"""

import torch
import torch.nn as nn


def tiled_inference(
    model: nn.Module,
    img_tensor: torch.Tensor,
    tile_size: int = 512,
    overlap: int = 64
) -> torch.Tensor:
    """
    Perform inference on large images using overlapping tiles
    
    Args:
        model: ISPNet model
        img_tensor: Input tensor [B, 4, H, W]
        tile_size: Size of each tile
        overlap: Overlap between tiles
        
    Returns:
        Output tensor [B, 3, H*2, W*2]
    """
    b, c, h, w = img_tensor.shape
    out_h, out_w = h * 2, w * 2
    
    output = torch.zeros((b, 3, out_h, out_w), device=img_tensor.device)
    count = torch.zeros((b, 3, out_h, out_w), device=img_tensor.device)
    
    stride = tile_size - overlap
    
    # Generate tile positions
    h_steps = list(range(0, h, stride))
    w_steps = list(range(0, w, stride))
    
    # Ensure we cover the entire image
    if h - tile_size not in h_steps and h > tile_size:
        h_steps.append(h - tile_size)
    if w - tile_size not in w_steps and w > tile_size:
        w_steps.append(w - tile_size)
    
    for hi in h_steps:
        for wi in w_steps:
            h_end = min(hi + tile_size, h)
            w_end = min(wi + tile_size, w)
            h_start = h_end - tile_size if h_end - tile_size >= 0 else 0
            w_start = w_end - tile_size if w_end - tile_size >= 0 else 0
            
            patch = img_tensor[:, :, h_start:h_end, w_start:w_end]
            
            with torch.no_grad():
                pred_patch = model(patch)
            
            # Map to output coordinates (2x upsampling)
            out_h_start, out_h_end = h_start * 2, h_end * 2
            out_w_start, out_w_end = w_start * 2, w_end * 2
            
            output[:, :, out_h_start:out_h_end, out_w_start:out_w_end] += pred_patch
            count[:, :, out_h_start:out_h_end, out_w_start:out_w_end] += 1.0
    
    return output / count
