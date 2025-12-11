# -*- coding: utf-8 -*-
"""
RAW Image Dataset for ISP Training
Handles Bayer pattern packing and data augmentation
"""

import os
import glob
import random
from typing import Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class RawImageDataset(Dataset):
    """
    Dataset for RAW to RGB image pairs
    
    Args:
        root_dir: Directory containing input/GT subdirectories (train) or RAW images (test)
        is_train: Whether this is training mode
        transform: Whether to apply augmentation
        img_size: Patch size for random cropping (training only)
        black_level: Black level for RAW normalization
        flip_prob: Probability of horizontal/vertical flip
        rot_prob: Probability of 90-degree rotation
    """
    
    def __init__(
        self,
        root_dir: str,
        is_train: bool = True,
        transform: bool = False,
        img_size: int = 512,
        black_level: float = 64.0,
        flip_prob: float = 0.5,
        rot_prob: float = 0.5
    ):
        self.is_train = is_train
        self.transform = transform
        self.img_size = img_size
        self.black_level = black_level
        self.flip_prob = flip_prob
        self.rot_prob = rot_prob
        
        if self.is_train:
            # Find input and GT directories
            subdirs = sorted([
                d for d in os.listdir(root_dir) 
                if os.path.isdir(os.path.join(root_dir, d))
            ])
            dir1 = os.path.join(root_dir, subdirs[0])
            dir2 = os.path.join(root_dir, subdirs[1])
            
            # Determine which is input (RAW) and which is GT
            if 'sony' in subdirs[1].lower() or 'raw' in subdirs[1].lower():
                self.input_dir, self.gt_dir = dir2, dir1
            else:
                self.input_dir, self.gt_dir = dir1, dir2
            
            self.input_paths = sorted(glob.glob(os.path.join(self.input_dir, '*.png')))
            self.gt_paths = sorted(glob.glob(os.path.join(self.gt_dir, '*.png')))
        else:
            self.input_paths = sorted(glob.glob(os.path.join(root_dir, '*.png')))
    
    def pack_raw(self, raw: np.ndarray) -> np.ndarray:
        """
        Pack Bayer pattern into 4 channels (RGGB)
        
        Args:
            raw: Raw image [H, W]
            
        Returns:
            Packed channels [4, H/2, W/2]
        """
        im = raw.astype(np.float32)
        im = np.maximum(im - self.black_level, 0)
        
        # Pack RGGB pattern
        out = np.stack((
            im[0::2, 0::2],  # R
            im[0::2, 1::2],  # G1
            im[1::2, 0::2],  # G2
            im[1::2, 1::2],  # B
        ), axis=0)
        
        return out / (65535.0 - self.black_level)
    
    def __len__(self) -> int:
        return len(self.input_paths)
    
    def __getitem__(self, idx: int):
        input_path = self.input_paths[idx]
        raw_np = np.array(Image.open(input_path))
        input_tensor = torch.from_numpy(self.pack_raw(raw_np)).float()
        
        if self.is_train:
            gt_path = self.gt_paths[idx]
            gt_img = Image.open(gt_path).convert('RGB')
            gt_np = np.array(gt_img).astype(np.float32) / 255.0
            gt_tensor = torch.from_numpy(gt_np.transpose(2, 0, 1)).float()
            
            if self.transform:
                input_tensor, gt_tensor = self._augment(input_tensor, gt_tensor)
            
            return input_tensor, gt_tensor
        
        # Test mode: return input and file ID
        file_id = os.path.basename(input_path).split('.')[0]
        return input_tensor, file_id
    
    def _augment(
        self, 
        inp: torch.Tensor, 
        gt: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply random augmentation"""
        _, h, w = inp.shape
        th, tw = self.img_size, self.img_size
        
        # Random crop
        if h > th and w > tw:
            i = random.randint(0, h - th)
            j = random.randint(0, w - tw)
            inp = inp[:, i:i+th, j:j+tw]
            gt = gt[:, i*2:i*2+th*2, j*2:j*2+tw*2]
        
        # Random horizontal flip
        if random.random() < self.flip_prob:
            inp = torch.flip(inp, [2])
            gt = torch.flip(gt, [2])
        
        # Random vertical flip
        if random.random() < self.flip_prob:
            inp = torch.flip(inp, [1])
            gt = torch.flip(gt, [1])
        
        # Random 90-degree rotation
        if random.random() < self.rot_prob:
            k = random.randint(1, 3)
            inp = torch.rot90(inp, k, [1, 2])
            gt = torch.rot90(gt, k, [1, 2])
        
        return inp, gt
