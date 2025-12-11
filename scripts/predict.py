# -*- coding: utf-8 -*-
"""
Prediction Script - Generate submission from trained model

Usage:
    python scripts/predict.py --model experiments/ISP_Restormer_v4_Final/best_model.pt
    python scripts/predict.py --model experiments/ISP_Restormer_v4_L1_Regression/best_model_l1.pt
"""

import os
import sys
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import io
import base64
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm

# Import from new modules
from src.config import BaseConfig as Config
from src.models import ISPNet
from src.data import RawImageDataset
from src.utils import tiled_inference


def load_model(path: str, device: str) -> ISPNet:
    """Load trained model"""
    model = ISPNet(
        dim=Config.DIM,
        enc_blocks=Config.ENC_BLOCKS,
        mid_blocks=Config.MID_BLOCKS,
        dec_blocks=Config.DEC_BLOCKS,
        heads=Config.HEADS
    ).to(device)
    
    weights = torch.load(path, map_location=device)
    
    # Handle torch.compile prefixes
    new_weights = {}
    for k, v in weights.items():
        new_weights[k.replace("_orig_mod.", "")] = v
    
    model.load_state_dict(new_weights)
    model.eval()
    
    return model


def run_tta(model, x: torch.Tensor) -> torch.Tensor:
    """
    8x Test-Time Augmentation
    4 rotations + 4 rotations with horizontal flip
    """
    preds = []
    
    # 4 rotations
    for k in range(4):
        x_rot = torch.rot90(x, k, [2, 3]) if k > 0 else x
        p = tiled_inference(model, x_rot)
        if k > 0:
            p = torch.rot90(p, -k, [2, 3])
        preds.append(p)
    
    # 4 rotations with horizontal flip
    x_flip = torch.flip(x, [3])
    for k in range(4):
        x_rot = torch.rot90(x_flip, k, [2, 3]) if k > 0 else x_flip
        p = tiled_inference(model, x_rot)
        if k > 0:
            p = torch.rot90(p, -k, [2, 3])
        p = torch.flip(p, [3])
        preds.append(p)
    
    return torch.stack(preds).mean(dim=0)


def main():
    parser = argparse.ArgumentParser(description='Generate submission')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model')
    parser.add_argument('--output', type=str, default='submission.csv',
                        help='Output CSV path')
    parser.add_argument('--no-tta', action='store_true',
                        help='Disable TTA (faster but less accurate)')
    args = parser.parse_args()
    
    device = Config.DEVICE
    print(f"Device: {device}")
    print(f"Loading model: {args.model}")
    
    model = load_model(args.model, device)
    
    # Create dataset
    test_ds = RawImageDataset(
        Config.TEST_DIR,
        is_train=False,
        black_level=Config.BLACK_LEVEL
    )
    loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)
    
    results = []
    print(f"Processing {len(test_ds)} images...")
    
    with torch.no_grad():
        for x, file_id in tqdm(loader, desc="Inference"):
            x = x.to(device)
            
            if args.no_tta:
                pred = tiled_inference(model, x)
            else:
                pred = run_tta(model, x)
            
            # Convert to base64
            img_np = np.clip(pred[0].permute(1, 2, 0).cpu().numpy() * 255, 0, 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)
            buffer = io.BytesIO()
            pil_img.save(buffer, format="PNG")
            b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
            results.append([file_id[0], b64_str])
    
    # Save submission
    df = pd.DataFrame(results, columns=['ID', 'IMAGE'])
    try:
        df['ID_num'] = df['ID'].astype(int)
        df = df.sort_values('ID_num').drop('ID_num', axis=1)
    except:
        df = df.sort_values('ID')
    
    df.to_csv(args.output, index=False)
    print(f"✅ Submission saved to: {args.output}")


if __name__ == '__main__':
    main()
