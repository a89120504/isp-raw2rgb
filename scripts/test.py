# -*- coding: utf-8 -*-
"""
Ensemble Inference Script - Combine multiple models for better results

Usage:
    python scripts/test.py \
        --model-a experiments/ISP_Restormer_v4_Finetune/best_model_finetuned.pt \
        --model-b experiments/ISP_Restormer_v4_L1_Regression/best_model_l1.pt \
        --output submission_ensemble.csv
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
    
    new_weights = {}
    for k, v in weights.items():
        new_weights[k.replace("_orig_mod.", "")] = v
    
    model.load_state_dict(new_weights)
    model.eval()
    
    return model


def run_tta(model, x: torch.Tensor) -> torch.Tensor:
    """8x Test-Time Augmentation"""
    preds = []
    
    for k in range(4):
        x_rot = torch.rot90(x, k, [2, 3]) if k > 0 else x
        p = tiled_inference(model, x_rot)
        if k > 0:
            p = torch.rot90(p, -k, [2, 3])
        preds.append(p)
    
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
    parser = argparse.ArgumentParser(description='Ensemble Inference')
    parser.add_argument('--model-a', type=str, required=True,
                        help='Path to first model (e.g., color-optimized)')
    parser.add_argument('--model-b', type=str, required=True,
                        help='Path to second model (e.g., L1-optimized)')
    parser.add_argument('--weight-a', type=float, default=0.5,
                        help='Weight for model A (default: 0.5)')
    parser.add_argument('--output', type=str, default='submission_ensemble.csv',
                        help='Output CSV path')
    parser.add_argument('--no-tta', action='store_true',
                        help='Disable TTA')
    args = parser.parse_args()
    
    device = Config.DEVICE
    weight_b = 1.0 - args.weight_a
    
    print(f"Device: {device}")
    print(f"Model A: {args.model_a} (weight: {args.weight_a})")
    print(f"Model B: {args.model_b} (weight: {weight_b})")
    
    model_a = load_model(args.model_a, device)
    model_b = load_model(args.model_b, device)
    
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
        for x, file_id in tqdm(loader, desc="Ensemble Inference"):
            x = x.to(device)
            
            if args.no_tta:
                pred_a = tiled_inference(model_a, x)
                pred_b = tiled_inference(model_b, x)
            else:
                pred_a = run_tta(model_a, x)
                pred_b = run_tta(model_b, x)
            
            # Weighted ensemble
            pred = args.weight_a * pred_a + weight_b * pred_b
            
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
    print(f"✅ Ensemble submission saved to: {args.output}")


if __name__ == '__main__':
    main()
