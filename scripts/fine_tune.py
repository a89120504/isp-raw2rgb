# -*- coding: utf-8 -*-
"""
Fine-tuning Script (Unified)
Supports both Delta E optimization and L1 Regression modes

Usage:
    python scripts/fine_tune.py --mode color    # Delta E optimization
    python scripts/fine_tune.py --mode l1       # L1 Regression (PSNR focus)
"""

import os
import sys
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import from new modules
from src.config import FineTuneConfig, L1RegressionConfig
from src.models import ISPNet
from src.data import RawImageDataset
from src.losses import CompositeLoss, DeltaELoss
from src.utils import tiled_inference, AverageMeter, setup_logger, seed_everything


def get_config(mode: str):
    """Get configuration based on fine-tuning mode"""
    if mode == 'color':
        return FineTuneConfig
    elif mode == 'l1':
        return L1RegressionConfig
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'color' or 'l1'.")


class FineTuneTrainer:
    """Fine-tuning Trainer"""
    
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.device = config.DEVICE
        
        self.model = ISPNet(
            dim=config.DIM,
            enc_blocks=config.ENC_BLOCKS,
            mid_blocks=config.MID_BLOCKS,
            dec_blocks=config.DEC_BLOCKS,
            heads=config.HEADS
        ).to(self.device)
        
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        self.scaler = GradScaler()
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.EPOCHS,
            eta_min=config.MIN_LR
        )
        
        self.criterion = CompositeLoss(
            l1_weight=config.L1_WEIGHT,
            ssim_weight=config.SSIM_WEIGHT,
            delta_e_weight=config.DELTA_E_WEIGHT,
            device=self.device
        ).to(self.device)
        
        self.metric_de = DeltaELoss().to(self.device)
        self.best_score = float('inf')
        self.top_models = []
        self.history = {'train_loss': [], 'val_score': [], 'lr': []}
        
        if hasattr(torch, 'compile') and sys.platform != 'win32':
            try:
                self.model = torch.compile(self.model)
            except:
                pass
    
    def _save_vis(self, pred, gt, epoch):
        """Save visualization"""
        def tensor2img(t):
            return np.clip(t.permute(1, 2, 0).cpu().numpy() * 255, 0, 255).astype(np.uint8)
        
        p_img = tensor2img(pred)
        g_img = tensor2img(gt)
        combined = np.hstack((p_img, g_img))
        Image.fromarray(combined).save(
            os.path.join(self.config.OUTPUT_DIR, 'vis_results', f'epoch_{epoch:03d}.png')
        )
    
    def load_pretrained(self):
        """Load pre-trained weights"""
        path = self.config.PRETRAINED_PATH
        if os.path.exists(path):
            self.logger.info(f"🔥 Loading Pre-trained weights from: {path}")
            weights = torch.load(path, map_location=self.device)
            
            # Handle torch.compile prefixes
            new_weights = {}
            for k, v in weights.items():
                new_weights[k.replace("_orig_mod.", "")] = v
            
            self.model.load_state_dict(new_weights)
            self.logger.info("✅ Weights loaded successfully. Starting fine-tuning.")
        else:
            self.logger.error(f"❌ Pretrained model not found at {path}")
            sys.exit(1)
    
    def save_checkpoint(self, epoch, val_score):
        """Save checkpoint"""
        if hasattr(self.model, "_orig_mod"):
            state_dict = self.model._orig_mod.state_dict()
        else:
            state_dict = self.model.state_dict()
        
        ckpt = {
            'epoch': epoch,
            'state_dict': state_dict,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'scaler': self.scaler.state_dict(),
            'best_score': self.best_score
        }
        torch.save(ckpt, os.path.join(self.config.OUTPUT_DIR, 'last.pth'))
        
        save_path = os.path.join(
            self.config.OUTPUT_DIR, 'checkpoints',
            f'epoch_{epoch:03d}_score_{val_score:.4f}.pth'
        )
        torch.save(state_dict, save_path)
        
        self.top_models.append((val_score, save_path))
        self.top_models.sort(key=lambda x: x[0])
        if len(self.top_models) > 3:
            worst = self.top_models.pop()
            if os.path.exists(worst[1]):
                os.remove(worst[1])
        
        if val_score <= self.top_models[0][0]:
            self.logger.info(f"🏆 New Best Score: {val_score:.4f}")
    
    def plot_metrics(self):
        """Plot metrics"""
        try:
            if len(self.history['train_loss']) == 0:
                return
            
            epochs = range(1, len(self.history['train_loss']) + 1)
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
            
            ax1.plot(epochs, self.history['train_loss'], 'b-')
            ax1.set_title('Training Loss')
            ax1.grid(True, alpha=0.3)
            
            ax2.plot(epochs, self.history['val_score'], 'r-')
            ax2.set_title('Validation Delta E')
            ax2.grid(True, alpha=0.3)
            
            ax3.plot(epochs, self.history['lr'], 'g-')
            ax3.set_title('Learning Rate')
            ax3.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.config.OUTPUT_DIR, 'finetune_metrics.png'))
            plt.close()
        except:
            pass
    
    def train_epoch(self, loader, epoch):
        """Train one epoch"""
        self.model.train()
        losses = AverageMeter()
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{self.config.EPOCHS} [Finetune]")
        
        for i, (x, y) in enumerate(pbar):
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            
            with autocast('cuda'):
                pred = self.model(x)
                loss = self.criterion(pred, y) / self.config.GRAD_ACCUM_STEPS
            
            self.scaler.scale(loss).backward()
            
            if (i + 1) % self.config.GRAD_ACCUM_STEPS == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
            
            losses.update(loss.item() * self.config.GRAD_ACCUM_STEPS)
            pbar.set_postfix({
                'loss': f"{losses.avg:.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
        
        return losses.avg
    
    @torch.no_grad()
    def validate(self, loader, epoch):
        """Validate model"""
        self.model.eval()
        de_meter = AverageMeter()
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{self.config.EPOCHS} [Val]")
        vis_done = False
        
        for x, y in pbar:
            x, y = x.to(self.device), y.to(self.device)
            
            with autocast('cuda'):
                pred = tiled_inference(self.model, x)
                de = self.metric_de(pred, y)
            
            de_meter.update(de.item())
            
            if not vis_done:
                self._save_vis(pred[0], y[0], epoch)
                vis_done = True
        
        return de_meter.avg
    
    def fit(self, train_loader, val_loader):
        """Training loop"""
        for epoch in range(self.config.EPOCHS):
            train_loss = self.train_epoch(train_loader, epoch)
            val_score = self.validate(val_loader, epoch)
            
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['train_loss'].append(train_loss)
            self.history['val_score'].append(val_score)
            self.history['lr'].append(current_lr)
            
            self.scheduler.step()
            self.logger.info(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val DeltaE: {val_score:.4f}")
            
            self.plot_metrics()
            self.save_checkpoint(epoch, val_score)
    
    def model_averaging(self, output_name: str):
        """Average top models"""
        self.logger.info("Performing Model Averaging...")
        if len(self.top_models) == 0:
            return
        
        avg_state_dict = {}
        for i, (_, path) in enumerate(self.top_models):
            state = torch.load(path, map_location=self.device)
            for k, v in state.items():
                if i == 0:
                    avg_state_dict[k] = v
                else:
                    avg_state_dict[k] += v
        
        for k in avg_state_dict.keys():
            avg_state_dict[k] = avg_state_dict[k] / len(self.top_models)
        
        if hasattr(self.model, "_orig_mod"):
            self.model._orig_mod.load_state_dict(avg_state_dict)
        else:
            self.model.load_state_dict(avg_state_dict)
        
        save_path = os.path.join(self.config.OUTPUT_DIR, output_name)
        torch.save(self.model.state_dict(), save_path)
        self.logger.info(f"💾 Finetuned model saved as: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Fine-tune ISP Model')
    parser.add_argument('--mode', type=str, default='color', choices=['color', 'l1'],
                        help='Fine-tuning mode: color (Delta E) or l1 (PSNR)')
    args = parser.parse_args()
    
    # Get config
    Config = get_config(args.mode)
    
    # Create directories
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(Config.OUTPUT_DIR, 'vis_results'), exist_ok=True)
    os.makedirs(os.path.join(Config.OUTPUT_DIR, 'checkpoints'), exist_ok=True)
    
    # Setup logger
    logger = setup_logger(Config.LOG_FILE)
    seed_everything(logger=logger)
    
    logger.info(f"Fine-tuning Mode: {args.mode.upper()}")
    logger.info(f"LR: {Config.LEARNING_RATE}, Epochs: {Config.EPOCHS}")
    logger.info(f"Loss Weights - L1: {Config.L1_WEIGHT}, SSIM: {Config.SSIM_WEIGHT}, DeltaE: {Config.DELTA_E_WEIGHT}")
    
    # Create datasets
    full_ds = RawImageDataset(
        Config.TRAIN_DIR,
        is_train=True,
        transform=True,
        img_size=Config.IMG_SIZE,
        black_level=Config.BLACK_LEVEL,
        flip_prob=Config.FLIP_PROB,
        rot_prob=Config.ROT_PROB
    )
    
    train_len = int((1 - Config.VAL_SPLIT) * len(full_ds))
    val_len = len(full_ds) - train_len
    train_set, val_set = random_split(
        full_ds,
        [train_len, val_len],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(
        train_set,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        persistent_workers=(Config.NUM_WORKERS > 0)
    )
    
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY
    )
    
    # Train
    trainer = FineTuneTrainer(Config, logger)
    trainer.load_pretrained()
    trainer.fit(train_loader, val_loader)
    
    # Save averaged model
    output_name = 'best_model_finetuned.pt' if args.mode == 'color' else 'best_model_l1.pt'
    trainer.model_averaging(output_name)


if __name__ == '__main__':
    main()
