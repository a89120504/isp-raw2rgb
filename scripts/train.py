# -*- coding: utf-8 -*-
"""
ISP Transformer Training Script (Refactored)
Target: Kaggle Top Tier + Moodle Submission
Hardware: NVIDIA RTX 3070 (8GB VRAM)

Usage:
    python scripts/train.py
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import io
import base64
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import from new modules
from src.config import TrainConfig as Config
from src.models import ISPNet
from src.data import RawImageDataset
from src.losses import CompositeLoss, DeltaELoss
from src.utils import tiled_inference, AverageMeter, setup_logger, seed_everything


# Create output directories
os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(Config.OUTPUT_DIR, 'vis_results'), exist_ok=True)
os.makedirs(os.path.join(Config.OUTPUT_DIR, 'checkpoints'), exist_ok=True)

# Setup logger
logger = setup_logger(Config.LOG_FILE)


class ISPTrainer:
    """ISP Model Trainer"""
    
    def __init__(self):
        self.device = Config.DEVICE
        self.model = ISPNet(
            dim=Config.DIM,
            enc_blocks=Config.ENC_BLOCKS,
            mid_blocks=Config.MID_BLOCKS,
            dec_blocks=Config.DEC_BLOCKS,
            heads=Config.HEADS
        ).to(self.device)
        
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=Config.LEARNING_RATE, 
            weight_decay=Config.WEIGHT_DECAY
        )
        self.scaler = GradScaler()
        
        # Learning rate scheduler with warmup
        warmup = LambdaLR(
            self.optimizer, 
            lr_lambda=lambda epoch: (epoch + 1) / Config.WARMUP_EPOCHS
        )
        cosine = CosineAnnealingLR(
            self.optimizer, 
            T_max=Config.EPOCHS - Config.WARMUP_EPOCHS, 
            eta_min=Config.MIN_LR
        )
        self.scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.optimizer, 
            schedulers=[warmup, cosine], 
            milestones=[Config.WARMUP_EPOCHS]
        )
        
        self.criterion = CompositeLoss(
            l1_weight=Config.L1_WEIGHT,
            ssim_weight=Config.SSIM_WEIGHT,
            delta_e_weight=Config.DELTA_E_WEIGHT,
            device=self.device
        ).to(self.device)
        
        self.metric_de = DeltaELoss().to(self.device)
        self.best_score = float('inf')
        self.start_epoch = 0
        self.top_models = []
        
        self.history = {'train_loss': [], 'val_score': [], 'lr': []}
        
        # Try torch.compile on non-Windows
        if hasattr(torch, 'compile') and sys.platform != 'win32':
            try:
                self.model = torch.compile(self.model)
            except:
                pass
    
    def _save_vis(self, pred, gt, epoch):
        """Save visualization of prediction vs ground truth"""
        def tensor2img(t):
            return np.clip(t.permute(1, 2, 0).cpu().numpy() * 255, 0, 255).astype(np.uint8)
        
        p_img = tensor2img(pred)
        g_img = tensor2img(gt)
        combined = np.hstack((p_img, g_img))
        Image.fromarray(combined).save(
            os.path.join(Config.OUTPUT_DIR, 'vis_results', f'epoch_{epoch:03d}.png')
        )
    
    def load_checkpoint(self, path):
        """Load training checkpoint"""
        if os.path.exists(path):
            logger.info(f"Loading checkpoint: {path}")
            ckpt = torch.load(path, map_location=self.device)
            state_dict = ckpt['state_dict']
            
            # Handle torch.compile prefixes
            if hasattr(self.model, "_orig_mod") and not list(state_dict.keys())[0].startswith("_orig_mod"):
                self.model._orig_mod.load_state_dict(state_dict)
            elif not hasattr(self.model, "_orig_mod") and list(state_dict.keys())[0].startswith("_orig_mod"):
                new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
                self.model.load_state_dict(new_state_dict)
            else:
                self.model.load_state_dict(state_dict)
            
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.scheduler.load_state_dict(ckpt['scheduler'])
            self.scaler.load_state_dict(ckpt['scaler'])
            self.start_epoch = ckpt['epoch'] + 1
            self.best_score = ckpt.get('best_score', float('inf'))
            logger.info(f"Resumed from epoch {self.start_epoch}, Best Score: {self.best_score:.4f}")
    
    def save_checkpoint(self, epoch, val_score):
        """Save training checkpoint"""
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
        torch.save(ckpt, os.path.join(Config.OUTPUT_DIR, 'last.pth'))
        
        # Save model checkpoint
        save_path = os.path.join(
            Config.OUTPUT_DIR, 'checkpoints', 
            f'epoch_{epoch:03d}_score_{val_score:.4f}.pth'
        )
        torch.save(state_dict, save_path)
        
        # Keep top 3 models
        self.top_models.append((val_score, save_path))
        self.top_models.sort(key=lambda x: x[0])
        if len(self.top_models) > 3:
            worst = self.top_models.pop()
            if os.path.exists(worst[1]):
                os.remove(worst[1])
        
        if val_score <= self.top_models[0][0]:
            logger.info(f"🏆 New Best Score: {val_score:.4f}")
    
    def plot_metrics(self):
        """Plot training metrics"""
        try:
            if len(self.history['train_loss']) == 0:
                return
            
            epochs = range(1, len(self.history['train_loss']) + 1)
            if self.start_epoch > 0:
                display_epochs = range(
                    self.start_epoch + 1, 
                    self.start_epoch + len(self.history['train_loss']) + 1
                )
            else:
                display_epochs = epochs
            
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
            
            ax1.plot(display_epochs, self.history['train_loss'], 'b-')
            ax1.set_title('Training Loss')
            ax1.set_xlabel('Epochs')
            ax1.grid(True, alpha=0.3)
            
            ax2.plot(display_epochs, self.history['val_score'], 'r-')
            ax2.set_title('Validation Delta E (Lower is Better)')
            ax2.set_xlabel('Epochs')
            ax2.grid(True, alpha=0.3)
            
            ax3.plot(display_epochs, self.history['lr'], 'g-')
            ax3.set_title('Learning Rate Schedule')
            ax3.set_xlabel('Epochs')
            ax3.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(Config.OUTPUT_DIR, 'training_metrics.png'))
            plt.close()
        except Exception as e:
            logger.warning(f"Failed to plot metrics: {e}")
    
    def train_epoch(self, loader, epoch):
        """Train for one epoch"""
        self.model.train()
        losses = AverageMeter()
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS} [Train]")
        
        for i, (x, y) in enumerate(pbar):
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            
            with autocast('cuda'):
                pred = self.model(x)
                use_warmup = epoch < Config.WARMUP_EPOCHS
                loss = self.criterion(pred, y, use_warmup) / Config.GRAD_ACCUM_STEPS
            
            self.scaler.scale(loss).backward()
            
            if (i + 1) % Config.GRAD_ACCUM_STEPS == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
            
            losses.update(loss.item() * Config.GRAD_ACCUM_STEPS)
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
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS} [Val]")
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
        """Full training loop"""
        for epoch in range(self.start_epoch, Config.EPOCHS):
            train_loss = self.train_epoch(train_loader, epoch)
            val_score = self.validate(val_loader, epoch)
            
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['train_loss'].append(train_loss)
            self.history['val_score'].append(val_score)
            self.history['lr'].append(current_lr)
            
            self.scheduler.step()
            logger.info(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val DeltaE: {val_score:.4f}")
            
            self.plot_metrics()
            self.save_checkpoint(epoch, val_score)
    
    def model_averaging(self):
        """Average top models for final submission"""
        logger.info("Performing Model Averaging...")
        if len(self.top_models) == 0:
            return
        
        avg_state_dict = {}
        for i, (_, path) in enumerate(self.top_models):
            logger.info(f"Averaging: {path}")
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
        
        logger.info("Model Averaging Complete.")
        
        # Save as best_model.pt for Moodle
        save_path = os.path.join(Config.OUTPUT_DIR, 'best_model.pt')
        torch.save(self.model.state_dict(), save_path)
        logger.info(f"💾 Averaged model saved as: {save_path}")
    
    def inference(self):
        """Generate submission file"""
        self.model_averaging()
        self.model.eval()
        
        test_ds = RawImageDataset(
            Config.TEST_DIR, 
            is_train=False,
            black_level=Config.BLACK_LEVEL
        )
        loader = DataLoader(
            test_ds, 
            batch_size=1, 
            shuffle=False, 
            num_workers=Config.NUM_WORKERS
        )
        
        results = []
        logger.info("Generating submission with TTA...")
        
        with torch.no_grad():
            for x, file_id in tqdm(loader, desc="Inference"):
                x = x.to(self.device)
                preds = []
                
                # 8x TTA: 4 rotations + 4 rotations with horizontal flip
                for k in range(4):
                    x_rot = torch.rot90(x, k, [2, 3]) if k > 0 else x
                    p = tiled_inference(self.model, x_rot)
                    if k > 0:
                        p = torch.rot90(p, -k, [2, 3])
                    preds.append(p)
                
                x_flip = torch.flip(x, [3])
                for k in range(4):
                    x_rot = torch.rot90(x_flip, k, [2, 3]) if k > 0 else x_flip
                    p = tiled_inference(self.model, x_rot)
                    if k > 0:
                        p = torch.rot90(p, -k, [2, 3])
                    p = torch.flip(p, [3])
                    preds.append(p)
                
                avg_pred = torch.stack(preds).mean(dim=0)
                
                # Convert to base64
                img_np = np.clip(avg_pred[0].permute(1, 2, 0).cpu().numpy() * 255, 0, 255).astype(np.uint8)
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
        
        df.to_csv(Config.SUBMISSION_FILE, index=False)
        logger.info(f"Submission saved to {Config.SUBMISSION_FILE}")


if __name__ == '__main__':
    seed_everything(logger=logger)
    logger.info(f"Config: Device={Config.DEVICE}, Batch={Config.BATCH_SIZE}, 3-Level Deep Arch")
    
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
    trainer = ISPTrainer()
    last_ckpt = os.path.join(Config.OUTPUT_DIR, 'last.pth')
    if os.path.exists(last_ckpt):
        trainer.load_checkpoint(last_ckpt)
    
    trainer.fit(train_loader, val_loader)
    trainer.inference()
