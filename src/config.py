# -*- coding: utf-8 -*-
"""
Unified Configuration Module
Supports different training modes via inheritance
"""

import os
import torch


def find_data_root():
    """自動尋找資料集根目錄"""
    candidates = [
        './dataset/dlhw3_public', './dlhw3_public',
        '../dataset/dlhw3_public', './data/dlhw3_public',
        'C:/dataset/dlhw3_public', 'D:/dataset/dlhw3_public'
    ]
    for p in candidates:
        if os.path.exists(os.path.join(p, 'test')):
            return p
    return './dataset/dlhw3_public'


class BaseConfig:
    """基礎設定 - 所有訓練模式共用"""
    
    # --- Paths ---
    DATA_ROOT = find_data_root()
    TRAIN_DIR = os.path.join(DATA_ROOT, 'train')
    TEST_DIR = os.path.join(DATA_ROOT, 'test')
    
    # --- Device ---
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_WORKERS = 2
    PIN_MEMORY = True
    
    # --- Model Architecture (3-Level Restormer) ---
    DIM = 48
    ENC_BLOCKS = [4, 4, 6]
    MID_BLOCKS = 4
    DEC_BLOCKS = [4, 4]
    HEADS = [1, 2, 4, 8]
    
    # --- ISP Specific ---
    BLACK_LEVEL = 64.0
    
    # --- Augmentation ---
    FLIP_PROB = 0.5
    ROT_PROB = 0.5
    
    # --- Training (Default) ---
    IMG_SIZE = 512
    BATCH_SIZE = 6
    GRAD_ACCUM_STEPS = 3
    VAL_SPLIT = 0.1


class TrainConfig(BaseConfig):
    """主訓練設定"""
    
    EXP_NAME = "ISP_Restormer_v4_Final"
    OUTPUT_DIR = os.path.join('./experiments', EXP_NAME)
    LOG_FILE = os.path.join(OUTPUT_DIR, 'train.log')
    SUBMISSION_FILE = 'submission.csv'
    
    # --- Hyperparameters ---
    EPOCHS = 200
    LEARNING_RATE = 3e-4
    MIN_LR = 1e-6
    WARMUP_EPOCHS = 15
    WEIGHT_DECAY = 1e-4
    
    # --- Loss Weights (After warmup) ---
    L1_WEIGHT = 1.0
    SSIM_WEIGHT = 0.25
    DELTA_E_WEIGHT = 0.15


class FineTuneConfig(BaseConfig):
    """微調設定 - Delta E 優化"""
    
    PRETRAINED_PATH = './experiments/ISP_Restormer_v4_Final/best_model.pt'
    
    EXP_NAME = "ISP_Restormer_v4_Finetune"
    OUTPUT_DIR = os.path.join('./experiments', EXP_NAME)
    LOG_FILE = os.path.join(OUTPUT_DIR, 'finetune.log')
    
    # --- Hyperparameters ---
    EPOCHS = 50
    LEARNING_RATE = 1e-5
    MIN_LR = 1e-7
    WARMUP_EPOCHS = 0
    WEIGHT_DECAY = 1e-4
    
    # --- Loss Weights (High color penalty) ---
    L1_WEIGHT = 1.0
    SSIM_WEIGHT = 0.2
    DELTA_E_WEIGHT = 0.3


class L1RegressionConfig(BaseConfig):
    """L1 回歸微調設定 - PSNR 優化"""
    
    PRETRAINED_PATH = './experiments/ISP_Restormer_v4_Finetune/best_model_finetuned.pt'
    
    EXP_NAME = "ISP_Restormer_v4_L1_Regression"
    OUTPUT_DIR = os.path.join('./experiments', EXP_NAME)
    LOG_FILE = os.path.join(OUTPUT_DIR, 'l1_regression.log')
    
    # --- Hyperparameters ---
    EPOCHS = 20
    LEARNING_RATE = 5e-6
    MIN_LR = 1e-7
    WARMUP_EPOCHS = 0
    WEIGHT_DECAY = 1e-4
    
    # --- Loss Weights (Heavy L1 bias) ---
    L1_WEIGHT = 2.0
    SSIM_WEIGHT = 0.05
    DELTA_E_WEIGHT = 0.1
