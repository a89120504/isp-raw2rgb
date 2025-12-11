# -*- coding: utf-8 -*-
"""
Logging and Reproducibility Utilities
"""

import os
import random
import logging

import numpy as np
import torch


def setup_logger(log_file: str) -> logging.Logger:
    """
    Setup logging to file and console
    
    Args:
        log_file: Path to log file
        
    Returns:
        Configured logger
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()
    
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    
    # File handler
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    return logger


def seed_everything(seed: int = 42, logger: logging.Logger = None):
    """
    Set random seeds for reproducibility
    
    Args:
        seed: Random seed value
        logger: Optional logger for logging seed info
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
    if logger:
        logger.info(f"Global Seed set to {seed}")
