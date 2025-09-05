"""
VCC Transformer: High-Performance Multi-Task Transformer for Virtual Cell Challenge

A PyTorch implementation of a multi-task learning Transformer model for predicting 
gene expression profiles in response to single-gene knockout perturbations.
"""

__version__ = "0.1.0"
__author__ = "VCC Team"

from .models.transformer import MultiTaskTransformer
from .data.dataset import VCCDataset
from .training.trainer import VCCTrainer
from .training.losses import CombinedLoss
from .utils.config import load_config

__all__ = [
    "MultiTaskTransformer",
    "VCCDataset", 
    "VCCTrainer",
    "CombinedLoss",
    "load_config"
]
