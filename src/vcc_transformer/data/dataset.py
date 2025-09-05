"""
Dataset implementation for VCC Transformer.

This module handles loading and preprocessing of single-cell RNA-seq data
for the Virtual Cell Challenge, including control cells, perturbed cells,
and perturbation metadata.
"""

import numpy as np
import pandas as pd
import polars as pl
import scanpy as sc
import anndata as ad
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import logging

logger = logging.getLogger(__name__)


class VCCDataset(Dataset):
    """
    PyTorch Dataset for Virtual Cell Challenge data.
    
    This dataset handles:
    - Loading control and perturbed cells from .h5ad files
    - Selecting highly variable genes
    - Normalizing and scaling expression data
    - Creating perturbation-to-index mappings
    - Generating input sequences for the transformer
    """
    
    def __init__(
        self,
        config,
        mode: str = "train",
        validation_perturbations: Optional[List[str]] = None
    ):
        """
        Initialize the VCC Dataset.
        
        Args:
            config: Configuration object containing data parameters
            mode: Dataset mode - "train", "val", or "predict"
            validation_perturbations: List of perturbations for validation mode
        """
        self.config = config
        self.mode = mode
        self.validation_perturbations = validation_perturbations
        
        # Initialize data containers
        self.adata = None
        self.control_cells = None
        self.perturbed_cells = None
        self.hvg_indices = None
        self.perturbation_to_idx = {}
        self.idx_to_perturbation = {}
        self.scaler = None
        
        # Load and preprocess data
        self._load_data()
        self._preprocess_data()
        self._create_pairs()
        
        logger.info(f"Dataset initialized in {mode} mode with {len(self)} samples")
    
    def _load_data(self) -> None:
        """Load data from .h5ad file."""
        data_path = Path(self.config.data.training_file)
        if not data_path.exists():
            raise FileNotFoundError(f"Training data not found: {data_path}")
        
        logger.info(f"Loading data from {data_path}")
        self.adata = sc.read_h5ad(data_path)
        
        # Log basic statistics
        logger.info(f"Loaded {self.adata.n_obs} cells, {self.adata.n_vars} genes")
        logger.info(f"Perturbations: {self.adata.obs['perturbation'].nunique()}")
    
    def _select_highly_variable_genes(self) -> None:
        """Select highly variable genes from the training data."""
        logger.info("Selecting highly variable genes...")
        
        # Use scanpy to identify highly variable genes
        sc.pp.highly_variable_genes(
            self.adata,
            n_top_genes=self.config.data.n_highly_variable_genes,
            flavor='seurat_v3'
        )
        
        # Get indices of highly variable genes
        hvg_mask = self.adata.var['highly_variable'].values
        self.hvg_indices = np.where(hvg_mask)[0]
        
        # Store gene names for reference
        self.hvg_names = self.adata.var_names[hvg_mask].tolist()
        
        logger.info(f"Selected {len(self.hvg_indices)} highly variable genes")
    
    def _normalize_data(self) -> None:
        """Normalize gene expression data."""
        logger.info("Normalizing expression data...")
        
        # Apply normalization method from config
        if self.config.data.normalize_method == "log1p":
            # Standard log1p normalization
            self.adata.X = np.log1p(self.adata.X)
        elif self.config.data.normalize_method == "zscore":
            # Z-score normalization per gene
            self.adata.X = (self.adata.X - np.mean(self.adata.X, axis=0)) / np.std(self.adata.X, axis=0)
        
        # Apply scaling method
        if self.config.data.scale_method == "standard":
            self.scaler = StandardScaler()
            # Only fit on training data, transform accordingly
            if self.mode == "train":
                self.adata.X = self.scaler.fit_transform(self.adata.X)
            else:
                self.adata.X = self.scaler.transform(self.adata.X)
        elif self.config.data.scale_method == "minmax":
            self.scaler = MinMaxScaler()
            if self.mode == "train":
                self.adata.X = self.scaler.fit_transform(self.adata.X)
            else:
                self.adata.X = self.scaler.transform(self.adata.X)
    
    def _create_perturbation_mapping(self) -> None:
        """Create mapping between perturbations and indices."""
        # Get unique perturbations
        perturbations = self.adata.obs['perturbation'].unique()
        
        # Reserve special indices
        # 0: CLS token, 1: UNK_PERT token
        self.perturbation_to_idx = {
            '<CLS>': self.config.model.cls_token_id,
            '<UNK_PERT>': self.config.model.unk_pert_token_id
        }
        
        # Add regular perturbations starting from index 2
        for i, pert in enumerate(sorted(perturbations)):
            if pert != 'Non-Targeting Control':  # Handle NTC specially
                self.perturbation_to_idx[pert] = i + 2
        
        # Create reverse mapping
        self.idx_to_perturbation = {v: k for k, v in self.perturbation_to_idx.items()}
        
        logger.info(f"Created mapping for {len(self.perturbation_to_idx)} perturbations")
    
    def _preprocess_data(self) -> None:
        """Run all preprocessing steps."""
        self._select_highly_variable_genes()
        self._normalize_data()
        self._create_perturbation_mapping()
        
        # Filter to highly variable genes
        self.adata = self.adata[:, self.hvg_indices].copy()
    
    def _create_pairs(self) -> None:
        """Create control-perturbation pairs for training."""
        # Separate control and perturbed cells
        control_mask = self.adata.obs['perturbation'] == 'Non-Targeting Control'
        self.control_cells = self.adata[control_mask].copy()
        self.perturbed_cells = self.adata[~control_mask].copy()
        
        # Create training pairs
        self.pairs = []
        
        if self.mode == "train":
            self._create_training_pairs()
        elif self.mode == "val":
            self._create_validation_pairs()
        elif self.mode == "predict":
            self._create_prediction_pairs()
    
    def _create_training_pairs(self) -> None:
        """Create training pairs from control and perturbed cells."""
        # Group perturbed cells by perturbation
        perturbation_groups = self.perturbed_cells.obs.groupby('perturbation')
        
        for perturbation, group_idx in perturbation_groups.indices.items():
            # Skip if too few or too many cells for this perturbation
            n_cells = len(group_idx)
            if n_cells < self.config.data.min_cells_per_perturbation:
                continue
            if n_cells > self.config.data.max_cells_per_perturbation:
                # Randomly sample cells
                group_idx = np.random.choice(
                    group_idx, 
                    self.config.data.max_cells_per_perturbation, 
                    replace=False
                )
            
            # Create pairs with random control cells
            for pert_idx in group_idx:
                control_idx = np.random.choice(len(self.control_cells))
                
                self.pairs.append({
                    'control_idx': control_idx,
                    'perturbed_idx': pert_idx,
                    'perturbation': perturbation,
                    'perturbation_idx': self.perturbation_to_idx.get(perturbation, 
                                                                   self.config.model.unk_pert_token_id)
                })
    
    def _create_validation_pairs(self) -> None:
        """Create validation pairs for specified perturbations."""
        if self.validation_perturbations is None:
            # Use a random subset of perturbations for validation
            all_perts = self.perturbed_cells.obs['perturbation'].unique()
            n_val_perts = int(len(all_perts) * self.config.data.validation_split)
            self.validation_perturbations = np.random.choice(
                all_perts, n_val_perts, replace=False
            ).tolist()
        
        # Create pairs only for validation perturbations
        for perturbation in self.validation_perturbations:
            pert_mask = self.perturbed_cells.obs['perturbation'] == perturbation
            pert_indices = np.where(pert_mask)[0]
            
            for pert_idx in pert_indices:
                control_idx = np.random.choice(len(self.control_cells))
                
                self.pairs.append({
                    'control_idx': control_idx,
                    'perturbed_idx': pert_idx,
                    'perturbation': perturbation,
                    'perturbation_idx': self.perturbation_to_idx.get(perturbation,
                                                                   self.config.model.unk_pert_token_id)
                })
    
    def _create_prediction_pairs(self) -> None:
        """Create pairs for prediction mode using validation file."""
        # Load validation perturbations from CSV
        val_file = Path(self.config.data.validation_file)
        if not val_file.exists():
            raise FileNotFoundError(f"Validation file not found: {val_file}")
        
        # Use polars for fast CSV reading
        val_df = pl.read_csv(val_file)
        required_perturbations = val_df['perturbation'].unique().to_list()
        
        # Create prediction pairs
        for perturbation in required_perturbations:
            # Use multiple control cells for robust predictions
            n_controls = min(50, len(self.control_cells))  # Use up to 50 control cells
            control_indices = np.random.choice(
                len(self.control_cells), n_controls, replace=False
            )
            
            for control_idx in control_indices:
                self.pairs.append({
                    'control_idx': control_idx,
                    'perturbed_idx': None,  # No ground truth for prediction
                    'perturbation': perturbation,
                    'perturbation_idx': self.perturbation_to_idx.get(perturbation,
                                                                   self.config.model.unk_pert_token_id)
                })
    
    def _create_sequence(self, control_expr: np.ndarray, perturbation_idx: int) -> torch.Tensor:
        """
        Create input sequence for transformer.
        
        Args:
            control_expr: Control cell expression values
            perturbation_idx: Index of perturbation
            
        Returns:
            torch.Tensor: Input sequence [CLS, PERT, gene1, gene2, ...]
        """
        # Create sequence
        sequence = np.zeros(self.config.model.max_seq_length)
        
        # Add special tokens
        sequence[0] = self.config.model.cls_token_id  # CLS token
        sequence[1] = perturbation_idx  # PERT token
        
        # Add gene expression values (starting from index 2)
        n_genes = len(control_expr)
        sequence[2:2+n_genes] = control_expr
        
        return torch.tensor(sequence, dtype=torch.float32)
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Dict containing input sequence, target expression, and perturbation label
        """
        pair = self.pairs[idx]
        
        # Get control cell expression
        control_expr = self.control_cells.X[pair['control_idx']].toarray().flatten()
        
        # Apply unknown perturbation masking during training
        perturbation_idx = pair['perturbation_idx']
        if (self.mode == "train" and 
            np.random.random() < self.config.model.unk_pert_probability):
            perturbation_idx = self.config.model.unk_pert_token_id
        
        # Create input sequence
        input_sequence = self._create_sequence(control_expr, perturbation_idx)
        
        # Get target data
        if pair['perturbed_idx'] is not None:
            # Training/validation mode - we have ground truth
            target_expr = self.perturbed_cells.X[pair['perturbed_idx']].toarray().flatten()
            target_expr = torch.tensor(target_expr, dtype=torch.float32)
        else:
            # Prediction mode - no ground truth
            target_expr = torch.zeros_like(torch.tensor(control_expr))
        
        # Perturbation label for classification
        pert_label = torch.tensor(pair['perturbation_idx'], dtype=torch.long)
        
        return {
            'input_sequence': input_sequence,
            'target_expression': target_expr,
            'perturbation_label': pert_label,
            'perturbation_name': pair['perturbation'],
            'control_expression': torch.tensor(control_expr, dtype=torch.float32)
        }
    
    def get_dataloader(
        self, 
        batch_size: Optional[int] = None,
        shuffle: Optional[bool] = None,
        **kwargs
    ) -> DataLoader:
        """
        Create a DataLoader for this dataset.
        
        Args:
            batch_size: Batch size (uses config default if None)
            shuffle: Whether to shuffle (auto-determined if None)
            **kwargs: Additional DataLoader arguments
            
        Returns:
            DataLoader: Configured PyTorch DataLoader
        """
        if batch_size is None:
            batch_size = self.config.training.batch_size
        
        if shuffle is None:
            shuffle = self.mode == "train"
        
        # Default dataloader settings from config
        dataloader_kwargs = {
            'batch_size': batch_size,
            'shuffle': shuffle,
            'num_workers': self.config.dataloader.num_workers,
            'pin_memory': self.config.dataloader.pin_memory,
            'persistent_workers': self.config.dataloader.persistent_workers and 
                                 self.config.dataloader.num_workers > 0,
            'prefetch_factor': self.config.dataloader.prefetch_factor
        }
        
        # Override with any provided kwargs
        dataloader_kwargs.update(kwargs)
        
        return DataLoader(self, **dataloader_kwargs)
    
    def save_preprocessing_info(self, save_path: Union[str, Path]) -> None:
        """
        Save preprocessing information for later use.
        
        Args:
            save_path: Path to save preprocessing info
        """
        save_path = Path(save_path)
        preprocessing_info = {
            'hvg_indices': self.hvg_indices.tolist(),
            'hvg_names': self.hvg_names,
            'perturbation_to_idx': self.perturbation_to_idx,
            'scaler_params': None
        }
        
        # Save scaler parameters if available
        if self.scaler is not None:
            if hasattr(self.scaler, 'mean_'):
                preprocessing_info['scaler_params'] = {
                    'type': type(self.scaler).__name__,
                    'mean': self.scaler.mean_.tolist(),
                    'scale': self.scaler.scale_.tolist()
                }
        
        # Save as numpy archive
        np.savez(save_path, **preprocessing_info)
        logger.info(f"Saved preprocessing info to {save_path}")


def create_datasets(config, validation_split: float = 0.1) -> Tuple[VCCDataset, VCCDataset]:
    """
    Create training and validation datasets.
    
    Args:
        config: Configuration object
        validation_split: Fraction of perturbations to use for validation
        
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    # Create training dataset first
    train_dataset = VCCDataset(config, mode="train")
    
    # Get perturbations for validation
    all_perturbations = list(train_dataset.perturbation_to_idx.keys())
    all_perturbations = [p for p in all_perturbations if not p.startswith('<')]
    
    n_val_perts = int(len(all_perturbations) * validation_split)
    val_perturbations = np.random.choice(
        all_perturbations, n_val_perts, replace=False
    ).tolist()
    
    # Create validation dataset
    val_dataset = VCCDataset(
        config, 
        mode="val", 
        validation_perturbations=val_perturbations
    )
    
    return train_dataset, val_dataset
