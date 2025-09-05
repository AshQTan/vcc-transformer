"""
Trainer for VCC Transformer with distributed training support.

This module implements the training loop with support for:
- Distributed Data Parallel (DDP)
- Automatic Mixed Precision (AMP)
- Gradient checkpointing
- Comprehensive logging and checkpointing
"""

import os
import time
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Any
import numpy as np

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import GradScaler, autocast

from ..data.dataset import VCCDataset, create_datasets
from ..models.transformer import MultiTaskTransformer, create_model
from ..training.losses import CombinedLoss, compute_challenge_metrics
from ..utils.config import get_device_config, setup_directories
from ..utils.visualization import TrainingProgressTracker

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

logger = logging.getLogger(__name__)


class VCCTrainer:
    """
    High-performance trainer for VCC Transformer with multi-GPU support.
    """
    
    def __init__(self, config, model: Optional[nn.Module] = None):
        """
        Initialize the trainer.
        
        Args:
            config: Configuration object
            model: Pre-initialized model (optional)
        """
        self.config = config
        self.device_config = get_device_config(config)
        
        # Setup distributed training
        self.is_distributed = self.device_config['world_size'] > 1
        self.local_rank = self.device_config['local_rank']
        self.device = torch.device(f"cuda:{self.local_rank}" if self.device_config['use_cuda'] else "cpu")
        
        if self.is_distributed:
            self._setup_distributed()
        
        # Initialize model
        self.model = model or create_model(config)
        self.model.to(self.device)
        
        # Compile model if requested
        if config.training.compile_model:
            self.model = torch.compile(
                self.model,
                mode=config.training.compile_mode,
                fullgraph=False
            )
            logger.info("Model compiled successfully")
        
        # Wrap model for distributed training
        if self.is_distributed:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=config.distributed.find_unused_parameters
            )
        
        # Initialize training components
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.criterion = None
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = float('inf')
        self.training_stats = []
        
        # Progress tracker
        self.progress_tracker = None
        
        # Setup directories
        setup_directories(config)
        
        # Initialize logging
        self._setup_logging()
    
    def _setup_distributed(self):
        """Setup distributed training."""
        if 'RANK' not in os.environ:
            os.environ['RANK'] = str(self.local_rank)
        if 'WORLD_SIZE' not in os.environ:
            os.environ['WORLD_SIZE'] = str(self.device_config['world_size'])
        
        dist.init_process_group(
            backend=self.config.distributed.backend,
            init_method='env://'
        )
        
        logger.info(f"Distributed training initialized. Rank: {self.local_rank}, World size: {self.device_config['world_size']}")
    
    def _setup_logging(self):
        """Setup logging and monitoring."""
        if self.local_rank == 0:  # Only log from main process
            if self.config.logging.use_wandb and WANDB_AVAILABLE:
                wandb.init(
                    project=self.config.logging.wandb_project,
                    entity=self.config.logging.wandb_entity,
                    config=dict(self.config),
                    name=f"vcc-transformer-{int(time.time())}"
                )
                logger.info("Weights & Biases logging initialized")
    
    def _setup_training_components(self):
        """Initialize optimizer, scheduler, and loss function."""
        # Optimizer
        if self.config.training.optimizer == "adamw":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.training.optimizer}")
        
        # Loss function
        self.criterion = CombinedLoss(
            reconstruction_weight=self.config.training.reconstruction_weight,
            classification_weight=self.config.training.classification_weight
        )
        
        # AMP scaler
        if self.config.training.use_amp:
            self.scaler = GradScaler()
        
        # Learning rate scheduler
        if self.config.training.scheduler == "cosine_with_warmup":
            from torch.optim.lr_scheduler import CosineAnnealingLR
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training.max_epochs,
                eta_min=self.config.training.learning_rate * 0.01
            )
        
        logger.info("Training components initialized")
    
    def train(
        self,
        train_dataset: Optional[VCCDataset] = None,
        val_dataset: Optional[VCCDataset] = None
    ):
        """
        Main training loop.
        
        Args:
            train_dataset: Training dataset (will create if None)
            val_dataset: Validation dataset (will create if None)
        """
        logger.info("Starting training...")
        
        # Create datasets if not provided
        if train_dataset is None or val_dataset is None:
            train_dataset, val_dataset = create_datasets(self.config)
        
        # Create data loaders
        train_loader = self._create_dataloader(train_dataset, shuffle=True)
        val_loader = self._create_dataloader(val_dataset, shuffle=False) if val_dataset else None
        
        # Setup training components
        self._setup_training_components()
        
        # Initialize progress tracker
        steps_per_epoch = len(train_loader)
        self.progress_tracker = TrainingProgressTracker(
            config=self.config,
            total_epochs=self.config.training.max_epochs,
            steps_per_epoch=steps_per_epoch
        )
        
        # Start beautiful progress tracking
        self.progress_tracker.start_training()
        
        # Training loop
        for epoch in range(self.current_epoch, self.config.training.max_epochs):
            self.current_epoch = epoch
            
            # Update progress tracker
            self.progress_tracker.update_epoch(epoch)
            
            # Training phase
            train_metrics = self._train_epoch(train_loader)
            
            # Validation phase
            val_metrics = {}
            if val_loader and self.config.validation.run_validation:
                if epoch % self.config.validation.val_every_n_epochs == 0:
                    val_metrics = self._validate_epoch(val_loader)
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
            
            # End epoch in progress tracker
            self.progress_tracker.end_epoch(train_metrics, val_metrics)
            
            # Log metrics
            self._log_epoch_metrics(epoch, train_metrics, val_metrics)
            
            # Save checkpoint
            if epoch % self.config.checkpointing.save_every_n_epochs == 0:
                self._save_checkpoint(epoch, val_metrics.get('val_combined_loss', float('inf')))
            
            # Early stopping check
            if self._should_stop_early(val_metrics):
                logger.info("Early stopping triggered")
                break
        
        # End beautiful progress tracking
        self.progress_tracker.end_training()
        
        # Save training history
        history_path = Path(self.config.logging.log_dir) / "training_history.json"
        self.progress_tracker.save_history(history_path)
        
        logger.info("Training completed")
    
    def _create_dataloader(self, dataset: VCCDataset, shuffle: bool) -> DataLoader:
        """Create data loader with appropriate sampler."""
        sampler = None
        if self.is_distributed:
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.device_config['world_size'],
                rank=self.local_rank,
                shuffle=shuffle
            )
            shuffle = False  # Sampler handles shuffling
        
        return DataLoader(
            dataset,
            batch_size=self.config.training.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.config.dataloader.num_workers,
            pin_memory=self.config.dataloader.pin_memory,
            persistent_workers=self.config.dataloader.persistent_workers and self.config.dataloader.num_workers > 0,
            prefetch_factor=self.config.dataloader.prefetch_factor
        )
    
    def _train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        total_recon_loss = 0.0
        total_class_loss = 0.0
        num_batches = 0
        
        if self.is_distributed:
            train_loader.sampler.set_epoch(self.current_epoch)
        
        for batch_idx, batch in enumerate(train_loader):
            batch = self._move_batch_to_device(batch)
            
            # Forward pass with AMP
            with autocast(enabled=self.config.training.use_amp):
                outputs = self.model(batch['input_sequence'])
                loss_dict = self.criterion(
                    outputs['reconstruction'],
                    batch['target_expression'],
                    outputs['classification'],
                    batch['perturbation_label']
                )
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.config.training.use_amp:
                self.scaler.scale(loss_dict['total_loss']).backward()
                
                # Gradient clipping
                if self.config.training.gradient_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.training.gradient_clip_norm
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss_dict['total_loss'].backward()
                
                # Gradient clipping
                if self.config.training.gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.training.gradient_clip_norm
                    )
                
                self.optimizer.step()
            
            # Update statistics
            total_loss += loss_dict['total_loss'].item()
            total_recon_loss += loss_dict['reconstruction_loss'].item()
            total_class_loss += loss_dict['classification_loss'].item()
            num_batches += 1
            self.global_step += 1
            
            # Log step metrics
            if batch_idx % self.config.logging.log_every_n_steps == 0:
                self._log_step_metrics(batch_idx, loss_dict, len(train_loader))
                
                # Update progress tracker with current metrics
                current_metrics = {
                    'loss': loss_dict['total_loss'].item(),
                    'recon_loss': loss_dict['reconstruction_loss'].item(),
                    'class_loss': loss_dict['classification_loss'].item(),
                    'lr': self.optimizer.param_groups[0]['lr']
                }
                self.progress_tracker.update_step(batch_idx, current_metrics)
            
            # Memory management
            if self.global_step % self.config.memory.empty_cache_every_n_steps == 0:
                torch.cuda.empty_cache()
        
        return {
            'train_loss': total_loss / num_batches,
            'train_recon_loss': total_recon_loss / num_batches,
            'train_class_loss': total_class_loss / num_batches
        }
    
    def _validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        
        total_loss = 0.0
        total_recon_loss = 0.0
        total_class_loss = 0.0
        all_predictions = []
        all_targets = []
        all_pert_labels = []
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = self._move_batch_to_device(batch)
                
                with autocast(enabled=self.config.training.use_amp):
                    outputs = self.model(batch['input_sequence'])
                    loss_dict = self.criterion(
                        outputs['reconstruction'],
                        batch['target_expression'],
                        outputs['classification'],
                        batch['perturbation_label']
                    )
                
                # Collect predictions for challenge metrics
                all_predictions.append(outputs['reconstruction'].cpu())
                all_targets.append(batch['target_expression'].cpu())
                all_pert_labels.append(batch['perturbation_label'].cpu())
                
                total_loss += loss_dict['total_loss'].item()
                total_recon_loss += loss_dict['reconstruction_loss'].item()
                total_class_loss += loss_dict['classification_loss'].item()
                num_batches += 1
        
        # Compute challenge metrics
        challenge_metrics = {}
        if self.config.validation.compute_challenge_metrics:
            all_predictions = torch.cat(all_predictions, dim=0)
            all_targets = torch.cat(all_targets, dim=0)
            all_pert_labels = torch.cat(all_pert_labels, dim=0)
            
            challenge_metrics = compute_challenge_metrics(
                all_predictions, all_targets, all_pert_labels
            )
        
        val_metrics = {
            'val_loss': total_loss / num_batches,
            'val_recon_loss': total_recon_loss / num_batches,
            'val_class_loss': total_class_loss / num_batches,
            'val_combined_loss': total_loss / num_batches
        }
        val_metrics.update({f'val_{k}': v for k, v in challenge_metrics.items()})
        
        return val_metrics
    
    def _move_batch_to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Move batch to appropriate device."""
        return {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
    
    def _log_step_metrics(self, batch_idx: int, loss_dict: Dict[str, torch.Tensor], total_batches: int):
        """Log metrics for current step."""
        if self.local_rank == 0:  # Only log from main process
            lr = self.optimizer.param_groups[0]['lr']
            
            logger.info(
                f"Epoch {self.current_epoch}, Batch {batch_idx}/{total_batches}, "
                f"Loss: {loss_dict['total_loss'].item():.4f}, "
                f"Recon: {loss_dict['reconstruction_loss'].item():.4f}, "
                f"Class: {loss_dict['classification_loss'].item():.4f}, "
                f"LR: {lr:.6f}"
            )
            
            if self.config.logging.use_wandb and WANDB_AVAILABLE:
                wandb.log({
                    'train/step_loss': loss_dict['total_loss'].item(),
                    'train/step_recon_loss': loss_dict['reconstruction_loss'].item(),
                    'train/step_class_loss': loss_dict['classification_loss'].item(),
                    'train/learning_rate': lr,
                    'train/global_step': self.global_step
                })
    
    def _log_epoch_metrics(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float]
    ):
        """Log metrics for current epoch."""
        if self.local_rank == 0:
            # Console logging
            log_msg = f"Epoch {epoch}: "
            log_msg += f"Train Loss: {train_metrics['train_loss']:.4f}, "
            if val_metrics:
                log_msg += f"Val Loss: {val_metrics['val_loss']:.4f}, "
                if 'val_mae' in val_metrics:
                    log_msg += f"Val MAE: {val_metrics['val_mae']:.4f}"
            
            logger.info(log_msg)
            
            # Wandb logging
            if self.config.logging.use_wandb and WANDB_AVAILABLE:
                log_dict = {f'train/{k}': v for k, v in train_metrics.items()}
                log_dict.update({f'val/{k}': v for k, v in val_metrics.items()})
                log_dict['epoch'] = epoch
                wandb.log(log_dict)
            
            # Store training stats
            self.training_stats.append({
                'epoch': epoch,
                **train_metrics,
                **val_metrics
            })
    
    def _save_checkpoint(self, epoch: int, metric_value: float):
        """Save model checkpoint."""
        if self.local_rank == 0:  # Only save from main process
            checkpoint_dir = Path(self.config.checkpointing.checkpoint_dir)
            checkpoint_dir.mkdir(exist_ok=True)
            
            # Prepare checkpoint data
            model_state_dict = self.model.state_dict()
            if self.is_distributed:
                model_state_dict = self.model.module.state_dict()
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model_state_dict,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                'config': dict(self.config),
                'training_stats': self.training_stats,
                'global_step': self.global_step,
                'best_metric': self.best_metric
            }
            
            # Save checkpoint
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
            torch.save(checkpoint, checkpoint_path)
            
            # Save best model
            if metric_value < self.best_metric:
                self.best_metric = metric_value
                best_path = checkpoint_dir / "best_model.pt"
                torch.save(checkpoint, best_path)
                logger.info(f"New best model saved with metric: {metric_value:.4f}")
            
            logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def _should_stop_early(self, val_metrics: Dict[str, float]) -> bool:
        """Check if training should stop early."""
        # Implement early stopping logic here if needed
        return False
    
    def load_checkpoint(self, checkpoint_path: str, load_optimizer: bool = True):
        """Load model from checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        if self.is_distributed:
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        if load_optimizer and self.optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load other states
        if self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Update training state
        self.current_epoch = checkpoint.get('epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)
        self.best_metric = checkpoint.get('best_metric', float('inf'))
        self.training_stats = checkpoint.get('training_stats', [])
        
        logger.info(f"Checkpoint loaded. Resuming from epoch {self.current_epoch}")
    
    def cleanup(self):
        """Cleanup distributed training."""
        if self.is_distributed:
            dist.destroy_process_group()
        
        if self.config.logging.use_wandb and WANDB_AVAILABLE and self.local_rank == 0:
            wandb.finish()
