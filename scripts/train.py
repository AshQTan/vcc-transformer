#!/usr/bin/env python3
"""
Training script for VCC Transformer.

This script handles the complete training pipeline including:
- Configuration loading and validation
- Dataset creation and preprocessing
- Model initialization and training
- Distributed training setup
- Checkpointing and logging
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.multiprocessing as mp
from omegaconf import DictConfig

from vcc_transformer.utils.config import load_config, validate_config, print_config
from vcc_transformer.training.trainer import VCCTrainer
from vcc_transformer.data.dataset import create_datasets


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('training.log')
        ]
    )


def train_worker(rank: int, world_size: int, config: DictConfig, args):
    """
    Training worker for distributed training.
    
    Args:
        rank: Process rank
        world_size: Total number of processes
        config: Configuration object
        args: Command line arguments
    """
    # Set environment variables for distributed training
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    
    # Initialize trainer
    trainer = VCCTrainer(config)
    
    try:
        # Load checkpoint if resuming
        if args.resume_from:
            trainer.load_checkpoint(args.resume_from)
        
        # Create datasets
        if rank == 0:  # Only create datasets on main process
            logging.info("Creating datasets...")
        
        train_dataset, val_dataset = create_datasets(config)
        
        if rank == 0:
            logging.info(f"Training dataset: {len(train_dataset)} samples")
            logging.info(f"Validation dataset: {len(val_dataset)} samples")
        
        # Start training
        trainer.train(train_dataset, val_dataset)
        
    except Exception as e:
        logging.error(f"Training failed on rank {rank}: {str(e)}")
        raise
    finally:
        # Cleanup
        trainer.cleanup()


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train VCC Transformer")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=None,
        help="Number of GPUs for distributed training (auto-detect if None)"
    )
    parser.add_argument(
        "--master-port",
        type=str,
        default="12355",
        help="Master port for distributed training"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Load and validate configuration
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    
    try:
        validate_config(config)
        logger.info("Configuration validation passed")
    except ValueError as e:
        logger.error(f"Configuration validation failed: {e}")
        sys.exit(1)
    
    # Print configuration
    if logger.isEnabledFor(logging.INFO):
        print_config(config, "Training Configuration")
    
    # Set random seeds for reproducibility
    if hasattr(config, 'seed'):
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
        logger.info(f"Random seed set to {config.seed}")
    
    # Configure CUDA settings
    if torch.cuda.is_available():
        if config.hardware.benchmark_cudnn:
            torch.backends.cudnn.benchmark = True
            logger.info("CuDNN benchmark enabled")
        
        if hasattr(config, 'deterministic') and config.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.use_deterministic_algorithms(True)
            logger.info("Deterministic mode enabled")
    
    # Determine world size for distributed training
    if args.world_size is None:
        world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
    else:
        world_size = args.world_size
    
    # Setup distributed training environment
    if world_size > 1:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = args.master_port
        
        logger.info(f"Starting distributed training with {world_size} processes")
        
        # Spawn training processes
        mp.spawn(
            train_worker,
            args=(world_size, config, args),
            nprocs=world_size,
            join=True
        )
    else:
        # Single GPU/CPU training
        logger.info("Starting single-process training")
        train_worker(0, 1, config, args)
    
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()
