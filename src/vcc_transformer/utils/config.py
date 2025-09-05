"""Configuration utilities for VCC Transformer"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Union
from omegaconf import DictConfig, OmegaConf


def load_config(config_path: Union[str, Path]) -> DictConfig:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        DictConfig: Configuration object
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return OmegaConf.create(config_dict)


def save_config(config: DictConfig, save_path: Union[str, Path]) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration object to save
        save_path: Path where to save the configuration
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        yaml.dump(OmegaConf.to_yaml(config), f, default_flow_style=False)


def merge_configs(base_config: DictConfig, override_config: Dict[str, Any]) -> DictConfig:
    """
    Merge base configuration with override values.
    
    Args:
        base_config: Base configuration
        override_config: Override values
        
    Returns:
        DictConfig: Merged configuration
    """
    override_cfg = OmegaConf.create(override_config)
    return OmegaConf.merge(base_config, override_cfg)


def validate_config(config: DictConfig) -> None:
    """
    Validate configuration values.
    
    Args:
        config: Configuration to validate
        
    Raises:
        ValueError: If configuration is invalid
    """
    # Check required sections
    required_sections = ['data', 'model', 'training']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")
    
    # Validate model config
    if config.model.d_model % config.model.n_heads != 0:
        raise ValueError("d_model must be divisible by n_heads")
    
    # Validate data paths
    if not Path(config.data.training_file).exists():
        print(f"Warning: Training file not found: {config.data.training_file}")
    
    # Validate training config
    if config.training.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    
    if config.training.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")


def setup_directories(config: DictConfig) -> None:
    """
    Create necessary directories based on config.
    
    Args:
        config: Configuration object
    """
    dirs_to_create = [
        config.checkpointing.checkpoint_dir,
        config.logging.log_dir,
        Path(config.data.output_file).parent
    ]
    
    for dir_path in dirs_to_create:
        Path(dir_path).mkdir(parents=True, exist_ok=True)


def get_device_config(config: DictConfig) -> Dict[str, Any]:
    """
    Get device configuration for training.
    
    Args:
        config: Configuration object
        
    Returns:
        Dict: Device configuration
    """
    import torch
    
    device_config = {
        'device': config.hardware.device,
        'use_cuda': torch.cuda.is_available() and config.hardware.device == 'cuda',
        'use_mps': torch.backends.mps.is_available() and config.hardware.device == 'mps',
        'world_size': 1,
        'local_rank': 0
    }
    
    if device_config['use_cuda']:
        device_config['world_size'] = torch.cuda.device_count()
        if 'LOCAL_RANK' in os.environ:
            device_config['local_rank'] = int(os.environ['LOCAL_RANK'])
    
    return device_config


def print_config(config: DictConfig, title: str = "Configuration") -> None:
    """
    Pretty print configuration.
    
    Args:
        config: Configuration to print
        title: Title for the print output
    """
    try:
        from rich.console import Console
        from rich.syntax import Syntax
        
        console = Console()
        config_yaml = OmegaConf.to_yaml(config)
        syntax = Syntax(config_yaml, "yaml", theme="monokai", line_numbers=True)
        
        console.print(f"\n[bold cyan]{title}[/bold cyan]")
        console.print(syntax)
    except ImportError:
        print(f"\n{title}:")
        print(OmegaConf.to_yaml(config))
