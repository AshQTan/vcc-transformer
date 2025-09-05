#!/usr/bin/env python3
"""
GPU Optimization Script for VCC Transformer

This script provides GPU optimization and undervolting capabilities
for improved power efficiency during training.

Usage:
    python scripts/optimize_gpu.py --config configs/base_config.yaml --find-optimal
    python scripts/optimize_gpu.py --config configs/base_config.yaml --apply-settings
    python scripts/optimize_gpu.py --monitor --duration 300
"""

import argparse
import sys
import time
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vcc_transformer.utils.config import load_config
from vcc_transformer.utils.gpu_optimization import GPUOptimizer, GPUSettings


def setup_logging():
    """Setup logging for the script"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('gpu_optimization.log')
        ]
    )
    return logging.getLogger(__name__)


def find_optimal_settings(optimizer: GPUOptimizer, config: dict):
    """Find optimal GPU settings"""
    logger = logging.getLogger(__name__)
    
    if not optimizer.enabled:
        logger.error("GPU optimization not available")
        return
    
    logger.info("Finding optimal GPU settings...")
    
    for gpu_id in range(optimizer.device_count):
        logger.info(f"Optimizing GPU {gpu_id}...")
        
        # Get current status
        status = optimizer.get_gpu_status(gpu_id)
        if status:
            logger.info(f"Current status - Temp: {status.temperature}°C, "
                       f"Power: {status.power_usage:.1f}W, "
                       f"Util: {status.utilization}%")
        
        # Find optimal settings
        optimal = optimizer.find_optimal_settings(gpu_id)
        if optimal:
            logger.info(f"Optimal settings for GPU {gpu_id}:")
            logger.info(f"  Core offset: {optimal.core_offset}mV")
            logger.info(f"  Memory offset: {optimal.memory_offset}mV")
            logger.info(f"  Power limit: {optimal.power_limit}%")
            logger.info(f"  Temperature limit: {optimal.temp_limit}°C")
        else:
            logger.warning(f"Could not find optimal settings for GPU {gpu_id}")


def apply_settings(optimizer: GPUOptimizer, config: dict):
    """Apply GPU optimization settings"""
    logger = logging.getLogger(__name__)
    
    if not optimizer.enabled:
        logger.error("GPU optimization not available")
        return
    
    gpu_config = config.get('gpu', {})
    undervolt_settings = gpu_config.get('undervolt_settings', {})
    
    settings = GPUSettings(
        core_offset=undervolt_settings.get('core_offset', -100),
        memory_offset=undervolt_settings.get('memory_offset', -50),
        power_limit=undervolt_settings.get('power_limit', 80),
        temp_limit=undervolt_settings.get('temp_limit', 83),
        fan_curve_aggressive=undervolt_settings.get('fan_curve_aggressive', True)
    )
    
    logger.info("Applying GPU optimization settings...")
    
    # Save original settings first
    optimizer.save_original_settings()
    
    for gpu_id in range(optimizer.device_count):
        logger.info(f"Applying settings to GPU {gpu_id}...")
        
        if optimizer.apply_settings(gpu_id, settings):
            logger.info(f"GPU {gpu_id}: Settings applied successfully")
            
            # Check stability
            if optimizer.check_stability(gpu_id, duration=30):
                logger.info(f"GPU {gpu_id}: Stability test passed")
            else:
                logger.warning(f"GPU {gpu_id}: Stability test failed, reverting...")
                optimizer.restore_original_settings()
                return
        else:
            logger.error(f"GPU {gpu_id}: Failed to apply settings")


def monitor_gpus(optimizer: GPUOptimizer, duration: int = None):
    """Monitor GPU status"""
    logger = logging.getLogger(__name__)
    
    if not optimizer.enabled:
        logger.error("GPU optimization not available")
        return
    
    logger.info(f"Starting GPU monitoring...")
    if duration:
        logger.info(f"Monitoring for {duration} seconds")
    else:
        logger.info("Monitoring until interrupted (Ctrl+C)")
    
    try:
        optimizer.monitor_gpus(interval=5, duration=duration)
    except KeyboardInterrupt:
        logger.info("Monitoring stopped by user")


def show_status(optimizer: GPUOptimizer):
    """Show current GPU status"""
    logger = logging.getLogger(__name__)
    
    if not optimizer.enabled:
        logger.error("GPU optimization not available")
        return
    
    logger.info("Current GPU Status:")
    
    for gpu_id in range(optimizer.device_count):
        status = optimizer.get_gpu_status(gpu_id)
        if status:
            logger.info(f"GPU {gpu_id} ({status.name}):")
            logger.info(f"  Temperature: {status.temperature}°C")
            logger.info(f"  Power Usage: {status.power_usage:.1f}W")
            logger.info(f"  Memory: {status.memory_used:.1f}/{status.memory_total:.1f}GB "
                       f"({status.memory_used/status.memory_total*100:.1f}%)")
            logger.info(f"  Utilization: {status.utilization}%")
            logger.info(f"  Power Limit: {status.power_limit:.1f}W")
    
    # Show recommendations
    recommendations = optimizer.get_optimization_recommendations()
    if recommendations:
        logger.info("\nOptimization Recommendations:")
        for gpu, rec in recommendations.items():
            if rec != "Optimal":
                logger.info(f"  {gpu}: {rec}")


def restore_settings(optimizer: GPUOptimizer):
    """Restore original GPU settings"""
    logger = logging.getLogger(__name__)
    
    if not optimizer.enabled:
        logger.error("GPU optimization not available")
        return
    
    logger.info("Restoring original GPU settings...")
    optimizer.restore_original_settings()
    logger.info("Original settings restored")


def main():
    parser = argparse.ArgumentParser(description="GPU Optimization for VCC Transformer")
    parser.add_argument("--config", type=str, help="Configuration file path")
    parser.add_argument("--find-optimal", action="store_true", 
                       help="Find optimal GPU settings automatically")
    parser.add_argument("--apply-settings", action="store_true",
                       help="Apply GPU settings from config")
    parser.add_argument("--monitor", action="store_true",
                       help="Monitor GPU status")
    parser.add_argument("--duration", type=int, default=None,
                       help="Duration for monitoring (seconds)")
    parser.add_argument("--status", action="store_true",
                       help="Show current GPU status")
    parser.add_argument("--restore", action="store_true",
                       help="Restore original GPU settings")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    # Load configuration
    config = {}
    if args.config:
        try:
            config = load_config(args.config)
            logger.info(f"Loaded configuration from {args.config}")
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            return 1
    
    # Create GPU optimizer
    optimizer = GPUOptimizer(config=config.get('gpu', {}))
    
    if not optimizer.enabled:
        logger.error("GPU optimization not available. Please install nvidia-ml-py3:")
        logger.error("pip install nvidia-ml-py3")
        return 1
    
    # Execute requested action
    if args.find_optimal:
        find_optimal_settings(optimizer, config)
    elif args.apply_settings:
        apply_settings(optimizer, config)
    elif args.monitor:
        monitor_gpus(optimizer, args.duration)
    elif args.status:
        show_status(optimizer)
    elif args.restore:
        restore_settings(optimizer)
    else:
        logger.error("No action specified. Use --help for options.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
