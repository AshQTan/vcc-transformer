#!/usr/bin/env python3
"""
Demo script to showcase the beautiful training progress visualization.

This script simulates training progress to demonstrate the rich console output.
"""

import time
import random
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vcc_transformer.utils.config import load_config
from vcc_transformer.utils.visualization import TrainingProgressTracker


def simulate_training():
    """Simulate training with beautiful progress tracking."""
    # Load configuration
    config = load_config('configs/small_config.yaml')  # Use small config for demo
    
    # Setup demo parameters
    total_epochs = 10
    steps_per_epoch = 50
    
    # Initialize progress tracker
    tracker = TrainingProgressTracker(
        config=config,
        total_epochs=total_epochs,
        steps_per_epoch=steps_per_epoch
    )
    
    print("🎭 Demo: VCC Transformer Training Progress Visualization")
    print("   This demonstrates the beautiful progress tracking during training")
    print("   Press Ctrl+C to exit early\n")
    
    try:
        # Start training
        tracker.start_training()
        
        # Simulate training epochs
        for epoch in range(total_epochs):
            tracker.update_epoch(epoch)
            
            # Simulate epoch training
            base_loss = 1.0
            base_recon = 0.8
            base_class = 0.3
            base_lr = config.training.learning_rate
            
            for step in range(steps_per_epoch):
                # Simulate loss decreasing with some noise
                progress = (epoch * steps_per_epoch + step) / (total_epochs * steps_per_epoch)
                noise = random.uniform(0.9, 1.1)
                
                current_metrics = {
                    'loss': (base_loss * (1 - progress * 0.7) + random.uniform(-0.1, 0.1)) * noise,
                    'recon_loss': (base_recon * (1 - progress * 0.6) + random.uniform(-0.05, 0.05)) * noise,
                    'class_loss': (base_class * (1 - progress * 0.8) + random.uniform(-0.02, 0.02)) * noise,
                    'lr': base_lr * (1 - progress * 0.9)  # Learning rate decay
                }
                
                tracker.update_step(step, current_metrics)
                time.sleep(0.1)  # Simulate training time
            
            # Simulate epoch end metrics
            train_metrics = {
                'train_loss': current_metrics['loss'],
                'train_recon_loss': current_metrics['recon_loss'],
                'train_class_loss': current_metrics['class_loss'],
                'learning_rate': current_metrics['lr']
            }
            
            val_metrics = None
            if epoch % 2 == 0:  # Validation every 2 epochs
                val_metrics = {
                    'val_loss': train_metrics['train_loss'] * 1.1,
                    'val_recon_loss': train_metrics['train_recon_loss'] * 1.05,
                    'val_class_loss': train_metrics['train_class_loss'] * 1.15,
                    'val_mae': 0.5 * (1 - progress * 0.6),
                    'val_correlation': 0.3 + progress * 0.4,
                    'val_rmse': 0.6 * (1 - progress * 0.5),
                    'val_combined_loss': train_metrics['train_loss'] * 1.1
                }
            
            tracker.end_epoch(train_metrics, val_metrics)
            time.sleep(0.5)  # Brief pause between epochs
        
        # End training
        tracker.end_training()
        
        # Save demo history
        demo_history_path = Path("demo_training_history.json")
        tracker.save_history(demo_history_path)
        
        print(f"\n🎉 Demo completed! Training history saved to: {demo_history_path}")
        print("\nTo generate a full report with plots, run:")
        print(f"python scripts/generate_report.py --history-file {demo_history_path} --output-dir demo_report")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrupted by user")
        tracker.end_training()
    except Exception as e:
        print(f"\n❌ Demo error: {e}")


def main():
    """Main demo function."""
    print("🧬 VCC Transformer - Beautiful Training Progress Demo")
    print("=" * 60)
    
    try:
        simulate_training()
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
