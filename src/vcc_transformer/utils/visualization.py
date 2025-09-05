"""
Enhanced logging and visualization utilities for VCC Transformer.

This module provides beautiful, easy-to-read training progress and results
with rich console output, progress bars, and comprehensive summaries.
"""

import time
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
import json

try:
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.live import Live
    from rich.text import Text
    from rich.tree import Tree
    from rich.columns import Columns
    from rich.align import Align
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

logger = logging.getLogger(__name__)


class TrainingProgressTracker:
    """
    Beautiful training progress tracker with rich console output.
    """
    
    def __init__(self, config, total_epochs: int, steps_per_epoch: int):
        """
        Initialize the progress tracker.
        
        Args:
            config: Configuration object
            total_epochs: Total number of training epochs
            steps_per_epoch: Number of steps per epoch
        """
        self.config = config
        self.total_epochs = total_epochs
        self.steps_per_epoch = steps_per_epoch
        self.console = Console() if RICH_AVAILABLE else None
        
        # Training state
        self.current_epoch = 0
        self.current_step = 0
        self.start_time = time.time()
        self.epoch_start_time = time.time()
        
        # Metrics history
        self.train_history = []
        self.val_history = []
        self.best_metrics = {}
        
        # Progress bars
        self.epoch_progress = None
        self.step_progress = None
        self.live_display = None
        
        if RICH_AVAILABLE:
            self._setup_rich_display()
    
    def _setup_rich_display(self):
        """Setup rich console display components."""
        self.epoch_progress = Progress(
            TextColumn("[bold blue]Epoch", justify="right"),
            BarColumn(bar_width=None),
            "[progress.percentage]{task.percentage:>3.1f}%",
            "•",
            TimeElapsedColumn(),
            "•",
            TimeRemainingColumn(),
            console=self.console,
            expand=True
        )
        
        self.step_progress = Progress(
            TextColumn("[bold green]Step", justify="right"),
            BarColumn(bar_width=None),
            "[progress.percentage]{task.percentage:>3.1f}%",
            "•",
            "[bold yellow]{task.completed}/{task.total}",
            console=self.console,
            expand=True
        )
    
    def start_training(self):
        """Start training progress tracking."""
        if not RICH_AVAILABLE:
            print("🚀 Starting VCC Transformer Training...")
            return
        
        # Print beautiful header
        header = Panel.fit(
            "[bold cyan]🧬 VCC Transformer Training[/bold cyan]\n"
            f"[dim]Model: {self.config.model.d_model}d, {self.config.model.n_layers} layers, {self.config.model.n_heads} heads[/dim]\n"
            f"[dim]Batch Size: {self.config.training.batch_size} | Learning Rate: {self.config.training.learning_rate}[/dim]",
            style="blue"
        )
        self.console.print(header)
        
        # Start progress tracking
        self.epoch_task = self.epoch_progress.add_task("Training", total=self.total_epochs)
        self.step_task = self.step_progress.add_task("Epoch Steps", total=self.steps_per_epoch)
        
        # Create live display layout
        layout = Layout()
        layout.split_column(
            Layout(self.epoch_progress, name="epoch"),
            Layout(self.step_progress, name="step"),
            Layout(self._create_metrics_panel(), name="metrics")
        )
        
        self.live_display = Live(layout, console=self.console, refresh_per_second=2)
        self.live_display.start()
    
    def update_epoch(self, epoch: int):
        """Update epoch progress."""
        self.current_epoch = epoch
        self.epoch_start_time = time.time()
        
        if RICH_AVAILABLE:
            self.epoch_progress.update(self.epoch_task, completed=epoch)
            # Reset step progress for new epoch
            self.step_progress.reset(self.step_task)
        else:
            print(f"\n📈 Epoch {epoch}/{self.total_epochs}")
    
    def update_step(self, step: int, metrics: Dict[str, float]):
        """Update step progress with metrics."""
        self.current_step = step
        
        if RICH_AVAILABLE:
            self.step_progress.update(self.step_task, completed=step)
            # Update live metrics display
            if self.live_display:
                layout = self.live_display.renderable
                layout["metrics"].update(self._create_metrics_panel(metrics))
        else:
            if step % 50 == 0:  # Log every 50 steps
                loss_str = f"Loss: {metrics.get('loss', 0):.4f}"
                lr_str = f"LR: {metrics.get('lr', 0):.2e}"
                print(f"  Step {step}/{self.steps_per_epoch} | {loss_str} | {lr_str}")
    
    def _create_metrics_panel(self, current_metrics: Optional[Dict[str, float]] = None) -> Panel:
        """Create a beautiful metrics display panel."""
        if not current_metrics:
            current_metrics = {}
        
        # Create metrics table
        table = Table(show_header=True, header_style="bold magenta", box=None)
        table.add_column("Metric", style="cyan", width=20)
        table.add_column("Current", style="green", justify="right", width=15)
        table.add_column("Best", style="yellow", justify="right", width=15)
        table.add_column("Trend", style="blue", justify="center", width=10)
        
        # Add current metrics
        metrics_to_show = [
            ("Loss", "loss", "total_loss"),
            ("Recon Loss", "recon_loss", "reconstruction_loss"),
            ("Class Loss", "class_loss", "classification_loss"),
            ("Learning Rate", "lr", "learning_rate"),
        ]
        
        for display_name, current_key, history_key in metrics_to_show:
            current_val = current_metrics.get(current_key, 0)
            best_val = self.best_metrics.get(history_key, 0)
            
            # Determine trend
            trend = "→"
            if len(self.train_history) > 1:
                if "loss" in history_key.lower():
                    trend = "↓" if current_val < best_val else "↑"
                else:
                    trend = "↑" if current_val > best_val else "↓"
            
            # Format values
            if "lr" in current_key.lower():
                current_str = f"{current_val:.2e}"
                best_str = f"{best_val:.2e}"
            else:
                current_str = f"{current_val:.4f}"
                best_str = f"{best_val:.4f}"
            
            table.add_row(display_name, current_str, best_str, trend)
        
        # Calculate time estimates
        elapsed = time.time() - self.start_time
        epoch_elapsed = time.time() - self.epoch_start_time
        
        if self.current_step > 0:
            time_per_step = epoch_elapsed / self.current_step
            eta_epoch = time_per_step * (self.steps_per_epoch - self.current_step)
            eta_total = (elapsed / max(1, self.current_epoch)) * (self.total_epochs - self.current_epoch)
        else:
            eta_epoch = 0
            eta_total = 0
        
        time_info = (
            f"[dim]Elapsed: {elapsed/3600:.1f}h | "
            f"Epoch ETA: {eta_epoch/60:.1f}m | "
            f"Total ETA: {eta_total/3600:.1f}h[/dim]"
        )
        
        return Panel(
            Align.center(Columns([table, Text(time_info)], equal=True)),
            title="[bold]📊 Training Metrics[/bold]",
            border_style="green"
        )
    
    def end_epoch(self, train_metrics: Dict[str, float], val_metrics: Dict[str, float] = None):
        """End epoch and record metrics."""
        # Record metrics
        epoch_data = {
            'epoch': self.current_epoch,
            'timestamp': time.time(),
            **train_metrics
        }
        self.train_history.append(epoch_data)
        
        if val_metrics:
            val_data = {
                'epoch': self.current_epoch,
                'timestamp': time.time(),
                **val_metrics
            }
            self.val_history.append(val_data)
        
        # Update best metrics
        for key, value in {**train_metrics, **(val_metrics or {})}.items():
            if "loss" in key.lower():
                if key not in self.best_metrics or value < self.best_metrics[key]:
                    self.best_metrics[key] = value
            else:
                if key not in self.best_metrics or value > self.best_metrics[key]:
                    self.best_metrics[key] = value
        
        # Display epoch summary
        if RICH_AVAILABLE:
            self._display_epoch_summary(train_metrics, val_metrics)
        else:
            self._print_epoch_summary(train_metrics, val_metrics)
    
    def _display_epoch_summary(self, train_metrics: Dict[str, float], val_metrics: Dict[str, float] = None):
        """Display beautiful epoch summary."""
        # Create summary table
        table = Table(title=f"Epoch {self.current_epoch} Summary", show_header=True)
        table.add_column("Split", style="cyan")
        table.add_column("Loss", style="red")
        table.add_column("Recon Loss", style="yellow")
        table.add_column("Class Loss", style="green")
        
        if val_metrics and 'val_mae' in val_metrics:
            table.add_column("MAE", style="blue")
            table.add_column("Correlation", style="magenta")
        
        # Add training row
        train_row = [
            "Train",
            f"{train_metrics.get('train_loss', 0):.4f}",
            f"{train_metrics.get('train_recon_loss', 0):.4f}",
            f"{train_metrics.get('train_class_loss', 0):.4f}"
        ]
        
        # Add validation row if available
        if val_metrics:
            val_row = [
                "Validation",
                f"{val_metrics.get('val_loss', 0):.4f}",
                f"{val_metrics.get('val_recon_loss', 0):.4f}",
                f"{val_metrics.get('val_class_loss', 0):.4f}"
            ]
            
            if 'val_mae' in val_metrics:
                train_row.extend(["-", "-"])
                val_row.extend([
                    f"{val_metrics.get('val_mae', 0):.4f}",
                    f"{val_metrics.get('val_correlation', 0):.4f}"
                ])
            
            table.add_row(*train_row)
            table.add_row(*val_row)
        else:
            table.add_row(*train_row)
        
        # Display with panel
        summary_panel = Panel(
            table,
            title=f"[bold green]✅ Epoch {self.current_epoch} Complete[/bold green]",
            border_style="green"
        )
        
        self.console.print("\n")
        self.console.print(summary_panel)
    
    def _print_epoch_summary(self, train_metrics: Dict[str, float], val_metrics: Dict[str, float] = None):
        """Print simple epoch summary."""
        print(f"\n✅ Epoch {self.current_epoch} Complete:")
        print(f"  Train Loss: {train_metrics.get('train_loss', 0):.4f}")
        if val_metrics:
            print(f"  Val Loss: {val_metrics.get('val_loss', 0):.4f}")
            if 'val_mae' in val_metrics:
                print(f"  Val MAE: {val_metrics.get('val_mae', 0):.4f}")
    
    def end_training(self):
        """End training and display final summary."""
        if RICH_AVAILABLE and self.live_display:
            self.live_display.stop()
            
            # Display final summary
            self._display_final_summary()
        else:
            print("\n🎉 Training Complete!")
            self._print_final_summary()
    
    def _display_final_summary(self):
        """Display beautiful final training summary."""
        total_time = time.time() - self.start_time
        
        # Create final summary
        summary_tree = Tree("🎉 [bold green]Training Complete![/bold green]")
        
        # Training info
        training_info = summary_tree.add("📊 [bold]Training Information[/bold]")
        training_info.add(f"Total Epochs: {self.current_epoch}")
        training_info.add(f"Total Time: {total_time/3600:.2f} hours")
        training_info.add(f"Average Time/Epoch: {total_time/max(1, self.current_epoch)/60:.1f} minutes")
        
        # Best metrics
        best_info = summary_tree.add("🏆 [bold]Best Metrics[/bold]")
        for key, value in self.best_metrics.items():
            if "loss" in key.lower():
                best_info.add(f"{key}: {value:.4f}")
            else:
                best_info.add(f"{key}: {value:.4f}")
        
        # Model info
        model_info = summary_tree.add("🤖 [bold]Model Configuration[/bold]")
        model_info.add(f"Architecture: {self.config.model.d_model}d, {self.config.model.n_layers} layers")
        model_info.add(f"Attention Heads: {self.config.model.n_heads}")
        model_info.add(f"Batch Size: {self.config.training.batch_size}")
        
        final_panel = Panel(
            summary_tree,
            title="[bold cyan]🧬 VCC Transformer Training Summary[/bold cyan]",
            border_style="cyan",
            padding=(1, 2)
        )
        
        self.console.print("\n")
        self.console.print(final_panel)
    
    def _print_final_summary(self):
        """Print simple final summary."""
        total_time = time.time() - self.start_time
        print(f"\n🎉 Training Complete!")
        print(f"  Total Time: {total_time/3600:.2f} hours")
        print(f"  Best Loss: {self.best_metrics.get('total_loss', 0):.4f}")
    
    def save_history(self, save_path: Path):
        """Save training history to file."""
        history_data = {
            'config': dict(self.config) if hasattr(self.config, '__dict__') else {},
            'train_history': self.train_history,
            'val_history': self.val_history,
            'best_metrics': self.best_metrics,
            'total_epochs': self.current_epoch,
            'total_time': time.time() - self.start_time
        }
        
        with open(save_path, 'w') as f:
            json.dump(history_data, f, indent=2)
        
        if RICH_AVAILABLE:
            self.console.print(f"📁 Training history saved to {save_path}")
        else:
            print(f"📁 Training history saved to {save_path}")


class ResultsVisualizer:
    """
    Create beautiful visualizations of training results.
    """
    
    def __init__(self, history_file: Path):
        """Initialize with training history."""
        with open(history_file, 'r') as f:
            self.history_data = json.load(f)
        
        self.train_history = self.history_data['train_history']
        self.val_history = self.history_data['val_history']
        
        if PLOTTING_AVAILABLE:
            plt.style.use('seaborn-v0_8')
            sns.set_palette("husl")
    
    def create_training_plots(self, save_dir: Path):
        """Create comprehensive training visualization plots."""
        if not PLOTTING_AVAILABLE:
            print("Matplotlib not available. Skipping plots.")
            return
        
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        # 1. Loss curves
        self._plot_loss_curves(save_dir / "loss_curves.png")
        
        # 2. Learning rate schedule
        self._plot_learning_rate(save_dir / "learning_rate.png")
        
        # 3. Challenge metrics
        if self.val_history and any('val_mae' in epoch for epoch in self.val_history):
            self._plot_challenge_metrics(save_dir / "challenge_metrics.png")
        
        # 4. Training speed
        self._plot_training_speed(save_dir / "training_speed.png")
        
        print(f"📊 Training plots saved to {save_dir}")
    
    def _plot_loss_curves(self, save_path: Path):
        """Plot training and validation loss curves."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('VCC Transformer Training Progress', fontsize=16, fontweight='bold')
        
        epochs = [epoch['epoch'] for epoch in self.train_history]
        
        # Total loss
        train_loss = [epoch['train_loss'] for epoch in self.train_history]
        axes[0, 0].plot(epochs, train_loss, label='Train', linewidth=2)
        
        if self.val_history:
            val_epochs = [epoch['epoch'] for epoch in self.val_history]
            val_loss = [epoch['val_loss'] for epoch in self.val_history]
            axes[0, 0].plot(val_epochs, val_loss, label='Validation', linewidth=2)
        
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Reconstruction loss
        recon_loss = [epoch['train_recon_loss'] for epoch in self.train_history]
        axes[0, 1].plot(epochs, recon_loss, label='Train Recon', linewidth=2)
        
        if self.val_history:
            val_recon = [epoch.get('val_recon_loss', 0) for epoch in self.val_history]
            axes[0, 1].plot(val_epochs, val_recon, label='Val Recon', linewidth=2)
        
        axes[0, 1].set_title('Reconstruction Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MSE Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Classification loss
        class_loss = [epoch['train_class_loss'] for epoch in self.train_history]
        axes[1, 0].plot(epochs, class_loss, label='Train Class', linewidth=2)
        
        if self.val_history:
            val_class = [epoch.get('val_class_loss', 0) for epoch in self.val_history]
            axes[1, 0].plot(val_epochs, val_class, label='Val Class', linewidth=2)
        
        axes[1, 0].set_title('Classification Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('CrossEntropy Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Loss ratio
        recon_weight = self.history_data.get('config', {}).get('training', {}).get('reconstruction_weight', 1.0)
        class_weight = self.history_data.get('config', {}).get('training', {}).get('classification_weight', 0.5)
        
        weighted_recon = [r * recon_weight for r in recon_loss]
        weighted_class = [c * class_weight for c in class_loss]
        
        axes[1, 1].plot(epochs, weighted_recon, label=f'Weighted Recon (×{recon_weight})', linewidth=2)
        axes[1, 1].plot(epochs, weighted_class, label=f'Weighted Class (×{class_weight})', linewidth=2)
        
        axes[1, 1].set_title('Weighted Loss Components')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Weighted Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_learning_rate(self, save_path: Path):
        """Plot learning rate schedule."""
        if not any('learning_rate' in epoch for epoch in self.train_history):
            return
        
        epochs = [epoch['epoch'] for epoch in self.train_history]
        lrs = [epoch.get('learning_rate', 0) for epoch in self.train_history]
        
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, lrs, linewidth=2, color='orange')
        plt.title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_challenge_metrics(self, save_path: Path):
        """Plot challenge-specific metrics."""
        val_epochs = [epoch['epoch'] for epoch in self.val_history]
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('Challenge Metrics Progress', fontsize=16, fontweight='bold')
        
        # MAE
        mae_values = [epoch.get('val_mae', 0) for epoch in self.val_history]
        axes[0].plot(val_epochs, mae_values, linewidth=3, color='red', marker='o')
        axes[0].set_title('Mean Absolute Error')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('MAE')
        axes[0].grid(True, alpha=0.3)
        
        # Correlation
        corr_values = [epoch.get('val_correlation', 0) for epoch in self.val_history]
        axes[1].plot(val_epochs, corr_values, linewidth=3, color='green', marker='s')
        axes[1].set_title('Pearson Correlation')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Correlation')
        axes[1].grid(True, alpha=0.3)
        
        # RMSE
        rmse_values = [epoch.get('val_rmse', 0) for epoch in self.val_history]
        axes[2].plot(val_epochs, rmse_values, linewidth=3, color='blue', marker='^')
        axes[2].set_title('Root Mean Square Error')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('RMSE')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_training_speed(self, save_path: Path):
        """Plot training speed metrics."""
        epochs = [epoch['epoch'] for epoch in self.train_history]
        timestamps = [epoch['timestamp'] for epoch in self.train_history]
        
        # Calculate time per epoch
        epoch_times = []
        for i in range(1, len(timestamps)):
            epoch_time = timestamps[i] - timestamps[i-1]
            epoch_times.append(epoch_time / 60)  # Convert to minutes
        
        plt.figure(figsize=(10, 6))
        plt.plot(epochs[1:], epoch_times, linewidth=2, marker='o', color='purple')
        plt.title('Training Speed', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Time per Epoch (minutes)')
        plt.grid(True, alpha=0.3)
        
        # Add average line
        if epoch_times:
            avg_time = np.mean(epoch_times)
            plt.axhline(y=avg_time, color='red', linestyle='--', 
                       label=f'Average: {avg_time:.1f} min')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


def create_training_report(history_file: Path, output_dir: Path):
    """Create a comprehensive training report."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Create visualizations
    visualizer = ResultsVisualizer(history_file)
    plots_dir = output_dir / "plots"
    visualizer.create_training_plots(plots_dir)
    
    # Create HTML report
    html_report = _generate_html_report(history_file, plots_dir)
    
    report_path = output_dir / "training_report.html"
    with open(report_path, 'w') as f:
        f.write(html_report)
    
    print(f"📄 Comprehensive training report created: {report_path}")
    return report_path


def _generate_html_report(history_file: Path, plots_dir: Path) -> str:
    """Generate HTML training report."""
    with open(history_file, 'r') as f:
        history_data = json.load(f)
    
    config = history_data.get('config', {})
    best_metrics = history_data.get('best_metrics', {})
    total_time = history_data.get('total_time', 0)
    
    # Convert plot paths to relative paths
    plot_files = list(plots_dir.glob("*.png"))
    plot_html = ""
    for plot_file in plot_files:
        plot_html += f'<img src="plots/{plot_file.name}" style="max-width: 100%; margin: 10px;">\n'
    
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>VCC Transformer Training Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                      color: white; padding: 20px; border-radius: 10px; }}
            .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
                           gap: 20px; margin: 20px 0; }}
            .metric-card {{ background: #f8f9fa; padding: 15px; border-radius: 8px; 
                           border-left: 4px solid #007bff; }}
            .plot-section {{ margin: 30px 0; }}
            .config-section {{ background: #f8f9fa; padding: 20px; border-radius: 8px; 
                             margin: 20px 0; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🧬 VCC Transformer Training Report</h1>
            <p>Training completed in {total_time/3600:.2f} hours</p>
        </div>
        
        <h2>📊 Best Metrics</h2>
        <div class="metric-grid">
            {_generate_metric_cards(best_metrics)}
        </div>
        
        <h2>🤖 Model Configuration</h2>
        <div class="config-section">
            {_generate_config_summary(config)}
        </div>
        
        <h2>📈 Training Progress</h2>
        <div class="plot-section">
            {plot_html}
        </div>
    </body>
    </html>
    """
    
    return html_template


def _generate_metric_cards(metrics: Dict[str, float]) -> str:
    """Generate HTML for metric cards."""
    cards_html = ""
    for metric, value in metrics.items():
        formatted_value = f"{value:.4f}"
        if "lr" in metric.lower():
            formatted_value = f"{value:.2e}"
        
        cards_html += f"""
        <div class="metric-card">
            <h4>{metric.replace('_', ' ').title()}</h4>
            <h2>{formatted_value}</h2>
        </div>
        """
    
    return cards_html


def _generate_config_summary(config: Dict[str, Any]) -> str:
    """Generate HTML for configuration summary."""
    model_config = config.get('model', {})
    training_config = config.get('training', {})
    
    return f"""
    <h3>Model Architecture</h3>
    <ul>
        <li><strong>Hidden Dimension:</strong> {model_config.get('d_model', 'N/A')}</li>
        <li><strong>Layers:</strong> {model_config.get('n_layers', 'N/A')}</li>
        <li><strong>Attention Heads:</strong> {model_config.get('n_heads', 'N/A')}</li>
        <li><strong>Feed Forward Dim:</strong> {model_config.get('d_ff', 'N/A')}</li>
    </ul>
    
    <h3>Training Configuration</h3>
    <ul>
        <li><strong>Batch Size:</strong> {training_config.get('batch_size', 'N/A')}</li>
        <li><strong>Learning Rate:</strong> {training_config.get('learning_rate', 'N/A')}</li>
        <li><strong>Optimizer:</strong> {training_config.get('optimizer', 'N/A')}</li>
        <li><strong>Mixed Precision:</strong> {training_config.get('use_amp', False)}</li>
    </ul>
    """
