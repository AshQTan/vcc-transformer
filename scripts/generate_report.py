#!/usr/bin/env python3
"""
Generate comprehensive training report with visualizations.

This script creates beautiful plots and HTML reports from training history.
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vcc_transformer.utils.visualization import create_training_report, ResultsVisualizer


def main():
    """Main report generation function."""
    parser = argparse.ArgumentParser(description="Generate VCC Transformer training report")
    parser.add_argument(
        "--history-file",
        type=str,
        required=True,
        help="Path to training history JSON file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports",
        help="Output directory for reports"
    )
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Generate plots only (no HTML report)"
    )
    
    args = parser.parse_args()
    
    # Check if history file exists
    history_path = Path(args.history_file)
    if not history_path.exists():
        print(f"❌ History file not found: {history_path}")
        return 1
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"📊 Generating training report from {history_path}")
    
    if args.plots_only:
        # Generate plots only
        visualizer = ResultsVisualizer(history_path)
        plots_dir = output_dir / "plots"
        visualizer.create_training_plots(plots_dir)
        print(f"✅ Plots generated in {plots_dir}")
    else:
        # Generate full report
        report_path = create_training_report(history_path, output_dir)
        print(f"✅ Full training report created: {report_path}")
        print(f"   Open {report_path} in your browser to view the report")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
