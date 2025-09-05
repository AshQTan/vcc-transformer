#!/usr/bin/env python3
"""
Prediction script for VCC Transformer.

This script handles inference and submission file generation:
- Loading trained model from checkpoint
- Creating predictions for validation set
- Formatting output for challenge submission
- Running cell-eval prep for final submission
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import polars as pl
import torch
import scanpy as sc
import anndata as ad
from tqdm import tqdm

from vcc_transformer.utils.config import load_config, print_config
from vcc_transformer.models.transformer import MultiTaskTransformer
from vcc_transformer.data.dataset import VCCDataset


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('prediction.log')
        ]
    )


class VCCPredictor:
    """
    Predictor class for generating submissions for VCC challenge.
    """
    
    def __init__(
        self,
        config,
        model_path: str,
        device: str = "cuda"
    ):
        """
        Initialize the predictor.
        
        Args:
            config: Configuration object
            model_path: Path to trained model checkpoint
            device: Device to run inference on
        """
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # Load preprocessing information
        self.preprocessing_info = None
        
        logging.info(f"Predictor initialized on {self.device}")
    
    def _load_model(self, model_path: str) -> MultiTaskTransformer:
        """Load trained model from checkpoint."""
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
        
        logging.info(f"Loading model from {model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Create model
        model = MultiTaskTransformer(self.config)
        
        # Load state dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint  # Assume direct state dict
        
        # Handle DataParallel/DDP state dict
        if any(key.startswith('module.') for key in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict)
        model.to(self.device)
        
        logging.info("Model loaded successfully")
        return model
    
    def predict_validation_set(
        self,
        validation_file: str,
        training_file: str,
        output_file: str,
        batch_size: int = 32
    ) -> None:
        """
        Generate predictions for validation set and save as .h5ad file.
        
        Args:
            validation_file: Path to validation CSV file
            training_file: Path to training .h5ad file
            output_file: Path for output .h5ad file
            batch_size: Batch size for inference
        """
        logging.info("Starting validation set prediction...")
        
        # Load validation perturbations
        val_df = pl.read_csv(validation_file)
        required_perturbations = val_df['perturbation'].unique().to_list()
        
        logging.info(f"Found {len(required_perturbations)} unique perturbations to predict")
        
        # Load training data for control cells and metadata
        logging.info(f"Loading training data from {training_file}")
        adata_train = sc.read_h5ad(training_file)
        
        # Get control cells and NTC cells
        control_mask = adata_train.obs['perturbation'] == 'Non-Targeting Control'
        control_cells = adata_train[control_mask].copy()
        
        # Extract NTC cells for final submission
        ntc_cells = control_cells.copy()
        
        logging.info(f"Found {len(control_cells)} control cells")
        
        # Create prediction dataset
        pred_dataset = VCCDataset(self.config, mode="predict")
        pred_loader = pred_dataset.get_dataloader(
            batch_size=batch_size,
            shuffle=False,
            num_workers=2
        )
        
        # Generate predictions
        all_predictions = []
        all_metadata = []
        
        logging.info("Generating predictions...")
        
        with torch.no_grad():
            for batch in tqdm(pred_loader, desc="Predicting"):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(batch['input_sequence'])
                predictions = outputs['reconstruction']
                
                # Store predictions and metadata
                all_predictions.append(predictions.cpu().numpy())
                all_metadata.extend([
                    {
                        'perturbation': pert_name,
                        'control_idx': i
                    }
                    for i, pert_name in enumerate(batch['perturbation_name'])
                ])
        
        # Concatenate all predictions
        all_predictions = np.concatenate(all_predictions, axis=0)
        
        logging.info(f"Generated {len(all_predictions)} predictions")
        
        # Create submission AnnData object
        submission_adata = self._create_submission_adata(
            all_predictions,
            all_metadata,
            control_cells,
            ntc_cells,
            pred_dataset
        )
        
        # Save submission file
        logging.info(f"Saving submission to {output_file}")
        submission_adata.write_h5ad(output_file)
        
        logging.info("Prediction completed successfully!")
    
    def _create_submission_adata(
        self,
        predictions: np.ndarray,
        metadata: List[Dict],
        control_cells: ad.AnnData,
        ntc_cells: ad.AnnData,
        dataset: VCCDataset
    ) -> ad.AnnData:
        """
        Create AnnData object for submission.
        
        Args:
            predictions: Predicted gene expressions
            metadata: Prediction metadata
            control_cells: Control cells from training data
            ntc_cells: NTC cells to include in submission
            dataset: Dataset object for gene names
            
        Returns:
            ad.AnnData: Submission AnnData object
        """
        # Create observation metadata
        obs_data = []
        
        # Add predictions
        for i, pred_meta in enumerate(metadata):
            obs_data.append({
                'perturbation': pred_meta['perturbation'],
                'cell_type': 'predicted',
                'is_control': False
            })
        
        # Add NTC cells
        for i in range(len(ntc_cells)):
            obs_data.append({
                'perturbation': 'Non-Targeting Control',
                'cell_type': 'control',
                'is_control': True
            })
        
        # Create observation DataFrame
        obs_df = pd.DataFrame(obs_data)
        obs_df.index = [f"cell_{i}" for i in range(len(obs_df))]
        
        # Combine expression data
        # Predictions
        pred_expr = predictions
        
        # NTC expression data (use original values)
        ntc_expr = ntc_cells.X.toarray() if hasattr(ntc_cells.X, 'toarray') else ntc_cells.X
        
        # Combine expression matrices
        combined_expr = np.vstack([pred_expr, ntc_expr])
        
        # Create variable (gene) metadata
        var_df = pd.DataFrame(index=dataset.hvg_names)
        var_df['highly_variable'] = True
        
        # Create AnnData object
        submission_adata = ad.AnnData(
            X=combined_expr,
            obs=obs_df,
            var=var_df
        )
        
        # Add metadata
        submission_adata.uns['method'] = 'MultiTaskTransformer'
        submission_adata.uns['submission_info'] = {
            'n_predictions': len(predictions),
            'n_ntc_cells': len(ntc_cells),
            'n_genes': len(dataset.hvg_names),
            'model_config': dict(self.config.model)
        }
        
        return submission_adata
    
    def predict_single_perturbation(
        self,
        control_expression: np.ndarray,
        perturbation: str,
        perturbation_idx: int = None
    ) -> Dict[str, np.ndarray]:
        """
        Predict effect of single perturbation on control cell.
        
        Args:
            control_expression: Control cell expression values
            perturbation: Perturbation name
            perturbation_idx: Perturbation index (optional)
            
        Returns:
            Dict containing predicted expression and confidence
        """
        self.model.eval()
        
        with torch.no_grad():
            # Prepare input
            if perturbation_idx is None:
                # Look up perturbation index (would need access to dataset)
                perturbation_idx = self.config.model.unk_pert_token_id
            
            # Create input sequence
            input_seq = torch.zeros(1, self.config.model.max_seq_length)
            input_seq[0, 0] = self.config.model.cls_token_id
            input_seq[0, 1] = perturbation_idx
            input_seq[0, 2:2+len(control_expression)] = torch.tensor(control_expression)
            
            input_seq = input_seq.to(self.device)
            
            # Forward pass
            outputs = self.model(input_seq)
            
            # Extract results
            predicted_expression = outputs['reconstruction'][0].cpu().numpy()
            perturbation_confidence = torch.softmax(outputs['classification'][0], dim=0).cpu().numpy()
            
            return {
                'predicted_expression': predicted_expression,
                'perturbation_confidence': perturbation_confidence,
                'perturbation': perturbation,
                'perturbation_idx': perturbation_idx
            }
    
    def run_cell_eval_prep(self, submission_file: str, output_dir: str = None) -> str:
        """
        Run cell-eval prep on submission file.
        
        Args:
            submission_file: Path to submission .h5ad file
            output_dir: Output directory for prepared file
            
        Returns:
            str: Path to prepared submission file
        """
        import subprocess
        
        if output_dir is None:
            output_dir = Path(submission_file).parent
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Construct cell-eval prep command
        cmd = [
            "cell-eval", "prep",
            "--input", str(submission_file),
            "--output-dir", str(output_dir)
        ]
        
        logging.info(f"Running cell-eval prep: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logging.info("cell-eval prep completed successfully")
            logging.info(result.stdout)
            
            # Find prepared file
            prepared_files = list(output_dir.glob("*_prepared.h5ad"))
            if prepared_files:
                return str(prepared_files[0])
            else:
                logging.warning("Prepared file not found")
                return None
                
        except subprocess.CalledProcessError as e:
            logging.error(f"cell-eval prep failed: {e}")
            logging.error(e.stderr)
            raise
        except FileNotFoundError:
            logging.error("cell-eval command not found. Please install cell-eval package.")
            raise


def main():
    """Main prediction function."""
    parser = argparse.ArgumentParser(description="Generate predictions with VCC Transformer")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--validation-file",
        type=str,
        required=True,
        help="Path to validation CSV file"
    )
    parser.add_argument(
        "--training-file",
        type=str,
        required=True,
        help="Path to training .h5ad file"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="predictions.h5ad",
        help="Output file for predictions"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for inference"
    )
    parser.add_argument(
        "--run-cell-eval",
        action="store_true",
        help="Run cell-eval prep on output file"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    
    # Print configuration
    print_config(config, "Prediction Configuration")
    
    # Create predictor
    predictor = VCCPredictor(
        config=config,
        model_path=args.model_path,
        device=args.device
    )
    
    # Generate predictions
    predictor.predict_validation_set(
        validation_file=args.validation_file,
        training_file=args.training_file,
        output_file=args.output_file,
        batch_size=args.batch_size
    )
    
    # Run cell-eval prep if requested
    if args.run_cell_eval:
        try:
            prepared_file = predictor.run_cell_eval_prep(args.output_file)
            if prepared_file:
                logger.info(f"Prepared submission file: {prepared_file}")
        except Exception as e:
            logger.warning(f"cell-eval prep failed: {e}")
    
    logger.info("Prediction pipeline completed successfully!")


if __name__ == "__main__":
    main()
