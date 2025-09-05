"""
Loss functions for VCC Transformer.

This module implements the combined loss function for multi-task learning,
including reconstruction loss and classification loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class CombinedLoss(nn.Module):
    """
    Combined loss function for multi-task learning.
    
    Combines reconstruction loss (MSE) and classification loss (CrossEntropy)
    with configurable weighting.
    """
    
    def __init__(
        self,
        reconstruction_weight: float = 1.0,
        classification_weight: float = 0.5,
        reduction: str = 'mean'
    ):
        """
        Initialize the combined loss function.
        
        Args:
            reconstruction_weight: Weight for reconstruction loss
            classification_weight: Weight for classification loss (beta parameter)
            reduction: Reduction method for losses ('mean', 'sum', 'none')
        """
        super().__init__()
        
        self.reconstruction_weight = reconstruction_weight
        self.classification_weight = classification_weight
        self.reduction = reduction
        
        # Individual loss functions
        self.mse_loss = nn.MSELoss(reduction=reduction)
        self.ce_loss = nn.CrossEntropyLoss(reduction=reduction)
        
        logger.info(
            f"CombinedLoss initialized with reconstruction_weight={reconstruction_weight}, "
            f"classification_weight={classification_weight}"
        )
    
    def forward(
        self,
        reconstruction_pred: torch.Tensor,
        reconstruction_target: torch.Tensor,
        classification_pred: torch.Tensor,
        classification_target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.
        
        Args:
            reconstruction_pred: Predicted gene expressions [batch_size, n_genes]
            reconstruction_target: Target gene expressions [batch_size, n_genes]
            classification_pred: Predicted perturbation logits [batch_size, n_classes]
            classification_target: Target perturbation indices [batch_size]
            mask: Optional mask for valid predictions [batch_size, n_genes]
            
        Returns:
            Dict containing individual and combined losses
        """
        # Compute reconstruction loss
        if mask is not None:
            # Apply mask to focus on relevant genes
            reconstruction_pred_masked = reconstruction_pred * mask
            reconstruction_target_masked = reconstruction_target * mask
            recon_loss = self.mse_loss(reconstruction_pred_masked, reconstruction_target_masked)
        else:
            recon_loss = self.mse_loss(reconstruction_pred, reconstruction_target)
        
        # Compute classification loss
        class_loss = self.ce_loss(classification_pred, classification_target)
        
        # Compute combined loss
        total_loss = (
            self.reconstruction_weight * recon_loss +
            self.classification_weight * class_loss
        )
        
        return {
            'total_loss': total_loss,
            'reconstruction_loss': recon_loss,
            'classification_loss': class_loss,
            'reconstruction_weight': self.reconstruction_weight,
            'classification_weight': self.classification_weight
        }


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in perturbation classification.
    
    Focal Loss is particularly useful when some perturbations are much more
    common than others in the training data.
    """
    
    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Class weights [n_classes]
            gamma: Focusing parameter
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super().__init__()
        
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            inputs: Predicted logits [batch_size, n_classes]
            targets: Target class indices [batch_size]
            
        Returns:
            torch.Tensor: Focal loss
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for learning better perturbation representations.
    
    This loss encourages similar perturbations to have similar representations
    while pushing different perturbations apart.
    """
    
    def __init__(
        self,
        temperature: float = 0.1,
        reduction: str = 'mean'
    ):
        """
        Initialize contrastive loss.
        
        Args:
            temperature: Temperature parameter for softmax
            reduction: Reduction method
        """
        super().__init__()
        
        self.temperature = temperature
        self.reduction = reduction
    
    def forward(
        self,
        representations: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute contrastive loss.
        
        Args:
            representations: Feature representations [batch_size, hidden_dim]
            labels: Class labels [batch_size]
            
        Returns:
            torch.Tensor: Contrastive loss
        """
        batch_size = representations.size(0)
        
        # Normalize representations
        representations = F.normalize(representations, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(representations, representations.T) / self.temperature
        
        # Create mask for positive pairs (same class)
        labels = labels.unsqueeze(1)
        mask = (labels == labels.T).float()
        
        # Remove diagonal (self-similarity)
        mask = mask - torch.eye(batch_size, device=mask.device)
        
        # Compute contrastive loss
        exp_sim = torch.exp(similarity_matrix)
        log_prob = similarity_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))
        
        # Apply mask and compute mean
        contrastive_loss = -(log_prob * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        
        if self.reduction == 'mean':
            return contrastive_loss.mean()
        elif self.reduction == 'sum':
            return contrastive_loss.sum()
        else:
            return contrastive_loss


class AdaptiveCombinedLoss(nn.Module):
    """
    Adaptive combined loss that adjusts weights based on training progress.
    
    This loss automatically adjusts the balance between reconstruction and
    classification losses based on their relative magnitudes.
    """
    
    def __init__(
        self,
        initial_reconstruction_weight: float = 1.0,
        initial_classification_weight: float = 0.5,
        adaptation_rate: float = 0.01,
        min_weight: float = 0.1,
        max_weight: float = 10.0
    ):
        """
        Initialize adaptive combined loss.
        
        Args:
            initial_reconstruction_weight: Initial weight for reconstruction loss
            initial_classification_weight: Initial weight for classification loss
            adaptation_rate: Rate of weight adaptation
            min_weight: Minimum weight value
            max_weight: Maximum weight value
        """
        super().__init__()
        
        self.reconstruction_weight = nn.Parameter(
            torch.tensor(initial_reconstruction_weight)
        )
        self.classification_weight = nn.Parameter(
            torch.tensor(initial_classification_weight)
        )
        
        self.adaptation_rate = adaptation_rate
        self.min_weight = min_weight
        self.max_weight = max_weight
        
        # Individual loss functions
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        
        # Running averages for loss magnitudes
        self.register_buffer('recon_loss_avg', torch.tensor(1.0))
        self.register_buffer('class_loss_avg', torch.tensor(1.0))
    
    def forward(
        self,
        reconstruction_pred: torch.Tensor,
        reconstruction_target: torch.Tensor,
        classification_pred: torch.Tensor,
        classification_target: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute adaptive combined loss.
        
        Args:
            reconstruction_pred: Predicted gene expressions
            reconstruction_target: Target gene expressions
            classification_pred: Predicted perturbation logits
            classification_target: Target perturbation indices
            
        Returns:
            Dict containing losses and adaptive weights
        """
        # Compute individual losses
        recon_loss = self.mse_loss(reconstruction_pred, reconstruction_target)
        class_loss = self.ce_loss(classification_pred, classification_target)
        
        # Update running averages
        with torch.no_grad():
            self.recon_loss_avg = (
                (1 - self.adaptation_rate) * self.recon_loss_avg +
                self.adaptation_rate * recon_loss.detach()
            )
            self.class_loss_avg = (
                (1 - self.adaptation_rate) * self.class_loss_avg +
                self.adaptation_rate * class_loss.detach()
            )
            
            # Adapt weights based on relative loss magnitudes
            if self.recon_loss_avg > self.class_loss_avg:
                self.reconstruction_weight.data = torch.clamp(
                    self.reconstruction_weight * 0.99,
                    self.min_weight, self.max_weight
                )
                self.classification_weight.data = torch.clamp(
                    self.classification_weight * 1.01,
                    self.min_weight, self.max_weight
                )
            else:
                self.reconstruction_weight.data = torch.clamp(
                    self.reconstruction_weight * 1.01,
                    self.min_weight, self.max_weight
                )
                self.classification_weight.data = torch.clamp(
                    self.classification_weight * 0.99,
                    self.min_weight, self.max_weight
                )
        
        # Compute weighted combined loss
        total_loss = (
            self.reconstruction_weight * recon_loss +
            self.classification_weight * class_loss
        )
        
        return {
            'total_loss': total_loss,
            'reconstruction_loss': recon_loss,
            'classification_loss': class_loss,
            'reconstruction_weight': self.reconstruction_weight.item(),
            'classification_weight': self.classification_weight.item(),
            'recon_loss_avg': self.recon_loss_avg.item(),
            'class_loss_avg': self.class_loss_avg.item()
        }


def create_loss_function(config) -> nn.Module:
    """
    Create loss function based on configuration.
    
    Args:
        config: Configuration object
        
    Returns:
        nn.Module: Configured loss function
    """
    loss_type = getattr(config.training, 'loss_type', 'combined')
    
    if loss_type == 'combined':
        return CombinedLoss(
            reconstruction_weight=config.training.reconstruction_weight,
            classification_weight=config.training.classification_weight
        )
    elif loss_type == 'adaptive':
        return AdaptiveCombinedLoss(
            initial_reconstruction_weight=config.training.reconstruction_weight,
            initial_classification_weight=config.training.classification_weight
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def compute_challenge_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    perturbation_labels: torch.Tensor
) -> Dict[str, float]:
    """
    Compute challenge-specific evaluation metrics.
    
    Args:
        predictions: Predicted gene expressions [batch_size, n_genes]
        targets: Target gene expressions [batch_size, n_genes]
        perturbation_labels: Perturbation labels [batch_size]
        
    Returns:
        Dict containing computed metrics
    """
    with torch.no_grad():
        # Mean Absolute Error
        mae = F.l1_loss(predictions, targets).item()
        
        # Mean Squared Error
        mse = F.mse_loss(predictions, targets).item()
        
        # Pearson correlation (approximation)
        pred_flat = predictions.view(-1)
        target_flat = targets.view(-1)
        
        pred_centered = pred_flat - pred_flat.mean()
        target_centered = target_flat - target_flat.mean()
        
        correlation = (
            (pred_centered * target_centered).sum() /
            (pred_centered.norm() * target_centered.norm() + 1e-8)
        ).item()
        
        return {
            'mae': mae,
            'mse': mse,
            'correlation': correlation,
            'rmse': mse ** 0.5
        }
