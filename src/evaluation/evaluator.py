import torch
from typing import Dict, List, Tuple, Union
import numpy as np


class VADEvaluator:
    def __init__(self, threshold: float = 0.5):
        """
        Initialize the VAD evaluator.

        Args:
            threshold (float): Threshold for converting probabilities to binary predictions
        """
        self.threshold = threshold
        self.reset()

    def reset(self):
        """Reset all accumulated statistics"""
        self.total_tp = 0
        self.total_fp = 0
        self.total_tn = 0
        self.total_fn = 0

    def update(self, predictions: torch.Tensor, labels: torch.Tensor) -> None:
        """
        Update metrics with a new batch of predictions and labels.

        Args:
            predictions (torch.Tensor): Model predictions (B, T) or (B, 1, T)
            labels (torch.Tensor): Ground truth labels (B, T) or (B, 1, T)
        """
        # Ensure tensors are on CPU and convert to binary
        predictions = (predictions.detach().cpu() > self.threshold).float()
        labels = labels.detach().cpu().float()

        # Ensure shapes match
        if predictions.dim() == 3:
            predictions = predictions.squeeze(1)
        if labels.dim() == 3:
            labels = labels.squeeze(1)

        # Calculate confusion matrix elements
        tp = ((predictions == 1) & (labels == 1)).sum().item()
        fp = ((predictions == 1) & (labels == 0)).sum().item()
        tn = ((predictions == 0) & (labels == 0)).sum().item()
        fn = ((predictions == 0) & (labels == 1)).sum().item()

        # Update totals
        self.total_tp += tp
        self.total_fp += fp
        self.total_tn += tn
        self.total_fn += fn

    def compute(self) -> Dict[str, float]:
        """
        Compute final metrics.

        Returns:
            Dict[str, float]: Dictionary containing various metrics
        """
        # Avoid division by zero
        eps = 1e-8

        # Calculate metrics
        precision = self.total_tp / (self.total_tp + self.total_fp + eps)
        recall = self.total_tp / (self.total_tp + self.total_fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        accuracy = (self.total_tp + self.total_tn) / (self.total_tp + self.total_tn +
                                                      self.total_fp + self.total_fn + eps)

        # Calculate false positive rate and true positive rate for ROC
        fpr = self.total_fp / (self.total_fp + self.total_tn + eps)
        tpr = recall  # Same as recall

        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'accuracy': accuracy,
            'false_positive_rate': fpr,
            'true_positive_rate': tpr
        }


def evaluate_vad_dataset(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device,
        threshold: float = 0.5
) -> Dict[str, float]:
    """
    Evaluate VAD model on entire dataset.

    Args:
        model (torch.nn.Module): The VAD model
        dataloader (DataLoader): DataLoader containing the evaluation dataset
        device (torch.device): Device to run evaluation on
        threshold (float): Threshold for binary classification

    Returns:
        Dict[str, float]: Dictionary containing evaluation metrics
    """
    model.eval()
    evaluator = VADEvaluator(threshold=threshold)

    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            predictions = model(inputs)

            # Update metrics
            evaluator.update(predictions, labels)

    # Compute final metrics
    metrics = evaluator.compute()
    return metrics


# Example usage:
"""
# Initialize model and dataloader
model = YourVADModel()
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Evaluate
metrics = evaluate_vad_dataset(model, dataloader, device)
print("Evaluation metrics:")
for metric_name, value in metrics.items():
    print(f"{metric_name}: {value:.4f}")
"""