import torch
from typing import Dict, List, Tuple, Union
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.models.silero.eval import get_vad_mask
from sklearn.metrics import roc_auc_score, roc_curve


class VADEvaluator:
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.reset()

    def reset(self):
        self.all_predictions = []
        self.all_labels = []

    def update(self, predictions: torch.Tensor, labels: torch.Tensor) -> None:
        # Store raw predictions for later computation
        predictions_cpu = predictions.detach().cpu()
        labels_cpu = labels.detach().cpu()
        
        if predictions_cpu.dim() == 3:
            predictions_cpu = predictions_cpu.squeeze(1)
        if labels_cpu.dim() == 3:
            labels_cpu = labels_cpu.squeeze(1)
            
        self.all_predictions.extend(predictions_cpu.flatten().numpy())
        self.all_labels.extend(labels_cpu.flatten().numpy())

    def compute(self, threshold: float = None) -> Dict[str, float]:
        if threshold is not None:
            self.threshold = threshold
            
        # Convert lists to numpy arrays for efficient computation
        predictions = np.array(self.all_predictions)
        labels = np.array(self.all_labels)
        binary_preds = (predictions > self.threshold).astype(float)
        
        # Compute confusion matrix
        tp = np.sum((binary_preds == 1) & (labels == 1))
        fp = np.sum((binary_preds == 1) & (labels == 0))
        tn = np.sum((binary_preds == 0) & (labels == 0))
        fn = np.sum((binary_preds == 0) & (labels == 1))

        eps = 1e-8
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
        fpr = fp / (fp + tn + eps)
        tpr = recall

        # Compute ROC-AUC
        roc_auc = roc_auc_score(labels, predictions)
        fpr_curve, tpr_curve, _ = roc_curve(labels, predictions)

        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'accuracy': accuracy,
            'false_positive_rate': fpr,
            'true_positive_rate': tpr,
            'roc_auc': roc_auc,
            'roc_curve': {
                'fpr': fpr_curve,
                'tpr': tpr_curve
            }
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



def validate_silero_vad(model, dataset, device, get_mask=get_vad_mask):
    """
    Validate Silero VAD model using the VADEvaluator
    
    Args:
        model: Silero VAD model instance
        dataset: Dataset instance providing audio samples and labels
        device: torch device
        batch_size: batch size for DataLoader
    """
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    evaluator = VADEvaluator(threshold=0.5)
    sample_rate = 16000
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader):
            # Move data to device
            wavs = batch['sample'].to(device)
            masks_true = batch['mask'].to(device)
            
            # Process each audio in batch
            mask_pred = get_mask(
                wavs, 
                model, 
                sample_rate=sample_rate)
            
            # Update evaluator
            evaluator.update(mask_pred, masks_true)
    
    # Compute and return metrics
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