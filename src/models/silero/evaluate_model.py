import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.evaluation.evaluator import VADEvaluator
from src.datasets.mixed_vad_datset import get_datset
from src.models.silero.eval import get_vad_mask
from src.models.adversarial_models.CNN1d_based import NoiseGenerator


def validate_silero_vad(model, dataset, batch_size=32, amodel=None, threshold=0.5):
    """
    Validate Silero VAD model using the VADEvaluator
    
    Args:
        model: Silero VAD model instance
        dataset: Dataset instance providing audio samples and labels
        device: torch device
        batch_size: batch size for DataLoader
    """
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    evaluator = VADEvaluator(threshold=0.5)
    sample_rate = 16000
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        for batch in tqdm(dataloader):
            # Move data to device
            wavs = batch['sample'].to(device)
            masks_true = batch['mask'].to(device)
            
            # Process each audio in batch
            mask_pred = get_vad_mask(
                wavs, 
                model,
                amodel,
                sample_rate=sample_rate,
                threshold=threshold)
            
            # Update evaluator
            evaluator.update(mask_pred, masks_true)
    
    # Compute and return metrics
    metrics = evaluator.compute()
    return metrics

def print_metrics(metrics:dict):
    for metric_name, value in metrics.items():
        if metric_name != 'roc_curve':
            print(f"{metric_name}: {value:.4f}")


if __name__ == '__main__':
    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
    print(model)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    # Create dataset
    print('Loading dataset...')
    dataset = get_datset(mode='test', erase_silence=True)
    print('Dataset loaded!')
    # Run validation
    metrics = validate_silero_vad(
                                   model,
                                  dataset,
                                  threshold=0.7,
                                  amodel=NoiseGenerator().to(device)
                                    )
    print("Validation metrics:")
    print_metrics(metrics)