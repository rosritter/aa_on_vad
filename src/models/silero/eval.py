import torch.nn.functional as F
import torch


@torch.no_grad()
def get_vad_mask(
    audio: torch.Tensor,
    model,
    threshold: float = 0.5,
    sample_rate: int = 16000,
    window_size_samples: int = 512
) -> torch.Tensor:
    """
    Convert VAD model predictions into a binary mask.
    
    Args:
        audio: torch.Tensor - Input audio (1D tensor)
        model: VAD model
        threshold: float - Speech probability threshold
        sample_rate: int - Audio sampling rate
        window_size_samples: int - Window size for processing
        
    Returns:
        torch.Tensor - Binary mask of same length as input audio
    """
    # Ensure audio is 1D
    if not torch.is_tensor(audio):
        audio = torch.tensor(audio)
    audio = audio.squeeze()
    
    # Handle sample_rate
    if sample_rate > 16000 and (sample_rate % 16000 == 0):
        step = sample_rate // 16000
        sample_rate = 16000
        audio = audio[::step]
    
    # Reset model states
    if hasattr(model, 'reset_states'):
        model.reset_states()
    
    # Initialize mask
    audio_length = len(audio)
    mask = torch.zeros(audio_length)
    # Process audio in windows
    for start_idx in range(0, audio_length, window_size_samples):
        # Get chunk
        chunk = audio[start_idx: start_idx + window_size_samples]
        
        # Pad last chunk if needed
        if len(chunk) < window_size_samples:
            chunk = F.pad(chunk, (0, window_size_samples - len(chunk)))
        
        # Get prediction
        speech_prob = model(chunk, sample_rate).item()
        
        # Fill mask for this window
        end_idx = min(start_idx + window_size_samples, audio_length)
        mask[start_idx:end_idx] = float(speech_prob >= threshold)
    
    return mask