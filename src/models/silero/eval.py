import torch.nn.functional as F
import torch
from torch.onnx.symbolic_opset11 import chunk


@torch.no_grad()
def get_vad_mask(
    audio: torch.Tensor,
    model,
    amodel = None,
    threshold: float = 0.5,
    sample_rate: int = 16000,
    num_samples: int = 512,
    context_size = 64,
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
    
    # Handle sample_rate
    if sample_rate > 16000 and (sample_rate % 16000 == 0):
        step = sample_rate // 16000
        sample_rate = 16000
        audio = audio[::step]
    
    # Reset model states
    # if hasattr(model, 'reset_states'):
    #     model.reset_states()
    
    state = torch.zeros((2, audio.shape[0], 128)).to(audio.device)  # fixed for silero
    # Initialize mask
    mask = torch.zeros(audio.shape)
    # Pad input for context
    x = torch.nn.functional.pad(audio, (context_size, audio.shape[-1] % num_samples))
    # Process audio in windows
    for i in range(context_size, x.shape[-1], num_samples):
        # Get current window with context
        input_window = x[:, i - context_size:i + num_samples]

        # Generate noise for main window (without context)
        if amodel:
            chunk_noise = amodel(input_window)

            noisy_window = input_window + chunk_noise
            chunk = noisy_window
        else:
            chunk = input_window
        # Get prediction
        out = model._model.stft(chunk)
        out = model._model.encoder(out)
        out, state = model._model.decoder(out, state)
        
        # Fill mask for this window
        mask[:, i :i + num_samples] = out.squeeze(1)  >= threshold
    
    return mask