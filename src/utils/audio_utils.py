import torchaudio
import torch
import numpy as np
import soundfile as sf
from typing import Union

def read_torch(file_path, reader:callable=torchaudio.load) -> Union[torch.tensor, np.ndarray]: 
    '''
    available readers
    
    - torchaudio.load
    - sf.read
    '''
    
    # Load audio

    waveform, sr = reader(file_path)
    if waveform.ndim > 1:
        waveform = waveform[0, :]
    # Resample if necessary
    return waveform, sr


def remove_silence(audio, sample_rate, energy_threshold=0.02, step_duration=0.01):
    """
    Removes silence from an audio waveform based on an energy threshold.

    Args:
        audio (torch.Tensor): The input audio waveform. Shape: (1, num_samples) or (num_samples,).
        sample_rate (int): The sampling rate of the audio.
        energy_threshold (float): The energy threshold below which audio is considered silence. Default: 0.02.
        step_duration (float): The duration (in seconds) of each step for energy evaluation. Default: 0.01.

    Returns:
        torch.Tensor: The audio waveform with silence removed.
    """
    if len(audio.shape) == 1:
        audio = audio.unsqueeze(0)  # Convert to 2D for consistency (1, num_samples)

    # Calculate step size in samples
    step_size = int(step_duration * sample_rate)

    # Ensure the step size is valid
    if step_size <= 0:
        raise ValueError("Step size must be greater than 0.")

    # Initialize the list to hold non-silent segments
    non_silent_segments = []

    # Process the audio in chunks
    for start in range(0, audio.shape[1], step_size):
        end = min(start + step_size, audio.shape[1])
        chunk = audio[:, start:end]

        # Compute energy of the chunk
        energy = torch.sqrt(torch.mean(chunk**2))

        # Retain chunk if energy exceeds the threshold
        if energy > energy_threshold:
            non_silent_segments.append(chunk)

    # Concatenate non-silent segments
    if non_silent_segments:
        waveform = torch.cat(non_silent_segments, dim=1)
    else:
        # If no audio remains, return a zero tensor
        waveform = torch.tensor([])
    return waveform
    

def vad_forward(waveform: torch.Tensor, vad: callable, sample_rate: int = 16000, step_sec=1, idx: int = 0) -> torch.Tensor:
    """
    Processes a waveform by removing silent segments using a given VAD function.

    Args:
        waveform (torch.Tensor): Input audio waveform. Shape: (n,) or (1, n).
        vad (callable): A function that processes a waveform chunk and removes silence.
        sample_rate (int): The sample rate of the audio. Default: 16000.
        idx (int): Optional identifier for the waveform (used for logging). Default: 0.

    Returns:
        torch.Tensor: The silence-free waveform. Shape: (1, m) or (m,).
    """
    silence_free_waveform = []
    chunk_size = int(sample_rate * step_sec)  # 200ms chunks

    # Ensure waveform is 2D for consistency
    if len(waveform.shape) == 1:
        waveform = waveform.unsqueeze(0)  # Convert to shape (1, num_samples)

    # Process waveform in chunks
    for start in range(0, waveform.shape[-1], chunk_size):
        chunk = waveform[:, start:start + chunk_size]  # Shape: (1, chunk_size)
        try:
            processed_chunk = vad(chunk)
            if processed_chunk.numel() > 0:  # Check if the chunk is non-empty
                silence_free_waveform.append(processed_chunk)
        except Exception as e:
            print(f"VAD processing failed for idx={idx}, chunk={start // chunk_size}: {e}")

    # Concatenate all non-silent chunks
    if silence_free_waveform:
        # Ensure all tensors have the same shape along dim=1 before concatenating
        silence_free_waveform = [chunk for chunk in silence_free_waveform if chunk.numel() > 0]
        waveform = torch.cat(silence_free_waveform, dim=1)
    else:
        waveform = torch.tensor([])  # Return empty tensor if no chunks remain

    return waveform