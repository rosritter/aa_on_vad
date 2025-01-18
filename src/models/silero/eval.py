import torch
import torchaudio

def process_audio_sample(model, audio_path, sr=16000):
    """
    Process a single audio file through Silero VAD model.
    
    Args:
        model: Loaded Silero VAD model
        audio_path: Path to audio file
        sr: Sampling rate (16000 or 8000)
    
    Returns:
        Tensor of speech probabilities for each time window
    """
    # First, let's load and preprocess the audio
    # torchaudio.load returns a tuple of (waveform, sample_rate)
    waveform, source_sr = torchaudio.load(audio_path)
    
    # If audio has multiple channels, convert to mono by averaging
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    # Resample if the source sample rate is different from target
    if source_sr != sr:
        resampler = torchaudio.transforms.Resample(source_sr, sr)
        waveform = resampler(waveform)
    
    # The model expects input shape [batch_size, audio_length]
    # waveform is currently [1, audio_length], which is correct
    
    # Get speech probabilities from the model
    # The model internally handles:
    # 1. STFT transformation
    # 2. Feature extraction through encoder
    # 3. Sequential processing through decoder
    speech_probs = model.audio_forward(waveform, sr=sr)
    
    return speech_probs

# Load the model
model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', 
                            model='silero_vad')
model.eval()  # Set to evaluation mode

# Example usage
audio_path = "datasets/speech16.wav"
speech_probabilities = process_audio_sample(model, audio_path)

# The output speech_probabilities has shape [1, num_windows]
# Each value is between 0 and 1, representing speech probability
# Time resolution is approximately 50ms per window

# To get binary speech/non-speech decisions, we can apply a threshold
# Default threshold is usually around 0.5
speech_mask = (speech_probabilities > 0.5).float()

# Print some information about the predictions
print(f"Number of time windows: {speech_probabilities.shape[1]}")
print(f"Detected speech in {speech_mask.sum().item()} windows")

# If you want to get time stamps of speech segments
def get_speech_timestamps(speech_probs, threshold=0.5, window_size_ms=50):
    """
    Convert model predictions to time stamps of speech segments.
    
    Args:
        speech_probs: Model predictions
        threshold: Classification threshold
        window_size_ms: Size of each window in milliseconds
    
    Returns:
        List of dictionaries containing start and end times in seconds
    """
    mask = (speech_probs[0] > threshold).cpu().numpy()
    speech_segments = []
    
    in_speech = False
    start_idx = 0
    
    for i, is_speech in enumerate(mask):
        if is_speech and not in_speech:
            start_idx = i
            in_speech = True
        elif not is_speech and in_speech:
            # Convert window indices to seconds
            start_time = start_idx * window_size_ms / 1000
            end_time = i * window_size_ms / 1000
            speech_segments.append({
                'start': start_time,
                'end': end_time
            })
            in_speech = False
    
    # Handle if audio ends during speech
    if in_speech:
        start_time = start_idx * window_size_ms / 1000
        end_time = len(mask) * window_size_ms / 1000
        speech_segments.append({
            'start': start_time,
            'end': end_time
        })
    
    return speech_segments

# Get time stamps of speech segments
timestamps = get_speech_timestamps(speech_probabilities)

# Print the speech segments
for segment in timestamps:
    print(f"Speech from {segment['start']:.2f}s to {segment['end']:.2f}s")