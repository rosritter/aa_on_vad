import torch
import torch.nn.functional as F
import random
import numpy as np
from typing import List, Dict, Optional, Union, Tuple
from torch.utils.data import Dataset
from dataclasses import dataclass

@dataclass
class LengthConfig:
    """Configuration for output audio length"""
    min_length: float  # in seconds
    max_length: float  # in seconds
    
    def get_length_samples(self, sample_rate: int, random_state: Optional[np.random.RandomState] = None) -> int:
        """Get random length in samples within configured range"""
        min_samples = int(self.min_length * sample_rate)
        max_samples = int(self.max_length * sample_rate)
        if random_state is not None:
            return random_state.randint(min_samples, max_samples + 1)
        return random.randint(min_samples, max_samples)

class VADMixedDataset(Dataset):
    def __init__(
        self,
        speech_datasets: List[Dataset],
        noise_datasets: List[Dataset],
        sample_rate: int = 16000,
        length_config: Optional[LengthConfig] = None,
        fixed_length: Optional[float] = None,  # in seconds
        target_speech_ratio: float = 0.3,  # target ratio of speech in generated audio
        speech_ratio_tolerance: float = 0.05,  # allowed deviation from target ratio
        speech_prob_weights: Optional[List[float]] = None,  # weights for each speech dataset
        noise_prob_weights: Optional[List[float]] = None,  # weights for each noise dataset
        min_speech_length: float = 0.2,  # minimum speech segment length in seconds
        deterministic: bool = False,  # whether to generate same samples each time
        seed: int = 42,  # seed for deterministic mode
    ):
        """
        Initialize the VAD mixed dataset.
        
        Args:
            speech_datasets: List of datasets containing speech samples
            noise_datasets: List of datasets containing non-speech samples
            sample_rate: Audio sample rate
            length_config: Configuration for random length range (if not using fixed_length)
            fixed_length: Fixed length for all outputs (in seconds, if not using length_config)
            target_speech_ratio: Target ratio of speech presence in generated audio (0.0 to 1.0)
            speech_ratio_tolerance: Allowed deviation from target speech ratio
            speech_prob_weights: Optional weights for sampling from speech datasets
            noise_prob_weights: Optional weights for sampling from noise datasets
            min_speech_length: Minimum length for individual speech segments
            deterministic: If True, will generate same samples each time for same index
            seed: Random seed for deterministic mode
        """
        if length_config is not None and fixed_length is not None:
            raise ValueError("Cannot specify both length_config and fixed_length")
        elif length_config is None and fixed_length is None:
            self.length_config = None
            self.fixed_length = 3.0
        else:
            self.length_config = length_config
            self.fixed_length = fixed_length
        
        self.speech_datasets = speech_datasets
        self.noise_datasets = noise_datasets
        self.sample_rate = sample_rate
        self.target_speech_ratio = target_speech_ratio
        self.speech_ratio_tolerance = speech_ratio_tolerance
        self.min_speech_length = int(min_speech_length * sample_rate)
        self.deterministic = deterministic
        self.seed = seed
        
        # Set default uniform weights
        if speech_prob_weights is None:
            self.speech_prob_weights = [1.0 / len(speech_datasets)] * len(speech_datasets)
        else:
            total = sum(speech_prob_weights)
            self.speech_prob_weights = [w / total for w in speech_prob_weights]
        
        if noise_prob_weights is None:
            self.noise_prob_weights = [1.0 / len(noise_datasets)] * len(noise_datasets)
        else:
            total = sum(noise_prob_weights)
            self.noise_prob_weights = [w / total for w in noise_prob_weights]
        
        # Store dataset lengths
        self.speech_lengths = [len(dataset) for dataset in speech_datasets]
        self.noise_lengths = [len(dataset) for dataset in noise_datasets]
        self.length = sum(self.speech_lengths) + sum(self.noise_lengths)

    def _get_random_generator(self, idx: int) -> Union[np.random.RandomState, None]:
        """Get random generator based on mode and index"""
        if self.deterministic:
            return np.random.RandomState(self.seed + idx)
        return None

    def _random_choice(self, 
                      options: List[int], 
                      weights: List[float], 
                      random_state: Optional[np.random.RandomState] = None) -> int:
        """Make a random choice with optional deterministic behavior"""
        if random_state is not None:
            return random_state.choice(options, p=weights)
        return random.choices(options, weights=weights)[0]

    def _random_int(self, 
                   low: int, 
                   high: int, 
                   random_state: Optional[np.random.RandomState] = None) -> int:
        """Get random integer with optional deterministic behavior"""
        if random_state is not None:
            return random_state.randint(low, high)
        return random.randint(low, high - 1)

    def _get_target_length(self, random_state: Optional[np.random.RandomState] = None) -> int:
        """Get target length in samples for current audio segment"""
        if self.length_config is not None:
            return self.length_config.get_length_samples(self.sample_rate, random_state)
        else:
            return int(self.fixed_length * self.sample_rate)

    def _get_random_slice(self, 
                         audio: torch.Tensor, 
                         target_length: int,
                         random_state: Optional[np.random.RandomState] = None) -> torch.Tensor:
        """Get a random slice of specified length from audio tensor"""
        if audio.size(-1) <= target_length:
            padding = target_length - audio.size(-1)
            return F.pad(audio, (0, padding))
        else:
            start = self._random_int(0, audio.size(-1) - target_length + 1, random_state)
            return audio[..., start:start + target_length]

    def _get_audio_from_dataset(self, 
                              is_speech: bool,
                              random_state: Optional[np.random.RandomState] = None) -> torch.Tensor:
        """Get a random audio sample from specified dataset type"""
        datasets = self.speech_datasets if is_speech else self.noise_datasets
        weights = self.speech_prob_weights if is_speech else self.noise_prob_weights
        
        dataset_idx = self._random_choice(
            range(len(datasets)), 
            weights=weights,
            random_state=random_state
        )
        
        dataset = datasets[dataset_idx]
        sample_idx = self._random_int(0, len(dataset), random_state)
        return dataset[sample_idx]['sample']

    def _calculate_current_speech_ratio(self, mask: torch.Tensor) -> float:
        """Calculate the current ratio of speech in the mask"""
        return mask.mean().item()

    def _generate_mixed_segment(self, 
                              target_length: int,
                              random_state: Optional[np.random.RandomState] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate a mixed audio segment with the target speech ratio"""
        output = torch.zeros(target_length)
        mask = torch.zeros(target_length)
        available_positions = set(range(target_length))
        current_ratio = 0.0

        while (abs(current_ratio - self.target_speech_ratio) > self.speech_ratio_tolerance and 
               len(available_positions) >= self.min_speech_length):
            
            if current_ratio < self.target_speech_ratio:
                # Need more speech
                audio = self._get_audio_from_dataset(is_speech=True, random_state=random_state)
                max_length = min(audio.size(-1), len(available_positions))
                segment_length = self._random_int(self.min_speech_length, max_length + 1, random_state)
            else:
                # Need more noise
                audio = self._get_audio_from_dataset(is_speech=False, random_state=random_state)
                max_length = min(audio.size(-1), len(available_positions))
                segment_length = self._random_int(self.min_speech_length, max_length + 1, random_state)

            if len(available_positions) < segment_length:
                break

            # Find a continuous segment of available positions
            available_list = sorted(list(available_positions))
            start_idx = self._random_int(0, len(available_list) - segment_length + 1, random_state)
            segment_positions = available_list[start_idx:start_idx + segment_length]
            
            # Update available positions
            available_positions -= set(segment_positions)
            
            # Add audio segment
            audio_segment = self._get_random_slice(audio, segment_length, random_state)
            output[segment_positions] = audio_segment[:segment_length]
            
            # Update mask for speech segments
            if current_ratio < self.target_speech_ratio:
                mask[segment_positions] = 1.0
            
            current_ratio = self._calculate_current_speech_ratio(mask)

        # Fill any remaining positions with noise
        if available_positions:
            noise = self._get_audio_from_dataset(is_speech=False, random_state=random_state)
            remaining_positions = sorted(list(available_positions))
            noise_segment = self._get_random_slice(noise, len(remaining_positions), random_state)
            output[remaining_positions] = noise_segment

        return output, mask

    def _maybe_normalize_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """Normalize audio to have max amplitude of 1"""
        max_val = torch.abs(audio).max()
        if max_val > 0:
            return audio / max_val
        return audio

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Get random generator based on mode
        random_state = self._get_random_generator(idx)
        
        # Get target length for this sample
        target_length = self._get_target_length(random_state)
        
        # Generate mixed segment with target speech ratio
        mixed_audio, mask = self._generate_mixed_segment(target_length, random_state)
        
        # Normalize final audio
        mixed_audio = self._maybe_normalize_audio(mixed_audio)
        
        return {
            'sample': mixed_audio,
            'mask': mask
        }
    


if __name__ == '__main__':
    from src.datasets.librispeech import get_librispeech_example, LibriSpeechWrapper
    from src.datasets.urbansound import UrbanSoundDataset, read_arrf
    from src.datasets.musan import MusanMusicDataset

    speech_datasets = [
    LibriSpeechWrapper(get_librispeech_example(), erase_silence=True)
]
    
    noise_datasets = [
    MusanMusicDataset(
    root_dir='datasets/musan/music',
    target_sample_rate=16000,
    segment_length=None
    ),
    UrbanSoundDataset(read_arrf())
]
    vad_dataset_fixed = VADMixedDataset(
    speech_datasets=speech_datasets,
    noise_datasets=noise_datasets,
    sample_rate=16000,
    fixed_length=6.0,
    target_speech_ratio=0.3,
    deterministic=True
)
    fixed_sample = vad_dataset_fixed[2]    # Will be exactly 6 seconds

# Example usage:
"""
# Initialize datasets
speech_datasets = [
    torchaudio.datasets.LIBRISPEECH("./data", url="train-clean-100", download=True),
    YourCustomSpeechDataset1(),
]

noise_datasets = [
    YourNoiseDataset1(),
    YourNoiseDataset2(),
]

# Example 1: Fixed length output
vad_dataset_fixed = VADMixedDataset(
    speech_datasets=speech_datasets,
    noise_datasets=noise_datasets,
    sample_rate=16000,
    fixed_length=3.0,  # 3 seconds fixed length
    target_speech_ratio=0.3
)

# Example 2: Random length output
length_config = LengthConfig(min_length=2.0, max_length=5.0)  # Random between 2-5 seconds
vad_dataset_random = VADMixedDataset(
    speech_datasets=speech_datasets,
    noise_datasets=noise_datasets,
    sample_rate=16000,
    length_config=length_config,
    target_speech_ratio=0.3
)

# Get samples
fixed_sample = vad_dataset_fixed[0]    # Will be exactly 3 seconds
random_sample = vad_dataset_random[0]  # Will be between 2-5 seconds
"""