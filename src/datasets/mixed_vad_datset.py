import torch
import torch.nn.functional as F
import random
import numpy as np
from typing import List, Dict, Optional, Union, Tuple, Literal
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

@dataclass
class SpeechConfig:
    """Configuration for speech presence in samples"""
    min_length: float  # minimum total speech length in seconds
    max_length: float  # maximum total speech length in seconds
    presence_prob: float = 0.7  # probability of having any speech in a sample

class NoiseGenerator:
    @staticmethod
    def white_noise(n: int, device: torch.device = torch.device('cpu')) -> torch.Tensor:
        """Generate white noise as torch tensor"""
        white = torch.randn(n, device=device)
        return white / torch.max(torch.abs(white))
    
    @staticmethod
    def pink_noise(n: int, device: torch.device = torch.device('cpu')) -> torch.Tensor:
        """Generate pink noise as torch tensor"""
        num_columns = int(np.ceil(np.log2(n)))
        array = torch.randn((num_columns, int(np.ceil(n / num_columns))), device=device)
        pink = torch.cumsum(array, dim=0) / torch.sqrt(torch.tensor(num_columns, device=device))
        pink = pink.flatten()[:n]
        return pink / torch.max(torch.abs(pink))
    
    @staticmethod
    def sine_wave(freq: float, n: int, sample_rate: int = 16000, device: torch.device = torch.device('cpu')) -> torch.Tensor:
        """Generate sine wave as torch tensor"""
        t = torch.linspace(0, n / sample_rate, n, device=device)
        return torch.sin(2 * torch.pi * freq * t)

class VADMixedDataset(Dataset):
    def __init__(
        self,
        speech_datasets: List[Dataset],
        noise_datasets: List[Dataset],
        sample_rate: int = 16000,
        length_config: Optional[LengthConfig] = None,
        fixed_length: Optional[float] = None,  # in seconds
        speech_config: Optional[SpeechConfig] = None,
        speech_proportion: float = 0.3,  # proportion of speech in all samples combined
        min_speech_length: float = 0.2,  # minimum speech segment length in seconds
        synthetic_noise_prob: float = 0.3,  # probability of using synthetic noise
        mode: Literal['train', 'val', 'test'] = 'val',
        deterministic: bool = False,
        seed: int = 42,
    ):
        """
        Initialize the enhanced VAD mixed dataset.
        
        Args:
            speech_datasets: List of datasets containing speech samples
            noise_datasets: List of datasets containing non-speech samples
            sample_rate: Audio sample rate
            length_config: Configuration for random length range
            fixed_length: Fixed length for all outputs (in seconds)
            speech_config: Configuration for speech presence and length
            speech_proportion: Target proportion of speech across all samples
            min_speech_length: Minimum length for individual speech segments
            synthetic_noise_prob: Probability of using synthetic noise vs dataset noise
            mode: Dataset mode ('train', 'val', or 'test')
            deterministic: If True, will generate same samples each time
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
        self.speech_proportion = speech_proportion
        self.min_speech_length = int(min_speech_length * sample_rate)
        self.synthetic_noise_prob = synthetic_noise_prob
        self.mode = mode
        self.deterministic = deterministic
        self.seed = seed
        
        # Initialize speech configuration
        self.speech_config = speech_config or SpeechConfig(
            min_length=2.0,
            max_length=5.0,
            presence_prob=0.7
        )
        
        # Initialize noise generator
        self.noise_gen = NoiseGenerator()
        
        # Store dataset lengths and calculate split sizes
        total_samples = sum(len(dataset) for dataset in speech_datasets)
        if mode == 'train':
            self.length = int(0.8 * total_samples)
        elif mode == 'val':
            self.length = int(0.1 * total_samples)
        else:  # test
            self.length = int(0.1 * total_samples)

    def _random_choice(self, 
                      options: List[any], 
                      weights: List[float], 
                      random_state: Optional[np.random.RandomState] = None) -> any:
        """Make a random choice with optional deterministic behavior"""
        if random_state is not None:
            return random_state.choice(options, p=weights)
        return random.choices(options, weights=weights)[0]

    def _random_int(self, 
                    low: int, 
                    high: int, 
                    random_state: Optional[np.random.RandomState] = None) -> int:
        """Get random integer with optional deterministic behavior"""
        if low >= high:
            return low
        if random_state is not None:
            return random_state.randint(low, high)
        return random.randint(low, high - 1)

    def _get_random_generator(self, idx: int) -> Union[np.random.RandomState, None]:
        """Get random generator based on mode and index"""
        if self.deterministic:
            return np.random.RandomState(self.seed + idx)
        return None

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
        dataset_idx = self._random_int(0, len(datasets), random_state)
        sample_idx = self._random_int(0, len(datasets[dataset_idx]), random_state)
        return datasets[dataset_idx][sample_idx]['sample']

    def _generate_synthetic_noise(self, 
                                length: int,
                                random_state: Optional[np.random.RandomState] = None) -> torch.Tensor:
        """Generate synthetic noise of specified type"""
        noise_type = self._random_choice(['white', 'pink', 'sine'], [0.4, 0.4, 0.2], random_state)
        
        if noise_type == 'white':
            return self.noise_gen.white_noise(length)
        elif noise_type == 'pink':
            return self.noise_gen.pink_noise(length)
        else:
            freq = self._random_int(50, 2000, random_state)
            return self.noise_gen.sine_wave(freq, length, self.sample_rate)

    def _get_noise_segment(self, 
                          length: int,
                          random_state: Optional[np.random.RandomState] = None) -> torch.Tensor:
        """Get noise segment either from dataset or generate synthetically"""
        use_synthetic = (random_state.random() if random_state else random.random()) < self.synthetic_noise_prob
        
        if use_synthetic:
            return self._generate_synthetic_noise(length, random_state)
        else:
            dataset_idx = self._random_int(0, len(self.noise_datasets), random_state)
            sample_idx = self._random_int(0, len(self.noise_datasets[dataset_idx]), random_state)
            noise = self.noise_datasets[dataset_idx][sample_idx]['sample']
            return self._get_random_slice(noise, length, random_state)

    def _should_include_speech(self, random_state: Optional[np.random.RandomState] = None) -> bool:
        """Determine if current sample should include speech"""
        if random_state is not None:
            return random_state.random() < self.speech_config.presence_prob
        return random.random() < self.speech_config.presence_prob

    def _get_speech_length(self, 
                          max_length: int,
                          random_state: Optional[np.random.RandomState] = None) -> int:
        """Get target speech length in samples"""
        min_samples = int(self.speech_config.min_length * self.sample_rate)
        max_samples = int(min(self.speech_config.max_length * self.sample_rate, max_length))
        
        if random_state is not None:
            return random_state.randint(min_samples, max_samples + 1)
        return random.randint(min_samples, max_samples)

    def _maybe_normalize_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """Normalize audio to have max amplitude of 1"""
        max_val = torch.abs(audio).max()
        if max_val > 0:
            return audio / max_val
        return audio

    def _generate_mixed_segment(self, 
                              target_length: int,
                              random_state: Optional[np.random.RandomState] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate a mixed audio segment with variable speech presence"""
        output = torch.zeros(target_length)
        mask = torch.zeros(target_length)
        
        # Decide if this sample should include speech
        if self._should_include_speech(random_state):
            # Get target speech length
            speech_length = self._get_speech_length(target_length, random_state)
            
            # Add speech segments until reaching target length
            remaining_speech = speech_length
            while remaining_speech >= self.min_speech_length:
                # Ensure valid range for segment length
                max_segment = min(remaining_speech, target_length)
                if max_segment < self.min_speech_length:
                    break
                    
                segment_length = self._random_int(
                    self.min_speech_length,
                    max_segment + 1,
                    random_state
                )
                
                # Get random position for speech segment
                available_positions = torch.where(mask == 0)[0]
                if len(available_positions) < segment_length:
                    break
                    
                start_idx = self._random_int(0, len(available_positions) - segment_length + 1, random_state)
                positions = available_positions[start_idx:start_idx + segment_length]
                
                # Add speech segment
                speech = self._get_audio_from_dataset(is_speech=True, random_state=random_state)
                speech_segment = self._get_random_slice(speech, segment_length, random_state)
                output[positions] = speech_segment
                mask[positions] = 1
                
                remaining_speech -= segment_length
        
        # Fill non-speech positions with noise
        non_speech_positions = torch.where(mask == 0)[0]
        if len(non_speech_positions) > 0:
            noise = self._get_noise_segment(len(non_speech_positions), random_state)
            output[non_speech_positions] = noise
        
        # Normalize final audio
        output = self._maybe_normalize_audio(output)
        
        return output, mask

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Adjust index based on mode
        if self.mode == 'val':
            idx += int(0.8 * self.length)
        elif self.mode == 'test':
            idx += int(0.9 * self.length)
        
        random_state = self._get_random_generator(idx)
        target_length = self._get_target_length(random_state)
        mixed_audio, mask = self._generate_mixed_segment(target_length, random_state)
        
        return {
            'sample': mixed_audio,
            'mask': mask,
            'mode': self.mode
        }
    

def get_datset():
    from src.datasets.librispeech import get_librispeech_example, LibriSpeechWrapper
    from src.datasets.urbansound import UrbanSoundDataset, read_arrf
    from src.datasets.musan import MusanMusicDataset

    speech_datasets = [
    LibriSpeechWrapper(get_librispeech_example(), remove_silence_on_edges=True)
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
                                        speech_proportion=0.3,
                                        synthetic_noise_prob=0.2,
                                        deterministic=True
                                        )
    return vad_dataset_fixed


if __name__ == '__main__':
    vad_dataset_fixed = get_datset()
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