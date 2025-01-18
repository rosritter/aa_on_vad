import torch
from torch.utils.data import Dataset
import torchaudio
from pathlib import Path
import numpy as np
import soundfile as sr
from typing import Union 
from src.utils.data_utils import read_torch


class MusanMusicDataset(Dataset):
    def __init__(self, root_dir, segment_length:Union[tuple[int,int],None]=(16000, 16000), target_sample_rate=16000, random_start_point:bool=True):
        """
        Args:
            root_dir (str): Directory with all the music wav files
            segment_length (int): Desired length of audio segments in samples
            sample_rate (int): Target sample rate
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.random_start_point = random_start_point
        self.root_dir = Path(root_dir)
        self.segment_length = segment_length
        self.target_sample_rate = target_sample_rate

        # Get all wav files recursively
        self.file_paths = list(self.root_dir.rglob("*.wav"))
        if not self.file_paths:
            raise RuntimeError(f"No .wav files found in {root_dir}")
        self.source_sample_rate = sr.read(self.file_paths[0])[1]
        self.resampler = torchaudio.transforms.Resample(self.source_sample_rate, self.target_sample_rate)
        # Cache file lengths to avoid repeated loading
        self.file_lengths = {}
        file_paths = []
        for path in self.file_paths:
            info = torchaudio.info(path)
            length = int(info.num_frames / self.source_sample_rate * self.target_sample_rate)
            if not self.segment_length or length >= self.segment_length[1]: # god please i`m trying to write oneline in python
                file_paths.append(path)
                self.file_lengths[path] = length
        self.file_paths = file_paths

    
    def get_random_segment(self, idx, target_length):
        """
        Args:
            target_length (int): Desired length in samples
            
        Returns:
            torch.Tensor: Audio segment of specified length
        """
        # Shuffle file paths to randomize search
    
        file_path = self.file_paths[idx]
        file_length = self.file_lengths[file_path]
        waveform = read_torch(file_path)
        if sr != self.target_sample_rate:
            waveform = self.resampler(waveform)
        # Get random start point
        if self.random_start_point:
            max_start = file_length - target_length
            start_idx = np.random.randint(low=0,high=max_start, size=1)[-1]
        else:
            start_idx:int = 0    
        
        # Extract segment
        segment = waveform[start_idx:start_idx + target_length]
        return segment
        
        
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        """
        Get a random segment of the specified length from the dataset.
        
        Args:
            idx (int): Index (used for compatibility with DataLoader)
            
        Returns:
            torch.Tensor: Audio segment of length self.segment_length
        """
        if self.segment_length:
            segment_length = np.random.randint(low=self.segment_length[0],high=self.segment_length[1]+1, size=1)[-1]
            return self.get_random_segment(idx, segment_length)
        else:
            return read_torch(self.file_paths[idx])[0]

    def check_length(self, idx):
        if self.file_lengths[self.file_paths[idx]] < self.segment_length[1]:
            return False
        return True

if __name__ == '__main__':
    dataset = MusanMusicDataset(
    root_dir='datasets/musan/music',
    target_sample_rate=16000,
    segment_length=None#(16000, 32000)
    )
    print(dataset[1].shape)