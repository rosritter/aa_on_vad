import torch
from torch.utils.data import Dataset
from scipy.io import arff
import pandas as pd
import torchaudio
import numpy as np
from scipy.io import arff


def read_arrf(path2file='datasets/UrbanSound/UrbanSound_TRAIN.arff') -> pd.DataFrame:
    train_data, meta = arff.loadarff(path2file)
    return pd.DataFrame(train_data)


# Convert features to tensor
class UrbanSoundDataset(Dataset):
    def __init__(self, dataframe, target_sample_rate:int=16000, source_sample_rate:int=44100):
        self.data = dataframe
        resampler = torchaudio.transforms.Resample(source_sample_rate, target_sample_rate)
        self.audios = [resampler(torch.tensor(audio.astype(np.float32))) for audio in dataframe.drop(columns=['target']).values]  # all columns except 'target'
        self.labels = dataframe['target'].values  # target column with labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio = torch.tensor(self.audios[idx], dtype=torch.float32)
        label = self.labels[idx].decode('utf-8')  # decode bytes to string
        return {'sample':audio, 'label': label}
    

if __name__ == '__main__':
    dataset = UrbanSoundDataset(read_arrf())
    dataset