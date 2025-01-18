import torchaudio
from torch.utils.data import Dataset
import torchaudio.transforms as T

from src.utils.audio_utils import vad_forward, remove_silence



class LibriSpeechWrapper(Dataset):
    def __init__(self, librispeech_dataset, sample_rate=16000, erase_silence=False, apply_vad=False):
        """
        Initialize the wrapper with a torchaudio.datasets.LIBRISPEECH dataset.

        Args:
            librispeech_dataset: Instance of torchaudio.datasets.LIBRISPEECH
            sample_rate: Sampling rate for VAD (default: 16000)
        """
        self.dataset = librispeech_dataset
        self.apply_vad = apply_vad
        self.erase_silence = erase_silence
        self.sample_rate = self.dataset[0][1]
        if apply_vad:
            self.vad = T.Vad(sample_rate=sample_rate)  # Voice Activity Detection transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset, remove silence, and return it in the desired dictionary format.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            A dictionary with keys:
            - 'sample': Audio waveform with silence removed
            - 'sample_rate': Sampling rate of the audio
            - 'Transcript': Transcript of the audio
            - 'speaker_id': Speaker ID
            - 'Chapter ID': Chapter ID
            - 'Utterance ID': Utterance ID
        """
        # Fetch original data
        waveform, sample_rate, transcript, speaker_id, chapter_id, utterance_id = self.dataset[idx]
        
        

        
        
        if self.apply_vad:
            waveform = vad_forward(waveform=waveform, vad=self.vad, sample_rate=self.sample_rate, idx=idx)
        
        if self.erase_silence:
            waveform = remove_silence(waveform, 16000, step_duration=0.02, energy_threshold=0.01)
        if waveform.ndim > 1:
            waveform = waveform[0,:]
        return {
            'sample': waveform,
            'sample_rate': sample_rate,
            'Transcript': transcript,
            'speaker_id': speaker_id,
            'Chapter ID': chapter_id,
            'Utterance ID': utterance_id,
        }


'''
https://pytorch.org/audio/main/generated/torchaudio.datasets.LIBRISPEECH.html#torchaudio.datasets.LIBRISPEECH
'''
def get_librispeech_example():
    ds = torchaudio.datasets.LIBRISPEECH(root='datasets/torchlibri',
                                     url ='test-clean',
                                     download=False)
    return ds

if __name__ == '__main__':
    print(LibriSpeechWrapper(get_librispeech_example())[0])
    