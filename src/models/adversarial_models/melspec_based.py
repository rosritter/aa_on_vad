import torch
import torch.nn as nn
import torchaudio

class SimpleMelNoiseGenerator(nn.Module):
    def __init__(self, sample_rate=16000, n_mels=80):
        super(SimpleMelNoiseGenerator, self).__init__()
        
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=400,
            hop_length=160
        )
        
        self.inverse = torchaudio.transforms.GriffinLim(
            n_fft=400,
            hop_length=160,
            n_iter=32
        )
        
        self.freq_generator = nn.Sequential(
            # Downsampling
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            
            # Process
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            
            # Upsampling
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),
            nn.Tanh()
        )
    
    def forward(self, x):
        mel = self.melspec(x).unsqueeze(1)
        noise_mel = self.freq_generator(mel)
        noise = self.inverse(noise_mel.squeeze(1))
        return noise

# Loss function
def noise_loss(original, noise, vad_model, target=0, alpha=0.1):
    noisy = original + noise
    pred = vad_model(noisy)
    adv_loss = nn.BCELoss()(pred, torch.full_like(pred, target))
    magnitude_loss = torch.mean(torch.abs(noise))
    return adv_loss + alpha * magnitude_loss