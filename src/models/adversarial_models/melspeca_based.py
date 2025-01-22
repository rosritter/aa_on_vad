import torch
import torch.nn as nn
import torchaudio

class MelSpecNoiseGenerator(nn.Module):
    def __init__(self, sample_rate=16000, n_mels=80):
        super(MelSpecNoiseGenerator, self).__init__()
        
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=400,
            hop_length=160
        )
        
        self.inverse_melspec = GriffinLim(
            n_fft=400,
            hop_length=160,
            n_iter=32
        )
        
        # Frequency-domain noise generator
        self.freq_generator = nn.Sequential(
            # Downsampling blocks
            self._conv_block(1, 32, kernel_size=(3, 3), stride=(2, 2)),
            self._conv_block(32, 64, kernel_size=(3, 3), stride=(2, 2)),
            self._conv_block(64, 128, kernel_size=(3, 3), stride=(2, 2)),
            
            # Transformer blocks for frequency-time attention
            TransformerBlock(128, heads=4),
            TransformerBlock(128, heads=4),
            
            # Upsampling blocks
            self._conv_block(128, 64, kernel_size=(3, 3), upsample=True),
            self._conv_block(64, 32, kernel_size=(3, 3), upsample=True),
            self._conv_block(32, 1, kernel_size=(3, 3), upsample=True),
            nn.Tanh()
        )
        
    def _conv_block(self, in_channels, out_channels, kernel_size, stride=(1, 1), upsample=False):
        layers = []
        if upsample:
            layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size[0]//2))
        else:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size[0]//2))
        
        layers.extend([
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        ])
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Convert to mel spectrogram
        mel = self.melspec(x)
        mel = mel.unsqueeze(1)  # Add channel dimension
        
        # Generate noise in mel-frequency domain
        noise_mel = self.freq_generator(mel)
        
        # Convert back to time domain
        noise = self.inverse_melspec(noise_mel.squeeze(1))
        return noise

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(dim, heads)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        
    def forward(self, x):
        # Reshape for attention
        b, c, h, w = x.shape
        x = x.flatten(2).permute(2, 0, 1)  # (h*w, batch, channels)
        
        # Self-attention
        attended = self.attention(x, x, x)[0]
        x = self.norm1(x + attended)
        
        # FFN
        x = self.norm2(x + self.ffn(x))
        
        # Reshape back
        x = x.permute(1, 2, 0).view(b, c, h, w)
        return x

class GriffinLim(nn.Module):
    def __init__(self, n_fft, hop_length, n_iter):
        super(GriffinLim, self).__init__()
        self.transform = torchaudio.transforms.GriffinLim(
            n_fft=n_fft,
            hop_length=hop_length,
            n_iter=n_iter
        )
    
    def forward(self, spec):
        return self.transform(spec)