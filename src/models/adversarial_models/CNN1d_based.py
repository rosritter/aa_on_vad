import torch
import torch.nn as nn

class NoiseGenerator(nn.Module):
    def __init__(self, input_channels=1):
        super(NoiseGenerator, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            self._make_layer(input_channels, 32, kernel_size=7),
            self._make_layer(32, 64, kernel_size=5),
            self._make_layer(64, 128, kernel_size=3),
            self._make_layer(128, 256, kernel_size=3),
        )
        
        # Residual blocks for temporal dependencies
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(256) for _ in range(3)
        ])
        
        # Decoder
        self.decoder = nn.Sequential(
            self._make_layer(256, 128, kernel_size=3, upsample=True),
            self._make_layer(128, 64, kernel_size=3, upsample=True),
            self._make_layer(64, 32, kernel_size=5, upsample=True),
            nn.Conv1d(32, 1, kernel_size=7, padding=3),
            nn.Tanh()  # Bound noise values
        )
        
    def _make_layer(self, in_channels, out_channels, kernel_size, upsample=False):
        layers = []
        if upsample:
            layers.append(nn.Upsample(scale_factor=2))
            
        layers.extend([
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
            nn.InstanceNorm1d(out_channels),
            nn.LeakyReLU(0.2)
        ])
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Encode
        x = self.encoder(x)
        
        # Apply residual blocks
        for block in self.residual_blocks:
            x = block(x)
            
        # Decode
        noise = self.decoder(x)
        return noise

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=1),
            nn.InstanceNorm1d(channels),
            nn.LeakyReLU(0.2),
            nn.Conv1d(channels, channels, 3, padding=1),
            nn.InstanceNorm1d(channels)
        )
        
    def forward(self, x):
        return x + self.block(x)

class AdversarialLoss(nn.Module):
    def __init__(self, vad_model, alpha=0.5):
        super(AdversarialLoss, self).__init__()
        self.vad_model = vad_model
        self.alpha = alpha
        self.bce = nn.BCELoss(reduction='none')
        
    def forward(self, original_speech, generated_noise, target_labels):
        # Combine speech and noise
        noisy_speech = original_speech + generated_noise
        
        # Get VAD predictions
        vad_predictions = self.vad_model(noisy_speech)
        
        # Adversarial loss (fool VAD model)
        adv_loss = self.bce(vad_predictions, target_labels)
        
        # Noise magnitude loss (keep noise small)
        noise_loss = torch.mean(torch.abs(generated_noise))
        
        # Combined loss
        total_loss = adv_loss - self.alpha * noise_loss
        return total_loss