import torch
import torch.nn as nn
import torch
import torch.nn as nn

import torch
import torch.nn as nn
import math

class NoiseGenerator(nn.Module):
    def __init__(self, input_channels=1):
        super(NoiseGenerator, self).__init__()
        
        self.encoder = nn.ModuleList([
            self._make_encoder_layer(input_channels, 32, kernel_size=1601, stride=2),
            self._make_encoder_layer(32, 64, kernel_size=801, stride=2),
            self._make_encoder_layer(64, 128, kernel_size=401, stride=2),
            self._make_encoder_layer(128, 256, kernel_size=201, stride=2)
        ])

        self.residual_blocks = nn.ModuleList([
            ResidualBlock(256) for _ in range(3)
        ])

        # Adjusted kernel sizes for exact matching
        self.decoder = nn.ModuleList([
            self._make_decoder_layer(256, 128, kernel_size=200, stride=2),
            self._make_decoder_layer(128, 64, kernel_size=400, stride=2),
            self._make_decoder_layer(64, 32, kernel_size=800, stride=2),
            self._make_decoder_layer(32, 32, kernel_size=1600, stride=2)
        ])

        self.final_conv = nn.Sequential(
            nn.Conv1d(32, 1, kernel_size=1601, padding=800),
            nn.Tanh()
        )

    def _make_encoder_layer(self, in_channels, out_channels, kernel_size, stride):
        padding = kernel_size // 2
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size,
                      stride=stride, padding=padding),
            nn.InstanceNorm1d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def _make_decoder_layer(self, in_channels, out_channels, kernel_size, stride):
        padding = (kernel_size - stride) // 2
        return nn.Sequential(
            nn.ConvTranspose1d(
                in_channels, out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=1
            ),
            nn.InstanceNorm1d(out_channels),
            nn.LeakyReLU(0.2)
        )
    
    def forward(self, x):
        orig_size = x.size(-1)
        x = x.unsqueeze(1)
        
        # Store encoder outputs for skip connections
        encoder_outputs = []
        for encoder_layer in self.encoder:
            x = encoder_layer(x)
            encoder_outputs.append(x)
            
        for block in self.residual_blocks:
            x = block(x)
            
        for decoder_layer in self.decoder:
            x = decoder_layer(x)
            # Trim any excess padding
            if x.size(-1) > orig_size:
                x = x[..., :orig_size]
                
        x = self.final_conv(x)
        return x.squeeze(1)




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
