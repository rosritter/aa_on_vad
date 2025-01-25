import torch
import torch.nn as nn
import torch
import torch.nn as nn


class NoiseGenerator(nn.Module):
    def __init__(self, input_channels=1):
        super(NoiseGenerator, self).__init__()

        # Encoder
        self.encoder = nn.ModuleList([
            self._make_encoder_layer(input_channels, 32, kernel_size=7, stride=2),
            self._make_encoder_layer(32, 64, kernel_size=5, stride=2),
            self._make_encoder_layer(64, 128, kernel_size=3, stride=2),
            self._make_encoder_layer(128, 256, kernel_size=3, stride=2)
        ])

        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(256) for _ in range(3)
        ])

        # Decoder - using ConvTranspose1d for precise upsampling
        self.decoder = nn.ModuleList([
            self._make_decoder_layer(256, 128, kernel_size=3, stride=2),
            self._make_decoder_layer(128, 64, kernel_size=3, stride=2),
            self._make_decoder_layer(64, 32, kernel_size=5, stride=2),
            self._make_decoder_layer(32, 32, kernel_size=5, stride=2)
        ])

        # Final convolution
        self.final_conv = nn.Sequential(
            nn.Conv1d(32, 1, kernel_size=7, padding=3),
            nn.Tanh()
        )

    def _make_encoder_layer(self, in_channels, out_channels, kernel_size, stride):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size,
                      stride=stride, padding=kernel_size // 2),
            nn.InstanceNorm1d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def _make_decoder_layer(self, in_channels, out_channels, kernel_size, stride):
        return nn.Sequential(
            nn.ConvTranspose1d(
                in_channels, out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                output_padding=stride - 1  # This ensures proper dimension matching
            ),
            nn.InstanceNorm1d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        # Track input size for dimension matching
        orig_size = x.size(-1)

        # Add channel dimension
        x = x.unsqueeze(1)

        # Store encoder outputs for potential skip connections
        encoder_outputs = []

        # Encode
        for encoder_layer in self.encoder:
            x = encoder_layer(x)
            encoder_outputs.append(x)

        # Apply residual blocks
        for block in self.residual_blocks:
            x = block(x)

        # Decode with precise dimension matching
        for i, decoder_layer in enumerate(self.decoder):
            x = decoder_layer(x)
            # Optional: Add skip connections
            x = x + encoder_outputs[-(i+1)] if x.shape == encoder_outputs[-(i+1)].shape else x

        # Ensure we're back to original temporal dimension
        assert x.size(-1) == orig_size, f"Output size {x.size(-1)} doesn't match input size {orig_size}"

        # Final convolution (preserves temporal dimension due to padding)
        x = self.final_conv(x)
        return x.squeeze()


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
