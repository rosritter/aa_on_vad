from torch import nn
import torch

class NoiseGenerator(nn.Module):
    def __init__(self, input_channels=1, d_model=256, nhead=8, num_layers=4):
        super().__init__()
        
        # Initial projection
        self.input_proj = nn.Sequential(
            nn.Conv1d(input_channels, d_model, kernel_size=15, stride=1, padding=7),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=15, stride=1, padding=7),
        )
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Conv1d(d_model, d_model//2, kernel_size=15, stride=1, padding=7),
            nn.GELU(),
            nn.Conv1d(d_model//2, input_channels, kernel_size=15, stride=1, padding=7),
            nn.Tanh()
        )

    def forward(self, x):
        # Input shape: [batch, time]
        x = x.unsqueeze(1)  # [batch, channels, time]
        
        # Initial projection
        x = self.input_proj(x)  # [batch, d_model, time]
        x = x.transpose(1, 2)  # [batch, time, d_model]
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer
        x = self.transformer(x)
        
        # Output projection
        x = x.transpose(1, 2)  # [batch, d_model, time]
        x = self.output_proj(x)  # [batch, channels, time]
        
        return x.squeeze(1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=96000):
        super().__init__()
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]