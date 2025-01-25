from torch import nn
import torch


class NoiseGenerator(nn.Module):
    def __init__(self, input_channels=1, hidden_size=256, num_layers=2):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Initial CNN feature extraction
        self.cnn_frontend = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=31, stride=1, padding=15),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=31, stride=1, padding=15),
            nn.GELU()
        )
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True
        )
        
        # Output projection
        self.output = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.GELU(),
            nn.Linear(64, input_channels),
            nn.Tanh()
        )
        
    def init_hidden(self, batch_size, device):
        # Initialize hidden state and cell state
        # Shape: (num_layers * num_directions, batch, hidden_size)
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(device)
        return (h0, c0)

    def forward(self, x, hidden=None):
        batch_size = x.size(0)
        device = x.device
        
        # Initialize hidden state if not provided
        if hidden is None:
            hidden = self.init_hidden(batch_size, device)
        
        # Add channel dimension and extract features
        x = x.unsqueeze(1)
        x = self.cnn_frontend(x)  # [batch, features, time]
        
        # Prepare for LSTM
        x = x.transpose(1, 2)  # [batch, time, features]
        
        # LSTM processing with hidden state
        lstm_out, hidden_out = self.lstm(x, hidden)
        
        # Output projection
        x = self.output(lstm_out)
        
        return x.squeeze(-1), hidden_out

    def generate_noise(self, input_length, batch_size=1, device='cuda'):
        # Generate noise incrementally to handle very long sequences
        chunk_size = 16000  # Process 1 second at a time
        num_chunks = (input_length + chunk_size - 1) // chunk_size
        
        output_chunks = []
        hidden = None
        
        for i in range(num_chunks):
            # Calculate current chunk size
            current_chunk_size = min(chunk_size, input_length - i * chunk_size)
            
            # Generate random input for current chunk
            chunk_input = torch.randn(batch_size, current_chunk_size).to(device)
            
            # Process chunk
            chunk_output, hidden = self.forward(chunk_input, hidden)
            output_chunks.append(chunk_output)
        
        # Concatenate all chunks
        return torch.cat(output_chunks, dim=1)
    

    '''
    model = NoiseGenerator()
# For training
x = torch.randn(32, 16000)  # 1 second of audio at 16kHz
output, hidden = model(x)

# For generating long sequences
noise = model.generate_noise(input_length=160000)  # 10 seconds
    '''