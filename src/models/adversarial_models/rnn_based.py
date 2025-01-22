import torch
import torch.nn as nn
import torchaudio

class RNNNoiseGenerator(nn.Module):
    def __init__(self, input_size=80, hidden_size=128):
        super(RNNNoiseGenerator, self).__init__()
        
        self.melspec = torchaudio.transforms.MelSpectrogram(
            n_mels=input_size,
            n_fft=400,
            hop_length=160
        )
        
        self.inverse = torchaudio.transforms.GriffinLim(
            n_fft=400,
            hop_length=160,
            n_iter=32
        )
        
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )
        
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size * 2, input_size),
            nn.Tanh()
        )
    
    def forward(self, x):
        # Convert to mel spectrogram
        mel = self.melspec(x)
        mel = mel.transpose(1, 2)  # [batch, time, freq]
        
        # Generate noise pattern
        rnn_out, _ = self.rnn(mel)
        noise_mel = self.output_layer(rnn_out)
        
        # Convert back to time domain
        noise_mel = noise_mel.transpose(1, 2)  # [batch, freq, time]
        noise = self.inverse(noise_mel)
        
        return noise

# Training loop example
def train_step(model, vad_model, speech, optimizer):
    noise = model(speech)
    noisy_speech = speech + noise
    
    # VAD prediction
    pred = vad_model(noisy_speech)
    
    # Losses
    adv_loss = nn.BCELoss()(pred, torch.zeros_like(pred))
    noise_loss = 0.1 * torch.mean(torch.abs(noise))
    loss = adv_loss + noise_loss
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item(), noise