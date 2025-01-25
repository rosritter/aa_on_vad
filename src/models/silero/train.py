import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.models.silero.loss import AdversarialLoss
from src.datasets.mixed_vad_datset import get_datset
from src.models.adversarial_models.CNN1d_based import NoiseGenerator
from src.models.silero.evaluate_model import validate_silero_vad, print_metrics


def train_step(batch, noise_producer, vad_model, criterion, optimizer, device):
    # Unpack batch with correct keys
    wavs = batch['sample'].to(device)
    masks = batch['mask'].to(device)

    # Model constants
    num_samples = 512  # main window size
    context_size = 64  # context size for 16khz mode
    state = torch.zeros((2, wavs.shape[0], 128)).to(device) # fixed for silero
    # Pad input for context
    x = torch.nn.functional.pad(wavs, (context_size, wavs.shape[-1] % num_samples))
    noise = noise_producer(x)
    x = x + noise
    outs = []
    noises = []

    for i in range(context_size, x.shape[1], num_samples):
        # Get current window with context
        input_window = x[:, i - context_size:i + num_samples]

        # Process through VAD
        out = vad_model._model.stft(input_window)
        out = vad_model._model.encoder(out)
        out, state = vad_model._model.decoder(out, state)
        outs.append(out)

    # Combine and interpolate to match mask shape
    vad_output = torch.cat(outs, dim=2).squeeze(1)
    vad_output = torch.nn.functional.interpolate(
        vad_output.unsqueeze(1),
        size=masks.shape[1],
        mode='linear'
    ).squeeze(1)

    # Calculate loss using masks
    loss = criterion(vad_output, noise, masks)

    # Backprop
    if optimizer:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item()

def eval_step(batch, noise_producer, vad_model, criterion, device):
    with torch.no_grad():
        return train_step(batch, noise_producer, vad_model, criterion, None, device)

def train_loop(dataloader, noise_producer, vad_model, criterion, optimizer, device, epoch):
    noise_producer.train()
    vad_model.eval()  # Keep VAD frozen
    
    total_loss = 0
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch in pbar:
        loss = train_step(batch, noise_producer, vad_model, criterion, optimizer, device)
        total_loss += loss

        # Update progress bar postfix without newlines
        pbar.set_postfix({'loss': f"{loss / batch['sample'].shape[0]:.4f}"})
        pbar.refresh()  # Force immediate update
    return total_loss / len(dataloader.dataset)

def eval_loop(dataloader, noise_producer, vad_model, criterion, device):
    noise_producer.eval()
    vad_model.eval()
    
    total_loss = 0
    for batch in tqdm(dataloader, desc='Evaluation'):
        loss = eval_step(batch, noise_producer, vad_model, criterion, device)
        total_loss += loss
        
    return total_loss / len(dataloader)

def train_noise_producer(noise_producer, vad_model, train_dataset, val_dataset, 
                        criterion=AdversarialLoss(), epochs=10, batch_size=32, lr=1e-4, device='cuda'):
    # Freeze VAD model
    for param in vad_model.parameters():
        param.requires_grad = False
    print(f'Noise producer model has {sum(p.numel() for p in noise_producer.parameters())} parametes')
    optimizer = torch.optim.Adam(noise_producer.parameters(), lr=lr)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    # metrics = validate_silero_vad(
    #     model,
    #     val_dataset,
    #     batch_size,
    #     # noise_producer,
    # )
    # print_metrics(metrics)
    for epoch in range(epochs):
        train_loss = train_loop(train_loader, noise_producer, vad_model, 
                              criterion, optimizer, device, epoch)
        print(f'Epoch {epoch}: Train Loss = {train_loss:.4f}')
        val_loss = eval_loop(val_loader, noise_producer, vad_model,
                               criterion, device)
        print(f'Epoch {epoch}: Eval Loss = {val_loss:.4f}')



if __name__ == '__main__':
    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
    print(model)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    amodel = NoiseGenerator().to(device)
    # Create dataset
    print('Loading dataset...')
    train_ds = get_datset(mode='train', erase_silence=True)
    eval_ds = get_datset(mode='eval', erase_silence=True)
    print('Dataset loaded!')

    train_noise_producer(
        amodel,
        model,
        train_ds,
        eval_ds,
    )
