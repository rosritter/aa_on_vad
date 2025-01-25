import torch


class AdversarialLoss(torch.nn.Module):
    def __init__(self, alpha=0.1):
        super().__init__()
        self.alpha = alpha
        self.bce = torch.nn.BCELoss()

    def forward(self, vad_output, noise, mask):
        # Use original mask as target - maximizing BCE will push predictions
        # away from true labels
        adv_loss = -self.bce(vad_output, mask)

        # Minimize noise magnitude
        noise_loss = torch.mean(torch.abs(noise))
        return adv_loss + self.alpha * noise_loss