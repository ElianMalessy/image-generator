import torch
import torch.nn as nn
import torch.nn.utils as utils

class EBM(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
    def forward(self, x):
        return self.model(x).squeeze()


def langevin_dynamics(x_init, ebm, num_steps=30, step_size=0.03, noise_scale=0.01):
    x = x_init.detach().clone()
    for _ in range(num_steps):
        x.requires_grad_(True)

        # energy and gradient
        energy = ebm(x).sum()
        grad = torch.autograd.grad(energy, x)[0]

        # do the update without tracking history
        with torch.no_grad():
            noise = torch.randn_like(x) * noise_scale
            x = x - 0.5 * step_size * grad + noise
            x = x.clamp(-1.0, 1.0)

    return x
