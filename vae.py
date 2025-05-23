import torch
import torch.nn as nn
import torch.nn.utils as utils

class VAE(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.ReLU(),
            nn.Flatten()
        )

        self.mu = nn.Linear(512 * 2 * 2, latent_dim)
        self.log_var = nn.Linear(512 * 2 * 2, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512 * 2 * 2),
            nn.Unflatten(1, (512, 2, 2)),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Sigmoid(),  # Keep output in [0,1]
        )

    def forward(self, x):
        downsampled_x = self.encoder(x)
        mu = self.mu(downsampled_x)
        log_var = self.log_var(downsampled_x)

        std_dev = torch.exp(0.5 * log_var)
        noise = torch.randn_like(std_dev)
        
        reparametrization = mu + std_dev * noise

        return self.decoder(reparametrization), mu, log_var

        
        # return reparametrization

