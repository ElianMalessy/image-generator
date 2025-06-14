import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, latent_dim=(4, 16, 16)):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, 128, 3, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(),
        )

        C, _, _ = latent_dim
        self.mu = nn.Conv2d(128, C, kernel_size=1)
        self.log_var = nn.Conv2d(128, C, kernel_size=1)

        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.GroupNorm(8, out_channels),
                nn.SiLU(),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.GroupNorm(8, out_channels),
                nn.SiLU(),
            )
            

        self.decoder = nn.Sequential(
            nn.Conv2d(C, 128, kernel_size=1),
            nn.SiLU(),

            conv_block(128, 128),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            conv_block(128, 64),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            conv_block(64, 32),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Tanh(),
        )

    def encode(self, x):
        x = self.encoder(x)
        mu = self.mu(x)
        log_var = self.log_var(x)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        reconstruction = self.decode(z)
        return reconstruction, mu, log_var
