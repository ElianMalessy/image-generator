import torch
import torch.nn as nn
from models.resnet import ResNetDecoder, Encoder

class VAE(nn.Module):
    def __init__(self, latent_dim=(16, 16, 16)):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim=latent_dim)
        self.decoder = ResNetDecoder(latent_dim=latent_dim)

    def encode(self, x):
        mu, log_var = self.encoder(x)
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

