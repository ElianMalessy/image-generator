import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, x):
        return x + self.block(x)

class ConvolutionalPositionalEncoding(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.dwconv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)

    def forward(self, x):
        x = x + self.dwconv(x)
        return x


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads)
    
    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.view(B, C, H*W).permute(0, 2, 1)  # (B, H*W, C)
        attn_output, _ = self.mha(x_flat, x_flat, x_flat)  # (B, H*W, C)
        x = attn_output.permute(0, 2, 1).view(B, C, H, W)
        return x

class VAE(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.SiLU(),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.SiLU(),
        )

        self.encoder_attention = nn.Sequential (
            ResidualBlock(
                nn.Sequential(
                    ConvolutionalPositionalEncoding(channels=512),
                    MultiHeadSelfAttention(embed_dim=512, num_heads=8)
                )
            ),

            nn.BatchNorm2d(512),
            nn.SiLU()
        )

        # spatial latents
        self.mu = nn.Conv2d(512, latent_dim, kernel_size=1)
        self.log_var = nn.Conv2d(512, latent_dim, kernel_size=1)

        self.decoder_attention = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, kernel_size=1),
            ResidualBlock(
                nn.Sequential(
                    ConvolutionalPositionalEncoding(channels=512),
                    MultiHeadSelfAttention(embed_dim=512, num_heads=8)
                )
            ),

            nn.BatchNorm2d(512),
            nn.SiLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.SiLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.SiLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.SiLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Tanh(),  # Keep output in [-1,1]
        )

    def encode(self, x):
        x = self.encoder(x)
        x = self.encoder_attention(x)

        mu = self.mu(x)
        log_var = self.log_var(x)

        return mu, log_var


    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        z = self.decoder_attention(z)
        return self.decoder(z)
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        reconstruction = self.decode(z)
        return reconstruction, mu, log_var

