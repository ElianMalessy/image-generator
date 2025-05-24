import torch
import torch.nn as nn

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
            nn.Flatten()
        )

        # self.attention = MultiHeadSelfAttention(embed_dim=512, num_heads=8)
        # self.avgpool = nn.AdaptiveAvgPool2d(1)
        # self.mu_conv = nn.Conv2d(512, latent_dim, 1)
        # self.log_var_conv = nn.Conv2d(512, latent_dim, 1)

        self.mu = nn.Linear(512 * 2 * 2, latent_dim)
        self.log_var = nn.Linear(512 * 2 * 2, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512 * 2 * 2),
            nn.Unflatten(1, (512, 2, 2)),
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
        # x = self.attention(x)
        # x = self.avgpool(x)
        # mu = self.mu_conv(x).squeeze(-1).squeeze(-1)
        # log_var = self.log_var_conv(x).squeeze(-1).squeeze(-1)
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

