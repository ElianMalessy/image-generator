import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block
        self.alpha = nn.Parameter(torch.tensor(0.01))

    def forward(self, x):
        return x + self.alpha * self.block(x)

class GlobalPositionalEncoding(nn.Module):
    def __init__(self, channels, dim):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, channels, dim, dim))

    def forward(self, x):
        x = x + self.pe

        return x

class DepthwiseEncoding(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.dwconv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)

    def forward(self, x):
        x = x + self.dwconv(x)
        return x


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
    
    def forward(self, x, token=None):
        B, C, H, W = x.shape
        x_flat = x.view(B, C, H*W).permute(0, 2, 1)

        if token is not None:
            x_flat = torch.cat([token, x_flat], dim=1)

        attn_output, _ = self.mha(x_flat, x_flat, x_flat)

        if token is not None:
            x_out, updated_global_token = attn_output[:, 1:], attn_output[:, :1]
            x_out = x_out.permute(0, 2, 1).view(B, C, H, W)
            return x_out, updated_global_token
        else:
            x_out = attn_output.permute(0, 2, 1).view(B, C, H, W)
            return x_out, None

class ConvAttention(nn.Module):
    def __init__(self, channels, num_heads):
        super().__init__()
        self.pe = DepthwiseEncoding(channels)

        self.attn = MultiHeadSelfAttention(embed_dim=channels, num_heads=num_heads)

    def forward(self, x):
        x = self.pe(x)
        x = self.attn(x)
        return x

class ViT(nn.Module):
    def __init__(self, channels, dim, heads, mlp_mult=4, dropout=0.1):
        super().__init__()
        self.pe = nn.Sequential(
            DepthwiseEncoding(channels),
            GlobalPositionalEncoding(channels, dim)
        )

        self.norm1 = nn.BatchNorm2d(channels)
        self.attn = MultiHeadSelfAttention(channels, heads)
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = nn.BatchNorm2d(channels)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels * mlp_mult, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(channels * mlp_mult, channels, kernel_size=1),
        )
        self.drop2 = nn.Dropout(dropout)
        self.res = ResidualBlock(nn.Sequential(self.norm2, self.mlp, self.drop2))

    def forward(self, x, global_token=None):
        x = self.pe(x)

        residual = x
        x_norm = self.norm1(x)
        attn_out, updated_global_token = self.attn(x_norm, global_token)
        x = residual + self.drop1(attn_out)
        x = self.res(x)

        if global_token is None:
            return x
        else:
            return x, updated_global_token


class VAE(nn.Module):
    def __init__(self, latent_dim=(4, 16, 16)):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
        )
        
        C, H, W = latent_dim

        # spatial latents
        self.mu = nn.Conv2d(128, C, kernel_size=1)
        self.log_var = nn.Conv2d(128, C, kernel_size=1)

        self.global_token = nn.Parameter(torch.zeros(1, 1, C))  # shared global token

        self.latent_attention = nn.ModuleList([
            ViT(C, H, heads=4, mlp_mult=4, dropout=0.1),
            ViT(C, H, heads=4, mlp_mult=4, dropout=0.1),
            ViT(C, H, heads=4, mlp_mult=4, dropout=0.1),
            ViT(C, H, heads=4, mlp_mult=4, dropout=0.1),
        ])


        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(C, 128, kernel_size=1),
            ViT(128, H, heads=4, mlp_mult=4, dropout=0.1),
            nn.SiLU(),

            nn.ConvTranspose2d(128, 64, 3, 2, 1, 1),
            ViT(64, H*2, heads=4, mlp_mult=4, dropout=0.1),
            nn.SiLU(),

            nn.ConvTranspose2d(64, 3, 3, 2, 1, 1),
            nn.Tanh(),  # Keep output in [-1,1]
        )

    def encode(self, x):
        x = self.encoder(x)

        mu = self.mu(x)
        log_var = self.log_var(x).clamp(-10.0, 10.0)

        return mu, log_var


    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        global_token = self.global_token
        global_token = global_token.expand(z.size(0), -1, -1)
    
        for block in self.latent_attention:
            z, global_token = block(z, global_token)

        return self.decoder(z)
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        reconstruction = self.decode(z)
        return reconstruction, mu, log_var

