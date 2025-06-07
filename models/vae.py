import torch
import torch.nn as nn

import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        self.embeddings = nn.Parameter(torch.randn(num_embeddings, embedding_dim))

    def forward(self, z_e):
        # Input shape: [B, D, H, W] → [BHW, D]
        B, D, H, W = z_e.shape
        z_flattened = z_e.permute(0, 2, 3, 1).contiguous().view(-1, D)

        # Compute L2 distance to each codebook vector
        distances = (
            torch.sum(z_flattened ** 2, dim=1, keepdim=True)
            - 2 * z_flattened @ self.embeddings.t()
            + torch.sum(self.embeddings ** 2, dim=1)
        )  # [BHW, K]

        encoding_indices = torch.argmin(distances, dim=1)
        z_q = self.embeddings[encoding_indices].view(B, H, W, D).permute(0, 3, 1, 2)

        # Codebook + commitment loss
        loss = F.mse_loss(z_q.detach(), z_e) + self.commitment_cost * F.mse_loss(z_q, z_e.detach())

        # Straight-through gradient estimator
        z_q = z_e + (z_q - z_e).detach()

        return z_q, loss

class VAE(nn.Module):
    def __init__(self, embedding_dim=64, num_embeddings=512, commitment_cost=0.25):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.Conv2d(128, embedding_dim, 4, 2, 1),  # → [B, D, H, W]
        )

        self.vq = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embedding_dim, 128, 4, 2, 1),
            nn.SiLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.SiLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Tanh()
        )

    def encode(self, x):
        z_e = self.encoder(x)
        z_q, vq_loss = self.vq(z_e) 

        return z_q, vq_loss

    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        z_q, vq_loss = self.encode(x)
        reconstruction = self.decode(z_q)
        return reconstruction, vq_loss

