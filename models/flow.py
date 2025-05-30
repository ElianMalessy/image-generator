import torch
import torch.nn as nn
from models import device
from torchdiffeq import odeint

class Flow(nn.Module):
    def __init__(self, in_channels=64, time_dim=16):
        super().__init__()

        self.dim = in_channels
        self.time_dim = time_dim
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels + time_dim, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, in_channels, 3, padding=1)
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        B, C, H, W = x.shape
        t_feats = fourier_features(t.view(t.size(0), 1), self.time_dim)
        t_map = t_feats.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
        x_aug = torch.cat([x, t_map], dim=1)
        return self.mlp(x_aug)


    def transform(self, x: torch.Tensor, steps=20):
        t = torch.linspace(0, 1, steps=steps).to(device)

        def dxdt(ti, xi) -> torch.Tensor:
            ti = ti.unsqueeze(0).expand(xi.size(0))
            return self.forward(xi, ti)

        x_traj = odeint(dxdt, x, t)
        return x_traj[-1]
            


def fourier_features(t: torch.Tensor, dim) -> torch.Tensor:
    half_dim = dim // 2
    freq_bands = 2 ** torch.arange(half_dim, device=device).float() * torch.pi
    args = t * freq_bands.unsqueeze(0)
    fourier_feats = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    return fourier_feats


