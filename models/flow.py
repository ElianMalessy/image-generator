import torch
import torch.nn as nn
from models import device
from torchdiffeq import odeint

class Flow(nn.Module):
    def __init__(self, dim=64, time_dim=16):
        super().__init__()

        self.dim = dim
        self.time_dim = time_dim
        self.mlp = nn.Sequential(
            nn.Linear(dim + time_dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        t_feats = fourier_features(t, self.time_dim)
        return self.mlp(torch.cat([x, t_feats], dim=-1))


    def transform(self, x: torch.Tensor, steps=15):
        t = torch.linspace(0, 1, steps=steps).to(device)

        def dxdt(ti, xi) -> torch.Tensor:
            ti = ti.unsqueeze(-1)
            return self.forward(xi, ti)

        x_traj = odeint(dxdt, x, t)
        return x_traj[-1]
            


def fourier_features(t: torch.Tensor, dim) -> torch.Tensor:
    half_dim = dim // 2
    freq_bands = 2 ** torch.arange(half_dim, device=device).float() * torch.pi
    args = t * freq_bands.unsqueeze(0)
    fourier_feats = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    return fourier_feats


