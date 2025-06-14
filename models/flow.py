import torch 
import torch.nn as nn
from models import device
from torchdiffeq import odeint

class FiLM(nn.Module):
    def __init__(self, feature_dim, label_dim):
        super().__init__()
        self.gamma = nn.Linear(label_dim, feature_dim)
        self.beta  = nn.Linear(label_dim, feature_dim)
    def forward(self, h, y):
        g = self.gamma(y)
        b = self.beta (y)
        return g * h + b

class Flow(nn.Module):
    def __init__(self, dim=64, time_dim=16, label_dim=40):
        super().__init__()
        self.dim = dim
        self.time_dim = time_dim

        self.film1 = FiLM(dim, label_dim)
        self.film2 = FiLM(dim, label_dim)

        def make_block(in_dim, out_dim):
            return nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.ReLU(),
                nn.Linear(out_dim, out_dim),
            )

        self.block1 = make_block(dim + time_dim, dim)
        self.block2 = make_block(dim, dim)
        self.block3 = make_block(dim, dim)

        self.out = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor):
        t_feats = fourier_features(t, self.time_dim)
        h = torch.cat([x, t_feats], dim=-1)
        h = self.block1(h)

        h = self.film1(h, y)

        res = self.block2(h)
        h = h + res

        h = self.film2(h, y)

        res = self.block3(h)
        h = h + res

        v_hat = self.out(h)

        return v_hat



    def transform(self, x: torch.Tensor, y: torch.Tensor, steps=20):
        def dxdt(ti, xi) -> torch.Tensor:
            ti = ti.unsqueeze(-1)
            v = self.forward(xi, ti, y)
            return v

        t = torch.linspace(0, 1, steps=steps).to(device)
        x_traj = odeint(dxdt, x, t, method='rk4')
        return x_traj[-1]
            


def fourier_features(t: torch.Tensor, dim) -> torch.Tensor:
    half_dim = dim // 2
    freq_bands = 2 ** torch.arange(half_dim, device=device).float() * torch.pi
    args = t * freq_bands.unsqueeze(0)
    fourier_feats = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    return fourier_feats
