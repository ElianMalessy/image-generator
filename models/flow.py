import torch 
import torch.nn as nn
from models import device
from torchdiffeq import odeint

class FiLM(nn.Module):
    def __init__(self, feature_dim, label_dim):
        super().__init__()
        self.gamma = nn.Linear(label_dim, feature_dim)
        self.beta  = nn.Linear(label_dim, feature_dim)
    def forward(self, h, y_embed):
        # h: (B, feature_dim), y_embed: (B, label_dim)
        g = self.gamma(y_embed)
        b = self.beta (y_embed)
        return g * h + b

class Flow(nn.Module):
    def __init__(self, dim=64, time_dim=16, label_dim=32, n_heads=4):
        super().__init__()
        self.dim       = dim
        self.time_dim  = time_dim
        self.label_dim = label_dim

        # → dim so cross-attn and FiLM both work in the same space
        self.label_proj = nn.Sequential(
            nn.Linear(40, label_dim),
            nn.ReLU(),
            nn.Linear(label_dim, dim),
        )

        # attention layers
        self.cross_attn1 = nn.MultiheadAttention(embed_dim=dim, num_heads=n_heads, batch_first=True)
        self.cross_attn2 = nn.MultiheadAttention(embed_dim=dim, num_heads=n_heads, batch_first=True)

        # FiLM layers
        self.film1 = FiLM(dim, dim)
        self.film2 = FiLM(dim, dim)

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
        B = x.size(0)

        # 1) time features
        t_feats = fourier_features(t, self.time_dim)   # (B, time_dim)
        h = torch.cat([x, t_feats], dim=-1)            # (B, dim+time_dim)
        h = self.block1(h)

        # 2) project labels once
        y_embed = self.label_proj(y)                   # (B, dim)

        # ── First Attention + FiLM ───────────────────────────
        # stack into length-2 sequence
        seq = torch.stack([h, y_embed], dim=1)         # (B, 2, dim)
        attn_out, _ = self.cross_attn1(seq, seq, seq)  # (B, 2, dim)
        h = attn_out[:,0]                              # (B, dim)
        # FiLM conditioning
        h = self.film1(h, y_embed)

        # ── First Residual Block ────────────────────────────
        res = self.block2(h)
        h = h + res

        # ── Second Attention + FiLM ──────────────────────────
        seq = torch.stack([h, y_embed], dim=1)
        attn_out, _ = self.cross_attn2(seq, seq, seq)
        h = attn_out[:,0]
        h = self.film2(h, y_embed)

        # ── Second Residual Block ────────────────────────────
        res = self.block3(h)
        h = h + res

        # ── Final projection ─────────────────────────────────
        v_hat = self.out(h)                             # (B, dim)

        return v_hat



    def transform(self, x: torch.Tensor, y: torch.Tensor, steps=20):
        def dxdt(ti, xi) -> torch.Tensor:
            ti = ti.unsqueeze(-1)
            v = self.forward(xi, ti, y)
            # alpha = (1 - ti).pow(2)
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
