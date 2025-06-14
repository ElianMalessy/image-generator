import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=1)
        return emb

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.sin_embed = SinusoidalTimeEmbedding(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, t):
        return self.mlp(self.sin_embed(t))


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, t_emb_dim, groups=4):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(groups, out_ch),
            nn.SiLU(),
        )
        self.time_proj = nn.Linear(t_emb_dim, out_ch)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(groups, out_ch),
            nn.SiLU(),
        )
        self.skip = nn.Identity()
        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x, t_emb):
        h = self.conv1(x)
        t = self.time_proj(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = F.silu(h + t)
        h = self.conv2(h)
        return self.skip(x) + h

class DownConv(nn.Module):
    def __init__(self, in_ch, out_ch, groups=4):
        super().__init__()
        self.down_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(groups, out_ch),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.down_conv(x)

class UpConv(nn.Module):
    def __init__(self, in_ch, out_ch, groups=4):
        super().__init__()
        self.up_conv = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2, bias=False),
            nn.GroupNorm(groups, out_ch),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.up_conv(x)
        

class UNet(nn.Module):
    def __init__(self, latent_dim=(4, 16, 16), time_dim=128, num_groups=4):
        super().__init__()
        C = 64
        
        self.time_mlp = TimeEmbedding(time_dim)

        self.encoder1 = ResBlock(latent_dim[0], C, time_dim, num_groups)
        self.down1 = DownConv(C, C*2, num_groups)

        self.encoder2 = ResBlock(C*2, C*2, time_dim, num_groups)
        self.down2 = DownConv(C*2, C*4, num_groups)

        self.encoder3 = ResBlock(C*4, C*4, time_dim, num_groups)
        self.down3 = DownConv(C*4, C*8, num_groups)

        self.bottleneck = ResBlock(C*8, C*8, time_dim, num_groups)

        self.up3 = UpConv(C*8, C*4)
        self.decoder3 = ResBlock(C*8, C*4, time_dim, num_groups)

        self.up2 = UpConv(C*4, C*2)
        self.decoder2 = ResBlock(C*4, C*2, time_dim, num_groups)
        
        self.up1 =  UpConv(C*2, C)
        self.decoder1 = ResBlock(C*2, C, time_dim, num_groups)

        self.out_conv = nn.Conv2d(C, latent_dim[0], kernel_size=1, bias=True)


    def forward(self, z, t):
        t_emb = self.time_mlp(t)

        e1 = self.encoder1(z, t_emb)
        e2 = self.encoder2(self.down1(e1), t_emb)
        e3 = self.encoder3(self.down2(e2), t_emb)

        bottle = self.bottleneck(self.down3(e3), t_emb)
        bottle = self.up3(bottle)

        d3 = torch.cat([bottle, e3], dim=1)
        d3 = self.up2(self.decoder3(d3, t_emb))

        d2 = torch.cat([d3, e2], dim=1)
        d2 = self.up1(self.decoder2(d2, t_emb))

        d1 = torch.cat([d2, e1], dim=1)
        d1 = self.decoder1(d1, t_emb)

        return self.out_conv(d1)
