import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlockUp(nn.Module):
    """
    A single residual block that upsamples by a factor of 2.
    - in_channels → out_channels
    - Bilinear/nearest upsampling on both main path and skip path.
    - Two 3×3 convs on the main path, each followed by BatchNorm2d + SiLU.
    - On the skip path, a 1×1 conv after upsampling to match out_channels.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # First conv (after upsampling)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
        # Second conv (keeps spatial dims)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
        # Skip branch: upsample + 1×1 conv to match out_channels
        self.skip_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_channels, H, W)
        
        # ------- main path -------
        # 1) BatchNorm + SiLU, then upsample
        out = self.bn1(x)
        out = F.silu(out, inplace=True)
        out = F.interpolate(out, scale_factor=2, mode='nearest')  # (B, in_channels, 2H, 2W)
        
        # 2) conv1
        out = self.conv1(out)
        
        # 3) BatchNorm + SiLU + conv2
        out = self.bn2(out)
        out = F.silu(out, inplace=True)
        out = self.conv2(out)
        # now out has shape (B, out_channels, 2H, 2W)
        
        # ------- skip path -------
        skip = F.interpolate(x, scale_factor=2, mode='nearest')  # (B, in_channels, 2H, 2W)
        skip = self.skip_conv(skip)                              # → (B, out_channels, 2H, 2W)
        
        return F.silu(out + skip, inplace=True)


class ResNetDecoder(nn.Module):
    def __init__(self, latent_dim=(16, 16, 16)):
        super().__init__()
        C, H, W = latent_dim
        
        # 1) Project latent (16 → 128 channels) at 16×16
        self.initial_conv = nn.Conv2d(C, 128, kernel_size=3, stride=1, padding=1)
        self.initial_bn   = nn.BatchNorm2d(128)
        
        # 2) Two ResBlockUp modules:
        #    - 16×16 → 32×32 (128→128)
        #    - 32×32 → 64×64 (128→64)
        self.resblock1 = ResBlockUp(in_channels=128, out_channels=128)  # 16→32
        self.resblock2 = ResBlockUp(in_channels=128, out_channels=64)   # 32→64
        
        # 3) Final conv to get 3 channels at 64×64
        #    (Use kernel_size=3, padding=1 to keep spatial dims)
        self.final_bn   = nn.BatchNorm2d(64)
        self.final_conv = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)

        self.output_act = nn.Tanh()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # 1) project up to 128 channels (still 16×16)
        x = self.initial_conv(z)   # now x is (B, 128, 16, 16)
        x = self.initial_bn(x)
        x = F.silu(x, inplace=True)
        
        # ResBlockUp #1: 16×16 → 32×32
        x = self.resblock1(x)      # (B, 128, 32, 32)
        
        # ResBlockUp #2: 32×32 → 64×64 (→ 64 channels)
        x = self.resblock2(x)      # (B, 64, 64, 64)
        
        # Final BN + SiLU + conv → 3 channels
        x = self.final_bn(x)
        x = F.silu(x, inplace=True)
        x = self.final_conv(x)     # (B, 3, 64, 64)
        
        return self.output_act(x)

class Encoder(nn.Module):
    def __init__(self, latent_dim=(16, 16, 16)):
        super().__init__()
        C, H, W = latent_dim
        # 1) Mix at 64×64 → 64×64
        self.conv1 = nn.Conv2d(3,  64,  3, stride=1, padding=1)
        self.bn1   = nn.BatchNorm2d(64)
        # 2) Downsample 64→32
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.bn2   = nn.BatchNorm2d(128)
        # 3) Mix at 32×32 → 32×32
        self.conv3 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.bn3   = nn.BatchNorm2d(256)
        # 4) Downsample 32→16
        self.conv4 = nn.Conv2d(256, 256, 3, stride=2, padding=1)
        self.bn4   = nn.BatchNorm2d(256)

        # mu/logvar heads from 256 → C
        self.mu = nn.Conv2d(256, C, kernel_size=1)
        self.logvar = nn.Conv2d(256, C, kernel_size=1)

    def forward(self, x):
        x = F.silu(self.bn1(self.conv1(x)))  # 64×64
        x = F.silu(self.bn2(self.conv2(x)))  # 32×32
        x = F.silu(self.bn3(self.conv3(x)))  # 32×32
        x = F.silu(self.bn4(self.conv4(x)))  # 16×16
        return self.mu(x), self.logvar(x)
 
