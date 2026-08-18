"""
U-Net backbone for the DDPM diffusion model.

Architecture (with base_channels=64, channel_mults=(1,2,4,8), image 224x224):

  init_conv :  3 → 64   (224)
  enc0      : 64 → 64   (224 → 112)
  enc1      : 64 → 128  (112 → 56)
  enc2      : 128 → 256 (56  → 28)   ← self-attention
  enc3      : 256 → 512 (28  → 14)   ← self-attention
  mid       : 512 → 512 (14)         ← self-attention
  dec0      : 512+512→256 (14 → 28)  ← self-attention
  dec1      : 256+256→128 (28 → 56)  ← self-attention
  dec2      : 128+128→64  (56 → 112)
  dec3      :  64+64→ 64  (112 → 224)
  final_conv:  64 → 3    (224)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def num_groups(channels: int, max_groups: int = 32) -> int:
    """Return largest divisor of `channels` that is <= max_groups."""
    g = max_groups
    while g > 1 and channels % g != 0:
        g -= 1
    return g


def sinusoidal_embedding(timesteps: torch.Tensor, dim: int,
                         max_period: int = 10000) -> torch.Tensor:
    """Classic sinusoidal positional embedding for timesteps."""
    assert dim % 2 == 0
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(half, dtype=torch.float32, device=timesteps.device)
        / half
    )
    args = timesteps[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # (B, dim)


# ---------------------------------------------------------------------------
# Core blocks
# ---------------------------------------------------------------------------

class TimeEmbedding(nn.Module):
    """Sinusoidal embedding → 2-layer MLP → out_dim."""

    def __init__(self, base_dim: int, out_dim: int):
        super().__init__()
        self.base_dim = base_dim
        self.mlp = nn.Sequential(
            nn.Linear(base_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(sinusoidal_embedding(t, self.base_dim))


class ResBlock(nn.Module):
    """
    ResNet-style block conditioned on timestep embedding.
      x → GroupNorm → SiLU → Conv → + time_proj(t_emb) → GroupNorm → SiLU → Dropout → Conv → + shortcut
    """

    def __init__(self, in_ch: int, out_ch: int, t_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups(in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.t_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_dim, out_ch),
        )

        self.norm2   = nn.GroupNorm(num_groups(out_ch), out_ch)
        self.drop    = nn.Dropout(dropout)
        self.conv2   = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.t_proj(t_emb)[:, :, None, None]   # broadcast over H, W
        h = self.conv2(self.drop(F.silu(self.norm2(h))))
        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    """
    Multi-head self-attention for spatial feature maps.
    Reshape (B,C,H,W) → (B, H*W, C), attend, reshape back.
    """

    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        # Ensure num_heads divides channels
        while channels % num_heads != 0 and num_heads > 1:
            num_heads //= 2
        self.norm = nn.GroupNorm(num_groups(channels), channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x).reshape(B, C, H * W).permute(0, 2, 1)  # (B, HW, C)
        h, _ = self.attn(h, h, h, need_weights=False)
        return x + h.permute(0, 2, 1).reshape(B, C, H, W)


# ---------------------------------------------------------------------------
# Encoder / Decoder building blocks
# ---------------------------------------------------------------------------

class EncoderBlock(nn.Module):
    """ResBlock (+ optional attention) → skip → strided-conv downsample."""

    def __init__(self, in_ch: int, out_ch: int, t_dim: int,
                 has_attn: bool = False, dropout: float = 0.1):
        super().__init__()
        self.res  = ResBlock(in_ch, out_ch, t_dim, dropout)
        self.attn = AttentionBlock(out_ch) if has_attn else nn.Identity()
        self.down = nn.Conv2d(out_ch, out_ch, 4, stride=2, padding=1)

    def forward(self, x, t_emb):
        x    = self.res(x, t_emb)
        x    = self.attn(x)
        skip = x                           # save before downsampling
        x    = self.down(x)
        return x, skip


class DecoderBlock(nn.Module):
    """ConvTranspose upsample → concat skip → ResBlock (+ optional attention)."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, t_dim: int,
                 has_attn: bool = False, dropout: float = 0.1):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, in_ch, 4, stride=2, padding=1)
        self.res  = ResBlock(in_ch + skip_ch, out_ch, t_dim, dropout)
        self.attn = AttentionBlock(out_ch) if has_attn else nn.Identity()

    def forward(self, x, skip, t_emb):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.res(x, t_emb)
        x = self.attn(x)
        return x


# ---------------------------------------------------------------------------
# Full U-Net
# ---------------------------------------------------------------------------

class UNet(nn.Module):
    """
    Time-conditioned U-Net for DDPM noise prediction.

    Parameters
    ----------
    in_channels   : image channels (3 for RGB)
    base_channels : channel width at level 0
    channel_mults : channel multipliers at each encoder depth
    t_dim_base    : sinusoidal embedding base dimension (projected to base_channels * 4)
    dropout       : dropout rate inside ResBlocks
    attn_depths   : tuple of encoder depth indices where attention is applied
    """

    def __init__(
        self,
        in_channels:   int   = 3,
        base_channels: int   = 64,
        channel_mults: tuple = (1, 2, 4, 8),
        t_dim_base:    int   = 256,
        dropout:       float = 0.1,
        attn_depths:   tuple = (2, 3),
    ):
        super().__init__()

        t_dim   = base_channels * 4
        chs     = [base_channels * m for m in channel_mults]   # e.g. [64,128,256,512]
        n_lvls  = len(chs)

        # Time embedding
        self.time_embed = TimeEmbedding(t_dim_base, t_dim)

        # Initial projection
        self.init_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # ---- Encoder ----
        self.enc_blocks = nn.ModuleList()
        in_ch = base_channels
        for depth, out_ch in enumerate(chs):
            has_attn = depth in attn_depths
            self.enc_blocks.append(EncoderBlock(in_ch, out_ch, t_dim, has_attn, dropout))
            in_ch = out_ch

        # ---- Middle ----
        mid_ch = chs[-1]
        self.mid_res1 = ResBlock(mid_ch, mid_ch, t_dim, dropout)
        self.mid_attn = AttentionBlock(mid_ch)
        self.mid_res2 = ResBlock(mid_ch, mid_ch, t_dim, dropout)

        # ---- Decoder ----
        # dec[i] receives:  in_ch (from previous), skip from enc[n_lvls-1-i]
        # dec[i] outputs:   chs[n_lvls-2-i]  (or base_channels for last)
        self.dec_blocks = nn.ModuleList()
        for i in range(n_lvls):
            skip_ch = chs[n_lvls - 1 - i]
            out_idx = n_lvls - 2 - i
            out_ch  = chs[out_idx] if out_idx >= 0 else base_channels
            has_attn = (n_lvls - 1 - i) in attn_depths
            self.dec_blocks.append(DecoderBlock(in_ch, skip_ch, out_ch, t_dim, has_attn, dropout))
            in_ch = out_ch

        # ---- Final projection ----
        self.final_conv = nn.Sequential(
            nn.GroupNorm(num_groups(in_ch), in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, in_channels, 1),
        )

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, C, H, W)  noisy image
        t : (B,)           integer timesteps

        Returns
        -------
        eps_pred : (B, C, H, W)  predicted noise
        """
        t_emb = self.time_embed(t)          # (B, t_dim)

        x = self.init_conv(x)               # (B, base_ch, H, W)

        # Encoder — collect skips
        skips = []
        for enc in self.enc_blocks:
            x, skip = enc(x, t_emb)
            skips.append(skip)

        # Middle
        x = self.mid_res1(x, t_emb)
        x = self.mid_attn(x)
        x = self.mid_res2(x, t_emb)

        # Decoder — consume skips in reverse
        for dec, skip in zip(self.dec_blocks, reversed(skips)):
            x = dec(x, skip, t_emb)

        return self.final_conv(x)