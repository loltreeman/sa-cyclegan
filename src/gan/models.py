"""SA-CycleGAN model components.

Architecture extracted verbatim from scripts/gan_memory_probe.py.
All hyperparameters come from configs/base.yaml gan block.

Architecture notes
------------------
Generator  — 3-stage enc / 9 res / 3-stage dec (§3.3.2).  Spatial sizes
at crop_size=256, ngf=64:
    initial : 256×256  64 ch
    down 1  : 128×128 128 ch
    down 2  :  64×64  256 ch
    down 3  :  32×32  256 ch  ← encoder attention (§3.3.3, bottleneck_resolution=32)
    res × 9 :  32×32  256 ch
                               ← decoder attention (before first upsample)
    up 1    :  64×64  128 ch
    up 2    : 128×128  64 ch
    up 3    : 256×256  64 ch
    output  : 256×256   3 ch  (tanh)

Discriminator — n_layers=4 from config means four stride-2 convolutions
(256→128→64→32→16→15→14 output map).  The config names this "patchgan_70"
but n_layers=3 is the standard 70×70 receptive field; n_layers=4 gives
~142×142.  The probe uses n_layers=4 as configured.  If §3.3.4 asserts a
70×70 receptive field, n_layers needs to be 3 — reconcile before Phase 3.

Feature consistency — VGG(G(real_A)) vs VGG(real_B), L1, equal layer
weights (§3.3.5, target="G(x)").  Gradient flows through G via VGG;
VGG parameters are frozen.
"""
from __future__ import annotations

import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm

__all__ = [
    "SelfAttention",
    "_ResnetBlock",
    "_down",
    "_up",
    "ResNet9Generator",
    "PatchGANDiscriminator",
    "VGGExtractor",
    "lsgan",
    "fc_loss",
    "ImageBuffer",
    "lr_lambda",
]

# VGG-19 layer names → index in vgg.features
_VGG_IDX = {"relu1_2": 3, "relu2_2": 8, "relu3_3": 15, "relu4_3": 24}


# ── Self-attention (SAGAN, §3.3.3) ──────────────────────────────────────────

class SelfAttention(nn.Module):
    """Learnable-gamma attention; gamma initialised to 0 (§3.3.3)."""

    def __init__(self, channels: int, gamma_init: float = 0.0) -> None:
        super().__init__()
        c = max(1, channels // 8)
        self.q = nn.Conv2d(channels, c, 1, bias=False)
        self.k = nn.Conv2d(channels, c, 1, bias=False)
        self.v = nn.Conv2d(channels, channels, 1, bias=False)
        self.gamma = nn.Parameter(torch.full((1,), gamma_init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        q = self.q(x).view(B, -1, H * W).permute(0, 2, 1)  # B, HW, c
        k = self.k(x).view(B, -1, H * W)                    # B, c,  HW
        attn = torch.softmax(q @ k, dim=-1)                  # B, HW, HW
        v = self.v(x).view(B, C, H * W)                     # B, C,  HW
        out = (v @ attn.permute(0, 2, 1)).view(B, C, H, W)
        return self.gamma * out + x


# ── ResNet-9 generator (§3.3.2–3.3.3) ──────────────────────────────────────

class _ResnetBlock(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(ch, ch, 3, bias=False),
            nn.InstanceNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(ch, ch, 3, bias=False),
            nn.InstanceNorm2d(ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


def _down(in_c: int, out_c: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, stride=2, padding=1, bias=False),
        nn.InstanceNorm2d(out_c),
        nn.ReLU(inplace=True),
    )


def _up(in_c: int, out_c: int) -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose2d(in_c, out_c, 3, stride=2, padding=1,
                           output_padding=1, bias=False),
        nn.InstanceNorm2d(out_c),
        nn.ReLU(inplace=True),
    )


class ResNet9Generator(nn.Module):
    def __init__(self, ngf: int = 64, n_res: int = 9,
                 enc_attn: bool = True, dec_attn: bool = True,
                 gamma_init: float = 0.0) -> None:
        super().__init__()
        # Encoder
        self.init = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, ngf, 7, bias=False),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True),
        )
        self.downs = nn.ModuleList([
            _down(ngf,     ngf * 2),   # 256 → 128
            _down(ngf * 2, ngf * 4),   # 128 →  64
            _down(ngf * 4, ngf * 4),   #  64 →  32  (third stage, keeps ch)
        ])
        self.enc_attn = (SelfAttention(ngf * 4, gamma_init)
                         if enc_attn else nn.Identity())
        # Residual stack at 32×32, 256 ch
        self.res = nn.Sequential(*[_ResnetBlock(ngf * 4) for _ in range(n_res)])
        self.dec_attn = (SelfAttention(ngf * 4, gamma_init)
                         if dec_attn else nn.Identity())
        # Decoder
        self.ups = nn.ModuleList([
            _up(ngf * 4, ngf * 2),    #  32 →  64
            _up(ngf * 2, ngf),        #  64 → 128
            _up(ngf,     ngf),        # 128 → 256  (third stage)
        ])
        self.out = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, 3, 7),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.init(x)
        for d in self.downs:
            x = d(x)
        x = self.enc_attn(x)
        x = self.res(x)
        x = self.dec_attn(x)
        for u in self.ups:
            x = u(x)
        return self.out(x)


# ── PatchGAN discriminator (§3.3.4) ─────────────────────────────────────────

def _sn(conv: nn.Conv2d) -> nn.Conv2d:
    return nn.utils.spectral_norm(conv)


class PatchGANDiscriminator(nn.Module):
    """Spectral norm on every conv layer (§3.3.4).

    n_layers=4 (from config) → four stride-2 convs, output map 14×14,
    receptive field ~142×142.  See module docstring note on patchgan_70.
    """

    def __init__(self, ndf: int = 64, n_layers: int = 4,
                 slope: float = 0.2) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            _sn(nn.Conv2d(3, ndf, 4, 2, 1)),
            nn.LeakyReLU(slope, inplace=True),
        ]
        mult = 1
        for i in range(1, n_layers):
            prev, mult = mult, min(2 ** i, 8)
            layers += [
                _sn(nn.Conv2d(ndf * prev, ndf * mult, 4, 2, 1)),
                nn.InstanceNorm2d(ndf * mult),
                nn.LeakyReLU(slope, inplace=True),
            ]
        prev, mult = mult, min(2 ** n_layers, 8)
        layers += [
            _sn(nn.Conv2d(ndf * prev, ndf * mult, 4, 1, 1)),
            nn.InstanceNorm2d(ndf * mult),
            nn.LeakyReLU(slope, inplace=True),
            _sn(nn.Conv2d(ndf * mult, 1, 4, 1, 1)),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── VGG-19 feature extractor (§3.3.5) ───────────────────────────────────────

class VGGExtractor(nn.Module):
    """Frozen VGG-19, captures relu1_2 / relu2_2 / relu3_3 / relu4_3."""

    def __init__(self, layer_names: list[str]) -> None:
        super().__init__()
        indices = {_VGG_IDX[n] for n in layer_names}
        depth = max(indices) + 1
        vgg = tvm.vgg19(weights=tvm.VGG19_Weights.IMAGENET1K_V1)
        self.features = vgg.features[:depth]
        self.capture = indices
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        out = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.capture:
                out.append(x)
        return out


# ── Losses ───────────────────────────────────────────────────────────────────

def lsgan(pred: torch.Tensor, real: bool) -> torch.Tensor:
    t = torch.ones_like(pred) if real else torch.zeros_like(pred)
    return F.mse_loss(pred, t)


def fc_loss(vgg: VGGExtractor, generated: torch.Tensor,
            target: torch.Tensor, lam: float) -> torch.Tensor:
    """VGG(G(x)) vs VGG(real_B), L1, equal weights (§3.3.5, target=G(x))."""
    feats_g = vgg(generated)
    with torch.no_grad():
        feats_t = [f.detach() for f in vgg(target)]
    loss = sum(F.l1_loss(g, t) for g, t in zip(feats_g, feats_t))
    return lam * loss / len(feats_g)


# ── Replay buffer ─────────────────────────────────────────────────────────────

class ImageBuffer:
    """Standard CycleGAN 50-image replay buffer.

    Stabilises discriminator training by presenting a mix of current and
    previously generated images rather than only the latest batch.
    """

    def __init__(self, max_size: int = 50) -> None:
        self.max_size = max_size
        self._pool: list[torch.Tensor] = []

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """Return a tensor of the same shape, drawn from the buffer or current batch.

        Each image in the batch is handled independently:
          - If the buffer is not full, the image is stored and returned as-is.
          - Otherwise, with p=0.5: swap a random stored image with the current
            one and return the stored image; with p=0.5: return the current image.
        Always returns a detached tensor — the discriminator must not
        backpropagate through the buffer's stored graphs.
        """
        result = []
        for img in images.detach():           # img: C×H×W
            img = img.unsqueeze(0)            # 1×C×H×W
            if len(self._pool) < self.max_size:
                self._pool.append(img)
                result.append(img)
            elif random.random() < 0.5:
                idx = random.randrange(len(self._pool))
                stored = self._pool[idx].clone()
                self._pool[idx] = img
                result.append(stored)
            else:
                result.append(img)
        return torch.cat(result, dim=0)


# ── LR schedule ──────────────────────────────────────────────────────────────

def lr_lambda(epoch: int, epochs_constant: int, epochs_decay: int) -> float:
    """Flat then linear-decay schedule for LambdaLR (§3.3.7).

    Returns 1.0 for epochs 0..epochs_constant-1, then decays linearly to 0
    over epochs_decay epochs.  Clipped at 0.0 to avoid negative multipliers
    past the end of training.
    """
    if epoch < epochs_constant:
        return 1.0
    return max(0.0, 1.0 - (epoch - epochs_constant) / epochs_decay)
