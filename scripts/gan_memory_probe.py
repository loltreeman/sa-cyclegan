"""GAN memory probe — §3.7.1.

Instantiates the full SA-CycleGAN component graph and runs ~N optimizer
steps with random tensors at 256×256 batch 1.  The only question is:

    Do two ResNet-9 generators + two PatchGAN discriminators + frozen VGG-19
    fit inside 6 GB at batch 1?

Components (all hyperparameters from configs/base.yaml gan block):
    G, F     — ResNet-9 generators with SAGAN self-attention (§3.3.2–3.3.3)
    D_A, D_B — PatchGAN discriminators with spectral norm (§3.3.4)
    VGG-19   — frozen, relu1_2 / relu2_2 / relu3_3 / relu4_3 (§3.3.5)

No dataloader, no real data, no convergence.  Weights are noise.
Peak memory is measured over all N steps so optimiser state (Adam moment
buffers, allocated after step 1) is fully included.

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

Usage:
    python scripts/gan_memory_probe.py
    python scripts/gan_memory_probe.py --steps 20
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
import yaml

ROOT = Path(__file__).resolve().parents[1]

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


# ── Probe ────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--steps", type=int, default=20,
                        help="optimizer steps (default: 20)")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: no CUDA device — this probe requires GPU")
        return 1

    with (ROOT / "configs" / "base.yaml").open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["gan"]

    device = torch.device("cuda")
    imgsz  = cfg["crop_size"]   # 256
    batch  = cfg["batch_size"]  # 1
    gcfg   = cfg["generator"]
    dcfg   = cfg["discriminator"]
    acfg   = cfg["attention"]
    lcfg   = cfg["loss"]
    fccfg  = lcfg["feature_consistency"]

    print(f"torch   {torch.__version__}")
    print(f"device  {torch.cuda.get_device_name(0)}")
    print(f"imgsz   {imgsz}   batch {batch}   steps {args.steps}")
    print()

    # ── Build ─────────────────────────────────────────────────────────────────
    gen_kw = dict(
        ngf=gcfg["ngf"],
        enc_attn=acfg["encoder"],
        dec_attn=acfg["decoder"],
        gamma_init=acfg["gamma_init"],
    )
    dis_kw = dict(
        ndf=dcfg["ndf"],
        n_layers=dcfg["n_layers"],
        slope=dcfg["leaky_relu_slope"],
    )

    G   = ResNet9Generator(**gen_kw).train().to(device)
    F_  = ResNet9Generator(**gen_kw).train().to(device)
    D_A = PatchGANDiscriminator(**dis_kw).train().to(device)
    D_B = PatchGANDiscriminator(**dis_kw).train().to(device)
    vgg = VGGExtractor(fccfg["layers"]).eval().to(device)

    p_G  = sum(p.numel() for p in G.parameters())
    p_DA = sum(p.numel() for p in D_A.parameters())
    p_vgg = sum(p.numel() for p in vgg.parameters())
    print(f"G  parameters : {p_G:,}  (each generator)")
    print(f"D  parameters : {p_DA:,}  (each discriminator)")
    print(f"VGG parameters: {p_vgg:,}  (frozen)")
    print()

    # ── Optimisers ────────────────────────────────────────────────────────────
    opt_G = torch.optim.Adam(
        list(G.parameters()) + list(F_.parameters()),
        lr=cfg["lr"], betas=(cfg["beta1"], cfg["beta2"]),
    )
    opt_D = torch.optim.Adam(
        list(D_A.parameters()) + list(D_B.parameters()),
        lr=cfg["lr"], betas=(cfg["beta1"], cfg["beta2"]),
    )

    lam_cyc = lcfg["lambda_cyc"]
    lam_id  = lcfg["lambda_id"]
    lam_fc  = lcfg["lambda_fc"]

    # ── Steps ─────────────────────────────────────────────────────────────────
    # Counter reset BEFORE the loop so Adam moment buffers (allocated after
    # step 1) are included in the reported peak.
    torch.cuda.reset_peak_memory_stats()

    for step in range(args.steps):
        real_A = torch.randn(batch, 3, imgsz, imgsz, device=device)
        real_B = torch.randn(batch, 3, imgsz, imgsz, device=device)

        # Generator update — all six forward passes accumulate before backward
        opt_G.zero_grad()

        fake_B = G(real_A)
        fake_A = F_(real_B)

        loss_G = (lsgan(D_B(fake_B), True) + lsgan(D_A(fake_A), True))

        rec_A = F_(fake_B)
        rec_B = G(fake_A)
        loss_G += lam_cyc * (F.l1_loss(rec_A, real_A) + F.l1_loss(rec_B, real_B))

        idt_B = F_(real_A)
        idt_A = G(real_B)
        loss_G += lam_id * (F.l1_loss(idt_A, real_B) + F.l1_loss(idt_B, real_A))

        # Feature consistency: VGG(G(real_A)) vs VGG(real_B) — target=G(x)
        loss_G += fc_loss(vgg, fake_B, real_B, lam_fc)

        loss_G.backward()
        opt_G.step()

        # Discriminator update — detach fakes so graph is independent
        opt_D.zero_grad()

        loss_D  = lsgan(D_A(real_A), True)          + lsgan(D_A(fake_A.detach()), False)
        loss_D += lsgan(D_B(real_B), True)          + lsgan(D_B(fake_B.detach()), False)
        (loss_D * 0.5).backward()
        opt_D.step()

        if step == 0 or (step + 1) % 5 == 0:
            live = torch.cuda.memory_allocated() / 1024 ** 3
            peak = torch.cuda.max_memory_allocated() / 1024 ** 3
            print(f"  step {step + 1:>3}  live {live:.3f} GB  peak_alloc {peak:.3f} GB")

    torch.cuda.synchronize()
    peak_alloc    = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()

    budget_gb = 6.0
    fits = (peak_alloc / 1024 ** 3) <= budget_gb

    print(f"\n── Result ──────────────────────────────────────────────")
    print(f"  peak memory allocated : {peak_alloc / 1024 ** 3:.3f} GB")
    print(f"  peak memory reserved  : {peak_reserved / 1024 ** 3:.3f} GB")
    print(f"  fits in {budget_gb} GB      : {'YES' if fits else 'NO — reduce batch or crop_size'}")

    # ── JSON ──────────────────────────────────────────────────────────────────
    out = ROOT / "runs" / "gan_memory_probe.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "imgsz": imgsz,
        "batch": batch,
        "steps": args.steps,
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "generator_params": p_G,
        "discriminator_params": p_DA,
        "vgg_params_frozen": p_vgg,
        "peak_memory_allocated_gb": round(peak_alloc / 1024 ** 3, 3),
        "peak_memory_reserved_gb":  round(peak_reserved / 1024 ** 3, 3),
        "fits_6gb": fits,
        "note": (
            "Random tensors; no dataloader. Attention: enc+dec (§3.3.3). "
            "Spectral norm on all D conv layers (§3.3.4). "
            "VGG frozen, relu1_2/2_2/3_3/4_3 (§3.3.5). "
            "Peak counter reset before step loop — includes Adam moment buffers. "
            "n_layers=4 as configured; receptive field ~142x142, not 70x70 — "
            "reconcile §3.3.4 before Phase 3."
        ),
    }, indent=2), encoding="utf-8")
    print(f"\nwritten: {out.relative_to(ROOT)}")
    return 0 if fits else 1


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
