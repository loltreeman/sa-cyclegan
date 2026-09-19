"""SA-CycleGAN training harness — §3.3, §3.7.1.

Implements the six production/ablation generator runs from Table 3.7 and the
λ_fc sensitivity sweep (§3.3.5).  Checkpoints are saved under
``<out_dir>/<run_name>/ckpt/``.  Loss scalars are appended as JSON lines to
``<out_dir>/<run_name>/losses.jsonl`` at the end of every epoch.

Six runs (RUN_CONFIGS)
----------------------
  rain_light        — full SA-CycleGAN (enc+dec attention), light rain pool
  rain_moderate     — full SA-CycleGAN, moderate rain pool
  rain_heavy        — full SA-CycleGAN, heavy rain pool
  lowlight          — full SA-CycleGAN, low-light pool (lum_severe)
  ablation_enc_only — encoder attention only, heavy rain (D4)
  ablation_no_attn  — no attention (baseline CycleGAN), heavy rain (D4)

λ_fc sweep (§3.3.5)
---------------------
  Three sequential runs at λ_fc ∈ [0.5, 1.0, 2.0] for any severity.
  Run names: sweep_fc0.5_<severity>, sweep_fc1.0_<severity>, sweep_fc2.0_<severity>.

Resuming
--------
  Pass ``--resume <path/to/epoch_NNNN.pt>`` — the checkpoint stores epoch index,
  G/F/D_A/D_B state_dicts, and both optimiser states.  Training resumes from
  the next epoch.

Usage
-----
  python src/gan/train.py --run rain_heavy \\
                          --manifest runs/manifest.parquet \\
                          --pool-a data/pool_a --pool-b data/pool_c \\
                          [--lambda-fc 1.0] \\
                          [--resume runs/gan/rain_heavy/ckpt/epoch_0050.pt] \\
                          [--sweep-fc] \\
                          [--device cuda] \\
                          [--workers 0]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

import yaml

from src.gan.models import (
    ResNet9Generator,
    PatchGANDiscriminator,
    VGGExtractor,
    ImageBuffer,
    lsgan,
    fc_loss,
    lr_lambda,
)
from src.gan.dataset import build_unpaired_loader

__all__ = ["GANTrainer", "RUN_CONFIGS"]

ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Table 3.7 — six run configurations
# ---------------------------------------------------------------------------

RUN_CONFIGS: dict[str, dict] = {
    "rain_light":        {"severity": "rain_light",   "enc_attn": True,  "dec_attn": True},
    "rain_moderate":     {"severity": "rain_moderate", "enc_attn": True,  "dec_attn": True},
    "rain_heavy":        {"severity": "rain_heavy",    "enc_attn": True,  "dec_attn": True},
    "lowlight":          {"severity": "lum_severe",    "enc_attn": True,  "dec_attn": True},
    "ablation_enc_only": {"severity": "rain_heavy",    "enc_attn": True,  "dec_attn": False},
    "ablation_no_attn":  {"severity": "rain_heavy",    "enc_attn": False, "dec_attn": False},
}


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class GANTrainer:
    """One complete SA-CycleGAN training run.

    Parameters
    ----------
    cfg:
        Contents of the ``gan:`` block from configs/base.yaml.
    run_name:
        Identifier used for checkpoint and log paths.
    run_cfg:
        One entry from RUN_CONFIGS (severity + attention flags).
    out_dir:
        Parent directory; outputs written to ``out_dir/run_name/``.
    manifest_path:
        Path to the manifest Parquet file.
    pool_a_root:
        Root directory for Pool A (clear-weather) frames.
    pool_b_root:
        Root directory for Pool B/C (rainy/low-light) frames.
    lambda_fc:
        Override for λ_fc; if None, uses ``cfg["loss"]["lambda_fc"]``.
    resume:
        Checkpoint ``.pt`` file to resume from; None starts from scratch.
    device:
        PyTorch device string (default "cuda").
    """

    def __init__(
        self,
        cfg: dict,
        run_name: str,
        run_cfg: dict,
        *,
        out_dir: Path,
        manifest_path: Path,
        pool_a_root: Path,
        pool_b_root: Path,
        lambda_fc: Optional[float] = None,
        resume: Optional[Path] = None,
        device: str = "cuda",
        workers: int = 0,
    ) -> None:
        self.cfg = cfg
        self.run_name = run_name
        self.run_cfg = run_cfg
        self.device = torch.device(device)
        self.workers = workers

        lcfg = cfg["loss"]
        self.lam_cyc: float = lcfg["lambda_cyc"]
        self.lam_id: float  = lcfg["lambda_id"]
        self.lam_fc: float  = lambda_fc if lambda_fc is not None else lcfg["lambda_fc"]

        self.epochs_total: int    = cfg["epochs_total"]
        self.epochs_constant: int = cfg["epochs_constant_lr"]
        self.epochs_decay: int    = cfg["epochs_decay_lr"]
        self.save_period: int     = cfg.get("save_period", 10)

        self.run_dir = out_dir / run_name
        self.ckpt_dir = self.run_dir / "ckpt"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.losses_path = self.run_dir / "losses.jsonl"

        gcfg = cfg["generator"]
        dcfg = cfg["discriminator"]
        acfg = cfg["attention"]
        fccfg = lcfg["feature_consistency"]

        gen_kw = dict(
            ngf=gcfg["ngf"],
            enc_attn=run_cfg["enc_attn"],
            dec_attn=run_cfg["dec_attn"],
            gamma_init=acfg["gamma_init"],
        )
        dis_kw = dict(
            ndf=dcfg["ndf"],
            n_layers=dcfg["n_layers"],
            slope=dcfg["leaky_relu_slope"],
        )

        self.G   = ResNet9Generator(**gen_kw).to(self.device)
        self.F_  = ResNet9Generator(**gen_kw).to(self.device)
        self.D_A = PatchGANDiscriminator(**dis_kw).to(self.device)
        self.D_B = PatchGANDiscriminator(**dis_kw).to(self.device)
        self.vgg = VGGExtractor(fccfg["layers"]).eval().to(self.device)

        self.opt_G = optim.Adam(
            list(self.G.parameters()) + list(self.F_.parameters()),
            lr=cfg["lr"],
            betas=(cfg["beta1"], cfg["beta2"]),
        )
        self.opt_D = optim.Adam(
            list(self.D_A.parameters()) + list(self.D_B.parameters()),
            lr=cfg["lr"],
            betas=(cfg["beta1"], cfg["beta2"]),
        )

        def _sched(epoch: int) -> float:
            return lr_lambda(epoch, self.epochs_constant, self.epochs_decay)

        self.sched_G = LambdaLR(self.opt_G, lr_lambda=_sched)
        self.sched_D = LambdaLR(self.opt_D, lr_lambda=_sched)

        # One buffer per fake domain; stale fakes reduce discriminator oscillation
        self.buf_A = ImageBuffer()
        self.buf_B = ImageBuffer()

        self._l1 = nn.L1Loss()

        self.manifest_path = manifest_path
        self.pool_a_root   = pool_a_root
        self.pool_b_root   = pool_b_root

        self._start_epoch = 0
        if resume is not None:
            self._start_epoch = self._load_checkpoint(resume)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self) -> None:
        loader = build_unpaired_loader(
            manifest_path=self.manifest_path,
            pool_a_root=self.pool_a_root,
            pool_b_root=self.pool_b_root,
            severity=self.run_cfg["severity"],
            load_size=self.cfg["load_size"],
            crop_size=self.cfg["crop_size"],
            hflip=self.cfg["hflip"],
            batch_size=self.cfg["batch_size"],
            workers=self.workers,
        )

        for epoch in range(self._start_epoch, self.epochs_total):
            self.G.train()
            self.F_.train()
            self.D_A.train()
            self.D_B.train()

            epoch_sums: dict[str, float] = {
                "G_adv": 0.0, "G_cyc": 0.0, "G_id": 0.0, "G_fc": 0.0, "D": 0.0,
            }
            n_steps = 0

            for real_A, real_B in loader:
                real_A = real_A.to(self.device)
                real_B = real_B.to(self.device)
                scalars = self._step(real_A, real_B)
                for k, v in scalars.items():
                    epoch_sums[k] += v
                n_steps += 1

            n = max(n_steps, 1)
            log_row = {"epoch": epoch + 1}
            log_row.update({k: round(v / n, 6) for k, v in epoch_sums.items()})

            with self.losses_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(log_row) + "\n")

            print(
                f"[{self.run_name}] epoch {epoch + 1:>3}/{self.epochs_total}  "
                f"G_adv={log_row['G_adv']:.4f}  G_cyc={log_row['G_cyc']:.4f}  "
                f"G_id={log_row['G_id']:.4f}  G_fc={log_row['G_fc']:.4f}  "
                f"D={log_row['D']:.4f}"
            )

            self.sched_G.step()
            self.sched_D.step()

            is_final = (epoch + 1 == self.epochs_total)
            if is_final or (epoch + 1) % self.save_period == 0:
                self._save_checkpoint(epoch + 1, is_final=is_final)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _step(self, real_A: torch.Tensor, real_B: torch.Tensor) -> dict[str, float]:
        # ── Generator update ──────────────────────────────────────────
        self.opt_G.zero_grad()

        fake_B = self.G(real_A)
        fake_A = self.F_(real_B)

        loss_adv = lsgan(self.D_B(fake_B), True) + lsgan(self.D_A(fake_A), True)

        rec_A = self.F_(fake_B)
        rec_B = self.G(fake_A)
        loss_cyc = self.lam_cyc * (self._l1(rec_A, real_A) + self._l1(rec_B, real_B))

        idt_B = self.F_(real_A)
        idt_A = self.G(real_B)
        loss_id = self.lam_id * (self._l1(idt_A, real_B) + self._l1(idt_B, real_A))

        # Feature consistency: VGG(G(real_A)) vs VGG(real_B), target=G(x) (§3.3.5)
        loss_fc = fc_loss(self.vgg, fake_B, real_B, self.lam_fc)

        loss_G = loss_adv + loss_cyc + loss_id + loss_fc
        loss_G.backward()
        self.opt_G.step()

        # ── Discriminator update ──────────────────────────────────────
        # Buffers supply previously generated fakes, stabilising D training
        self.opt_D.zero_grad()

        fake_A_buf = self.buf_A.query(fake_A.detach())
        fake_B_buf = self.buf_B.query(fake_B.detach())

        loss_D = (
            lsgan(self.D_A(real_A), True)    + lsgan(self.D_A(fake_A_buf), False)
            + lsgan(self.D_B(real_B), True)  + lsgan(self.D_B(fake_B_buf), False)
        )
        (loss_D * 0.5).backward()
        self.opt_D.step()

        return {
            "G_adv": loss_adv.item(),
            "G_cyc": loss_cyc.item(),
            "G_id":  loss_id.item(),
            "G_fc":  loss_fc.item(),
            "D":     (loss_D * 0.5).item(),
        }

    def _save_checkpoint(self, epoch: int, is_final: bool = False) -> Path:
        fname = "final.pt" if is_final else f"epoch_{epoch:04d}.pt"
        path = self.ckpt_dir / fname
        torch.save(
            {
                "epoch":      epoch,
                "G":          self.G.state_dict(),
                "F_":         self.F_.state_dict(),
                "D_A":        self.D_A.state_dict(),
                "D_B":        self.D_B.state_dict(),
                "opt_G":      self.opt_G.state_dict(),
                "opt_D":      self.opt_D.state_dict(),
                "run_name":   self.run_name,
                "run_cfg":    self.run_cfg,
                "lambda_fc":  self.lam_fc,
            },
            path,
        )
        print(f"  saved {path}")
        return path

    def _load_checkpoint(self, path: Path) -> int:
        ckpt = torch.load(path, map_location=self.device)
        self.G.load_state_dict(ckpt["G"])
        self.F_.load_state_dict(ckpt["F_"])
        self.D_A.load_state_dict(ckpt["D_A"])
        self.D_B.load_state_dict(ckpt["D_B"])
        self.opt_G.load_state_dict(ckpt["opt_G"])
        self.opt_D.load_state_dict(ckpt["opt_D"])
        epoch = int(ckpt["epoch"])
        print(f"  resumed from {path}  (epoch {epoch})")
        return epoch


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_cfg() -> dict:
    with (ROOT / "configs" / "base.yaml").open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)["gan"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train one SA-CycleGAN run (§3.3, §3.7.1).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--run", required=True, choices=list(RUN_CONFIGS),
        help="Run name from Table 3.7",
    )
    parser.add_argument("--manifest",  required=True, type=Path,
                        help="Path to manifest.parquet")
    parser.add_argument("--pool-a",    required=True, type=Path,
                        help="Pool A root (clear-weather frames)")
    parser.add_argument("--pool-b",    required=True, type=Path,
                        help="Pool B/C root (rainy or low-light frames)")
    parser.add_argument("--out-dir",   type=Path, default=ROOT / "runs" / "gan",
                        help="Output root (default: runs/gan)")
    parser.add_argument("--lambda-fc", type=float, default=None,
                        help="Override λ_fc (§3.3.5); ignored when --sweep-fc is set")
    parser.add_argument("--resume",    type=Path, default=None,
                        help="Checkpoint .pt to resume from")
    parser.add_argument("--sweep-fc",  action="store_true",
                        help="Run λ_fc sweep [0.5, 1.0, 2.0] for the chosen run's severity")
    parser.add_argument("--device",    default="cuda",
                        help="PyTorch device (default: cuda)")
    parser.add_argument("--workers",   type=int, default=0,
                        help="DataLoader workers (default: 0)")
    args = parser.parse_args()

    cfg = _load_cfg()
    base_run_cfg = RUN_CONFIGS[args.run]

    if args.sweep_fc:
        sweep_values: list[float] = cfg["loss"]["lambda_fc_sweep"]
        severity = base_run_cfg["severity"]
        for lfc in sweep_values:
            name = f"sweep_fc{lfc}_{severity}"
            print(f"\n=== sweep run: {name}  λ_fc={lfc} ===")
            trainer = GANTrainer(
                cfg=cfg,
                run_name=name,
                run_cfg=base_run_cfg,
                out_dir=args.out_dir,
                manifest_path=args.manifest,
                pool_a_root=args.pool_a,
                pool_b_root=args.pool_b,
                lambda_fc=lfc,
                resume=None,
                device=args.device,
                workers=args.workers,
            )
            trainer.train()
    else:
        print(f"\n=== run: {args.run} ===")
        trainer = GANTrainer(
            cfg=cfg,
            run_name=args.run,
            run_cfg=base_run_cfg,
            out_dir=args.out_dir,
            manifest_path=args.manifest,
            pool_a_root=args.pool_a,
            pool_b_root=args.pool_b,
            lambda_fc=args.lambda_fc,
            resume=args.resume,
            device=args.device,
            workers=args.workers,
        )
        trainer.train()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
