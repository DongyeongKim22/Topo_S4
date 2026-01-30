<<<<<<< HEAD
"""
train_s4d_hf.py

Minimal S4D training on sequential CIFAR10 / MNIST with optional high-frequency (HF) perturbations.

Key points:
- Datasets return images (C,H,W) after ToTensor+Normalize (no flatten in dataset).
- We flatten inside the train/eval step: (B,C,H,W) -> (B,L,C), L=H*W.
- HF perturbations are applied on (B,C,H,W) BEFORE flatten, i.e. along the same sequence axis H*W.
- Evaluation (when --hf_test) reports BOTH clean and HF metrics *on the same samples*,
  plus sensitivity metrics (flip-rate, logit L2 diff, input delta RMS).

Requires the S4 repo structure:
  from models.s4.s4d import S4D
"""

import argparse
import math
import random
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

from models.s4.s4d import S4D


# -------------------------
# Reproducibility
# -------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -------------------------
# Helpers
# -------------------------
def flatten_to_seq(x_img: torch.Tensor) -> torch.Tensor:
    """
    (B,C,H,W) -> (B,L,C) where L=H*W
    """
    if x_img.ndim != 4:
        raise ValueError(f"Expected (B,C,H,W), got {tuple(x_img.shape)}")
    B, C, H, W = x_img.shape
    return x_img.reshape(B, C, H * W).transpose(1, 2).contiguous()


def _band_to_bins(band: Tuple[float, float], F: int) -> Tuple[int, int]:
    """band fraction of Nyquist -> rFFT bin indices [0, F-1]"""
    lo_f, hi_f = band
    lo = int(round(lo_f * (F - 1)))
    hi = int(round(hi_f * (F - 1)))
    lo = max(0, min(lo, F - 1))
    hi = max(0, min(hi, F - 1))
    return lo, hi


# -------------------------
# HF perturbations
# -------------------------
class HFBandNoiseTimeRMS:
    """
    Add band-limited *random* noise whose TIME-DOMAIN RMS is controlled.

    This avoids the common pitfall where you set a frequency-domain sigma,
    but the iFFT scaling makes the actual time-domain perturbation tiny.

    Args:
      band: (lo, hi) as fraction of Nyquist in [0,1]. (0.9,1.0) == 0.9π~π.
      rms: absolute time-domain RMS to add (after Normalize scale).
      rms_rel: if not None, rms = rms_rel * rms(x) per-sample.
      p: probability to apply per sample.
    """
    def __init__(
        self,
        band: Tuple[float, float] = (0.9, 1.0),
        rms: float = 0.2,
        rms_rel: Optional[float] = None,
        p: float = 1.0,
        eps: float = 1e-12,
    ):
        lo, hi = band
        if not (0.0 <= lo <= hi <= 1.0):
            raise ValueError(f"band must be within [0,1] and lo<=hi, got {band}")
        self.band = (float(lo), float(hi))
        self.rms = float(rms)
        self.rms_rel = None if rms_rel is None else float(rms_rel)
        self.p = float(p)
        self.eps = float(eps)

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,H,W) or (C,H,W)
        if x.ndim == 3:
            x = x.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        if x.ndim != 4:
            raise ValueError(f"Expected x shape (B,C,H,W) or (C,H,W), got {tuple(x.shape)}")

        B, C, H, W = x.shape
        L = H * W
        F = L // 2 + 1

        x0 = x.to(torch.float32)

        # per-sample apply mask
        if self.p < 1.0:
            mask = (torch.rand((B, 1, 1, 1), device=x0.device) < self.p).to(x0.dtype)
        else:
            mask = None

        # determine target RMS per sample (and broadcast)
        if self.rms_rel is not None:
            # per-sample rms over all dims
            x_rms = x0.reshape(B, -1).pow(2).mean(dim=-1).sqrt()  # (B,)
            target = (self.rms_rel * x_rms).clamp_min(0.0)        # (B,)
        else:
            target = torch.full((B,), self.rms, device=x0.device, dtype=x0.dtype)

        # build complex spectrum noise N: (B,C,F)
        N = torch.zeros((B, C, F), dtype=torch.complex64, device=x0.device)
        lo, hi = _band_to_bins(self.band, F)
        if hi < lo:
            return x if not squeeze else x.squeeze(0)

        K = hi - lo + 1
        nr = torch.randn((B, C, K), device=x0.device, dtype=torch.float32)
        ni = torch.randn((B, C, K), device=x0.device, dtype=torch.float32)
        N[:, :, lo:hi + 1] = torch.complex(nr, ni)

        # enforce real constraints
        N[:, :, 0] = torch.complex(N[:, :, 0].real, torch.zeros_like(N[:, :, 0].real))
        if L % 2 == 0:
            N[:, :, -1] = torch.complex(N[:, :, -1].real, torch.zeros_like(N[:, :, -1].real))

        noise = torch.fft.irfft(N, n=L, dim=-1)  # (B,C,L), real
        # scale to target RMS per (B,C) (use per-channel rms, but target is per-sample)
        cur = noise.pow(2).mean(dim=-1, keepdim=True).sqrt()  # (B,C,1)
        # target: (B,) -> (B,1,1)
        tgt = target.view(B, 1, 1)
        noise = noise * (tgt / (cur + self.eps))

        if mask is not None:
            noise = noise.reshape(B, C, H, W) * mask
            out = x0 + noise
        else:
            out = x0 + noise.reshape(B, C, H, W)

        out = out.to(x.dtype) if x.dtype in (torch.float16, torch.float32, torch.float64) else out
        return out.squeeze(0) if squeeze else out


class HFSingleCosInjectSeq1D:
    """
    Add a *single* cosine Fourier mode along the flattened axis.
    This is the closest to "Fourier mode perturbation" setups.

    Args:
      rho: in [0,1], maps to rFFT bin m ≈ rho*(L//2), i.e. Ω ≈ rho*π.
      eps: absolute time-domain amplitude (after Normalize).
      eps_rel: if not None, eps = eps_rel * rms(x) per-sample.
      phase: fixed phase (ignored if random_phase=True)
      random_phase: if True, sample phase ~ Uniform[0,2π) per-sample.
      p: probability to apply per sample.
      all_channels: if True, add to all channels; else to channel 0 only.
    """
    def __init__(
        self,
        rho: float = 0.97,
        eps: float = 0.2,
        eps_rel: Optional[float] = None,
        phase: float = 0.0,
        random_phase: bool = False,
        p: float = 1.0,
        all_channels: bool = True,
    ):
        if not (0.0 <= rho <= 1.0):
            raise ValueError(f"rho must be in [0,1], got {rho}")
        self.rho = float(rho)
        self.eps = float(eps)
        self.eps_rel = None if eps_rel is None else float(eps_rel)
        self.phase = float(phase)
        self.random_phase = bool(random_phase)
        self.p = float(p)
        self.all_channels = bool(all_channels)

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        if x.ndim != 4:
            raise ValueError(f"Expected x shape (B,C,H,W) or (C,H,W), got {tuple(x.shape)}")

        B, C, H, W = x.shape
        L = H * W
        # choose DFT bin (avoid DC and exact Nyquist)
        m = int(round(self.rho * (L // 2)))
        m = max(1, min(m, (L // 2) - 1))

        x0 = x.to(torch.float32)

        # per-sample apply mask
        if self.p < 1.0:
            apply = (torch.rand((B,), device=x0.device) < self.p)
        else:
            apply = torch.ones((B,), device=x0.device, dtype=torch.bool)

        # eps per sample
        if self.eps_rel is not None:
            x_rms = x0.reshape(B, -1).pow(2).mean(dim=-1).sqrt()  # (B,)
            eps_b = (self.eps_rel * x_rms)                        # (B,)
        else:
            eps_b = torch.full((B,), self.eps, device=x0.device, dtype=x0.dtype)

        # phases per sample
        if self.random_phase:
            phi = torch.rand((B,), device=x0.device, dtype=x0.dtype) * (2.0 * math.pi)
        else:
            phi = torch.full((B,), self.phase, device=x0.device, dtype=x0.dtype)

        n = torch.arange(L, device=x0.device, dtype=x0.dtype)  # (L,)
        # s: (B,L)
        s = torch.cos(2.0 * math.pi * m * n / float(L) + phi[:, None]) * eps_b[:, None]
        s = s.reshape(B, 1, H, W)  # broadcast over channels

        out = x0.clone()
        if self.all_channels:
            out[apply] = out[apply] + s[apply]
        else:
            out[apply, 0:1] = out[apply, 0:1] + s[apply]

        out = out.to(x.dtype) if x.dtype in (torch.float16, torch.float32, torch.float64) else out
        return out.squeeze(0) if squeeze else out


# -------------------------
# Model
# -------------------------
class S4Model(nn.Module):
    def __init__(self, d_input, d_output, d_model, n_layers, dropout, prenorm, lr_s4):
        super().__init__()
        self.prenorm = prenorm

        self.encoder = nn.Linear(d_input, d_model)

        self.s4_layers = nn.ModuleList(
            [S4D(d_model, dropout=dropout, transposed=True, lr=lr_s4) for _ in range(n_layers)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(n_layers)])

        self.decoder = nn.Linear(d_model, d_output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, d_input)
        x = self.encoder(x)             # (B, L, d_model)
        x = x.transpose(-1, -2)         # (B, d_model, L)

        for layer, norm, drop in zip(self.s4_layers, self.norms, self.dropouts):
            z = x
            if self.prenorm:
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)

            z, _ = layer(z)
            z = drop(z)

            x = x + z

            if not self.prenorm:
                x = norm(x.transpose(-1, -2)).transpose(-1, -2)

        x = x.transpose(-1, -2)         # (B, L, d_model)
        x = x.mean(dim=1)               # (B, d_model) mean pool
        return self.decoder(x)          # (B, d_output)


def setup_optimizer(model: nn.Module, lr: float, weight_decay: float, epochs: int):
    """
    Keep S4(D) special param groups via _optim.
    """
    all_params = list(model.parameters())
    base_params = [p for p in all_params if not hasattr(p, "_optim")]
    opt = optim.AdamW(base_params, lr=lr, weight_decay=weight_decay)

    # collect unique hyperparam dicts from S4 params
    hps = [getattr(p, "_optim") for p in all_params if hasattr(p, "_optim")]
    hps = [dict(s) for s in sorted(list(dict.fromkeys(frozenset(hp.items()) for hp in hps)))]
    for hp in hps:
        params = [p for p in all_params if getattr(p, "_optim", None) == hp]
        opt.add_param_group({"params": params, **hp})

    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    return opt, sch


# -------------------------
# Train / Eval
# -------------------------
@dataclass
class Metrics:
    loss: float
    acc: float

@dataclass
class PairMetrics:
    clean: Metrics
    hf: Metrics
    drop: float
    flip_rate: float
    logit_l2: float
    delta_rms: float
    delta_rel: float


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion,
    optimizer,
    device: str,
    injector=None,
):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0

    for x_img, y in loader:
        x_img = x_img.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if injector is not None:
            x_img = injector(x_img)

        x = flatten_to_seq(x_img)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        bs = y.size(0)
        loss_sum += float(loss.item()) * bs
        total += bs
        correct += int((logits.argmax(dim=1) == y).sum().item())

    return Metrics(loss=loss_sum / total, acc=100.0 * correct / total)


@torch.no_grad()
def eval_clean_and_hf(
    model: nn.Module,
    loader: DataLoader,
    criterion,
    device: str,
    injector=None,
) -> PairMetrics:
    model.eval()

    loss_c, corr_c, loss_h, corr_h, total = 0.0, 0, 0.0, 0, 0
    flip_sum = 0
    logit_l2_sum = 0.0
    delta_rms_sum = 0.0
    x_rms_sum = 0.0

    for x_img, y in loader:
        x_img = x_img.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        x = flatten_to_seq(x_img)
        logits_c = model(x)
        l_c = criterion(logits_c, y)

        bs = y.size(0)
        total += bs
        loss_c += float(l_c.item()) * bs
        corr_c += int((logits_c.argmax(dim=1) == y).sum().item())

        if injector is not None:
            x_hf_img = injector(x_img)
            x_hf = flatten_to_seq(x_hf_img)
            logits_h = model(x_hf)
            l_h = criterion(logits_h, y)

            loss_h += float(l_h.item()) * bs
            corr_h += int((logits_h.argmax(dim=1) == y).sum().item())

            pred_c = logits_c.argmax(dim=1)
            pred_h = logits_h.argmax(dim=1)
            flip_sum += int((pred_c != pred_h).sum().item())

            # mean L2 over batch
            diff = logits_h - logits_c
            logit_l2_sum += float(diff.pow(2).sum(dim=1).sqrt().sum().item())

            d = (x_hf_img - x_img).reshape(bs, -1)
            delta_rms_sum += float(d.pow(2).mean(dim=1).sqrt().sum().item())
            xr = x_img.reshape(bs, -1).pow(2).mean(dim=1).sqrt()
            x_rms_sum += float(xr.sum().item())

    clean = Metrics(loss=loss_c / total, acc=100.0 * corr_c / total)

    if injector is None:
        # dummy values
        hf = Metrics(loss=float("nan"), acc=float("nan"))
        return PairMetrics(
            clean=clean, hf=hf, drop=float("nan"),
            flip_rate=float("nan"), logit_l2=float("nan"),
            delta_rms=float("nan"), delta_rel=float("nan"),
        )

    hf = Metrics(loss=loss_h / total, acc=100.0 * corr_h / total)
    drop = clean.acc - hf.acc
    flip_rate = 100.0 * flip_sum / total
    logit_l2 = logit_l2_sum / total
    delta_rms = delta_rms_sum / total
    x_rms = x_rms_sum / total
    delta_rel = delta_rms / (x_rms + 1e-12)

    return PairMetrics(
        clean=clean, hf=hf, drop=drop,
        flip_rate=flip_rate, logit_l2=logit_l2,
        delta_rms=delta_rms, delta_rel=delta_rel,
    )


# -------------------------
# CLI / Main
# -------------------------
def build_injector(args):
    if not (args.hf_train or args.hf_test):
        return None

    if args.hf_kind == "band_rms":
        if args.hf_rms_rel is None and args.hf_rms <= 0:
            raise ValueError("--hf_kind band_rms requires --hf_rms > 0 or --hf_rms_rel")
        return HFBandNoiseTimeRMS(
            band=(args.hf_band_lo, args.hf_band_hi),
            rms=args.hf_rms,
            rms_rel=args.hf_rms_rel,
            p=args.hf_p,
        )

    if args.hf_kind == "single_cos":
        if args.hf_eps_rel is None and args.hf_eps == 0:
            raise ValueError("--hf_kind single_cos requires --hf_eps != 0 or --hf_eps_rel")
        return HFSingleCosInjectSeq1D(
            rho=args.hf_rho,
            eps=args.hf_eps,
            eps_rel=args.hf_eps_rel,
            phase=args.hf_phase,
            random_phase=args.hf_random_phase,
            p=args.hf_p,
            all_channels=args.hf_all_channels,
        )

    raise ValueError(f"Unknown hf_kind: {args.hf_kind}")


def parse_float_list(s: str) -> List[float]:
    s = s.strip()
    if not s:
        return []
    return [float(x) for x in s.split(",")]


def main():
    p = argparse.ArgumentParser()

    # Core
    p.add_argument("--dataset", choices=["cifar10", "mnist"], default="cifar10")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)

    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--weight_decay", type=float, default=0.01)

    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--prenorm", action="store_true")
    p.add_argument("--seed", type=int, default=0)

    # HF control
    p.add_argument("--hf_train", action="store_true", help="apply HF perturbation during training")
    p.add_argument("--hf_test", action="store_true", help="evaluate on HF-perturbed inputs (paired with clean)")

    p.add_argument("--hf_kind", choices=["band_rms", "single_cos"], default="single_cos")

    # band_rms params
    p.add_argument("--hf_band_lo", type=float, default=0.9)
    p.add_argument("--hf_band_hi", type=float, default=1.0)
    p.add_argument("--hf_rms", type=float, default=0.2, help="time-domain RMS (absolute)")
    p.add_argument("--hf_rms_rel", type=float, default=None, help="time-domain RMS relative to input RMS (e.g. 0.1)")

    # single_cos params
    p.add_argument("--hf_rho", type=float, default=0.97, help="normalized freq rho in [0,1], Ω≈rho*π")
    p.add_argument("--hf_eps", type=float, default=0.2, help="cosine amplitude (absolute)")
    p.add_argument("--hf_eps_rel", type=float, default=None, help="cosine amplitude relative to input RMS (e.g. 0.1)")
    p.add_argument("--hf_phase", type=float, default=0.0)
    p.add_argument("--hf_random_phase", action="store_true")
    p.add_argument("--hf_all_channels", action="store_true", help="(single_cos) add to all channels (default True)")
    p.set_defaults(hf_all_channels=True)

    p.add_argument("--hf_p", type=float, default=1.0, help="probability apply per sample")

    # Sweep (useful to find sensitive frequencies)
    p.add_argument("--sweep_rho", type=str, default="", help="comma-separated rhos for single_cos sweep after training")
    p.add_argument("--sweep_eps_rel", type=float, default=None, help="override eps_rel during sweep (optional)")
    p.add_argument("--sweep_eps", type=float, default=None, help="override eps during sweep (optional)")

    args = p.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        cudnn.benchmark = True

    # Dataset
    if args.dataset == "cifar10":
        c = 3
        base = T.Compose([
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = torchvision.datasets.CIFAR10("./data/cifar", train=True, download=True, transform=base)
        testset = torchvision.datasets.CIFAR10("./data/cifar", train=False, download=True, transform=base)
        d_input, d_output = c, 10
    else:
        c = 1
        base = T.Compose([
            T.ToTensor(),
            T.Normalize((0.1307,), (0.3081,)),
        ])
        trainset = torchvision.datasets.MNIST("./data/mnist", train=True, download=True, transform=base)
        testset = torchvision.datasets.MNIST("./data/mnist", train=False, download=True, transform=base)
        d_input, d_output = c, 10

    trainloader = DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )
    testloader = DataLoader(
        testset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )

    # Model / optim
    lr_s4 = min(1e-3, args.lr)
    model = S4Model(
        d_input=d_input,
        d_output=d_output,
        d_model=args.d_model,
        n_layers=args.n_layers,
        dropout=args.dropout,
        prenorm=args.prenorm,
        lr_s4=lr_s4,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer, scheduler = setup_optimizer(model, args.lr, args.weight_decay, args.epochs)

    # Build injectors
    inj = build_injector(args)
    inj_train = inj if args.hf_train else None
    inj_test = inj if args.hf_test else None

    # Train
    best_clean = -1.0
    best_hf = -1.0
    best_drop = 0.0

    for epoch in range(args.epochs):
        tr = train_epoch(model, trainloader, criterion, optimizer, device, injector=inj_train)
        te = eval_clean_and_hf(model, testloader, criterion, device, injector=inj_test)

        scheduler.step()

        if inj_test is None:
            print(
                f"Epoch {epoch:03d} | "
                f"train loss {tr.loss:.4f} acc {tr.acc:6.2f}% | "
                f"test  loss {te.clean.loss:.4f} acc {te.clean.acc:6.2f}%"
            )
            best_clean = max(best_clean, te.clean.acc)
        else:
            print(
                f"Epoch {epoch:03d} | "
                f"train {tr.loss:.4f} {tr.acc:6.2f}% | "
                f"test(clean) {te.clean.loss:.4f} {te.clean.acc:6.2f}% | "
                f"test(HF) {te.hf.loss:.4f} {te.hf.acc:6.2f}% | "
                f"drop {te.drop:+.2f}% | "
                f"flip {te.flip_rate:5.2f}% | "
                f"logitL2 {te.logit_l2:7.4f} | "
                f"Δx_rms {te.delta_rms:7.4f} (rel {te.delta_rel:6.3f})"
            )

            if te.clean.acc > best_clean:
                best_clean = te.clean.acc
            if te.hf.acc > best_hf:
                best_hf = te.hf.acc
            best_drop = max(best_drop, te.drop)

    if inj_test is None:
        print(f"[best] test(clean) acc {best_clean:.2f}%")
    else:
        print(f"[best] test(clean) acc {best_clean:.2f}% | test(HF) acc {best_hf:.2f}% | max drop {best_drop:+.2f}%")

    # Optional: sweep rho after training (single_cos only)
    rhos = parse_float_list(args.sweep_rho)
    if rhos:
        if args.hf_kind != "single_cos":
            print("[sweep] sweep_rho is only supported for --hf_kind single_cos (ignoring).")
            return

        print("\n[sweep] Evaluating rho sweep (clean-train; test-only perturbation)")
        # build a base injector template but override rho/eps if requested
        base_eps_rel = args.hf_eps_rel
        base_eps = args.hf_eps
        if args.sweep_eps_rel is not None:
            base_eps_rel = args.sweep_eps_rel
        if args.sweep_eps is not None:
            base_eps = args.sweep_eps

        for rho in rhos:
            inj_sweep = HFSingleCosInjectSeq1D(
                rho=rho,
                eps=base_eps,
                eps_rel=base_eps_rel,
                phase=args.hf_phase,
                random_phase=args.hf_random_phase,
                p=1.0,  # always apply for sweep
                all_channels=args.hf_all_channels,
            )
            te = eval_clean_and_hf(model, testloader, criterion, device, injector=inj_sweep)
            print(
                f"rho={rho:5.3f} | clean {te.clean.acc:6.2f}% | HF {te.hf.acc:6.2f}% | "
                f"drop {te.drop:+6.2f}% | flip {te.flip_rate:6.2f}% | relΔ {te.delta_rel:6.3f}"
            )


if __name__ == "__main__":
=======
"""
train_s4d_hf.py

Minimal S4D training on sequential CIFAR10 / MNIST with optional high-frequency (HF) perturbations.

Key points:
- Datasets return images (C,H,W) after ToTensor+Normalize (no flatten in dataset).
- We flatten inside the train/eval step: (B,C,H,W) -> (B,L,C), L=H*W.
- HF perturbations are applied on (B,C,H,W) BEFORE flatten, i.e. along the same sequence axis H*W.
- Evaluation (when --hf_test) reports BOTH clean and HF metrics *on the same samples*,
  plus sensitivity metrics (flip-rate, logit L2 diff, input delta RMS).

Requires the S4 repo structure:
  from models.s4.s4d import S4D
"""

import argparse
import math
import random
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

from models.s4.s4d import S4D


# -------------------------
# Reproducibility
# -------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -------------------------
# Helpers
# -------------------------
def flatten_to_seq(x_img: torch.Tensor) -> torch.Tensor:
    """
    (B,C,H,W) -> (B,L,C) where L=H*W
    """
    if x_img.ndim != 4:
        raise ValueError(f"Expected (B,C,H,W), got {tuple(x_img.shape)}")
    B, C, H, W = x_img.shape
    return x_img.reshape(B, C, H * W).transpose(1, 2).contiguous()


def _band_to_bins(band: Tuple[float, float], F: int) -> Tuple[int, int]:
    """band fraction of Nyquist -> rFFT bin indices [0, F-1]"""
    lo_f, hi_f = band
    lo = int(round(lo_f * (F - 1)))
    hi = int(round(hi_f * (F - 1)))
    lo = max(0, min(lo, F - 1))
    hi = max(0, min(hi, F - 1))
    return lo, hi


# -------------------------
# HF perturbations
# -------------------------
class HFBandNoiseTimeRMS:
    """
    Add band-limited *random* noise whose TIME-DOMAIN RMS is controlled.

    This avoids the common pitfall where you set a frequency-domain sigma,
    but the iFFT scaling makes the actual time-domain perturbation tiny.

    Args:
      band: (lo, hi) as fraction of Nyquist in [0,1]. (0.9,1.0) == 0.9π~π.
      rms: absolute time-domain RMS to add (after Normalize scale).
      rms_rel: if not None, rms = rms_rel * rms(x) per-sample.
      p: probability to apply per sample.
    """
    def __init__(
        self,
        band: Tuple[float, float] = (0.9, 1.0),
        rms: float = 0.2,
        rms_rel: Optional[float] = None,
        p: float = 1.0,
        eps: float = 1e-12,
    ):
        lo, hi = band
        if not (0.0 <= lo <= hi <= 1.0):
            raise ValueError(f"band must be within [0,1] and lo<=hi, got {band}")
        self.band = (float(lo), float(hi))
        self.rms = float(rms)
        self.rms_rel = None if rms_rel is None else float(rms_rel)
        self.p = float(p)
        self.eps = float(eps)

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,H,W) or (C,H,W)
        if x.ndim == 3:
            x = x.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        if x.ndim != 4:
            raise ValueError(f"Expected x shape (B,C,H,W) or (C,H,W), got {tuple(x.shape)}")

        B, C, H, W = x.shape
        L = H * W
        F = L // 2 + 1

        x0 = x.to(torch.float32)

        # per-sample apply mask
        if self.p < 1.0:
            mask = (torch.rand((B, 1, 1, 1), device=x0.device) < self.p).to(x0.dtype)
        else:
            mask = None

        # determine target RMS per sample (and broadcast)
        if self.rms_rel is not None:
            # per-sample rms over all dims
            x_rms = x0.reshape(B, -1).pow(2).mean(dim=-1).sqrt()  # (B,)
            target = (self.rms_rel * x_rms).clamp_min(0.0)        # (B,)
        else:
            target = torch.full((B,), self.rms, device=x0.device, dtype=x0.dtype)

        # build complex spectrum noise N: (B,C,F)
        N = torch.zeros((B, C, F), dtype=torch.complex64, device=x0.device)
        lo, hi = _band_to_bins(self.band, F)
        if hi < lo:
            return x if not squeeze else x.squeeze(0)

        K = hi - lo + 1
        nr = torch.randn((B, C, K), device=x0.device, dtype=torch.float32)
        ni = torch.randn((B, C, K), device=x0.device, dtype=torch.float32)
        N[:, :, lo:hi + 1] = torch.complex(nr, ni)

        # enforce real constraints
        N[:, :, 0] = torch.complex(N[:, :, 0].real, torch.zeros_like(N[:, :, 0].real))
        if L % 2 == 0:
            N[:, :, -1] = torch.complex(N[:, :, -1].real, torch.zeros_like(N[:, :, -1].real))

        noise = torch.fft.irfft(N, n=L, dim=-1)  # (B,C,L), real
        # scale to target RMS per (B,C) (use per-channel rms, but target is per-sample)
        cur = noise.pow(2).mean(dim=-1, keepdim=True).sqrt()  # (B,C,1)
        # target: (B,) -> (B,1,1)
        tgt = target.view(B, 1, 1)
        noise = noise * (tgt / (cur + self.eps))

        if mask is not None:
            noise = noise.reshape(B, C, H, W) * mask
            out = x0 + noise
        else:
            out = x0 + noise.reshape(B, C, H, W)

        out = out.to(x.dtype) if x.dtype in (torch.float16, torch.float32, torch.float64) else out
        return out.squeeze(0) if squeeze else out


class HFSingleCosInjectSeq1D:
    """
    Add a *single* cosine Fourier mode along the flattened axis.
    This is the closest to "Fourier mode perturbation" setups.

    Args:
      rho: in [0,1], maps to rFFT bin m ≈ rho*(L//2), i.e. Ω ≈ rho*π.
      eps: absolute time-domain amplitude (after Normalize).
      eps_rel: if not None, eps = eps_rel * rms(x) per-sample.
      phase: fixed phase (ignored if random_phase=True)
      random_phase: if True, sample phase ~ Uniform[0,2π) per-sample.
      p: probability to apply per sample.
      all_channels: if True, add to all channels; else to channel 0 only.
    """
    def __init__(
        self,
        rho: float = 0.97,
        eps: float = 0.2,
        eps_rel: Optional[float] = None,
        phase: float = 0.0,
        random_phase: bool = False,
        p: float = 1.0,
        all_channels: bool = True,
    ):
        if not (0.0 <= rho <= 1.0):
            raise ValueError(f"rho must be in [0,1], got {rho}")
        self.rho = float(rho)
        self.eps = float(eps)
        self.eps_rel = None if eps_rel is None else float(eps_rel)
        self.phase = float(phase)
        self.random_phase = bool(random_phase)
        self.p = float(p)
        self.all_channels = bool(all_channels)

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        if x.ndim != 4:
            raise ValueError(f"Expected x shape (B,C,H,W) or (C,H,W), got {tuple(x.shape)}")

        B, C, H, W = x.shape
        L = H * W
        # choose DFT bin (avoid DC and exact Nyquist)
        m = int(round(self.rho * (L // 2)))
        m = max(1, min(m, (L // 2) - 1))

        x0 = x.to(torch.float32)

        # per-sample apply mask
        if self.p < 1.0:
            apply = (torch.rand((B,), device=x0.device) < self.p)
        else:
            apply = torch.ones((B,), device=x0.device, dtype=torch.bool)

        # eps per sample
        if self.eps_rel is not None:
            x_rms = x0.reshape(B, -1).pow(2).mean(dim=-1).sqrt()  # (B,)
            eps_b = (self.eps_rel * x_rms)                        # (B,)
        else:
            eps_b = torch.full((B,), self.eps, device=x0.device, dtype=x0.dtype)

        # phases per sample
        if self.random_phase:
            phi = torch.rand((B,), device=x0.device, dtype=x0.dtype) * (2.0 * math.pi)
        else:
            phi = torch.full((B,), self.phase, device=x0.device, dtype=x0.dtype)

        n = torch.arange(L, device=x0.device, dtype=x0.dtype)  # (L,)
        # s: (B,L)
        s = torch.cos(2.0 * math.pi * m * n / float(L) + phi[:, None]) * eps_b[:, None]
        s = s.reshape(B, 1, H, W)  # broadcast over channels

        out = x0.clone()
        if self.all_channels:
            out[apply] = out[apply] + s[apply]
        else:
            out[apply, 0:1] = out[apply, 0:1] + s[apply]

        out = out.to(x.dtype) if x.dtype in (torch.float16, torch.float32, torch.float64) else out
        return out.squeeze(0) if squeeze else out


# -------------------------
# Model
# -------------------------
class S4Model(nn.Module):
    def __init__(self, d_input, d_output, d_model, n_layers, dropout, prenorm, lr_s4):
        super().__init__()
        self.prenorm = prenorm

        self.encoder = nn.Linear(d_input, d_model)

        self.s4_layers = nn.ModuleList(
            [S4D(d_model, dropout=dropout, transposed=True, lr=lr_s4) for _ in range(n_layers)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(n_layers)])

        self.decoder = nn.Linear(d_model, d_output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, d_input)
        x = self.encoder(x)             # (B, L, d_model)
        x = x.transpose(-1, -2)         # (B, d_model, L)

        for layer, norm, drop in zip(self.s4_layers, self.norms, self.dropouts):
            z = x
            if self.prenorm:
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)

            z, _ = layer(z)
            z = drop(z)

            x = x + z

            if not self.prenorm:
                x = norm(x.transpose(-1, -2)).transpose(-1, -2)

        x = x.transpose(-1, -2)         # (B, L, d_model)
        x = x.mean(dim=1)               # (B, d_model) mean pool
        return self.decoder(x)          # (B, d_output)


def setup_optimizer(model: nn.Module, lr: float, weight_decay: float, epochs: int):
    """
    Keep S4(D) special param groups via _optim.
    """
    all_params = list(model.parameters())
    base_params = [p for p in all_params if not hasattr(p, "_optim")]
    opt = optim.AdamW(base_params, lr=lr, weight_decay=weight_decay)

    # collect unique hyperparam dicts from S4 params
    hps = [getattr(p, "_optim") for p in all_params if hasattr(p, "_optim")]
    hps = [dict(s) for s in sorted(list(dict.fromkeys(frozenset(hp.items()) for hp in hps)))]
    for hp in hps:
        params = [p for p in all_params if getattr(p, "_optim", None) == hp]
        opt.add_param_group({"params": params, **hp})

    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    return opt, sch


# -------------------------
# Train / Eval
# -------------------------
@dataclass
class Metrics:
    loss: float
    acc: float

@dataclass
class PairMetrics:
    clean: Metrics
    hf: Metrics
    drop: float
    flip_rate: float
    logit_l2: float
    delta_rms: float
    delta_rel: float


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion,
    optimizer,
    device: str,
    injector=None,
):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0

    for x_img, y in loader:
        x_img = x_img.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if injector is not None:
            x_img = injector(x_img)

        x = flatten_to_seq(x_img)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        bs = y.size(0)
        loss_sum += float(loss.item()) * bs
        total += bs
        correct += int((logits.argmax(dim=1) == y).sum().item())

    return Metrics(loss=loss_sum / total, acc=100.0 * correct / total)


@torch.no_grad()
def eval_clean_and_hf(
    model: nn.Module,
    loader: DataLoader,
    criterion,
    device: str,
    injector=None,
) -> PairMetrics:
    model.eval()

    loss_c, corr_c, loss_h, corr_h, total = 0.0, 0, 0.0, 0, 0
    flip_sum = 0
    logit_l2_sum = 0.0
    delta_rms_sum = 0.0
    x_rms_sum = 0.0

    for x_img, y in loader:
        x_img = x_img.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        x = flatten_to_seq(x_img)
        logits_c = model(x)
        l_c = criterion(logits_c, y)

        bs = y.size(0)
        total += bs
        loss_c += float(l_c.item()) * bs
        corr_c += int((logits_c.argmax(dim=1) == y).sum().item())

        if injector is not None:
            x_hf_img = injector(x_img)
            x_hf = flatten_to_seq(x_hf_img)
            logits_h = model(x_hf)
            l_h = criterion(logits_h, y)

            loss_h += float(l_h.item()) * bs
            corr_h += int((logits_h.argmax(dim=1) == y).sum().item())

            pred_c = logits_c.argmax(dim=1)
            pred_h = logits_h.argmax(dim=1)
            flip_sum += int((pred_c != pred_h).sum().item())

            # mean L2 over batch
            diff = logits_h - logits_c
            logit_l2_sum += float(diff.pow(2).sum(dim=1).sqrt().sum().item())

            d = (x_hf_img - x_img).reshape(bs, -1)
            delta_rms_sum += float(d.pow(2).mean(dim=1).sqrt().sum().item())
            xr = x_img.reshape(bs, -1).pow(2).mean(dim=1).sqrt()
            x_rms_sum += float(xr.sum().item())

    clean = Metrics(loss=loss_c / total, acc=100.0 * corr_c / total)

    if injector is None:
        # dummy values
        hf = Metrics(loss=float("nan"), acc=float("nan"))
        return PairMetrics(
            clean=clean, hf=hf, drop=float("nan"),
            flip_rate=float("nan"), logit_l2=float("nan"),
            delta_rms=float("nan"), delta_rel=float("nan"),
        )

    hf = Metrics(loss=loss_h / total, acc=100.0 * corr_h / total)
    drop = clean.acc - hf.acc
    flip_rate = 100.0 * flip_sum / total
    logit_l2 = logit_l2_sum / total
    delta_rms = delta_rms_sum / total
    x_rms = x_rms_sum / total
    delta_rel = delta_rms / (x_rms + 1e-12)

    return PairMetrics(
        clean=clean, hf=hf, drop=drop,
        flip_rate=flip_rate, logit_l2=logit_l2,
        delta_rms=delta_rms, delta_rel=delta_rel,
    )


# -------------------------
# CLI / Main
# -------------------------
def build_injector(args):
    if not (args.hf_train or args.hf_test):
        return None

    if args.hf_kind == "band_rms":
        if args.hf_rms_rel is None and args.hf_rms <= 0:
            raise ValueError("--hf_kind band_rms requires --hf_rms > 0 or --hf_rms_rel")
        return HFBandNoiseTimeRMS(
            band=(args.hf_band_lo, args.hf_band_hi),
            rms=args.hf_rms,
            rms_rel=args.hf_rms_rel,
            p=args.hf_p,
        )

    if args.hf_kind == "single_cos":
        if args.hf_eps_rel is None and args.hf_eps == 0:
            raise ValueError("--hf_kind single_cos requires --hf_eps != 0 or --hf_eps_rel")
        return HFSingleCosInjectSeq1D(
            rho=args.hf_rho,
            eps=args.hf_eps,
            eps_rel=args.hf_eps_rel,
            phase=args.hf_phase,
            random_phase=args.hf_random_phase,
            p=args.hf_p,
            all_channels=args.hf_all_channels,
        )

    raise ValueError(f"Unknown hf_kind: {args.hf_kind}")


def parse_float_list(s: str) -> List[float]:
    s = s.strip()
    if not s:
        return []
    return [float(x) for x in s.split(",")]


def main():
    p = argparse.ArgumentParser()

    # Core
    p.add_argument("--dataset", choices=["cifar10", "mnist"], default="cifar10")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)

    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--weight_decay", type=float, default=0.01)

    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--prenorm", action="store_true")
    p.add_argument("--seed", type=int, default=0)

    # HF control
    p.add_argument("--hf_train", action="store_true", help="apply HF perturbation during training")
    p.add_argument("--hf_test", action="store_true", help="evaluate on HF-perturbed inputs (paired with clean)")

    p.add_argument("--hf_kind", choices=["band_rms", "single_cos"], default="single_cos")

    # band_rms params
    p.add_argument("--hf_band_lo", type=float, default=0.9)
    p.add_argument("--hf_band_hi", type=float, default=1.0)
    p.add_argument("--hf_rms", type=float, default=0.2, help="time-domain RMS (absolute)")
    p.add_argument("--hf_rms_rel", type=float, default=None, help="time-domain RMS relative to input RMS (e.g. 0.1)")

    # single_cos params
    p.add_argument("--hf_rho", type=float, default=0.97, help="normalized freq rho in [0,1], Ω≈rho*π")
    p.add_argument("--hf_eps", type=float, default=0.2, help="cosine amplitude (absolute)")
    p.add_argument("--hf_eps_rel", type=float, default=None, help="cosine amplitude relative to input RMS (e.g. 0.1)")
    p.add_argument("--hf_phase", type=float, default=0.0)
    p.add_argument("--hf_random_phase", action="store_true")
    p.add_argument("--hf_all_channels", action="store_true", help="(single_cos) add to all channels (default True)")
    p.set_defaults(hf_all_channels=True)

    p.add_argument("--hf_p", type=float, default=1.0, help="probability apply per sample")

    # Sweep (useful to find sensitive frequencies)
    p.add_argument("--sweep_rho", type=str, default="", help="comma-separated rhos for single_cos sweep after training")
    p.add_argument("--sweep_eps_rel", type=float, default=None, help="override eps_rel during sweep (optional)")
    p.add_argument("--sweep_eps", type=float, default=None, help="override eps during sweep (optional)")

    args = p.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        cudnn.benchmark = True

    # Dataset
    if args.dataset == "cifar10":
        c = 3
        base = T.Compose([
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = torchvision.datasets.CIFAR10("./data/cifar", train=True, download=True, transform=base)
        testset = torchvision.datasets.CIFAR10("./data/cifar", train=False, download=True, transform=base)
        d_input, d_output = c, 10
    else:
        c = 1
        base = T.Compose([
            T.ToTensor(),
            T.Normalize((0.1307,), (0.3081,)),
        ])
        trainset = torchvision.datasets.MNIST("./data/mnist", train=True, download=True, transform=base)
        testset = torchvision.datasets.MNIST("./data/mnist", train=False, download=True, transform=base)
        d_input, d_output = c, 10

    trainloader = DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )
    testloader = DataLoader(
        testset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )

    # Model / optim
    lr_s4 = min(1e-3, args.lr)
    model = S4Model(
        d_input=d_input,
        d_output=d_output,
        d_model=args.d_model,
        n_layers=args.n_layers,
        dropout=args.dropout,
        prenorm=args.prenorm,
        lr_s4=lr_s4,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer, scheduler = setup_optimizer(model, args.lr, args.weight_decay, args.epochs)

    # Build injectors
    inj = build_injector(args)
    inj_train = inj if args.hf_train else None
    inj_test = inj if args.hf_test else None

    # Train
    best_clean = -1.0
    best_hf = -1.0
    best_drop = 0.0

    for epoch in range(args.epochs):
        tr = train_epoch(model, trainloader, criterion, optimizer, device, injector=inj_train)
        te = eval_clean_and_hf(model, testloader, criterion, device, injector=inj_test)

        scheduler.step()

        if inj_test is None:
            print(
                f"Epoch {epoch:03d} | "
                f"train loss {tr.loss:.4f} acc {tr.acc:6.2f}% | "
                f"test  loss {te.clean.loss:.4f} acc {te.clean.acc:6.2f}%"
            )
            best_clean = max(best_clean, te.clean.acc)
        else:
            print(
                f"Epoch {epoch:03d} | "
                f"train {tr.loss:.4f} {tr.acc:6.2f}% | "
                f"test(clean) {te.clean.loss:.4f} {te.clean.acc:6.2f}% | "
                f"test(HF) {te.hf.loss:.4f} {te.hf.acc:6.2f}% | "
                f"drop {te.drop:+.2f}% | "
                f"flip {te.flip_rate:5.2f}% | "
                f"logitL2 {te.logit_l2:7.4f} | "
                f"Δx_rms {te.delta_rms:7.4f} (rel {te.delta_rel:6.3f})"
            )

            if te.clean.acc > best_clean:
                best_clean = te.clean.acc
            if te.hf.acc > best_hf:
                best_hf = te.hf.acc
            best_drop = max(best_drop, te.drop)

    if inj_test is None:
        print(f"[best] test(clean) acc {best_clean:.2f}%")
    else:
        print(f"[best] test(clean) acc {best_clean:.2f}% | test(HF) acc {best_hf:.2f}% | max drop {best_drop:+.2f}%")

    # Optional: sweep rho after training (single_cos only)
    rhos = parse_float_list(args.sweep_rho)
    if rhos:
        if args.hf_kind != "single_cos":
            print("[sweep] sweep_rho is only supported for --hf_kind single_cos (ignoring).")
            return

        print("\n[sweep] Evaluating rho sweep (clean-train; test-only perturbation)")
        # build a base injector template but override rho/eps if requested
        base_eps_rel = args.hf_eps_rel
        base_eps = args.hf_eps
        if args.sweep_eps_rel is not None:
            base_eps_rel = args.sweep_eps_rel
        if args.sweep_eps is not None:
            base_eps = args.sweep_eps

        for rho in rhos:
            inj_sweep = HFSingleCosInjectSeq1D(
                rho=rho,
                eps=base_eps,
                eps_rel=base_eps_rel,
                phase=args.hf_phase,
                random_phase=args.hf_random_phase,
                p=1.0,  # always apply for sweep
                all_channels=args.hf_all_channels,
            )
            te = eval_clean_and_hf(model, testloader, criterion, device, injector=inj_sweep)
            print(
                f"rho={rho:5.3f} | clean {te.clean.acc:6.2f}% | HF {te.hf.acc:6.2f}% | "
                f"drop {te.drop:+6.2f}% | flip {te.flip_rate:6.2f}% | relΔ {te.delta_rel:6.3f}"
            )


if __name__ == "__main__":
>>>>>>> 2a5624c (Add project fiels)
    main()