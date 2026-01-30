"""
paper_spectral_grid_bench.py

Paper-ready spectral sensitivity benchmark for S4D and baselines.

What it does:
- Train a model on clean data (seq-CIFAR10 or seq-MNIST setting).
- Evaluate test-only perturbation by adding a single cosine Fourier mode at frequency rho (0..1, 1->Nyquist).
- Supports:
  * 1D rho sweep at fixed perturbation strength (relΔ or eps)
  * 2D grid sweep: rho x pert_rel (relΔ) -> heatmap saved

Models:
- s4d          : S4D sequence model (LTI-ish conv kernel)
- cnn1d        : simple 1D CNN over sequence (O(L))
- transformer  : vanilla TransformerEncoder over sequence (O(L^2)) (may require smaller batch size)
- resnet18     : ResNet18 modified for CIFAR (conv1 3x3 stride1, no maxpool) on image input

Important:
- To keep perturbation consistent across models, perturbation is defined on the flattened sequence axis (H*W)
  and reshaped back to image for image-based models.
- Clean outputs are cached to compute flip/logitΔ/marginΔ.

Run from the state-spaces/s4 repo root (or adjust PYTHONPATH) so that:
  from models.s4.s4d import S4D
works.

Example commands:

(1) Train S4D + 1D rho sweep (fixed relΔ):
  python paper_spectral_grid_bench.py --dataset cifar10 --model s4d --epochs 80 \
    --sweep --rho_values 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.97,0.99 \
    --pert_rel 0.05 --phase_mode sample --n_phase 5 \
    --out_dir runs/cifar_s4d --name s4d_rel005 --save_plot

(2) Train S4D + 2D grid sweep (rho x pert_rel) + heatmap:
  python paper_spectral_grid_bench.py --dataset cifar10 --model s4d --epochs 80 \
    --sweep_grid \
    --rho_values 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.97,0.99 \
    --pert_rel_values 0.01,0.03,0.05,0.10 \
    --phase_mode sample --n_phase 3 \
    --out_dir runs/cifar_s4d --name s4d_grid --save_plot

(3) ResNet18 baseline (same perturbation definition):
  python paper_spectral_grid_bench.py --dataset cifar10 --model resnet18 --epochs 200 \
    --augment --lr 0.1 --weight_decay 5e-4 --batch_size 128 \
    --sweep --pert_rel 0.05 --n_phase 5 \
    --out_dir runs/cifar_resnet18 --name resnet18_rel005 --save_plot

(4) Transformer baseline (reduce batch size!):
  python paper_spectral_grid_bench.py --dataset cifar10 --model transformer --epochs 80 \
    --batch_size 16 --lr 3e-4 --weight_decay 0.01 \
    --sweep --pert_rel 0.05 --n_phase 3 \
    --out_dir runs/cifar_transformer --name tf_rel005 --save_plot
"""

import os
import math
import time
import json
import csv
import inspect
import argparse
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


# ---------- S4D import (repo-dependent) ----------
try:
    from models.s4.s4d import S4D
except Exception as e:
    raise ImportError(
        "Failed to import S4D from models.s4.s4d.\n"
        "Run from the state-spaces/s4 repo root or adjust PYTHONPATH.\n"
        f"Original error: {repr(e)}"
    )


# ----------------- Utils -----------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def set_seed(seed: int, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        cudnn.deterministic = True
        cudnn.benchmark = False


class Logger:
    def __init__(self, log_path: Optional[str]):
        self.log_path = log_path
        self.f = None
        if log_path is not None:
            ensure_dir(os.path.dirname(log_path))
            self.f = open(log_path, "w", encoding="utf-8")

    def log(self, msg: str):
        print(msg, flush=True)
        if self.f is not None:
            self.f.write(msg + "\n")
            self.f.flush()

    def close(self):
        if self.f is not None:
            self.f.close()
            self.f = None


def parse_list_floats(s: str) -> List[float]:
    s = s.strip()
    if not s:
        return []
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if tok:
            out.append(float(tok))
    return out


# ----------------- Data helpers -----------------
def img_to_seq(x_img: torch.Tensor) -> torch.Tensor:
    # x_img: (B,C,H,W) -> (B,L,C)
    B, C, H, W = x_img.shape
    return x_img.reshape(B, C, H * W).transpose(1, 2).contiguous()


def seq_to_img(x_seq: torch.Tensor, C: int, H: int, W: int) -> torch.Tensor:
    # x_seq: (B,L,C) -> (B,C,H,W)
    B, L, C2 = x_seq.shape
    assert C2 == C
    assert L == H * W
    return x_seq.transpose(1, 2).reshape(B, C, H, W).contiguous()


# ----------------- Models -----------------
class S4SeqModel(nn.Module):
    def __init__(
        self,
        d_input: int,
        d_output: int,
        d_model: int,
        n_layers: int,
        dropout: float,
        prenorm: bool,
        lr_s4: float,
    ):
        super().__init__()
        self.prenorm = prenorm

        self.encoder = nn.Linear(d_input, d_model)

        self.s4_layers = nn.ModuleList([
            S4D(d_model, dropout=dropout, transposed=True, lr=lr_s4)
            for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(n_layers)])

        self.decoder = nn.Linear(d_model, d_output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,L,d_input)
        x = self.encoder(x)          # (B,L,d_model)
        x = x.transpose(-1, -2)      # (B,d_model,L)

        for layer, norm, drop in zip(self.s4_layers, self.norms, self.dropouts):
            z = x
            if self.prenorm:
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)

            out = layer(z)
            z = out[0] if isinstance(out, tuple) else out

            z = drop(z)
            x = x + z

            if not self.prenorm:
                x = norm(x.transpose(-1, -2)).transpose(-1, -2)

        x = x.transpose(-1, -2)      # (B,L,d_model)
        x = x.mean(dim=1)            # (B,d_model)
        x = self.decoder(x)          # (B,d_output)
        return x


class CNN1DSeqModel(nn.Module):
    """
    Lightweight 1D CNN baseline over sequence: (B,L,C) -> (B,num_classes)
    """
    def __init__(self, d_input: int, d_output: int, d_model: int, n_layers: int, kernel: int, dropout: float):
        super().__init__()
        assert kernel % 2 == 1, "kernel should be odd for same padding"
        self.in_proj = nn.Conv1d(d_input, d_model, kernel_size=1)

        blocks = []
        for _ in range(n_layers):
            blocks.append(nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=kernel, padding=kernel // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Conv1d(d_model, d_model, kernel_size=kernel, padding=kernel // 2),
                nn.Dropout(dropout),
            ))
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.BatchNorm1d(d_model)
        self.head = nn.Linear(d_model, d_output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,L,C) -> (B,C,L)
        x = x.transpose(1, 2).contiguous()
        x = self.in_proj(x)
        for blk in self.blocks:
            z = blk(x)
            x = x + z
        x = self.norm(x)
        x = x.mean(dim=-1)  # global avg over L
        return self.head(x)


class TransformerSeqModel(nn.Module):
    """
    Vanilla TransformerEncoder baseline over sequence. O(L^2) – reduce batch_size for CIFAR (L=1024).
    """
    def __init__(
        self,
        d_input: int,
        d_output: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        ff_mult: int,
        dropout: float,
        seq_len: int,
        prenorm: bool,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.encoder = nn.Linear(d_input, d_model)
        self.pos = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.normal_(self.pos, std=0.02)

        dim_ff = ff_mult * d_model

        # Support old/new PyTorch: norm_first may not exist in very old versions
        layer_kwargs = dict(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        sig = inspect.signature(nn.TransformerEncoderLayer.__init__)
        if "norm_first" in sig.parameters:
            layer_kwargs["norm_first"] = prenorm

        enc_layer = nn.TransformerEncoderLayer(**layer_kwargs)
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, d_output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,L,C)
        B, L, _ = x.shape
        assert L == self.seq_len, f"Expected L={self.seq_len}, got L={L}"
        x = self.encoder(x) + self.pos[:, :L, :]
        x = self.drop(x)
        x = self.enc(x)
        x = x.mean(dim=1)
        return self.head(x)


def build_resnet18_cifar(in_ch: int, num_classes: int) -> nn.Module:
    m = torchvision.models.resnet18(num_classes=num_classes)
    # CIFAR-style stem
    m.conv1 = nn.Conv2d(in_ch, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    return m


def build_model(
    model_name: str,
    d_input: int,
    d_output: int,
    seq_len: int,
    args,
) -> Tuple[nn.Module, str]:
    """
    Returns: (model, input_mode) where input_mode in {"seq","img"}.
    """
    if model_name == "s4d":
        lr_s4 = min(0.001, args.lr)
        return S4SeqModel(
            d_input=d_input, d_output=d_output,
            d_model=args.d_model, n_layers=args.n_layers,
            dropout=args.dropout, prenorm=args.prenorm,
            lr_s4=lr_s4,
        ), "seq"

    if model_name == "cnn1d":
        return CNN1DSeqModel(
            d_input=d_input, d_output=d_output,
            d_model=args.d_model, n_layers=args.n_layers,
            kernel=args.cnn_kernel, dropout=args.dropout,
        ), "seq"

    if model_name == "transformer":
        return TransformerSeqModel(
            d_input=d_input, d_output=d_output,
            d_model=args.d_model, n_layers=args.n_layers,
            n_heads=args.tf_heads, ff_mult=args.tf_ff_mult,
            dropout=args.dropout, seq_len=seq_len,
            prenorm=args.prenorm,
        ), "seq"

    if model_name == "resnet18":
        return build_resnet18_cifar(d_input, d_output), "img"

    raise ValueError(f"Unknown model: {model_name}")


# ----------------- Optimizer setup -----------------
def setup_optimizer(model: nn.Module, lr: float, weight_decay: float, epochs: int, model_name: str):
    """
    S4D in state-spaces/s4 expects special param groups via _optim.
    Other models can just use AdamW or SGD.
    """
    if model_name == "s4d":
        all_parameters = list(model.parameters())
        base_params = [p for p in all_parameters if not hasattr(p, "_optim")]

        optimizer = optim.AdamW(base_params, lr=lr, weight_decay=weight_decay)

        hps = [getattr(p, "_optim") for p in all_parameters if hasattr(p, "_optim")]
        hps = [dict(s) for s in sorted(list(dict.fromkeys(frozenset(hp.items()) for hp in hps)))]
        for hp in hps:
            params = [p for p in all_parameters if getattr(p, "_optim", None) == hp]
            optimizer.add_param_group({"params": params, **hp})

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
        return optimizer, scheduler

    # CIFAR ResNet common choice: SGD + cosine
    if model_name == "resnet18":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
        return optimizer, scheduler

    # transformer/cnn baseline: AdamW + cosine
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    return optimizer, scheduler


# ----------------- Train / Eval loops -----------------
def train_epoch(model, input_mode, loader, criterion, optimizer, device, amp=False, grad_clip=0.0):
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    loss_sum, correct, total = 0.0, 0, 0

    for x_img, y in loader:
        x_img = x_img.to(device)
        y = y.to(device)

        if input_mode == "seq":
            x = img_to_seq(x_img)
        else:
            x = x_img

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=amp):
            logits = model(x)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()

        if grad_clip and grad_clip > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()

        bs = y.size(0)
        loss_sum += float(loss.item()) * bs
        total += bs
        correct += (logits.argmax(dim=1) == y).sum().item()

    return loss_sum / total, 100.0 * correct / total


@torch.no_grad()
def eval_epoch(model, input_mode, loader, criterion, device, amp=False):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0

    for x_img, y in loader:
        x_img = x_img.to(device)
        y = y.to(device)

        if input_mode == "seq":
            x = img_to_seq(x_img)
        else:
            x = x_img

        with torch.cuda.amp.autocast(enabled=amp):
            logits = model(x)
            loss = criterion(logits, y)

        bs = y.size(0)
        loss_sum += float(loss.item()) * bs
        total += bs
        correct += (logits.argmax(dim=1) == y).sum().item()

    return loss_sum / total, 100.0 * correct / total


# ----------------- Perturbation: single cosine mode on sequence -----------------
@dataclass
class CosPerturbCfg:
    rho: float
    target_rel: Optional[float] = 0.05  # match relΔ per sample if not None
    eps: Optional[float] = None         # fixed eps if target_rel is None
    phase_mode: str = "sample"          # fixed|batch|sample
    phase_value: float = 0.0
    channels: str = "all"               # all or comma list "0,1" etc
    clamp_valid_bin: bool = True        # clamp m in [1, L//2-1]
    eps_floor: float = 0.0
    eps_cap: Optional[float] = None


def _channel_mask(ch: str, C: int, device, dtype) -> torch.Tensor:
    if ch == "all":
        return torch.ones(C, device=device, dtype=dtype)
    idx = [int(s) for s in ch.split(",") if s.strip() != ""]
    mask = torch.zeros(C, device=device, dtype=dtype)
    for i in idx:
        if i < 0 or i >= C:
            raise ValueError(f"Invalid channel index {i} for C={C}")
        mask[i] = 1.0
    if float(mask.sum().item()) == 0.0:
        raise ValueError(f"channels='{ch}' selects no channel")
    return mask


@torch.no_grad()
def add_single_cos_mode_seq(
    x_seq: torch.Tensor,  # (B,L,C)
    cfg: CosPerturbCfg,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns (x_noisy, delta, relΔ_per_sample).
    relΔ computed over (L,C) L2 norms.
    """
    assert x_seq.ndim == 3
    B, L, C = x_seq.shape
    device = x_seq.device
    dtype = x_seq.dtype

    m = int(round(cfg.rho * (L // 2)))
    if cfg.clamp_valid_bin:
        m = max(1, min(m, (L // 2) - 1))

    n = torch.arange(L, device=device, dtype=dtype)  # (L,)

    # phase selection
    if cfg.phase_mode == "fixed":
        phi = torch.tensor(cfg.phase_value, device=device, dtype=dtype).view(1, 1)
    elif cfg.phase_mode == "batch":
        r = torch.rand(1, device=device, dtype=dtype, generator=generator)
        phi = (2 * math.pi * r).view(1, 1)
    elif cfg.phase_mode == "sample":
        r = torch.rand(B, device=device, dtype=dtype, generator=generator)
        phi = (2 * math.pi * r).view(B, 1)
    else:
        raise ValueError(f"phase_mode must be fixed|batch|sample, got {cfg.phase_mode}")

    base = (2 * math.pi * m / L) * n.view(1, L)   # (1,L)
    s = torch.cos(base + phi)                      # (B,L) or (1,L)
    if s.shape[0] == 1 and B != 1:
        s = s.expand(B, -1)
    s = s.unsqueeze(-1)                            # (B,L,1)

    ch_mask = _channel_mask(cfg.channels, C, device, dtype)  # (C,)
    n_ch = float(ch_mask.sum().item())
    ch_factor = math.sqrt(max(n_ch, 1e-12))

    x_norm = torch.sqrt(torch.sum(x_seq.float() ** 2, dim=(1, 2)) + 1e-12)  # (B,)
    s_norm = torch.sqrt(torch.sum(s.float() ** 2, dim=(1, 2)) + 1e-12)      # (B,)

    if cfg.target_rel is not None:
        eps_i = (cfg.target_rel * x_norm) / (s_norm * ch_factor + 1e-12)    # (B,)
    else:
        assert cfg.eps is not None
        eps_i = torch.full((B,), float(cfg.eps), device=device, dtype=torch.float32)

    if cfg.eps_floor and cfg.eps_floor > 0:
        eps_i = torch.maximum(eps_i, torch.tensor(cfg.eps_floor, device=device, dtype=eps_i.dtype))
    if cfg.eps_cap is not None:
        eps_i = torch.minimum(eps_i, torch.tensor(cfg.eps_cap, device=device, dtype=eps_i.dtype))

    eps_i = eps_i.to(device=device, dtype=dtype).view(B, 1, 1)  # (B,1,1)

    delta_base = eps_i * s                     # (B,L,1)
    delta = delta_base.expand(B, L, C) * ch_mask.view(1, 1, C)
    x_noisy = x_seq + delta

    delta_norm = torch.sqrt(torch.sum(delta.float() ** 2, dim=(1, 2)) + 1e-12)
    rel_delta = delta_norm / (x_norm + 1e-12)

    return x_noisy, delta, rel_delta


# ----------------- Cache clean outputs -----------------
@dataclass
class CleanCache:
    clean_acc: float
    logits_cpu: torch.Tensor  # (N,K)
    pred_cpu: torch.Tensor    # (N,)
    y_cpu: torch.Tensor       # (N,)


@torch.no_grad()
def cache_clean(model, input_mode, loader, device, amp=False) -> CleanCache:
    model.eval()
    correct = 0
    total = 0
    all_logits = []
    all_pred = []
    all_y = []

    for x_img, y in loader:
        x_img = x_img.to(device)
        y = y.to(device)

        x = img_to_seq(x_img) if input_mode == "seq" else x_img

        with torch.cuda.amp.autocast(enabled=amp):
            logits = model(x)

        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)

        all_logits.append(logits.detach().cpu())
        all_pred.append(pred.detach().cpu())
        all_y.append(y.detach().cpu())

    logits_cpu = torch.cat(all_logits, dim=0)
    pred_cpu = torch.cat(all_pred, dim=0)
    y_cpu = torch.cat(all_y, dim=0)

    return CleanCache(
        clean_acc=100.0 * correct / max(total, 1),
        logits_cpu=logits_cpu,
        pred_cpu=pred_cpu,
        y_cpu=y_cpu,
    )


# ----------------- Evaluate one (rho, pert_rel) -----------------
@torch.no_grad()
def eval_perturb(
    model,
    input_mode,
    loader,
    cache: CleanCache,
    device,
    cfg: CosPerturbCfg,
    C: int,
    H: int,
    W: int,
    amp=False,
    generator_seed: Optional[int] = None,
) -> Dict[str, float]:
    model.eval()
    g = None
    if generator_seed is not None:
        g = torch.Generator(device=device)
        g.manual_seed(int(generator_seed))

    N = cache.pred_cpu.numel()
    idx = 0

    correct_noisy = 0
    flip = 0
    total = 0

    rel_sum = 0.0
    logit_l2_sum = 0.0
    logit_rel_sum = 0.0
    margin_drop_sum = 0.0

    for x_img, y in loader:
        bs = y.size(0)
        x_img = x_img.to(device)
        y = y.to(device)

        clean_logits = cache.logits_cpu[idx:idx+bs].to(device)
        clean_pred = cache.pred_cpu[idx:idx+bs].to(device)

        # always perturb in seq space (flatten axis), then reshape depending on model
        x_seq = img_to_seq(x_img)                       # (B,L,C)
        x_seq_noisy, _, rel_delta = add_single_cos_mode_seq(x_seq, cfg, generator=g)

        if input_mode == "seq":
            x_in = x_seq_noisy
        else:
            x_in = seq_to_img(x_seq_noisy, C=C, H=H, W=W)

        with torch.cuda.amp.autocast(enabled=amp):
            logits_noisy = model(x_in)

        pred_noisy = logits_noisy.argmax(dim=1)
        correct_noisy += (pred_noisy == y).sum().item()
        flip += (pred_noisy != clean_pred).sum().item()
        total += bs

        rel_sum += float(rel_delta.mean().item()) * bs

        dlog = logits_noisy - clean_logits
        dlog_l2 = torch.sqrt(torch.sum(dlog.float() ** 2, dim=1) + 1e-12)
        clog_l2 = torch.sqrt(torch.sum(clean_logits.float() ** 2, dim=1) + 1e-12)

        logit_l2_sum += float(dlog_l2.mean().item()) * bs
        logit_rel_sum += float((dlog_l2 / (clog_l2 + 1e-12)).mean().item()) * bs

        # margin drop: (true - best_other) clean minus noisy
        true_idx = y.view(-1, 1)
        clean_true = clean_logits.gather(1, true_idx).squeeze(1)
        noisy_true = logits_noisy.gather(1, true_idx).squeeze(1)

        mask = torch.ones_like(clean_logits, dtype=torch.bool)
        mask.scatter_(1, true_idx, False)
        clean_best_other = clean_logits.masked_fill(~mask, -1e9).max(dim=1).values
        noisy_best_other = logits_noisy.masked_fill(~mask, -1e9).max(dim=1).values

        clean_margin = clean_true - clean_best_other
        noisy_margin = noisy_true - noisy_best_other
        margin_drop_sum += float((clean_margin - noisy_margin).mean().item()) * bs

        idx += bs

    assert idx == N, f"Cache mismatch: iter {idx} vs cache {N}"

    hf_acc = 100.0 * correct_noisy / max(total, 1)
    flip_rate = 100.0 * flip / max(total, 1)
    rel_delta_avg = rel_sum / max(total, 1)
    logit_l2_avg = logit_l2_sum / max(total, 1)
    logit_rel_avg = logit_rel_sum / max(total, 1)
    margin_drop_avg = margin_drop_sum / max(total, 1)

    drop = cache.clean_acc - hf_acc

    return {
        "rho": float(cfg.rho),
        "pert_rel": float(cfg.target_rel) if cfg.target_rel is not None else float("nan"),
        "clean_acc": float(cache.clean_acc),
        "hf_acc": float(hf_acc),
        "drop": float(drop),
        "flip": float(flip_rate),
        "rel_delta": float(rel_delta_avg),
        "logit_l2": float(logit_l2_avg),
        "logit_rel": float(logit_rel_avg),
        "margin_drop": float(margin_drop_avg),
    }


# ----------------- CSV + Plots -----------------
def write_csv(path: str, rows: List[Dict[str, float]], extra_cols: Optional[Dict[str, str]] = None):
    ensure_dir(os.path.dirname(path))
    if not rows:
        return

    # stable columns
    base_cols = ["model", "dataset", "rho", "pert_rel",
                 "clean_acc_mean", "hf_acc_mean", "drop_mean", "drop_std",
                 "flip_mean", "flip_std",
                 "rel_delta_mean", "rel_delta_std",
                 "logit_rel_mean", "logit_rel_std",
                 "margin_drop_mean", "margin_drop_std"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=base_cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def agg_stats(stats: List[Dict[str, float]]) -> Dict[str, float]:
    # stats: list of per-rep dicts with same keys
    out = {}
    keys = list(stats[0].keys())
    for k in keys:
        vals = np.array([s[k] for s in stats], dtype=np.float64)
        out[k + "_mean"] = float(vals.mean())
        out[k + "_std"] = float(vals.std(ddof=0))
    return out


def maybe_plot_1d(out_png: str, rows: List[Dict[str, float]], title_prefix: str):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    rhos = [r["rho"] for r in rows]
    clean = [r["clean_acc_mean"] for r in rows]
    hf = [r["hf_acc_mean"] for r in rows]
    drop = [r["drop_mean"] for r in rows]

    ensure_dir(os.path.dirname(out_png))

    plt.figure()
    plt.plot(rhos, clean, label="clean")
    plt.plot(rhos, hf, label="HF")
    plt.xlabel("rho (1.0 = Nyquist)")
    plt.ylabel("Accuracy (%)")
    plt.title(f"{title_prefix} | Accuracy vs rho")
    plt.grid(True)
    plt.legend()
    plt.savefig(out_png.replace(".png", "_acc.png"), dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(rhos, drop, label="drop")
    plt.xlabel("rho (1.0 = Nyquist)")
    plt.ylabel("clean - HF (%)")
    plt.title(f"{title_prefix} | Drop vs rho")
    plt.grid(True)
    plt.legend()
    plt.savefig(out_png.replace(".png", "_drop.png"), dpi=200, bbox_inches="tight")
    plt.close()


def maybe_plot_heatmap(out_png: str, rhos: List[float], rels: List[float], grid: np.ndarray,
                       title: str, xlabel: str = "rho", ylabel: str = "pert_rel"):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    ensure_dir(os.path.dirname(out_png))

    plt.figure()
    # grid shape: (len(rels), len(rhos))
    plt.imshow(grid, aspect="auto", origin="lower")
    plt.colorbar()
    plt.xticks(ticks=list(range(len(rhos))), labels=[f"{r:.2f}" for r in rhos], rotation=45, ha="right")
    plt.yticks(ticks=list(range(len(rels))), labels=[f"{e:.3f}" for e in rels])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close()


# ----------------- Main -----------------
def main():
    p = argparse.ArgumentParser()

    # Data
    p.add_argument("--dataset", choices=["cifar10", "mnist"], default="cifar10")
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--augment", action="store_true", help="(CIFAR10) RandomCrop+Flip for training")
    p.add_argument("--grayscale", action="store_true", help="(CIFAR10) convert to grayscale")

    # Model
    p.add_argument("--model", choices=["s4d", "cnn1d", "transformer", "resnet18"], default="s4d")
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--prenorm", action="store_true")

    # CNN1D
    p.add_argument("--cnn_kernel", type=int, default=5)

    # Transformer
    p.add_argument("--tf_heads", type=int, default=4)
    p.add_argument("--tf_ff_mult", type=int, default=4)

    # Train
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=0.0)

    # Misc
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--amp", action="store_true")

    # Sweep (1D)
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--rho_values", type=str, default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.97,0.99")
    p.add_argument("--pert_rel", type=float, default=0.05)

    # Grid sweep (2D)
    p.add_argument("--sweep_grid", action="store_true")
    p.add_argument("--pert_rel_values", type=str, default="0.01,0.03,0.05,0.10")
    p.add_argument("--heat_metric", choices=["drop", "flip", "logit_rel", "margin_drop"], default="drop")

    # Phase averaging
    p.add_argument("--n_phase", type=int, default=1, help="repeat per point with different random phase seeds")
    p.add_argument("--phase_mode", choices=["fixed", "batch", "sample"], default="sample")
    p.add_argument("--phase_value", type=float, default=0.0)

    # Perturb misc
    p.add_argument("--pert_channels", type=str, default="all")
    p.add_argument("--clamp_valid_bin", action="store_true", help="clamp m in [1, L//2-1]")
    p.add_argument("--eps_floor", type=float, default=0.0)
    p.add_argument("--eps_cap", type=float, default=None)

    # IO
    p.add_argument("--out_dir", type=str, default="./runs/spectral_bench")
    p.add_argument("--name", type=str, default="run")
    p.add_argument("--log_path", type=str, default=None)
    p.add_argument("--save_plot", action="store_true")

    # CKPT
    p.add_argument("--skip_train", action="store_true")
    p.add_argument("--ckpt_path", type=str, default=None)

    args = p.parse_args()

    ensure_dir(args.out_dir)
    if args.log_path is None:
        args.log_path = os.path.join(args.out_dir, f"{args.name}.log")
    logger = Logger(args.log_path)

    logger.log("[args]\n" + json.dumps(vars(args), indent=2))

    set_seed(args.seed, deterministic=args.deterministic)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not args.deterministic:
        cudnn.benchmark = True

    # --- Dataset / transforms (return image tensor) ---
    if args.dataset == "cifar10":
        H, W = 32, 32
        if args.grayscale:
            C = 1
            mean = (122.6 / 255.0,)
            std = (61.0 / 255.0,)
        else:
            C = 3
            mean = (0.4914, 0.4822, 0.4465)
            std = (0.2023, 0.1994, 0.2010)

        train_tf = []
        if args.augment:
            train_tf += [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
        if args.grayscale:
            train_tf += [transforms.Grayscale(num_output_channels=1)]
        train_tf += [transforms.ToTensor(), transforms.Normalize(mean, std)]

        test_tf = []
        if args.grayscale:
            test_tf += [transforms.Grayscale(num_output_channels=1)]
        test_tf += [transforms.ToTensor(), transforms.Normalize(mean, std)]

        trainset = torchvision.datasets.CIFAR10(
            root=os.path.join(args.data_root, "cifar"),
            train=True,
            download=True,
            transform=transforms.Compose(train_tf),
        )
        testset = torchvision.datasets.CIFAR10(
            root=os.path.join(args.data_root, "cifar"),
            train=False,
            download=True,
            transform=transforms.Compose(test_tf),
        )
        d_input, d_output = C, 10

    else:
        # MNIST
        H, W = 28, 28
        C = 1
        train_tf = [transforms.ToTensor()]
        test_tf = [transforms.ToTensor()]
        trainset = torchvision.datasets.MNIST(
            root=os.path.join(args.data_root, "mnist"),
            train=True,
            download=True,
            transform=transforms.Compose(train_tf),
        )
        testset = torchvision.datasets.MNIST(
            root=os.path.join(args.data_root, "mnist"),
            train=False,
            download=True,
            transform=transforms.Compose(test_tf),
        )
        d_input, d_output = 1, 10

    seq_len = H * W

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

    # --- Model ---
    if args.model == "transformer" and args.dataset == "cifar10" and args.batch_size > 32:
        logger.log("[warn] Transformer on CIFAR(L=1024) may OOM with large batch_size. Consider --batch_size 16.")

    model, input_mode = build_model(args.model, d_input, d_output, seq_len, args)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer, scheduler = setup_optimizer(model, args.lr, args.weight_decay, args.epochs, args.model)

    ckpt_last = os.path.join(args.out_dir, f"{args.name}_ckpt_last.pt")

    # --- Load ckpt ---
    if args.ckpt_path is not None and os.path.isfile(args.ckpt_path):
        ckpt = torch.load(args.ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"], strict=True)
        if not args.skip_train and "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if not args.skip_train and "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        logger.log(f"[ckpt] loaded {args.ckpt_path}")

    # --- Train ---
    if not args.skip_train:
        logger.log("[train] start")
        best = -1.0
        for epoch in range(args.epochs):
            t0 = time.time()
            tr_loss, tr_acc = train_epoch(model, input_mode, trainloader, criterion, optimizer, device,
                                          amp=args.amp, grad_clip=args.grad_clip)
            te_loss, te_acc = eval_epoch(model, input_mode, testloader, criterion, device, amp=args.amp)
            scheduler.step()
            dt = time.time() - t0
            best = max(best, te_acc)
            logger.log(f"Epoch {epoch:03d} | train {tr_loss:.4f}/{tr_acc:.2f}% | test {te_loss:.4f}/{te_acc:.2f}% | {dt:.1f}s")

            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "args": vars(args),
                    "epoch": epoch,
                },
                ckpt_last,
            )
        logger.log(f"[train] done. best_test={best:.2f}% | ckpt={ckpt_last}")
    else:
        logger.log("[train] skipped")

    # --- Cache clean ---
    logger.log("[cache] computing clean outputs on test set")
    cache = cache_clean(model, input_mode, testloader, device=device, amp=args.amp)
    logger.log(f"[cache] clean_acc={cache.clean_acc:.2f}% (N={cache.pred_cpu.numel()})")

    rhos = parse_list_floats(args.rho_values)
    if not rhos:
        raise ValueError("rho_values is empty")

    # --- 1D sweep ---
    if args.sweep:
        rows_out = []
        logger.log("[sweep] 1D rho sweep (fixed pert_rel)")
        for rho in rhos:
            stats = []
            for rep in range(args.n_phase):
                cfg = CosPerturbCfg(
                    rho=float(rho),
                    target_rel=float(args.pert_rel),
                    eps=None,
                    phase_mode=args.phase_mode,
                    phase_value=float(args.phase_value),
                    channels=args.pert_channels,
                    clamp_valid_bin=args.clamp_valid_bin,
                    eps_floor=float(args.eps_floor),
                    eps_cap=float(args.eps_cap) if args.eps_cap is not None else None,
                )
                seed_rep = args.seed * 100000 + int(round(rho * 10000)) * 100 + rep
                out = eval_perturb(model, input_mode, testloader, cache, device, cfg, C=C, H=H, W=W,
                                   amp=args.amp, generator_seed=seed_rep)
                stats.append(out)

            ag = agg_stats(stats)  # keys: ..._mean/_std for each metric in out dict
            row = {
                "model": args.model,
                "dataset": args.dataset,
                "rho": float(rho),
                "pert_rel": float(args.pert_rel),

                "clean_acc_mean": ag["clean_acc_mean"],
                "hf_acc_mean": ag["hf_acc_mean"],
                "drop_mean": ag["drop_mean"],
                "drop_std": ag["drop_std"],

                "flip_mean": ag["flip_mean"],
                "flip_std": ag["flip_std"],

                "rel_delta_mean": ag["rel_delta_mean"],
                "rel_delta_std": ag["rel_delta_std"],

                "logit_rel_mean": ag["logit_rel_mean"],
                "logit_rel_std": ag["logit_rel_std"],

                "margin_drop_mean": ag["margin_drop_mean"],
                "margin_drop_std": ag["margin_drop_std"],
            }
            rows_out.append(row)

            logger.log(
                f"rho={rho:.3f} | clean {row['clean_acc_mean']:.2f}% | HF {row['hf_acc_mean']:.2f}% | "
                f"drop {row['drop_mean']:+.2f}% ±{row['drop_std']:.2f} | flip {row['flip_mean']:.2f}% | relΔ {row['rel_delta_mean']:.3f}"
            )

        csv_path = os.path.join(args.out_dir, f"{args.name}_{args.model}_{args.dataset}_rho_sweep.csv")
        write_csv(csv_path, rows_out)
        logger.log(f"[sweep] saved CSV: {csv_path}")

        if args.save_plot:
            png_path = os.path.join(args.out_dir, f"{args.name}_{args.model}_{args.dataset}_rho_sweep.png")
            maybe_plot_1d(png_path, rows_out, title_prefix=f"{args.model} | {args.dataset} | relΔ={args.pert_rel}")
            logger.log(f"[sweep] saved plots: {png_path.replace('.png','_acc.png')} and _drop.png")

    # --- 2D grid sweep ---
    if args.sweep_grid:
        rels = parse_list_floats(args.pert_rel_values)
        if not rels:
            raise ValueError("pert_rel_values is empty")

        logger.log("[grid] 2D sweep: rho x pert_rel")
        rows_grid = []

        # store heatmap matrix for chosen metric (mean)
        metric = args.heat_metric  # drop / flip / logit_rel / margin_drop
        grid_mat = np.zeros((len(rels), len(rhos)), dtype=np.float64)

        for i, rel in enumerate(rels):
            for j, rho in enumerate(rhos):
                stats = []
                for rep in range(args.n_phase):
                    cfg = CosPerturbCfg(
                        rho=float(rho),
                        target_rel=float(rel),
                        eps=None,
                        phase_mode=args.phase_mode,
                        phase_value=float(args.phase_value),
                        channels=args.pert_channels,
                        clamp_valid_bin=args.clamp_valid_bin,
                        eps_floor=float(args.eps_floor),
                        eps_cap=float(args.eps_cap) if args.eps_cap is not None else None,
                    )
                    seed_rep = args.seed * 100000 + int(round(rel * 10000)) * 1000 + int(round(rho * 10000)) * 10 + rep
                    out = eval_perturb(model, input_mode, testloader, cache, device, cfg, C=C, H=H, W=W,
                                       amp=args.amp, generator_seed=seed_rep)
                    stats.append(out)

                ag = agg_stats(stats)

                row = {
                    "model": args.model,
                    "dataset": args.dataset,
                    "rho": float(rho),
                    "pert_rel": float(rel),

                    "clean_acc_mean": ag["clean_acc_mean"],
                    "hf_acc_mean": ag["hf_acc_mean"],
                    "drop_mean": ag["drop_mean"],
                    "drop_std": ag["drop_std"],

                    "flip_mean": ag["flip_mean"],
                    "flip_std": ag["flip_std"],

                    "rel_delta_mean": ag["rel_delta_mean"],
                    "rel_delta_std": ag["rel_delta_std"],

                    "logit_rel_mean": ag["logit_rel_mean"],
                    "logit_rel_std": ag["logit_rel_std"],

                    "margin_drop_mean": ag["margin_drop_mean"],
                    "margin_drop_std": ag["margin_drop_std"],
                }
                rows_grid.append(row)

                # fill heatmap cell
                if metric == "drop":
                    grid_mat[i, j] = row["drop_mean"]
                elif metric == "flip":
                    grid_mat[i, j] = row["flip_mean"]
                elif metric == "logit_rel":
                    grid_mat[i, j] = row["logit_rel_mean"]
                elif metric == "margin_drop":
                    grid_mat[i, j] = row["margin_drop_mean"]
                else:
                    raise ValueError(metric)

                logger.log(
                    f"[grid] rel={rel:.3f} rho={rho:.3f} | drop {row['drop_mean']:+.2f}% | flip {row['flip_mean']:.2f}% | relΔ {row['rel_delta_mean']:.3f}"
                )

        csv_path = os.path.join(args.out_dir, f"{args.name}_{args.model}_{args.dataset}_grid.csv")
        write_csv(csv_path, rows_grid)
        logger.log(f"[grid] saved CSV: {csv_path}")

        if args.save_plot:
            png_path = os.path.join(args.out_dir, f"{args.name}_{args.model}_{args.dataset}_heat_{metric}.png")
            maybe_plot_heatmap(
                png_path,
                rhos=rhos,
                rels=rels,
                grid=grid_mat,
                title=f"{args.model} | {args.dataset} | heatmap ({metric}) | phase={args.phase_mode} n_phase={args.n_phase}",
                xlabel="rho (1.0 = Nyquist)",
                ylabel="pert_rel (target relΔ)",
            )
            logger.log(f"[grid] saved heatmap: {png_path}")

    logger.close()


if __name__ == "__main__":
    main()
