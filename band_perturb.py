<<<<<<< HEAD
# band_perturb.py
import math
import torch

class BandPerturb:
    """
    Inject band-limited perturbation into x: (B, L, C)

    - band_mode: "mid" or "high" or "custom"
    - kind: "tone" (single sinusoid) or "noise" (band-limited random)
    - ratio: target RMS ratio ||delta|| / ||x||  (per-sample)
    - deterministic: if True and idx provided, noise is deterministic per (idx, epoch)
    """
    def __init__(
        self,
        band_mode="high",
        kind="tone",
        ratio=0.1,
        band_lo=0.9,
        band_hi=1.0,
        tone_omega_frac=None,   # if None: mid->0.35, high->0.95
        deterministic=True,
        base_seed=0,
        eps=1e-12,
    ):
        assert band_mode in ["mid", "high", "custom"]
        assert kind in ["tone", "noise"]
        self.band_mode = band_mode
        self.kind = kind
        self.ratio = float(ratio)
        self.band_lo = float(band_lo)
        self.band_hi = float(band_hi)
        self.tone_omega_frac = tone_omega_frac
        self.deterministic = bool(deterministic)
        self.base_seed = int(base_seed)
        self.eps = float(eps)

    @staticmethod
    def _rms(x, dims):
        return torch.sqrt(torch.mean(x * x, dim=dims) + 1e-12)

    def _get_band(self):
        if self.band_mode == "mid":
            return 0.30, 0.40
        if self.band_mode == "high":
            return 0.90, 1.00
        return self.band_lo, self.band_hi

    def _get_tone_frac(self):
        if self.tone_omega_frac is not None:
            return float(self.tone_omega_frac)
        return 0.35 if self.band_mode == "mid" else 0.95

    def __call__(self, x: torch.Tensor, idx=None, epoch: int = 0):
        """
        x: (B, L, C), float
        idx: (B,) optional (for deterministic noise)
        """
        if self.ratio <= 0.0:
            return x

        B, L, C = x.shape
        device = x.device
        dtype = x.dtype

        # per-sample RMS of x
        rms_x = self._rms(x, dims=(1,2))  # (B,)

        if self.kind == "tone":
            frac = self._get_tone_frac()
            omega = math.pi * frac
            n = torch.arange(L, device=device, dtype=dtype)  # (L,)
            # phase: deterministic if idx given, else random but repeatable-ish
            if idx is not None and self.deterministic:
                # phase seeded by idx + epoch
                # (do on CPU generator for reproducibility across devices)
                phases = []
                for b in range(B):
                    g = torch.Generator(device="cpu")
                    g.manual_seed(self.base_seed + int(idx[b]) + 1000003 * int(epoch))
                    phases.append((2*math.pi) * torch.rand((), generator=g).item())
                phi = torch.tensor(phases, device=device, dtype=dtype)  # (B,)
            else:
                phi = (2*math.pi) * torch.rand((B,), device=device, dtype=dtype)

            tone = torch.cos(omega * n[None, :] + phi[:, None])  # (B,L)
            delta = tone[:, :, None].expand(B, L, C)            # (B,L,C)

        else:  # kind == "noise"
            lo_frac, hi_frac = self._get_band()
            F = L // 2 + 1
            lo = int(round(lo_frac * (F - 1)))
            hi = int(round(hi_frac * (F - 1)))
            lo = max(0, min(lo, F-1))
            hi = max(0, min(hi, F-1))
            if hi < lo:
                return x

            deltas = torch.zeros((B, L, C), device=device, dtype=dtype)

            # per-sample loop for deterministic noise
            for b in range(B):
                g = None
                if idx is not None and self.deterministic:
                    g = torch.Generator(device="cpu")
                    g.manual_seed(self.base_seed + int(idx[b]) + 1000003 * int(epoch))

                # complex spectrum noise (C, F)
                Z = torch.zeros((C, F), device=device, dtype=torch.complex64)

                # sample re/im on CPU if deterministic, then move
                if g is None:
                    re = torch.randn((C, hi - lo + 1), device=device, dtype=torch.float32)
                    im = torch.randn((C, hi - lo + 1), device=device, dtype=torch.float32)
                else:
                    re = torch.randn((C, hi - lo + 1), generator=g, device="cpu", dtype=torch.float32).to(device)
                    im = torch.randn((C, hi - lo + 1), generator=g, device="cpu", dtype=torch.float32).to(device)

                Z[:, lo:hi+1] = torch.complex(re, im)

                # enforce DC real, and Nyquist real if even L
                Z[:, 0] = torch.complex(Z[:, 0].real, torch.zeros_like(Z[:, 0].real))
                if L % 2 == 0:
                    Z[:, -1] = torch.complex(Z[:, -1].real, torch.zeros_like(Z[:, -1].real))

                # back to time domain: (C,L) -> (L,C)
                d = torch.fft.irfft(Z, n=L, dim=-1).transpose(0, 1).to(dtype)
                deltas[b] = d

            delta = deltas

        # scale per-sample to match target ratio
        rms_d = self._rms(delta, dims=(1,2))  # (B,)
        scale = (self.ratio * rms_x) / (rms_d + self.eps)  # (B,)
        x_out = x + scale[:, None, None] * delta
        return x_out
=======
# band_perturb.py
import math
import torch

class BandPerturb:
    """
    Inject band-limited perturbation into x: (B, L, C)

    - band_mode: "mid" or "high" or "custom"
    - kind: "tone" (single sinusoid) or "noise" (band-limited random)
    - ratio: target RMS ratio ||delta|| / ||x||  (per-sample)
    - deterministic: if True and idx provided, noise is deterministic per (idx, epoch)
    """
    def __init__(
        self,
        band_mode="high",
        kind="tone",
        ratio=0.1,
        band_lo=0.9,
        band_hi=1.0,
        tone_omega_frac=None,   # if None: mid->0.35, high->0.95
        deterministic=True,
        base_seed=0,
        eps=1e-12,
    ):
        assert band_mode in ["mid", "high", "custom"]
        assert kind in ["tone", "noise"]
        self.band_mode = band_mode
        self.kind = kind
        self.ratio = float(ratio)
        self.band_lo = float(band_lo)
        self.band_hi = float(band_hi)
        self.tone_omega_frac = tone_omega_frac
        self.deterministic = bool(deterministic)
        self.base_seed = int(base_seed)
        self.eps = float(eps)

    @staticmethod
    def _rms(x, dims):
        return torch.sqrt(torch.mean(x * x, dim=dims) + 1e-12)

    def _get_band(self):
        if self.band_mode == "mid":
            return 0.30, 0.40
        if self.band_mode == "high":
            return 0.90, 1.00
        return self.band_lo, self.band_hi

    def _get_tone_frac(self):
        if self.tone_omega_frac is not None:
            return float(self.tone_omega_frac)
        return 0.35 if self.band_mode == "mid" else 0.95

    def __call__(self, x: torch.Tensor, idx=None, epoch: int = 0):
        """
        x: (B, L, C), float
        idx: (B,) optional (for deterministic noise)
        """
        if self.ratio <= 0.0:
            return x

        B, L, C = x.shape
        device = x.device
        dtype = x.dtype

        # per-sample RMS of x
        rms_x = self._rms(x, dims=(1,2))  # (B,)

        if self.kind == "tone":
            frac = self._get_tone_frac()
            omega = math.pi * frac
            n = torch.arange(L, device=device, dtype=dtype)  # (L,)
            # phase: deterministic if idx given, else random but repeatable-ish
            if idx is not None and self.deterministic:
                # phase seeded by idx + epoch
                # (do on CPU generator for reproducibility across devices)
                phases = []
                for b in range(B):
                    g = torch.Generator(device="cpu")
                    g.manual_seed(self.base_seed + int(idx[b]) + 1000003 * int(epoch))
                    phases.append((2*math.pi) * torch.rand((), generator=g).item())
                phi = torch.tensor(phases, device=device, dtype=dtype)  # (B,)
            else:
                phi = (2*math.pi) * torch.rand((B,), device=device, dtype=dtype)

            tone = torch.cos(omega * n[None, :] + phi[:, None])  # (B,L)
            delta = tone[:, :, None].expand(B, L, C)            # (B,L,C)

        else:  # kind == "noise"
            lo_frac, hi_frac = self._get_band()
            F = L // 2 + 1
            lo = int(round(lo_frac * (F - 1)))
            hi = int(round(hi_frac * (F - 1)))
            lo = max(0, min(lo, F-1))
            hi = max(0, min(hi, F-1))
            if hi < lo:
                return x

            deltas = torch.zeros((B, L, C), device=device, dtype=dtype)

            # per-sample loop for deterministic noise
            for b in range(B):
                g = None
                if idx is not None and self.deterministic:
                    g = torch.Generator(device="cpu")
                    g.manual_seed(self.base_seed + int(idx[b]) + 1000003 * int(epoch))

                # complex spectrum noise (C, F)
                Z = torch.zeros((C, F), device=device, dtype=torch.complex64)

                # sample re/im on CPU if deterministic, then move
                if g is None:
                    re = torch.randn((C, hi - lo + 1), device=device, dtype=torch.float32)
                    im = torch.randn((C, hi - lo + 1), device=device, dtype=torch.float32)
                else:
                    re = torch.randn((C, hi - lo + 1), generator=g, device="cpu", dtype=torch.float32).to(device)
                    im = torch.randn((C, hi - lo + 1), generator=g, device="cpu", dtype=torch.float32).to(device)

                Z[:, lo:hi+1] = torch.complex(re, im)

                # enforce DC real, and Nyquist real if even L
                Z[:, 0] = torch.complex(Z[:, 0].real, torch.zeros_like(Z[:, 0].real))
                if L % 2 == 0:
                    Z[:, -1] = torch.complex(Z[:, -1].real, torch.zeros_like(Z[:, -1].real))

                # back to time domain: (C,L) -> (L,C)
                d = torch.fft.irfft(Z, n=L, dim=-1).transpose(0, 1).to(dtype)
                deltas[b] = d

            delta = deltas

        # scale per-sample to match target ratio
        rms_d = self._rms(delta, dims=(1,2))  # (B,)
        scale = (self.ratio * rms_x) / (rms_d + self.eps)  # (B,)
        x_out = x + scale[:, None, None] * delta
        return x_out
>>>>>>> 2a5624c (Add project fiels)
