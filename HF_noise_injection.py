<<<<<<< HEAD
import math
import torch
def plot_one_example_spectrum_cifar(args, save_path="hf_spectrum.png"):
    import matplotlib.pyplot as plt
    import torch
    import torchvision
    import torchvision.transforms as transforms

    # raw dataset (no transform)
    raw = torchvision.datasets.CIFAR10(root='./data/cifar/', train=True, download=True, transform=None)
    img, _ = raw[0]  # PIL image

    # base transforms that keep (C,H,W)
    if args.grayscale:
        base = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((122.6/255.0,), (61.0/255.0,)),
        ])
        c = 1
    else:
        base = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        c = 3

    # HF injector (same params as training)
    hf = HFNoiseInjectSeq1D(
        band=(args.hf_band_lo, args.hf_band_hi),
        mode=args.hf_mode,
        gain=args.hf_gain,
        noise_std=args.hf_noise_std,
        snr_db=args.hf_snr_db,
        p=1.0,
        clamp=None,
    )

    # Apply transforms
    torch.manual_seed(0)  # so the plotted noise is reproducible
    x0 = base(img)        # (C,H,W)
    x1 = hf(base(img))    # (C,H,W) with HF injected

    # Flatten to sequential (L, C)
    L = 32 * 32
    x0_seq = x0.view(c, L).t()  # (L,C)
    x1_seq = x1.view(c, L).t()  # (L,C)

    # rFFT along L, average across channels
    X0 = torch.fft.rfft(x0_seq.t(), dim=-1)  # (C, F)
    X1 = torch.fft.rfft(x1_seq.t(), dim=-1)  # (C, F)
    mag0 = X0.abs().mean(dim=0).cpu().numpy()  # (F,)
    mag1 = X1.abs().mean(dim=0).cpu().numpy()  # (F,)

    F = mag0.shape[0]
    lo = int(round(args.hf_band_lo * (F - 1)))
    hi = int(round(args.hf_band_hi * (F - 1)))

    plt.figure()
    plt.plot(mag0, label="no HF")
    plt.plot(mag1, label="with HF")
    plt.axvspan(lo, hi, alpha=0.2)  # show injected band
    plt.xlabel("rFFT bin (0..Nyquist)")
    plt.ylabel("mean |X|")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"[Saved spectrum plot] {save_path}")

class HFNoiseInjectSeq1D:
    """
    Treat image as a 1D sequence by flattening spatial dims (H*W),
    then inject noise / gain in a high-frequency band near Nyquist (Ω≈π).

    - band: (lo, hi) as fraction of Nyquist in [0,1]. e.g. (0.9, 1.0) == 0.9π~π
    - mode:
        * "add": add complex Gaussian noise in the band
        * "gain": multiply existing spectrum in the band by `gain`
        * "add+gain": do both (gain first, then add)
    - noise_std: absolute std for real/imag parts (if snr_db is None)
    - snr_db: if set, noise power in the band is set relative to signal power in the band
    - p: probability to apply (use 1.0 for always)
    """

    def __init__(
        self,
        band=(0.9, 1.0),
        mode="add",
        gain=1.0,
        noise_std=0.0,
        snr_db=None,
        p=1.0,
        eps=1e-12,
        clamp=None,  # e.g. (0.0, 1.0) if you want to keep valid pixel range
    ):
        assert 0.0 <= band[0] <= band[1] <= 1.0
        assert mode in ["add", "gain", "add+gain"]
        self.band = band
        self.mode = mode
        self.gain = float(gain)
        self.noise_std = float(noise_std)
        self.snr_db = snr_db
        self.p = float(p)
        self.eps = float(eps)
        self.clamp = clamp

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(x):
            x = torch.as_tensor(x)

        # Apply with probability p (torch RNG, good with DataLoader workers)
        if self.p < 1.0 and torch.rand(()) > self.p:
            return x

        # Expect image tensor [C,H,W] (C=1 or 3 typically)
        if x.ndim != 3:
            raise ValueError(f"Expected x.ndim==3 (C,H,W), got {tuple(x.shape)}")

        C, H, W = x.shape
        L = H * W

        # Work in float32 for stable FFT
        x0 = x.to(torch.float32)
        x_seq = x0.reshape(C, L)  # [C, L]

        # rFFT along sequence axis
        X = torch.fft.rfft(x_seq, dim=-1)  # [C, L//2+1], complex

        n_bins = X.shape[-1]
        if n_bins <= 2:
            return x  # too short to meaningfully band-select

        # Map band fraction to rFFT bin indices
        # rFFT bins correspond to Ω in [0, π], where bin (n_bins-1) is Ω=π (Nyquist)
        lo = int(round(self.band[0] * (n_bins - 1)))
        hi = int(round(self.band[1] * (n_bins - 1)))
        lo = max(0, min(lo, n_bins - 1))
        hi = max(0, min(hi, n_bins - 1))
        if hi < lo:
            return x

        band_slice = slice(lo, hi + 1)
        X_band = X[..., band_slice]

        # Optional gain
        if self.mode in ["gain", "add+gain"] and self.gain != 1.0:
            X_band = X_band * self.gain

        # Optional noise add
        if self.mode in ["add", "add+gain"]:
            if self.snr_db is not None:
                # Match noise power to signal power within the band
                # SNR = P_signal / P_noise  =>  P_noise = P_signal / SNR
                snr = 10.0 ** (float(self.snr_db) / 10.0)
                P_sig = (X[..., band_slice].abs() ** 2).mean()
                P_noise = P_sig / max(snr, self.eps)
                # If noise = a + j b with a,b ~ N(0, sigma^2), then E|noise|^2 = 2*sigma^2
                sigma = torch.sqrt(P_noise / 2.0 + self.eps)
            else:
                sigma = torch.tensor(self.noise_std, dtype=torch.float32)

            if float(sigma) > 0.0:
                nr = torch.randn_like(X_band.real)
                ni = torch.randn_like(X_band.real)
                noise = torch.complex(nr, ni) * sigma
                X_band = X_band + noise

        # Write back modified band
        X[..., band_slice] = X_band

        # Keep DC (0) real, and Nyquist (last bin, if even L) real to avoid tiny numerical weirdness
        X[..., 0] = torch.complex(X[..., 0].real, torch.zeros_like(X[..., 0].real))
        if L % 2 == 0:
            X[..., -1] = torch.complex(X[..., -1].real, torch.zeros_like(X[..., -1].real))

        # iRFFT back to real sequence and reshape to image
        x_out = torch.fft.irfft(X, n=L, dim=-1).reshape(C, H, W)

        if self.clamp is not None:
            lo_c, hi_c = self.clamp
            x_out = x_out.clamp(lo_c, hi_c)

        # Cast back to original dtype if it was float-like
        if x.dtype in (torch.float16, torch.float32, torch.float64):
            x_out = x_out.to(x.dtype)

        return x_out
=======
import math
import torch
def plot_one_example_spectrum_cifar(args, save_path="hf_spectrum.png"):
    import matplotlib.pyplot as plt
    import torch
    import torchvision
    import torchvision.transforms as transforms

    # raw dataset (no transform)
    raw = torchvision.datasets.CIFAR10(root='./data/cifar/', train=True, download=True, transform=None)
    img, _ = raw[0]  # PIL image

    # base transforms that keep (C,H,W)
    if args.grayscale:
        base = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((122.6/255.0,), (61.0/255.0,)),
        ])
        c = 1
    else:
        base = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        c = 3

    # HF injector (same params as training)
    hf = HFNoiseInjectSeq1D(
        band=(args.hf_band_lo, args.hf_band_hi),
        mode=args.hf_mode,
        gain=args.hf_gain,
        noise_std=args.hf_noise_std,
        snr_db=args.hf_snr_db,
        p=1.0,
        clamp=None,
    )

    # Apply transforms
    torch.manual_seed(0)  # so the plotted noise is reproducible
    x0 = base(img)        # (C,H,W)
    x1 = hf(base(img))    # (C,H,W) with HF injected

    # Flatten to sequential (L, C)
    L = 32 * 32
    x0_seq = x0.view(c, L).t()  # (L,C)
    x1_seq = x1.view(c, L).t()  # (L,C)

    # rFFT along L, average across channels
    X0 = torch.fft.rfft(x0_seq.t(), dim=-1)  # (C, F)
    X1 = torch.fft.rfft(x1_seq.t(), dim=-1)  # (C, F)
    mag0 = X0.abs().mean(dim=0).cpu().numpy()  # (F,)
    mag1 = X1.abs().mean(dim=0).cpu().numpy()  # (F,)

    F = mag0.shape[0]
    lo = int(round(args.hf_band_lo * (F - 1)))
    hi = int(round(args.hf_band_hi * (F - 1)))

    plt.figure()
    plt.plot(mag0, label="no HF")
    plt.plot(mag1, label="with HF")
    plt.axvspan(lo, hi, alpha=0.2)  # show injected band
    plt.xlabel("rFFT bin (0..Nyquist)")
    plt.ylabel("mean |X|")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"[Saved spectrum plot] {save_path}")

class HFNoiseInjectSeq1D:
    """
    Treat image as a 1D sequence by flattening spatial dims (H*W),
    then inject noise / gain in a high-frequency band near Nyquist (Ω≈π).

    - band: (lo, hi) as fraction of Nyquist in [0,1]. e.g. (0.9, 1.0) == 0.9π~π
    - mode:
        * "add": add complex Gaussian noise in the band
        * "gain": multiply existing spectrum in the band by `gain`
        * "add+gain": do both (gain first, then add)
    - noise_std: absolute std for real/imag parts (if snr_db is None)
    - snr_db: if set, noise power in the band is set relative to signal power in the band
    - p: probability to apply (use 1.0 for always)
    """

    def __init__(
        self,
        band=(0.9, 1.0),
        mode="add",
        gain=1.0,
        noise_std=0.0,
        snr_db=None,
        p=1.0,
        eps=1e-12,
        clamp=None,  # e.g. (0.0, 1.0) if you want to keep valid pixel range
    ):
        assert 0.0 <= band[0] <= band[1] <= 1.0
        assert mode in ["add", "gain", "add+gain"]
        self.band = band
        self.mode = mode
        self.gain = float(gain)
        self.noise_std = float(noise_std)
        self.snr_db = snr_db
        self.p = float(p)
        self.eps = float(eps)
        self.clamp = clamp

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(x):
            x = torch.as_tensor(x)

        # Apply with probability p (torch RNG, good with DataLoader workers)
        if self.p < 1.0 and torch.rand(()) > self.p:
            return x

        # Expect image tensor [C,H,W] (C=1 or 3 typically)
        if x.ndim != 3:
            raise ValueError(f"Expected x.ndim==3 (C,H,W), got {tuple(x.shape)}")

        C, H, W = x.shape
        L = H * W

        # Work in float32 for stable FFT
        x0 = x.to(torch.float32)
        x_seq = x0.reshape(C, L)  # [C, L]

        # rFFT along sequence axis
        X = torch.fft.rfft(x_seq, dim=-1)  # [C, L//2+1], complex

        n_bins = X.shape[-1]
        if n_bins <= 2:
            return x  # too short to meaningfully band-select

        # Map band fraction to rFFT bin indices
        # rFFT bins correspond to Ω in [0, π], where bin (n_bins-1) is Ω=π (Nyquist)
        lo = int(round(self.band[0] * (n_bins - 1)))
        hi = int(round(self.band[1] * (n_bins - 1)))
        lo = max(0, min(lo, n_bins - 1))
        hi = max(0, min(hi, n_bins - 1))
        if hi < lo:
            return x

        band_slice = slice(lo, hi + 1)
        X_band = X[..., band_slice]

        # Optional gain
        if self.mode in ["gain", "add+gain"] and self.gain != 1.0:
            X_band = X_band * self.gain

        # Optional noise add
        if self.mode in ["add", "add+gain"]:
            if self.snr_db is not None:
                # Match noise power to signal power within the band
                # SNR = P_signal / P_noise  =>  P_noise = P_signal / SNR
                snr = 10.0 ** (float(self.snr_db) / 10.0)
                P_sig = (X[..., band_slice].abs() ** 2).mean()
                P_noise = P_sig / max(snr, self.eps)
                # If noise = a + j b with a,b ~ N(0, sigma^2), then E|noise|^2 = 2*sigma^2
                sigma = torch.sqrt(P_noise / 2.0 + self.eps)
            else:
                sigma = torch.tensor(self.noise_std, dtype=torch.float32)

            if float(sigma) > 0.0:
                nr = torch.randn_like(X_band.real)
                ni = torch.randn_like(X_band.real)
                noise = torch.complex(nr, ni) * sigma
                X_band = X_band + noise

        # Write back modified band
        X[..., band_slice] = X_band

        # Keep DC (0) real, and Nyquist (last bin, if even L) real to avoid tiny numerical weirdness
        X[..., 0] = torch.complex(X[..., 0].real, torch.zeros_like(X[..., 0].real))
        if L % 2 == 0:
            X[..., -1] = torch.complex(X[..., -1].real, torch.zeros_like(X[..., -1].real))

        # iRFFT back to real sequence and reshape to image
        x_out = torch.fft.irfft(X, n=L, dim=-1).reshape(C, H, W)

        if self.clamp is not None:
            lo_c, hi_c = self.clamp
            x_out = x_out.clamp(lo_c, hi_c)

        # Cast back to original dtype if it was float-like
        if x.dtype in (torch.float16, torch.float32, torch.float64):
            x_out = x_out.to(x.dtype)

        return x_out
>>>>>>> 2a5624c (Add project fiels)
