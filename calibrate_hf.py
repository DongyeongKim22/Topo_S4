import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from HF_noise_injection import HFNoiseInjectSeq1D
class FlattenToSeq:
    """(C,H,W) -> (L,C) where L=H*W"""
    def __init__(self, c: int):
        self.c = c

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x: (C,H,W)
        return x.view(self.c, -1).t().contiguous()

def rms(x): return torch.sqrt(torch.mean(x**2))

def measure(noise_std, n=200):
    base = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010)),
    ])
    hf = HFNoiseInjectSeq1D(band=(0.9,1.0), mode="add", noise_std=noise_std, p=1.0)

    ds = torchvision.datasets.CIFAR10(root="./data/cifar/", train=True, download=True, transform=None)
    ratios = []
    torch.manual_seed(0)
    for i in range(n):
        img, _ = ds[i]
        x0 = base(img)             # (3,32,32)
        x1 = hf(base(img))
        ratio = (rms(x1-x0) / (rms(x0) + 1e-12)).item()
        ratios.append(ratio)
    return float(np.mean(ratios)), float(np.std(ratios))

for s in [0.0, 0.1, 0.2, 0.3, 0.5, 0.8]:
    m, sd = measure(s)
    print(f"noise_std={s:.2f}  delta_rms_ratio={m:.3f} ± {sd:.3f}")
