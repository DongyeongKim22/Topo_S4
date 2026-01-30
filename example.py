'''
Train an S4 model on sequential CIFAR10 / sequential MNIST with PyTorch for demonstration purposes.
This code borrows heavily from https://github.com/kuangliu/pytorch-cifar.

This file only depends on the standalone S4 layer
available in /models/s4/

* Train standard sequential CIFAR:
    python -m example
* Train sequential CIFAR grayscale:
    python -m example --grayscale
* Train MNIST:
    python -m example --dataset mnist --d_model 256 --weight_decay 0.0

The `S4Model` class defined in this file provides a simple backbone to train S4 models.
This backbone is a good starting point for many problems, although some tasks (especially generation)
may require using other backbones.

The default CIFAR10 model trained by this file should get
89+% accuracy on the CIFAR10 test set in 80 epochs.

Each epoch takes approximately 7m20s on a T4 GPU (will be much faster on V100 / A100).
'''

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os, sys
import argparse
import math, time, json, random
from datetime import datetime
import numpy as np

from models.s4.s4 import S4Block as S4  # Can use full version instead of minimal S4D standalone below
from models.s4.s4d import S4D
from tqdm.auto import tqdm

from HF_noise_injection import *
class FlattenToSeq:
    """(C,H,W) -> (L,C) where L=H*W"""
    def __init__(self, c: int):
        self.c = c

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x: (C,H,W)
        return x.view(self.c, -1).t().contiguous()
    
def main():
    # Dropout broke in PyTorch 1.11
    ver = tuple(map(int, torch.__version__.split('.')[:2]))
    if ver == (1, 11):
        print("WARNING: Dropout is bugged in PyTorch 1.11. Results may be worse.")
        dropout_fn = nn.Dropout
    elif ver >= (1, 12):
        dropout_fn = nn.Dropout1d
    else:
        dropout_fn = nn.Dropout2d



    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    # Optimizer
    parser.add_argument('--lr', default=0.01, type=float, help='Learning rate')
    parser.add_argument('--weight_decay', default=0.01, type=float, help='Weight decay')
    # Scheduler
    # parser.add_argument('--patience', default=10, type=float, help='Patience for learning rate scheduler')
    parser.add_argument('--epochs', default=100, type=int, help='Training epochs')
    # Dataset
    parser.add_argument('--dataset', default='cifar10', choices=['mnist', 'cifar10'], type=str, help='Dataset')
    parser.add_argument('--grayscale', action='store_true', help='Use grayscale CIFAR10')
    # Dataloader
    parser.add_argument('--num_workers', default=4, type=int, help='Number of workers to use for dataloader')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
    # Model
    parser.add_argument('--n_layers', default=4, type=int, help='Number of layers')
    parser.add_argument('--d_model', default=128, type=int, help='Model dimension')
    parser.add_argument('--dropout', default=0.1, type=float, help='Dropout')
    parser.add_argument('--prenorm', action='store_true', help='Prenorm')
    # General
    parser.add_argument('--resume', '-r', action='store_true', help='Resume from checkpoint')
    # HF noise injection (train-time corruption / augmentation)
    parser.add_argument('--hf', action='store_true', help='Enable HF noise injection on train set')
    parser.add_argument('--hf_eval', action='store_true', help='Also apply HF noise injection on val/test')
    parser.add_argument('--hf_band_lo', default=0.9, type=float, help='Band low edge as fraction of Nyquist (0~1)')
    parser.add_argument('--hf_band_hi', default=1.0, type=float, help='Band high edge as fraction of Nyquist (0~1)')
    parser.add_argument('--hf_mode', default='add', choices=['add', 'gain', 'add+gain'])
    parser.add_argument('--hf_gain', default=1.0, type=float)
    parser.add_argument('--hf_noise_std', default=0.0, type=float, help='Absolute complex noise std (if hf_snr_db is None)')
    parser.add_argument('--hf_snr_db', default=None, type=float, help='If set, noise power in band matches this SNR (dB)')
    parser.add_argument('--hf_p', default=1.0, type=float, help='Probability to apply per sample')
    parser.add_argument('--plot_hf_example', action='store_true',
                        help='Plot spectrum of one example (with/without HF) and exit')

    parser.add_argument('--seed', default=0, type=int)
    # Stability logging
    parser.add_argument('--loss_ema_decay', default=0.98, type=float)
    parser.add_argument('--spike_factor', default=3.0, type=float)
    parser.add_argument('--grad_log_every', default=1, type=int)  # 매 배치 gradnorm 계산(1) / N배치마다
    parser.add_argument('--abort_on_nonfinite', action='store_true')
    # Logging / plotting
    parser.add_argument('--run_name', default=None, type=str, help='Name for this run (used in log filename)')
    parser.add_argument('--log_dir', default='./logs', type=str, help='Directory to save json logs')

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    global best_acc
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    def set_seed(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    set_seed(args.seed)

    def grad_norm_l2(model: nn.Module) -> float:
        total = 0.0
        for p in model.parameters():
            if p.grad is None:
                continue
            g = p.grad.detach()
            # g.norm()이 NaN/Inf면 바로 잡힘
            gn = torch.norm(g, p=2)
            if not torch.isfinite(gn):
                return float('nan')
            total += (gn.item() ** 2)
        return math.sqrt(total)
    # Data
    print(f'==> Preparing {args.dataset} data..')

    def split_train_val(train, val_split):
        train_len = int(len(train) * (1.0-val_split))
        train, val = torch.utils.data.random_split(
            train,
            (train_len, len(train) - train_len),
            generator=torch.Generator().manual_seed(42),
        )
        return train, val

    if args.dataset == 'cifar10':

        # HF injector (only used if args.hf or args.hf_eval)
        hf = HFNoiseInjectSeq1D(
            band=(args.hf_band_lo, args.hf_band_hi),
            mode=args.hf_mode,
            gain=args.hf_gain,
            noise_std=args.hf_noise_std,
            snr_db=args.hf_snr_db,
            p=args.hf_p,
            clamp=None,  # NOTE: keep None if applying after Normalize
        )

        if args.grayscale:
            c = 1
            # NOTE: Normalize는 (mean,), (std,) 형태가 안전합니다.
            base = [
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize((122.6/255.0,), (61.0/255.0,)),
            ]
            # seq = transforms.Lambda(lambda x: x.view(c, 1024).t())  # (1,32,32)->(1024,1)
            seq = FlattenToSeq(1)
  # (1,32,32)->(1024,1)
        else:
            c = 3
            base = [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
            # seq = transforms.Lambda(lambda x: x.view(c, 1024).t())  # (3,32,32)->(1024,3)
            seq = FlattenToSeq(3)
  # (3,32,32)->(1024,3)

        # Train transform: base + (optional HF) + flatten-to-seq
        train_tf = list(base)
        if args.hf:
            train_tf.append(hf)   # <-- 여기에서 HF 주입
        train_tf.append(seq)
        transform_train = transforms.Compose(train_tf)

        # Test/Val transform: base + (optional HF if hf_eval) + flatten-to-seq
        test_tf = list(base)
        if args.hf and args.hf_eval:
            test_tf.append(hf)    # <-- val/test에도 넣고 싶으면 --hf_eval
        test_tf.append(seq)
        transform_test = transforms.Compose(test_tf)

        trainset = torchvision.datasets.CIFAR10(
            root='./data/cifar/', train=True, download=True, transform=transform_train)
        trainset, _ = split_train_val(trainset, val_split=0.1)

        valset = torchvision.datasets.CIFAR10(
            root='./data/cifar/', train=True, download=True, transform=transform_test)
        _, valset = split_train_val(valset, val_split=0.1)

        testset = torchvision.datasets.CIFAR10(
            root='./data/cifar/', train=False, download=True, transform=transform_test)

        d_input = 3 if not args.grayscale else 1
        d_output = 10
        if args.plot_hf_example:
            plot_one_example_spectrum_cifar(args, save_path="hf_spectrum.png")
            raise SystemExit

    elif args.dataset == 'mnist':

        hf = HFNoiseInjectSeq1D(
            band=(args.hf_band_lo, args.hf_band_hi),
            mode=args.hf_mode,
            gain=args.hf_gain,
            noise_std=args.hf_noise_std,
            snr_db=args.hf_snr_db,
            p=args.hf_p,
            clamp=None,
        )

        base = [transforms.ToTensor()]
        seq  = transforms.Lambda(lambda x: x.view(1, 784).t())

        train_tf = list(base)
        if args.hf:
            train_tf.append(hf)
        train_tf.append(seq)
        transform_train = transforms.Compose(train_tf)

        test_tf = list(base)
        if args.hf and args.hf_eval:
            test_tf.append(hf)
        test_tf.append(seq)
        transform_test = transforms.Compose(test_tf)

        trainset = torchvision.datasets.MNIST(
            root='./data', train=True, download=True, transform=transform_train)
        trainset, _ = split_train_val(trainset, val_split=0.1)

        valset = torchvision.datasets.MNIST(
            root='./data', train=True, download=True, transform=transform_test)
        _, valset = split_train_val(valset, val_split=0.1)

        testset = torchvision.datasets.MNIST(
            root='./data', train=False, download=True, transform=transform_test)

        d_input = 1
        d_output = 10
    else: raise NotImplementedError

    # Dataloaders
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valloader = torch.utils.data.DataLoader(
        valset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    class S4Model(nn.Module):

        def __init__(
            self,
            d_input,
            d_output=10,
            d_model=256,
            n_layers=4,
            dropout=0.2,
            prenorm=False,
        ):
            super().__init__()

            self.prenorm = prenorm

            # Linear encoder (d_input = 1 for grayscale and 3 for RGB)
            self.encoder = nn.Linear(d_input, d_model)

            # Stack S4 layers as residual blocks
            self.s4_layers = nn.ModuleList()
            self.norms = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            for _ in range(n_layers):
                self.s4_layers.append(
                    S4D(d_model, dropout=dropout, transposed=True, lr=min(0.001, args.lr))
                )
                self.norms.append(nn.LayerNorm(d_model))
                self.dropouts.append(dropout_fn(dropout))

            # Linear decoder
            self.decoder = nn.Linear(d_model, d_output)

        def forward(self, x):
            """
            Input x is shape (B, L, d_input)
            """
            x = self.encoder(x)  # (B, L, d_input) -> (B, L, d_model)

            x = x.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L)
            for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
                # Each iteration of this loop will map (B, d_model, L) -> (B, d_model, L)

                z = x
                if self.prenorm:
                    # Prenorm
                    z = norm(z.transpose(-1, -2)).transpose(-1, -2)

                # Apply S4 block: we ignore the state input and output
                z, _ = layer(z)

                # Dropout on the output of the S4 block
                z = dropout(z)

                # Residual connection
                x = z + x

                if not self.prenorm:
                    # Postnorm
                    x = norm(x.transpose(-1, -2)).transpose(-1, -2)

            x = x.transpose(-1, -2)

            # Pooling: average pooling over the sequence length
            x = x.mean(dim=1)

            # Decode the outputs
            x = self.decoder(x)  # (B, d_model) -> (B, d_output)

            return x

    # Model
    print('==> Building model..')
    model = S4Model(
        d_input=d_input,
        d_output=d_output,
        d_model=args.d_model,
        n_layers=args.n_layers,
        dropout=args.dropout,
        prenorm=args.prenorm,
    )

    model = model.to(device)
    if device == 'cuda':
        cudnn.benchmark = True

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt.pth')
        model.load_state_dict(checkpoint['model'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    def setup_optimizer(model, lr, weight_decay, epochs):
        """
        S4 requires a specific optimizer setup.

        The S4 layer (A, B, C, dt) parameters typically
        require a smaller learning rate (typically 0.001), with no weight decay.

        The rest of the model can be trained with a higher learning rate (e.g. 0.004, 0.01)
        and weight decay (if desired).
        """

        # All parameters in the model
        all_parameters = list(model.parameters())

        # General parameters don't contain the special _optim key
        params = [p for p in all_parameters if not hasattr(p, "_optim")]

        # Create an optimizer with the general parameters
        optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)

        # Add parameters with special hyperparameters
        hps = [getattr(p, "_optim") for p in all_parameters if hasattr(p, "_optim")]
        hps = [
            dict(s) for s in sorted(list(dict.fromkeys(frozenset(hp.items()) for hp in hps)))
        ]  # Unique dicts
        for hp in hps:
            params = [p for p in all_parameters if getattr(p, "_optim", None) == hp]
            optimizer.add_param_group(
                {"params": params, **hp}
            )

        # Create a lr scheduler
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=0.2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

        # Print optimizer info
        keys = sorted(set([k for hp in hps for k in hp.keys()]))
        for i, g in enumerate(optimizer.param_groups):
            group_hps = {k: g.get(k, None) for k in keys}
            print(' | '.join([
                f"Optimizer group {i}",
                f"{len(g['params'])} tensors",
            ] + [f"{k} {v}" for k, v in group_hps.items()]))

        return optimizer, scheduler

    criterion = nn.CrossEntropyLoss()
    optimizer, scheduler = setup_optimizer(
        model, lr=args.lr, weight_decay=args.weight_decay, epochs=args.epochs
    )

    ###############################################################################
    # Everything after this point is standard PyTorch training!
    ###############################################################################

    # Training
    def train():
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        grad_norms = []
        nonfinite_loss_batches = 0
        nonfinite_grad_batches = 0
        loss_spikes = 0

        ema = None

        t0 = time.time()
        pbar = tqdm(
                    enumerate(trainloader),
                    total=len(trainloader),
                    file=sys.stdout,      # 중요
                    leave=False,
                    mininterval=1.0,
                    miniters=500,
                )

        for batch_idx, (inputs, targets) in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            loss_val = float(loss.detach().cpu())
            if not math.isfinite(loss_val):
                nonfinite_loss_batches += 1
                if args.abort_on_nonfinite:
                    break
                else:
                    continue

            # spike check (EMA 기준)
            if ema is None:
                ema = loss_val
            else:
                if loss_val > args.spike_factor * ema:
                    loss_spikes += 1
                ema = args.loss_ema_decay * ema + (1 - args.loss_ema_decay) * loss_val

            loss.backward()

            # grad norm logging
            if (batch_idx % args.grad_log_every) == 0:
                gn = grad_norm_l2(model)
                grad_norms.append(gn)
                if not math.isfinite(gn):
                    nonfinite_grad_batches += 1
                    if args.abort_on_nonfinite:
                        break

            optimizer.step()

            bs = targets.size(0)
            total_loss += loss_val * bs
            _, predicted = outputs.max(1)
            total += bs
            correct += predicted.eq(targets).sum().item()

            # pbar.set_description(
            #     f"Train | Loss {total_loss/max(total,1):.3f} | Acc {100.*correct/max(total,1):.2f}%"
            # )

        epoch_time = time.time() - t0

        avg_loss = total_loss / max(total, 1)
        avg_acc = 100. * correct / max(total, 1)

        # grad norm stats
        g = np.array([x for x in grad_norms if np.isfinite(x)], dtype=np.float64)
        if g.size > 0:
            gn_p95 = float(np.percentile(g, 95))
            gn_p99 = float(np.percentile(g, 99))
            gn_max = float(np.max(g))
        else:
            gn_p95 = float('nan')
            gn_p99 = float('nan')
            gn_max = float('nan')

        metrics = dict(
            epoch_time_sec=float(epoch_time),
            loss_spikes=int(loss_spikes),
            nonfinite_loss_batches=int(nonfinite_loss_batches),
            nonfinite_grad_batches=int(nonfinite_grad_batches),
            grad_norm_p95=gn_p95,
            grad_norm_p99=gn_p99,
            grad_norm_max=gn_max,
            aborted=bool(args.abort_on_nonfinite and (nonfinite_loss_batches > 0 or nonfinite_grad_batches > 0)),
        )
        return avg_loss, avg_acc, metrics


    def eval(epoch, dataloader, checkpoint=False):
        global best_acc
        model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            pbar = tqdm(enumerate(dataloader), total=len(dataloader))
            for batch_idx, (inputs, targets) in pbar:
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                bs = targets.size(0)
                total_loss += loss.item() * bs
                _, predicted = outputs.max(1)
                total += bs
                correct += predicted.eq(targets).sum().item()

                # pbar.set_description(
                #     'Eval  | Batch (%d/%d) | Loss: %.3f | Acc: %.2f%% (%d/%d)' %
                #     (batch_idx, len(dataloader),
                #     total_loss / max(total, 1),
                #     100. * correct / max(total, 1),
                #     correct, total)
                # )

        avg_loss = total_loss / max(total, 1)
        acc = 100. * correct / max(total, 1)

        # Save checkpoint.
        if checkpoint and acc > best_acc:
            state = {'model': model.state_dict(), 'acc': acc, 'epoch': epoch}
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/ckpt.pth')
            best_acc = acc

        return avg_loss, acc

    os.makedirs(args.log_dir, exist_ok=True)
    run_name = args.run_name
    if run_name is None:
        tag = f"{args.dataset}"
        if args.hf:
            tag += f"_HF-{args.hf_mode}"
            if args.hf_snr_db is not None:
                tag += f"_snr{args.hf_snr_db}"
            else:
                tag += f"_std{args.hf_noise_std}"
        else:
            tag += "_HF0"
        tag += f"_seed{args.seed}"
        run_name = datetime.now().strftime(f"{tag}_%Y%m%d-%H%M%S")

    log_path = os.path.join(args.log_dir, f"{run_name}.json")
    history = {
        "run_name": run_name,
        "args": vars(args),
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "test_loss": [],
        "test_acc": [],
        "lr": [],
    }
    history.update({
        "train_grad_norm_p95": [],
        "train_grad_norm_p99": [],
        "train_grad_norm_max": [],
        "train_loss_spikes": [],
        "train_nonfinite_loss_batches": [],
        "train_nonfinite_grad_batches": [],
        "epoch_time_sec": [],
        "aborted_epoch": None,
    })

    pbar = tqdm(range(start_epoch, args.epochs))
    for epoch in pbar:
        # train
        tr_loss, tr_acc, trm = train()
        val_loss, val_acc = eval(epoch, valloader, checkpoint=True)
        te_loss, te_acc = eval(epoch, testloader)

        scheduler.step()

        # log
        history["epoch"].append(epoch)
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["test_loss"].append(te_loss)
        history["test_acc"].append(te_acc)
        history["lr"].append(scheduler.get_last_lr()[0])
        history["train_grad_norm_p95"].append(trm["grad_norm_p95"])
        history["train_grad_norm_p99"].append(trm["grad_norm_p99"])
        history["train_grad_norm_max"].append(trm["grad_norm_max"])
        history["train_loss_spikes"].append(trm["loss_spikes"])
        history["train_nonfinite_loss_batches"].append(trm["nonfinite_loss_batches"])
        history["train_nonfinite_grad_batches"].append(trm["nonfinite_grad_batches"])
        history["epoch_time_sec"].append(trm["epoch_time_sec"])
        
        if trm["aborted"] and history["aborted_epoch"] is None:
            history["aborted_epoch"] = epoch
            # 로그 저장 후 중단
            with open(log_path, "w") as f:
                json.dump(history, f, indent=2)
            print(f"[ABORT] non-finite detected at epoch {epoch}. Saved log: {log_path}")
            break
        # progress bar text
        # pbar.set_description(
        #     f"Epoch {epoch} | "
        #     f"tr_acc {tr_acc:.2f} | val_acc {val_acc:.2f} | te_acc {te_acc:.2f}"
        # )

        # save json every epoch (중간에 죽어도 로그 남게)
        with open(log_path, "w") as f:
            json.dump(history, f, indent=2)

    print(f"[Saved log] {log_path}")

if __name__ == "__main__":
    main()