$ErrorActionPreference = "Stop"

$LOGDIR = ".\logs_sweep"
$EPOCHS = 50
$SEEDS = @(0,1,2)

$MID_STD = 0.20
$HIGH_STD = 0.50

New-Item -ItemType Directory -Force -Path $LOGDIR | Out-Null

foreach ($s in $SEEDS) {
  # HF=0
  python -m example --dataset cifar10 --epochs $EPOCHS --seed $s `
    --run_name "cifar_seed${s}_hf0" --log_dir $LOGDIR `
    --abort_on_nonfinite

  # HF=mid
  python -m example --dataset cifar10 --epochs $EPOCHS --seed $s `
    --hf --hf_mode add --hf_band_lo 0.9 --hf_band_hi 1.0 --hf_noise_std $MID_STD `
    --run_name "cifar_seed${s}_hfmid" --log_dir $LOGDIR `
    --abort_on_nonfinite

  # HF=high
  python -m example --dataset cifar10 --epochs $EPOCHS --seed $s `
    --hf --hf_mode add --hf_band_lo 0.9 --hf_band_hi 1.0 --hf_noise_std $HIGH_STD `
    --run_name "cifar_seed${s}_hfhigh" --log_dir $LOGDIR `
    --abort_on_nonfinite
}
