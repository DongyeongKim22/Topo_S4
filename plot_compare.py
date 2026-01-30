<<<<<<< HEAD
import json
import argparse
import matplotlib.pyplot as plt

def load(path):
    with open(path, "r") as f:
        return json.load(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log_a", type=str)
    ap.add_argument("log_b", type=str)
    ap.add_argument("--out", type=str, default="compare.png")
    args = ap.parse_args()

    A = load(args.log_a)
    B = load(args.log_b)

    ea = A["epoch"]
    eb = B["epoch"]

    # 1) Accuracy
    plt.figure()
    plt.plot(ea, A["train_acc"], label=f'{A["run_name"]} train')
    plt.plot(ea, A["val_acc"],   label=f'{A["run_name"]} val')
    plt.plot(ea, A["test_acc"],  label=f'{A["run_name"]} test')
    plt.plot(eb, B["train_acc"], label=f'{B["run_name"]} train')
    plt.plot(eb, B["val_acc"],   label=f'{B["run_name"]} val')
    plt.plot(eb, B["test_acc"],  label=f'{B["run_name"]} test')
    plt.xlabel("epoch")
    plt.ylabel("accuracy (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("acc_" + args.out, dpi=150)

    # 2) Loss
    plt.figure()
    plt.plot(ea, A["train_loss"], label=f'{A["run_name"]} train')
    plt.plot(ea, A["val_loss"],   label=f'{A["run_name"]} val')
    plt.plot(ea, A["test_loss"],  label=f'{A["run_name"]} test')
    plt.plot(eb, B["train_loss"], label=f'{B["run_name"]} train')
    plt.plot(eb, B["val_loss"],   label=f'{B["run_name"]} val')
    plt.plot(eb, B["test_loss"],  label=f'{B["run_name"]} test')
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("loss_" + args.out, dpi=150)

    print(f"Saved: acc_{args.out}, loss_{args.out}")

if __name__ == "__main__":
    main()
=======
import json
import argparse
import matplotlib.pyplot as plt

def load(path):
    with open(path, "r") as f:
        return json.load(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log_a", type=str)
    ap.add_argument("log_b", type=str)
    ap.add_argument("--out", type=str, default="compare.png")
    args = ap.parse_args()

    A = load(args.log_a)
    B = load(args.log_b)

    ea = A["epoch"]
    eb = B["epoch"]

    # 1) Accuracy
    plt.figure()
    plt.plot(ea, A["train_acc"], label=f'{A["run_name"]} train')
    plt.plot(ea, A["val_acc"],   label=f'{A["run_name"]} val')
    plt.plot(ea, A["test_acc"],  label=f'{A["run_name"]} test')
    plt.plot(eb, B["train_acc"], label=f'{B["run_name"]} train')
    plt.plot(eb, B["val_acc"],   label=f'{B["run_name"]} val')
    plt.plot(eb, B["test_acc"],  label=f'{B["run_name"]} test')
    plt.xlabel("epoch")
    plt.ylabel("accuracy (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("acc_" + args.out, dpi=150)

    # 2) Loss
    plt.figure()
    plt.plot(ea, A["train_loss"], label=f'{A["run_name"]} train')
    plt.plot(ea, A["val_loss"],   label=f'{A["run_name"]} val')
    plt.plot(ea, A["test_loss"],  label=f'{A["run_name"]} test')
    plt.plot(eb, B["train_loss"], label=f'{B["run_name"]} train')
    plt.plot(eb, B["val_loss"],   label=f'{B["run_name"]} val')
    plt.plot(eb, B["test_loss"],  label=f'{B["run_name"]} test')
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("loss_" + args.out, dpi=150)

    print(f"Saved: acc_{args.out}, loss_{args.out}")

if __name__ == "__main__":
    main()
>>>>>>> 2a5624c (Add project fiels)
