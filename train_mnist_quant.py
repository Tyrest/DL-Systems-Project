#!/usr/bin/env python3
"""
Train a simple MNIST MLP and report accuracy for:
1) float32 model
2) weight-quantized (int8) model

Usage (defaults are light for demo):
  python train_mnist_quant.py --epochs 1 --batch-size 256 --lr 0.01
"""

import argparse
import os
import sys
import numpy as np
import time
from pathlib import Path

def parse_args() -> argparse.Namespace:
    """CLI args for MNIST training demo (float32 vs quantized)."""
    p = argparse.ArgumentParser(description="Train MNIST float32 vs int8 models.")
    p.add_argument("--data-dir", type=str, default="data", help="Where to store MNIST gzip files.")
    p.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    p.add_argument("--batch-size", type=int, default=256, help="Batch size.")
    p.add_argument("--lr", type=float, default=0.01, help="Learning rate.")
    p.add_argument("--hidden", type=int, default=512, help="Hidden size for MLP.")
    p.add_argument("--device", type=str, default="cpu", help="Device to run on (cpu or cuda)")
    return p.parse_args()


def ensure_backend() -> None:
    """Ensure Needle backend env and import path are set."""
    if "NEEDLE_BACKEND" not in os.environ:
        os.environ["NEEDLE_BACKEND"] = "nd"
    sys.path.append("./python")


def build_model(in_dim: int, hidden: int, out_dim: int, device=None):
    import needle as ndl

    return ndl.nn.Sequential(
        ndl.nn.Flatten(),
        ndl.nn.Linear(in_dim, hidden, device=device),
        ndl.nn.ReLU(),
        ndl.nn.Linear(hidden, out_dim, device=device),
    )


def accuracy(model, dataloader, device=None):
    """Compute accuracy over a dataset using mini-batches."""
    import needle as ndl

    correct = 0
    total = 0
    model.eval()
    for batch in dataloader:
        xb, yb = batch
        xb, yb = ndl.Tensor(xb, device=device), ndl.Tensor(yb, device=device)
        logits = model(xb)
        preds = np.argmax(logits.numpy(), axis=1)
        correct += (preds == yb.numpy()).sum()
        total += yb.shape[0]
    return correct / total


def main() -> None:
    """Train/evaluate separate float32 and int8-weight MNIST models and report stats."""
    args = parse_args()
    ensure_backend()
    import needle as ndl
    from needle.data import MNISTDataset, DataLoader

    if args.device == "cpu":
        device = ndl.cpu()
    elif args.device == "cuda":
        device = ndl.cuda()
    else:
        raise ValueError(f"Unknown device: {args.device}")

    # Data
    data_dir = Path(args.data_dir)
    train_dataset = MNISTDataset(
        str(data_dir / "train-images-idx3-ubyte.gz"),
        str(data_dir / "train-labels-idx1-ubyte.gz")
    )
    test_dataset = MNISTDataset(
        str(data_dir / "t10k-images-idx3-ubyte.gz"),
        str(data_dir / "t10k-labels-idx1-ubyte.gz")
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    # Input dim is 28*28 = 784
    in_dim, out_dim = 784, 10
    model = build_model(in_dim, args.hidden, out_dim, device=device)
    loss_fn = ndl.nn.SoftmaxLoss()

    def train_model(model, opt, label):
        """Train a model for the configured epochs and return elapsed time."""
        model.train()
        t0 = time.perf_counter()
        for epoch in range(args.epochs):
            losses = []
            for batch in train_loader:
                xb, yb = batch
                xb, yb = ndl.Tensor(xb, device=device), ndl.Tensor(yb, device=device)
                logits = model(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                opt.step()
                opt.reset_grad()
                losses.append(loss.numpy())
            print(f"[{label}] Epoch {epoch+1}: loss={np.mean(losses):.4f}")
        return time.perf_counter() - t0

    # Train float32 model
    opt_float = ndl.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    train_time_float = train_model(model, opt_float, "float32")
    
    t_eval_float = time.perf_counter()
    acc_float = accuracy(model, test_loader, device=device)
    eval_float_time = time.perf_counter() - t_eval_float

    # Quantize and evaluate the SAME model
    print("Quantizing model...")
    model.eval()
    # Enable quantization for Linear layers
    model.quantize(axis=1)
    
    # Enable int8 matmul kernel
    os.environ["NEEDLE_USE_INT8_MATMUL"] = "1"
            
    t_eval_int8 = time.perf_counter()
    acc_int8 = accuracy(model, test_loader, device=device)
    eval_int8_time = time.perf_counter() - t_eval_int8

    print(f"Float32 accuracy: {acc_float*100:.2f}%")
    print(f"Int8 (weights) accuracy: {acc_int8*100:.2f}%")
    print(f"Eval times: float={eval_float_time:.2f}s | int8={eval_int8_time:.2f}s")


if __name__ == "__main__":
    main()
