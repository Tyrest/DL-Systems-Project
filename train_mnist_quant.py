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
import gzip
import urllib.request
from pathlib import Path
import numpy as np
import time


MNIST_URLS = {
    "train-images": [
        "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
        "https://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
        "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
    ],
    "train-labels": [
        "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
        "https://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
        "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
    ],
    "test-images": [
        "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
        "https://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
        "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
    ],
    "test-labels": [
        "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
        "https://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz",
        "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz",
    ],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train MNIST float32 vs int8 models.")
    p.add_argument("--data-dir", type=str, default="data/mnist", help="Where to store MNIST gzip files.")
    p.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    p.add_argument("--batch-size", type=int, default=256, help="Batch size.")
    p.add_argument("--lr", type=float, default=0.01, help="Learning rate.")
    p.add_argument("--hidden", type=int, default=512, help="Hidden size for MLP.")
    return p.parse_args()


def ensure_backend() -> None:
    if "NEEDLE_BACKEND" not in os.environ:
        os.environ["NEEDLE_BACKEND"] = "nd"
    sys.path.append("./python")


def download_mnist(data_dir: Path) -> dict:
    data_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for key, url_list in MNIST_URLS.items():
        fname = data_dir / f"{key}.gz"
        paths[key] = fname
        if not fname.exists():
            success = False
            for url in url_list:
                try:
                    print(f"Downloading {url} -> {fname}")
                    urllib.request.urlretrieve(url, fname)
                    success = True
                    break
                except Exception as e:
                    print(f"Failed {url}: {e}")
            if not success:
                raise RuntimeError(f"Could not download {key} from any mirror")
    return paths


def load_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic = int.from_bytes(f.read(4), "big")
        assert magic == 2051
        num = int.from_bytes(f.read(4), "big")
        rows = int.from_bytes(f.read(4), "big")
        cols = int.from_bytes(f.read(4), "big")
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)
        return (data.astype(np.float32) / 255.0).reshape(num, -1)


def load_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic = int.from_bytes(f.read(4), "big")
        assert magic == 2049
        num = int.from_bytes(f.read(4), "big")
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        assert labels.shape[0] == num
        return labels


def iterate_batches(X, y, batch_size, shuffle=True):
    idx = np.arange(len(X))
    if shuffle:
        np.random.shuffle(idx)
    for i in range(0, len(idx), batch_size):
        batch_idx = idx[i : i + batch_size]
        yield X[batch_idx], y[batch_idx]


def build_model(in_dim: int, hidden: int, out_dim: int):
    import needle as ndl

    return ndl.nn.Sequential(
        ndl.nn.Linear(in_dim, hidden),
        ndl.nn.ReLU(),
        ndl.nn.Linear(hidden, out_dim),
    )


def accuracy(model, X, y, batch_size=512):
    import needle as ndl

    correct = 0
    total = 0
    for xb, yb in iterate_batches(X, y, batch_size, shuffle=False):
        logits = model(ndl.Tensor(xb, requires_grad=False))
        preds = np.argmax(logits.numpy(), axis=1)
        correct += (preds == yb).sum()
        total += len(yb)
    return correct / total


def main() -> None:
    args = parse_args()
    ensure_backend()
    import needle as ndl

    # Data
    paths = download_mnist(Path(args.data_dir))
    X_train = load_images(paths["train-images"])
    y_train = load_labels(paths["train-labels"])
    X_test = load_images(paths["test-images"])
    y_test = load_labels(paths["test-labels"])

    in_dim, out_dim = X_train.shape[1], 10
    model = build_model(in_dim, args.hidden, out_dim)
    loss_fn = ndl.nn.SoftmaxLoss()

    def train_model(model, opt, label):
        model.train()
        t0 = time.perf_counter()
        for epoch in range(args.epochs):
            losses = []
            for xb, yb in iterate_batches(X_train, y_train, args.batch_size, shuffle=True):
                xb_t = ndl.Tensor(xb, requires_grad=False)
                yb_t = ndl.Tensor(yb, requires_grad=False)
                logits = model(xb_t)
                loss = loss_fn(logits, yb_t)
                loss.backward()
                opt.step()
                opt.reset_grad()
                losses.append(loss.numpy())
            print(f"[{label}] Epoch {epoch+1}: loss={np.mean(losses):.4f}")
        return time.perf_counter() - t0

    # Train float32 model
    opt_float = ndl.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    train_time_float = train_model(model, opt_float, "float32")
    model.eval()
    t_eval_float = time.perf_counter()
    acc_float = accuracy(model, X_test, y_test, batch_size=512)
    eval_float_time = time.perf_counter() - t_eval_float

    # Independent int8 model (trained separately)
    model_int8 = build_model(in_dim, args.hidden, out_dim)
    opt_int8 = ndl.optim.SGD(model_int8.parameters(), lr=args.lr, momentum=0.9)
    train_time_int8 = train_model(model_int8, opt_int8, "int8-train")
    model_int8.eval()
    model_int8.quantize()
    t_eval_int8 = time.perf_counter()
    acc_int8 = accuracy(model_int8, X_test, y_test, batch_size=512)
    eval_int8_time = time.perf_counter() - t_eval_int8

    # Memory footprint
    float_params = sum(int(np.prod(p.shape)) for p in model.parameters())
    float_mem = float_params * 4  # bytes
    quant_bytes = 0
    for m in model_int8._children():
        if hasattr(m, "_weight_q") and m._weight_q is not None:
            quant_bytes += m._weight_q.memory_bytes()
            if hasattr(m, "bias"):
                quant_bytes += m.bias.detach().numpy().size * 4

    print(f"Float32 accuracy: {acc_float*100:.2f}%")
    print(f"Int8 (weights) accuracy: {acc_int8*100:.2f}%")
    print(f"Float32 params: {float_params:,}, approx memory: {float_mem/1e6:.3f} MB")
    print(f"Int8 weight memory (with scale/zp + bias): {quant_bytes/1e6:.3f} MB")
    print(f"Eval times: float={eval_float_time:.2f}s | int8={eval_int8_time:.2f}s")


if __name__ == "__main__":
    main()
