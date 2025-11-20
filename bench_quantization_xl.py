#!/usr/bin/env python3
"""
Extra-large benchmark: four Linear layers (approx MLP stack) with float32 vs int8-quantized weights.

Defaults are heavy to amplify any potential int8 speedup:
  batch=1024, dims=8192, steps=10
"""

import argparse
import os
import sys
import time
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="XL Needle quantization benchmark.")
    p.add_argument("--batch", type=int, default=1024, help="Batch size.")
    p.add_argument("--dim", type=int, default=8192, help="Hidden dimension for all layers.")
    p.add_argument("--steps", type=int, default=10, help="Timed iterations per mode.")
    return p.parse_args()


def ensure_backend() -> None:
    if "NEEDLE_BACKEND" not in os.environ:
        os.environ["NEEDLE_BACKEND"] = "nd"
    sys.path.append("./python")


def _stack_linear(dim: int):
    import needle as ndl

    return [
        ndl.nn.Linear(dim, dim, bias=True, dtype="float32")
        for _ in range(4)
    ]


def forward(model, x, ndl):
    out = x
    for layer in model:
        out = ndl.ops.relu(layer(out))
    return out


def benchmark(dim: int, batch: int, steps: int) -> dict:
    import needle as ndl

    model = _stack_linear(dim)
    x = ndl.Tensor(np.random.randn(batch, dim).astype(np.float32), requires_grad=False)

    # Float32 path
    _ = forward(model, x, ndl)  # warmup
    t0 = time.perf_counter()
    for _ in range(steps):
        _ = forward(model, x, ndl)
    float_time = time.perf_counter() - t0

    # Quantized weights path (weights cached as float once)
    for layer in model:
        layer.eval()
        layer.enable_quantization(axis=1)
    _ = forward(model, x, ndl)  # warmup
    t1 = time.perf_counter()
    for _ in range(steps):
        _ = forward(model, x, ndl)
    int8_time = time.perf_counter() - t1

    return {
        "float32_total_s": float_time,
        "int8_total_s": int8_time,
        "float32_avg_ms": float_time / steps * 1e3,
        "int8_avg_ms": int8_time / steps * 1e3,
        "speedup_x": float_time / int8_time if int8_time > 0 else float("inf"),
    }


def main() -> None:
    args = parse_args()
    ensure_backend()
    print(f"Backend: {os.environ.get('NEEDLE_BACKEND', 'nd')}")
    res = benchmark(dim=args.dim, batch=args.batch, steps=args.steps)
    print(f"float32 avg: {res['float32_avg_ms']:.3f} ms")
    print(f"int8 avg:   {res['int8_avg_ms']:.3f} ms")
    print(f"speedup:    {res['speedup_x']:.2f}x")
    print("\nFull results:", res)


if __name__ == "__main__":
    main()
