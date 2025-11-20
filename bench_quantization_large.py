#!/usr/bin/env python3
"""
Larger-mode benchmark: two Linear layers (feedforward-style) with float32 vs int8-quantized weights.

Defaults are tuned to be heavier than the small benchmark:
  batch=512, in_features=4096, hidden=4096, out_features=4096, steps=20
"""

import argparse
import os
import sys
import time
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Large-mode Needle quantization benchmark.")
    parser.add_argument("--batch", type=int, default=512, help="Batch size.")
    parser.add_argument("--in-features", type=int, default=4096, help="Input feature dimension.")
    parser.add_argument("--hidden", type=int, default=4096, help="Hidden feature dimension.")
    parser.add_argument("--out-features", type=int, default=4096, help="Output feature dimension.")
    parser.add_argument("--steps", type=int, default=20, help="Timed iterations per mode.")
    return parser.parse_args()


def ensure_backend() -> None:
    if "NEEDLE_BACKEND" not in os.environ:
        os.environ["NEEDLE_BACKEND"] = "nd"
    sys.path.append("./python")


def benchmark_mlp(batch: int, in_features: int, hidden: int, out_features: int, steps: int) -> dict:
    import needle as ndl

    l1 = ndl.nn.Linear(in_features, hidden, bias=True, dtype="float32")
    l2 = ndl.nn.Linear(hidden, out_features, bias=True, dtype="float32")
    x = ndl.Tensor(
        np.random.randn(batch, in_features).astype(np.float32), requires_grad=False
    )

    # Float32 path
    _ = l2(ndl.ops.relu(l1(x)))  # warmup
    t0 = time.perf_counter()
    for _ in range(steps):
        _ = l2(ndl.ops.relu(l1(x)))
    float_time = time.perf_counter() - t0

    # Quantized weights path
    l1.eval()
    l2.eval()
    l1.enable_quantization(axis=1)
    l2.enable_quantization(axis=1)
    _ = l2(ndl.ops.relu(l1(x)))  # warmup
    t1 = time.perf_counter()
    for _ in range(steps):
        _ = l2(ndl.ops.relu(l1(x)))
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
    results = benchmark_mlp(
        batch=args.batch,
        in_features=args.in_features,
        hidden=args.hidden,
        out_features=args.out_features,
        steps=args.steps,
    )
    print(f"float32 avg: {results['float32_avg_ms']:.3f} ms")
    print(f"int8 avg:   {results['int8_avg_ms']:.3f} ms")
    print(f"speedup:    {results['speedup_x']:.2f}x")
    print("\nFull results:", results)


if __name__ == "__main__":
    main()
