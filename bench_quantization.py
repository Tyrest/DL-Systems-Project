#!/usr/bin/env python3
"""
Benchmark float32 vs int8-quantized Linear forward passes.

Usage:
  python bench_quantization.py --batch 256 --in-features 1024 --out-features 4096 --steps 30
"""

import argparse
import os
import sys
import time
import numpy as np


def parse_args() -> argparse.Namespace:
    """CLI args for simple linear benchmark."""
    parser = argparse.ArgumentParser(description="Benchmark Needle quantization performance.")
    parser.add_argument("--batch", type=int, default=256, help="Batch size.")
    parser.add_argument("--in-features", type=int, default=1024, help="Input feature dimension.")
    parser.add_argument("--out-features", type=int, default=4096, help="Output feature dimension.")
    parser.add_argument("--steps", type=int, default=30, help="Number of timed iterations per mode.")
    return parser.parse_args()


def ensure_backend() -> None:
    """Set backend env and path for imports."""
    if "NEEDLE_BACKEND" not in os.environ:
        os.environ["NEEDLE_BACKEND"] = "nd"
    sys.path.append("./python")


def benchmark_linear(batch: int, in_features: int, out_features: int, steps: int) -> dict:
    """Run float32 vs quantized Linear forward passes and time them."""
    import needle as ndl  # imported after backend is set

    lin = ndl.nn.Linear(in_features, out_features, bias=True, dtype="float32")
    x = ndl.Tensor(
        np.random.randn(batch, in_features).astype(np.float32), requires_grad=False
    )

    # Float32 path
    _ = lin(x)  # warmup
    t0 = time.perf_counter()
    for _ in range(steps):
        _ = lin(x)
    float_time = time.perf_counter() - t0

    # Quantized weights path (activations remain float32)
    lin.eval()
    lin.enable_quantization(axis=1)
    _ = lin(x)  # warmup
    t1 = time.perf_counter()
    for _ in range(steps):
        _ = lin(x)
    int8_time = time.perf_counter() - t1

    return {
        "float32_total_s": float_time,
        "int8_total_s": int8_time,
        "float32_avg_ms": float_time / steps * 1e3,
        "int8_avg_ms": int8_time / steps * 1e3,
        "speedup_x": float_time / int8_time if int8_time > 0 else float("inf"),
    }


def main() -> None:
    """Entry point: parse args, run benchmark, print summary."""
    args = parse_args()
    ensure_backend()
    print(f"Backend: {os.environ.get('NEEDLE_BACKEND', 'nd')}")
    results = benchmark_linear(
        batch=args.batch,
        in_features=args.in_features,
        out_features=args.out_features,
        steps=args.steps,
    )
    print(f"float32 avg: {results['float32_avg_ms']:.3f} ms")
    print(f"int8 avg:   {results['int8_avg_ms']:.3f} ms")
    print(f"speedup:    {results['speedup_x']:.2f}x")
    print("\nFull results:", results)


if __name__ == "__main__":
    main()
