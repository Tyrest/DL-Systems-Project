#!/usr/bin/env python3
"""
Report parameter counts and memory for float32 vs int8-quantized weights.

Defaults: single Linear layer (in=1024, out=4096).
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Memory comparison for quantization.")
    p.add_argument("--in-features", type=int, default=1024, help="Input dim for single-layer report.")
    p.add_argument("--out-features", type=int, default=4096, help="Output dim for single-layer report.")
    p.add_argument(
        "--sizes",
        type=str,
        default="256,512,1024,2048,4096",
        help="Comma-separated list of square layer sizes to plot (in=out=size).",
    )
    p.add_argument("--save", type=str, default="memory_plot.png", help="Path to save plot.")
    return p.parse_args()


def ensure_backend() -> None:
    if "NEEDLE_BACKEND" not in os.environ:
        os.environ["NEEDLE_BACKEND"] = "nd"
    sys.path.append("./python")


def bytes_to_mb(num_bytes: int) -> float:
    return num_bytes / (1024 * 1024)


def main() -> None:
    args = parse_args()
    ensure_backend()
    import needle as ndl

    lin = ndl.nn.Linear(args.in_features, args.out_features, bias=True, dtype="float32")
    # Parameter counts
    weight_elems = args.in_features * args.out_features
    bias_elems = args.out_features
    total_params = weight_elems + bias_elems

    # Float32 memory
    float_bytes = total_params * 4  # float32 = 4 bytes

    # Quantize weights (bias stays float32)
    q = lin.enable_quantization(axis=1)
    int8_bytes = q.memory_bytes() + bias_elems * 4  # int8 weights + scales/zp + float32 bias

    print(f"Backend: {os.environ.get('NEEDLE_BACKEND', 'nd')}")
    print(f"Layer: Linear({args.in_features}, {args.out_features})")
    print(f"Parameters: weights={weight_elems:,}, bias={bias_elems:,}, total={total_params:,}")
    print(f"Float32 memory: {float_bytes/1e6:.3f} MB")
    print(
        f"Int8 memory:   {int8_bytes/1e6:.3f} MB "
        f"(weights int8 + scale/zp + float32 bias)"
    )
    reduction = 1 - (int8_bytes / float_bytes)
    print(f"Memory reduction: {reduction*100:.2f}%")

    # Plot across sizes
    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    float_list = []
    int8_list = []
    for sz in sizes:
        tmp = ndl.nn.Linear(sz, sz, bias=True, dtype="float32")
        q_sz = tmp.enable_quantization(axis=1)
        params = sz * sz + sz
        float_b = params * 4
        int8_b = q_sz.memory_bytes() + sz * 4
        float_list.append(float_b / 1e6)
        int8_list.append(int8_b / 1e6)

    plt.figure(figsize=(6, 4))
    plt.plot(sizes, float_list, label="float32", marker="o")
    plt.plot(sizes, int8_list, label="int8 (weights)", marker="o")
    plt.xlabel("Layer size (in=out)")
    plt.ylabel("Memory (MB)")
    plt.title("Parameter memory vs layer size")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(args.save)
    print(f"Saved plot to {args.save}")


if __name__ == "__main__":
    main()
