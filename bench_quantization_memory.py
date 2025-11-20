#!/usr/bin/env python3
"""
Report parameter counts and memory for float32 vs int8-quantized weights.

Defaults: single Linear layer (in=1024, out=4096).
"""

import argparse
import os
import sys
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Memory comparison for quantization.")
    p.add_argument("--in-features", type=int, default=1024, help="Input dim.")
    p.add_argument("--out-features", type=int, default=4096, help="Output dim.")
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


if __name__ == "__main__":
    main()
