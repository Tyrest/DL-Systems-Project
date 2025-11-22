#!/usr/bin/env python3
"""
Consolidated benchmark script for Needle quantization.
Supports: linear, mlp, and memory modes.
"""

import argparse
import os
import sys
import time
import numpy as np

def ensure_backend() -> None:
    """Set backend env and sys.path for Needle imports."""
    if "NEEDLE_BACKEND" not in os.environ:
        os.environ["NEEDLE_BACKEND"] = "nd"
    sys.path.append("./python")

def run_linear(args, device):
    import needle as ndl
    print(f"Running Linear benchmark: batch={args.batch}, in={args.in_features}, out={args.out_features}, steps={args.steps}, device={device}")
    
    lin = ndl.nn.Linear(args.in_features, args.out_features, bias=True, dtype="float32", device=device)
    x = ndl.Tensor(
        np.random.randn(args.batch, args.in_features).astype(np.float32), requires_grad=False, device=device
    )

    # Float32 path
    _ = lin(x)  # warmup
    t0 = time.perf_counter()
    for _ in range(args.steps):
        _ = lin(x)
    float_time = time.perf_counter() - t0

    # Quantized weights path
    lin.eval()
    lin.enable_quantization(axis=1)
    _ = lin(x)  # warmup
    t1 = time.perf_counter()
    for _ in range(args.steps):
        _ = lin(x)
    int8_time = time.perf_counter() - t1

    results = {
        "float32_total_s": float_time,
        "int8_total_s": int8_time,
        "float32_avg_ms": float_time / args.steps * 1e3,
        "int8_avg_ms": int8_time / args.steps * 1e3,
        "speedup_x": float_time / int8_time if int8_time > 0 else float("inf"),
    }
    print(f"float32 avg: {results['float32_avg_ms']:.3f} ms")
    print(f"int8 avg:   {results['int8_avg_ms']:.3f} ms")
    print(f"speedup:    {results['speedup_x']:.2f}x")
    print("\nFull results:", results)

def run_mlp(args, device):
    import needle as ndl
    print(f"Running MLP benchmark: batch={args.batch}, in={args.in_features}, hidden={args.hidden}, out={args.out_features}, steps={args.steps}, device={device}")

    l1 = ndl.nn.Linear(args.in_features, args.hidden, bias=True, dtype="float32", device=device)
    l2 = ndl.nn.Linear(args.hidden, args.out_features, bias=True, dtype="float32", device=device)
    x = ndl.Tensor(
        np.random.randn(args.batch, args.in_features).astype(np.float32), requires_grad=False, device=device
    )

    # Float32 path
    _ = l2(ndl.ops.relu(l1(x)))  # warmup
    t0 = time.perf_counter()
    for _ in range(args.steps):
        _ = l2(ndl.ops.relu(l1(x)))
    float_time = time.perf_counter() - t0

    # Quantized weights path
    l1.eval()
    l2.eval()
    l1.enable_quantization(axis=1)
    l2.enable_quantization(axis=1)
    _ = l2(ndl.ops.relu(l1(x)))  # warmup
    t1 = time.perf_counter()
    for _ in range(args.steps):
        _ = l2(ndl.ops.relu(l1(x)))
    int8_time = time.perf_counter() - t1

    results = {
        "float32_total_s": float_time,
        "int8_total_s": int8_time,
        "float32_avg_ms": float_time / args.steps * 1e3,
        "int8_avg_ms": int8_time / args.steps * 1e3,
        "speedup_x": float_time / int8_time if int8_time > 0 else float("inf"),
    }
    print(f"float32 avg: {results['float32_avg_ms']:.3f} ms")
    print(f"int8 avg:   {results['int8_avg_ms']:.3f} ms")
    print(f"speedup:    {results['speedup_x']:.2f}x")
    print("\nFull results:", results)

def run_memory(args, device):
    import needle as ndl
    import matplotlib.pyplot as plt
    import psutil
    import gc
    import ctypes
    import subprocess
    import threading
    import time

    process = psutil.Process(os.getpid())

    def trim_memory():
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass

    def get_cuda_memory_mb():
        try:
            result = subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,nounits,noheader"],
                encoding="utf-8"
            )
            for line in result.strip().split("\n"):
                if not line: continue
                parts = line.split(",")
                if len(parts) < 2: continue
                pid, mem = parts[0], parts[1]
                if int(pid) == os.getpid():
                    # nvidia-smi returns MiB, convert to MB (1e6 bytes) to match psutil
                    return float(mem) * 1024 * 1024 / 1e6
        except Exception:
            pass
        return 0.0

    def get_current_memory_mb():
        if args.device == "cuda":
            return get_cuda_memory_mb()
        return process.memory_info().rss / 1e6

    class MemoryMonitor(threading.Thread):
        def __init__(self):
            super().__init__()
            self.keep_running = True
            self.peak_memory = 0.0
            self.lock = threading.Lock()

        def run(self):
            while self.keep_running:
                try:
                    mem = get_current_memory_mb()
                    with self.lock:
                        self.peak_memory = max(self.peak_memory, mem)
                except:
                    pass
                time.sleep(0.0001)

        def stop(self):
            self.keep_running = False
            self.join()
            return self.peak_memory

    def get_memory_mb():
        gc.collect()
        trim_memory()
        return get_current_memory_mb()

    print(f"Running Memory benchmark (Actual {'GPU' if args.device == 'cuda' else 'Process'} Memory): in={args.in_features}, out={args.out_features}, device={device}")
    
    # Warmup to ensure CUDA context is initialized before measurements
    if args.device == "cuda":
        print("Warming up CUDA context...")
        dummy = ndl.Tensor([1.0], device=device)
        del dummy
        gc.collect()
        # Allow time for driver to settle
        time.sleep(1)

    # Define a measurement function
    def measure_layer(in_feat, out_feat, quantize=False):
        gc.collect()
        trim_memory()
        base = get_current_memory_mb()
        
        model = ndl.nn.Linear(in_feat, out_feat, bias=True, dtype="float32", device=device)
        
        # Inference mode: disable gradients to avoid graph overhead
        for p in model.parameters():
            p.requires_grad = False
        del p  # Ensure loop variable doesn't keep reference to parameters


        if quantize:
            model.eval()
            model.enable_quantization(axis=1)
            # Simulate deployment: drop original float32 weights and cached dequantized weights
            # This allows us to measure the memory footprint of just the int8 weights + bias
            model.weight = None
            model.bias = None  # Drop bias reference too if we want to be super clean (though bias is small)
            model._weight_deq = None
            if model._weight_q is not None:
                model._weight_q.cached_dequantized = None
            
            # Force garbage collection of the float weights we just dropped
            gc.collect()
            trim_memory()
            
            # Force use of int8 kernel if available
            os.environ["NEEDLE_USE_INT8_MATMUL"] = "1"
        else:
            os.environ["NEEDLE_USE_INT8_MATMUL"] = "0"
        
        # Run data through (batch size 128)
        x = ndl.Tensor(np.random.randn(128, in_feat).astype(np.float32), device=device, requires_grad=False)
        
        if args.device == "cuda":
            monitor = MemoryMonitor()
            monitor.peak_memory = get_current_memory_mb()
            monitor.start()
            
            # Run multiple times to ensure we catch the peak usage
            # especially for fast GPU operations
            for _ in range(20):
                out = model(x)
                # Force realization
                _ = out.numpy()
                
            peak = monitor.stop()
        else:
            out = model(x)
            # Force realization
            _ = out.numpy()
            peak = get_current_memory_mb()

        del model, x, out
        return peak - base

    float_usage = measure_layer(args.in_features, args.out_features, quantize=False)
    int8_usage = measure_layer(args.in_features, args.out_features, quantize=True)

    print(f"Layer: Linear({args.in_features}, {args.out_features})")
    print(f"Float32 memory usage: {float_usage:.2f} MB")
    print(f"Int8 memory usage:    {int8_usage:.2f} MB")
    if float_usage > 0:
        print(f"Memory reduction:     {1 - int8_usage/float_usage:.2%}")

    # Plot
    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    if not sizes:
        return

    print(f"Plotting sizes: {sizes}")
    float_list = []
    int8_list = []
    
    for sz in sizes:
        f_mem = measure_layer(sz, sz, quantize=False)
        i_mem = measure_layer(sz, sz, quantize=True)
        float_list.append(f_mem)
        int8_list.append(i_mem)

    print(f"Float32 memory (MB): {[round(x, 2) for x in float_list]}")
    print(f"Int8 memory (MB):    {[round(x, 2) for x in int8_list]}")
    
    reduction_pct = []
    for f, i in zip(float_list, int8_list):
        if f > 1e-6:
            reduction_pct.append(100 * (1 - i/f))
        else:
            reduction_pct.append(0.0)

    print(f"Reduction:           {[f'{r:.1f}%' for r in reduction_pct]}")

    plt.figure(figsize=(6, 4))
    plt.plot(sizes, reduction_pct, label="Memory Saving %", marker="o", color='green')
    plt.xlabel("Layer size (in=out)")
    plt.ylabel("Memory Saving (%)")
    plt.title("Memory Saving Percentage vs Layer Size")
    plt.ylim(0, 100)  # Percentage is 0-100
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(args.save)
    print(f"Saved plot to {args.save}")

def main():
    parser = argparse.ArgumentParser(description="Needle Quantization Benchmarks")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run on (cpu or cuda)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Linear
    p_lin = subparsers.add_parser("linear", help="Benchmark single Linear layer")
    p_lin.add_argument("--batch", type=int, default=256)
    p_lin.add_argument("--in-features", type=int, default=1024)
    p_lin.add_argument("--out-features", type=int, default=4096)
    p_lin.add_argument("--steps", type=int, default=30)

    # MLP
    p_mlp = subparsers.add_parser("mlp", help="Benchmark two-layer MLP")
    p_mlp.add_argument("--batch", type=int, default=512)
    p_mlp.add_argument("--in-features", type=int, default=4096)
    p_mlp.add_argument("--hidden", type=int, default=4096)
    p_mlp.add_argument("--out-features", type=int, default=4096)
    p_mlp.add_argument("--steps", type=int, default=20)

    # Memory
    p_mem = subparsers.add_parser("memory", help="Memory usage benchmark")
    p_mem.add_argument("--in-features", type=int, default=1024)
    p_mem.add_argument("--out-features", type=int, default=4096)
    p_mem.add_argument("--sizes", type=str, default="256,512,1024,2048,4096")
    p_mem.add_argument("--save", type=str, default="memory_plot.png")

    args = parser.parse_args()
    ensure_backend()
    import needle as ndl
    print(f"Backend: {os.environ.get('NEEDLE_BACKEND', 'nd')}")
    
    if args.device == "cpu":
        device = ndl.cpu()
    elif args.device == "cuda":
        device = ndl.cuda()
    else:
        raise ValueError(f"Unknown device: {args.device}")

    if args.command == "linear":
        run_linear(args, device)
    elif args.command == "mlp":
        run_mlp(args, device)
    elif args.command == "memory":
        run_memory(args, device)

if __name__ == "__main__":
    main()
