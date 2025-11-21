#!/usr/bin/env python3
"""
Consolidated benchmark suite for Needle quantization.
Includes inference time and memory usage benchmarks for Linear and MLP models.
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

# Ensure backend is set
if "NEEDLE_BACKEND" not in os.environ:
    os.environ["NEEDLE_BACKEND"] = "nd"
sys.path.append("./python")

try:
    import needle as ndl
except ImportError:
    pass

def get_model(model_type, in_features, out_features, hidden=None):
    """
    Factory to create models.
    model_type: 'linear', 'mlp2', 'mlp4'
    """
    if model_type == 'linear':
        return ndl.nn.Linear(in_features, out_features, bias=True, dtype="float32")
    elif model_type == 'mlp2':
        h = hidden if hidden else in_features
        l1 = ndl.nn.Linear(in_features, h, bias=True, dtype="float32")
        l2 = ndl.nn.Linear(h, out_features, bias=True, dtype="float32")
        return [l1, l2]
    elif model_type == 'mlp4':
        # Assuming square for MLP4 as per original XL benchmark
        return [ndl.nn.Linear(in_features, out_features, bias=True, dtype="float32") for _ in range(4)]
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def forward_pass(model, x, model_type):
    """
    Executes forward pass based on model structure.
    """
    if model_type == 'linear':
        return model(x)
    elif model_type == 'mlp2':
        l1, l2 = model
        return l2(ndl.ops.relu(l1(x)))
    elif model_type == 'mlp4':
        out = x
        for layer in model:
            out = ndl.ops.relu(layer(out))
        return out

def benchmark_time(model_type='linear', dim=None, in_features=None, out_features=None, hidden=None, batch=256, steps=20, verbose=True):
    """
    Benchmark inference time for float32 vs int8.
    """
    if dim:
        in_features = dim
        out_features = dim
        hidden = dim
    
    # Defaults if not provided
    in_features = in_features or 1024
    out_features = out_features or 1024
    hidden = hidden or 1024

    if verbose:
        print(f"--- Benchmarking Time: {model_type} (in={in_features}, out={out_features}, batch={batch}) ---")
    
    model = get_model(model_type, in_features, out_features, hidden)
    x = ndl.Tensor(np.random.randn(batch, in_features).astype(np.float32), requires_grad=False)

    # Float32 path
    # Warmup
    forward_pass(model, x, model_type)
    
    t0 = time.perf_counter()
    for _ in range(steps):
        forward_pass(model, x, model_type)
    float_time = time.perf_counter() - t0

    # Quantized weights path
    layers = model if isinstance(model, list) else [model]
    for layer in layers:
        layer.eval()
        layer.enable_quantization(axis=1)
    
    # Warmup
    forward_pass(model, x, model_type)
    
    t1 = time.perf_counter()
    for _ in range(steps):
        forward_pass(model, x, model_type)
    int8_time = time.perf_counter() - t1

    res = {
        "model": model_type,
        "in_features": in_features,
        "out_features": out_features,
        "batch": batch,
        "float32_ms": float_time / steps * 1000,
        "int8_ms": int8_time / steps * 1000,
        "speedup": float_time / int8_time if int8_time > 0 else float('inf')
    }
    
    if verbose:
        print(f"Float32: {res['float32_ms']:.3f} ms")
        print(f"Int8:    {res['int8_ms']:.3f} ms")
        print(f"Speedup: {res['speedup']:.2f}x")
        print("")
        
    return res

def benchmark_memory(dim=None, in_features=None, out_features=None, verbose=True):
    """
    Benchmark memory usage for a single Linear layer.
    """
    if dim:
        in_features = dim
        out_features = dim
        
    in_features = in_features or 1024
    out_features = out_features or 1024

    if verbose:
        print(f"--- Benchmarking Memory: Linear Layer (in={in_features}, out={out_features}) ---")
        
    lin = ndl.nn.Linear(in_features, out_features, bias=True, dtype="float32")
    
    weight_elems = in_features * out_features
    bias_elems = out_features
    total_params = weight_elems + bias_elems
    
    float_bytes = total_params * 4  # float32 = 4 bytes
    
    # Quantize
    q = lin.enable_quantization(axis=1)
    int8_bytes = q.memory_bytes() + bias_elems * 4
    
    res = {
        "in_features": in_features,
        "out_features": out_features,
        "float32_mb": float_bytes / 1e6,
        "int8_mb": int8_bytes / 1e6,
        "reduction": 1 - (int8_bytes / float_bytes)
    }
    
    if verbose:
        print(f"Float32: {res['float32_mb']:.3f} MB")
        print(f"Int8:    {res['int8_mb']:.3f} MB")
        print(f"Reduction: {res['reduction']*100:.2f}%")
        print("")
        
    return res

def plot_memory_scaling(sizes=[256, 512, 1024, 2048, 4096], save_path="memory_plot.png"):
    """
    Generates memory scaling plot.
    """
    print(f"--- Generating Memory Plot (sizes={sizes}) ---")
    float_list = []
    int8_list = []
    
    for sz in sizes:
        res = benchmark_memory(dim=sz, verbose=False)
        float_list.append(res['float32_mb'])
        int8_list.append(res['int8_mb'])

    plt.figure(figsize=(8, 5))
    plt.plot(sizes, float_list, label="float32", marker="o")
    plt.plot(sizes, int8_list, label="int8 (weights)", marker="o")
    plt.xlabel("Layer size (in=out)")
    plt.ylabel("Memory (MB)")
    plt.title("Parameter memory vs layer size")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")
    
    return sizes, float_list, int8_list

def run_full_suite():
    """
    Runs all standard benchmarks.
    """
    print("=== Running Full Benchmark Suite ===\n")
    print(f"Backend: {os.environ.get('NEEDLE_BACKEND', 'nd')}\n")
    
    # 1. Standard Linear Benchmark (bench_quantization.py)
    benchmark_time('linear', in_features=1024, out_features=4096, batch=256, steps=30)
    
    # 2. Large MLP Benchmark (bench_quantization_large.py)
    benchmark_time('mlp2', in_features=4096, out_features=4096, hidden=4096, batch=512, steps=20)
    
    # 3. XL MLP Benchmark (bench_quantization_xl.py)
    benchmark_time('mlp4', dim=8192, batch=1024, steps=10)
    
    # 4. Memory Benchmark (bench_quantization_memory.py)
    benchmark_memory(in_features=1024, out_features=4096)
    
    # 5. Plot
    plot_memory_scaling()

if __name__ == "__main__":
    run_full_suite()
