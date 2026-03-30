"""
TensorRT-style autotuner for TileLang kernels.

On first run, exhaustively benchmarks all tile configurations for every kernel
shape used in the CNN, then caches the optimal configs to disk.
Subsequent runs load from cache instantly.

Usage:
    python autotune.py              # run full autotuning
    python autotune.py --quick      # quick mode (fewer configs)
"""

import torch
import tilelang
import json
import os
import sys
import time
import itertools

sys.path.insert(0, os.path.dirname(__file__))
from kernels import (
    make_gemm_kernel,
    make_conv2d_forward_kernel,
    make_conv2d_backward_data_kernel,
    make_conv2d_backward_weight_kernel,
)

CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "autotune_cache.json")

# ============================================================
# Search space
# ============================================================

# Full search space for SM80
GEMM_SEARCH_SPACE = {
    "block_M":    [16, 32, 64, 128],
    "block_N":    [16, 32, 64, 128],
    "block_K":    [8, 16, 32],
    "num_stages": [2, 3, 4],
    "threads":    [64, 128, 256],
}

CONV_SEARCH_SPACE = {
    "block_M":    [16, 32, 64, 128],
    "block_N":    [16, 32, 64, 128],
    "block_K":    [8, 16, 32],
    "num_stages": [2, 3],
    "threads":    [64, 128, 256],
}

# Quick search space (subset)
GEMM_SEARCH_SPACE_QUICK = {
    "block_M":    [16, 32, 64, 128],
    "block_N":    [16, 32, 64, 128],
    "block_K":    [16, 32],
    "num_stages": [2, 3],
    "threads":    [64, 128],
}

CONV_SEARCH_SPACE_QUICK = {
    "block_M":    [16, 32, 64],
    "block_N":    [16, 32, 64],
    "block_K":    [16, 32],
    "num_stages": [2, 3],
    "threads":    [64, 128],
}


def generate_configs(search_space):
    """Generate all combinations from the search space."""
    keys = list(search_space.keys())
    values = list(search_space.values())
    configs = []
    for combo in itertools.product(*values):
        configs.append(dict(zip(keys, combo)))
    return configs


def is_valid_config(config, M, N, K):
    """Filter out obviously invalid configs (block > dimension, etc.)."""
    bM, bN, bK = config["block_M"], config["block_N"], config["block_K"]
    threads = config["threads"]

    # Block shouldn't be absurdly larger than the dimension
    if bM > max(M, 128) or bN > max(N, 128) or bK > max(K, 64):
        return False

    # Shared memory constraint: rough estimate for SM80 (164KB max)
    # Two shared tiles + output shared: 2 * bM*bK + bK*bN (in fp16 = 2 bytes)
    shared_bytes = 2 * (bM * bK + bK * bN + bM * bN)
    if shared_bytes > 163840:  # 160KB safety margin
        return False

    return True


def benchmark_kernel(make_fn, args, config, warmup=5, iters=20):
    """
    Compile and benchmark a single kernel configuration.
    Returns median time in microseconds, or float('inf') on failure.
    """
    try:
        prim_func = make_fn(*args, **config)
        kernel = tilelang.compile(prim_func, out_idx=[2], target="auto")
        profiler = kernel.get_profiler(tensor_supply_type="randn")
        latency = profiler.do_bench(warmup=warmup, rep=iters)
        return latency
    except Exception as e:
        return float('inf')


def autotune_kernel(name, make_fn, shape_args, search_space, extra_kwargs=None):
    """
    Exhaustively benchmark all valid configs for a kernel shape.
    Returns the best config and its latency.
    """
    extra_kwargs = extra_kwargs or {}
    configs = generate_configs(search_space)

    # Determine M, N, K for validity check
    # For GEMM: args = (M, N, K)
    # For conv: we pass M_equiv, N_equiv, K_equiv separately
    M_equiv = extra_kwargs.pop("M_equiv", shape_args[0] if len(shape_args) >= 1 else 128)
    N_equiv = extra_kwargs.pop("N_equiv", shape_args[1] if len(shape_args) >= 2 else 128)
    K_equiv = extra_kwargs.pop("K_equiv", shape_args[2] if len(shape_args) >= 3 else 32)

    valid_configs = [c for c in configs if is_valid_config(c, M_equiv, N_equiv, K_equiv)]
    total = len(valid_configs)
    print(f"\n  [{name}] Testing {total} configurations (filtered from {len(configs)})...")

    best_config = None
    best_latency = float('inf')

    for idx, config in enumerate(valid_configs):
        full_args = list(shape_args) + [
            config["block_M"], config["block_N"], config["block_K"],
            config["num_stages"], config["threads"],
        ]

        latency = benchmark_kernel(make_fn, full_args, extra_kwargs.copy())

        status = ""
        if latency < best_latency:
            best_latency = latency
            best_config = config.copy()
            status = " <-- NEW BEST"

        if (idx + 1) % 20 == 0 or idx == 0 or status:
            print(f"    [{idx+1:3d}/{total}] bM={config['block_M']:3d} bN={config['block_N']:3d} "
                  f"bK={config['block_K']:2d} stages={config['num_stages']} "
                  f"threads={config['threads']:3d}  "
                  f"latency={latency:8.1f}us{status}")

    print(f"  [{name}] Best: {best_config}  latency={best_latency:.1f}us")
    return best_config, best_latency


# ============================================================
# Enumerate all kernel shapes used in our CNN
# ============================================================

def get_cnn_kernel_shapes(batch_size=64):
    """
    Return all (name, make_fn, shape_args, search_space, extra_kwargs)
    for every kernel invocation in our CNN.
    """
    shapes = []

    # -- Conv2d forward --
    # Block 1a: (N,1,28,28) -> (32, 3x3, pad=1)
    # Block 1b: (N,32,28,28) -> (32, 3x3, pad=1)
    # Block 2a: (N,32,14,14) -> (64, 3x3, pad=1)
    # Block 2b: (N,64,14,14) -> (64, 3x3, pad=1)
    # Block 3:  (N,64,7,7) -> (128, 3x3, pad=1)
    conv_fwd_specs = [
        ("conv2d_fwd_1a", batch_size, 1, 28, 28, 32, 3, 3),
        ("conv2d_fwd_1b", batch_size, 32, 28, 28, 32, 3, 3),
        ("conv2d_fwd_2a", batch_size, 32, 14, 14, 64, 3, 3),
        ("conv2d_fwd_2b", batch_size, 64, 14, 14, 64, 3, 3),
        ("conv2d_fwd_3",  batch_size, 64, 7, 7, 128, 3, 3),
    ]
    for name, N, C_in, H, W, C_out, KH, KW in conv_fwd_specs:
        OH = (H + 2 * 1 - KH) // 1 + 1
        OW = (W + 2 * 1 - KW) // 1 + 1
        shapes.append((
            name, make_conv2d_forward_kernel,
            (N, C_in, H, W, C_out, KH, KW, 1, 1),
            {"M_equiv": N * OH * OW, "N_equiv": C_out, "K_equiv": KH * KW * C_in},
        ))

    # -- Conv2d backward data --
    for name_prefix, N, C_in, H, W, C_out, KH, KW in conv_fwd_specs:
        name = name_prefix.replace("fwd", "bwd_data")
        OH = (H + 2 * 1 - KH) // 1 + 1
        OW = (W + 2 * 1 - KW) // 1 + 1
        shapes.append((
            name, make_conv2d_backward_data_kernel,
            (N, C_in, H, W, C_out, KH, KW, 1, 1),
            {"M_equiv": N * H * W, "N_equiv": C_in, "K_equiv": KH * KW * C_out},
        ))

    # -- Conv2d backward weight --
    for name_prefix, N, C_in, H, W, C_out, KH, KW in conv_fwd_specs:
        name = name_prefix.replace("fwd", "bwd_weight")
        OH = (H + 2 * 1 - KH) // 1 + 1
        OW = (W + 2 * 1 - KW) // 1 + 1
        shapes.append((
            name, make_conv2d_backward_weight_kernel,
            (N, C_in, H, W, C_out, KH, KW, 1, 1),
            {"M_equiv": KH * KW * C_in, "N_equiv": C_out, "K_equiv": N * OH * OW},
        ))

    # -- Linear (GEMM) forward --
    # FC1: (N_flat, 128) @ (128, 64) -> (N_flat, 64)
    # FC2: (N_flat, 64) @ (64, 12) -> (N_flat, 12)
    gemm_specs = [
        ("gemm_fc1_fwd", batch_size, 64, 128),
        ("gemm_fc2_fwd", batch_size, 12, 64),
    ]
    for name, M, N, K in gemm_specs:
        shapes.append((
            name, make_gemm_kernel,
            (M, N, K),
            {"M_equiv": M, "N_equiv": N, "K_equiv": K},
        ))

    # -- Linear backward (grad_input) --
    # grad_input = grad_output @ weight  ->  (M, N_out) @ (N_out, K) = (M, K)
    gemm_bwd_input_specs = [
        ("gemm_fc1_bwd_input", batch_size, 128, 64),   # (M, K, N_out)
        ("gemm_fc2_bwd_input", batch_size, 64, 12),
    ]
    for name, M, N, K in gemm_bwd_input_specs:
        shapes.append((
            name, make_gemm_kernel,
            (M, N, K),
            {"M_equiv": M, "N_equiv": N, "K_equiv": K},
        ))

    # -- Linear backward (grad_weight) --
    # grad_weight = grad_output^T @ input -> (N_out, K, M)
    gemm_bwd_weight_specs = [
        ("gemm_fc1_bwd_weight", 64, 128, batch_size),
        ("gemm_fc2_bwd_weight", 12, 64, batch_size),
    ]
    for name, M, N, K in gemm_bwd_weight_specs:
        shapes.append((
            name, make_gemm_kernel,
            (M, N, K),
            {"M_equiv": M, "N_equiv": N, "K_equiv": K},
        ))

    return shapes


# ============================================================
# Cache management
# ============================================================

def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_cache(cache):
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=2)
    print(f"\nCache saved to {CACHE_FILE}")


def get_cache_key(name, shape_args):
    """Create a unique key from kernel name and shape."""
    return f"{name}_{','.join(str(x) for x in shape_args)}"


# ============================================================
# Main autotuning entry point
# ============================================================

def run_autotune(batch_size=64, quick=False):
    """Run full autotuning for all CNN kernel shapes."""
    print("=" * 70)
    print("TileLang Autotuner — TensorRT-style exhaustive profiling")
    print("=" * 70)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Batch size: {batch_size}")
    print(f"Mode: {'quick' if quick else 'full'}")

    conv_space = CONV_SEARCH_SPACE_QUICK if quick else CONV_SEARCH_SPACE
    gemm_space = GEMM_SEARCH_SPACE_QUICK if quick else GEMM_SEARCH_SPACE

    print(f"Conv search space: {len(generate_configs(conv_space))} configs")
    print(f"GEMM search space: {len(generate_configs(gemm_space))} configs")

    shapes = get_cnn_kernel_shapes(batch_size)
    cache = load_cache()

    total_shapes = len(shapes)
    total_start = time.time()

    for idx, (name, make_fn, shape_args, equiv_kwargs) in enumerate(shapes):
        cache_key = get_cache_key(name, shape_args)

        # Skip if already cached
        if cache_key in cache:
            print(f"\n[{idx+1}/{total_shapes}] {name}: CACHED "
                  f"(latency={cache[cache_key]['latency']:.1f}us)")
            continue

        print(f"\n[{idx+1}/{total_shapes}] Autotuning: {name}")
        print(f"  Shape args: {shape_args}")

        # Select search space based on kernel type
        is_conv = "conv2d" in name
        search_space = conv_space if is_conv else gemm_space

        best_config, best_latency = autotune_kernel(
            name, make_fn, shape_args, search_space, equiv_kwargs.copy())

        if best_config is not None:
            cache[cache_key] = {
                "config": best_config,
                "latency": best_latency,
                "shape_args": list(shape_args),
                "name": name,
            }
            # Save after each kernel (resume-friendly)
            save_cache(cache)

    total_time = time.time() - total_start
    print(f"\n{'=' * 70}")
    print(f"Autotuning complete! Total time: {total_time:.1f}s")
    print(f"Results cached in: {CACHE_FILE}")

    # Summary table
    print(f"\n{'Kernel':<35s} {'Config':>40s} {'Latency':>10s}")
    print("-" * 85)
    for name, _, shape_args, _ in shapes:
        key = get_cache_key(name, shape_args)
        if key in cache:
            cfg = cache[key]["config"]
            lat = cache[key]["latency"]
            cfg_str = f"bM={cfg['block_M']:3d} bN={cfg['block_N']:3d} bK={cfg['block_K']:2d} st={cfg['num_stages']} th={cfg['threads']:3d}"
            print(f"  {name:<33s} {cfg_str:>40s} {lat:>8.1f}us")


def lookup_config(name, shape_args):
    """Look up the best config from cache. Returns None if not found."""
    cache = load_cache()
    key = get_cache_key(name, shape_args)
    if key in cache:
        return cache[key]["config"]
    return None


def lookup_or_default(name, shape_args):
    """Look up cached config, fall back to conservative defaults."""
    config = lookup_config(name, shape_args)
    if config is not None:
        return config
    # Fallback defaults (conservative, works for any shape)
    return {
        "block_M": 32,
        "block_N": 32,
        "block_K": 16,
        "num_stages": 2,
        "threads": 128,
    }


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    batch_size = 64
    for arg in sys.argv[1:]:
        if arg.startswith("--batch="):
            batch_size = int(arg.split("=")[1])
    run_autotune(batch_size=batch_size, quick=quick)
