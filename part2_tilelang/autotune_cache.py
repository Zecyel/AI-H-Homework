"""
Persistent autotuning cache for TileLang kernels.
First run: benchmarks all TileLang configs AND cuDNN, saves the winner.
Subsequent runs: loads from cache instantly (zero overhead).
If cuDNN is faster for a shape, stores {"use_cudnn": true} — TileLang is skipped at runtime.
"""

import json
import os
import logging
import torch

logger = logging.getLogger(__name__)

CACHE_FILE = os.path.join(os.path.dirname(__file__), "autotune_cache.json")

_cache = None


def _load_cache():
    global _cache
    if _cache is not None:
        return _cache
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE) as f:
            _cache = json.load(f)
        logger.info(f"Loaded {len(_cache)} cached autotune results from {CACHE_FILE}")
    else:
        _cache = {}
    return _cache


def _save_cache():
    with open(CACHE_FILE, 'w') as f:
        json.dump(_cache, f, indent=2)


def _bench_fn(fn, warmup=3, rep=10):
    """Benchmark a callable, return time in ms or inf on failure."""
    try:
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(rep):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / rep
    except Exception:
        return float('inf')


def get_cached_kernel(kernel_fn, shape_args, configs, cache_key, real_inputs,
                      cudnn_fn=None):
    """
    Get the best kernel for given shape, using cache if available.
    Compares TileLang's best config against cuDNN and picks the winner.

    Args:
        kernel_fn: The @tilelang.jit decorated kernel function
        shape_args: Tuple of shape parameters (e.g., (M, N, K) for GEMM)
        configs: List of config dicts to try
        cache_key: String key for the cache
        real_inputs: Tuple of actual input tensors (used for benchmarking)
        cudnn_fn: Optional callable for cuDNN benchmark (e.g., lambda: F.conv2d(...))

    Returns:
        Compiled kernel ready to call, or None if cuDNN wins / all configs failed.
        Also returns a boolean: (compiled_kernel_or_None, use_cudnn)
    """
    cache = _load_cache()

    # Cache hit
    if cache_key in cache:
        entry = cache[cache_key]
        if entry.get("use_cudnn", False):
            return None  # Caller should use cuDNN
        try:
            compiled = kernel_fn(*shape_args, **{k: v for k, v in entry.items()
                                                  if k != "tilelang_ms" and k != "cudnn_ms"})
            return compiled
        except Exception:
            logger.warning(f"Cached config for {cache_key} failed, re-tuning...")
            del cache[cache_key]

    # Cache miss — benchmark all TileLang configs
    logger.info(f"Autotuning {cache_key} ({len(configs)} configs)...")
    best_time = float('inf')
    best_config = None
    best_compiled = None

    for i, config in enumerate(configs):
        try:
            compiled = kernel_fn(*shape_args, **config)
        except Exception:
            continue

        t = _bench_fn(lambda c=compiled, ri=real_inputs: c(*ri))
        if t < best_time:
            best_time = t
            best_config = config
            best_compiled = compiled
            logger.info(f"  [{i+1}/{len(configs)}] {config} -> {t:.3f} ms (new best)")

    # Benchmark cuDNN for comparison
    cudnn_time = float('inf')
    if cudnn_fn is not None:
        cudnn_time = _bench_fn(cudnn_fn)
        logger.info(f"  cuDNN -> {cudnn_time:.3f} ms")

    # Decide winner
    if best_config is None and cudnn_fn is not None:
        # All TileLang configs failed, cuDNN wins by default
        cache[cache_key] = {"use_cudnn": True, "cudnn_ms": round(cudnn_time, 4)}
        _save_cache()
        logger.info(f"  Winner: cuDNN (all TileLang configs failed)")
        return None

    if best_config is None:
        logger.warning(f"All {len(configs)} configs failed for {cache_key}")
        return None

    if cudnn_time < best_time:
        # cuDNN wins
        cache[cache_key] = {"use_cudnn": True,
                            "cudnn_ms": round(cudnn_time, 4),
                            "tilelang_ms": round(best_time, 4)}
        _save_cache()
        logger.info(f"  Winner: cuDNN ({cudnn_time:.3f} ms) beats TileLang ({best_time:.3f} ms)")
        return None

    # TileLang wins
    entry = dict(best_config)
    entry["tilelang_ms"] = round(best_time, 4)
    if cudnn_fn is not None:
        entry["cudnn_ms"] = round(cudnn_time, 4)
    cache[cache_key] = entry
    _save_cache()
    logger.info(f"  Winner: TileLang ({best_time:.3f} ms) beats cuDNN ({cudnn_time:.3f} ms) — saved")
    return best_compiled


# ============================================================
# Config search spaces (curated for SM80 / A800)
# ============================================================

def get_gemm_configs():
    configs = []
    for bM, bN in [(32, 32), (64, 64), (128, 128), (64, 32), (32, 64),
                    (128, 64), (64, 128), (32, 128), (128, 32)]:
        for bK in [16, 32]:
            for stages in [2, 3]:
                configs.append({
                    "block_M": bM, "block_N": bN, "block_K": bK,
                    "num_stages": stages, "threads": 128,
                })
    return configs


def get_conv_configs():
    configs = []
    for bM, bN in [(32, 32), (64, 64), (32, 64), (64, 32), (16, 32),
                    (32, 16), (16, 64), (64, 16)]:
        for bK in [16, 32]:
            for stages in [2, 3]:
                configs.append({
                    "block_M": bM, "block_N": bN, "block_K": bK,
                    "num_stages": stages, "threads": 128,
                })
    return configs
