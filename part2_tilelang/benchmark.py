"""
Benchmark TileLang (autotuned) vs PyTorch/cuDNN on A800.
Run `python autotune.py` first to populate the cache.
"""

import torch
import torch.nn.functional as F
import time
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))


def benchmark_fn(fn, warmup=10, iters=100, label=""):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / iters * 1000

    print(f"  {label:45s} {elapsed:8.3f} ms")
    return elapsed


def main():
    if not torch.cuda.is_available():
        print("CUDA not available.")
        return

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"PyTorch: {torch.__version__}")

    from model import TileLangConv2d, TileLangLinear

    print("\n=== Conv2d Benchmark (forward, autotuned) ===")
    configs = [
        (64, 1, 28, 28, 32, 3, 1, 1),
        (64, 32, 28, 28, 32, 3, 1, 1),
        (64, 32, 14, 14, 64, 3, 1, 1),
        (64, 64, 14, 14, 64, 3, 1, 1),
        (64, 64, 7, 7, 128, 3, 1, 1),
        (128, 3, 224, 224, 64, 7, 2, 3),
        (128, 64, 56, 56, 128, 3, 1, 1),
    ]

    for batch, cin, h, w, cout, k, s, p in configs:
        config_str = f"({batch},{cin},{h},{w})->({cout},{k}x{k},s={s},p={p})"
        x = torch.randn(batch, cin, h, w, device='cuda')
        w_pt = torch.randn(cout, cin, k, k, device='cuda')

        t_pt = benchmark_fn(lambda: F.conv2d(x, w_pt, stride=s, padding=p),
                            label=f"PyTorch  {config_str}")

        tl = TileLangConv2d(cin, cout, k, stride=s, padding=p).cuda()
        tl.weight.data = w_pt.clone()
        t_tl = benchmark_fn(lambda: tl(x), label=f"TileLang {config_str}")

        ratio = t_tl / t_pt
        print(f"  {'Ratio (TL/PT):':45s} {ratio:.2f}x\n")

    print("\n=== Linear (GEMM) Benchmark (forward, autotuned) ===")
    lin_configs = [
        (4096, 128, 64),
        (4096, 64, 12),
        (1024, 768, 768),
        (1024, 768, 3072),
        (4096, 4096, 4096),
    ]

    for m, k, n in lin_configs:
        config_str = f"({m},{k})->({n})"
        x = torch.randn(m, k, device='cuda')
        w = torch.randn(n, k, device='cuda')

        t_pt = benchmark_fn(lambda: F.linear(x, w), label=f"PyTorch  {config_str}")

        tl = TileLangLinear(k, n, bias=False).cuda()
        tl.weight.data = w.clone()
        t_tl = benchmark_fn(lambda: tl(x), label=f"TileLang {config_str}")

        ratio = t_tl / t_pt
        print(f"  {'Ratio (TL/PT):':45s} {ratio:.2f}x\n")


if __name__ == '__main__':
    main()
