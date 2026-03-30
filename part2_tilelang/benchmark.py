"""
Benchmark TileLang kernels vs PyTorch/cuDNN on A800.
Measures throughput for Conv2d and Linear at various sizes.
"""

import torch
import torch.nn.functional as F
import time
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))


def benchmark_fn(fn, warmup=10, iters=100, label=""):
    """Benchmark a function, return average time in ms."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / iters * 1000  # ms

    print(f"  {label:40s} {elapsed:8.3f} ms")
    return elapsed


def benchmark_conv2d():
    print("\n=== Conv2d Benchmark (forward) ===")
    print(f"{'Config':40s} {'Time (ms)':>10s}")
    print("-" * 55)

    from model import TileLangConv2d

    configs = [
        # (batch, C_in, H, W, C_out, kernel, stride, padding)
        (64, 1, 28, 28, 32, 3, 1, 1),      # Block 1a (our CNN)
        (64, 32, 28, 28, 32, 3, 1, 1),      # Block 1b
        (64, 32, 14, 14, 64, 3, 1, 1),      # Block 2a
        (64, 64, 14, 14, 64, 3, 1, 1),      # Block 2b
        (64, 64, 7, 7, 128, 3, 1, 1),       # Block 3
        (128, 3, 224, 224, 64, 7, 2, 3),     # ResNet-like first conv
        (128, 64, 56, 56, 128, 3, 1, 1),     # ResNet-like mid conv
    ]

    for batch, cin, h, w, cout, k, s, p in configs:
        config_str = f"({batch},{cin},{h},{w})->({cout},{k}x{k},s={s},p={p})"

        x = torch.randn(batch, cin, h, w, device='cuda')
        w_torch = torch.randn(cout, cin, k, k, device='cuda')

        # PyTorch/cuDNN
        def pytorch_conv():
            return F.conv2d(x, w_torch, stride=s, padding=p)

        t_pytorch = benchmark_fn(pytorch_conv, label=f"PyTorch  {config_str}")

        # TileLang
        tl_conv = TileLangConv2d(cin, cout, k, stride=s, padding=p).cuda()
        tl_conv.weight.data = w_torch.clone()

        def tilelang_conv():
            return tl_conv(x)

        t_tl = benchmark_fn(tilelang_conv, label=f"TileLang {config_str}")

        ratio = t_tl / t_pytorch
        print(f"  {'Ratio (TL/PT):':40s} {ratio:.2f}x")
        print()


def benchmark_linear():
    print("\n=== Linear (GEMM) Benchmark (forward) ===")
    print(f"{'Config':40s} {'Time (ms)':>10s}")
    print("-" * 55)

    from model import TileLangLinear

    configs = [
        (4096, 128, 64),     # Our CNN FC1
        (4096, 64, 12),      # Our CNN FC2
        (1024, 768, 768),    # Transformer-like
        (1024, 768, 3072),   # Transformer FFN
        (4096, 4096, 4096),  # Large GEMM
    ]

    for m, k, n in configs:
        config_str = f"({m},{k})->({n})"

        x = torch.randn(m, k, device='cuda')
        w = torch.randn(n, k, device='cuda')

        def pytorch_linear():
            return F.linear(x, w)

        t_pytorch = benchmark_fn(pytorch_linear, label=f"PyTorch  {config_str}")

        tl_linear = TileLangLinear(k, n, bias=False).cuda()
        tl_linear.weight.data = w.clone()

        def tilelang_linear():
            return tl_linear(x)

        t_tl = benchmark_fn(tilelang_linear, label=f"TileLang {config_str}")

        ratio = t_tl / t_pytorch
        print(f"  {'Ratio (TL/PT):':40s} {ratio:.2f}x")
        print()


def main():
    if not torch.cuda.is_available():
        print("CUDA not available, skipping benchmarks.")
        return

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"PyTorch: {torch.__version__}")

    try:
        import tilelang
        print(f"TileLang: {tilelang.__version__}")
    except (ImportError, AttributeError):
        print("TileLang: (version unknown)")

    benchmark_conv2d()
    benchmark_linear()


if __name__ == '__main__':
    main()
