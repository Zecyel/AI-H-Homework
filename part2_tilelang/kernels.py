"""
TileLang GPU kernels for CNN operations, tuned for NVIDIA A800 (SM80).

Implements:
- Conv2d forward (implicit im2col GEMM)
- Conv2d backward (weight gradient + input gradient)
- Linear forward/backward (GEMM)
"""

import tilelang
import tilelang.language as T

# ============================================================
# SM80 (A800) tuning parameters
# ============================================================
# A800 has 80 SMs, 48KB L1/shared per SM (configurable up to 164KB),
# 40MB L2 cache, 2TB/s HBM bandwidth, Tensor Cores (FP16 MMA).

SM80_GEMM_BLOCK_M = 128
SM80_GEMM_BLOCK_N = 128
SM80_GEMM_BLOCK_K = 32
SM80_GEMM_STAGES = 3
SM80_GEMM_THREADS = 128

SM80_CONV_BLOCK_M = 64
SM80_CONV_BLOCK_N = 64
SM80_CONV_BLOCK_K = 32
SM80_CONV_STAGES = 2
SM80_CONV_THREADS = 128


# ============================================================
# GEMM kernel (for linear layers)
# ============================================================

def make_gemm_kernel(M, N, K, dtype="float16", accum_dtype="float32",
                     block_M=None, block_N=None, block_K=None,
                     num_stages=None, threads=None, transpose_B=False):
    """
    Create a GEMM kernel: C = A @ B (or C = A @ B^T if transpose_B=True).
    A: (M, K), B: (K, N) or (N, K) if transposed, C: (M, N)
    """
    block_M = block_M or SM80_GEMM_BLOCK_M
    block_N = block_N or SM80_GEMM_BLOCK_N
    block_K = block_K or SM80_GEMM_BLOCK_K
    num_stages = num_stages or SM80_GEMM_STAGES
    threads = threads or SM80_GEMM_THREADS

    if not transpose_B:
        @T.prim_func
        def kernel(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((K, N), dtype),
            C: T.Tensor((M, N), dtype),
        ):
            with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
                A_shared = T.alloc_shared((block_M, block_K), dtype)
                B_shared = T.alloc_shared((block_K, block_N), dtype)
                C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
                T.clear(C_local)

                for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                    T.copy(A[by * block_M, ko * block_K], A_shared)
                    T.copy(B[ko * block_K, bx * block_N], B_shared)
                    T.gemm(A_shared, B_shared, C_local)

                T.copy(C_local, C[by * block_M, bx * block_N])
    else:
        # C = A @ B^T, where B is (N, K)
        @T.prim_func
        def kernel(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((N, K), dtype),
            C: T.Tensor((M, N), dtype),
        ):
            with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
                A_shared = T.alloc_shared((block_M, block_K), dtype)
                B_shared = T.alloc_shared((block_N, block_K), dtype)
                C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
                T.clear(C_local)

                for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                    T.copy(A[by * block_M, ko * block_K], A_shared)
                    T.copy(B[bx * block_N, ko * block_K], B_shared)
                    T.gemm(A_shared, B_shared, C_local, transpose_B=True)

                T.copy(C_local, C[by * block_M, bx * block_N])

    return kernel


# ============================================================
# Conv2d forward kernel (implicit im2col + GEMM)
# ============================================================

def make_conv2d_forward_kernel(N, C_in, H, W, C_out, KH, KW, stride=1, padding=0,
                                dtype="float16", accum_dtype="float32",
                                block_M=None, block_N=None, block_K=None,
                                num_stages=None, threads=None):
    """
    Conv2d forward: output = conv2d(input, weight)
    Input: (N, H, W, C_in) -- NHWC layout
    Weight: (KH, KW, C_in, C_out) -- HWCF layout
    Output: (N, OH, OW, C_out) -- NHWC layout

    Implemented as implicit im2col GEMM:
    - M dimension: N * OH * OW (output spatial)
    - K dimension: KH * KW * C_in (filter volume)
    - N dimension: C_out (output channels)
    """
    block_M = block_M or SM80_CONV_BLOCK_M
    block_N = block_N or SM80_CONV_BLOCK_N
    block_K = block_K or SM80_CONV_BLOCK_K
    num_stages = num_stages or SM80_CONV_STAGES
    threads = threads or SM80_CONV_THREADS

    OH = (H + 2 * padding - KH) // stride + 1
    OW = (W + 2 * padding - KW) // stride + 1
    M_total = N * OH * OW
    K_total = KH * KW * C_in

    @T.prim_func
    def kernel(
        data: T.Tensor((N, H, W, C_in), dtype),
        weight: T.Tensor((KH, KW, C_in, C_out), dtype),
        out: T.Tensor((N, OH, OW, C_out), dtype),
    ):
        with T.Kernel(T.ceildiv(C_out, block_N), T.ceildiv(M_total, block_M), threads=threads) as (bx, by):
            data_shared = T.alloc_shared((block_M, block_K), dtype)
            weight_shared = T.alloc_shared((block_K, block_N), dtype)
            out_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            out_shared = T.alloc_shared((block_M, block_N), dtype)

            weight_flat = T.Tensor((K_total, C_out), dtype, weight.data)
            out_flat = T.Tensor((M_total, C_out), dtype, out.data)

            T.clear(out_local)

            for k_iter in T.Pipelined(T.ceildiv(K_total, block_K), num_stages=num_stages):
                # Implicit im2col: compute input coordinates on-the-fly
                for i, j in T.Parallel(block_M, block_K):
                    k = k_iter * block_K + j
                    m = by * block_M + i
                    # Decode m -> (n, oh, ow)
                    n_idx = m // (OH * OW)
                    ohow = m % (OH * OW)
                    oh = ohow // OW
                    ow = ohow % OW
                    # Decode k -> (kh, kw, c_in)
                    kh = k // (KW * C_in)
                    kw = (k // C_in) % KW
                    c = k % C_in
                    # Compute input coordinates with stride and padding
                    ih = oh * stride + kh - padding
                    iw = ow * stride + kw - padding
                    in_bound = (n_idx < N) and (ih >= 0) and (iw >= 0) and (ih < H) and (iw < W)
                    data_shared[i, j] = T.if_then_else(in_bound, data[n_idx, ih, iw, c], 0)

                T.copy(weight_flat[k_iter * block_K, bx * block_N], weight_shared)
                T.gemm(data_shared, weight_shared, out_local)

            T.copy(out_local, out_shared)
            T.copy(out_shared, out_flat[by * block_M, bx * block_N])

    return kernel


# ============================================================
# Conv2d backward kernels
# ============================================================

def make_conv2d_backward_data_kernel(N, C_in, H, W, C_out, KH, KW,
                                      stride=1, padding=0,
                                      dtype="float16", accum_dtype="float32",
                                      block_M=None, block_N=None, block_K=None,
                                      num_stages=None, threads=None):
    """
    Compute gradient w.r.t. input data.
    grad_input = full_conv(grad_output, weight_rotated_180)

    This is implemented as a transposed GEMM:
    grad_input_flat (N*H*W, C_in) = col2im(grad_output) @ weight^T

    For simplicity, we use a scatter approach:
    For each output position and filter tap, accumulate gradient.
    """
    block_M = block_M or SM80_CONV_BLOCK_M
    block_N = block_N or SM80_CONV_BLOCK_N
    block_K = block_K or SM80_CONV_BLOCK_K
    num_stages = num_stages or SM80_CONV_STAGES
    threads = threads or SM80_CONV_THREADS

    OH = (H + 2 * padding - KH) // stride + 1
    OW = (W + 2 * padding - KW) // stride + 1
    M_total = N * H * W
    K_total = KH * KW * C_out

    @T.prim_func
    def kernel(
        grad_out: T.Tensor((N, OH, OW, C_out), dtype),
        weight: T.Tensor((KH, KW, C_in, C_out), dtype),
        grad_input: T.Tensor((N, H, W, C_in), dtype),
    ):
        with T.Kernel(T.ceildiv(C_in, block_N), T.ceildiv(M_total, block_M), threads=threads) as (bx, by):
            grad_shared = T.alloc_shared((block_M, block_K), dtype)
            weight_shared = T.alloc_shared((block_K, block_N), dtype)
            out_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            out_shared = T.alloc_shared((block_M, block_N), dtype)

            grad_input_flat = T.Tensor((M_total, C_in), dtype, grad_input.data)

            T.clear(out_local)

            for k_iter in T.Pipelined(T.ceildiv(K_total, block_K), num_stages=num_stages):
                # For each input position (n, ih, iw), gather from grad_output
                for i, j in T.Parallel(block_M, block_K):
                    k = k_iter * block_K + j
                    m = by * block_M + i
                    n_idx = m // (H * W)
                    hw = m % (H * W)
                    ih = hw // W
                    iw = hw % W
                    # k -> (kh, kw, c_out)
                    kh = k // (KW * C_out)
                    kw = (k // C_out) % KW
                    c_out = k % C_out
                    # The output position that uses this input+filter combo
                    oh_check = ih + padding - kh
                    ow_check = iw + padding - kw
                    valid_stride = (oh_check % stride == 0) and (ow_check % stride == 0)
                    oh = oh_check // stride
                    ow = ow_check // stride
                    in_bound = (n_idx < N) and valid_stride and (oh >= 0) and (ow >= 0) and (oh < OH) and (ow < OW)
                    grad_shared[i, j] = T.if_then_else(in_bound, grad_out[n_idx, oh, ow, c_out], 0)

                # Weight: need (KH*KW*C_out, C_in) view
                # Original weight is (KH, KW, C_in, C_out) -> reshape
                # We need W^T effectively: (K_total_out, C_in)
                # weight[kh, kw, :, c_out] -> for fixed (kh,kw,c_out), all C_in
                # This is a transposed gather, use parallel loop
                for i, j in T.Parallel(block_K, block_N):
                    k = k_iter * block_K + i
                    c_in = bx * block_N + j
                    kh = k // (KW * C_out)
                    kw = (k // C_out) % KW
                    c_out = k % C_out
                    in_bound2 = (kh < KH) and (kw < KW) and (c_in < C_in) and (c_out < C_out)
                    weight_shared[i, j] = T.if_then_else(in_bound2, weight[kh, kw, c_in, c_out], 0)

                T.gemm(grad_shared, weight_shared, out_local)

            T.copy(out_local, out_shared)
            T.copy(out_shared, grad_input_flat[by * block_M, bx * block_N])

    return kernel


def make_conv2d_backward_weight_kernel(N, C_in, H, W, C_out, KH, KW,
                                        stride=1, padding=0,
                                        dtype="float16", accum_dtype="float32",
                                        block_M=None, block_N=None, block_K=None,
                                        num_stages=None, threads=None):
    """
    Compute gradient w.r.t. weight.
    grad_weight = im2col(input)^T @ grad_output_flat

    im2col(input)^T: (KH*KW*C_in, N*OH*OW)
    grad_output_flat: (N*OH*OW, C_out)
    Result: (KH*KW*C_in, C_out)
    """
    block_M = block_M or SM80_CONV_BLOCK_M
    block_N = block_N or SM80_CONV_BLOCK_N
    block_K = block_K or SM80_CONV_BLOCK_K
    num_stages = num_stages or SM80_CONV_STAGES
    threads = threads or SM80_CONV_THREADS

    OH = (H + 2 * padding - KH) // stride + 1
    OW = (W + 2 * padding - KW) // stride + 1
    M_total = KH * KW * C_in
    K_total = N * OH * OW

    @T.prim_func
    def kernel(
        data: T.Tensor((N, H, W, C_in), dtype),
        grad_out: T.Tensor((N, OH, OW, C_out), dtype),
        grad_weight: T.Tensor((KH, KW, C_in, C_out), dtype),
    ):
        with T.Kernel(T.ceildiv(C_out, block_N), T.ceildiv(M_total, block_M), threads=threads) as (bx, by):
            data_shared = T.alloc_shared((block_M, block_K), dtype)
            grad_shared = T.alloc_shared((block_K, block_N), dtype)
            out_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            out_shared = T.alloc_shared((block_M, block_N), dtype)

            grad_out_flat = T.Tensor((K_total, C_out), dtype, grad_out.data)
            grad_weight_flat = T.Tensor((M_total, C_out), dtype, grad_weight.data)

            T.clear(out_local)

            for k_iter in T.Pipelined(T.ceildiv(K_total, block_K), num_stages=num_stages):
                # im2col^T: rows = (kh, kw, c_in), cols = (n, oh, ow)
                for i, j in T.Parallel(block_M, block_K):
                    m = by * block_M + i  # (kh*KW*C_in + kw*C_in + c_in)
                    k = k_iter * block_K + j  # (n*OH*OW + oh*OW + ow)
                    kh = m // (KW * C_in)
                    kw = (m // C_in) % KW
                    c_in = m % C_in
                    n_idx = k // (OH * OW)
                    ohow = k % (OH * OW)
                    oh = ohow // OW
                    ow = ohow % OW
                    ih = oh * stride + kh - padding
                    iw = ow * stride + kw - padding
                    in_bound = (n_idx < N) and (ih >= 0) and (iw >= 0) and (ih < H) and (iw < W)
                    data_shared[i, j] = T.if_then_else(in_bound, data[n_idx, ih, iw, c_in], 0)

                T.copy(grad_out_flat[k_iter * block_K, bx * block_N], grad_shared)
                T.gemm(data_shared, grad_shared, out_local)

            T.copy(out_local, out_shared)
            T.copy(out_shared, grad_weight_flat[by * block_M, bx * block_N])

    return kernel
