"""
TileLang GPU kernels with built-in autotuning.
Uses @tilelang.autotune + @tilelang.jit for native fast profiling.
"""

import tilelang
import tilelang.language as T
from tilelang.autotuner import *


# ============================================================
# Autotune configs for SM80
# ============================================================

def get_gemm_configs():
    """Search space for GEMM kernels."""
    configs = []
    for bM in [8, 16, 32, 64, 128]:
        for bN in [8, 16, 32, 64, 128]:
            for bK in [4, 8, 16, 32]:
                for stages in [2, 3]:
                    for threads in [32, 64, 128]:
                        configs.append({
                            "block_M": bM, "block_N": bN, "block_K": bK,
                            "num_stages": stages, "threads": threads,
                        })
    return configs


def get_conv_configs():
    """Search space for conv kernels."""
    configs = []
    for bM in [8, 16, 32, 64]:
        for bN in [8, 16, 32, 64]:
            for bK in [4, 8, 16, 32]:
                for stages in [2, 3]:
                    for threads in [32, 64, 128]:
                        configs.append({
                            "block_M": bM, "block_N": bN, "block_K": bK,
                            "num_stages": stages, "threads": threads,
                        })
    return configs


# ============================================================
# GEMM kernel
# ============================================================

@tilelang.autotune(configs=get_gemm_configs(), skip_check=True)
@tilelang.jit(out_idx=[2])
def gemm_kernel(M, N, K, block_M, block_N, block_K, num_stages, threads,
                dtype=T.float16, accum_dtype=T.float32):
    @T.prim_func
    def main(
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
    return main


# ============================================================
# Conv2d forward
# ============================================================

@tilelang.autotune(configs=get_conv_configs(), skip_check=True)
@tilelang.jit(out_idx=[2])
def conv2d_forward(N, C_in, H, W, C_out, KH, KW, stride, padding,
                   block_M, block_N, block_K, num_stages, threads,
                   dtype=T.float16, accum_dtype=T.float32):
    OH = (H + 2 * padding - KH) // stride + 1
    OW = (W + 2 * padding - KW) // stride + 1
    M_total = N * OH * OW
    K_total = KH * KW * C_in

    @T.prim_func
    def main(
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
                for i, j in T.Parallel(block_M, block_K):
                    k = k_iter * block_K + j
                    m = by * block_M + i
                    n_idx = m // (OH * OW)
                    ohow = m % (OH * OW)
                    oh = ohow // OW
                    ow = ohow % OW
                    kh = k // (KW * C_in)
                    kw = (k // C_in) % KW
                    c = k % C_in
                    ih = oh * stride + kh - padding
                    iw = ow * stride + kw - padding
                    in_bound = (n_idx < N) and (ih >= 0) and (iw >= 0) and (ih < H) and (iw < W)
                    data_shared[i, j] = T.if_then_else(in_bound, data[n_idx, ih, iw, c], 0)

                T.copy(weight_flat[k_iter * block_K, bx * block_N], weight_shared)
                T.gemm(data_shared, weight_shared, out_local)

            T.copy(out_local, out_shared)
            T.copy(out_shared, out_flat[by * block_M, bx * block_N])
    return main


# ============================================================
# Conv2d backward data
# ============================================================

@tilelang.autotune(configs=get_conv_configs(), skip_check=True)
@tilelang.jit(out_idx=[2])
def conv2d_backward_data(N, C_in, H, W, C_out, KH, KW, stride, padding,
                          block_M, block_N, block_K, num_stages, threads,
                          dtype=T.float16, accum_dtype=T.float32):
    OH = (H + 2 * padding - KH) // stride + 1
    OW = (W + 2 * padding - KW) // stride + 1
    M_total = N * H * W
    K_total = KH * KW * C_out

    @T.prim_func
    def main(
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
                for i, j in T.Parallel(block_M, block_K):
                    k = k_iter * block_K + j
                    m = by * block_M + i
                    n_idx = m // (H * W)
                    hw = m % (H * W)
                    ih = hw // W
                    iw = hw % W
                    kh = k // (KW * C_out)
                    kw = (k // C_out) % KW
                    c_out = k % C_out
                    oh_check = ih + padding - kh
                    ow_check = iw + padding - kw
                    valid_stride = (oh_check % stride == 0) and (ow_check % stride == 0)
                    oh = oh_check // stride
                    ow = ow_check // stride
                    in_bound = (n_idx < N) and valid_stride and (oh >= 0) and (ow >= 0) and (oh < OH) and (ow < OW)
                    grad_shared[i, j] = T.if_then_else(in_bound, grad_out[n_idx, oh, ow, c_out], 0)

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
    return main


# ============================================================
# Conv2d backward weight
# ============================================================

@tilelang.autotune(configs=get_conv_configs(), skip_check=True)
@tilelang.jit(out_idx=[2])
def conv2d_backward_weight(N, C_in, H, W, C_out, KH, KW, stride, padding,
                            block_M, block_N, block_K, num_stages, threads,
                            dtype=T.float16, accum_dtype=T.float32):
    OH = (H + 2 * padding - KH) // stride + 1
    OW = (W + 2 * padding - KW) // stride + 1
    M_total = KH * KW * C_in
    K_total = N * OH * OW

    @T.prim_func
    def main(
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
                for i, j in T.Parallel(block_M, block_K):
                    m = by * block_M + i
                    k = k_iter * block_K + j
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
    return main
