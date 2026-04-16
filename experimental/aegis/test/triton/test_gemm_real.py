"""Realistic GEMM profiling test — sizes from transformer workloads."""
import torch
import triton
import triton.language as tl
import time


@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)

        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

        mask_a = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        mask_b = (offs_k[:, None] < K) & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=mask_a, other=0.0)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)

        acc += tl.dot(a, b)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask_c)


def run_gemm(M, N, K, BLOCK_M=128, BLOCK_N=128, BLOCK_K=32, label=""):
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    c = torch.empty(M, N, device="cuda", dtype=torch.float32)

    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)

    # Warmup (not profiled if AEGISBIT captures all dispatches)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    ref = (a.float() @ b.float())
    max_diff = (c - ref).abs().max().item()

    num_wgs = grid[0]
    k_iters = triton.cdiv(K, BLOCK_K)

    print(f"\n{'='*60}")
    print(f"GEMM {label}: M={M} N={N} K={K}  "
          f"BLOCK={BLOCK_M}x{BLOCK_N}x{BLOCK_K}")
    print(f"  Workgroups: {num_wgs}  K-loop iterations: {k_iters}")
    print(f"  Wall time: {(t1-t0)*1000:.1f} ms")
    print(f"  Max diff vs reference: {max_diff:.2e}")
    assert max_diff < 1.0, f"FAIL: max diff {max_diff}"
    print(f"  PASS")
    return t1 - t0


if __name__ == "__main__":
    print(f"PyTorch: {torch.__version__}")
    print(f"Triton:  {triton.__version__}")
    print(f"Device:  {torch.cuda.get_device_name(0)}")

    # Transformer-scale GEMMs (single layer, batch=1):
    #   QKV projection: (seq_len, hidden) × (hidden, 3*hidden)
    #   Attention output: (seq_len, hidden) × (hidden, hidden)
    #   MLP up-proj:     (seq_len, hidden) × (hidden, 4*hidden)

    run_gemm(1024, 4096, 4096, label="QKV projection (1k seq)")

    print(f"\n{'='*60}")
    print("All GEMM tests passed.")
