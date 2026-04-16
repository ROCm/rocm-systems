#!/usr/bin/env python3
"""
Coalescing validation test — runs two kernels back-to-back:
  1. Coalesced: each lane reads a consecutive element (stride-1)
  2. Scattered: each lane reads a random element (worst-case)

Expected profiling output:
  - Coalesced kernel: ~1-2 cache lines per access, eff ≈ 100%
  - Scattered kernel: many cache lines per access, eff << 100%

Run:
    AEGISBIT_ENABLED=1 AEGISBIT_MODE=MEMORY_ONLY \
    AEGISBIT_KERNELS="*coalesced*,*scattered*" \
    AEGISBIT_DRY_PAYLOAD=99 \
    LD_PRELOAD=build/src/libaegisbit.so \
    python3 test/triton/coalescing_test.py
"""
import torch
import triton
import triton.language as tl

# --- Kernel 1: perfectly coalesced (stride-1 access) ---
@triton.jit
def coalesced_copy_kernel(
    In_ptr, Out_ptr, N,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    # Stride-1: lane k reads element (pid*BLOCK + k) — consecutive
    x = tl.load(In_ptr + offs, mask=mask)
    tl.store(Out_ptr + offs, x, mask=mask)

# --- Kernel 2: scattered (random index per lane) ---
@triton.jit
def scattered_copy_kernel(
    In_ptr, Out_ptr, Idx_ptr, N,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    # Each lane reads a random index — worst-case coalescing
    idx = tl.load(Idx_ptr + offs, mask=mask)
    x = tl.load(In_ptr + idx, mask=mask)
    tl.store(Out_ptr + offs, x, mask=mask)

# --- Kernel 3: strided (every 64th element — column-major-like) ---
@triton.jit
def strided_copy_kernel(
    In_ptr, Out_ptr, N, stride,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    # Lane k reads element k*stride — touches a different cache line per lane
    src_offs = offs * stride
    x = tl.load(In_ptr + src_offs, mask=mask)
    tl.store(Out_ptr + offs, x, mask=mask)


def main():
    print(f"PyTorch: {torch.__version__}")
    print(f"Triton:  {triton.__version__}")
    print(f"Device:  {torch.cuda.get_device_name(0)}")
    print()

    N = 65536
    BLOCK = 64

    inp = torch.randn(N, device="cuda", dtype=torch.float32)
    out = torch.empty(N, device="cuda", dtype=torch.float32)
    grid = (triton.cdiv(N, BLOCK),)

    # --- Test 1: Coalesced ---
    print("=" * 60)
    print("Kernel 1: coalesced_copy (stride-1, expect ~2 cachelines)")
    print("=" * 60)
    coalesced_copy_kernel[grid](inp, out, N, BLOCK=BLOCK)
    torch.cuda.synchronize()
    diff = (out - inp).abs().max().item()
    print(f"  Correctness: max_diff = {diff:.2e}")
    assert diff < 1e-6, f"FAIL: {diff}"
    print("  PASS\n")

    # --- Test 2: Scattered ---
    print("=" * 60)
    print("Kernel 2: scattered_copy (random idx, expect ~64 cachelines)")
    print("=" * 60)
    idx = torch.randint(0, N, (N,), device="cuda", dtype=torch.int32)
    out2 = torch.empty(N, device="cuda", dtype=torch.float32)
    scattered_copy_kernel[grid](inp, out2, idx, N, BLOCK=BLOCK)
    torch.cuda.synchronize()
    ref = inp[idx.long()]
    diff = (out2 - ref).abs().max().item()
    print(f"  Correctness: max_diff = {diff:.2e}")
    assert diff < 1e-6, f"FAIL: {diff}"
    print("  PASS\n")

    # --- Test 3: Strided (stride=128 elements = 512 bytes, every 4 cache lines) ---
    print("=" * 60)
    print("Kernel 3: strided_copy (stride=128, expect ~64 cachelines)")
    print("=" * 60)
    big_inp = torch.randn(N * 128, device="cuda", dtype=torch.float32)
    out3 = torch.empty(N, device="cuda", dtype=torch.float32)
    strided_copy_kernel[grid](big_inp, out3, N, 128, BLOCK=BLOCK)
    torch.cuda.synchronize()
    ref3 = big_inp[torch.arange(N, device="cuda") * 128]
    diff = (out3 - ref3).abs().max().item()
    print(f"  Correctness: max_diff = {diff:.2e}")
    assert diff < 1e-6, f"FAIL: {diff}"
    print("  PASS\n")

    print("All tests passed.")


if __name__ == "__main__":
    main()
