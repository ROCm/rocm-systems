##############################################################################
# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
##############################################################################

import torch
import triton
import argparse
from pathlib import Path
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def v3_lds_swizzling(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,  #
    stride_bk,
    stride_bn,  #
    stride_cm,
    stride_cn,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,  #
):

    pid = gl.program_id(axis=0)
    num_pid_n = gl.cdiv(N, BLOCK_N)

    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    gLoadLayoutA: gl.constexpr = gl.BlockedLayout(
        [1, 8],  # sizePerThread
        [512 // BLOCK_K, BLOCK_K // 8],  # threadsPerWarp
        [4, 1],  # warpsPerCTA
        [1, 0],  # order
    )

    gLoadLayoutB: gl.constexpr = gl.BlockedLayout(
        [8, 1],  # sizePerThread
        [BLOCK_K // 8, 512 // BLOCK_K],  # threadsPerWarp
        [1, 4],  # warpsPerCTA
        [0, 1],  # order
    )

    sharedLayoutA: gl.constexpr = gl.SwizzledSharedLayout(8, 2, 8, order=[1, 0])
    sharedLayoutB: gl.constexpr = gl.SwizzledSharedLayout(8, 2, 8, order=[0, 1])

    smemA = gl.allocate_shared_memory(
        a_ptr.dtype.element_ty, [BLOCK_M, BLOCK_K], sharedLayoutA
    )
    smemB = gl.allocate_shared_memory(
        b_ptr.dtype.element_ty, [BLOCK_K, BLOCK_N], sharedLayoutB
    )

    offs_am = gl.arange(0, BLOCK_M, gl.SliceLayout(1, gLoadLayoutA))
    offs_ak = gl.arange(0, BLOCK_K, gl.SliceLayout(0, gLoadLayoutA))

    offs_bn = gl.arange(0, BLOCK_N, gl.SliceLayout(0, gLoadLayoutB))
    offs_bk = gl.arange(0, BLOCK_K, gl.SliceLayout(1, gLoadLayoutB))

    a_base = a_ptr + pid_m * BLOCK_M * stride_am
    b_base = b_ptr + pid_n * BLOCK_N * stride_bn

    a_offsets = offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak
    b_offsets = offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    mfmaLayout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[2, 2]
    )

    dotOpLayoutA: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfmaLayout, k_width=8
    )
    dotOpLayoutB: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfmaLayout, k_width=8
    )

    acc = gl.zeros((BLOCK_M, BLOCK_N), gl.float32, mfmaLayout)

    max_iter = gl.cdiv(K, BLOCK_K)
    gl.assume(max_iter > 0)

    for k in range(0, max_iter):
        gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA, a_base, a_offsets)
        gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB, b_base, b_offsets)
        gl.amd.cdna4.async_copy.commit_group()
        gl.amd.cdna4.async_copy.wait_group(0)
        a = smemA.load(dotOpLayoutA)
        b = smemB.load(dotOpLayoutB)

        acc = gl.amd.cdna3.mfma(a, b, acc)

        a_base += BLOCK_K * stride_ak
        b_base += BLOCK_K * stride_bk

    c = acc.to(a_ptr.dtype.element_ty)

    gStoreLayoutC: gl.constexpr = mfmaLayout
    c = gl.convert_layout(c, layout=gStoreLayoutC)
    offs_cm = gl.arange(0, BLOCK_M, gl.SliceLayout(1, gStoreLayoutC))
    offs_cn = gl.arange(0, BLOCK_N, gl.SliceLayout(0, gStoreLayoutC))
    c_base = c_ptr + pid_m * BLOCK_M * stride_cm + pid_n * BLOCK_N * stride_cn
    c_offsets = stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    gl.amd.cdna3.buffer_store(
        ptr=c_base, offsets=c_offsets, stored_value=c, mask=c_mask
    )


@gluon.jit
def v3_lds_padding(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,  #
    stride_bk,
    stride_bn,  #
    stride_cm,
    stride_cn,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,  #
):

    pid = gl.program_id(axis=0)
    num_pid_n = gl.cdiv(N, BLOCK_N)

    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    gLoadLayoutA: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2], [0, 4], [4, 0], [8, 0], [128, 0]],
        lane_bases=[[0, 8], [0, 16], [0, 32], [16, 0], [32, 0], [64, 0]],
        warp_bases=[[1, 0], [2, 0]],
        block_bases=[],
        shape=[BLOCK_M, BLOCK_K],
    )
    gLoadLayoutB: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[1, 0], [2, 0], [4, 0], [0, 4], [0, 8], [0, 128]],
        lane_bases=[[8, 0], [16, 0], [32, 0], [0, 16], [0, 32], [0, 64]],
        warp_bases=[[0, 1], [0, 2]],
        block_bases=[],
        shape=[BLOCK_K, BLOCK_N],
    )

    sharedLayoutA: gl.constexpr = gl.PaddedSharedLayout(
        [[512, 16]],
        [
            [0, 1],
            [0, 2],
            [0, 4],
            [0, 8],
            [0, 16],
            [0, 32],
            [16, 0],
            [32, 0],
            [64, 0],
            [1, 0],
            [2, 0],
            [4, 0],
            [8, 0],
            [128, 0],
        ],
        [],
        [BLOCK_M, BLOCK_K],
    )
    sharedLayoutB: gl.constexpr = gl.PaddedSharedLayout(
        [[512, 16]],
        [
            [1, 0],
            [2, 0],
            [4, 0],
            [8, 0],
            [16, 0],
            [32, 0],
            [0, 16],
            [0, 32],
            [0, 64],
            [0, 1],
            [0, 2],
            [0, 4],
            [0, 8],
            [0, 128],
        ],
        [],
        [BLOCK_K, BLOCK_N],
    )

    smemA = gl.allocate_shared_memory(
        a_ptr.dtype.element_ty, [BLOCK_M, BLOCK_K], sharedLayoutA
    )
    smemB = gl.allocate_shared_memory(
        b_ptr.dtype.element_ty, [BLOCK_K, BLOCK_N], sharedLayoutB
    )

    offs_am = gl.arange(0, BLOCK_M, gl.SliceLayout(1, gLoadLayoutA))
    offs_ak = gl.arange(0, BLOCK_K, gl.SliceLayout(0, gLoadLayoutA))

    offs_bn = gl.arange(0, BLOCK_N, gl.SliceLayout(0, gLoadLayoutB))
    offs_bk = gl.arange(0, BLOCK_K, gl.SliceLayout(1, gLoadLayoutB))

    a_base = a_ptr + pid_m * BLOCK_M * stride_am
    b_base = b_ptr + pid_n * BLOCK_N * stride_bn

    a_offsets = offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak
    b_offsets = offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    mfmaLayout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[2, 2]
    )

    dotOpLayoutA: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfmaLayout, k_width=8
    )
    dotOpLayoutB: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfmaLayout, k_width=8
    )

    acc = gl.zeros((BLOCK_M, BLOCK_N), gl.float32, mfmaLayout)

    for k in range(0, gl.cdiv(K, BLOCK_K)):
        gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA, a_base, a_offsets)
        gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB, b_base, b_offsets)
        gl.amd.cdna4.async_copy.commit_group()
        gl.amd.cdna4.async_copy.wait_group(0)
        a = smemA.load(dotOpLayoutA)
        b = smemB.load(dotOpLayoutB)

        acc = gl.amd.cdna3.mfma(a, b, acc)

        a_base += BLOCK_K * stride_ak
        b_base += BLOCK_K * stride_bk

    c = acc.to(a_ptr.dtype.element_ty)

    gStoreLayoutC: gl.constexpr = mfmaLayout
    c = gl.convert_layout(c, layout=gStoreLayoutC)
    offs_cm = gl.arange(0, BLOCK_M, gl.SliceLayout(1, gStoreLayoutC))
    offs_cn = gl.arange(0, BLOCK_N, gl.SliceLayout(0, gStoreLayoutC))
    c_base = c_ptr + pid_m * BLOCK_M * stride_cm + pid_n * BLOCK_N * stride_cn
    c_offsets = stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    gl.amd.cdna3.buffer_store(
        ptr=c_base, offsets=c_offsets, stored_value=c, mask=c_mask
    )


def matmul(a, b, c=None, *, block_m=256, block_n=256, block_k=128, use_swizzling=True):
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert b.is_contiguous(), "Matrix B must be contiguous"
    M, K = a.shape
    K, N = b.shape
    BLOCK_M, BLOCK_N, BLOCK_K = block_m, block_n, block_k
    num_warps = 4
    if c is None:
        c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    GRID_MN = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    grid = (GRID_MN, 1)
    kernel = v3_lds_swizzling if use_swizzling else v3_lds_padding
    kernel[grid](
        a,
        b,
        c,  #
        M,
        N,
        K,  #
        a.stride(0),
        a.stride(1),  #
        b.stride(0),
        b.stride(1),  #
        c.stride(0),
        c.stride(1),  #
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=num_warps,
        warp_size=64,
    )
    return c


def compile_kernel(*, layout, block_m, block_n, block_k, arch, output):
    from triton.backends.compiler import GPUTarget
    from triton.compiler import compile
    from triton.experimental.gluon._runtime import GluonASTSource

    kernel = v3_lds_swizzling if layout == "swizzling" else v3_lds_padding
    signature = {
        "a_ptr": "*fp16",
        "b_ptr": "*fp16",
        "c_ptr": "*fp16",
        "M": "i32",
        "N": "i32",
        "K": "i32",
        "stride_am": "i32",
        "stride_ak": "i32",
        "stride_bk": "i32",
        "stride_bn": "i32",
        "stride_cm": "i32",
        "stride_cn": "i32",
    }
    constexprs = {"BLOCK_M": block_m, "BLOCK_N": block_n, "BLOCK_K": block_k}
    source = GluonASTSource(kernel, signature, constexprs, attrs={})
    compiled = compile(
        source,
        target=GPUTarget("hip", arch, 64),
        options={"num_warps": 4, "num_ctas": 1, "num_stages": 2},
    )
    output.write_bytes(compiled.asm["hsaco"])
    print("compiled", output)
    print("kernel", compiled.metadata.name)
    print("bytes", output.stat().st_size)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=256)
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--block-m", type=int, default=256)
    parser.add_argument("--block-n", type=int, default=256)
    parser.add_argument("--block-k", type=int, default=64)
    parser.add_argument("--layout", choices=("padding", "swizzling"), default="padding")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--arch", default="gfx950")
    parser.add_argument(
        "--output", type=Path, default=Path("/tmp/gfx950_gluon_lds_matmul.hsaco")
    )
    args = parser.parse_args()

    if args.compile_only:
        compile_kernel(
            layout=args.layout,
            block_m=args.block_m,
            block_n=args.block_n,
            block_k=args.block_k,
            arch=args.arch,
            output=args.output,
        )
        return

    torch.manual_seed(0)
    dtype = torch.float16
    device = "cuda"
    m = args.m
    n = args.n
    k = args.k
    block_m = args.block_m
    block_n = args.block_n
    block_k = args.block_k

    a = torch.randn((m, k), device=device, dtype=dtype)
    b = torch.randn((k, n), device=device, dtype=dtype)
    c = matmul(
        a,
        b,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        use_swizzling=args.layout == "swizzling",
    )
    torch.cuda.synchronize()

    ref = torch.matmul(a, b)
    torch.cuda.synchronize()
    diff = (c.float() - ref.float()).abs()
    print("shape", tuple(c.shape))
    print("dtype", c.dtype)
    print("block", block_m, block_n, block_k)
    print("layout", args.layout)
    print("shared_bytes_nominal", 2 * (block_m * block_k + block_k * block_n))
    print("finite", bool(torch.isfinite(c).all().item()))
    print("max_abs_diff", float(diff.max().item()))
    print("sum", float(c.float().sum().item()))
    print("first8", [float(v) for v in c.flatten()[:8].float().cpu()])
    print("ptrs", hex(a.data_ptr()), hex(b.data_ptr()), hex(c.data_ptr()))


if __name__ == "__main__":
    main()
