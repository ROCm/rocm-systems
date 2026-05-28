# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Torch-dependent helpers + OP_SPECS for test_torch_trace_coverage.

OP_SPECS: ATen "<name>[.overload]" -> OpSpec (builder or skip reason).
Use CoverageTensorArg for structural inputs (probs, SPD, etc.).
"""

import os
import shutil
import signal
import subprocess
import sys
import textwrap
import threading
import uuid
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Set, Tuple

import pandas as pd
import pytest
import torch


class CoverageTensorArg(NamedTuple):
    """Tensor slot with a fixed factory (rand/spd/pivots_1based/etc.)
    for cases where randn would violate a structural invariant.
    """

    shape: Tuple[int, ...]
    emit: str
    scale: float = 4.0


ArgBuilder = Callable[[str], Tuple[List[Any], Dict[str, Any]]]


@dataclass(frozen=True)
class OpSpec:
    """One op: either a builder that produces (args, kwargs), or a skip reason."""

    build: Optional[ArgBuilder] = None
    skip: Optional[str] = None


# -----------------------------------------------------------------------------
# Compact tensor factories used by builders
# -----------------------------------------------------------------------------


def f(device: str, *shape: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.randn(shape, device=device, dtype=dtype)


def i(device: str, *shape: int, low: int = 0, high: int = 8) -> torch.Tensor:
    return torch.randint(low, high, shape, device=device, dtype=torch.int64)


def i1(device: str, n: int = 8) -> torch.Tensor:
    return torch.randint(0, 4, (n,), device=device, dtype=torch.int64)


def b(device: str, shape: Tuple[int, ...] = (4, 4)) -> torch.Tensor:
    return torch.ones(shape, device=device, dtype=torch.bool)


def r01(device: str, *shape: int) -> torch.Tensor:
    return torch.rand(shape, device=device)


# Symmetric positive-definite, for Cholesky-family ops.
def spd(device: str, n: int = 4) -> torch.Tensor:
    a = torch.randn(n, n, device=device)
    return a @ a.mT + torch.eye(n, device=device) * 0.1


# -----------------------------------------------------------------------------
# Multi-line builder helpers (kept out of the table for readability)
# -----------------------------------------------------------------------------


def linalg_householder_product_args(device: str) -> Tuple[List[Any], Dict[str, Any]]:
    """Build (input, tau) for linalg_householder_product across torch builds.

    Some ROCm wheels omit torch.linalg.geqrf; fall back to the top-level
    torch.geqrf, then to a CPU geqrf copied to device. Last
    resort: tensors with valid ranks only (may be numerically weaker).
    """
    a = torch.randn(6, 4, device=device)
    for geqrf in (getattr(torch.linalg, "geqrf", None), getattr(torch, "geqrf", None)):
        if geqrf is None:
            continue
        try:
            qr_tau = geqrf(a)
            return [qr_tau[0], qr_tau[1]], {}
        except Exception:
            pass
        try:
            qr_tau = geqrf(torch.randn(6, 4))
        except Exception:
            continue
        return [qr_tau[0].to(device), qr_tau[1].to(device)], {}
    inp = torch.randn(6, 4, device=device)
    tau = torch.randn(4, device=device)
    return [inp, tau], {}


def lu_unpack_args(device: str) -> Tuple[List[Any], Dict[str, Any]]:
    # Schema: (LU_data, LU_pivots, unpack_data=True, unpack_pivots=True).
    # LU_pivots is used by the kernel as row-swap indices, so its *values*
    # matter, not just shape/dtype.  CoverageTensorArg(emit="pivots_1based")
    # tells the emitter to produce torch.arange(1, n+1, dtype=int32) — a
    # valid identity permutation.  Plain randint indices would walk off
    # the row table and trigger HIP 719.
    return [
        torch.randn(4, 4, device=device),
        CoverageTensorArg(shape=(4,), emit="pivots_1based"),
        True,
        True,
    ], {}


def lu_solve_args(device: str) -> Tuple[List[Any], Dict[str, Any]]:
    # Schema: (LU, pivots, B, *, left=True, adjoint=False).  See lu_unpack_args
    # comment for why pivots uses the pivots_1based emit mode.
    return [
        torch.randn(4, 4, device=device),
        CoverageTensorArg(shape=(4,), emit="pivots_1based"),
        torch.randn(4, 2, device=device),
    ], {}


def ldl_solve_args(device: str) -> Tuple[List[Any], Dict[str, Any]]:
    ld, pivots, _ = torch.linalg.ldl_factor_ex(spd(device))
    return [ld, pivots.to(dtype=torch.int64), torch.randn(4, 2, device=device)], {}


def solve_triangular_args(device: str) -> Tuple[List[Any], Dict[str, Any]]:
    # Schema: linalgsolve_triangular_args(self=A, B, *, upper, left=True,
    # unitriangular=False).  upper is kwarg-only, and the first positional
    # is A (triangular), not B.
    a = torch.randn(4, 4, device=device).tril()
    b = torch.randn(4, 2, device=device)
    return [a, b], {"upper": False, "left": True, "unitriangular": False}


def ormqr_args(device: str) -> Tuple[List[Any], Dict[str, Any]]:
    # Schema: (self[m,k], tau[k], other[m,n], left=True, transpose=False).
    # With left=True the constraint is other.shape[-2] == self.shape[-2]
    # so pick square A = (4,4), tau = (4,), other = (4, 2).
    return (
        [
            torch.randn(4, 4, device=device),
            torch.randn(4, device=device),
            torch.randn(4, 2, device=device),
            True,
        ],
        {},
    )


def convolution_backward_args(device: str) -> Tuple[List[Any], Dict[str, Any]]:
    # Schema (11 positionals):
    #   grad_output, input, weight, bias_sizes?, stride, padding, dilation,
    #   transposed, output_padding, groups, output_mask[3]
    return (
        [
            torch.randn(1, 1, 6, 6, device=device),  # grad_output (N,C_out,H,W)
            torch.randn(1, 1, 8, 8, device=device),  # input
            torch.randn(1, 1, 3, 3, device=device),  # weight
            None,  # bias_sizes (no bias)
            [1, 1],  # stride
            [0, 0],  # padding
            [1, 1],  # dilation
            False,  # transposed
            [0, 0],  # output_padding
            1,  # groups
            [True, True, False],  # output_mask
        ],
        {},
    )


def native_batch_norm_backward_args(
    device: str,
) -> Tuple[List[Any], Dict[str, Any]]:
    # Schema: (grad_out, input, weight?, running_mean?, running_var?,
    #          save_mean?, save_invstd?, train, eps, output_mask[3]).
    return (
        [
            torch.randn(2, 3, 4, 4, device=device),  # grad_out
            torch.randn(2, 3, 4, 4, device=device),  # input
            torch.ones(3, device=device),  # weight
            torch.zeros(3, device=device),  # running_mean
            torch.ones(3, device=device),  # running_var
            torch.zeros(3, device=device),  # save_mean
            torch.ones(3, device=device),  # save_invstd
            True,  # train
            1e-5,  # eps
            [True, True, True],  # output_mask
        ],
        {},
    )


def batch_norm_backward_args(device: str) -> Tuple[List[Any], Dict[str, Any]]:
    # Pass an empty reserve tensor for compatibility with schemas that expose
    # a trailing reserve argument on this op.
    args, kwargs = native_batch_norm_backward_args(device)
    args.append(torch.empty(0, device=device))  # reserve
    return args, kwargs


def native_group_norm_backward_args(device: str) -> Tuple[List[Any], Dict[str, Any]]:
    # Schema: (grad_out, input, mean, rstd, weight?, N, C, HxW, group,
    #          output_mask[3]). mean/rstd shape is (N, groups).
    return (
        [
            torch.randn(2, 6, 4, 4, device=device),  # grad_out
            torch.randn(2, 6, 4, 4, device=device),  # input
            torch.zeros(2, 2, device=device),  # mean (N, groups)
            torch.ones(2, 2, device=device),  # rstd (N, groups)
            torch.ones(6, device=device),  # weight
            2,  # N
            6,  # C
            16,  # HxW
            2,  # group
            [True, True, True],  # output_mask
        ],
        {},
    )


def fused_adam_args(device: str) -> Tuple[List[Any], Dict[str, Any]]:
    """Build list-of-tensors positionals for _fused_adam_/_fused_adamw_.
    Lengths must match; max_exp_avg_sqs empty when amsgrad=False;
    state_steps elements are scalar float tensors.
    """
    params = [torch.randn(4, device=device)]
    grads = [torch.randn(4, device=device)]
    exp_avgs = [torch.zeros(4, device=device)]
    exp_avg_sqs = [torch.zeros(4, device=device)]
    max_exp_avg_sqs: List[torch.Tensor] = []
    state_steps = [torch.tensor(1.0, device=device)]
    return (
        [params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps],
        {
            "lr": 1e-3,
            "beta1": 0.9,
            "beta2": 0.999,
            "weight_decay": 0.0,
            "eps": 1e-8,
            "amsgrad": False,
            "maximize": False,
        },
    )


def native_layer_norm_backward_args(device: str) -> Tuple[List[Any], Dict[str, Any]]:
    # Schema: (grad_out, input, normalized_shape, mean, rstd, weight?, bias?,
    #          output_mask[3]). mean/rstd broadcast across non-normalized dims.
    return (
        [
            torch.randn(2, 4, 4, device=device),  # grad_out
            torch.randn(2, 4, 4, device=device),  # input
            [4],  # normalized_shape
            torch.randn(2, 4, 1, device=device),  # mean
            torch.randn(2, 4, 1, device=device),  # rstd
            torch.ones(4, device=device),  # weight
            torch.zeros(4, device=device),  # bias
            [True, True, True],  # output_mask
        ],
        {},
    )


# -----------------------------------------------------------------------------
# The unified table.  Keys are ATen short names.
# -----------------------------------------------------------------------------
#
# Skip entries appear first, followed by hardcoded-argument builders
# grouped by operator category. Adding a new operator to the coverage
# workload requires exactly one entry in this table (OpSpec.build or
# OpSpec.skip); removing an entry causes build_args_for_op to
# report the operator as SKIP.

OP_SPECS: Dict[str, OpSpec] = {
    # ---------------------------------------------------------------
    # Skipped entirely: the op advertises CUDA dispatch but faults at
    # runtime under our synthetic workload (or specifically under
    # torch.profiler).
    # ---------------------------------------------------------------
    "_sparse_semi_structured_mm": OpSpec(skip="CUTLASS 2:4 sparse; NVIDIA-only."),
    # FFT ops SIGSEGV inside torch.profiler on ROCm (kernel + ROCTracer race).
    # They run fine outside the profiler, but the test needs profiler ground truth.
    "_fft_r2c": OpSpec(skip="SIGSEGV under torch.profiler on ROCm."),
    "_fft_c2c": OpSpec(skip="SIGSEGV under torch.profiler on ROCm."),
    "_fft_c2r": OpSpec(skip="SIGSEGV under torch.profiler on ROCm."),
    "_cudnn_rnn": OpSpec(skip="cuDNN-only; ROCm uses miopen_rnn instead."),
    "_cdist_backward": OpSpec(
        skip="Requires cdist output tied to x1/x2; serialized workload cannot "
        "preserve that correlation.",
    ),
    # igamma / igammac kernels hit HIP 719 (illegal memory access) on ROCm
    # and poison the HIP context.
    "igamma": OpSpec(skip="ROCm kernel: HIP 719 illegal memory access."),
    "igammac": OpSpec(skip="ROCm kernel: HIP 719 illegal memory access."),
    "cudnn_batch_norm_backward": OpSpec(
        skip="Needs save_mean / save_var / reserveSpace from a matching forward.",
    ),
    "cudnn_batch_norm": OpSpec(skip="cuDNN-only; ROCm uses miopen_batch_norm."),
    # Specialised quantised / low-precision GEMMs: signatures require matching
    # qScaleAndZeros / scales / layouts that are backend-specific. Skip until a
    # proper per-backend builder exists.
    "_weight_int4pack_mm": OpSpec(
        skip="Requires packed int4 weights + matching qScaleAndZeros layout.",
    ),
    "_weight_int8pack_mm": OpSpec(
        skip="Requires int8 packed weights + matching scales on supported builds.",
    ),
    "_native_multi_head_attention": OpSpec(
        skip="Requires specific QKV/mask layout + matching projection weights.",
    ),
    # observer_on, fake_quant_on, zero_point must be int32 (op rejects int64).
    "_fused_moving_avg_obs_fq_helper": OpSpec(
        build=lambda d: (
            [
                torch.randn(4, 4, device=d),
                torch.ones(1, device=d, dtype=torch.int32),
                torch.ones(1, device=d, dtype=torch.int32),
                torch.zeros(1, device=d),
                torch.zeros(1, device=d),
                torch.ones(1, device=d),
                torch.zeros(1, device=d, dtype=torch.int32),
                0.01,
                0,
                255,
                -1,
                False,
                False,
            ],
            {},
        ),
    ),
    "_scaled_grouped_mm": OpSpec(
        skip="Requires mat2 to be transposed with scale tensors; backend-specific.",
    ),
    # Nested-tensor APIs need a coherent values/offsets/max_lengths contract
    # the emitter can't yet synthesize.
    "_jagged_to_padded_dense_forward": OpSpec(skip="shape contract TBD."),
    "_padded_dense_to_jagged_forward": OpSpec(skip="shape contract TBD."),
    # ---------------------------------------------------------------
    # matmul / batched matmul / conv
    # ---------------------------------------------------------------
    "bmm": OpSpec(build=lambda d: ([f(d, 2, 4, 4), f(d, 2, 4, 4)], {})),
    "conv1d": OpSpec(build=lambda d: ([f(d, 1, 3, 16), f(d, 6, 3, 3)], {})),
    "conv2d": OpSpec(build=lambda d: ([f(d, 1, 3, 8, 8), f(d, 6, 3, 3, 3)], {})),
    "conv3d": OpSpec(
        build=lambda d: ([f(d, 1, 3, 4, 4, 4), f(d, 6, 3, 3, 3, 3)], {}),
    ),
    "conv_transpose1d": OpSpec(
        build=lambda d: ([f(d, 1, 3, 16), f(d, 3, 6, 3)], {}),
    ),
    "conv_transpose2d": OpSpec(
        build=lambda d: ([f(d, 1, 3, 8, 8), f(d, 3, 6, 3, 3)], {}),
    ),
    "conv_transpose3d": OpSpec(
        build=lambda d: ([f(d, 1, 3, 4, 4, 4), f(d, 3, 6, 3, 3, 3)], {}),
    ),
    "addmm": OpSpec(build=lambda d: ([f(d, 4), f(d, 4, 4), f(d, 4, 4)], {})),
    "addbmm": OpSpec(
        build=lambda d: ([f(d, 4, 4), f(d, 2, 4, 4), f(d, 2, 4, 4)], {}),
    ),
    "addbmm_": OpSpec(
        build=lambda d: ([f(d, 4, 4), f(d, 2, 4, 4), f(d, 2, 4, 4)], {}),
    ),
    "baddbmm": OpSpec(
        build=lambda d: ([f(d, 2, 4, 4), f(d, 2, 4, 4), f(d, 2, 4, 4)], {}),
    ),
    "baddbmm_": OpSpec(
        build=lambda d: ([f(d, 2, 4, 4), f(d, 2, 4, 4), f(d, 2, 4, 4)], {}),
    ),
    "addmv": OpSpec(build=lambda d: ([f(d, 4), f(d, 4, 4), f(d, 4)], {})),
    "addmv_": OpSpec(build=lambda d: ([f(d, 4), f(d, 4, 4), f(d, 4)], {})),
    "addr": OpSpec(build=lambda d: ([f(d, 4, 4), f(d, 4), f(d, 4)], {})),
    "dot": OpSpec(build=lambda d: ([f(d, 4), f(d, 4)], {})),
    "vdot": OpSpec(build=lambda d: ([f(d, 4), f(d, 4)], {})),
    # ---------------------------------------------------------------
    # embedding / embedding_bag
    # ---------------------------------------------------------------
    "embedding": OpSpec(
        build=lambda d: ([f(d, 10, 8), torch.randint(0, 10, (4,), device=d)], {}),
    ),
    # grad_output (*, embed_dim), indices must be Long; num_weights > max index.
    "embedding_dense_backward": OpSpec(
        build=lambda d: (
            [f(d, 8, 16), torch.randint(0, 32, (8,), device=d), 32, -1, False],
            {},
        ),
    ),
    # Schema: (weight, indices, offsets, scale_grad_by_freq=False, mode=0,
    #          sparse=False, per_sample_weights=None, include_last_offset=False,
    #          padding_idx=-1). offsets must be int64 (not float) and
    #          describe the bag boundaries into indices.
    "_embedding_bag": OpSpec(
        build=lambda d: (
            [
                f(d, 10, 8),
                i1(d, 6),
                torch.tensor([0, 3], device=d, dtype=torch.int64),
            ],
            {},
        ),
    ),
    "_embedding_bag_forward_only": OpSpec(
        build=lambda d: (
            [
                f(d, 10, 8),
                i1(d, 6),
                torch.tensor([0, 3], device=d, dtype=torch.int64),
                False,
                0,
                False,
            ],
            {},
        ),
    ),
    # Backward variants require matching offset2bag / bag_size /
    # maximum_indices tensors produced by the forward; faking them reliably
    # across all modes (sum/mean/max) is brittle, so skip them.
    "_embedding_bag_backward": OpSpec(
        skip="Requires offset2bag/bag_size from matched forward run.",
    ),
    "_embedding_bag_dense_backward": OpSpec(
        skip="Requires offset2bag/bag_size from matched forward run.",
    ),
    "_embedding_bag_per_sample_weights_backward": OpSpec(
        skip="Requires offset2bag from matched forward run.",
    ),
    # ---------------------------------------------------------------
    # Linalg / Cholesky (need structured input)
    # ---------------------------------------------------------------
    "linalg_householder_product": OpSpec(build=linalg_householder_product_args),
    # cholesky raises _LinAlgError on non-PD input. CoverageTensorArg(
    # emit="spd") tells the emitter to produce a @ a.mT + n * I —
    # diagonal-dominant, always SPD. The build-time _spd helper was not
    # enough because serialize_arg only reads shape+dtype and would drop
    # the SPD structure. linalg_cholesky_ex / cholesky_inverse do not
    # raise, but they still need an SPD input for meaningful kernels.
    "cholesky": OpSpec(
        build=lambda d: ([CoverageTensorArg(shape=(4, 4), emit="spd")], {}),
    ),
    "linalg_cholesky_ex": OpSpec(
        build=lambda d: ([CoverageTensorArg(shape=(4, 4), emit="spd")], {}),
    ),
    "cholesky_inverse": OpSpec(
        build=lambda d: ([CoverageTensorArg(shape=(4, 4), emit="spd")], {}),
    ),
    # Schema: _cholesky_solve_helper(self, A, bool upper).
    "_cholesky_solve_helper": OpSpec(
        build=lambda d: ([f(d, 4, 2), spd(d).tril(), False], {}),
    ),
    "linalg_cross": OpSpec(build=lambda d: ([f(d, 2, 3), f(d, 2, 3)], {})),
    # ldl_solve on CUDA/ROCm requires PyTorch built with MAGMA/LAPACK support;
    # not available in common ROCm builds.
    "linalg_ldl_solve": OpSpec(
        skip="Requires the MAGMA library (libmagma) for the GPU path.  On "
        "ROCm the equivalent would be rocSOLVER, but PyTorch's LDL solve "
        "currently only dispatches to MAGMA.  Cannot be enabled by "
        "import: PyTorch must be built against MAGMA for this op to be "
        "callable on GPU.",
    ),
    # lu_unpack / linalglu_solve_args feed LU_pivots to the kernel as
    # row-swap indices, so the *values* must form a valid 1-based
    # permutation (otherwise the kernel reads past the end of the row
    # table → HIP 719).  lu_unpack_args / lu_solve_args use
    # CoverageTensorArg(emit="pivots_1based") which serializes to
    # torch.arange(1, n+1, dtype=int32) — a safe identity permutation.
    "linalg_lu_solve": OpSpec(build=lu_solve_args),
    "linalg_solve_triangular": OpSpec(build=solve_triangular_args),
    "lu_unpack": OpSpec(build=lu_unpack_args),
    "ormqr": OpSpec(build=ormqr_args),
    # ---------------------------------------------------------------
    # Loss functions
    # ---------------------------------------------------------------
    "cross_entropy_loss": OpSpec(
        build=lambda d: ([f(d, 4, 10), torch.randint(0, 10, (4,), device=d)], {}),
    ),
    # input / target in [0, 1]; randn has triggered ROCm HSA
    # exceptions in binary_cross_entropy*_cuda.
    "binary_cross_entropy": OpSpec(
        build=lambda _d: (
            [CoverageTensorArg((4, 4), "rand"), CoverageTensorArg((4, 4), "rand")],
            {},
        ),
    ),
    "binary_cross_entropy_backward": OpSpec(
        build=lambda d: (
            [
                f(d, 4, 4),
                CoverageTensorArg((4, 4), "rand"),
                CoverageTensorArg((4, 4), "rand"),
            ],
            {},
        ),
    ),
    "nll_loss_forward": OpSpec(
        build=lambda d: (
            [
                f(d, 4, 10).log_softmax(1),
                torch.randint(0, 10, (4,), device=d),
                torch.ones(10, device=d),
                0,
                -100,
            ],
            {},
        ),
    ),
    # Schema: nll_loss_backward(grad_output, self, target, weight?, reduction,
    #                          ignore_index, total_weight).
    # With reduction=Mean (=1), grad_output/total_weight are scalars.
    "nll_loss_backward": OpSpec(
        build=lambda d: (
            [
                torch.tensor(1.0, device=d),  # grad_output
                f(d, 4, 5).log_softmax(1),  # self
                torch.randint(0, 5, (4,), device=d),  # target
                None,  # weight
                1,  # reduction (Mean)
                -100,  # ignore_index
                torch.tensor(4.0, device=d),  # total_weight
            ],
            {},
        ),
    ),
    "nll_loss2d_forward": OpSpec(
        build=lambda d: (
            [
                f(d, 2, 5, 4, 4).log_softmax(1),
                torch.randint(0, 5, (2, 4, 4), device=d),
                torch.ones(5, device=d),
                0,
                -100,
            ],
            {},
        ),
    ),
    # Same schema as the 1-D variant but with (N, C, H, W) inputs / (N, H, W)
    # targets.  total_weight approximates N*H*W for reduction=Mean.
    "nll_loss2d_backward": OpSpec(
        build=lambda d: (
            [
                torch.tensor(1.0, device=d),  # grad_output
                f(d, 2, 5, 4, 4).log_softmax(1),  # self
                torch.randint(0, 5, (2, 4, 4), device=d),  # target
                None,  # weight
                1,  # reduction (Mean)
                -100,  # ignore_index
                torch.tensor(32.0, device=d),  # total_weight
            ],
            {},
        ),
    ),
    # Schema: multi_margin_loss(self, target, p=1, margin=1, weight?, reduction=Mean).
    # weight is Tensor? — passing 0 trips the dispatcher; use None
    # to fall back to unit weights.
    "multi_margin_loss": OpSpec(
        build=lambda d: (
            [
                f(d, 4, 5),
                torch.randint(0, 5, (4,), device=d),
                1.0,
                1.0,
                None,
            ],
            {},
        ),
    ),
    "multilabel_margin_loss_forward": OpSpec(
        build=lambda d: (
            [f(d, 4, 5), torch.randint(0, 5, (4, 5), device=d), 0],
            {},
        ),
    ),
    # ---------------------------------------------------------------
    # Norm / batch / group / layer
    # ---------------------------------------------------------------
    "batch_norm": OpSpec(
        build=lambda d: (
            [
                f(d, 2, 3, 4, 4),
                f(d, 3),
                f(d, 3),
                f(d, 3),
                f(d, 3),
                True,
                0.1,
                1e-5,
                False,
            ],
            {},
        ),
    ),
    "native_batch_norm": OpSpec(
        build=lambda d: (
            [
                f(d, 2, 3, 4, 4),
                f(d, 3),
                f(d, 3),
                f(d, 3),
                f(d, 3),
                True,
                0.1,
                1e-5,
            ],
            {},
        ),
    ),
    "miopen_batch_norm": OpSpec(
        build=lambda d: (
            [
                f(d, 2, 3, 4, 4),
                torch.ones(3, device=d),
                torch.zeros(3, device=d),
                torch.ones(3, device=d),
                torch.zeros(3, device=d),
                True,
                0.1,
                1e-5,
            ],
            {},
        ),
    ),
    # Schema: (input, grad_output, weight, running_mean?, running_var?,
    #          save_mean?, save_var?, epsilon). grad_output must match input
    #          shape (not a 1-D tensor).
    "miopen_batch_norm_backward": OpSpec(
        build=lambda d: (
            [
                f(d, 2, 3, 4, 4),  # input
                f(d, 2, 3, 4, 4),  # grad_output
                torch.ones(3, device=d),  # weight
                torch.zeros(3, device=d),  # running_mean
                torch.ones(3, device=d),  # running_var
                torch.zeros(3, device=d),  # save_mean
                torch.ones(3, device=d),  # save_var
                1e-5,  # epsilon
            ],
            {},
        ),
    ),
    "_batch_norm_with_update": OpSpec(
        build=lambda d: (
            [
                f(d, 2, 3, 4, 4),
                torch.ones(3, device=d),
                torch.zeros(3, device=d),
                torch.ones(3, device=d),
                torch.zeros(3, device=d),
                0.1,
                1e-5,
            ],
            {},
        ),
    ),
    "native_group_norm": OpSpec(
        build=lambda d: (
            [
                f(d, 2, 6, 4, 4),
                torch.ones(6, device=d),
                torch.zeros(6, device=d),
                2,
                6,
                16,
                2,
                1e-5,
            ],
            {},
        ),
    ),
    "native_layer_norm": OpSpec(
        build=lambda d: (
            [
                f(d, 2, 4, 4),
                [4],
                torch.ones(4, device=d),
                torch.zeros(4, device=d),
                1e-5,
            ],
            {},
        ),
    ),
    "_fused_rms_norm": OpSpec(
        build=lambda d: ([f(d, 2, 4, 4), [4], torch.ones(4, device=d), 1e-5], {}),
    ),
    # Schema: (grad_out, input, normalized_shape, rstd, weight?, bool[2] output_mask).
    "_fused_rms_norm_backward": OpSpec(
        build=lambda d: (
            [
                f(d, 2, 4, 4),  # grad_out
                f(d, 2, 4, 4),  # input
                [4],  # normalized_shape
                torch.ones(2, 4, 1, device=d),  # rstd
                torch.ones(4, device=d),  # weight
                [True, True],  # output_mask
            ],
            {},
        ),
    ),
    "batch_norm_backward": OpSpec(build=batch_norm_backward_args),
    "native_batch_norm_backward": OpSpec(build=native_batch_norm_backward_args),
    "native_group_norm_backward": OpSpec(build=native_group_norm_backward_args),
    "native_layer_norm_backward": OpSpec(build=native_layer_norm_backward_args),
    # ---------------------------------------------------------------
    # Pooling + unpooling (strict shape requirements)
    # ---------------------------------------------------------------
    "_adaptive_avg_pool2d": OpSpec(
        build=lambda d: ([f(d, 1, 3, 8, 8), [4, 4]], {}),
    ),
    "_adaptive_avg_pool2d_backward": OpSpec(
        build=lambda d: ([f(d, 1, 3, 4, 4), f(d, 1, 3, 8, 8)], {}),
    ),
    "_adaptive_avg_pool3d": OpSpec(
        build=lambda d: ([f(d, 1, 1, 8, 8, 8), [4, 4, 4]], {}),
    ),
    "_adaptive_avg_pool3d_backward": OpSpec(
        build=lambda d: ([f(d, 1, 1, 4, 4, 4), f(d, 1, 1, 8, 8, 8)], {}),
    ),
    "adaptive_max_pool2d": OpSpec(build=lambda d: ([f(d, 1, 3, 8, 8), [2, 2]], {})),
    # adaptive_max_pool2d_backward: grad_output and indices must match the
    # pool-output shape (N, C, out_h, out_w); self carries the input shape.
    "adaptive_max_pool2d_backward": OpSpec(
        build=lambda d: (
            [f(d, 1, 3, 2, 2), f(d, 1, 3, 8, 8), i(d, 1, 3, 2, 2)],
            {},
        ),
    ),
    "adaptive_max_pool3d": OpSpec(
        build=lambda d: ([f(d, 1, 2, 6, 6, 6), [2, 2, 2]], {}),
    ),
    "adaptive_max_pool3d_backward": OpSpec(
        build=lambda d: (
            [f(d, 1, 2, 2, 2, 2), f(d, 1, 2, 6, 6, 6), i(d, 1, 2, 2, 2, 2)],
            {},
        ),
    ),
    "avg_pool2d": OpSpec(build=lambda d: ([f(d, 1, 1, 8, 8), [2, 2]], {})),
    # Schema takes 8 positionals; divisor_override is int? and must be
    # supplied (as None) because it is positional, not kwarg-only.
    "avg_pool2d_backward": OpSpec(
        build=lambda d: (
            [
                f(d, 1, 1, 4, 4),
                f(d, 1, 1, 8, 8),
                [2, 2],
                [2, 2],
                [0, 0],
                True,
                True,
                None,
            ],
            {},
        ),
    ),
    "avg_pool3d": OpSpec(build=lambda d: ([f(d, 1, 1, 8, 8, 8), [2, 2, 2]], {})),
    "avg_pool3d_backward": OpSpec(
        build=lambda d: (
            [
                f(d, 1, 1, 4, 4, 4),
                f(d, 1, 1, 8, 8, 8),
                [2, 2, 2],
                [2, 2, 2],
                [0, 0, 0],
                False,
                True,
                None,
            ],
            {},
        ),
    ),
    # Schema: (self, kernel_size, output_size, random_samples[N,C,2]).
    "fractional_max_pool2d": OpSpec(
        build=lambda d: (
            [
                f(d, 1, 3, 8, 8),
                [2, 2],
                [4, 4],
                torch.rand(1, 3, 2, device=d),
            ],
            {},
        ),
    ),
    # Schema: (self, kernel_size, stride, padding, dilation, ceil_mode).
    # Stride must be > 0, and backward takes (grad_output, self, kernel, stride,
    # padding, dilation, ceil_mode, indices) — grad_output / indices shapes
    # match the pool output, not the input.
    "max_pool2d_with_indices": OpSpec(
        build=lambda d: (
            [f(d, 1, 1, 8, 8), [2, 2], [2, 2], [0, 0], [1, 1], False],
            {},
        ),
    ),
    "max_pool2d_with_indices_backward": OpSpec(
        build=lambda d: (
            [
                f(d, 1, 1, 4, 4),  # grad_output
                f(d, 1, 1, 8, 8),  # self
                [2, 2],
                [2, 2],
                [0, 0],
                [1, 1],  # kernel, stride, padding, dilation
                False,  # ceil_mode
                i(d, 1, 1, 4, 4),  # indices
            ],
            {},
        ),
    ),
    "max_pool3d_with_indices": OpSpec(
        build=lambda d: (
            [
                f(d, 1, 1, 8, 8, 8),
                [2, 2, 2],
                [2, 2, 2],
                [0, 0, 0],
                [1, 1, 1],
                False,
            ],
            {},
        ),
    ),
    "max_pool3d_with_indices_backward": OpSpec(
        build=lambda d: (
            [
                f(d, 1, 1, 4, 4, 4),  # grad_output
                f(d, 1, 1, 8, 8, 8),  # self
                [2, 2, 2],
                [2, 2, 2],
                [0, 0, 0],
                [1, 1, 1],
                False,
                i(d, 1, 1, 4, 4, 4),  # indices
            ],
            {},
        ),
    ),
    "max_unpool2d": OpSpec(
        build=lambda d: ([f(d, 1, 1, 4, 4), i(d, 1, 1, 4, 4), [8, 8]], {}),
    ),
    "max_unpool3d": OpSpec(
        build=lambda d: (
            [
                f(d, 1, 1, 2, 2, 2),
                i(d, 1, 1, 2, 2, 2),
                [4, 4, 4],
                [2, 2, 2],
                [0, 0, 0],
            ],
            {},
        ),
    ),
    # ---------------------------------------------------------------
    # Upsample
    # ---------------------------------------------------------------
    "upsample_nearest2d": OpSpec(
        build=lambda d: ([f(d, 1, 3, 8, 8), [4, 4], None, None], {}),
    ),
    "upsample_nearest3d": OpSpec(
        build=lambda d: ([f(d, 1, 1, 4, 4, 4), [8, 8, 8]], {}),
    ),
    "upsample_nearest1d": OpSpec(build=lambda d: ([f(d, 1, 3, 8), [16], None], {})),
    # Schema: (grad_output, output_size, input_size, scales_h?, scales_w?).
    "upsample_nearest2d_backward": OpSpec(
        build=lambda d: (
            [f(d, 1, 3, 16, 16), [16, 16], [1, 3, 8, 8], None, None],
            {},
        ),
    ),
    "upsample_linear1d": OpSpec(
        build=lambda d: ([f(d, 1, 3, 8), [16], False, False], {}),
    ),
    "upsample_bilinear2d": OpSpec(
        build=lambda d: ([f(d, 1, 3, 8, 8), [16, 16], False], {}),
    ),
    "upsample_bicubic2d": OpSpec(
        build=lambda d: ([f(d, 1, 3, 8, 8), [16, 16], False], {}),
    ),
    "upsample_trilinear3d": OpSpec(
        build=lambda d: ([f(d, 1, 2, 4, 4, 4), [8, 8, 8], False], {}),
    ),
    "_upsample_nearest_exact1d": OpSpec(
        build=lambda d: ([f(d, 1, 3, 8), [16]], {}),
    ),
    "_upsample_nearest_exact2d": OpSpec(
        build=lambda d: ([f(d, 1, 3, 8, 8), [16, 16]], {}),
    ),
    "_upsample_nearest_exact2d_backward": OpSpec(
        build=lambda d: (
            [f(d, 1, 3, 16, 16), [16, 16], [1, 3, 8, 8], None, None],
            {},
        ),
    ),
    "_upsample_nearest_exact3d": OpSpec(
        build=lambda d: ([f(d, 1, 2, 4, 4, 4), [8, 8, 8]], {}),
    ),
    "_upsample_bilinear2d_aa": OpSpec(
        build=lambda d: ([f(d, 1, 3, 8, 8), [16, 16], False, None], {}),
    ),
    "_upsample_bicubic2d_aa": OpSpec(
        build=lambda d: ([f(d, 1, 3, 8, 8), [16, 16], False, None], {}),
    ),
    # Schema: (grad_output, output_size, input_size, align_corners, scales_h?,
    # scales_w?).
    "_upsample_bilinear2d_aa_backward": OpSpec(
        build=lambda d: (
            [f(d, 1, 3, 16, 16), [16, 16], [1, 3, 8, 8], False, None, None],
            {},
        ),
    ),
    # ---------------------------------------------------------------
    # Grid sampler / im2col
    # ---------------------------------------------------------------
    "grid_sampler_2d": OpSpec(
        build=lambda d: (
            [f(d, 1, 1, 8, 8), torch.zeros(1, 8, 8, 2, device=d), 0, 0, False],
            {},
        ),
    ),
    # Schema: (grad_output, input, grid, interp, padding, align_corners,
    # bool[2] output_mask).
    "grid_sampler_2d_backward": OpSpec(
        build=lambda d: (
            [
                f(d, 1, 1, 8, 8),  # grad_output
                f(d, 1, 1, 8, 8),  # input
                torch.zeros(1, 8, 8, 2, device=d),  # grid
                0,
                0,
                False,
                [True, True],  # output_mask
            ],
            {},
        ),
    ),
    "grid_sampler_3d": OpSpec(
        build=lambda d: (
            [f(d, 1, 1, 4, 4, 4), torch.zeros(1, 4, 4, 4, 3, device=d), 0, 0, False],
            {},
        ),
    ),
    "grid_sampler_3d_backward": OpSpec(
        build=lambda d: (
            [
                f(d, 1, 1, 4, 4, 4),  # grad_output
                f(d, 1, 1, 4, 4, 4),  # input
                torch.zeros(1, 4, 4, 4, 3, device=d),  # grid
                0,
                0,
                False,
                [True, True],
            ],
            {},
        ),
    ),
    # Schema: im2col(self, kernel_size, dilation, padding, stride).
    # The final argument is stride and must be > 0; swapping stride/padding
    # produces stride should be greater than zero.
    "im2col": OpSpec(
        build=lambda d: ([f(d, 1, 1, 8, 8), [3, 3], [1, 1], [0, 0], [1, 1]], {}),
    ),
    # Schema: col2im(self, output_size[2], kernel_size, dilation, padding, stride).
    # Input (1, C*kH*kW, L) unfolds back to (N, C, H, W) with
    # output_size=[H, W] — must be a 2-element list, not 4.
    "col2im": OpSpec(
        build=lambda d: (
            [f(d, 1, 4, 9), [4, 4], [2, 2], [1, 1], [0, 0], [1, 1]],
            {},
        ),
    ),
    # ---------------------------------------------------------------
    # Padding
    # ---------------------------------------------------------------
    "reflection_pad1d_backward": OpSpec(
        build=lambda d: ([f(d, 1, 1, 8), f(d, 1, 1, 6), [1, 1]], {}),
    ),
    "reflection_pad2d": OpSpec(
        build=lambda d: ([f(d, 1, 1, 6, 6), [1, 1, 1, 1]], {}),
    ),
    "reflection_pad2d_backward": OpSpec(
        build=lambda d: ([f(d, 1, 1, 8, 8), f(d, 1, 1, 6, 6), [1, 1, 1, 1]], {}),
    ),
    "reflection_pad3d": OpSpec(
        build=lambda d: ([f(d, 1, 1, 4, 4, 4), [1, 1, 1, 1, 1, 1]], {}),
    ),
    "reflection_pad3d_backward": OpSpec(
        build=lambda d: (
            [f(d, 2, 3, 8, 8, 8), f(d, 2, 3, 6, 6, 6), [1, 1, 1, 1, 1, 1]],
            {},
        ),
    ),
    "replication_pad1d_backward": OpSpec(
        build=lambda d: ([f(d, 1, 1, 8), f(d, 1, 1, 6), [1, 1]], {}),
    ),
    "replication_pad2d": OpSpec(
        build=lambda d: ([f(d, 1, 1, 6, 6), [1, 1, 1, 1]], {}),
    ),
    "replication_pad2d_backward": OpSpec(
        build=lambda d: ([f(d, 1, 1, 8, 8), f(d, 1, 1, 6, 6), [1, 1, 1, 1]], {}),
    ),
    "replication_pad3d": OpSpec(
        build=lambda d: ([f(d, 1, 1, 4, 4, 4), [1, 1, 1, 1, 1, 1]], {}),
    ),
    "replication_pad3d_backward": OpSpec(
        build=lambda d: (
            [f(d, 1, 1, 6, 6, 6), f(d, 1, 1, 4, 4, 4), [1, 1, 1, 1, 1, 1]],
            {},
        ),
    ),
    # ---------------------------------------------------------------
    # Integer / bitwise (CUDA rejects float)
    # ---------------------------------------------------------------
    "__ilshift__": OpSpec(build=lambda d: ([i(d, 4, 4), i(d, 4, 4, high=4)], {})),
    "__irshift__": OpSpec(build=lambda d: ([i(d, 4, 4), i(d, 4, 4)], {})),
    "__rshift__": OpSpec(build=lambda d: ([i(d, 4, 4), i(d, 4, 4)], {})),
    "__lshift__": OpSpec(build=lambda d: ([i(d, 4, 4), i(d, 4, 4)], {})),
    "bitwise_and": OpSpec(build=lambda d: ([i(d, 4, 4), i(d, 4, 4)], {})),
    "bitwise_and_": OpSpec(build=lambda d: ([i(d, 4, 4), i(d, 4, 4)], {})),
    "bitwise_or": OpSpec(build=lambda d: ([i(d, 4, 4), i(d, 4, 4)], {})),
    "bitwise_or_": OpSpec(build=lambda d: ([i(d, 4, 4), i(d, 4, 4)], {})),
    "bitwise_xor": OpSpec(build=lambda d: ([i(d, 4, 4), i(d, 4, 4)], {})),
    "bitwise_xor_": OpSpec(build=lambda d: ([i(d, 4, 4), i(d, 4, 4)], {})),
    "bitwise_left_shift": OpSpec(build=lambda d: ([i(d, 4, 4), i(d, 4, 4)], {})),
    "bitwise_right_shift": OpSpec(build=lambda d: ([i(d, 4, 4), i(d, 4, 4)], {})),
    "bitwise_not": OpSpec(
        build=lambda d: (
            [torch.randint(0, 256, (4, 4), device=d, dtype=torch.int64)],
            {},
        ),
    ),
    "bitwise_not_": OpSpec(build=lambda d: ([i(d, 4, 4)], {})),
    "gcd": OpSpec(build=lambda d: ([i(d, 4, 4), i(d, 4, 4)], {})),
    "gcd_": OpSpec(build=lambda d: ([i(d, 4, 4), i(d, 4, 4)], {})),
    "lcm": OpSpec(
        build=lambda d: (
            [
                torch.randint(1, 100, (4, 4), device=d, dtype=torch.int64),
                torch.randint(1, 100, (4, 4), device=d, dtype=torch.int64),
            ],
            {},
        ),
    ),
    "lcm_": OpSpec(build=lambda d: ([i(d, 4, 4), i(d, 4, 4)], {})),
    # ---------------------------------------------------------------
    # Random distributions
    # ---------------------------------------------------------------
    "one_hot": OpSpec(
        build=lambda d: ([torch.randint(0, 5, (4,), device=d)], {}),
    ),
    # Poisson expects non-negative rates; randn can trigger ROCm HSA
    # faults. The emitted workload must use rand * scale.
    "poisson": OpSpec(
        build=lambda _d: ([CoverageTensorArg((4, 4), "rand_uniform", 4.0), None], {}),
    ),
    # bernoulli_ expects probabilities in [0, 1]; see note above.
    "bernoulli_": OpSpec(
        build=lambda d: ([f(d, 4, 4), CoverageTensorArg((4, 4), "rand")], {}),
    ),
    # multinomial runs fine standalone but SIGSEGVs under
    # torch.profiler(CPU+CUDA) on ROCm (renormRowsL1 kernel path
    # interacting with ROCTracer). Confirmed by scripts/torch_trace_coverage_
    # scan_ops.py with --workers=1. Revisit after driver/profiler updates.
    "multinomial": OpSpec(
        skip="torch.profiler cannot collect ground truth: ROCm ROCTracer + "
        "renormRowsL1 kernel race produces SIGSEGV inside the profiler "
        "subprocess (op itself runs fine outside the profiler).",
    ),
    "geometric_": OpSpec(build=lambda d: ([f(d, 4, 4), 0.5], {})),
    "repeat_interleave": OpSpec(build=lambda d: ([i1(d, 16)], {})),
    "bincount": OpSpec(build=lambda d: ([i1(d, 32).clamp_min(0)], {})),
    # ---------------------------------------------------------------
    # Masks / clamp
    # ---------------------------------------------------------------
    "clamp": OpSpec(build=lambda d: ([f(d, 4, 4), 0.0, 1.0], {})),
    "clamp_": OpSpec(build=lambda d: ([f(d, 4, 4), 0.0, 1.0], {})),
    "masked_fill_": OpSpec(build=lambda d: ([f(d, 4, 4), b(d), 0.0], {})),
    "masked_scatter_": OpSpec(
        build=lambda d: ([f(d, 4, 4), b(d), f(d, 4, 4)], {}),
    ),
    "masked_select": OpSpec(build=lambda d: ([f(d, 4, 4), b(d)], {})),
    "_masked_scale": OpSpec(
        build=lambda d: (
            [
                f(d, 4, 4),
                torch.randint(0, 2, (4, 4), device=d, dtype=torch.uint8),
                1.0,
            ],
            {},
        ),
    ),
    # Schema: native_dropout_backward(Tensor grad_output, Tensor mask, float scale).
    # Passing a second float tensor between grad_output and mask produces
    # "expected at most 3 argument(s) but received 4".
    "native_dropout_backward": OpSpec(
        build=lambda d: ([f(d, 4, 4), b(d), 0.5], {}),
    ),
    # ---------------------------------------------------------------
    # Indexing / scatter / gather
    # ---------------------------------------------------------------
    # aten.index.Tensor needs a real Tensor?[], not None.
    "index": OpSpec(
        build=lambda d: (
            [
                f(d, 4, 4),
                [None, torch.randint(0, 4, (2, 2), device=d, dtype=torch.int64)],
            ],
            {},
        ),
    ),
    # List[int] defaults like [1, 1] duplicate a dim on 2D tensors.
    "flip": OpSpec(build=lambda d: ([f(d, 4, 4), [0]], {})),
    "index_copy": OpSpec(
        build=lambda d: ([f(d, 4, 4), 0, i1(d, 2), f(d, 2, 4)], {}),
    ),
    "index_copy_": OpSpec(
        build=lambda d: ([f(d, 4, 4), 0, i1(d, 2), f(d, 2, 4)], {}),
    ),
    "index_add": OpSpec(
        build=lambda d: ([f(d, 4, 4), 0, i1(d, 2), f(d, 2, 4)], {}),
    ),
    "index_add_": OpSpec(
        build=lambda d: ([f(d, 4, 4), 0, i1(d, 2), f(d, 2, 4)], {}),
    ),
    # include_self is kwarg-only; passing as 6th positional trips the
    # dispatcher with "takes 5 positional argument(s) but 6 was/were given".
    "index_reduce": OpSpec(
        build=lambda d: (
            [f(d, 4, 4), 0, i1(d, 4), f(d, 4, 4), "prod"],
            {"include_self": False},
        ),
    ),
    "index_reduce_": OpSpec(
        build=lambda d: (
            [f(d, 4, 4), 0, i1(d, 4), f(d, 4, 4), "prod"],
            {"include_self": False},
        ),
    ),
    "index_fill_": OpSpec(
        build=lambda d: ([f(d, 4, 4), 0, i1(d, 2), torch.tensor(0.0, device=d)], {}),
    ),
    "index_select": OpSpec(build=lambda d: ([f(d, 4, 4), 0, i1(d, 2)], {})),
    # gather needs index to be int64; schema fallback gives float.
    "gather": OpSpec(
        build=lambda d: ([f(d, 4, 4), 0, i(d, 4, 4, low=0, high=4)], {}),
    ),
    "scatter": OpSpec(
        build=lambda d: ([f(d, 4, 4), 0, i(d, 4, 4), f(d, 4, 4)], {}),
    ),
    "scatter_": OpSpec(
        build=lambda d: ([f(d, 4, 4), 0, i(d, 4, 4), f(d, 4, 4)], {}),
    ),
    "scatter_add": OpSpec(
        build=lambda d: ([f(d, 4, 4), 0, i(d, 4, 4), f(d, 4, 4)], {}),
    ),
    "scatter_add_": OpSpec(
        build=lambda d: ([f(d, 4, 4), 0, i(d, 4, 4), f(d, 4, 4)], {}),
    ),
    # include_self is kwarg-only on both the functional and in-place form.
    "scatter_reduce": OpSpec(
        build=lambda d: (
            [f(d, 4, 4), 0, i(d, 4, 4), f(d, 4, 4), "sum"],
            {"include_self": False},
        ),
    ),
    "scatter_reduce_": OpSpec(
        build=lambda d: (
            [f(d, 4, 4), 0, i(d, 4, 4), f(d, 4, 4), "sum"],
            {"include_self": False},
        ),
    ),
    "take": OpSpec(build=lambda d: ([f(d, 4, 4), i1(d, 8)], {})),
    # segment_reduce: most arguments (lengths, indices,
    # offsets, axis, unsafe, initial) are kwarg-only. The
    # kernel normally checks lengths.sum() == data.size(axis), but the
    # emitter replaces tensor literals with random torch.randint
    # values; unsafe=True bypasses the check so the reduction kernel
    # still launches.
    "segment_reduce": OpSpec(
        build=lambda d: (
            [f(d, 8), "sum"],
            {
                "lengths": torch.tensor([4, 4], device=d, dtype=torch.int64),
                "unsafe": True,
            },
        ),
    ),
    # _segment_reduce_backward: (grad, output, data, reduce, *, lengths?,
    # offsets?, axis=0, initial?).
    "_segment_reduce_backward": OpSpec(
        build=lambda d: (
            [
                torch.randn(2, device=d),  # grad (output shape)
                torch.randn(2, device=d),  # output
                torch.randn(8, device=d),  # data
                "sum",  # reduce
            ],
            {"lengths": torch.tensor([4, 4], device=d, dtype=torch.int64)},
        ),
    ),
    # ---------------------------------------------------------------
    # Misc shape / dtype / layout
    # ---------------------------------------------------------------
    "_chunk_cat": OpSpec(
        build=lambda d: ([[f(d, 2, 4), f(d, 2, 4)], 0, 2], {}),
    ),
    # Schema takes 4 positionals; compute_mode is int? but positional.
    "_cdist_forward": OpSpec(
        build=lambda d: ([f(d, 2, 4, 8), f(d, 2, 5, 8), 2.0, None], {}),
    ),
    "_thnn_fused_lstm_cell": OpSpec(
        build=lambda d: (
            [f(d, 2, 16), f(d, 2, 16), f(d, 2, 4), None, None],
            {},
        ),
    ),
    "channel_shuffle": OpSpec(build=lambda d: ([f(d, 1, 8, 4, 4), 2], {})),
    "bucketize": OpSpec(build=lambda d: ([f(d, 8), f(d, 5)], {})),
    "roll": OpSpec(build=lambda d: ([f(d, 4, 4), [1], [0]], {})),
    "unfold": OpSpec(build=lambda d: ([f(d, 2, 8), 1, 4, 2], {})),
    # Schema: unfold_backward(grad_in, SymInt[] input_sizes, dim, size, step).
    # input_sizes is a list of ints, not a tensor.
    "unfold_backward": OpSpec(
        build=lambda d: ([f(d, 2, 4, 4), [2, 10], 1, 4, 2], {}),
    ),
    "view_as_complex": OpSpec(build=lambda d: ([f(d, 4, 2)], {})),
    "view_as_real": OpSpec(
        build=lambda d: ([torch.randn(4, 2, device=d, dtype=torch.complex64)], {}),
    ),
    # Schema: view(self, SymInt[] size).  Passing 8, 2 as two ints
    # does not match the List[int] slot.
    "view": OpSpec(build=lambda d: ([f(d, 4, 4), [8, 2]], {})),
    # Schema: glu_backward(grad_output, self, dim).  grad_output matches the
    # glu output shape (self halved along dim); don't swap the two.
    "glu_backward": OpSpec(build=lambda d: ([f(d, 2, 4), f(d, 2, 8), 1], {})),
    # Quantized / packed matmul (working subset; layout-specific ones are
    # in the skip block above).
    "_int_mm": OpSpec(
        build=lambda d: (
            [
                torch.randint(-2, 2, (32, 32), device=d, dtype=torch.int8),
                torch.randint(-2, 2, (32, 32), device=d, dtype=torch.int8),
            ],
            {},
        ),
    ),
    # _convert_weight_to_int4pack expects uint8 weight shaped [N][K/2] and
    # legal innerKTiles values; use a shape/value pair that satisfies those
    # constraints and reaches dispatch.
    "_convert_weight_to_int4pack": OpSpec(
        build=lambda d: (
            [torch.randint(0, 255, (128, 64), device=d, dtype=torch.uint8), 8],
            {},
        ),
    ),
    # ---------------------------------------------------------------
    # Softmax backward (mixed-dtype accumulation)
    # ---------------------------------------------------------------
    # ROCm/CPU enforce grad.scalar_type() == input_dtype OR
    # (grad==Float && input_dtype==Half). With grad and output already
    # in fp16, the input_dtype param must also be fp16 (passing fp32
    # there hits the dtype-mismatch check before any kernel launches).
    "_log_softmax_backward_data": OpSpec(
        build=lambda d: (
            [
                f(d, 2, 8, dtype=torch.float16),
                f(d, 2, 8, dtype=torch.float16),
                1,
                torch.float16,
            ],
            {},
        ),
    ),
    "_softmax_backward_data": OpSpec(
        build=lambda d: (
            [
                f(d, 2, 8, dtype=torch.float16),
                f(d, 2, 8, dtype=torch.float16),
                1,
                torch.float16,
            ],
            {},
        ),
    ),
    # ---------------------------------------------------------------
    # Misc / AMP / assertions
    # ---------------------------------------------------------------
    # Schema: (Tensor[] self, Tensor found_inf, Tensor inv_scale) -> ().
    # Omitting inv_scale fails the dispatcher.
    "_amp_foreach_non_finite_check_and_unscale_": OpSpec(
        build=lambda d: (
            [
                [f(d, 4), f(d, 4)],
                torch.zeros(1, device=d),  # found_inf
                torch.ones(1, device=d),  # inv_scale
            ],
            {},
        ),
    ),
    # _assert_async traps (SIGABRT) on any False element, so the input must
    # be all-True; default randint(bool) would abort the ground-truth subprocess.
    "_assert_async": OpSpec(
        build=lambda _d: (
            [CoverageTensorArg(shape=(), emit="bool_all_true")],
            {},
        ),
    ),
    # nonzero_static: size and fill_value are kwarg-only; the schema
    # fallback can't synthesise the kwargs, so hand them in explicitly.
    "nonzero_static": OpSpec(
        build=lambda d: ([f(d, 4, 4)], {"size": 8}),
    ),
    # ---------------------------------------------------------------
    # Convolution backward (needs explicit output_sizes + masks)
    # ---------------------------------------------------------------
    "convolution_backward": OpSpec(build=convolution_backward_args),
    # ---------------------------------------------------------------
    # Fused optimizers (list-of-tensors + scalar state_steps)
    # ---------------------------------------------------------------
    "_fused_adam_": OpSpec(build=fused_adam_args),
    "_fused_adamw_": OpSpec(build=fused_adam_args),
    # Backend-gated families (flash-attn, fp8 gemm, 2:4 sparse, grouped mm,
    # nested-tensor): need dedicated workloads, not schema-driven args.
    "_scaled_dot_product_cudnn_attention": OpSpec(skip="cuDNN-only."),
    "_scaled_dot_product_cudnn_attention_backward": OpSpec(skip="cuDNN-only."),
    "_scaled_dot_product_efficient_attention": OpSpec(skip="xFormers CUDA-only."),
    "_scaled_dot_product_efficient_attention_backward": OpSpec(
        skip="xFormers CUDA-only."
    ),
    "_scaled_dot_product_flash_attention": OpSpec(
        skip="FlashAttention forward; schema-driven synthesis raises IndexError "
        "from broadcast/mask dims, needs real 4D QKV + optional mask.",
    ),
    "_scaled_dot_product_flash_attention_backward": OpSpec(
        skip="FlashAttention backward; requires matching fwd output + softmax_lse.",
    ),
    "_flash_attention_forward": OpSpec(
        skip="FlashAttention forward; requires bf16/fp16 QKV with specific layout.",
    ),
    "_flash_attention_backward": OpSpec(
        skip="FlashAttention backward; requires matching fwd output + softmax_lse.",
    ),
    "_efficient_attention_forward": OpSpec(skip="xFormers CUDA-only kernel."),
    "_efficient_attention_backward": OpSpec(skip="xFormers CUDA-only kernel."),
    "_scaled_mm": OpSpec(
        skip="fp8 gemm; requires fp8 tensors + scale tensors on supported hardware.",
    ),
    "_scaled_mm_v2": OpSpec(
        skip="fp8 gemm v2; requires fp8 tensors + scale tensors on supported hardware.",
    ),
    "_grouped_mm": OpSpec(
        skip="grouped matmul; requires per-group offset metadata on supported hw.",
    ),
    "_cslt_sparse_mm": OpSpec(skip="cuSPARSELt; no ROCm equivalent."),
    "_sparse_semi_structured_linear": OpSpec(skip="CUTLASS 2:4 sparse; NVIDIA-only."),
    "_sparse_semi_structured_addmm": OpSpec(skip="CUTLASS 2:4 sparse; NVIDIA-only."),
    "_nested_view_from_buffer": OpSpec(
        skip="Nested-tensor constructor; requires matching offsets/lengths buffers.",
    ),
}


# -----------------------------------------------------------------------------
# Bulk-registered builders
# -----------------------------------------------------------------------------
# Register repetitive short-name builders in bulk to avoid OP_SPECS bloat.
# dict.setdefault preserves explicit per-op overrides in the main table.
# Buckets that need custom argument design remain explicitly listed there.


def _register_bulk_unary_float_builders() -> None:
    """Register short-name builders for unary float Tensor->Tensor ops
    (single 8x8 float32 input, no kwargs). Output dtype follows input
    so .out variants are covered too.
    """

    def template(d: str) -> Tuple[List[Any], Dict[str, Any]]:
        return ([f(d, 8, 8)], {})

    # fmt: off
    bulk_short_names = (
        # trig / inverse trig / hyperbolic
        "acos", "acosh", "asin", "asinh", "atan", "atanh",
        "cos", "cosh", "sin", "sinh", "tan", "tanh", "sinc",
        # exponentials / logarithms / power
        "exp", "exp2", "expm1", "log", "log10", "log1p", "log2",
        "logit",
        "reciprocal", "rsqrt", "sqrt", "square",
        # rounding / sign / fractional
        "ceil", "floor", "trunc", "round", "neg", "frac", "sgn", "sign",
        "positive",
        # gamma / Bessel-ish
        "digamma", "erf", "erfc", "erfinv", "i0", "lgamma",
        # complex / angle helpers
        "angle", "conj_physical",
        # angle conversion
        "deg2rad", "rad2deg",
        # activations
        "elu", "gelu", "hardshrink", "hardsigmoid", "hardswish",
        "hardtanh", "leaky_relu", "mish", "relu", "sigmoid", "silu",
        "softplus", "softshrink",
    )
    # fmt: on
    for name in bulk_short_names:
        OP_SPECS.setdefault(name, OpSpec(build=template))


def _register_bulk_whole_tensor_reduction_builders() -> None:
    """Register builders for ops that reduce over every element.

    Single 8x8 float32 input; the dispatcher routes to the no-dim
    default overload. `.dim` / `.dim_IntList` variants of these names
    still need explicit entries with a `dim` arg (see the
    "reduction along dim" bucket).
    """

    def template(d: str) -> Tuple[List[Any], Dict[str, Any]]:
        return ([f(d, 8, 8)], {})

    bulk_short_names = (
        "all",
        "any",
        "amax",
        "amin",
        "aminmax",
        "max",
        "min",
        "median",
        "mode",
        "nanmedian",
        "prod",
        "sum",
        "mean",
        "nansum",
        "trace",
        "std",
        "var",
        "std_mean",
        "var_mean",
    )
    for name in bulk_short_names:
        OP_SPECS.setdefault(name, OpSpec(build=template))


def _register_bulk_unary_special_builders() -> None:
    """Register builders for unary special_* functions (rand-uniform
    in (0, 2)). Binary polynomial families need per-op entries.
    """

    def template(d: str) -> Tuple[List[Any], Dict[str, Any]]:
        return ([torch.rand(8, 8, device=d) * 2.0], {})

    bulk_short_names = (
        "special_airy_ai",
        "special_bessel_j0",
        "special_bessel_j1",
        "special_bessel_y0",
        "special_bessel_y1",
        "special_modified_bessel_i0",
        "special_modified_bessel_i1",
        "special_modified_bessel_k0",
        "special_modified_bessel_k1",
        "special_scaled_modified_bessel_k0",
        "special_scaled_modified_bessel_k1",
        "special_spherical_bessel_j0",
        "special_entr",
        "special_erfcx",
        "special_ndtri",
        "special_i0e",
        "special_i1e",
    )
    for name in bulk_short_names:
        OP_SPECS.setdefault(name, OpSpec(build=template))


def _register_bulk_binary_tensor_elementwise_builders() -> None:
    """Register builders for binary elementwise short names (two
    same-shape 8x8 float32 tensors).
    """

    def template(d: str) -> Tuple[List[Any], Dict[str, Any]]:
        return ([f(d, 8, 8), f(d, 8, 8)], {})

    bulk_short_names = (
        "add",
        "sub",
        "mul",
        "div",
        "maximum",
        "minimum",
        "fmax",
        "fmin",
        "copysign",
        "heaviside",
        "hypot",
        "nextafter",
        "logaddexp",
        "logaddexp2",
        "floor_divide",
        "xlogy",
        "atan2",
        "clamp_max",
        "clamp_min",
    )
    for name in bulk_short_names:
        OP_SPECS.setdefault(name, OpSpec(build=template))


def _register_high_impact_individual_builders() -> None:
    """Per-op short-name builders for high-impact gaps. Each entry is
    one builder; see the inline comment for the schema.
    """
    individual = {
        # (input, value, value) — fused multiply-add variants
        "addcmul": OpSpec(
            build=lambda d: ([f(d, 4, 4), f(d, 4, 4), f(d, 4, 4)], {}),
        ),
        "addcdiv": OpSpec(
            build=lambda d: ([f(d, 4, 4), f(d, 4, 4), f(d, 4, 4)], {}),
        ),
        # matmul: 2D x 2D
        "mm": OpSpec(build=lambda d: ([f(d, 4, 4), f(d, 4, 4)], {})),
        # cat: list of tensors + dim
        "cat": OpSpec(
            build=lambda d: ([[f(d, 2, 4), f(d, 2, 4)], 0], {}),
        ),
        # where.self: (condition Bool, self, other) - same shapes
        "where": OpSpec(
            build=lambda d: (
                [
                    torch.zeros(4, 4, dtype=torch.bool, device=d),
                    f(d, 4, 4),
                    f(d, 4, 4),
                ],
                {},
            ),
        ),
        # softmax / log_softmax forward — (input, dim, half_to_float=False)
        "_softmax": OpSpec(
            build=lambda d: ([f(d, 4, 8), -1, False], {}),
        ),
        "_log_softmax": OpSpec(
            build=lambda d: ([f(d, 4, 8), -1, False], {}),
        ),
        # dim-reduction helpers — (input, dim, keepdim?)
        "all.dim": OpSpec(build=lambda d: ([f(d, 4, 4), -1], {})),
        "any.dim": OpSpec(build=lambda d: ([f(d, 4, 4), -1], {})),
        "max.dim": OpSpec(build=lambda d: ([f(d, 4, 4), -1], {})),
        "min.dim": OpSpec(build=lambda d: ([f(d, 4, 4), -1], {})),
        "mean.dim": OpSpec(build=lambda d: ([f(d, 4, 4), [-1]], {})),
        "prod.dim_int": OpSpec(
            build=lambda d: ([f(d, 4, 4), -1], {}),
        ),
        "count_nonzero.dim_IntList": OpSpec(
            build=lambda d: ([f(d, 4, 4), [-1]], {}),
        ),
        # cumulative reductions
        "cumsum": OpSpec(build=lambda d: ([f(d, 4, 4), -1], {})),
        "cumprod": OpSpec(build=lambda d: ([f(d, 4, 4), -1], {})),
        # losses: (input, target, reduction='mean')
        "mse_loss": OpSpec(
            build=lambda d: ([f(d, 4, 4), f(d, 4, 4)], {}),
        ),
        "smooth_l1_loss": OpSpec(
            build=lambda d: ([f(d, 4, 4), f(d, 4, 4)], {}),
        ),
        "huber_loss": OpSpec(
            build=lambda d: ([f(d, 4, 4), f(d, 4, 4)], {}),
        ),
        # padding (1d): (input, pad)
        "reflection_pad1d": OpSpec(
            build=lambda d: ([f(d, 1, 4, 8), [1, 1]], {}),
        ),
        "replication_pad1d": OpSpec(
            build=lambda d: ([f(d, 1, 4, 8), [1, 1]], {}),
        ),
        # threshold: (self, threshold, value)
        "threshold": OpSpec(
            build=lambda d: ([f(d, 4, 4), 0.0, 0.0], {}),
        ),
        # histc: (input, bins, min, max)
        "histc": OpSpec(
            build=lambda d: ([f(d, 64), 10, -3.0, 3.0], {}),
        ),
        # native_dropout: (input, p, train) → (Tensor, Bool mask)
        "native_dropout": OpSpec(
            build=lambda d: ([f(d, 4, 4), 0.5, True], {}),
        ),
        # tril / triu / *_indices: (n, k=0)
        "tril": OpSpec(build=lambda d: ([f(d, 4, 4)], {})),
        "triu": OpSpec(build=lambda d: ([f(d, 4, 4)], {})),
        "tril_indices": OpSpec(build=lambda _d: ([4, 4], {})),
        "triu_indices": OpSpec(build=lambda _d: ([4, 4], {})),
        # comparison with explicit Scalar overload short names
        "eq": OpSpec(build=lambda d: ([f(d, 4, 4), f(d, 4, 4)], {})),
        "ne": OpSpec(build=lambda d: ([f(d, 4, 4), f(d, 4, 4)], {})),
        "lt": OpSpec(build=lambda d: ([f(d, 4, 4), f(d, 4, 4)], {})),
        "gt": OpSpec(build=lambda d: ([f(d, 4, 4), f(d, 4, 4)], {})),
        "ge": OpSpec(build=lambda d: ([f(d, 4, 4), f(d, 4, 4)], {})),
        "le": OpSpec(build=lambda d: ([f(d, 4, 4), f(d, 4, 4)], {})),
        # isin: needs (elements, test_elements). Short name "isin"
        # covers both .Tensor_Tensor and .Tensor_Scalar overloads.
        "isin": OpSpec(
            build=lambda d: (
                [
                    torch.tensor(
                        [0.0, 1.0, 2.0, 3.0],
                        device=d,
                        dtype=torch.float32,
                    ),
                    torch.tensor(
                        [1.0, 3.0],
                        device=d,
                        dtype=torch.float32,
                    ),
                ],
                {},
            ),
        ),
        # logical_*: bool inputs
        "logical_and": OpSpec(
            build=lambda d: (
                [b(d, (4, 4)), b(d, (4, 4))],
                {},
            ),
        ),
        "logical_or": OpSpec(
            build=lambda d: (
                [b(d, (4, 4)), b(d, (4, 4))],
                {},
            ),
        ),
        "logical_xor": OpSpec(
            build=lambda d: (
                [b(d, (4, 4)), b(d, (4, 4))],
                {},
            ),
        ),
        "logical_not": OpSpec(
            build=lambda d: ([b(d, (4, 4))], {}),
        ),
        # argmax / argmin: (input, dim?, keepdim?)
        "argmax": OpSpec(
            build=lambda d: ([f(d, 4, 4)], {"dim": -1}),
        ),
        "argmin": OpSpec(
            build=lambda d: ([f(d, 4, 4)], {"dim": -1}),
        ),
        # topk: (input, k)
        "topk": OpSpec(build=lambda d: ([f(d, 16), 4], {})),
        # sort: (input, dim?, descending?)
        "sort": OpSpec(build=lambda d: ([f(d, 16)], {})),
        # nonzero: (input,) → indices
        "nonzero": OpSpec(build=lambda d: ([f(d, 4, 4)], {})),
        # unique_consecutive / unique_dim
        "unique_consecutive": OpSpec(
            build=lambda d: (
                [torch.tensor([1, 1, 2, 2, 3, 1], device=d)],
                {},
            ),
        ),
        # renorm: (input, p, dim, maxnorm)
        "renorm": OpSpec(
            build=lambda d: ([f(d, 4, 4), 2.0, 0, 5.0], {}),
        ),
    }
    for name, spec in individual.items():
        OP_SPECS.setdefault(name, spec)


_register_bulk_unary_float_builders()
_register_bulk_whole_tensor_reduction_builders()
_register_bulk_unary_special_builders()
_register_bulk_binary_tensor_elementwise_builders()
_register_high_impact_individual_builders()


# -----------------------------------------------------------------------------
# Zero-copy view ops that intentionally launch no GPU kernels
# -----------------------------------------------------------------------------
#
# These ATen ops only reinterpret a tensor's metadata
# (shape / stride / dtype-view / storage offset) and return a new tensor
# that shares the underlying storage. They issue no HIP/CUDA kernel
# launches by design, so a correct ROCTX trace MUST show the marker
# present with an empty kernel-correlation set.
#
# torch.profiler's ground truth is misleading for these ops: it
# attributes any GPU kernel whose dispatch happens to overlap the op's
# CPU span to the op itself, even when that kernel was launched by an
# unrelated adjacent op (e.g. a copy_ that follows view). This is a
# documented limitation of CPU-span-based kernel attribution in
# torch.profiler and is the sole source of the historical "marker found
# but no correlated kernels" failures for this family.
#
# compare_single_op treats these ops as PASS once the ROCTX marker is
# observed and emits an explanatory note explaining (a) why the op
# launches no kernels and (b) why the empty ROCTX correlation set is
# the correct outcome.
KNOWN_KERNEL_FREE_ATEN_OPS: Set[str] = {
    "view",
    "view_as_real",
    "view_as_complex",
    "unfold",
}


# -----------------------------------------------------------------------------
# Hardware-incompatible CUDA-only ops (filtered out at discovery)
# -----------------------------------------------------------------------------
#
# A subset of ATen ops are advertised as CUDA-dispatched but require a
# strictly NVIDIA-only backend (CUTLASS, cuDNN, xFormers, cuSPARSELt,
# FlashAttention, fp8 GEMM, packed int4/int8 weights, grouped matmul,
# ...). These ops cannot run on ROCm hardware regardless of how the
# workload synthesizes arguments, so including them in the coverage
# sample only contributes noise to the SKIP count.
#
# discover_operators uses is_hardware_incompatible_cuda_only to filter
# such ops out at enumeration time. They never reach the workload, the
# ground-truth subprocess, the per-op log, or the SKIP breakdown — but
# the count is surfaced once in the session header so the exclusion is
# auditable.
#
# The keyword set is shared with categorize_skip_reason so the
# discovery-time filter and the runtime SKIP categorizer remain in
# lockstep; OpSpec.skip entries in OP_SPECS are the single source of
# truth for which ops are CUDA-only.
_HARDWARE_INCOMPATIBLE_CUDA_KEYWORDS: Tuple[str, ...] = (
    "cuda-only",
    "cuda only",
    "nvidia-only",
    "nvidia only",
    "cudnn-only",
    "cudnn only",
    "cutlass",
    "cusparselt",
    "xformers",
    "flashattention",
    "flash attention",
    "fp8 gemm",
    "fp8 tensors",
    "packed int4",
    "int8 packed",
    "grouped matmul",
    "mat2 to be transposed",
)


def is_hardware_incompatible_cuda_only(text: Optional[str]) -> bool:
    """True if text contains any CUDA-only / NVIDIA-only backend keyword.

    Used both at discovery (to drop the op from the coverage sample) and
    at runtime SKIP categorization (to bucket any reason that slipped
    through). Case-insensitive; returns False for None / empty input.
    """
    if not text:
        return False
    lowered = text.lower()
    return any(k in lowered for k in _HARDWARE_INCOMPATIBLE_CUDA_KEYWORDS)


# One row in the coverage sample. category is "aten" or "structural";
# schema is a torch FunctionSchema for ATen rows, None for structural.
class OpEntry(NamedTuple):
    name: str
    category: str
    schema: object


# status is "pass" / "skip" / "fail"; reason is empty on pass, otherwise the
# skip explanation or failure reason.
class OpCompareOutcome(NamedTuple):
    status: str
    reason: str
    log_lines: Tuple[str, ...]


# Standalone runner source: one torch.profiler window per op, dumped to JSON.
# Stored as a string so the generated workload stays free of profiler/JSON code.
COVERAGE_GROUND_TRUTH_RUNNER_SOURCE = textwrap.dedent(
    """
import importlib.util
import json
import os
import sys

# Configure synchronous kernel dispatch before the HIP/CUDA runtime is
# initialised. AMD_SERIALIZE_KERNEL is read once at context creation;
# synchronous dispatch is required for asynchronous faults to surface at
# their launch site so per-operator attribution (START / OK / FAIL banners
# and the device-liveness probe) remains correct.
os.environ.setdefault("AMD_SERIALIZE_KERNEL", "3")

import torch
from torch.profiler import profile, ProfilerActivity


def probe_device() -> None:
    \"\"\"Dispatch and synchronise a minimal CUDA kernel to surface sticky faults.

    torch.cuda.synchronize alone is not a reliable liveness check on
    ROCm: the runtime clears a sticky error on the synchronise call that
    reports it, so a subsequent synchronise with no pending work returns
    success even though the next kernel launch would fail. Launching a
    real kernel reproduces the sticky state and raises deterministically.
    \"\"\"
    t = torch.empty(1, device=\"cuda\")
    t.add_(1)
    torch.cuda.synchronize()


def main() -> None:
    workload_path, output_path = sys.argv[1], sys.argv[2]
    spec = importlib.util.spec_from_file_location(
        "_torch_trace_coverage_workload",
        workload_path,
    )
    wl = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(wl)
    results: dict = {}
    for op_name, run_fn in zip(wl.OP_NAMES, wl.ALL_OPS):
        print(f"[torch-trace-cov] START {op_name}", file=sys.stderr, flush=True)
        try:
            with profile(
                activities=[
                    ProfilerActivity.CPU,
                    ProfilerActivity.CUDA,
                ],
                acc_events=True,
            ) as prof:
                run_fn()
                torch.cuda.synchronize()
            results[op_name] = {
                "aten_ops": [
                    e.name
                    for e in prof.events()
                    if e.name.startswith("aten::")
                ],
                "cuda_kernels": [
                    e.name
                    for e in prof.events()
                    if e.device_type == torch.autograd.DeviceType.CUDA
                ],
            }
        except Exception as exc:
            results[op_name] = {
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(
                f"[torch-trace-cov] FAIL  {op_name} {type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            try:
                probe_device()
            except Exception as exc2:
                print(
                    f"[torch-trace-cov] DEVICE_DEAD after {op_name} "
                    f"{type(exc2).__name__}: {exc2}",
                    file=sys.stderr,
                    flush=True,
                )
                break
            continue
        print(f"[torch-trace-cov] OK    {op_name}", file=sys.stderr, flush=True)
    with open(output_path, "w", encoding="utf-8") as out_f:
        json.dump(results, out_f)


if __name__ == "__main__":
    main()
"""
).strip()


def unique_get_output_param_id(prefix: str) -> str:
    """Unique param_id (xdist worker + pid + tid + uuid) to avoid path races."""
    worker = os.environ.get("PYTEST_XDIST_WORKER", "main")
    return f"{prefix}_{worker}_{os.getpid()}_{threading.get_ident()}_{uuid.uuid4().hex}"


# -- ATen name helpers --
#
# ATen sample labels take one of two forms: torch.ops.aten.<op> for the
# default overload, or torch.ops.aten.<op>.<overload> for a named
# overload (.Tensor, .Scalar, .out, etc.).
# inject_roctx.instrument_all_torch_ops wraps the packet rather
# than individual overloads, so the generated workload always calls
# torch.ops.aten.<op> and relies on the dispatcher to route to the
# target overload from the synthesized argument types.


def aten_op_short_name(op_name: str) -> str:
    """Return the ATen op name without any overload suffix.

    torch.ops.aten.add → "add"; torch.ops.aten.add.Tensor → "add".
    Returns "" for non-ATen labels.
    """
    parts = op_name.split(".")
    return parts[3] if len(parts) >= 4 and parts[:3] == ["torch", "ops", "aten"] else ""


def aten_overload_key(op_name: str) -> Optional[str]:
    """Return <op>.<overload> for non-default overload labels, else None.

    torch.ops.aten.add → None; torch.ops.aten.add.Tensor → "add.Tensor".
    """
    parts = op_name.split(".")
    if len(parts) >= 5 and parts[:3] == ["torch", "ops", "aten"]:
        return ".".join(parts[3:])
    return None


def aten_packet_call_path(op_name: str) -> str:
    """Strip any overload suffix so the emitted call hits the wrapped packet.

    torch.ops.aten.add.Tensor → "torch.ops.aten.add"; names without
    an overload suffix are returned unchanged.
    """
    parts = op_name.split(".")
    if len(parts) >= 5 and parts[:3] == ["torch", "ops", "aten"]:
        return ".".join(parts[:4])
    return op_name


# -- Argument synthesis --
#
# Argument generation is restricted to hand-authored, per-operator builders
# registered in torch_trace_coverage_utils. An ATen operator is
# emitted into the coverage workload only when OP_SPECS contains an
# OpSpec.build entry for it (looked up by overload-specific key first,
# then by packet key). Operators with OpSpec.skip set, and operators
# with no entry at all, are reported as SKIP. Structural entries carry no
# arguments.
#
# Keep ROCm-safe argument generation in per-op verified builders.
# Two coverage extensions reuse existing builders:
#   * .out overloads: add out=torch.empty(0, ...) when schema/shape guards pass
#   * in-place overloads: reuse the out-of-place sibling builder
# Both only apply when a base short-name builder already exists in OP_SPECS.


# Overload suffixes whose .out variant accepts a single out=Tensor argument.
# Suffixes that bind a different parameter name or return multiple tensors are
# intentionally excluded and handled by the schema guards.
_SINGLE_TENSOR_OUT_OVERLOAD_SUFFIXES: frozenset = frozenset({
    "out",
    "Tensor_out",
    "OutTensor",
    "m_out",
    "out_mode",
})


# Short names excluded from generic .out injection because output dtype does not
# follow input dtype (predicates, argsort family, bool reductions, index ops).
# Multi-output .out variants are filtered by schema-return-count checks.
_OUT_MECHANISM_EXCLUDED_SHORT_NAMES: frozenset = frozenset({
    "eq",
    "ne",
    "lt",
    "gt",
    "ge",
    "le",
    "isnan",
    "isinf",
    "isfinite",
    "isneginf",
    "isposinf",
    "logical_and",
    "logical_or",
    "logical_xor",
    "logical_not",
    "isclose",
    "isin",
    "argmax",
    "argmin",
    "argsort",
    "all",
    "any",
    "nonzero",
    "tril_indices",
    "triu_indices",
})


def _first_tensor_dtype(args: List[Any]) -> Optional[torch.dtype]:
    """Return the dtype of the first Tensor found in args (recurses one
    level into the first element when it's a list/tuple, to handle
    list-of-tensors signatures like cat / stack).

    Returns None when no Tensor is found, which lets callers fall back
    to the default float dtype.
    """
    for arg in args:
        if isinstance(arg, torch.Tensor):
            return arg.dtype
        if isinstance(arg, (list, tuple)) and arg:
            first = arg[0]
            if isinstance(first, torch.Tensor):
                return first.dtype
    return None


def _schema_returns_single_tensor(schema: object) -> bool:
    """True iff a torch FunctionSchema declares exactly one return.
    Used to guard single-Tensor out= injection from multi-output
    .out variants. Returns False when schema is None.
    """
    if schema is None:
        return False
    returns = getattr(schema, "returns", None)
    if returns is None:
        return False
    try:
        return len(returns) == 1
    except TypeError:
        return False


def build_args_for_op(
    op: OpEntry,
) -> Optional[Tuple[List[Any], dict]]:
    """Return (args, kwargs) for op on CUDA, or None to SKIP.
    Structural -> ([], {}). ATen lookup: overload key, packet key,
    .out / trailing-underscore in-place fallbacks; else None.
    """
    device = "cuda"

    if op.category == "structural":
        return [], {}

    # Lookup order:
    #   1) overload-specific key (<op>.<overload>)
    #   2) .out mechanism (single-Tensor out= reuse of short-name builder)
    #   3) packet/short-name key (<op>)
    #   4) in-place mechanism (reuse out-of-place sibling builder)
    overload_key = aten_overload_key(op.name)
    short_name = aten_op_short_name(op.name) or op.name.rsplit(".", 1)[-1]

    if overload_key is not None:
        spec = OP_SPECS.get(overload_key)
        if spec is not None:
            if spec.skip is not None:
                return None
            if spec.build is not None:
                return spec.build(device)

    if overload_key is not None and "." in overload_key:
        overload_suffix = overload_key.split(".", 1)[1]
        if (
            overload_suffix in _SINGLE_TENSOR_OUT_OVERLOAD_SUFFIXES
            and short_name not in _OUT_MECHANISM_EXCLUDED_SHORT_NAMES
            and _schema_returns_single_tensor(op.schema)
        ):
            base = OP_SPECS.get(short_name)
            if base is not None and base.build is not None:
                base_args, base_kwargs = base.build(device)
                # The output dtype matches the first Tensor arg's
                # dtype for every short-name builder cleared by the
                # exclusion list above (pure float→float, int→int, or
                # complex→complex elementwise ops). torch.empty(0, ...)
                # is then resized to the actual output shape by
                # PyTorch's .out contract without raising.
                out_dtype = _first_tensor_dtype(base_args)
                if out_dtype is None:
                    out_dtype = torch.get_default_dtype()
                out_kwargs = dict(base_kwargs)
                out_kwargs["out"] = torch.empty(
                    0,
                    device=device,
                    dtype=out_dtype,
                )
                return base_args, out_kwargs

    spec = OP_SPECS.get(short_name)
    if spec is not None:
        if spec.skip is not None:
            return None
        if spec.build is not None:
            return spec.build(device)

    if (
        short_name.endswith("_")
        and not short_name.endswith("__")
        and not short_name.startswith("_")
    ):
        out_of_place = OP_SPECS.get(short_name[:-1])
        if out_of_place is not None and out_of_place.build is not None:
            return out_of_place.build(device)

    return None


# -- Operator discovery --


def _op_skip_text(op: OpEntry) -> Optional[str]:
    """OpSpec.skip text for op (overload-key first, short-name second), else None."""
    overload_key = aten_overload_key(op.name)
    spec = None
    if overload_key is not None:
        spec = OP_SPECS.get(overload_key)
    if spec is None:
        spec = OP_SPECS.get(aten_op_short_name(op.name) or op.name.rsplit(".", 1)[-1])
    return spec.skip if spec is not None else None


def discover_operators() -> Tuple[List[OpEntry], List[OpEntry], List[OpEntry]]:
    """Enumerate ATen + structural operators for the coverage sample.

    Walks torch.ops.aten emitting one OpEntry per overload with a CUDA
    kernel. Returns (kept_aten_ops, structural_ops, excluded_aten_ops);
    excluded ops are NVIDIA-/CUDA-only backends incompatible with ROCm
    and are surfaced for reporting only.
    """
    aten_ops: List[OpEntry] = []
    seen: Set[str] = set()

    for ns_name in ("aten",):
        ns = getattr(torch.ops, ns_name, None)
        if ns is None:
            continue
        for op_name in dir(ns):
            packet = getattr(ns, op_name, None)
            if packet is None or not hasattr(packet, "overloads"):
                continue
            try:
                overload_names = list(packet.overloads())
            except Exception:
                continue

            for ov_name in overload_names:
                try:
                    overload = getattr(packet, ov_name)
                except AttributeError:
                    continue
                try:
                    has_cuda = torch._C._dispatch_has_kernel_for_dispatch_key(
                        overload.name(),
                        torch._C.DispatchKey.CUDA,
                    )
                except Exception:
                    continue
                if not has_cuda:
                    continue

                if ov_name == "default":
                    full = f"torch.ops.{ns_name}.{op_name}"
                else:
                    full = f"torch.ops.{ns_name}.{op_name}.{ov_name}"
                if full in seen:
                    continue
                seen.add(full)

                schema = getattr(overload, "_schema", None)
                aten_ops.append(OpEntry(full, "aten", schema))

    # Partition the enumerated ATen ops: CUDA-only backends are dropped
    # at discovery so they never enter the workload or the SKIP report.
    # Structural ops are not partitioned — they don't carry an OpSpec.
    kept_aten_ops: List[OpEntry] = []
    excluded_aten_ops: List[OpEntry] = []
    for op in aten_ops:
        if is_hardware_incompatible_cuda_only(_op_skip_text(op)):
            excluded_aten_ops.append(op)
        else:
            kept_aten_ops.append(op)

    # Structural entries come from STRUCTURAL_BUILDERS; sorted keeps
    # emission deterministic across hash-seed randomization.
    structural_ops = [
        OpEntry(name, "structural", None) for name in sorted(STRUCTURAL_BUILDERS.keys())
    ]

    return kept_aten_ops, structural_ops, excluded_aten_ops


# -- Marker matching --


def marker_matches_op(op_name: str, marker_leaf: str) -> bool:
    """Return whether a ROCTX Function leaf counts as coverage for op_name.

    Class-specific structural entries (e.g. nn.Module.Conv2d.__call__) only
    match class-specific markers (nn.Module.Conv2d.forward); generic
    labels fall through to *.forward / *.step suffix matching.
    """
    if op_name == marker_leaf:
        return True

    # Class-specific structural patterns: nn.Module.<Class>.__call__ only
    # matches markers ending in <Class>.forward (inject_roctx rewrites
    # __call__ to forward). Generic labels fall through to the suffix rule.
    if (
        op_name.startswith("nn.Module.")
        and op_name.endswith(".__call__")
        and op_name != "nn.Module.__call__"
    ):
        cls = op_name[len("nn.Module.") : -len(".__call__")]
        if marker_leaf.endswith(f".{cls}.forward") or marker_leaf == (
            f"nn.Module.{cls}.forward"
        ):
            return True
        return False

    if (
        op_name.startswith("Optimizer.")
        and op_name.endswith(".step")
        and op_name != "Optimizer.step"
    ):
        cls = op_name[len("Optimizer.") : -len(".step")]
        if cls and marker_leaf.endswith(f".{cls}.step"):
            return True
        return False

    # ATen overload labels (torch.ops.aten.<op>.<overload>) match the
    # packet-level marker torch.ops.aten.<op> that inject_roctx emits.
    if op_name.startswith("torch.ops.aten."):
        op_short = aten_op_short_name(op_name)
        if op_short:
            packet = aten_packet_call_path(op_name)
            if marker_leaf == packet:
                return True
            marker_norm = marker_leaf.rsplit("::", 1)[-1].rsplit(".", 1)[-1]
            if marker_norm == op_short:
                return True
            if marker_leaf.endswith(f".{op_short}"):
                return True
            return False

    op_leaf = op_name.rsplit("::", 1)[-1].rsplit(".", 1)[-1]
    marker_norm = marker_leaf.rsplit("::", 1)[-1].rsplit(".", 1)[-1]
    if marker_norm == op_leaf:
        return True
    if marker_leaf.endswith(f".{op_leaf}"):
        return True

    # Generic "any module / any optimizer" fallbacks.
    if op_name == "Optimizer.step" and marker_leaf.endswith(".step"):
        return True
    if (
        op_name == "nn.Module.__call__"
        and ".forward" in marker_leaf
        and marker_leaf.startswith("nn.Module.")
    ):
        return True
    return False


# -- Workload script generation --

# Ruff E501 uses 88 columns; workload def bodies are indented by 4 spaces.
WORKLOAD_EMIT_BODY_CONTENT_MAX = 84


def workload_emit_tensor_rand_uniform_setup(
    vname: str,
    shape: Tuple[Any, ...],
    scale: float,
) -> List[str]:
    """Emit torch.rand times scale (non-negative rates, etc.)."""
    scale_repr = repr(scale)
    single = f"{vname} = torch.rand({shape}, device=device) * {scale_repr}"
    if len(single) <= WORKLOAD_EMIT_BODY_CONTENT_MAX:
        return [single]
    return [
        f"{vname} = torch.rand(",
        f"    {shape},",
        "    device=device,",
        f") * {scale_repr}",
    ]


def workload_emit_tensor_rand_setup(
    vname: str,
    shape: Tuple[Any, ...],
) -> List[str]:
    """Emit torch.rand in [0, 1) (Bernoulli probabilities, etc.)."""
    single = f"{vname} = torch.rand({shape}, device=device)"
    if len(single) <= WORKLOAD_EMIT_BODY_CONTENT_MAX:
        return [single]
    return [
        f"{vname} = torch.rand(",
        f"    {shape},",
        "    device=device,",
        ")",
    ]


def factory_expr_for_dtype(shape: Tuple[Any, ...], dtype: torch.dtype) -> str:
    """Return a random-tensor factory expression that preserves dtype.

    randn for real/complex floats, randint for integers, and
    randint(0, 2, ..., dtype=bool) for bool.
    """
    if dtype == torch.bool:
        return f"torch.randint(0, 2, {shape}, device=device, dtype=torch.bool)"
    if dtype.is_floating_point or dtype.is_complex:
        return f"torch.randn({shape}, device=device, dtype={dtype!r})"
    low, high = (-4, 4) if dtype == torch.int8 else (0, 4)
    return f"torch.randint({low}, {high}, {shape}, device=device, dtype={dtype!r})"


def workload_emit_assignment(vname: str, rhs: str) -> List[str]:
    """Emit vname = rhs on one line, or wrap across lines for Ruff E501."""
    single = f"{vname} = {rhs}"
    if len(single) <= WORKLOAD_EMIT_BODY_CONTENT_MAX:
        return [single]
    return [f"{vname} = (", f"    {rhs}", ")"]


def workload_emit_tensor_random_setup(
    vname: str,
    shape: Tuple[Any, ...],
    dtype: torch.dtype,
) -> List[str]:
    """Emit a dtype-preserving random tensor construction."""
    return workload_emit_assignment(vname, factory_expr_for_dtype(shape, dtype))


def workload_emit_tensor_spd_setup(
    vname: str,
    shape: Tuple[Any, ...],
) -> List[str]:
    """Emit an SPD matrix (a @ a.mT + n*I) for cholesky et al."""
    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError(
            f"CoverageTensorArg(emit='spd') expects a square shape, got {shape!r}"
        )
    n = shape[0]
    return [
        f"{vname}_a = torch.randn({shape}, device=device)",
        (f"{vname} = {vname}_a @ {vname}_a.mT + {n} * torch.eye({n}, device=device)"),
    ]


def workload_emit_tensor_pivots_1based_setup(
    vname: str,
    shape: Tuple[Any, ...],
) -> List[str]:
    """Emit a 1-D 1-based identity int32 permutation for LU pivots.

    lu_unpack / linalg_lu_solve require pivots in [1, n]; random
    int32 values would walk off the row table and trigger HIP 719.
    """
    if len(shape) != 1:
        raise ValueError(
            f"CoverageTensorArg(emit='pivots_1based') expects a 1-D shape, "
            f"got {shape!r}"
        )
    n = shape[0]
    return [
        f"{vname} = torch.arange(1, {n + 1}, dtype=torch.int32, device=device)",
    ]


def workload_emit_tensor_cumsum_offsets_setup(
    vname: str,
    shape: Tuple[Any, ...],
    segment_length: int,
) -> List[str]:
    """Emit [0, k, 2k, ..., n*k] int64 offsets for nested/jagged ops
    (length n+1; terminal entry == values.numel()).
    """
    if len(shape) != 1:
        raise ValueError(
            f"CoverageTensorArg(emit='cumsum_offsets') expects a 1-D shape, "
            f"got {shape!r}"
        )
    n = shape[0]
    return [
        (
            f"{vname} = torch.arange(0, {n * segment_length + 1}, "
            f"{segment_length}, dtype=torch.int64, device=device)"
        ),
    ]


def workload_emit_tensor_bool_all_true_setup(
    vname: str,
    shape: Tuple[Any, ...],
) -> List[str]:
    """Emit an all-True bool tensor of shape.

    Needed by _assert_async and similar ops that trap on False input;
    the default random emitter would produce False half the time.
    """
    return [
        f"{vname} = torch.ones({shape}, device=device, dtype=torch.bool)",
    ]


def workload_emit_multiline_call(
    call_expr: str,
    arg_expression_strings: List[str],
) -> List[str]:
    """Emit call_expr(...) on one line or split across lines for Ruff E501."""
    parts = list(arg_expression_strings)
    if not parts:
        single = f"{call_expr}()"
        if len(single) <= WORKLOAD_EMIT_BODY_CONTENT_MAX:
            return [single]
        return [
            f"{call_expr}(",
            ")",
        ]
    one_line = f"{call_expr}(" + ", ".join(parts) + ")"
    if len(one_line) <= WORKLOAD_EMIT_BODY_CONTENT_MAX:
        return [one_line]
    out_lines = [f"{call_expr}("]
    for index, part in enumerate(parts):
        last = index == len(parts) - 1
        suffix = "" if last else ","
        out_lines.append(f"    {part}{suffix}")
    out_lines.append(")")
    return out_lines


COVERAGE_TENSOR_ARG_EMITTERS: Dict[
    str,
    Callable[["CoverageTensorArg", str], List[str]],
] = {
    "rand": lambda a, v: workload_emit_tensor_rand_setup(v, a.shape),
    "rand_uniform": lambda a, v: workload_emit_tensor_rand_uniform_setup(
        v,
        a.shape,
        a.scale,
    ),
    "spd": lambda a, v: workload_emit_tensor_spd_setup(v, a.shape),
    "pivots_1based": lambda a, v: workload_emit_tensor_pivots_1based_setup(
        v,
        a.shape,
    ),
    "cumsum_offsets": lambda a, v: workload_emit_tensor_cumsum_offsets_setup(
        v,
        a.shape,
        int(a.scale),
    ),
    "bool_all_true": lambda a, v: workload_emit_tensor_bool_all_true_setup(
        v,
        a.shape,
    ),
}


def serialize_arg(argument_value: Any, vname: str) -> Tuple[List[str], str]:
    """Build emitted-source lines defining vname plus an expression
    to pass. Returns (setup_lines, expr): setup_lines run before the
    profiled call (empty for scalars), expr appears inside call_expr.

    Value-lossy for plain torch.Tensor inputs: only shape+dtype survive
    and values are re-randomized by the emitted factory. Use
    CoverageTensorArg when the builder needs specific values.
    """
    if isinstance(argument_value, CoverageTensorArg):
        emitter = COVERAGE_TENSOR_ARG_EMITTERS.get(argument_value.emit)
        if emitter is None:
            raise ValueError(
                f"unsupported CoverageTensorArg.emit: {argument_value.emit!r}"
            )
        return emitter(argument_value, vname), vname

    if isinstance(argument_value, torch.Tensor):
        return (
            workload_emit_tensor_random_setup(
                vname,
                tuple(argument_value.shape),
                argument_value.dtype,
            ),
            vname,
        )

    # Serialize list / tuple elements independently: preserves heterogeneous
    # shapes and routes CoverageTensorArg / nested tensor entries through
    # their emitters instead of repr (which would emit unbound names).
    if isinstance(argument_value, (list, tuple)):
        open_b, close_b = ("[", "]") if isinstance(argument_value, list) else ("(", ")")
        if not argument_value:
            return [], f"{open_b}{close_b}" if open_b == "[" else "()"
        stmts: List[str] = []
        parts: List[str] = []
        for index, elem in enumerate(argument_value):
            sub_stmts, sub_expr = serialize_arg(elem, f"{vname}_sub{index}")
            stmts.extend(sub_stmts)
            parts.append(sub_expr)
        body = ", ".join(parts)
        if isinstance(argument_value, tuple) and len(argument_value) == 1:
            body += ","
        return stmts, f"{open_b}{body}{close_b}"

    if argument_value is None:
        return [], "None"
    # Scalars / dtypes / strings: repr round-trips to valid Python source.
    return [], repr(argument_value)


# -- Structural builder registry ----------------------------------------------
#
# A builder is safe_var -> (setup_lines, call_expr, call_args): statements
# placed inside def _run_<safe_var>() before the call, a callable expression,
# and the comma-separated argument substring. STRUCTURAL_BUILDERS is the
# single source of truth for the structural sample — both
# discover_operators and emit_structural_preamble consume it, so
# adding a new API surface is one entry.
StructuralBuilder = Callable[[str], Tuple[List[str], str, str]]


def builder_nn_module_call(
    ctor_expr: str,
    input_expr: str,
    *,
    extra_setup: Tuple[str, ...] = (),
) -> StructuralBuilder:
    """Instantiate ctor_expr on CUDA and call it with input_expr.

    extra_setup prepends extra statements for auxiliary tensors referenced
    by input_expr.
    """

    def build(safe_var: str) -> Tuple[List[str], str, str]:
        setup: List[str] = ["import torch.nn as nn"]
        setup.extend(extra_setup)
        setup.append(f"_mod_{safe_var} = ({ctor_expr}).cuda()")
        return setup, f"_mod_{safe_var}", input_expr

    return build


def builder_optimizer_step(optimizer_ctor: str) -> StructuralBuilder:
    """Run forward+backward on a tiny Linear, then call optim.step().

    optimizer_ctor substitutes PARAMS with the Linear's parameters
    (e.g. "torch.optim.Adam(PARAMS, lr=1e-3)"). Real gradients are needed
    so stateful optimizers actually launch kernels.
    """

    def build(safe_var: str) -> Tuple[List[str], str, str]:
        params_expr = f"_m_{safe_var}.parameters()"
        optim_src = optimizer_ctor.replace("PARAMS", params_expr)
        return (
            [
                "import torch.nn as nn",
                f"_m_{safe_var} = nn.Linear(4, 4).cuda()",
                f"_opt_{safe_var} = {optim_src}",
                (f"_m_{safe_var}(torch.randn(2, 4, device=device)).sum().backward()"),
            ],
            f"_opt_{safe_var}.step",
            "",
        )

    return build


# Idempotent single-GPU process-group bootstrap inlined by distributed
# builders. Uses a file:// rendezvous in a per-process tmpdir so concurrent
# pytest workers (xdist, CI shards) cannot collide on a TCP port. The
# atexit hook destroys the process group on interpreter shutdown regardless
# of HIP/CUDA context state; without it, rocBLAS or NCCL cleanup on a
# faulted device can abort the interpreter with SIGABRT and mask the
# original failure.
DISTRIBUTED_BOOTSTRAP_SETUP: Tuple[str, ...] = (
    "import atexit as _atexit",
    "import tempfile as _tempfile",
    "import torch.distributed as _dist",
    "if not _dist.is_initialized():",
    "    _dist_init_dir = _tempfile.mkdtemp(prefix='torch_trace_cov_dist_')",
    "    _dist.init_process_group(",
    '        backend="nccl",',
    '        init_method=f"file://{_dist_init_dir}/init",',
    "        rank=0,",
    "        world_size=1,",
    "    )",
    "    def _shutdown_dist():",
    "        try:",
    "            if _dist.is_initialized():",
    "                _dist.destroy_process_group()",
    "        except Exception:",
    "            pass",
    "    _atexit.register(_shutdown_dist)",
)


def builder_distributed(
    call_expr: str,
    *,
    tensor_setup: Tuple[str, ...] = ("_t = torch.randn(4, 4, device=device)",),
    call_args: str = "_t",
) -> StructuralBuilder:
    """Run a torch.distributed.* collective on a single-GPU process group.

    Systems without a working NCCL/RCCL raise at init_process_group and the
    op is reported as SKIP — multi-GPU topology is not required.
    """

    def build(_safe_var: str) -> Tuple[List[str], str, str]:
        setup = list(DISTRIBUTED_BOOTSTRAP_SETUP) + list(tensor_setup)
        return setup, call_expr, call_args

    return build


# Per-module builders: name -> (ctor_expr, input_expr, extra_setup).
NN_MODULE_BUILDERS: Dict[str, Tuple[str, str, Tuple[str, ...]]] = {
    # Dense / linear layers
    "nn.Module.Linear.__call__": (
        "nn.Linear(4, 4)",
        "torch.randn(2, 4, device=device)",
        (),
    ),
    "nn.Module.Bilinear.__call__": (
        "nn.Bilinear(4, 4, 4)",
        "torch.randn(2, 4, device=device), torch.randn(2, 4, device=device)",
        (),
    ),
    # Convolutional layers
    "nn.Module.Conv1d.__call__": (
        "nn.Conv1d(3, 8, 3, padding=1)",
        "torch.randn(1, 3, 16, device=device)",
        (),
    ),
    "nn.Module.Conv2d.__call__": (
        "nn.Conv2d(3, 8, 3, padding=1)",
        "torch.randn(1, 3, 16, 16, device=device)",
        (),
    ),
    "nn.Module.Conv3d.__call__": (
        "nn.Conv3d(3, 8, 3, padding=1)",
        "torch.randn(1, 3, 4, 8, 8, device=device)",
        (),
    ),
    "nn.Module.ConvTranspose2d.__call__": (
        "nn.ConvTranspose2d(3, 8, 3, padding=1)",
        "torch.randn(1, 3, 16, 16, device=device)",
        (),
    ),
    # Normalization
    "nn.Module.BatchNorm1d.__call__": (
        "nn.BatchNorm1d(4)",
        "torch.randn(8, 4, device=device)",
        (),
    ),
    "nn.Module.BatchNorm2d.__call__": (
        "nn.BatchNorm2d(3)",
        "torch.randn(2, 3, 8, 8, device=device)",
        (),
    ),
    "nn.Module.LayerNorm.__call__": (
        "nn.LayerNorm(8)",
        "torch.randn(2, 8, device=device)",
        (),
    ),
    "nn.Module.GroupNorm.__call__": (
        "nn.GroupNorm(2, 4)",
        "torch.randn(2, 4, 8, 8, device=device)",
        (),
    ),
    "nn.Module.InstanceNorm2d.__call__": (
        "nn.InstanceNorm2d(3)",
        "torch.randn(2, 3, 8, 8, device=device)",
        (),
    ),
    # Embedding
    "nn.Module.Embedding.__call__": (
        "nn.Embedding(16, 4)",
        "torch.randint(0, 16, (2, 4), device=device)",
        (),
    ),
    "nn.Module.EmbeddingBag.__call__": (
        "nn.EmbeddingBag(16, 4, mode='mean')",
        "torch.randint(0, 16, (8,), device=device), "
        "torch.tensor([0, 4], device=device)",
        (),
    ),
    # Attention / transformer
    "nn.Module.MultiheadAttention.__call__": (
        "nn.MultiheadAttention(8, 2, batch_first=True)",
        "_q, _q, _q",
        ("_q = torch.randn(2, 4, 8, device=device)",),
    ),
    "nn.Module.TransformerEncoderLayer.__call__": (
        "nn.TransformerEncoderLayer(d_model=8, nhead=2, batch_first=True)",
        "torch.randn(2, 4, 8, device=device)",
        (),
    ),
    # Recurrent
    "nn.Module.RNN.__call__": (
        "nn.RNN(4, 4, batch_first=True)",
        "torch.randn(2, 3, 4, device=device)",
        (),
    ),
    "nn.Module.LSTM.__call__": (
        "nn.LSTM(4, 4, batch_first=True)",
        "torch.randn(2, 3, 4, device=device)",
        (),
    ),
    "nn.Module.GRU.__call__": (
        "nn.GRU(4, 4, batch_first=True)",
        "torch.randn(2, 3, 4, device=device)",
        (),
    ),
    # Activations (trainable / parameter-free)
    "nn.Module.ReLU.__call__": (
        "nn.ReLU()",
        "torch.randn(4, 4, device=device)",
        (),
    ),
    "nn.Module.GELU.__call__": (
        "nn.GELU()",
        "torch.randn(4, 4, device=device)",
        (),
    ),
    "nn.Module.SiLU.__call__": (
        "nn.SiLU()",
        "torch.randn(4, 4, device=device)",
        (),
    ),
    "nn.Module.Softmax.__call__": (
        "nn.Softmax(dim=-1)",
        "torch.randn(4, 4, device=device)",
        (),
    ),
    "nn.Module.LogSoftmax.__call__": (
        "nn.LogSoftmax(dim=-1)",
        "torch.randn(4, 4, device=device)",
        (),
    ),
    # Dropout
    "nn.Module.Dropout.__call__": (
        "nn.Dropout(p=0.5)",
        "torch.randn(4, 4, device=device)",
        (),
    ),
    # Pooling
    "nn.Module.MaxPool2d.__call__": (
        "nn.MaxPool2d(2)",
        "torch.randn(1, 3, 8, 8, device=device)",
        (),
    ),
    "nn.Module.AvgPool2d.__call__": (
        "nn.AvgPool2d(2)",
        "torch.randn(1, 3, 8, 8, device=device)",
        (),
    ),
    "nn.Module.AdaptiveAvgPool2d.__call__": (
        "nn.AdaptiveAvgPool2d((4, 4))",
        "torch.randn(1, 3, 8, 8, device=device)",
        (),
    ),
    # Upsample / pixel shuffle / reshape
    "nn.Module.Upsample.__call__": (
        "nn.Upsample(scale_factor=2, mode='nearest')",
        "torch.randn(1, 3, 4, 4, device=device)",
        (),
    ),
    "nn.Module.PixelShuffle.__call__": (
        "nn.PixelShuffle(2)",
        "torch.randn(1, 4, 4, 4, device=device)",
        (),
    ),
    "nn.Module.Flatten.__call__": (
        "nn.Flatten()",
        "torch.randn(2, 3, 4, 4, device=device)",
        (),
    ),
    # Containers (just to exercise the wrapper forward)
    "nn.Module.Sequential.__call__": (
        "nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 4))",
        "torch.randn(2, 4, device=device)",
        (),
    ),
}


# Per-optimizer builders ------------------------------------------------------
OPTIMIZER_BUILDERS: Dict[str, str] = {
    "Optimizer.SGD.step": "torch.optim.SGD(PARAMS, lr=0.01)",
    "Optimizer.Adam.step": "torch.optim.Adam(PARAMS, lr=1e-3)",
    "Optimizer.AdamW.step": "torch.optim.AdamW(PARAMS, lr=1e-3)",
    "Optimizer.RMSprop.step": "torch.optim.RMSprop(PARAMS, lr=1e-3)",
    "Optimizer.Adagrad.step": "torch.optim.Adagrad(PARAMS, lr=1e-2)",
    "Optimizer.Adadelta.step": "torch.optim.Adadelta(PARAMS, lr=1.0)",
    "Optimizer.NAdam.step": "torch.optim.NAdam(PARAMS, lr=1e-3)",
    "Optimizer.RAdam.step": "torch.optim.RAdam(PARAMS, lr=1e-3)",
    "Optimizer.Adamax.step": "torch.optim.Adamax(PARAMS, lr=1e-3)",
    "Optimizer.ASGD.step": "torch.optim.ASGD(PARAMS, lr=1e-3)",
}


# Autograd builders: tiny Linear + scalar loss so backward launches real kernels.


def builder_autograd_grad(safe_var: str) -> Tuple[List[str], str, str]:
    """torch.autograd.grad(loss, params) on a 1-layer MLP."""
    return (
        [
            "import torch.nn as nn",
            f"_m_{safe_var} = nn.Linear(4, 4).cuda()",
            (
                f"_loss_{safe_var} = _m_{safe_var}("
                "torch.randn(2, 4, device=device)).sum()"
            ),
        ],
        "torch.autograd.grad",
        f"_loss_{safe_var}, list(_m_{safe_var}.parameters())",
    )


def builder_autograd_backward(safe_var: str) -> Tuple[List[str], str, str]:
    """torch.autograd.backward([loss]) — functional form of .backward()."""
    return (
        [
            "import torch.nn as nn",
            f"_m_{safe_var} = nn.Linear(4, 4).cuda()",
            (
                f"_loss_{safe_var} = _m_{safe_var}("
                "torch.randn(2, 4, device=device)).sum()"
            ),
        ],
        "torch.autograd.backward",
        f"[_loss_{safe_var}]",
    )


def builder_autograd_function_apply(safe_var: str) -> Tuple[List[str], str, str]:
    """Define a tiny torch.autograd.Function and call its .apply."""
    return (
        [
            f"class _Fn_{safe_var}(torch.autograd.Function):",
            "    @staticmethod",
            "    def forward(ctx, x):",
            "        ctx.save_for_backward(x)",
            "        return x * 2.0 + 1.0",
            "    @staticmethod",
            "    def backward(ctx, grad_output):",
            "        (x,) = ctx.saved_tensors",
            "        return grad_output * 2.0",
            (f"_x_{safe_var} = torch.randn(4, 4, device=device, requires_grad=True)"),
        ],
        f"_Fn_{safe_var}.apply",
        f"_x_{safe_var}",
    )


def builder_autograd_functional_jacobian(
    safe_var: str,
) -> Tuple[List[str], str, str]:
    """torch.autograd.functional.jacobian(f, x) for a 1-layer MLP wrapper."""
    return (
        [
            "import torch.nn as nn",
            f"_m_{safe_var} = nn.Linear(4, 4).cuda()",
            f"_x_{safe_var} = torch.randn(4, device=device)",
        ],
        "torch.autograd.functional.jacobian",
        f"lambda z: _m_{safe_var}(z).sum(), _x_{safe_var}",
    )


def builder_autograd_functional_hessian(
    safe_var: str,
) -> Tuple[List[str], str, str]:
    """torch.autograd.functional.hessian(f, x) of a quadratic form."""
    return (
        [f"_x_{safe_var} = torch.randn(4, device=device)"],
        "torch.autograd.functional.hessian",
        f"lambda z: (z * z).sum(), _x_{safe_var}",
    )


def builder_autograd_functional_vjp(safe_var: str) -> Tuple[List[str], str, str]:
    """torch.autograd.functional.vjp(f, x, v) for a tiny MLP."""
    return (
        [
            "import torch.nn as nn",
            f"_m_{safe_var} = nn.Linear(4, 4).cuda()",
            f"_x_{safe_var} = torch.randn(4, device=device)",
            f"_v_{safe_var} = torch.randn(4, device=device)",
        ],
        "torch.autograd.functional.vjp",
        f"lambda z: _m_{safe_var}(z), _x_{safe_var}, _v_{safe_var}",
    )


def builder_autograd_functional_jvp(safe_var: str) -> Tuple[List[str], str, str]:
    """torch.autograd.functional.jvp(f, x, v) for a tiny MLP."""
    return (
        [
            "import torch.nn as nn",
            f"_m_{safe_var} = nn.Linear(4, 4).cuda()",
            f"_x_{safe_var} = torch.randn(4, device=device)",
            f"_v_{safe_var} = torch.randn(4, device=device)",
        ],
        "torch.autograd.functional.jvp",
        f"lambda z: _m_{safe_var}(z), _x_{safe_var}, _v_{safe_var}",
    )


# Compile / JIT builders: best-effort — compile/trace failures map to SKIP.


def builder_torch_compile(safe_var: str) -> Tuple[List[str], str, str]:
    """Compile a trivial fn with torch.compile and invoke it once."""
    return (
        [
            f"def _fn_{safe_var}(x):",
            "    return (x * 2.0 + 1.0).relu().sum()",
            f"_cfn_{safe_var} = torch.compile(_fn_{safe_var})",
            f"_x_{safe_var} = torch.randn(64, device=device)",
        ],
        f"_cfn_{safe_var}",
        f"_x_{safe_var}",
    )


def builder_torch_jit_trace(safe_var: str) -> Tuple[List[str], str, str]:
    """Trace a tiny Linear with torch.jit.trace and run the traced module."""
    return (
        [
            "import torch.nn as nn",
            f"_m_{safe_var} = nn.Linear(4, 4).cuda().eval()",
            f"_ex_{safe_var} = torch.randn(2, 4, device=device)",
            (f"_tr_{safe_var} = torch.jit.trace(_m_{safe_var}, _ex_{safe_var})"),
        ],
        f"_tr_{safe_var}",
        f"_ex_{safe_var}",
    )


def builder_torch_jit_script(safe_var: str) -> Tuple[List[str], str, str]:
    """Script a pure-python fn with torch.jit.script and run it on the GPU."""
    return (
        [
            f"def _fn_{safe_var}(x):",
            "    return x * 2.0 + 1.0",
            f"_sc_{safe_var} = torch.jit.script(_fn_{safe_var})",
            f"_x_{safe_var} = torch.randn(4, 4, device=device)",
        ],
        f"_sc_{safe_var}",
        f"_x_{safe_var}",
    )


# CUDA utilities: mostly host-side. "Marker present, no kernels" counts as PASS
# in compare_single_op, so a missing marker flags a wrapper gap.


def builder_cuda_synchronize(_safe_var: str) -> Tuple[List[str], str, str]:
    return [], "torch.cuda.synchronize", ""


def builder_cuda_current_device(_safe_var: str) -> Tuple[List[str], str, str]:
    return [], "torch.cuda.current_device", ""


def builder_cuda_device_count(_safe_var: str) -> Tuple[List[str], str, str]:
    return [], "torch.cuda.device_count", ""


def builder_cuda_empty_cache(_safe_var: str) -> Tuple[List[str], str, str]:
    return [], "torch.cuda.empty_cache", ""


def builder_cuda_memory_allocated(_safe_var: str) -> Tuple[List[str], str, str]:
    return [], "torch.cuda.memory_allocated", ""


def builder_cuda_reset_peak_memory_stats(
    _safe_var: str,
) -> Tuple[List[str], str, str]:
    return [], "torch.cuda.reset_peak_memory_stats", ""


def builder_cuda_manual_seed(_safe_var: str) -> Tuple[List[str], str, str]:
    return [], "torch.cuda.manual_seed", "0"


def builder_cuda_set_device(_safe_var: str) -> Tuple[List[str], str, str]:
    return [], "torch.cuda.set_device", "0"


def builder_cuda_stream(safe_var: str) -> Tuple[List[str], str, str]:
    """Create a torch.cuda.Stream and submit one op on it.

    The real work (stream context + kernel + sync) runs in setup; the emitted
    "call" is a no-op because markers have already fired.
    """
    return (
        [
            f"_stream_{safe_var} = torch.cuda.Stream()",
            f"with torch.cuda.stream(_stream_{safe_var}):",
            f"    _y_{safe_var} = torch.randn(8, 8, device=device) * 2.0",
            f"_stream_{safe_var}.synchronize()",
        ],
        "(lambda: None)",
        "",
    )


def builder_cuda_event(safe_var: str) -> Tuple[List[str], str, str]:
    """Record and query a torch.cuda.Event around a small op."""
    return (
        [
            f"_ev_start_{safe_var} = torch.cuda.Event(enable_timing=True)",
            f"_ev_end_{safe_var} = torch.cuda.Event(enable_timing=True)",
            f"_ev_start_{safe_var}.record()",
            f"_y_{safe_var} = torch.randn(8, 8, device=device) * 2.0",
            f"_ev_end_{safe_var}.record()",
            f"_ev_end_{safe_var}.synchronize()",
        ],
        "(lambda: None)",
        "",
    )


# Assembly of the registry ----------------------------------------------------


def build_structural_builder_registry() -> Dict[str, StructuralBuilder]:
    """Flat name -> StructuralBuilder map. Duplicate names raise."""
    registry: Dict[str, StructuralBuilder] = {}

    # Generic fallback entries: nn.Module.__call__, Optimizer.step, and
    # torch.Tensor.backward match any marker whose class-specific variant
    # below is not present (or is bypassed by a subclass override).
    def _generic_nn_call(safe_var: str) -> Tuple[List[str], str, str]:
        return (
            [
                "import torch.nn as nn",
                f"_model_{safe_var} = nn.Linear(4, 4).cuda()",
            ],
            f"_model_{safe_var}",
            "torch.randn(2, 4, device=device)",
        )

    def _generic_optimizer_step(safe_var: str) -> Tuple[List[str], str, str]:
        return (
            [
                "import torch.nn as nn",
                f"_m_{safe_var} = nn.Linear(4, 4).cuda()",
                f"_opt_{safe_var} = torch.optim.SGD(",
                f"    _m_{safe_var}.parameters(),",
                "    lr=0.01,",
                ")",
                f"_m_{safe_var}(",
                "    torch.randn(2, 4, device=device),",
                ").sum().backward()",
            ],
            f"_opt_{safe_var}.step",
            "",
        )

    def _generic_tensor_backward(safe_var: str) -> Tuple[List[str], str, str]:
        return (
            [
                f"_lin_{safe_var} = torch.nn.Linear(4, 4).cuda()",
                f"_loss_{safe_var} = _lin_{safe_var}(",
                "    torch.randn(2, 4, device=device),",
                ").sum()",
            ],
            f"_loss_{safe_var}.backward",
            "",
        )

    registry["nn.Module.__call__"] = _generic_nn_call
    registry["Optimizer.step"] = _generic_optimizer_step
    registry["torch.Tensor.backward"] = _generic_tensor_backward

    for name, (ctor, input_expr, extra_setup) in NN_MODULE_BUILDERS.items():
        if name in registry:
            raise AssertionError(f"duplicate structural builder: {name!r}")
        registry[name] = builder_nn_module_call(
            ctor,
            input_expr,
            extra_setup=extra_setup,
        )

    for name, optim_ctor in OPTIMIZER_BUILDERS.items():
        if name in registry:
            raise AssertionError(f"duplicate structural builder: {name!r}")
        registry[name] = builder_optimizer_step(optim_ctor)

    for name, fn in (
        ("torch.autograd.grad", builder_autograd_grad),
        ("torch.autograd.backward", builder_autograd_backward),
        ("torch.autograd.Function.apply", builder_autograd_function_apply),
        (
            "torch.autograd.functional.jacobian",
            builder_autograd_functional_jacobian,
        ),
        (
            "torch.autograd.functional.hessian",
            builder_autograd_functional_hessian,
        ),
        ("torch.autograd.functional.vjp", builder_autograd_functional_vjp),
        ("torch.autograd.functional.jvp", builder_autograd_functional_jvp),
        ("torch.compile", builder_torch_compile),
        ("torch.jit.trace", builder_torch_jit_trace),
        ("torch.jit.script", builder_torch_jit_script),
        ("torch.distributed.all_reduce", builder_distributed("_dist.all_reduce")),
        (
            "torch.distributed.broadcast",
            builder_distributed(
                "_dist.broadcast",
                call_args="_t, src=0",
            ),
        ),
        (
            "torch.distributed.reduce",
            builder_distributed(
                "_dist.reduce",
                call_args="_t, dst=0",
            ),
        ),
        (
            "torch.distributed.all_gather",
            builder_distributed(
                "_dist.all_gather",
                tensor_setup=(
                    "_t = torch.randn(4, 4, device=device)",
                    "_out = [torch.empty_like(_t)]",
                ),
                call_args="_out, _t",
            ),
        ),
        (
            "torch.distributed.reduce_scatter",
            builder_distributed(
                "_dist.reduce_scatter",
                tensor_setup=(
                    "_t = torch.randn(4, 4, device=device)",
                    "_inp = [torch.randn(4, 4, device=device)]",
                ),
                call_args="_t, _inp",
            ),
        ),
        (
            "torch.distributed.barrier",
            builder_distributed(
                "_dist.barrier",
                tensor_setup=(),
                call_args="",
            ),
        ),
        ("torch.cuda.synchronize", builder_cuda_synchronize),
        ("torch.cuda.current_device", builder_cuda_current_device),
        ("torch.cuda.device_count", builder_cuda_device_count),
        ("torch.cuda.empty_cache", builder_cuda_empty_cache),
        ("torch.cuda.memory_allocated", builder_cuda_memory_allocated),
        (
            "torch.cuda.reset_peak_memory_stats",
            builder_cuda_reset_peak_memory_stats,
        ),
        ("torch.cuda.manual_seed", builder_cuda_manual_seed),
        ("torch.cuda.set_device", builder_cuda_set_device),
        ("torch.cuda.Stream", builder_cuda_stream),
        ("torch.cuda.Event", builder_cuda_event),
    ):
        if name in registry:
            raise AssertionError(f"duplicate structural builder: {name!r}")
        registry[name] = fn

    return registry


STRUCTURAL_BUILDERS: Dict[str, StructuralBuilder] = build_structural_builder_registry()


def emit_structural_preamble(
    op_name: str,
    safe_var: str,
) -> Tuple[List[str], str, str]:
    """Dispatch into STRUCTURAL_BUILDERS for one structural workload.

    Unknown names fall through to ([], op_name, ""); the resulting bare
    call raises at runtime and the comparison layer maps it to SKIP.
    """
    builder = STRUCTURAL_BUILDERS.get(op_name)
    if builder is None:
        return [], op_name, ""
    return builder(safe_var)


def build_workload_module_lines(operators: List[OpEntry]) -> List[str]:
    """Emit source lines for a minimal coverage_workload.py module.

    One def per operator plus run_all. The ground-truth runner imports
    this module and wraps each ALL_OPS entry with torch.profiler;
    run_all() is the entry point when rocprof-compute runs this script.
    """
    lines = [
        "import os",
        "import sys",
        # AMD_SERIALIZE_KERNEL must be set before import torch: HIP
        # reads it once at runtime initialisation. Synchronous dispatch is
        # required for asynchronous faults to surface at their launch site
        # so per-operator attribution is correct.
        'os.environ.setdefault("AMD_SERIALIZE_KERNEL", "3")',
        "import torch",
        "",
        'device = "cuda"',
        "",
    ]
    op_name_literals: List[str] = []
    runner_fn_names: List[str] = []

    for op in operators:
        build_result = build_args_for_op(op)
        if build_result is None:
            continue

        args, kwargs = build_result
        safe_var = op.name.replace(".", "_").replace("::", "_")

        op_setup: List[str] = []
        arg_strs = []
        for arg_index, arg_value in enumerate(args):
            vname = f"_arg_{safe_var}_{arg_index}"
            stmts, expr = serialize_arg(arg_value, vname)
            op_setup.extend(stmts)
            arg_strs.append(expr)

        # Route kwargs through serialize_arg so tensor kwargs emit factory
        # calls instead of bare tensor([...]) reprs (which NameError).
        kwarg_strs: List[str] = []
        for keyword, value in kwargs.items():
            vname = f"_kwarg_{safe_var}_{keyword}"
            stmts, expr = serialize_arg(value, vname)
            op_setup.extend(stmts)
            kwarg_strs.append(f"{keyword}={expr}")
        call_args = ", ".join(arg_strs + kwarg_strs)

        if op.name.startswith("torch.ops."):
            # Always call the packet — inject_roctx wraps packets, not specific
            # overloads, and replaces the packet attribute with a plain
            # function (so <packet>.<overload> is not accessible in the
            # instrumented run). Argument types drive dispatch to the target
            # overload.
            call_expr = aten_packet_call_path(op.name)
        elif op.category == "structural":
            extra_setup, call_expr, call_args = emit_structural_preamble(
                op.name,
                safe_var,
            )
            op_setup.extend(extra_setup)
        else:
            call_expr = op.name

        if op.category == "structural" and not call_args.strip():
            call_segments: List[str] = []
        elif op.category == "structural":
            call_segments = [call_args]
        else:
            call_segments = arg_strs + kwarg_strs
        call_body_lines = workload_emit_multiline_call(call_expr, call_segments)
        fn_name = f"_run_{safe_var}"
        lines.append(f"def {fn_name}():")
        for setup_line in op_setup:
            lines.append(f"    {setup_line}")
        for call_body_line in call_body_lines:
            lines.append(f"    {call_body_line}")
        lines.append("")

        op_name_literals.append(repr(op.name))
        runner_fn_names.append(fn_name)

    lines.append("OP_NAMES = [")
    for lit in op_name_literals:
        lines.append(f"    {lit},")
    lines.append("]")
    lines.append("")
    lines.append("ALL_OPS = [")
    for fn in runner_fn_names:
        lines.append(f"    {fn},")
    lines.append("]")
    lines.append("")
    # run_all iterates over every emitted operator. Per-operator
    # exceptions are logged and do not halt iteration. START / OK / FAIL
    # banners are flushed to stderr so that a native-code termination can
    # be attributed to the last operator whose START has no matching
    # terminal banner. torch.cuda.synchronize() anchors asynchronous
    # completion before OK is logged. After a per-operator failure,
    # probe_device() dispatches a minimal kernel to test context
    # liveness; if the probe also raises, iteration terminates with a
    # DEVICE_DEAD banner identifying the terminal operator.
    lines.append("def probe_device():")
    lines.append("    t = torch.empty(1, device=device)")
    lines.append("    t.add_(1)")
    lines.append("    torch.cuda.synchronize()")
    lines.append("")
    lines.append("def run_all():")
    lines.append("    for op_label, fn in zip(OP_NAMES, ALL_OPS):")
    lines.append(
        '        print(f"[torch-trace-cov] START {op_label}", '
        "file=sys.stderr, flush=True)"
    )
    lines.append("        try:")
    lines.append("            fn()")
    lines.append("            torch.cuda.synchronize()")
    lines.append("        except Exception as exc:")
    lines.append(
        '            print(f"[torch-trace-cov] FAIL  {op_label} '
        '{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)'
    )
    lines.append("            try:")
    lines.append("                probe_device()")
    lines.append("            except Exception as exc2:")
    lines.append(
        '                print(f"[torch-trace-cov] DEVICE_DEAD after {op_label} '
        '{type(exc2).__name__}: {exc2}", file=sys.stderr, flush=True)'
    )
    lines.append("                break")
    lines.append("            continue")
    lines.append(
        '        print(f"[torch-trace-cov] OK    {op_label}", '
        "file=sys.stderr, flush=True)"
    )
    lines.append("")
    lines.append('if __name__ == "__main__":')
    lines.append("    run_all()")
    lines.append("")

    return lines


def write_coverage_ground_truth_runner_script(path: str) -> None:
    """Write the static torch.profiler runner next to coverage_workload.py."""
    Path(path).write_text(
        COVERAGE_GROUND_TRUTH_RUNNER_SOURCE + "\n",
        encoding="utf-8",
    )


def write_coverage_workload_artifacts(
    operators: List[OpEntry],
    workload_script_path: str,
    ground_truth_runner_script_path: str,
) -> None:
    """Write coverage_workload.py and coverage_ground_truth_runner.py."""
    Path(workload_script_path).write_text(
        "\n".join(build_workload_module_lines(operators)),
        encoding="utf-8",
    )
    write_coverage_ground_truth_runner_script(ground_truth_runner_script_path)


# -- ROCTX marker CSV parsing --


def with_correlation_id_standard_name(marker_df: pd.DataFrame) -> pd.DataFrame:
    """Normalise the correlation-id column name across rocprof-compute versions."""
    if "Correlation_Id" not in marker_df.columns:
        return marker_df
    return marker_df.rename(columns={"Correlation_Id": "Correlation_ID"})


def segments_from_function_cell(func: object) -> List[str]:
    """Return every chain segment of a ROCTX Function cell, in order.

    A cell looks like ``aten::view/aten::_to_copy:#1@main.py:60``; we
    strip the ``:#...`` context tail and split the remainder on ``/``.
    Empty cells return ``[]``.
    """
    if not isinstance(func, str):
        return []
    op_path = func.split(":#")[0] if ":#" in func else func
    return [seg for seg in (s.strip() for s in op_path.split("/")) if seg]


def leaf_from_function_cell(func: object) -> Optional[str]:
    """Return the last path segment of a ROCTX Function cell, or None."""
    segments = segments_from_function_cell(func)
    return segments[-1] if segments else None


def collect_marker_ops_and_correlations(
    marker_df: pd.DataFrame,
) -> tuple[set[str], dict[str, set]]:
    """Return (marker_leaves, segment -> correlation_ids).

    The correlation map attributes each row's Correlation_ID to every
    segment of its chain so a kernel inside a nested marker is visible
    when the harness probes the enclosing op. Monotonic: only adds
    correlations the outer marker genuinely enclosed.
    """
    marker_ops: set[str] = set()
    op_to_corr: dict[str, set] = {}
    func_col = marker_df.get("Function")
    corr_col = marker_df.get("Correlation_ID")
    if func_col is None:
        return marker_ops, op_to_corr

    for row_index, function_cell in enumerate(func_col):
        segments = segments_from_function_cell(function_cell)
        if not segments:
            continue
        marker_ops.add(segments[-1])
        if corr_col is None:
            continue
        correlation_id = corr_col.iloc[row_index]
        if not pd.notna(correlation_id):
            continue
        for segment in segments:
            op_to_corr.setdefault(segment, set()).add(correlation_id)

    return marker_ops, op_to_corr


def merge_kernel_names_for_correlation_ids(
    correlation_ids: set,
    correlation_id_to_kernels: dict,
) -> set[str]:
    """Union the kernel-name sets for every correlation id in correlation_ids."""
    kernels: set[str] = set()
    for correlation_id in correlation_ids:
        kernels |= correlation_id_to_kernels.get(correlation_id, set())
    return kernels


def kernels_by_marker_leaf(
    op_to_corr: dict[str, set],
    counter_files: list[Path],
) -> dict[str, set[str]]:
    """Return leaf -> {kernel_name} by joining marker correlations to counters."""
    counter_df = pd.concat(
        [pd.read_csv(f) for f in counter_files],
        ignore_index=True,
    )
    counter_df = with_correlation_id_standard_name(counter_df)
    if "Kernel_Name" not in counter_df.columns:
        return {}

    all_corr_ids: set = set()
    for ids in op_to_corr.values():
        all_corr_ids |= ids

    matched = counter_df[counter_df["Correlation_ID"].isin(all_corr_ids)]
    correlation_id_to_kernels: dict = {}
    for correlation_id, kernel_name in zip(
        matched["Correlation_ID"],
        matched["Kernel_Name"],
    ):
        if not pd.notna(kernel_name):
            continue
        correlation_id_to_kernels.setdefault(correlation_id, set()).add(kernel_name)

    op_to_kernels: dict[str, set[str]] = {}
    for leaf, correlation_ids_for_leaf in op_to_corr.items():
        kernels = merge_kernel_names_for_correlation_ids(
            correlation_ids_for_leaf,
            correlation_id_to_kernels,
        )
        if kernels:
            op_to_kernels[leaf] = kernels
    return op_to_kernels


def _read_roctx_marker_dataframe(workload_dir: str) -> Optional[pd.DataFrame]:
    """Concatenate marker_api_trace.csv files under workload_dir;
    returns None when no marker CSV is present.
    """
    marker_files = sorted(Path(workload_dir).glob("**/*marker_api_trace.csv"))
    if not marker_files:
        return None
    marker_df = pd.concat(
        [pd.read_csv(f) for f in marker_files],
        ignore_index=True,
    )
    return with_correlation_id_standard_name(marker_df)


def parse_roctx_markers(
    workload_dir: str,
) -> Tuple[Dict[str, Set[str]], Set[str]]:
    """Parse rocprof-compute marker + counter CSVs under workload_dir.

    Returns (op_to_kernels, marker_ops): marker_ops is every distinct
    marker leaf, op_to_kernels maps a leaf to correlated GPU kernel names
    (leaves without kernels are omitted). Returns ({}, set()) if no marker
    CSV exists.
    """
    marker_df = _read_roctx_marker_dataframe(workload_dir)
    if marker_df is None:
        return {}, set()

    counter_files = sorted(Path(workload_dir).glob("**/*counter_collection.csv"))
    marker_ops, op_to_corr = collect_marker_ops_and_correlations(marker_df)

    op_to_kernels: Dict[str, Set[str]] = {}
    if counter_files and op_to_corr:
        op_to_kernels = kernels_by_marker_leaf(op_to_corr, counter_files)

    return op_to_kernels, marker_ops


# C++ RecordFunction tier signature detection. compare_single_op()
# PASSes on either tier (markers + kernel intersection); these
# sentinels are the only way to prove the C++ tier ran. Source of
# truth: src/lib/roctx_recordfn/roctx_recordfn.cpp:default_leaf_context().


C_TIER_LEAF_CONTEXT_SENTINELS: Tuple[str, ...] = (
    "aten:0",
    "aten.nested:0",
    "autograd.engine:0",
    "autograd.bwd:0",
)

# Subset that proves backward-pass instrumentation specifically.
C_TIER_BACKWARD_SENTINELS: Tuple[str, ...] = (
    "autograd.engine:0",
    "autograd.bwd:0",
)


def cpp_tier_signature_in_markers(
    marker_df: pd.DataFrame,
) -> Tuple[bool, Dict[str, int]]:
    """Scan marker_df["Function"] for "@<sentinel>" tokens. Returns
    (cpp_tier_active, sentinel_counts). The "@" prefix avoids false
    positives against file paths. Missing column -> (False, {}).
    """
    counts: Dict[str, int] = {}
    if marker_df is None or marker_df.empty:
        return False, counts
    func_col = marker_df.get("Function")
    if func_col is None:
        return False, counts

    tokens = tuple(f"@{sentinel}" for sentinel in C_TIER_LEAF_CONTEXT_SENTINELS)
    for cell in func_col:
        if not isinstance(cell, str):
            continue
        for sentinel, token in zip(C_TIER_LEAF_CONTEXT_SENTINELS, tokens):
            n = cell.count(token)
            if n:
                counts[sentinel] = counts.get(sentinel, 0) + n
    return bool(counts), counts


def detect_cpp_tier_signature(
    workload_dir: str,
) -> Tuple[bool, Dict[str, int]]:
    """Workload-dir entry point for cpp_tier_signature_in_markers.
    Returns (False, {}) when no marker CSV exists.
    """
    marker_df = _read_roctx_marker_dataframe(workload_dir)
    if marker_df is None:
        return False, {}
    return cpp_tier_signature_in_markers(marker_df)


def format_cpp_tier_signature_report(
    cpp_tier_active: bool,
    sentinel_counts: Dict[str, int],
) -> Tuple[str, ...]:
    """Render the C++ tier outcome as 1-4 indented stdout lines:
    NONE (Python fallback), FORWARD-ONLY (no autograd in workload),
    or FULL (forward + backward sentinels).
    """
    if not cpp_tier_active:
        return (
            "C++ tier signature: NONE — markers carry no aten:0 / "
            "aten.nested:0 / autograd.engine:0 / autograd.bwd:0 "
            "sentinel. Workload ran on the Python TorchDispatchMode "
            "fallback (degraded backward-thread context propagation).",
        )

    backward_observed = any(
        sentinel_counts.get(name, 0) > 0 for name in C_TIER_BACKWARD_SENTINELS
    )
    forward_observed = any(
        sentinel_counts.get(name, 0) > 0 for name in ("aten:0", "aten.nested:0")
    )
    parts = []
    if forward_observed:
        parts.append(
            f"forward (aten:0={sentinel_counts.get('aten:0', 0)}, "
            f"aten.nested:0={sentinel_counts.get('aten.nested:0', 0)})"
        )
    if backward_observed:
        parts.append(
            f"backward (autograd.engine:0="
            f"{sentinel_counts.get('autograd.engine:0', 0)}, "
            f"autograd.bwd:0={sentinel_counts.get('autograd.bwd:0', 0)})"
        )
    summary = "  ".join(parts) if parts else "(no sentinels counted)"

    if backward_observed:
        return (f"C++ tier signature: FULL — {summary}.",)

    return (
        f"C++ tier signature: FORWARD-ONLY — {summary}.",
        "  No autograd.engine:0 / autograd.bwd:0 sentinels observed; "
        "this is expected for workloads that do no .backward() but "
        "indicates degraded backward propagation if your workload "
        "*should* have triggered autograd.",
    )


# -- Per-operator comparison --


def torch_trace_coverage_color_enabled() -> bool:
    """True when ANSI colors are allowed for coverage report lines (stdout)."""
    if os.environ.get("NO_COLOR", "").strip():
        return False
    force = os.environ.get("FORCE_COLOR", "").strip().lower()
    if force in ("1", "true", "yes"):
        return True
    return sys.stdout.isatty()


def torch_trace_coverage_red(text: str) -> str:
    if not torch_trace_coverage_color_enabled():
        return text
    return f"\033[31m{text}\033[0m"


def coverage_log_pass(op_name: str, *, note: str = "") -> Tuple[str, ...]:
    """Stdout lines for a passing operator (pytest -s).

    note may contain newlines; each line is rendered as a separate
    indented stdout line so multi-line rationales remain readable.
    """
    if not note:
        return (f"PASS  {op_name}",)
    body = note.splitlines() or [note]
    lines = [f"PASS  {op_name}"]
    lines.extend(f"    {ln}" for ln in body)
    return tuple(lines)


def coverage_log_fail(op_name: str, reason: str) -> Tuple[str, ...]:
    """Stdout lines for a failing operator; reason is red when color is allowed."""
    body = reason.splitlines() or [reason]
    first = f"FAIL  {op_name}"
    if len(body) == 1:
        return (first, f"    {torch_trace_coverage_red(body[0])}")
    lines = [first]
    for ln in body:
        lines.append(f"    {torch_trace_coverage_red(ln)}")
    return tuple(lines)


def coverage_log_skip(op_name: str, reason: str) -> Tuple[str, ...]:
    """Stdout lines for a skipped operator."""
    body = reason.splitlines() or [reason]
    lines = [f"SKIP  {op_name}"]
    lines.extend(f"    {ln}" for ln in body)
    return tuple(lines)


def describe_missing_or_errored_op(
    op: OpEntry,
    ground_truth_entry: Optional[Dict[str, Any]],
) -> str:
    """Human-readable SKIP reason: workload error, OpSpec skip, or missing builder."""
    if ground_truth_entry is not None and "error" in ground_truth_entry:
        return f"workload raised at runtime: {ground_truth_entry['error']}"

    if op.category == "structural":
        return "structural op missing from generated workload"

    overload_key = aten_overload_key(op.name)
    short_name = aten_op_short_name(op.name) or op.name.rsplit(".", 1)[-1]
    spec = None
    if overload_key is not None:
        spec = OP_SPECS.get(overload_key)
    if spec is None:
        spec = OP_SPECS.get(short_name)
    if spec is not None and spec.skip:
        return f"OpSpec skip: {spec.skip}"
    if spec is not None and spec.build is not None:
        return "OpSpec builder exists but op was not emitted into the workload"
    return (
        "no OpSpec.build entry in torch_trace_coverage_utils.OP_SPECS; "
        "argument synthesis is restricted to per-operator builders"
    )


# -- Skip-reason categorization --
#
# SKIP outcomes have many distinct underlying causes (missing builder,
# upstream ROCm fault, NVIDIA-only feature, paired-forward dependency,
# missing external library, ...). The end-of-test summary aggregates
# these into a small set of operator-author-facing buckets so the SKIP
# count is actionable rather than opaque.
#
# Categorization is keyword-based: each branch is keyed on a stable
# phrase emitted by describe_missing_or_errored_op or by the literal
# text of an OpSpec.skip reason in OP_SPECS. Adding a new bucket
# requires (1) a new key in SKIP_CATEGORY_LABELS and (2) a matching
# branch in categorize_skip_reason.

SKIP_CATEGORY_LABELS: Dict[str, str] = {
    "argument_builder_gap": (
        "argument-builder coverage gap "
        "(no OpSpec.build registered for this op in OP_SPECS — "
        "register a per-operator builder to enable coverage)"
    ),
    "workload_runtime_error": (
        "workload runtime error (synthetic workload raised before profiling completed)"
    ),
    "upstream_rocm_bug": (
        "upstream ROCm bug (SIGSEGV / HIP illegal memory access under torch.profiler)"
    ),
    "hardware_incompatible_cuda_only": (
        "hardware-incompatible "
        "(CUDA-only feature: CUTLASS / cuDNN / xFormers / cuSPARSELt / "
        "FlashAttention / fp8 / packed int4-int8 / grouped matmul)"
    ),
    "stateful_op_dependency": (
        "stateful-op dependency "
        "(backward op needs state / output from a matched forward run)"
    ),
    "external_library_required": (
        "external library required (e.g. MAGMA) — not present in this build"
    ),
    "incomplete_op_spec": (
        "OpSpec pending (builder declared but shape / argument contract TBD)"
    ),
    "no_kernel_in_ground_truth": (
        "no GPU kernels in ground truth "
        "(torch.profiler reported zero kernels for this op)"
    ),
    "structural_marker_gap": (
        "structural marker gap "
        "(structural pattern not exercised or no ROCTX marker emitted)"
    ),
    "opspec_builder_present_but_not_emitted": (
        "OpSpec builder present but op was not emitted into the workload"
    ),
    "other": "other / uncategorized",
}


def categorize_skip_reason(reason: str) -> str:
    """Return the stable SKIP-category key for an outcome.reason string.

    Branches are ordered: structural lookup paths first (origin-based),
    then OpSpec.skip text matched by keyword. Returns "other" when no
    branch matches — adding a new SKIP source should extend both this
    function and SKIP_CATEGORY_LABELS.
    """
    r = reason.lower()

    if "no opspec.build entry" in r:
        return "argument_builder_gap"
    if "workload raised at runtime" in r:
        return "workload_runtime_error"
    if "no gpu kernels in ground truth" in r:
        return "no_kernel_in_ground_truth"
    if "no roctx marker" in r or "structural op missing" in r:
        return "structural_marker_gap"
    if "opspec builder exists but op was not emitted" in r:
        return "opspec_builder_present_but_not_emitted"

    if any(
        k in r
        for k in (
            "sigsegv",
            "hip 719",
            "illegal memory access",
            "kernel race",
        )
    ):
        return "upstream_rocm_bug"
    # discover_operators filters CUDA-only ops out of the sample using the
    # same helper, so this branch normally only fires if an OpSpec.skip
    # text matches the keyword set but wasn't caught at enumeration time
    # (e.g. a structural label or a future code path that bypasses
    # discover_operators).
    if is_hardware_incompatible_cuda_only(r):
        return "hardware_incompatible_cuda_only"
    if any(
        k in r
        for k in (
            "matching forward",
            "matched forward",
            "matched fwd",
            "offset2bag",
            "save_mean",
            "save_var",
            "reservespace",
            "cdist output",
            "nested-tensor constructor",
            "matching offsets/lengths",
        )
    ):
        return "stateful_op_dependency"
    if "magma" in r or "libmagma" in r:
        return "external_library_required"
    if "shape contract" in r:
        return "incomplete_op_spec"
    return "other"


def _short_op_name(op_name: str) -> str:
    """Strip the torch.ops.aten. prefix for compact breakdown listings."""
    prefix = "torch.ops.aten."
    return op_name[len(prefix) :] if op_name.startswith(prefix) else op_name


# -- argument-builder-gap Pareto (family classifier) -------------------------
#
# The argument_builder_gap SKIP bucket dominates the coverage report because
# OP_SPECS only carries hand-authored short-name builders. The remaining ops
# are not random — they cluster into a small number of structural families,
# each addressable by one shared template builder. missing_builder_family
# returns a stable family key for an op name; format_missing_arg_builder_report
# prints the family Pareto so the developer can attack the largest families
# first.
#
# Family keys are also used as keys into _MISSING_BUILDER_FAMILY_HINTS, which
# records the one-line "fix pattern" for each family. Adding a new family
# requires (1) a new branch in missing_builder_family and (2) a matching
# hint entry; otherwise the report will fall back to the generic "other".


_MISSING_BUILDER_FAMILY_HINTS: Dict[str, str] = {
    "out / output-buffer overload (.out / .grad_input)": (
        "wrap the default-overload builder and pass out=torch.empty_like(...)"
    ),
    "autograd backward (*_backward)": (
        "synthesize grad_output + the forward inputs/outputs the schema needs"
    ),
    "in-place (trailing underscore)": (
        "same args as the out-of-place sibling; just call the in-place variant"
    ),
    "foreach / vectorized list ops": (
        "wrap inputs in [t1, t2, t3] lists; lengths must match across args"
    ),
    "special functions (Bessel / Chebyshev / Hermite / ...)": (
        "single Tensor → Tensor; randn(d, N) works for most "
        "(some need x ∈ [-1, 1] or x > 0)"
    ),
    "linear algebra (linalg_*)": (
        "well-conditioned square matrices (A = randn(N,N) + N*eye(N))"
    ),
    "softmax / log_softmax family": (
        "(input, dim, half_to_float=False) for forward; backward needs "
        "grad_output + output (not input)"
    ),
    "batch normalization": (
        "input (N,C,*) + weight/bias (C,) + running_mean/var (C,) + "
        "training flag; backward also needs save_mean / save_invstd"
    ),
    "layer / group / instance normalization": (
        "input (N,C,*) + normalized_shape + weight/bias matching the normalized dims"
    ),
    "FFT family": ("complex input, or real input with appropriate signal_ndim"),
    "sparse tensor ops": ("explicit indices/values constructor; never randn"),
    "quantization": ("needs scale + zero_point tensors matching the quantized dtype"),
    "distribution / random sampler": (
        "shape arg + optional generator; some take param tensors (mean, std)"
    ),
    "convolution": ("input (N,C,H,W) + weight (C_out,C_in,Kh,Kw) + optional bias"),
    "pooling": ("input (N,C,H,W) + kernel_size + stride/padding"),
    "loss function": ("input + target with matching shapes / dtypes"),
    "indexing / gather / scatter": (
        "source tensor + int64 index tensor; bounds must lie inside source"
    ),
    "cumulative reduction": ("input tensor + dim; dtype kwarg optional"),
    "reduction along dim": ("input tensor + dim (int or list-of-int) + keepdim kwarg"),
    "elementwise — Tensor overload": ("two same-shape tensors"),
    "elementwise — Scalar overload": ("tensor + python scalar"),
    "elementwise — unary float Tensor → Tensor": (
        "single template: lambda d: ([f(d, N, N)], {}) — most accept any "
        "float tensor (a few need x ∈ [-1, 1] or x > 0)"
    ),
    "reduction (whole-tensor)": (
        "single template: lambda d: ([f(d, N, N)], {}) — operates over "
        "all elements (some have .dim variants which fall in the "
        "'reduction along dim' bucket)"
    ),
    "other / uncategorized": ("inspect the schema; no shared template applies"),
}


def missing_builder_family(op_name: str) -> str:
    """Heuristic family bucket for ops missing an OpSpec.build entry.

    Name-based classification (no schema parsing) so the function works
    even when called from offline tooling that doesn't have torch loaded.
    Priority-ordered: more-specific buckets are checked first.
    """
    short = aten_op_short_name(op_name) or op_name.rsplit(".", 1)[-1]
    overload_key = aten_overload_key(op_name)
    suffix = (
        overload_key.split(".", 1)[1] if overload_key and "." in overload_key else ""
    )

    # Highest-priority structural buckets: same builder pattern fits the
    # entire family regardless of which op short name the variant belongs
    # to.
    if suffix == "out" or suffix.endswith("_out") or suffix == "grad_input":
        return "out / output-buffer overload (.out / .grad_input)"
    if short.endswith("_backward"):
        return "autograd backward (*_backward)"
    # Trailing-underscore in-place ops, e.g. add_, sin_, log_normal_.
    # Filter out leading-underscore private ops to avoid mis-bucketing
    # things like _fft_r2c.
    if short.endswith("_") and not short.startswith("_"):
        return "in-place (trailing underscore)"
    if short.startswith("_foreach_"):
        return "foreach / vectorized list ops"

    # Domain-specific families. Match private (_underscored) variants to
    # the same family as their public sibling so e.g. _linalg_det groups
    # with linalg_*.
    if short.startswith("special_"):
        return "special functions (Bessel / Chebyshev / Hermite / ...)"
    if short.startswith("linalg_") or short.startswith("_linalg_"):
        return "linear algebra (linalg_*)"
    if "softmax" in short:
        return "softmax / log_softmax family"
    if "batch_norm" in short:
        return "batch normalization"
    if "layer_norm" in short or "group_norm" in short or "instance_norm" in short:
        return "layer / group / instance normalization"
    if "fft" in short:
        return "FFT family"
    if "sparse" in short:
        return "sparse tensor ops"
    if (
        short.startswith("quantize")
        or "quantized" in short
        or "_pack" in short
        or "_unpack" in short
        or "dequantize" in short
    ):
        return "quantization"
    if any(
        k in short
        for k in (
            "normal",
            "uniform",
            "cauchy",
            "exponential",
            "geometric",
            "bernoulli",
            "multinomial",
            "random",
            "poisson",
        )
    ):
        return "distribution / random sampler"
    if "conv" in short or "convolution" in short:
        return "convolution"
    if "pool" in short:
        return "pooling"
    if short.endswith("_loss") or short.endswith("_loss_backward"):
        return "loss function"
    if (
        short.startswith("index_")
        or short.startswith("_index_")
        or "gather" in short
        or "scatter" in short
    ):
        return "indexing / gather / scatter"
    if (
        short.startswith("cumsum")
        or short.startswith("cumprod")
        or short.startswith("cummax")
        or short.startswith("cummin")
        or short.startswith("logcumsumexp")
    ):
        return "cumulative reduction"
    if suffix in ("dim", "dim_IntList", "dim_int", "dim_values"):
        return "reduction along dim"

    # Generic elementwise overloads — apply only when no domain-family
    # match fired above, so e.g. add.Tensor still buckets here but
    # softmax.int does not.
    if suffix in ("Tensor", "Tensor_Tensor"):
        return "elementwise — Tensor overload"
    if suffix in ("Scalar", "Tensor_Scalar", "Scalar_Tensor"):
        return "elementwise — Scalar overload"

    # Hand-curated short names that are unary float Tensor → Tensor
    # primitives (math, activations, predicates). All share the same
    # one-line template builder: lambda d: ([f(d, N, N)], {}). Added
    # here rather than via heuristics because the names have no
    # exploitable pattern.
    if short in _KNOWN_UNARY_FLOAT_ELEMENTWISE:
        return "elementwise — unary float Tensor → Tensor"
    # Hand-curated whole-tensor reductions (no dim arg) — single
    # template (input tensor) covers all.
    if short in _KNOWN_WHOLE_TENSOR_REDUCTIONS:
        return "reduction (whole-tensor)"
    return "other / uncategorized"


# Curated short-name allowlists for the family classifier. These names
# have no exploitable structural pattern (e.g. cos / sin are plain
# words), but they all accept the same minimal builder signature, so
# bucketing them lets the report flag a single template that knocks
# out the entire group.
_KNOWN_UNARY_FLOAT_ELEMENTWISE: frozenset = frozenset({
    # trig / hyperbolic
    "acos",
    "acosh",
    "asin",
    "asinh",
    "atan",
    "atanh",
    "cos",
    "cosh",
    "sin",
    "sinc",
    "sinh",
    "tan",
    "tanh",
    # exponentials / logs
    "exp",
    "exp2",
    "expm1",
    "log",
    "log10",
    "log1p",
    "log2",
    "logit",
    # roots, powers, reciprocals
    "reciprocal",
    "rsqrt",
    "sqrt",
    "square",
    # rounding / sign
    "ceil",
    "floor",
    "frac",
    "neg",
    "round",
    "sgn",
    "sign",
    "signbit",
    "trunc",
    "positive",
    # gamma / bessel-ish
    "digamma",
    "erf",
    "erfc",
    "erfinv",
    "i0",
    "lgamma",
    "polygamma",
    # complex / angle
    "angle",
    "conj_physical",
    # predicates → BoolTensor
    "isfinite",
    "isinf",
    "isnan",
    "isneginf",
    "isposinf",
    "isreal",
    # angle conversion
    "deg2rad",
    "rad2deg",
    # activations
    "elu",
    "gelu",
    "glu",
    "hardshrink",
    "hardsigmoid",
    "hardswish",
    "hardtanh",
    "leaky_relu",
    "mish",
    "relu",
    "sigmoid",
    "silu",
    "softplus",
    "softshrink",
    "threshold",
    # other
    "nonzero",
})

_KNOWN_WHOLE_TENSOR_REDUCTIONS: frozenset = frozenset({
    "all",
    "any",
    "amax",
    "amin",
    "aminmax",
    "argmax",
    "argmin",
    "max",
    "min",
    "median",
    "mode",
    "nanmedian",
    "prod",
    "sum",
    "mean",
    "nansum",
    "trace",
    "std",
    "var",
    "std_mean",
    "var_mean",
    "norm",
})


def format_missing_arg_builder_report(
    op_names: List[str],
    *,
    max_examples: int = 5,
    include_hints: bool = True,
    indent: str = "    ",
) -> List[str]:
    """Family Pareto for ops missing an OpSpec.build entry.

    Triage workflow: pick the family with the highest count, write one
    template builder that covers every op in it, and watch the
    argument_builder_gap SKIP bucket shrink in the next run.

    Returns an empty list when op_names is empty so callers can drop the
    section entirely. Examples per family are alphabetized and truncated
    to max_examples; the trailing "+ N more" makes the truncation
    auditable.
    """
    if not op_names:
        return []

    by_family: Dict[str, List[str]] = {}
    for n in op_names:
        by_family.setdefault(missing_builder_family(n), []).append(
            _short_op_name(n),
        )

    rows = sorted(by_family.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    width = max(len(str(len(ops))) for _, ops in rows)
    sub_indent = f"{indent}{' ' * width}  "

    lines = [
        f"  argument-builder gap — by family ({len(op_names)} ops, "
        f"target the largest families first):"
    ]
    for family, names in rows:
        sample = sorted(names)[:max_examples]
        suffix = (
            f", … (+{len(names) - max_examples} more)"
            if len(names) > max_examples
            else ""
        )
        lines.append(f"{indent}{len(names):>{width}}  {family}")
        lines.append(f"{sub_indent}e.g. {', '.join(sample)}{suffix}")
        if include_hints:
            hint = _MISSING_BUILDER_FAMILY_HINTS.get(family)
            if hint:
                lines.append(f"{sub_indent}hint: {hint}")
    return lines


def format_skip_breakdown_lines(
    skip_counts: Dict[str, int],
    *,
    skip_op_names: Optional[Dict[str, List[str]]] = None,
    max_names_per_category: int = 12,
    indent: str = "    ",
) -> List[str]:
    """Render a SKIP breakdown table (sorted by count desc, then by label).

    When skip_op_names is provided, each category line is followed by a
    secondary ``ops: …`` line listing the offending operators (short
    form, alphabetized). Lists longer than max_names_per_category are
    truncated with a ``(+N more)`` tail so the section stays bounded
    even when one bucket dominates.

    Returns an empty list when there are zero skips so callers can drop
    the section entirely without a special case. The total line is
    appended last as a defensive cross-check against the loop's
    skipped counter.
    """
    if not skip_counts:
        return []

    rows = sorted(
        skip_counts.items(),
        key=lambda kv: (-kv[1], SKIP_CATEGORY_LABELS.get(kv[0], kv[0])),
    )
    width = max(len(str(count)) for _, count in rows)
    # Secondary "ops:" lines align under the label column, i.e. past
    # the count + the two-space gutter that follows it.
    sub_indent = f"{indent}{' ' * width}  "
    lines = ["  SKIP breakdown:"]
    for category, count in rows:
        label = SKIP_CATEGORY_LABELS.get(category, category)
        lines.append(f"{indent}{count:>{width}}  {label}")
        if skip_op_names:
            names = skip_op_names.get(category) or []
            if names:
                short_names = sorted(_short_op_name(n) for n in names)
                if len(short_names) <= max_names_per_category:
                    listed = ", ".join(short_names)
                    suffix = ""
                else:
                    listed = ", ".join(short_names[:max_names_per_category])
                    suffix = f", … (+{len(short_names) - max_names_per_category} more)"
                lines.append(f"{sub_indent}ops: {listed}{suffix}")
    total = sum(skip_counts.values())
    lines.append(f"{indent}{total:>{width}}  total")
    return lines


def multiline_coverage_failure_warning(
    failure_detail: List[Tuple[str, str]],
    *,
    max_ops: int,
    seed: int,
    sample_budget: int,
) -> str:
    """Bounded multiline text for warnings.warn when stdout is captured."""
    lines = [
        f"{len(failure_detail)} operator(s) failed ROCTX/kernel coverage "
        "(report only; test still passes).",
        f"Re-run with -s: pytest tests/test_torch_trace_coverage.py -m "
        f"torch_trace --coverage-seed={seed} --coverage-n={sample_budget} -s",
        "",
    ]
    shown = failure_detail[:max_ops]
    for name, reason in shown:
        r = reason.replace("\n", " ")[:500]
        lines.append(f"FAIL  {name}")
        lines.append(f"    {r}")
        lines.append("")
    rest = len(failure_detail) - len(shown)
    if rest > 0:
        lines.append(f"… and {rest} more (see pytest -s for full list).")
    return "\n".join(lines)


def compare_single_op(
    op: OpEntry,
    ground_truth: Dict[str, Any],
    roctx_marker_names: Set[str],
    roctx_kernels_map: Dict[str, Set[str]],
    *,
    match_verbose: bool = False,
) -> OpCompareOutcome:
    """Compare one op's profiler JSON entry to parsed ROCTX markers and kernels.

    PASS rule: structural ops need a matching marker leaf only; ATen
    ops with profiler kernels also need a non-empty intersection of
    profiler cuda_kernels and ROCTX-correlated kernels. Marker-to-op
    matching is structural (leaf name); kernel intersection uses the
    torch.profiler ground truth. With match_verbose=True, every
    outcome logs the matched marker leaves and both kernel sets.
    """
    ground_truth_entry = ground_truth.get(op.name)

    if ground_truth_entry is None or "error" in ground_truth_entry:
        err_msg = describe_missing_or_errored_op(op, ground_truth_entry)
        reason = err_msg if len(err_msg) <= 4000 else f"{err_msg[:4000]}…"
        log_lines = coverage_log_skip(op.name, reason)
        if match_verbose:
            log_lines = log_lines + (
                "    [match] no ground-truth entry (op skipped before "
                "marker/kernel comparison)",
            )
        return OpCompareOutcome("skip", reason, log_lines)

    profiler_kernels = ground_truth_entry.get("cuda_kernels", [])
    profiler_kernel_set = set(profiler_kernels)

    matched_marker_leaves: List[str] = []
    roctx_kernels: Set[str] = set()
    for observed_marker_leaf in roctx_marker_names:
        if marker_matches_op(op.name, observed_marker_leaf):
            matched_marker_leaves.append(observed_marker_leaf)
            roctx_kernels |= roctx_kernels_map.get(observed_marker_leaf, set())
    marker_found = bool(matched_marker_leaves)

    def _match_verbose_lines() -> Tuple[str, ...]:
        """Indented per-op match diagnostic, truncated at 8 entries."""
        if not match_verbose:
            return ()

        def _trunc(label: str, items) -> str:
            items_sorted = sorted(items)
            head = items_sorted[:8]
            ellipsis = "…" if len(items_sorted) > 8 else ""
            return f"    [match] {label}: {head}{ellipsis}"

        return (
            _trunc("matched_marker_leaves", matched_marker_leaves),
            _trunc("profiler_kernels", profiler_kernel_set),
            _trunc("roctx_kernels", roctx_kernels),
            _trunc(
                "kernel_overlap",
                profiler_kernel_set & roctx_kernels,
            ),
        )

    verbose_tail = _match_verbose_lines()

    # Structural ops are hierarchical wrappers; inner ATen ops own
    # the kernels. Marker presence == PASS.
    if op.category == "structural":
        if marker_found:
            return OpCompareOutcome(
                "pass",
                "",
                coverage_log_pass(op.name, note="structural: marker present")
                + verbose_tail,
            )
        skip_msg = "no ROCTX marker — inject_roctx instrumentation gap"
        return OpCompareOutcome(
            "skip",
            skip_msg,
            coverage_log_skip(op.name, skip_msg) + verbose_tail,
        )

    if not profiler_kernel_set:
        if marker_found:
            return OpCompareOutcome(
                "pass",
                "",
                coverage_log_pass(
                    op.name,
                    note="marker present; no kernels in ground truth",
                )
                + verbose_tail,
            )
        skip_msg = "no GPU kernels in ground truth"
        return OpCompareOutcome(
            "skip",
            skip_msg,
            coverage_log_skip(op.name, skip_msg) + verbose_tail,
        )

    if marker_found and roctx_kernels:
        # Require >=1 kernel in common with profiler ground truth;
        # roctx kernels alone may belong to unrelated dispatches.
        overlap = profiler_kernel_set & roctx_kernels
        if overlap:
            return OpCompareOutcome(
                "pass",
                "",
                coverage_log_pass(op.name) + verbose_tail,
            )
        reason = (
            "marker found and ROCTX correlated kernels, but none overlap "
            f"torch.profiler cuda_kernels "
            f"(profiler={sorted(profiler_kernel_set)[:6]}"
            f"{'…' if len(profiler_kernel_set) > 6 else ''}, "
            f"roctx={sorted(roctx_kernels)[:6]}"
            f"{'…' if len(roctx_kernels) > 6 else ''})"
        )
        return OpCompareOutcome(
            "fail",
            reason,
            coverage_log_fail(op.name, reason) + verbose_tail,
        )

    if marker_found:
        # Zero-copy view ops issue no kernel launches; ROCTX correctly
        # reports an empty kernel-correlation set while profiler's
        # CPU-span overlap attributes adjacent kernels.
        short_name = aten_op_short_name(op.name)
        if short_name and short_name in KNOWN_KERNEL_FREE_ATEN_OPS:
            note_lines = (
                "kernel-free aten op (zero-copy view).",
                "ROCTX reports zero correlated kernels; profiler "
                "attributes adjacent kernels via CPU-span overlap.",
            )
            return OpCompareOutcome(
                "pass",
                "",
                coverage_log_pass(op.name, note="\n".join(note_lines)) + verbose_tail,
            )
        reason = "marker found but no correlated kernels"
        return OpCompareOutcome(
            "fail",
            reason,
            coverage_log_fail(op.name, reason) + verbose_tail,
        )

    reason = "marker not found"
    return OpCompareOutcome(
        "fail",
        reason,
        coverage_log_fail(op.name, reason) + verbose_tail,
    )


def print_torch_trace_coverage_session_header(
    seed: int,
    sample_budget: int,
    sampled_operator_count: int,
    aten_operator_count: int,
    structural_operator_count: int,
    excluded_operator_count: int = 0,
) -> None:
    """Print seed / sampling summary (pytest -s); warn for default capture.

    excluded_operator_count is the number of CUDA-only ATen ops dropped
    by discover_operators (hardware-incompatible on ROCm). It's printed
    on its own line when non-zero so the exclusion is auditable; the
    "selected from" count is post-exclusion.
    """
    print(
        f"\n  Seed: {seed} | {sampled_operator_count} operators"
        f" selected from {aten_operator_count} CUDA ATen ops"
        f" + {structural_operator_count} structural"
        f" (budget={sample_budget})"
    )
    if excluded_operator_count:
        print(
            f"  Excluded {excluded_operator_count} CUDA-only ATen ops "
            f"(hardware-incompatible: CUTLASS / cuDNN / xFormers / "
            f"cuSPARSELt / FlashAttention / fp8 / packed int4-int8 / "
            f"grouped matmul)"
        )
    print()
    reproduce_cmd = (
        "pytest tests/test_torch_trace_coverage.py -m torch_trace "
        f"--coverage-seed={seed} --coverage-n={sample_budget}"
    )
    warnings.warn(
        f"torch_trace_coverage RNG: seed={seed}, n={sample_budget}. "
        f"Re-run: {reproduce_cmd}",
        UserWarning,
        stacklevel=2,
    )


def signal_name(returncode: int) -> str:
    """SIGSEGV / SIGABRT / ... for negative returncodes; '' otherwise."""
    if returncode >= 0:
        return ""
    try:
        return signal.Signals(-returncode).name
    except ValueError:
        return ""


def describe_subprocess_exit_code(returncode: int) -> str:
    """Human-readable explanation for subprocess returncode (POSIX)."""
    if returncode < 0:
        name = signal_name(returncode)
        name_part = f" ({name})" if name else ""
        return (
            f"Child process terminated by signal {-returncode}{name_part} "
            f"(exit {returncode}): native code fault (GPU kernel, driver, or "
            "profiler hook)."
        )
    return f"Child process exited with status {returncode}."


def op_workload_failure_due_to(
    *,
    timed_out: bool,
    returncode: Optional[int],
) -> str:
    """Short phrase for pytest.fail lead-in."""
    if timed_out:
        return "ground-truth subprocess exceeded 120s timeout"
    assert returncode is not None
    if returncode < 0:
        name = signal_name(returncode)
        name_part = f", {name}" if name else ""
        return (
            f"ground-truth subprocess killed by signal {-returncode}"
            f"{name_part} (exit {returncode})"
        )
    return f"ground-truth subprocess exited with status {returncode}"


def stderr_tail_collapsed(
    stderr: str,
    *,
    max_lines: int = 32,
    max_chars: int = 6000,
) -> str:
    """Return a shorter stderr view: drop leading duplicate spam, collapse repeats."""
    if not stderr or not stderr.strip():
        return "(no stderr)"

    lines = stderr.splitlines()
    collapsed: List[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        run_end = index + 1
        while run_end < len(lines) and lines[run_end] == line:
            run_end += 1
        repeat_count = run_end - index
        if repeat_count > 1:
            collapsed.append(f"{line}  [repeated {repeat_count} times]")
        else:
            collapsed.append(line)
        index = run_end

    if len(collapsed) <= max_lines:
        body_lines = collapsed
    else:
        head_n = min(10, max_lines // 3)
        tail_n = max_lines - head_n - 1
        head = collapsed[:head_n]
        tail = collapsed[-tail_n:]
        omitted = len(collapsed) - head_n - tail_n
        body_lines = head + [f"... ({omitted} line(s) omitted) ..."] + tail
    body = "\n".join(body_lines)
    if len(body) > max_chars:
        body = body[-max_chars:]
        body = f"... (stderr tail, {max_chars} char cap) ...\n{body}"
    return body


def copy_failed_coverage_artifacts_to_cwd(
    workload_script_path: str,
    runner_script_path: str,
) -> str:
    """Copy workload + profiler runner from the failed run into pytest cwd."""
    workload_dest = Path.cwd() / "failed_torch_trace_coverage_workload.py"
    runner_dest = Path.cwd() / "failed_torch_trace_coverage_runner.py"
    notes: List[str] = []
    for src, dest in (
        (workload_script_path, workload_dest),
        (runner_script_path, runner_dest),
    ):
        try:
            shutil.copy2(src, dest)
            notes.append(str(dest.resolve()))
        except OSError as exc:
            return f"\n\nCould not copy {src} to {dest}: {exc}"
    return (
        f"\n\nSaved workload to {notes[0]} and runner to {notes[1]}. "
        f"Re-run: python {runner_dest.name} {workload_dest.name} ground_truth.json"
    )


def extract_device_fault_banner(stderr_text: str) -> str:
    """Summarise device-fault markers emitted by the generated workload.

    Scans stderr_text for DEVICE_DEAD banners and for the most
    recent START without a matching OK or FAIL terminal
    banner; both identify the operator implicated in a context-faulting
    failure. Returns the combined hint, or an empty string when neither
    signal is present.
    """
    if not stderr_text:
        return ""
    dead: List[str] = []
    last_start: Optional[str] = None
    last_terminal_label: Optional[str] = None
    for line in stderr_text.splitlines():
        if "[torch-trace-cov] DEVICE_DEAD" in line:
            dead.append(line.strip())
            continue
        if "[torch-trace-cov] START " in line:
            last_start = line.split("[torch-trace-cov] START ", 1)[1].strip()
            continue
        for tag in ("[torch-trace-cov] OK    ", "[torch-trace-cov] FAIL  "):
            if tag in line:
                last_terminal_label = line.split(tag, 1)[1].split(" ", 1)[0]
                break
    hints: List[str] = []
    if dead:
        hints.append("Device-dead banner(s):")
        hints.extend(f"  {d}" for d in dead)
    if last_start is not None and last_start != last_terminal_label:
        hints.append(f"Last START without matching OK/FAIL: {last_start}")
    return "\n".join(hints)


def run_ground_truth_torch_profiler_subprocess(
    runner_script_path: str,
    workload_script_path: str,
    ground_truth_json_path: str,
    *,
    coverage_seed: Optional[int] = None,
    coverage_sample_budget: Optional[int] = None,
) -> None:
    """Run coverage_ground_truth_runner.py (torch.profiler + JSON write)."""
    repro = ""
    if coverage_seed is not None and coverage_sample_budget is not None:
        repro = f" (seed={coverage_seed}, n={coverage_sample_budget})"
    argv = [
        sys.executable,
        str(Path(runner_script_path).resolve()),
        str(Path(workload_script_path).resolve()),
        str(Path(ground_truth_json_path).resolve()),
    ]
    try:
        completed = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired as exc:
        copy_note = copy_failed_coverage_artifacts_to_cwd(
            workload_script_path,
            runner_script_path,
        )
        due = op_workload_failure_due_to(timed_out=True, returncode=None)
        out_tail = stderr_tail_collapsed(exc.stdout or "", max_lines=8)
        err_tail = stderr_tail_collapsed(exc.stderr or "")
        fault_hint = extract_device_fault_banner(exc.stderr or "")
        hint_block = f"--- device fault ---\n{fault_hint}\n\n" if fault_hint else ""
        pytest.fail(
            f"Op workload failed: {due}.{repro}{copy_note}\n\n"
            f"{hint_block}"
            f"--- stdout (tail) ---\n{out_tail}\n\n"
            f"--- stderr (tail) ---\n{err_tail}"
        )
    if completed.returncode != 0:
        copy_note = copy_failed_coverage_artifacts_to_cwd(
            workload_script_path,
            runner_script_path,
        )
        due = op_workload_failure_due_to(
            timed_out=False,
            returncode=completed.returncode,
        )
        exit_expl = describe_subprocess_exit_code(completed.returncode)
        err_tail = stderr_tail_collapsed(completed.stderr or "")
        out_tail = stderr_tail_collapsed(completed.stdout or "", max_lines=12)
        cmdline = subprocess.list2cmdline(argv)
        fault_hint = extract_device_fault_banner(completed.stderr or "")
        hint_block = f"--- device fault ---\n{fault_hint}\n\n" if fault_hint else ""
        pytest.fail(
            f"Op workload failed: {due}.{repro}{copy_note}\n\n"
            f"{exit_expl}\n\n"
            f"{hint_block}"
            f"--- command ---\n{cmdline}\n\n"
            f"--- stdout (tail) ---\n{out_tail}\n\n"
            f"--- stderr (tail) ---\n{err_tail}"
        )
