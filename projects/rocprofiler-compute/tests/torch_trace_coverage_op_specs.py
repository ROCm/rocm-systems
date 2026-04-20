# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Per-operator argument builders for :mod:`test_torch_trace_coverage`.

Single source of truth for ATen operators that appear in the generated
``coverage_workload.py``. Keys of :data:`OP_SPECS` are ATen short names
(last segment of ``torch.ops.aten.<name>``) or overload-specific keys
(``<name>.<overload>``). Each value is an :class:`OpSpec` with exactly
one of:

- ``build``: a ``(device) -> (args_list, kwargs_dict)`` callable returning
  the positional arguments and keyword arguments for the call. An entry
  is required for every ATen operator that must appear in the workload;
  :func:`build_args_for_op` reports any ATen operator without an entry as
  SKIP.
- ``skip``: a short human-readable reason string. The operator is dropped
  from the workload and reported as SKIP.

Builder return values are consumed by ``serialize_arg`` in the test
module, which reads each tensor's ``shape`` and ``dtype`` and emits a
matching factory call into the generated workload. Dtype is preserved
end-to-end, so builders that need int32 pivots, complex FFT inputs, fp16
softmax, bool masks, or int8 operands can return tensors of that dtype
directly.

:class:`_CoverageTensorArg` is available for tensors that require a
value *distribution* the default ``randn`` factory cannot produce
(Bernoulli probabilities in ``[0, 1)``, Poisson rates ``>= 0``, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple

import torch


class _CoverageTensorArg(NamedTuple):
    """Shape-only tensor slot for workload emission when ``randn`` is invalid.

    The emitter normally preserves shape and dtype only, so builders that make
    tensors with *structural* invariants (positive-definite, 1-based
    permutation, monotonic offsets, ...) would lose them on serialization.
    This node tells ``serialize_arg`` to emit a specific factory expression:

    ==================  ======================================================
    ``emit``            Factory expression in ``coverage_workload.py``
    ==================  ======================================================
    ``rand``            ``torch.rand(shape, device=device)``
                        — uniform in ``[0, 1)`` (Bernoulli ``p``, dropout rate).
    ``rand_uniform``    ``torch.rand(...) * scale``
                        — non-negative reals (Poisson rates).
    ``spd``             ``a = torch.randn(n, n, ...); a @ a.mT + n * eye(n)``
                        — symmetric positive-definite (``cholesky``).
                        ``shape`` must be ``(n, n)``.
    ``pivots_1based``   ``torch.arange(1, n + 1, dtype=int32, device=device)``
                        — identity permutation as 1-based LU pivots
                        (``lu_unpack``, ``linalg_lu_solve``).
    ``cumsum_offsets``  ``torch.tensor([0, k, 2k, …, shape[0]*k], int64)``
                        — monotonic offsets for nested / jagged tensors.
                        ``scale`` is the per-row segment length ``k``.
    ``bool_all_true``   ``torch.ones(shape, dtype=bool, device=device)``
                        — all-``True`` mask required by ``_assert_async``
                        (any ``False`` element trips the device-side
                        ``__trap`` and aborts the process).
    ==================  ======================================================
    """

    shape: Tuple[int, ...]
    emit: str
    scale: float = 4.0


ArgBuilder = Callable[[str], Tuple[List[Any], Dict[str, Any]]]


@dataclass(frozen=True)
class OpSpec:
    """Declarative argument builder or skip directive for one ATen operator.

    Exactly one of ``build`` / ``skip`` is expected to be set. Entries
    with ``build`` contribute the operator to the generated workload;
    entries with ``skip`` drop the operator with an explanatory reason.
    """

    build: Optional[ArgBuilder] = None
    skip: Optional[str] = None


# -----------------------------------------------------------------------------
# Compact tensor factories used by builders
# -----------------------------------------------------------------------------

def _f(device: str, *shape: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Float tensor from ``torch.randn``."""
    return torch.randn(shape, device=device, dtype=dtype)


def _i(device: str, *shape: int, low: int = 0, high: int = 8) -> torch.Tensor:
    """int64 tensor from ``torch.randint``."""
    return torch.randint(low, high, shape, device=device, dtype=torch.int64)


def _i1(device: str, n: int = 8) -> torch.Tensor:
    """1-D int64 indices tensor in ``[0, 4)``."""
    return torch.randint(0, 4, (n,), device=device, dtype=torch.int64)


def _b(device: str, shape: Tuple[int, ...] = (4, 4)) -> torch.Tensor:
    """Boolean mask tensor of all ones."""
    return torch.ones(shape, device=device, dtype=torch.bool)


def _r01(device: str, *shape: int) -> torch.Tensor:
    """Float tensor in ``[0, 1)`` from ``torch.rand``."""
    return torch.rand(shape, device=device)


def _spd(device: str, n: int = 4) -> torch.Tensor:
    """Symmetric positive-definite matrix for Cholesky-family ops."""
    a = torch.randn(n, n, device=device)
    return a @ a.mT + torch.eye(n, device=device) * 0.1


# -----------------------------------------------------------------------------
# Multi-line builder helpers (kept out of the table for readability)
# -----------------------------------------------------------------------------

def _linalg_householder_product(device: str) -> Tuple[List[Any], Dict[str, Any]]:
    """Build ``(input, tau)`` for ``linalg_householder_product`` across torch builds.

    Some ROCm wheels omit ``torch.linalg.geqrf``; fall back to legacy
    ``torch.geqrf``, then to CPU ``geqrf`` + ``.to(device)``. Last resort:
    tensors with valid ranks only (may be numerically weaker).
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


def _lu_unpack(device: str) -> Tuple[List[Any], Dict[str, Any]]:
    # Schema: (LU_data, LU_pivots, unpack_data=True, unpack_pivots=True).
    # ``LU_pivots`` is used by the kernel as row-swap indices, so its *values*
    # matter, not just shape/dtype.  ``_CoverageTensorArg(emit="pivots_1based")``
    # tells the emitter to produce ``torch.arange(1, n+1, dtype=int32)`` — a
    # valid identity permutation.  Plain ``randint`` indices would walk off
    # the row table and trigger HIP 719.
    return [
        torch.randn(4, 4, device=device),
        _CoverageTensorArg(shape=(4,), emit="pivots_1based"),
        True,
        True,
    ], {}


def _lu_solve(device: str) -> Tuple[List[Any], Dict[str, Any]]:
    # Schema: (LU, pivots, B, *, left=True, adjoint=False).  See ``_lu_unpack``
    # comment for why ``pivots`` uses the ``pivots_1based`` emit mode.
    return [
        torch.randn(4, 4, device=device),
        _CoverageTensorArg(shape=(4,), emit="pivots_1based"),
        torch.randn(4, 2, device=device),
    ], {}


def _ldl_solve(device: str) -> Tuple[List[Any], Dict[str, Any]]:
    ld, pivots, _ = torch.linalg.ldl_factor_ex(_spd(device))
    return [ld, pivots.to(dtype=torch.int64), torch.randn(4, 2, device=device)], {}


def _solve_triangular(device: str) -> Tuple[List[Any], Dict[str, Any]]:
    # Schema: linalg_solve_triangular(self=A, B, *, upper, left=True,
    # unitriangular=False).  ``upper`` is kwarg-only, and the first positional
    # is ``A`` (triangular), not ``B``.
    a = torch.randn(4, 4, device=device).tril()
    b = torch.randn(4, 2, device=device)
    return [a, b], {"upper": False, "left": True, "unitriangular": False}


def _ormqr(device: str) -> Tuple[List[Any], Dict[str, Any]]:
    # Schema: (self[m,k], tau[k], other[m,n], left=True, transpose=False).
    # With ``left=True`` the constraint is ``other.shape[-2] == self.shape[-2]``
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


def _convolution_backward(device: str) -> Tuple[List[Any], Dict[str, Any]]:
    # Schema (11 positionals):
    #   grad_output, input, weight, bias_sizes?, stride, padding, dilation,
    #   transposed, output_padding, groups, output_mask[3]
    return (
        [
            torch.randn(1, 1, 6, 6, device=device),   # grad_output (N,C_out,H,W)
            torch.randn(1, 1, 8, 8, device=device),   # input
            torch.randn(1, 1, 3, 3, device=device),   # weight
            None,                                      # bias_sizes (no bias)
            [1, 1],                                    # stride
            [0, 0],                                    # padding
            [1, 1],                                    # dilation
            False,                                     # transposed
            [0, 0],                                    # output_padding
            1,                                         # groups
            [True, True, False],                       # output_mask
        ],
        {},
    )


def _batch_norm_backward_common(device: str) -> Tuple[List[Any], Dict[str, Any]]:
    # Schema: (grad_out, input, weight?, running_mean?, running_var?,
    #          save_mean?, save_invstd?, train, eps, output_mask[3]).
    return (
        [
            torch.randn(2, 3, 4, 4, device=device),  # grad_out
            torch.randn(2, 3, 4, 4, device=device),  # input
            torch.ones(3, device=device),            # weight
            torch.zeros(3, device=device),           # running_mean
            torch.ones(3, device=device),            # running_var
            torch.zeros(3, device=device),           # save_mean
            torch.ones(3, device=device),            # save_invstd
            True,                                    # train
            1e-5,                                    # eps
            [True, True, True],                      # output_mask
        ],
        {},
    )


def _native_group_norm_backward(device: str) -> Tuple[List[Any], Dict[str, Any]]:
    # Schema: (grad_out, input, mean, rstd, weight?, N, C, HxW, group,
    #          output_mask[3]). mean/rstd shape is (N, groups).
    return (
        [
            torch.randn(2, 6, 4, 4, device=device),  # grad_out
            torch.randn(2, 6, 4, 4, device=device),  # input
            torch.zeros(2, 2, device=device),        # mean (N, groups)
            torch.ones(2, 2, device=device),         # rstd (N, groups)
            torch.ones(6, device=device),            # weight
            2,                                        # N
            6,                                        # C
            16,                                       # HxW
            2,                                        # group
            [True, True, True],                      # output_mask
        ],
        {},
    )


def _fused_adam_args(device: str) -> Tuple[List[Any], Dict[str, Any]]:
    """Build list-of-tensors positionals for ``_fused_adam_`` / ``_fused_adamw_``.

    Schema: ``(Tensor[] self, Tensor[] grads, Tensor[] exp_avgs,
    Tensor[] exp_avg_sqs, Tensor[] max_exp_avg_sqs, Tensor[] state_steps, *,
    float lr, ...)``. All list lengths must match; ``max_exp_avg_sqs`` is empty
    when ``amsgrad=False``. ``state_steps`` elements must be scalar float
    tensors. Remaining kwargs use schema defaults.
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


def _native_layer_norm_backward(device: str) -> Tuple[List[Any], Dict[str, Any]]:
    # Schema: (grad_out, input, normalized_shape, mean, rstd, weight?, bias?,
    #          output_mask[3]). mean/rstd broadcast across non-normalized dims.
    return (
        [
            torch.randn(2, 4, 4, device=device),     # grad_out
            torch.randn(2, 4, 4, device=device),     # input
            [4],                                     # normalized_shape
            torch.randn(2, 4, 1, device=device),     # mean
            torch.randn(2, 4, 1, device=device),     # rstd
            torch.ones(4, device=device),            # weight
            torch.zeros(4, device=device),           # bias
            [True, True, True],                      # output_mask
        ],
        {},
    )


# -----------------------------------------------------------------------------
# The unified table.  Keys are ATen short names.
# -----------------------------------------------------------------------------
#
# Skip entries appear first, followed by hardcoded-argument builders
# grouped by operator category. Adding a new operator to the coverage
# workload requires exactly one entry in this table (``OpSpec.build`` or
# ``OpSpec.skip``); removing an entry causes :func:`build_args_for_op` to
# report the operator as SKIP.

OP_SPECS: Dict[str, OpSpec] = {
    # ---------------------------------------------------------------
    # Skipped entirely: the op advertises CUDA dispatch but faults at
    # runtime under our synthetic workload (or specifically under
    # ``torch.profiler``).
    # ---------------------------------------------------------------
    "_sparse_semi_structured_mm": OpSpec(
        skip="Kernel ``sparse_semi_structured_mm`` is CUTLASS-only (2:4 NVIDIA "
        "sparse tensor cores); not built for the ROCm stack.",
    ),
    # The FFT family segfaults the ground-truth subprocess whenever
    # ``torch.profiler`` is active: the ROCm FFT kernel and ROCTracer's
    # flow-event hooks race, and the operator aborts before ROCTracer can
    # emit a completion event. The kernels themselves run correctly
    # outside the profiler. Because the test requires ``torch.profiler``
    # to collect ground truth, these operators are skipped entirely;
    # coverage can only be restored by a driver / profiler update that
    # resolves the race.
    "_fft_r2c": OpSpec(
        skip="torch.profiler cannot collect ground truth: ROCm FFT kernel + "
        "ROCTracer race produces SIGSEGV inside the profiler subprocess "
        "(op itself runs fine outside the profiler).",
    ),
    "_fft_c2c": OpSpec(
        skip="torch.profiler cannot collect ground truth: ROCm FFT kernel + "
        "ROCTracer race produces SIGSEGV inside the profiler subprocess "
        "(op itself runs fine outside the profiler).",
    ),
    "_fft_c2r": OpSpec(
        skip="torch.profiler cannot collect ground truth: ROCm FFT kernel + "
        "ROCTracer race produces SIGSEGV inside the profiler subprocess "
        "(op itself runs fine outside the profiler).",
    ),
    "_cudnn_rnn": OpSpec(
        skip="Requires the cuDNN library (``libcudnn``) — NVIDIA-only.  ROCm "
        "provides the equivalent RNN path through MIOpen (``miopen_rnn``), "
        "which is a different ATen op.  Cannot be enabled by ``import``: "
        "PyTorch must be built against cuDNN for this op to be callable.",
    ),
    "_cdist_backward": OpSpec(
        skip="Requires cdist output tied to x1/x2; serialized workload cannot "
        "preserve that correlation.",
    ),
    # ``igamma`` / ``igammac`` (and their ``.out`` overloads) trigger a
    # kernel-side illegal memory access on ROCm (HIP error 719,
    # ``hipErrorLaunchFailure``) that places the HIP context in a
    # sticky-error state.
    "igamma": OpSpec(
        skip="ROCm kernel: HIP error 719 (unspecified launch failure).",
    ),
    "igammac": OpSpec(
        skip="ROCm kernel: HIP error 719 (unspecified launch failure).",
    ),
    "cudnn_batch_norm_backward": OpSpec(
        skip="Needs save_mean / save_var / reserveSpace from a matching forward.",
    ),
    "cudnn_batch_norm": OpSpec(
        skip="Kernel ``cudnnBatchNormalizationForwardTraining`` lives in cuDNN; "
        "ATen is not compiled with cuDNN on ROCm builds (MIOpen is used "
        "instead, under ``miopen_batch_norm``).",
    ),
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
    # ``_fused_moving_avg_obs_fq_helper(self, observer_on, fake_quant_on,
    # running_min, running_max, scale, zero_point, averaging_const,
    # quant_min, quant_max, ch_axis, per_row_fake_quant=False,
    # symmetric_quant=False) -> (out, mask)``. The operator updates the
    # running min/max tensors in place based on ``self``, so a one-shot
    # call only requires shape-correct buffers — the emitter's random
    # values satisfy the kernel (``observer_on`` / ``fake_quant_on`` are
    # treated truthily; ``running_min`` / ``running_max`` need not be
    # ordered on the first call). ``observer_on`` / ``fake_quant_on`` /
    # ``zero_point`` must be ``int32`` (the operator rejects ``int64``);
    # ``.to(dtype=torch.int32)`` pins the dtype so the emitter preserves it.
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
    # Nested-tensor APIs take ``offsets: Tensor[]`` — a Python list of int64
    # tensors.  Each offsets tensor must be monotonic and end at
    # ``values.numel()`` along the ragged dim, otherwise the packer reads
    # past the values buffer.  ``serialize_arg`` now dispatches through each
    # list element (so ``[_CoverageTensorArg(emit="cumsum_offsets")]`` emits
    # correctly), but the builders still need ``values.shape[0]`` /
    # ``max_lengths`` / ``offsets[-1]`` to agree, which is op-specific and
    # not yet worked out; leaving them skipped until a coherent shape contract
    # is added.
    "_jagged_to_padded_dense_forward": OpSpec(
        skip="Nested-tensor API: needs a list of monotonic int64 offsets "
        "tensors whose tail equals ``values.numel`` along the ragged dim; "
        "emitter supports the list-of-_CoverageTensorArg form, but a "
        "consistent shape contract (values / offsets / max_lengths) is TBD.",
    ),
    "_padded_dense_to_jagged_forward": OpSpec(
        skip="Nested-tensor API: needs a list of monotonic int64 offsets "
        "tensors whose tail equals ``values.numel`` along the ragged dim; "
        "emitter supports the list-of-_CoverageTensorArg form, but a "
        "consistent shape contract (values / offsets / max_lengths) is TBD.",
    ),

    # ---------------------------------------------------------------
    # matmul / batched matmul / conv
    # ---------------------------------------------------------------
    "bmm": OpSpec(build=lambda d: ([_f(d, 2, 4, 4), _f(d, 2, 4, 4)], {})),
    "conv1d": OpSpec(build=lambda d: ([_f(d, 1, 3, 16), _f(d, 6, 3, 3)], {})),
    "conv2d": OpSpec(build=lambda d: ([_f(d, 1, 3, 8, 8), _f(d, 6, 3, 3, 3)], {})),
    "conv3d": OpSpec(
        build=lambda d: ([_f(d, 1, 3, 4, 4, 4), _f(d, 6, 3, 3, 3, 3)], {}),
    ),
    "conv_transpose1d": OpSpec(
        build=lambda d: ([_f(d, 1, 3, 16), _f(d, 3, 6, 3)], {}),
    ),
    "conv_transpose2d": OpSpec(
        build=lambda d: ([_f(d, 1, 3, 8, 8), _f(d, 3, 6, 3, 3)], {}),
    ),
    "conv_transpose3d": OpSpec(
        build=lambda d: ([_f(d, 1, 3, 4, 4, 4), _f(d, 3, 6, 3, 3, 3)], {}),
    ),
    "addmm": OpSpec(build=lambda d: ([_f(d, 4), _f(d, 4, 4), _f(d, 4, 4)], {})),
    "addbmm": OpSpec(
        build=lambda d: ([_f(d, 4, 4), _f(d, 2, 4, 4), _f(d, 2, 4, 4)], {}),
    ),
    "addbmm_": OpSpec(
        build=lambda d: ([_f(d, 4, 4), _f(d, 2, 4, 4), _f(d, 2, 4, 4)], {}),
    ),
    "baddbmm": OpSpec(
        build=lambda d: ([_f(d, 2, 4, 4), _f(d, 2, 4, 4), _f(d, 2, 4, 4)], {}),
    ),
    "baddbmm_": OpSpec(
        build=lambda d: ([_f(d, 2, 4, 4), _f(d, 2, 4, 4), _f(d, 2, 4, 4)], {}),
    ),
    "addmv": OpSpec(build=lambda d: ([_f(d, 4), _f(d, 4, 4), _f(d, 4)], {})),
    "addmv_": OpSpec(build=lambda d: ([_f(d, 4), _f(d, 4, 4), _f(d, 4)], {})),
    "addr": OpSpec(build=lambda d: ([_f(d, 4, 4), _f(d, 4), _f(d, 4)], {})),
    "dot": OpSpec(build=lambda d: ([_f(d, 4), _f(d, 4)], {})),
    "vdot": OpSpec(build=lambda d: ([_f(d, 4), _f(d, 4)], {})),

    # ---------------------------------------------------------------
    # embedding / embedding_bag
    # ---------------------------------------------------------------
    "embedding": OpSpec(
        build=lambda d: ([_f(d, 10, 8), torch.randint(0, 10, (4,), device=d)], {}),
    ),
    # grad_output (*, embed_dim), indices must be Long; num_weights > max index.
    "embedding_dense_backward": OpSpec(
        build=lambda d: (
            [_f(d, 8, 16), torch.randint(0, 32, (8,), device=d), 32, -1, False],
            {},
        ),
    ),
    # Schema: (weight, indices, offsets, scale_grad_by_freq=False, mode=0,
    #          sparse=False, per_sample_weights=None, include_last_offset=False,
    #          padding_idx=-1). ``offsets`` must be int64 (not float) and
    #          describe the bag boundaries into ``indices``.
    "_embedding_bag": OpSpec(
        build=lambda d: (
            [
                _f(d, 10, 8),
                _i1(d, 6),
                torch.tensor([0, 3], device=d, dtype=torch.int64),
            ],
            {},
        ),
    ),
    "_embedding_bag_forward_only": OpSpec(
        build=lambda d: (
            [
                _f(d, 10, 8),
                _i1(d, 6),
                torch.tensor([0, 3], device=d, dtype=torch.int64),
                False, 0, False,
            ],
            {},
        ),
    ),
    # Backward variants require matching ``offset2bag`` / ``bag_size`` /
    # ``maximum_indices`` tensors produced by the forward; faking them reliably
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
    "linalg_householder_product": OpSpec(build=_linalg_householder_product),
    # ``cholesky`` raises _LinAlgError on non-PD input. ``_CoverageTensorArg(
    # emit="spd")`` tells the emitter to produce ``a @ a.mT + n * I`` —
    # diagonal-dominant, always SPD. The build-time ``_spd`` helper was not
    # enough because ``serialize_arg`` only reads shape+dtype and would drop
    # the SPD structure. ``linalg_cholesky_ex`` / ``cholesky_inverse`` do not
    # raise, but they still need an SPD input for meaningful kernels.
    "cholesky": OpSpec(
        build=lambda d: ([_CoverageTensorArg(shape=(4, 4), emit="spd")], {}),
    ),
    "linalg_cholesky_ex": OpSpec(
        build=lambda d: ([_CoverageTensorArg(shape=(4, 4), emit="spd")], {}),
    ),
    "cholesky_inverse": OpSpec(
        build=lambda d: ([_CoverageTensorArg(shape=(4, 4), emit="spd")], {}),
    ),
    # Schema: _cholesky_solve_helper(self, A, bool upper).
    "_cholesky_solve_helper": OpSpec(
        build=lambda d: ([_f(d, 4, 2), _spd(d).tril(), False], {}),
    ),
    "linalg_cross": OpSpec(build=lambda d: ([_f(d, 2, 3), _f(d, 2, 3)], {})),
    # ldl_solve on CUDA/ROCm requires PyTorch built with MAGMA/LAPACK support;
    # not available in common ROCm builds.
    "linalg_ldl_solve": OpSpec(
        skip="Requires the MAGMA library (``libmagma``) for the GPU path.  On "
        "ROCm the equivalent would be rocSOLVER, but PyTorch's LDL solve "
        "currently only dispatches to MAGMA.  Cannot be enabled by "
        "``import``: PyTorch must be built against MAGMA for this op to be "
        "callable on GPU.",
    ),
    # lu_unpack / linalg_lu_solve feed ``LU_pivots`` to the kernel as
    # row-swap indices, so the *values* must form a valid 1-based
    # permutation (otherwise the kernel reads past the end of the row
    # table → HIP 719).  ``_lu_unpack`` / ``_lu_solve`` use
    # ``_CoverageTensorArg(emit="pivots_1based")`` which serializes to
    # ``torch.arange(1, n+1, dtype=int32)`` — a safe identity permutation.
    "linalg_lu_solve": OpSpec(build=_lu_solve),
    "linalg_solve_triangular": OpSpec(build=_solve_triangular),
    "lu_unpack": OpSpec(build=_lu_unpack),
    "ormqr": OpSpec(build=_ormqr),

    # ---------------------------------------------------------------
    # Loss functions
    # ---------------------------------------------------------------
    "cross_entropy_loss": OpSpec(
        build=lambda d: ([_f(d, 4, 10), torch.randint(0, 10, (4,), device=d)], {}),
    ),
    # ``input`` / ``target`` in [0, 1]; ``randn`` has triggered ROCm HSA
    # exceptions in binary_cross_entropy*_cuda.
    "binary_cross_entropy": OpSpec(
        build=lambda _d: (
            [_CoverageTensorArg((4, 4), "rand"), _CoverageTensorArg((4, 4), "rand")],
            {},
        ),
    ),
    "binary_cross_entropy_backward": OpSpec(
        build=lambda d: (
            [
                _f(d, 4, 4),
                _CoverageTensorArg((4, 4), "rand"),
                _CoverageTensorArg((4, 4), "rand"),
            ],
            {},
        ),
    ),
    "nll_loss_forward": OpSpec(
        build=lambda d: (
            [
                _f(d, 4, 10).log_softmax(1),
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
                torch.tensor(1.0, device=d),                   # grad_output
                _f(d, 4, 5).log_softmax(1),                    # self
                torch.randint(0, 5, (4,), device=d),           # target
                None,                                           # weight
                1,                                              # reduction (Mean)
                -100,                                           # ignore_index
                torch.tensor(4.0, device=d),                   # total_weight
            ],
            {},
        ),
    ),
    "nll_loss2d_forward": OpSpec(
        build=lambda d: (
            [
                _f(d, 2, 5, 4, 4).log_softmax(1),
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
                torch.tensor(1.0, device=d),                   # grad_output
                _f(d, 2, 5, 4, 4).log_softmax(1),              # self
                torch.randint(0, 5, (2, 4, 4), device=d),      # target
                None,                                           # weight
                1,                                              # reduction (Mean)
                -100,                                           # ignore_index
                torch.tensor(32.0, device=d),                  # total_weight
            ],
            {},
        ),
    ),
    # Schema: multi_margin_loss(self, target, p=1, margin=1, weight?, reduction=Mean).
    # ``weight`` is Tensor? — passing ``0`` trips the dispatcher; use ``None``
    # to fall back to unit weights.
    "multi_margin_loss": OpSpec(
        build=lambda d: (
            [
                _f(d, 4, 5),
                torch.randint(0, 5, (4,), device=d),
                1.0, 1.0, None,
            ],
            {},
        ),
    ),
    "multilabel_margin_loss_forward": OpSpec(
        build=lambda d: (
            [_f(d, 4, 5), torch.randint(0, 5, (4, 5), device=d), 0],
            {},
        ),
    ),

    # ---------------------------------------------------------------
    # Norm / batch / group / layer
    # ---------------------------------------------------------------
    "batch_norm": OpSpec(
        build=lambda d: (
            [_f(d, 2, 3, 4, 4), _f(d, 3), _f(d, 3), _f(d, 3), _f(d, 3),
             True, 0.1, 1e-5, False],
            {},
        ),
    ),
    "native_batch_norm": OpSpec(
        build=lambda d: (
            [_f(d, 2, 3, 4, 4), _f(d, 3), _f(d, 3), _f(d, 3), _f(d, 3),
             True, 0.1, 1e-5],
            {},
        ),
    ),
    "miopen_batch_norm": OpSpec(
        build=lambda d: (
            [
                _f(d, 2, 3, 4, 4),
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
                _f(d, 2, 3, 4, 4),                 # input
                _f(d, 2, 3, 4, 4),                 # grad_output
                torch.ones(3, device=d),           # weight
                torch.zeros(3, device=d),          # running_mean
                torch.ones(3, device=d),           # running_var
                torch.zeros(3, device=d),          # save_mean
                torch.ones(3, device=d),           # save_var
                1e-5,                               # epsilon
            ],
            {},
        ),
    ),
    "_batch_norm_with_update": OpSpec(
        build=lambda d: (
            [
                _f(d, 2, 3, 4, 4),
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
                _f(d, 2, 6, 4, 4),
                torch.ones(6, device=d),
                torch.zeros(6, device=d),
                2, 6, 16, 2, 1e-5,
            ],
            {},
        ),
    ),
    "native_layer_norm": OpSpec(
        build=lambda d: (
            [
                _f(d, 2, 4, 4),
                [4],
                torch.ones(4, device=d),
                torch.zeros(4, device=d),
                1e-5,
            ],
            {},
        ),
    ),
    "_fused_rms_norm": OpSpec(
        build=lambda d: ([_f(d, 2, 4, 4), [4], torch.ones(4, device=d), 1e-5], {}),
    ),
    # Schema: (grad_out, input, normalized_shape, rstd, weight?, bool[2] output_mask).
    "_fused_rms_norm_backward": OpSpec(
        build=lambda d: (
            [
                _f(d, 2, 4, 4),                   # grad_out
                _f(d, 2, 4, 4),                   # input
                [4],                               # normalized_shape
                torch.ones(2, 4, 1, device=d),    # rstd
                torch.ones(4, device=d),          # weight
                [True, True],                      # output_mask
            ],
            {},
        ),
    ),
    "batch_norm_backward": OpSpec(build=_batch_norm_backward_common),
    "native_batch_norm_backward": OpSpec(build=_batch_norm_backward_common),
    "native_group_norm_backward": OpSpec(build=_native_group_norm_backward),
    "native_layer_norm_backward": OpSpec(build=_native_layer_norm_backward),

    # ---------------------------------------------------------------
    # Pooling + unpooling (strict shape requirements)
    # ---------------------------------------------------------------
    "_adaptive_avg_pool2d": OpSpec(
        build=lambda d: ([_f(d, 1, 3, 8, 8), [4, 4]], {}),
    ),
    "_adaptive_avg_pool2d_backward": OpSpec(
        build=lambda d: ([_f(d, 1, 3, 4, 4), _f(d, 1, 3, 8, 8)], {}),
    ),
    "_adaptive_avg_pool3d": OpSpec(
        build=lambda d: ([_f(d, 1, 1, 8, 8, 8), [4, 4, 4]], {}),
    ),
    "_adaptive_avg_pool3d_backward": OpSpec(
        build=lambda d: ([_f(d, 1, 1, 4, 4, 4), _f(d, 1, 1, 8, 8, 8)], {}),
    ),
    "adaptive_max_pool2d": OpSpec(build=lambda d: ([_f(d, 1, 3, 8, 8), [2, 2]], {})),
    # adaptive_max_pool2d_backward: grad_output and indices must match the
    # pool-output shape (N, C, out_h, out_w); self carries the input shape.
    "adaptive_max_pool2d_backward": OpSpec(
        build=lambda d: (
            [_f(d, 1, 3, 2, 2), _f(d, 1, 3, 8, 8), _i(d, 1, 3, 2, 2)],
            {},
        ),
    ),
    "adaptive_max_pool3d": OpSpec(
        build=lambda d: ([_f(d, 1, 2, 6, 6, 6), [2, 2, 2]], {}),
    ),
    "adaptive_max_pool3d_backward": OpSpec(
        build=lambda d: (
            [_f(d, 1, 2, 2, 2, 2), _f(d, 1, 2, 6, 6, 6), _i(d, 1, 2, 2, 2, 2)],
            {},
        ),
    ),
    "avg_pool2d": OpSpec(build=lambda d: ([_f(d, 1, 1, 8, 8), [2, 2]], {})),
    # Schema takes 8 positionals; ``divisor_override`` is int? and must be
    # supplied (as None) because it is positional, not kwarg-only.
    "avg_pool2d_backward": OpSpec(
        build=lambda d: (
            [
                _f(d, 1, 1, 4, 4), _f(d, 1, 1, 8, 8),
                [2, 2], [2, 2], [0, 0],
                True, True, None,
            ],
            {},
        ),
    ),
    "avg_pool3d": OpSpec(build=lambda d: ([_f(d, 1, 1, 8, 8, 8), [2, 2, 2]], {})),
    "avg_pool3d_backward": OpSpec(
        build=lambda d: (
            [
                _f(d, 1, 1, 4, 4, 4),
                _f(d, 1, 1, 8, 8, 8),
                [2, 2, 2], [2, 2, 2], [0, 0, 0],
                False, True, None,
            ],
            {},
        ),
    ),
    # Schema: (self, kernel_size, output_size, random_samples[N,C,2]).
    "fractional_max_pool2d": OpSpec(
        build=lambda d: (
            [
                _f(d, 1, 3, 8, 8), [2, 2], [4, 4],
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
            [_f(d, 1, 1, 8, 8), [2, 2], [2, 2], [0, 0], [1, 1], False],
            {},
        ),
    ),
    "max_pool2d_with_indices_backward": OpSpec(
        build=lambda d: (
            [
                _f(d, 1, 1, 4, 4),                # grad_output
                _f(d, 1, 1, 8, 8),                # self
                [2, 2], [2, 2], [0, 0], [1, 1],   # kernel, stride, padding, dilation
                False,                             # ceil_mode
                _i(d, 1, 1, 4, 4),                # indices
            ],
            {},
        ),
    ),
    "max_pool3d_with_indices": OpSpec(
        build=lambda d: (
            [
                _f(d, 1, 1, 8, 8, 8),
                [2, 2, 2], [2, 2, 2], [0, 0, 0], [1, 1, 1], False,
            ],
            {},
        ),
    ),
    "max_pool3d_with_indices_backward": OpSpec(
        build=lambda d: (
            [
                _f(d, 1, 1, 4, 4, 4),             # grad_output
                _f(d, 1, 1, 8, 8, 8),             # self
                [2, 2, 2], [2, 2, 2], [0, 0, 0], [1, 1, 1],
                False,
                _i(d, 1, 1, 4, 4, 4),             # indices
            ],
            {},
        ),
    ),
    "max_unpool2d": OpSpec(
        build=lambda d: ([_f(d, 1, 1, 4, 4), _i(d, 1, 1, 4, 4), [8, 8]], {}),
    ),
    "max_unpool3d": OpSpec(
        build=lambda d: (
            [
                _f(d, 1, 1, 2, 2, 2), _i(d, 1, 1, 2, 2, 2),
                [4, 4, 4], [2, 2, 2], [0, 0, 0],
            ],
            {},
        ),
    ),

    # ---------------------------------------------------------------
    # Upsample
    # ---------------------------------------------------------------
    "upsample_nearest2d": OpSpec(
        build=lambda d: ([_f(d, 1, 3, 8, 8), [4, 4], None, None], {}),
    ),
    "upsample_nearest3d": OpSpec(
        build=lambda d: ([_f(d, 1, 1, 4, 4, 4), [8, 8, 8]], {}),
    ),
    "upsample_nearest1d": OpSpec(build=lambda d: ([_f(d, 1, 3, 8), [16], None], {})),
    # Schema: (grad_output, output_size, input_size, scales_h?, scales_w?).
    "upsample_nearest2d_backward": OpSpec(
        build=lambda d: (
            [_f(d, 1, 3, 16, 16), [16, 16], [1, 3, 8, 8], None, None],
            {},
        ),
    ),
    "upsample_linear1d": OpSpec(
        build=lambda d: ([_f(d, 1, 3, 8), [16], False, False], {}),
    ),
    "upsample_bilinear2d": OpSpec(
        build=lambda d: ([_f(d, 1, 3, 8, 8), [16, 16], False], {}),
    ),
    "upsample_bicubic2d": OpSpec(
        build=lambda d: ([_f(d, 1, 3, 8, 8), [16, 16], False], {}),
    ),
    "upsample_trilinear3d": OpSpec(
        build=lambda d: ([_f(d, 1, 2, 4, 4, 4), [8, 8, 8], False], {}),
    ),
    "_upsample_nearest_exact1d": OpSpec(
        build=lambda d: ([_f(d, 1, 3, 8), [16]], {}),
    ),
    "_upsample_nearest_exact2d": OpSpec(
        build=lambda d: ([_f(d, 1, 3, 8, 8), [16, 16]], {}),
    ),
    "_upsample_nearest_exact2d_backward": OpSpec(
        build=lambda d: (
            [_f(d, 1, 3, 16, 16), [16, 16], [1, 3, 8, 8], None, None],
            {},
        ),
    ),
    "_upsample_nearest_exact3d": OpSpec(
        build=lambda d: ([_f(d, 1, 2, 4, 4, 4), [8, 8, 8]], {}),
    ),
    "_upsample_bilinear2d_aa": OpSpec(
        build=lambda d: ([_f(d, 1, 3, 8, 8), [16, 16], False, None], {}),
    ),
    "_upsample_bicubic2d_aa": OpSpec(
        build=lambda d: ([_f(d, 1, 3, 8, 8), [16, 16], False, None], {}),
    ),
    # Schema: (grad_output, output_size, input_size, align_corners, scales_h?,
    # scales_w?).
    "_upsample_bilinear2d_aa_backward": OpSpec(
        build=lambda d: (
            [_f(d, 1, 3, 16, 16), [16, 16], [1, 3, 8, 8], False, None, None],
            {},
        ),
    ),

    # ---------------------------------------------------------------
    # Grid sampler / im2col
    # ---------------------------------------------------------------
    "grid_sampler_2d": OpSpec(
        build=lambda d: (
            [_f(d, 1, 1, 8, 8), torch.zeros(1, 8, 8, 2, device=d), 0, 0, False],
            {},
        ),
    ),
    # Schema: (grad_output, input, grid, interp, padding, align_corners,
    # bool[2] output_mask).
    "grid_sampler_2d_backward": OpSpec(
        build=lambda d: (
            [
                _f(d, 1, 1, 8, 8),                    # grad_output
                _f(d, 1, 1, 8, 8),                    # input
                torch.zeros(1, 8, 8, 2, device=d),    # grid
                0, 0, False,
                [True, True],                          # output_mask
            ],
            {},
        ),
    ),
    "grid_sampler_3d": OpSpec(
        build=lambda d: (
            [_f(d, 1, 1, 4, 4, 4), torch.zeros(1, 4, 4, 4, 3, device=d), 0, 0, False],
            {},
        ),
    ),
    "grid_sampler_3d_backward": OpSpec(
        build=lambda d: (
            [
                _f(d, 1, 1, 4, 4, 4),                     # grad_output
                _f(d, 1, 1, 4, 4, 4),                     # input
                torch.zeros(1, 4, 4, 4, 3, device=d),     # grid
                0, 0, False,
                [True, True],
            ],
            {},
        ),
    ),
    # Schema: im2col(self, kernel_size, dilation, padding, stride).
    # The final argument is ``stride`` and must be > 0; swapping stride/padding
    # produces ``stride should be greater than zero``.
    "im2col": OpSpec(
        build=lambda d: ([_f(d, 1, 1, 8, 8), [3, 3], [1, 1], [0, 0], [1, 1]], {}),
    ),
    # Schema: col2im(self, output_size[2], kernel_size, dilation, padding, stride).
    # Input (1, C*kH*kW, L) unfolds back to (N, C, H, W) with
    # ``output_size=[H, W]`` — must be a 2-element list, not 4.
    "col2im": OpSpec(
        build=lambda d: (
            [_f(d, 1, 4, 9), [4, 4], [2, 2], [1, 1], [0, 0], [1, 1]],
            {},
        ),
    ),

    # ---------------------------------------------------------------
    # Padding
    # ---------------------------------------------------------------
    "reflection_pad1d_backward": OpSpec(
        build=lambda d: ([_f(d, 1, 1, 8), _f(d, 1, 1, 6), [1, 1]], {}),
    ),
    "reflection_pad2d": OpSpec(
        build=lambda d: ([_f(d, 1, 1, 6, 6), [1, 1, 1, 1]], {}),
    ),
    "reflection_pad2d_backward": OpSpec(
        build=lambda d: ([_f(d, 1, 1, 8, 8), _f(d, 1, 1, 6, 6), [1, 1, 1, 1]], {}),
    ),
    "reflection_pad3d": OpSpec(
        build=lambda d: ([_f(d, 1, 1, 4, 4, 4), [1, 1, 1, 1, 1, 1]], {}),
    ),
    "reflection_pad3d_backward": OpSpec(
        build=lambda d: (
            [_f(d, 2, 3, 8, 8, 8), _f(d, 2, 3, 6, 6, 6), [1, 1, 1, 1, 1, 1]],
            {},
        ),
    ),
    "replication_pad1d_backward": OpSpec(
        build=lambda d: ([_f(d, 1, 1, 8), _f(d, 1, 1, 6), [1, 1]], {}),
    ),
    "replication_pad2d": OpSpec(
        build=lambda d: ([_f(d, 1, 1, 6, 6), [1, 1, 1, 1]], {}),
    ),
    "replication_pad2d_backward": OpSpec(
        build=lambda d: ([_f(d, 1, 1, 8, 8), _f(d, 1, 1, 6, 6), [1, 1, 1, 1]], {}),
    ),
    "replication_pad3d": OpSpec(
        build=lambda d: ([_f(d, 1, 1, 4, 4, 4), [1, 1, 1, 1, 1, 1]], {}),
    ),
    "replication_pad3d_backward": OpSpec(
        build=lambda d: (
            [_f(d, 1, 1, 6, 6, 6), _f(d, 1, 1, 4, 4, 4), [1, 1, 1, 1, 1, 1]],
            {},
        ),
    ),

    # ---------------------------------------------------------------
    # Integer / bitwise (CUDA rejects float)
    # ---------------------------------------------------------------
    "__ilshift__": OpSpec(build=lambda d: ([_i(d, 4, 4), _i(d, 4, 4, high=4)], {})),
    "__irshift__": OpSpec(build=lambda d: ([_i(d, 4, 4), _i(d, 4, 4)], {})),
    "__rshift__": OpSpec(build=lambda d: ([_i(d, 4, 4), _i(d, 4, 4)], {})),
    "__lshift__": OpSpec(build=lambda d: ([_i(d, 4, 4), _i(d, 4, 4)], {})),
    "bitwise_and": OpSpec(build=lambda d: ([_i(d, 4, 4), _i(d, 4, 4)], {})),
    "bitwise_and_": OpSpec(build=lambda d: ([_i(d, 4, 4), _i(d, 4, 4)], {})),
    "bitwise_or": OpSpec(build=lambda d: ([_i(d, 4, 4), _i(d, 4, 4)], {})),
    "bitwise_or_": OpSpec(build=lambda d: ([_i(d, 4, 4), _i(d, 4, 4)], {})),
    "bitwise_xor": OpSpec(build=lambda d: ([_i(d, 4, 4), _i(d, 4, 4)], {})),
    "bitwise_xor_": OpSpec(build=lambda d: ([_i(d, 4, 4), _i(d, 4, 4)], {})),
    "bitwise_left_shift": OpSpec(build=lambda d: ([_i(d, 4, 4), _i(d, 4, 4)], {})),
    "bitwise_right_shift": OpSpec(build=lambda d: ([_i(d, 4, 4), _i(d, 4, 4)], {})),
    "bitwise_not": OpSpec(
        build=lambda d: (
            [torch.randint(0, 256, (4, 4), device=d, dtype=torch.int64)],
            {},
        ),
    ),
    "bitwise_not_": OpSpec(build=lambda d: ([_i(d, 4, 4)], {})),
    "gcd": OpSpec(build=lambda d: ([_i(d, 4, 4), _i(d, 4, 4)], {})),
    "gcd_": OpSpec(build=lambda d: ([_i(d, 4, 4), _i(d, 4, 4)], {})),
    "lcm": OpSpec(
        build=lambda d: (
            [
                torch.randint(1, 100, (4, 4), device=d, dtype=torch.int64),
                torch.randint(1, 100, (4, 4), device=d, dtype=torch.int64),
            ],
            {},
        ),
    ),
    "lcm_": OpSpec(build=lambda d: ([_i(d, 4, 4), _i(d, 4, 4)], {})),

    # ---------------------------------------------------------------
    # Random distributions
    # ---------------------------------------------------------------
    "one_hot": OpSpec(
        build=lambda d: ([torch.randint(0, 5, (4,), device=d)], {}),
    ),
    # Poisson expects non-negative rates; ``randn`` can trigger ROCm HSA
    # faults. The emitted workload must use ``rand * scale``.
    "poisson": OpSpec(
        build=lambda _d: ([_CoverageTensorArg((4, 4), "rand_uniform", 4.0), None], {}),
    ),
    # ``bernoulli_`` expects probabilities in [0, 1]; see note above.
    "bernoulli_": OpSpec(
        build=lambda d: ([_f(d, 4, 4), _CoverageTensorArg((4, 4), "rand")], {}),
    ),
    # ``multinomial`` runs fine standalone but SIGSEGVs under
    # ``torch.profiler(CPU+CUDA)`` on ROCm (``renormRowsL1`` kernel path
    # interacting with ROCTracer). Confirmed by scripts/torch_trace_coverage_
    # scan_ops.py with --workers=1. Revisit after driver/profiler updates.
    "multinomial": OpSpec(
        skip="torch.profiler cannot collect ground truth: ROCm ROCTracer + "
        "renormRowsL1 kernel race produces SIGSEGV inside the profiler "
        "subprocess (op itself runs fine outside the profiler).",
    ),
    "geometric_": OpSpec(build=lambda d: ([_f(d, 4, 4), 0.5], {})),
    "repeat_interleave": OpSpec(build=lambda d: ([_i1(d, 16)], {})),
    "bincount": OpSpec(build=lambda d: ([_i1(d, 32).clamp_min(0)], {})),

    # ---------------------------------------------------------------
    # Masks / clamp
    # ---------------------------------------------------------------
    "clamp": OpSpec(build=lambda d: ([_f(d, 4, 4), 0.0, 1.0], {})),
    "clamp_": OpSpec(build=lambda d: ([_f(d, 4, 4), 0.0, 1.0], {})),
    "masked_fill_": OpSpec(build=lambda d: ([_f(d, 4, 4), _b(d), 0.0], {})),
    "masked_scatter_": OpSpec(
        build=lambda d: ([_f(d, 4, 4), _b(d), _f(d, 4, 4)], {}),
    ),
    "masked_select": OpSpec(build=lambda d: ([_f(d, 4, 4), _b(d)], {})),
    "_masked_scale": OpSpec(
        build=lambda d: (
            [
                _f(d, 4, 4),
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
        build=lambda d: ([_f(d, 4, 4), _b(d), 0.5], {}),
    ),

    # ---------------------------------------------------------------
    # Indexing / scatter / gather
    # ---------------------------------------------------------------
    # ``aten.index.Tensor`` needs a real ``Tensor?[]``, not ``None``.
    "index": OpSpec(
        build=lambda d: (
            [
                _f(d, 4, 4),
                [None, torch.randint(0, 4, (2, 2), device=d, dtype=torch.int64)],
            ],
            {},
        ),
    ),
    # ``List[int]`` defaults like ``[1, 1]`` duplicate a dim on 2D tensors.
    "flip": OpSpec(build=lambda d: ([_f(d, 4, 4), [0]], {})),
    "index_copy": OpSpec(
        build=lambda d: ([_f(d, 4, 4), 0, _i1(d, 2), _f(d, 2, 4)], {}),
    ),
    "index_copy_": OpSpec(
        build=lambda d: ([_f(d, 4, 4), 0, _i1(d, 2), _f(d, 2, 4)], {}),
    ),
    "index_add": OpSpec(
        build=lambda d: ([_f(d, 4, 4), 0, _i1(d, 2), _f(d, 2, 4)], {}),
    ),
    "index_add_": OpSpec(
        build=lambda d: ([_f(d, 4, 4), 0, _i1(d, 2), _f(d, 2, 4)], {}),
    ),
    # ``include_self`` is kwarg-only; passing as 6th positional trips the
    # dispatcher with "takes 5 positional argument(s) but 6 was/were given".
    "index_reduce": OpSpec(
        build=lambda d: (
            [_f(d, 4, 4), 0, _i1(d, 4), _f(d, 4, 4), "prod"],
            {"include_self": False},
        ),
    ),
    "index_reduce_": OpSpec(
        build=lambda d: (
            [_f(d, 4, 4), 0, _i1(d, 4), _f(d, 4, 4), "prod"],
            {"include_self": False},
        ),
    ),
    "index_fill_": OpSpec(
        build=lambda d: ([_f(d, 4, 4), 0, _i1(d, 2), torch.tensor(0.0, device=d)], {}),
    ),
    "index_select": OpSpec(build=lambda d: ([_f(d, 4, 4), 0, _i1(d, 2)], {})),
    # ``gather`` needs ``index`` to be int64; schema fallback gives float.
    "gather": OpSpec(
        build=lambda d: ([_f(d, 4, 4), 0, _i(d, 4, 4, low=0, high=4)], {}),
    ),
    "scatter": OpSpec(
        build=lambda d: ([_f(d, 4, 4), 0, _i(d, 4, 4), _f(d, 4, 4)], {}),
    ),
    "scatter_": OpSpec(
        build=lambda d: ([_f(d, 4, 4), 0, _i(d, 4, 4), _f(d, 4, 4)], {}),
    ),
    "scatter_add": OpSpec(
        build=lambda d: ([_f(d, 4, 4), 0, _i(d, 4, 4), _f(d, 4, 4)], {}),
    ),
    "scatter_add_": OpSpec(
        build=lambda d: ([_f(d, 4, 4), 0, _i(d, 4, 4), _f(d, 4, 4)], {}),
    ),
    # ``include_self`` is kwarg-only on both the functional and in-place form.
    "scatter_reduce": OpSpec(
        build=lambda d: (
            [_f(d, 4, 4), 0, _i(d, 4, 4), _f(d, 4, 4), "sum"],
            {"include_self": False},
        ),
    ),
    "scatter_reduce_": OpSpec(
        build=lambda d: (
            [_f(d, 4, 4), 0, _i(d, 4, 4), _f(d, 4, 4), "sum"],
            {"include_self": False},
        ),
    ),
    "take": OpSpec(build=lambda d: ([_f(d, 4, 4), _i1(d, 8)], {})),
    # ``segment_reduce``: most arguments (``lengths``, ``indices``,
    # ``offsets``, ``axis``, ``unsafe``, ``initial``) are kwarg-only. The
    # kernel normally checks ``lengths.sum() == data.size(axis)``, but the
    # emitter replaces tensor literals with random ``torch.randint``
    # values; ``unsafe=True`` bypasses the check so the reduction kernel
    # still launches.
    "segment_reduce": OpSpec(
        build=lambda d: (
            [_f(d, 8), "sum"],
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
                torch.randn(2, device=d),     # grad (output shape)
                torch.randn(2, device=d),     # output
                torch.randn(8, device=d),     # data
                "sum",                         # reduce
            ],
            {"lengths": torch.tensor([4, 4], device=d, dtype=torch.int64)},
        ),
    ),

    # ---------------------------------------------------------------
    # Misc shape / dtype / layout
    # ---------------------------------------------------------------
    "_chunk_cat": OpSpec(
        build=lambda d: ([[_f(d, 2, 4), _f(d, 2, 4)], 0, 2], {}),
    ),
    # Schema takes 4 positionals; ``compute_mode`` is int? but positional.
    "_cdist_forward": OpSpec(
        build=lambda d: ([_f(d, 2, 4, 8), _f(d, 2, 5, 8), 2.0, None], {}),
    ),
    "_thnn_fused_lstm_cell": OpSpec(
        build=lambda d: (
            [_f(d, 2, 16), _f(d, 2, 16), _f(d, 2, 4), None, None],
            {},
        ),
    ),
    "channel_shuffle": OpSpec(build=lambda d: ([_f(d, 1, 8, 4, 4), 2], {})),
    "bucketize": OpSpec(build=lambda d: ([_f(d, 8), _f(d, 5)], {})),
    "roll": OpSpec(build=lambda d: ([_f(d, 4, 4), [1], [0]], {})),
    "unfold": OpSpec(build=lambda d: ([_f(d, 2, 8), 1, 4, 2], {})),
    # Schema: unfold_backward(grad_in, SymInt[] input_sizes, dim, size, step).
    # ``input_sizes`` is a list of ints, not a tensor.
    "unfold_backward": OpSpec(
        build=lambda d: ([_f(d, 2, 4, 4), [2, 10], 1, 4, 2], {}),
    ),
    "view_as_complex": OpSpec(build=lambda d: ([_f(d, 4, 2)], {})),
    "view_as_real": OpSpec(
        build=lambda d: ([torch.randn(4, 2, device=d, dtype=torch.complex64)], {}),
    ),
    # Schema: view(self, SymInt[] size).  Passing ``8, 2`` as two ints
    # does not match the List[int] slot.
    "view": OpSpec(build=lambda d: ([_f(d, 4, 4), [8, 2]], {})),
    # Schema: glu_backward(grad_output, self, dim).  grad_output matches the
    # glu output shape (self halved along ``dim``); don't swap the two.
    "glu_backward": OpSpec(build=lambda d: ([_f(d, 2, 4), _f(d, 2, 8), 1], {})),

    # ---------------------------------------------------------------
    # FFT
    #
    # Forward/backward FFT ops (`_fft_c2c`, `_fft_c2r`, `_fft_r2c`) are
    # declared as skips at the top of this table — they SIGSEGV under
    # ``torch.profiler`` on ROCm even with correctly-typed complex input.
    # ---------------------------------------------------------------

    # ---------------------------------------------------------------
    # Quantized / packed matmul (keep the few that work with synthetic args;
    # the int4/int8 packed-mm and scaled-grouped-mm variants are declared
    # above in the skip block because they need backend-specific layouts).
    # ---------------------------------------------------------------
    "_int_mm": OpSpec(
        build=lambda d: (
            [
                torch.randint(-2, 2, (32, 32), device=d, dtype=torch.int8),
                torch.randint(-2, 2, (32, 32), device=d, dtype=torch.int8),
            ],
            {},
        ),
    ),
    "_convert_weight_to_int4pack": OpSpec(
        build=lambda d: (
            [torch.randint(0, 255, (128, 64), device=d, dtype=torch.uint8), 128],
            {},
        ),
    ),

    # ---------------------------------------------------------------
    # Softmax backward (mixed-dtype accumulation)
    # ---------------------------------------------------------------
    "_log_softmax_backward_data": OpSpec(
        build=lambda d: (
            [
                _f(d, 2, 8, dtype=torch.float16),
                _f(d, 2, 8, dtype=torch.float16),
                1,
                torch.float32,
            ],
            {},
        ),
    ),
    "_softmax_backward_data": OpSpec(
        build=lambda d: (
            [
                _f(d, 2, 8, dtype=torch.float16),
                _f(d, 2, 8, dtype=torch.float16),
                1,
                torch.float32,
            ],
            {},
        ),
    ),

    # ---------------------------------------------------------------
    # Misc / AMP / assertions
    # ---------------------------------------------------------------
    # Schema: (Tensor[] self, Tensor found_inf, Tensor inv_scale) -> ().
    # Omitting ``inv_scale`` fails the dispatcher.
    "_amp_foreach_non_finite_check_and_unscale_": OpSpec(
        build=lambda d: (
            [
                [_f(d, 4), _f(d, 4)],
                torch.zeros(1, device=d),   # found_inf
                torch.ones(1, device=d),    # inv_scale
            ],
            {},
        ),
    ),
    # ``_assert_async(self)`` evaluates ``self`` on the device and invokes
    # ``__trap`` if any element is ``False``, which surfaces in the host as
    # SIGABRT (exit 134). The default emitter reduces a plain bool tensor to
    # ``randint(0, 2, ..., dtype=bool)``, which is ``False`` with p=0.5 and
    # would abort the ground-truth subprocess; ``emit="bool_all_true"``
    # pins the value to ``torch.ones(..., dtype=bool)``.
    "_assert_async": OpSpec(
        build=lambda _d: (
            [_CoverageTensorArg(shape=(), emit="bool_all_true")],
            {},
        ),
    ),
    # nonzero_static: ``size`` and ``fill_value`` are kwarg-only; the schema
    # fallback can't synthesise the kwargs, so hand them in explicitly.
    "nonzero_static": OpSpec(
        build=lambda d: ([_f(d, 4, 4)], {"size": 8}),
    ),

    # ---------------------------------------------------------------
    # Convolution backward (needs explicit output_sizes + masks)
    # ---------------------------------------------------------------
    "convolution_backward": OpSpec(build=_convolution_backward),

    # ---------------------------------------------------------------
    # Fused optimizers (list-of-tensors + scalar state_steps)
    # ---------------------------------------------------------------
    "_fused_adam_": OpSpec(build=_fused_adam_args),
    "_fused_adamw_": OpSpec(build=_fused_adam_args),

    # ---------------------------------------------------------------
    # Backend-gated families: skip rather than fake args.
    #
    # Each op below fails under generic arg synthesis because it needs
    # backend support that is either absent on ROCm or requires a very
    # specific tensor layout (flash-attn kernels, fp8 gemms, 2:4
    # semi-structured sparse metadata, nested/jagged buffers, grouped
    # mm). Covering these op requires dedicated workloads, not the
    # schema-driven matrix. Remove the skip when the op is known to work
    # on the target stack.
    # ---------------------------------------------------------------
    "_scaled_dot_product_cudnn_attention": OpSpec(
        skip="Kernel ``cudnnMultiHeadAttnForward`` lives in cuDNN (NVIDIA-only); "
        "not available on ROCm — ``_scaled_dot_product_flash_attention`` is the "
        "analogous path on AMD.",
    ),
    "_scaled_dot_product_cudnn_attention_backward": OpSpec(
        skip="Kernel ``cudnnMultiHeadAttnBackwardData/Weights`` lives in cuDNN "
        "(NVIDIA-only); not available on ROCm.",
    ),
    "_scaled_dot_product_efficient_attention": OpSpec(
        skip="Kernel ``mem_efficient_attention_forward_generic_cuda`` is a "
        "CUDA-only xFormers-style kernel; not built for ROCm.",
    ),
    "_scaled_dot_product_efficient_attention_backward": OpSpec(
        skip="Kernel ``mem_efficient_attention_backward_generic_cuda`` is a "
        "CUDA-only xFormers-style kernel; not built for ROCm.",
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
    "_efficient_attention_forward": OpSpec(
        skip="Kernel ``mem_efficient_attention_forward_generic_cuda`` (xFormers) "
        "is not built for ROCm; layout-sensitive 4-D QKV in fp16/bf16 required.",
    ),
    "_efficient_attention_backward": OpSpec(
        skip="Kernel ``mem_efficient_attention_backward_generic_cuda`` (xFormers) "
        "is not built for ROCm.",
    ),
    "_scaled_mm": OpSpec(
        skip="fp8 gemm; requires fp8 tensors + scale tensors on supported hardware.",
    ),
    "_scaled_mm_v2": OpSpec(
        skip="fp8 gemm v2; requires fp8 tensors + scale tensors on supported hardware.",
    ),
    "_grouped_mm": OpSpec(
        skip="grouped matmul; requires per-group offset metadata on supported hw.",
    ),
    "_cslt_sparse_mm": OpSpec(
        skip="Kernel ``cusparseLtMatmul`` is in cuSPARSELt (NVIDIA-only); not "
        "available on ROCm (no hipSPARSELt equivalent wired into PyTorch).",
    ),
    "_sparse_semi_structured_linear": OpSpec(
        skip="Kernel ``sparse_semi_structured_linear`` is the CUTLASS 2:4 "
        "sparse path (NVIDIA sparse tensor cores); not built for ROCm.",
    ),
    "_sparse_semi_structured_addmm": OpSpec(
        skip="Kernel ``sparse_semi_structured_addmm`` is the CUTLASS 2:4 "
        "sparse path (NVIDIA sparse tensor cores); not built for ROCm.",
    ),
    "_nested_view_from_buffer": OpSpec(
        skip="Nested-tensor constructor; requires matching offsets/lengths buffers.",
    ),
}
