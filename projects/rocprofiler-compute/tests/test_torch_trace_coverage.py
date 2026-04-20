# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""ROCTX marker coverage test for ``inject_roctx.py``.

Samples CUDA ATen operators plus the structural entries in
:data:`_STRUCTURAL_BUILDERS` (``nn.Module`` subclass forwards, ``Optimizer.step``,
``torch.autograd`` / ``torch.compile`` / ``torch.jit`` / ``torch.distributed``
/ ``torch.cuda`` surfaces), writes a minimal ``coverage_workload.py`` plus a
``coverage_ground_truth_runner.py`` that runs ``torch.profiler`` per op, then
runs rocprof-compute with ``--torch-trace`` on the workload and compares
ROCTX markers + kernel correlation against the profiler ground truth.

Sampling is controlled by ``--coverage-seed`` / ``--coverage-n`` (defaults in
``conftest.py``). To exercise the full matrix, set ``--coverage-n`` to at least
``len(aten_ops) + len(structural_ops)``::

    pytest tests/test_torch_trace_coverage.py -m torch_trace \\
        --coverage-seed=0 --coverage-n=10000 -s

If the ground-truth subprocess fails, copies of the generated workload and
runner are saved to the pytest cwd as
``failed_torch_trace_coverage_workload.py`` /
``failed_torch_trace_coverage_runner.py`` for manual reruns.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import sys
import textwrap
import threading
import uuid
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Set, Tuple

import pandas as pd
import pytest
import test_utils
from torch_trace_coverage_op_specs import (
    OP_SPECS,
    OpSpec,
    _CoverageTensorArg,
)

try:
    import torch
except ImportError as _torch_err:
    print(
        f"SKIPPED test_torch_trace_coverage: import torch failed ({_torch_err}).",
        file=sys.stderr,
        flush=True,
    )
    pytest.skip(
        f"import torch failed: {_torch_err}",
        allow_module_level=True,
    )

if not torch.cuda.is_available():
    print(
        "SKIPPED test_torch_trace_coverage: torch.cuda.is_available() is False.",
        file=sys.stderr,
        flush=True,
    )
    pytest.skip(
        "torch.cuda.is_available() is False",
        allow_module_level=True,
    )

COVERAGE_TEST_CONFIG: Dict[str, Any] = {"cleanup": True}

# POSIX signal names for subprocess failure messages; unknown numbers print
# numerically.
_POSIX_SIGNAL_NUMBER_TO_NAME: Dict[int, str] = {
    4: "SIGILL",
    6: "SIGABRT",
    7: "SIGBUS",
    8: "SIGFPE",
    9: "SIGKILL",
    11: "SIGSEGV",
    15: "SIGTERM",
}


class OpEntry(NamedTuple):
    """One row in the coverage sample: what to call and how it was discovered.

    Attributes:
        name: Human-readable op id, e.g. ``torch.ops.aten.mm`` or
            ``nn.Module.__call__`` for structural patterns.
        category: ``aten`` for ``torch.ops`` entries, ``structural`` for
            synthetic workload patterns (module forward, optimizer step, etc.).
        schema: PyTorch ``FunctionSchema`` for the chosen overload, or ``None``
            for structural entries.
    """

    name: str
    category: str
    schema: object


class OpCompareOutcome(NamedTuple):
    """Result of :func:`compare_single_op` (status + log lines for stdout).

    ``reason`` is empty for passing ops and carries the human-readable skip
    explanation (``status == "skip"``) or failure reason (``status == "fail"``).
    """

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
# initialised. ``AMD_SERIALIZE_KERNEL`` is read once at context creation;
# synchronous dispatch is required for asynchronous faults to surface at
# their launch site so per-operator attribution (START / OK / FAIL banners
# and the device-liveness probe) remains correct.
os.environ.setdefault("AMD_SERIALIZE_KERNEL", "3")

import torch
from torch.profiler import profile, ProfilerActivity


def _probe_device() -> None:
    \"\"\"Dispatch and synchronise a minimal CUDA kernel to surface sticky faults.

    ``torch.cuda.synchronize`` alone is not a reliable liveness check on
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
                _probe_device()
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
    """Unique ``param_id`` (xdist worker + pid + tid + uuid) to avoid path races."""
    worker = os.environ.get("PYTEST_XDIST_WORKER", "main")
    return f"{prefix}_{worker}_{os.getpid()}_{threading.get_ident()}_{uuid.uuid4().hex}"


# -- ATen name helpers --
#
# ATen sample labels take one of two forms: ``torch.ops.aten.<op>`` for the
# default overload, or ``torch.ops.aten.<op>.<overload>`` for a named
# overload (``.Tensor``, ``.Scalar``, ``.out``, etc.).
# :func:`inject_roctx.instrument_all_torch_ops` wraps the packet rather
# than individual overloads, so the generated workload always calls
# ``torch.ops.aten.<op>`` and relies on the dispatcher to route to the
# target overload from the synthesized argument types.


def _aten_op_short_name(op_name: str) -> str:
    """Return the ATen op name without any overload suffix.

    ``torch.ops.aten.add`` → ``"add"``; ``torch.ops.aten.add.Tensor`` → ``"add"``.
    Returns ``""`` for non-ATen labels.
    """
    parts = op_name.split(".")
    return parts[3] if len(parts) >= 4 and parts[:3] == ["torch", "ops", "aten"] else ""


def _aten_overload_key(op_name: str) -> Optional[str]:
    """Return ``<op>.<overload>`` for non-default overload labels, else ``None``.

    ``torch.ops.aten.add`` → ``None``; ``torch.ops.aten.add.Tensor`` → ``"add.Tensor"``.
    """
    parts = op_name.split(".")
    if len(parts) >= 5 and parts[:3] == ["torch", "ops", "aten"]:
        return ".".join(parts[3:])
    return None


def _aten_packet_call_path(op_name: str) -> str:
    """Strip any overload suffix so the emitted call hits the wrapped packet.

    ``torch.ops.aten.add.Tensor`` → ``"torch.ops.aten.add"``; ``torch.ops.aten.add``
    is returned unchanged.
    """
    parts = op_name.split(".")
    if len(parts) >= 5 and parts[:3] == ["torch", "ops", "aten"]:
        return ".".join(parts[:4])
    return op_name


# -- Argument synthesis --
#
# Argument generation is restricted to hand-authored, per-operator builders
# registered in :mod:`torch_trace_coverage_op_specs`. An ATen operator is
# emitted into the coverage workload only when :data:`OP_SPECS` contains an
# ``OpSpec.build`` entry for it (looked up by overload-specific key first,
# then by packet key). Operators with ``OpSpec.skip`` set, and operators
# with no entry at all, are reported as SKIP. Structural entries carry no
# arguments.
#
# Rationale: arguments that satisfy an operator's ATen type schema may still
# violate kernel-level constraints (value ranges, shape invariants, index
# bounds, backend-specific layouts). On ROCm, violating such a constraint
# can trigger an illegal memory access that places the HIP context in a
# sticky-error state, forcing the host process to abort. Restricting
# argument generation to per-operator builders verified on ROCm makes this
# failure mode structurally unreachable.


def build_args_for_op(
    op: OpEntry,
) -> Optional[Tuple[List[Any], dict]]:
    """Return ``(args, kwargs)`` for ``op`` on CUDA, or ``None`` to SKIP.

    Structural entries return ``([], {})``. ATen entries return the output
    of :attr:`OpSpec.build` when :data:`OP_SPECS` has a matching entry under
    either the overload-specific key (``<op>.<overload>``) or the packet
    key (``<op>``). All other cases return ``None``.
    """
    device = "cuda"

    if op.category == "structural":
        return [], {}

    overload_key = _aten_overload_key(op.name)
    spec: Optional[OpSpec] = None
    if overload_key is not None:
        spec = OP_SPECS.get(overload_key)
    if spec is None:
        spec = OP_SPECS.get(_aten_op_short_name(op.name) or op.name.rsplit(".", 1)[-1])
    if spec is not None:
        if spec.skip is not None:
            return None
        if spec.build is not None:
            return spec.build(device)

    return None


# -- Operator discovery --


def discover_operators() -> Tuple[List[OpEntry], List[OpEntry]]:
    """Enumerate all CUDA-dispatched ATen overloads plus structural labels.

    Walks ``torch.ops.aten`` and emits one entry per overload with a CUDA
    kernel. The ``default`` overload keeps the bare name ``torch.ops.aten.<op>``;
    non-default overloads are labeled ``torch.ops.aten.<op>.<overload>``. The
    workload always calls the packet (see :func:`_aten_packet_call_path`), so
    dispatch is driven by the synthesized argument types. Structural entries
    come from :data:`_STRUCTURAL_BUILDERS` in sorted order.
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

    # Structural entries come from ``_STRUCTURAL_BUILDERS``; ``sorted`` keeps
    # emission deterministic across hash-seed randomization.
    structural_ops = [
        OpEntry(name, "structural", None)
        for name in sorted(_STRUCTURAL_BUILDERS.keys())
    ]

    return aten_ops, structural_ops


# -- Marker matching --


def marker_matches_op(op_name: str, marker_leaf: str) -> bool:
    """Return whether a ROCTX ``Function`` leaf counts as coverage for ``op_name``.

    Class-specific structural entries (e.g. ``nn.Module.Conv2d.__call__``) only
    match class-specific markers (``nn.Module.Conv2d.forward``); generic
    labels fall through to ``*.forward`` / ``*.step`` suffix matching.
    """
    if op_name == marker_leaf:
        return True

    # Class-specific structural patterns: ``nn.Module.<Class>.__call__`` only
    # matches markers ending in ``<Class>.forward`` (inject_roctx rewrites
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

    # ATen overload labels (``torch.ops.aten.<op>.<overload>``) match the
    # packet-level marker ``torch.ops.aten.<op>`` that ``inject_roctx`` emits.
    if op_name.startswith("torch.ops.aten."):
        op_short = _aten_op_short_name(op_name)
        if op_short:
            packet = _aten_packet_call_path(op_name)
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

# Ruff E501 uses 88 columns; workload ``def`` bodies are indented by 4 spaces.
_WORKLOAD_EMIT_BODY_CONTENT_MAX = 84


def _workload_emit_tensor_rand_uniform_setup(
    vname: str,
    shape: Tuple[Any, ...],
    scale: float,
) -> List[str]:
    """Emit ``torch.rand`` times ``scale`` (non-negative rates, etc.)."""
    scale_repr = repr(scale)
    single = f"{vname} = torch.rand({shape}, device=device) * {scale_repr}"
    if len(single) <= _WORKLOAD_EMIT_BODY_CONTENT_MAX:
        return [single]
    return [
        f"{vname} = torch.rand(",
        f"    {shape},",
        "    device=device,",
        f") * {scale_repr}",
    ]


def _workload_emit_tensor_rand_setup(
    vname: str,
    shape: Tuple[Any, ...],
) -> List[str]:
    """Emit ``torch.rand`` in ``[0, 1)`` (Bernoulli probabilities, etc.)."""
    single = f"{vname} = torch.rand({shape}, device=device)"
    if len(single) <= _WORKLOAD_EMIT_BODY_CONTENT_MAX:
        return [single]
    return [
        f"{vname} = torch.rand(",
        f"    {shape},",
        "    device=device,",
        ")",
    ]


def _factory_expr_for_dtype(shape: Tuple[Any, ...], dtype: torch.dtype) -> str:
    """Return a random-tensor factory expression that preserves ``dtype``.

    ``randn`` for real/complex floats, ``randint`` for integers, and
    ``randint(0, 2, ..., dtype=bool)`` for bool.
    """
    if dtype == torch.bool:
        return f"torch.randint(0, 2, {shape}, device=device, dtype=torch.bool)"
    if dtype.is_floating_point or dtype.is_complex:
        return f"torch.randn({shape}, device=device, dtype={dtype!r})"
    low, high = (-4, 4) if dtype == torch.int8 else (0, 4)
    return f"torch.randint({low}, {high}, {shape}, device=device, dtype={dtype!r})"


def _workload_emit_assignment(vname: str, rhs: str) -> List[str]:
    """Emit ``vname = rhs`` on one line, or wrap across lines for Ruff E501."""
    single = f"{vname} = {rhs}"
    if len(single) <= _WORKLOAD_EMIT_BODY_CONTENT_MAX:
        return [single]
    return [f"{vname} = (", f"    {rhs}", ")"]


def _workload_emit_tensor_random_setup(
    vname: str,
    shape: Tuple[Any, ...],
    dtype: torch.dtype,
) -> List[str]:
    """Emit a dtype-preserving random tensor construction."""
    return _workload_emit_assignment(vname, _factory_expr_for_dtype(shape, dtype))


def _workload_emit_tensor_spd_setup(
    vname: str,
    shape: Tuple[Any, ...],
) -> List[str]:
    """Emit an SPD matrix (``a @ a.mT + n*I``) for ``cholesky`` et al."""
    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError(
            f"_CoverageTensorArg(emit='spd') expects a square shape, got {shape!r}"
        )
    n = shape[0]
    return [
        f"{vname}_a = torch.randn({shape}, device=device)",
        (f"{vname} = {vname}_a @ {vname}_a.mT + {n} * torch.eye({n}, device=device)"),
    ]


def _workload_emit_tensor_pivots_1based_setup(
    vname: str,
    shape: Tuple[Any, ...],
) -> List[str]:
    """Emit a 1-D 1-based identity ``int32`` permutation for LU pivots.

    ``lu_unpack`` / ``linalg_lu_solve`` require pivots in ``[1, n]``; random
    int32 values would walk off the row table and trigger HIP 719.
    """
    if len(shape) != 1:
        raise ValueError(
            f"_CoverageTensorArg(emit='pivots_1based') expects a 1-D shape, "
            f"got {shape!r}"
        )
    n = shape[0]
    return [
        f"{vname} = torch.arange(1, {n + 1}, dtype=torch.int32, device=device)",
    ]


def _workload_emit_tensor_cumsum_offsets_setup(
    vname: str,
    shape: Tuple[Any, ...],
    segment_length: int,
) -> List[str]:
    """Emit ``[0, k, 2k, ..., (n-1)*k]`` int64 offsets for nested / jagged ops."""
    if len(shape) != 1:
        raise ValueError(
            f"_CoverageTensorArg(emit='cumsum_offsets') expects a 1-D shape, "
            f"got {shape!r}"
        )
    n = shape[0]
    return [
        (
            f"{vname} = torch.arange(0, {n * segment_length}, "
            f"{segment_length}, dtype=torch.int64, device=device)"
        ),
    ]


def _workload_emit_tensor_bool_all_true_setup(
    vname: str,
    shape: Tuple[Any, ...],
) -> List[str]:
    """Emit an all-``True`` bool tensor of ``shape``.

    Needed by ``_assert_async`` and similar ops that trap on ``False`` input;
    the default random emitter would produce ``False`` half the time.
    """
    return [
        f"{vname} = torch.ones({shape}, device=device, dtype=torch.bool)",
    ]


def _workload_emit_multiline_call(
    call_expr: str,
    arg_expression_strings: List[str],
) -> List[str]:
    """Emit ``call_expr(...)`` on one line or split across lines for Ruff E501."""
    parts = list(arg_expression_strings)
    if not parts:
        single = f"{call_expr}()"
        if len(single) <= _WORKLOAD_EMIT_BODY_CONTENT_MAX:
            return [single]
        return [
            f"{call_expr}(",
            ")",
        ]
    one_line = f"{call_expr}(" + ", ".join(parts) + ")"
    if len(one_line) <= _WORKLOAD_EMIT_BODY_CONTENT_MAX:
        return [one_line]
    out_lines = [f"{call_expr}("]
    for index, part in enumerate(parts):
        last = index == len(parts) - 1
        suffix = "" if last else ","
        out_lines.append(f"    {part}{suffix}")
    out_lines.append(")")
    return out_lines


_COVERAGE_TENSOR_ARG_EMITTERS: Dict[
    str,
    Callable[["_CoverageTensorArg", str], List[str]],
] = {
    "rand": lambda a, v: _workload_emit_tensor_rand_setup(v, a.shape),
    "rand_uniform": lambda a, v: _workload_emit_tensor_rand_uniform_setup(
        v,
        a.shape,
        a.scale,
    ),
    "spd": lambda a, v: _workload_emit_tensor_spd_setup(v, a.shape),
    "pivots_1based": lambda a, v: _workload_emit_tensor_pivots_1based_setup(
        v,
        a.shape,
    ),
    "cumsum_offsets": lambda a, v: _workload_emit_tensor_cumsum_offsets_setup(
        v,
        a.shape,
        int(a.scale),
    ),
    "bool_all_true": lambda a, v: _workload_emit_tensor_bool_all_true_setup(
        v,
        a.shape,
    ),
}


def serialize_arg(argument_value: Any, vname: str) -> Tuple[List[str], str]:
    """Build emitted-source lines that define ``vname`` plus an expression to pass.

    Args:
        argument_value: Runtime value from :func:`build_args_for_op` (tensor,
            :class:`_CoverageTensorArg`, list, scalar, ``None``, etc.).
        vname: Unique variable name in the generated workload script.

    Returns:
        ``(setup_lines, expr)`` where ``setup_lines`` are statements to run before
        the profiled call (empty for scalars), and ``expr`` is what appears inside
        ``call_expr(expr, ...)`` — usually ``vname`` for tensors, a list literal
        ``[sub0, sub1, ...]`` for sequences, or a Python literal for scalars.

    Emission is value-lossy for plain :class:`torch.Tensor` inputs: only
    ``shape`` and ``dtype`` survive, so values are re-randomized by the
    emitted factory. Builders that depend on specific tensor *values*
    (positive-definite, 1-based permutation, all-``True`` mask, ...) must
    use :class:`_CoverageTensorArg` with the appropriate ``emit`` mode.
    """
    if isinstance(argument_value, _CoverageTensorArg):
        emitter = _COVERAGE_TENSOR_ARG_EMITTERS.get(argument_value.emit)
        if emitter is None:
            raise ValueError(
                f"unsupported _CoverageTensorArg.emit: {argument_value.emit!r}"
            )
        return emitter(argument_value, vname), vname

    if isinstance(argument_value, torch.Tensor):
        return (
            _workload_emit_tensor_random_setup(
                vname,
                tuple(argument_value.shape),
                argument_value.dtype,
            ),
            vname,
        )

    # Serialize list / tuple elements independently: preserves heterogeneous
    # shapes and routes ``_CoverageTensorArg`` / nested tensor entries through
    # their emitters instead of ``repr`` (which would emit unbound names).
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
    # Scalars / dtypes / strings: ``repr`` round-trips to valid Python source.
    return [], repr(argument_value)


# -- Structural builder registry ----------------------------------------------
#
# A builder is ``safe_var -> (setup_lines, call_expr, call_args)``: statements
# placed inside ``def _run_<safe_var>()`` before the call, a callable expression,
# and the comma-separated argument substring. :data:`_STRUCTURAL_BUILDERS` is the
# single source of truth for the structural sample — both
# :func:`discover_operators` and :func:`emit_structural_preamble` consume it, so
# adding a new API surface is one entry.
StructuralBuilder = Callable[[str], Tuple[List[str], str, str]]


def _builder_nn_module_call(
    ctor_expr: str,
    input_expr: str,
    *,
    extra_setup: Tuple[str, ...] = (),
) -> StructuralBuilder:
    """Instantiate ``ctor_expr`` on CUDA and call it with ``input_expr``.

    ``extra_setup`` prepends extra statements for auxiliary tensors referenced
    by ``input_expr``.
    """

    def build(safe_var: str) -> Tuple[List[str], str, str]:
        setup: List[str] = ["import torch.nn as nn"]
        setup.extend(extra_setup)
        setup.append(f"_mod_{safe_var} = ({ctor_expr}).cuda()")
        return setup, f"_mod_{safe_var}", input_expr

    return build


def _builder_optimizer_step(optimizer_ctor: str) -> StructuralBuilder:
    """Run forward+backward on a tiny Linear, then call ``optim.step()``.

    ``optimizer_ctor`` substitutes ``PARAMS`` with the Linear's parameters
    (e.g. ``"torch.optim.Adam(PARAMS, lr=1e-3)"``). Real gradients are needed
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
# builders. The ``atexit`` hook destroys the process group on interpreter
# shutdown regardless of HIP/CUDA context state; without it, rocBLAS or
# NCCL cleanup on a faulted device can abort the interpreter with SIGABRT
# and mask the original failure.
_DISTRIBUTED_BOOTSTRAP_SETUP: Tuple[str, ...] = (
    "import atexit as _atexit",
    "import os as _os",
    "import torch.distributed as _dist",
    '_os.environ.setdefault("MASTER_ADDR", "127.0.0.1")',
    '_os.environ.setdefault("MASTER_PORT", "29500")',
    "if not _dist.is_initialized():",
    '    _dist.init_process_group(backend="nccl", rank=0, world_size=1)',
    "    def _shutdown_dist():",
    "        try:",
    "            if _dist.is_initialized():",
    "                _dist.destroy_process_group()",
    "        except Exception:",
    "            pass",
    "    _atexit.register(_shutdown_dist)",
)


def _builder_distributed(
    call_expr: str,
    *,
    tensor_setup: Tuple[str, ...] = ("_t = torch.randn(4, 4, device=device)",),
    call_args: str = "_t",
) -> StructuralBuilder:
    """Run a ``torch.distributed.*`` collective on a single-GPU process group.

    Systems without a working NCCL/RCCL raise at ``init_process_group`` and the
    op is reported as SKIP — multi-GPU topology is not required.
    """

    def build(_safe_var: str) -> Tuple[List[str], str, str]:
        setup = list(_DISTRIBUTED_BOOTSTRAP_SETUP) + list(tensor_setup)
        return setup, call_expr, call_args

    return build


# Per-module builders: ``name -> (ctor_expr, input_expr, extra_setup)``.
_NN_MODULE_BUILDERS: Dict[str, Tuple[str, str, Tuple[str, ...]]] = {
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
_OPTIMIZER_BUILDERS: Dict[str, str] = {
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


def _builder_autograd_grad(safe_var: str) -> Tuple[List[str], str, str]:
    """``torch.autograd.grad(loss, params)`` on a 1-layer MLP."""
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


def _builder_autograd_backward(safe_var: str) -> Tuple[List[str], str, str]:
    """``torch.autograd.backward([loss])`` — functional form of ``.backward()``."""
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


def _builder_autograd_function_apply(safe_var: str) -> Tuple[List[str], str, str]:
    """Define a tiny :class:`torch.autograd.Function` and call its ``.apply``."""
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


def _builder_autograd_functional_jacobian(
    safe_var: str,
) -> Tuple[List[str], str, str]:
    """``torch.autograd.functional.jacobian(f, x)`` for a 1-layer MLP wrapper."""
    return (
        [
            "import torch.nn as nn",
            f"_m_{safe_var} = nn.Linear(4, 4).cuda()",
            f"_x_{safe_var} = torch.randn(4, device=device)",
        ],
        "torch.autograd.functional.jacobian",
        f"lambda z: _m_{safe_var}(z).sum(), _x_{safe_var}",
    )


def _builder_autograd_functional_hessian(
    safe_var: str,
) -> Tuple[List[str], str, str]:
    """``torch.autograd.functional.hessian(f, x)`` of a quadratic form."""
    return (
        [f"_x_{safe_var} = torch.randn(4, device=device)"],
        "torch.autograd.functional.hessian",
        f"lambda z: (z * z).sum(), _x_{safe_var}",
    )


def _builder_autograd_functional_vjp(safe_var: str) -> Tuple[List[str], str, str]:
    """``torch.autograd.functional.vjp(f, x, v)`` for a tiny MLP."""
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


def _builder_autograd_functional_jvp(safe_var: str) -> Tuple[List[str], str, str]:
    """``torch.autograd.functional.jvp(f, x, v)`` for a tiny MLP."""
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


def _builder_torch_compile(safe_var: str) -> Tuple[List[str], str, str]:
    """Compile a trivial fn with ``torch.compile`` and invoke it once."""
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


def _builder_torch_jit_trace(safe_var: str) -> Tuple[List[str], str, str]:
    """Trace a tiny Linear with ``torch.jit.trace`` and run the traced module."""
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


def _builder_torch_jit_script(safe_var: str) -> Tuple[List[str], str, str]:
    """Script a pure-python fn with ``torch.jit.script`` and run it on the GPU."""
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
# in :func:`compare_single_op`, so a missing marker flags a wrapper gap.


def _builder_cuda_synchronize(_safe_var: str) -> Tuple[List[str], str, str]:
    return [], "torch.cuda.synchronize", ""


def _builder_cuda_current_device(_safe_var: str) -> Tuple[List[str], str, str]:
    return [], "torch.cuda.current_device", ""


def _builder_cuda_device_count(_safe_var: str) -> Tuple[List[str], str, str]:
    return [], "torch.cuda.device_count", ""


def _builder_cuda_empty_cache(_safe_var: str) -> Tuple[List[str], str, str]:
    return [], "torch.cuda.empty_cache", ""


def _builder_cuda_memory_allocated(_safe_var: str) -> Tuple[List[str], str, str]:
    return [], "torch.cuda.memory_allocated", ""


def _builder_cuda_reset_peak_memory_stats(
    _safe_var: str,
) -> Tuple[List[str], str, str]:
    return [], "torch.cuda.reset_peak_memory_stats", ""


def _builder_cuda_manual_seed(_safe_var: str) -> Tuple[List[str], str, str]:
    return [], "torch.cuda.manual_seed", "0"


def _builder_cuda_set_device(_safe_var: str) -> Tuple[List[str], str, str]:
    return [], "torch.cuda.set_device", "0"


def _builder_cuda_stream(safe_var: str) -> Tuple[List[str], str, str]:
    """Create a :class:`torch.cuda.Stream` and submit one op on it.

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


def _builder_cuda_event(safe_var: str) -> Tuple[List[str], str, str]:
    """Record and query a :class:`torch.cuda.Event` around a small op."""
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


def _build_structural_builder_registry() -> Dict[str, StructuralBuilder]:
    """Flat ``name -> StructuralBuilder`` map. Duplicate names raise."""
    registry: Dict[str, StructuralBuilder] = {}

    # Generic ``nn.Module.__call__`` / ``Optimizer.step`` / ``torch.Tensor.backward``
    # entries kept for back-compat; class-specific variants below are a superset.
    def _legacy_nn_call(safe_var: str) -> Tuple[List[str], str, str]:
        return (
            [
                "import torch.nn as nn",
                f"_model_{safe_var} = nn.Linear(4, 4).cuda()",
            ],
            f"_model_{safe_var}",
            "torch.randn(2, 4, device=device)",
        )

    def _legacy_optimizer_step(safe_var: str) -> Tuple[List[str], str, str]:
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

    def _legacy_tensor_backward(safe_var: str) -> Tuple[List[str], str, str]:
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

    registry["nn.Module.__call__"] = _legacy_nn_call
    registry["Optimizer.step"] = _legacy_optimizer_step
    registry["torch.Tensor.backward"] = _legacy_tensor_backward

    for name, (ctor, input_expr, extra_setup) in _NN_MODULE_BUILDERS.items():
        if name in registry:
            raise AssertionError(f"duplicate structural builder: {name!r}")
        registry[name] = _builder_nn_module_call(
            ctor,
            input_expr,
            extra_setup=extra_setup,
        )

    for name, optim_ctor in _OPTIMIZER_BUILDERS.items():
        if name in registry:
            raise AssertionError(f"duplicate structural builder: {name!r}")
        registry[name] = _builder_optimizer_step(optim_ctor)

    for name, fn in (
        ("torch.autograd.grad", _builder_autograd_grad),
        ("torch.autograd.backward", _builder_autograd_backward),
        ("torch.autograd.Function.apply", _builder_autograd_function_apply),
        (
            "torch.autograd.functional.jacobian",
            _builder_autograd_functional_jacobian,
        ),
        (
            "torch.autograd.functional.hessian",
            _builder_autograd_functional_hessian,
        ),
        ("torch.autograd.functional.vjp", _builder_autograd_functional_vjp),
        ("torch.autograd.functional.jvp", _builder_autograd_functional_jvp),
        ("torch.compile", _builder_torch_compile),
        ("torch.jit.trace", _builder_torch_jit_trace),
        ("torch.jit.script", _builder_torch_jit_script),
        ("torch.distributed.all_reduce", _builder_distributed("_dist.all_reduce")),
        (
            "torch.distributed.broadcast",
            _builder_distributed(
                "_dist.broadcast",
                call_args="_t, src=0",
            ),
        ),
        (
            "torch.distributed.reduce",
            _builder_distributed(
                "_dist.reduce",
                call_args="_t, dst=0",
            ),
        ),
        (
            "torch.distributed.all_gather",
            _builder_distributed(
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
            _builder_distributed(
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
            _builder_distributed(
                "_dist.barrier",
                tensor_setup=(),
                call_args="",
            ),
        ),
        ("torch.cuda.synchronize", _builder_cuda_synchronize),
        ("torch.cuda.current_device", _builder_cuda_current_device),
        ("torch.cuda.device_count", _builder_cuda_device_count),
        ("torch.cuda.empty_cache", _builder_cuda_empty_cache),
        ("torch.cuda.memory_allocated", _builder_cuda_memory_allocated),
        (
            "torch.cuda.reset_peak_memory_stats",
            _builder_cuda_reset_peak_memory_stats,
        ),
        ("torch.cuda.manual_seed", _builder_cuda_manual_seed),
        ("torch.cuda.set_device", _builder_cuda_set_device),
        ("torch.cuda.Stream", _builder_cuda_stream),
        ("torch.cuda.Event", _builder_cuda_event),
    ):
        if name in registry:
            raise AssertionError(f"duplicate structural builder: {name!r}")
        registry[name] = fn

    return registry


_STRUCTURAL_BUILDERS: Dict[str, StructuralBuilder] = (
    _build_structural_builder_registry()
)


def emit_structural_preamble(
    op_name: str,
    safe_var: str,
) -> Tuple[List[str], str, str]:
    """Dispatch into :data:`_STRUCTURAL_BUILDERS` for one structural workload.

    Unknown names fall through to ``([], op_name, "")``; the resulting bare
    call raises at runtime and the comparison layer maps it to SKIP.
    """
    builder = _STRUCTURAL_BUILDERS.get(op_name)
    if builder is None:
        return [], op_name, ""
    return builder(safe_var)


def build_workload_module_lines(operators: List[OpEntry]) -> List[str]:
    """Emit source lines for a minimal ``coverage_workload.py`` module.

    One ``def`` per operator plus ``run_all``. The ground-truth runner imports
    this module and wraps each ``ALL_OPS`` entry with ``torch.profiler``;
    ``run_all()`` is the entry point when rocprof-compute runs this script.
    """
    lines = [
        "import os",
        "import sys",
        # ``AMD_SERIALIZE_KERNEL`` must be set before ``import torch``: HIP
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
        # calls instead of bare ``tensor([...])`` reprs (which NameError).
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
            # function (so ``<packet>.<overload>`` is not accessible in the
            # instrumented run). Argument types drive dispatch to the target
            # overload.
            call_expr = _aten_packet_call_path(op.name)
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
        call_body_lines = _workload_emit_multiline_call(call_expr, call_segments)
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
    # ``run_all`` iterates over every emitted operator. Per-operator
    # exceptions are logged and do not halt iteration. START / OK / FAIL
    # banners are flushed to stderr so that a native-code termination can
    # be attributed to the last operator whose START has no matching
    # terminal banner. ``torch.cuda.synchronize()`` anchors asynchronous
    # completion before OK is logged. After a per-operator failure,
    # ``_probe_device()`` dispatches a minimal kernel to test context
    # liveness; if the probe also raises, iteration terminates with a
    # DEVICE_DEAD banner identifying the terminal operator.
    lines.append("def _probe_device():")
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
    lines.append("                _probe_device()")
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
    """Write the static torch.profiler runner next to ``coverage_workload.py``."""
    Path(path).write_text(
        COVERAGE_GROUND_TRUTH_RUNNER_SOURCE + "\n",
        encoding="utf-8",
    )


def write_coverage_workload_artifacts(
    operators: List[OpEntry],
    workload_script_path: str,
    ground_truth_runner_script_path: str,
) -> None:
    """Write ``coverage_workload.py`` and ``coverage_ground_truth_runner.py``."""
    Path(workload_script_path).write_text(
        "\n".join(build_workload_module_lines(operators)),
        encoding="utf-8",
    )
    write_coverage_ground_truth_runner_script(ground_truth_runner_script_path)


# -- ROCTX marker CSV parsing --


def parse_roctx_markers(
    workload_dir: str,
) -> Tuple[Dict[str, Set[str]], Set[str]]:
    """Parse rocprof-compute marker + counter CSVs under ``workload_dir``.

    Returns ``(op_to_kernels, marker_ops)``: ``marker_ops`` is every distinct
    marker leaf, ``op_to_kernels`` maps a leaf to correlated GPU kernel names
    (leaves without kernels are omitted). Returns ``({}, set())`` if no marker
    CSV exists.
    """
    marker_files = sorted(Path(workload_dir).glob("**/*marker_api_trace.csv"))
    counter_files = sorted(Path(workload_dir).glob("**/*counter_collection.csv"))
    if not marker_files:
        return {}, set()

    marker_df = pd.concat(
        [pd.read_csv(f) for f in marker_files],
        ignore_index=True,
    )
    marker_df = _with_correlation_id_standard_name(marker_df)
    marker_ops, op_to_corr = _collect_marker_ops_and_correlations(marker_df)

    op_to_kernels: Dict[str, Set[str]] = {}
    if counter_files and op_to_corr:
        op_to_kernels = _kernels_by_marker_leaf(op_to_corr, counter_files)

    return op_to_kernels, marker_ops


# -- Per-operator comparison --


def _torch_trace_coverage_color_enabled() -> bool:
    """True when ANSI colors are allowed for coverage report lines (stdout)."""
    if os.environ.get("NO_COLOR", "").strip():
        return False
    force = os.environ.get("FORCE_COLOR", "").strip().lower()
    if force in ("1", "true", "yes"):
        return True
    return sys.stdout.isatty()


def _torch_trace_coverage_red(text: str) -> str:
    if not _torch_trace_coverage_color_enabled():
        return text
    return f"\033[31m{text}\033[0m"


def _coverage_log_pass(op_name: str, *, note: str = "") -> Tuple[str, ...]:
    """Stdout lines for a passing operator (``pytest -s``)."""
    if note:
        return (f"PASS  {op_name}", f"    {note}")
    return (f"PASS  {op_name}",)


def _coverage_log_fail(op_name: str, reason: str) -> Tuple[str, ...]:
    """Stdout lines for a failing operator; reason is red when color is allowed."""
    body = reason.splitlines() or [reason]
    first = f"FAIL  {op_name}"
    if len(body) == 1:
        return (first, f"    {_torch_trace_coverage_red(body[0])}")
    lines = [first]
    for ln in body:
        lines.append(f"    {_torch_trace_coverage_red(ln)}")
    return tuple(lines)


def _coverage_log_skip(op_name: str, reason: str) -> Tuple[str, ...]:
    """Stdout lines for a skipped operator."""
    body = reason.splitlines() or [reason]
    lines = [f"SKIP  {op_name}"]
    lines.extend(f"    {ln}" for ln in body)
    return tuple(lines)


def _describe_missing_or_errored_op(
    op: OpEntry,
    ground_truth_entry: Optional[Dict[str, Any]],
) -> str:
    """Human-readable SKIP reason: workload error, OpSpec skip, or missing builder."""
    if ground_truth_entry is not None and "error" in ground_truth_entry:
        return f"workload raised at runtime: {ground_truth_entry['error']}"

    if op.category == "structural":
        return "structural op missing from generated workload"

    overload_key = _aten_overload_key(op.name)
    short_name = _aten_op_short_name(op.name) or op.name.rsplit(".", 1)[-1]
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
        "no OpSpec.build entry in torch_trace_coverage_op_specs.OP_SPECS; "
        "argument synthesis is restricted to per-operator builders"
    )


def _multiline_coverage_failure_warning(
    failure_detail: List[Tuple[str, str]],
    *,
    max_ops: int,
    seed: int,
    sample_budget: int,
) -> str:
    """Bounded multiline text for :func:`warnings.warn` when stdout is captured."""
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
) -> OpCompareOutcome:
    """Compare one op's profiler JSON entry to parsed ROCTX markers and kernels.

    Returns an :class:`OpCompareOutcome` with ``status`` ``pass`` / ``fail`` /
    ``skip``, a reason (empty for passes), and the log lines to print.
    """
    ground_truth_entry = ground_truth.get(op.name)

    if ground_truth_entry is None or "error" in ground_truth_entry:
        err_msg = _describe_missing_or_errored_op(op, ground_truth_entry)
        reason = err_msg if len(err_msg) <= 4000 else f"{err_msg[:4000]}…"
        return OpCompareOutcome(
            "skip",
            reason,
            _coverage_log_skip(op.name, reason),
        )

    profiler_kernels = ground_truth_entry.get("cuda_kernels", [])
    profiler_kernel_set = set(profiler_kernels)

    marker_found = any(
        marker_matches_op(op.name, observed_marker_leaf)
        for observed_marker_leaf in roctx_marker_names
    )
    roctx_kernels: Set[str] = set()
    for observed_marker_leaf in roctx_marker_names:
        if marker_matches_op(op.name, observed_marker_leaf):
            roctx_kernels |= roctx_kernels_map.get(observed_marker_leaf, set())

    # Structural ops are hierarchical wrappers; their markers don't correlate
    # to GPU kernels directly (inner ATen ops do). Marker presence == PASS.
    if op.category == "structural":
        if marker_found:
            return OpCompareOutcome(
                "pass",
                "",
                _coverage_log_pass(op.name, note="structural: marker present"),
            )
        skip_msg = (
            f"no ROCTX marker for '{op.name}' — inject_roctx may not "
            "instrument this call site (subclass overrides can bypass the "
            "base-class wrapper)"
        )
        return OpCompareOutcome(
            "skip",
            skip_msg,
            _coverage_log_skip(op.name, skip_msg),
        )

    if not profiler_kernel_set:
        if marker_found:
            return OpCompareOutcome(
                "pass",
                "",
                _coverage_log_pass(
                    op.name,
                    note="marker present; no kernels in ground truth",
                ),
            )
        skip_msg = "no GPU kernels in ground truth"
        return OpCompareOutcome(
            "skip",
            skip_msg,
            _coverage_log_skip(op.name, skip_msg),
        )

    if marker_found and roctx_kernels:
        return OpCompareOutcome(
            "pass",
            "",
            _coverage_log_pass(op.name),
        )

    if marker_found:
        reason = "marker found but no correlated kernels"
        return OpCompareOutcome(
            "fail",
            reason,
            _coverage_log_fail(op.name, reason),
        )

    reason = "marker not found"
    return OpCompareOutcome(
        "fail",
        reason,
        _coverage_log_fail(op.name, reason),
    )


def print_torch_trace_coverage_session_header(
    seed: int,
    sample_budget: int,
    sampled_operator_count: int,
    aten_operator_count: int,
    structural_operator_count: int,
) -> None:
    """Print seed / sampling summary (``pytest -s``); warn for default capture."""
    print(
        f"\n  Seed: {seed} | {sampled_operator_count} operators"
        f" selected from {aten_operator_count} CUDA ATen ops"
        f" + {structural_operator_count} structural"
        f" (budget={sample_budget})\n"
    )
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


def _describe_subprocess_exit_code(returncode: int) -> str:
    """Human-readable explanation for ``subprocess`` ``returncode`` (POSIX)."""
    if returncode < 0:
        signal_number = -returncode
        name = _POSIX_SIGNAL_NUMBER_TO_NAME.get(signal_number, "")
        name_part = f" ({name})" if name else ""
        return (
            f"Child process terminated by signal {signal_number}{name_part} "
            f"(exit {returncode}): native code fault (GPU kernel, driver, or "
            "profiler hook)."
        )
    return f"Child process exited with status {returncode}."


def _op_workload_failure_due_to(
    *,
    timed_out: bool,
    returncode: int | None,
) -> str:
    """Short phrase for ``pytest.fail`` lead-in."""
    if timed_out:
        return "ground-truth subprocess exceeded 120s timeout"
    assert returncode is not None
    if returncode < 0:
        signal_number = -returncode
        name = _POSIX_SIGNAL_NUMBER_TO_NAME.get(signal_number, "")
        name_part = f", {name}" if name else ""
        return (
            f"ground-truth subprocess killed by signal {signal_number}"
            f"{name_part} (exit {returncode})"
        )
    return f"ground-truth subprocess exited with status {returncode}"


def _stderr_tail_collapsed(
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


def _copy_failed_coverage_artifacts_to_cwd(
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


def _extract_device_fault_banner(stderr_text: str) -> str:
    """Summarise device-fault markers emitted by the generated workload.

    Scans ``stderr_text`` for ``DEVICE_DEAD`` banners and for the most
    recent ``START`` without a matching ``OK`` or ``FAIL`` terminal
    banner; both identify the operator implicated in a context-faulting
    failure. Returns the combined hint, or an empty string when neither
    signal is present.
    """
    if not stderr_text:
        return ""
    dead: List[str] = []
    last_start: str | None = None
    last_terminal_label: str | None = None
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
    coverage_seed: int | None = None,
    coverage_sample_budget: int | None = None,
) -> None:
    """Run ``coverage_ground_truth_runner.py`` (torch.profiler + JSON write)."""
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
        copy_note = _copy_failed_coverage_artifacts_to_cwd(
            workload_script_path,
            runner_script_path,
        )
        due = _op_workload_failure_due_to(timed_out=True, returncode=None)
        out_tail = _stderr_tail_collapsed(exc.stdout or "", max_lines=8)
        err_tail = _stderr_tail_collapsed(exc.stderr or "")
        fault_hint = _extract_device_fault_banner(exc.stderr or "")
        hint_block = f"--- device fault ---\n{fault_hint}\n\n" if fault_hint else ""
        pytest.fail(
            f"Op workload failed: {due}.{repro}{copy_note}\n\n"
            f"{hint_block}"
            f"--- stdout (tail) ---\n{out_tail}\n\n"
            f"--- stderr (tail) ---\n{err_tail}"
        )
    if completed.returncode != 0:
        copy_note = _copy_failed_coverage_artifacts_to_cwd(
            workload_script_path,
            runner_script_path,
        )
        due = _op_workload_failure_due_to(
            timed_out=False,
            returncode=completed.returncode,
        )
        exit_expl = _describe_subprocess_exit_code(completed.returncode)
        err_tail = _stderr_tail_collapsed(completed.stderr or "")
        out_tail = _stderr_tail_collapsed(completed.stdout or "", max_lines=12)
        cmdline = subprocess.list2cmdline(argv)
        fault_hint = _extract_device_fault_banner(completed.stderr or "")
        hint_block = f"--- device fault ---\n{fault_hint}\n\n" if fault_hint else ""
        pytest.fail(
            f"Op workload failed: {due}.{repro}{copy_note}\n\n"
            f"{exit_expl}\n\n"
            f"{hint_block}"
            f"--- command ---\n{cmdline}\n\n"
            f"--- stdout (tail) ---\n{out_tail}\n\n"
            f"--- stderr (tail) ---\n{err_tail}"
        )


# -- Main test --


@pytest.mark.torch_trace
def test_random_operator_kernel_coverage(
    request,
    binary_handler_profile_rocprof_compute,
    torch_trace_coverage_sampling,
):
    """Verify ``--torch-trace`` ROCTX output matches profiler ground truth.

    Steps: sample ops → emit workload + runner → run runner for JSON → run
    rocprof-compute on the workload → parse CSVs → compare per op. Per-op
    mismatches are reported (stdout + :class:`UserWarning`) but do **not**
    fail the test item, so CI can collect signal without blocking on gaps.
    """
    seed, sample_budget = torch_trace_coverage_sampling
    rng = random.Random(seed)

    aten_ops, structural_ops = discover_operators()

    n_aten = min(
        max(0, sample_budget - len(structural_ops)),
        len(aten_ops),
    )
    sampled = rng.sample(aten_ops, n_aten) + structural_ops

    print_torch_trace_coverage_session_header(
        seed,
        sample_budget,
        len(sampled),
        len(aten_ops),
        len(structural_ops),
    )

    gt_work_dir = test_utils.get_output_dir(
        param_id=unique_get_output_param_id("torch_trace_gt"),
        suffix="_tmp",
        clean_existing=True,
    )
    workload_dir = test_utils.get_output_dir(
        param_id=unique_get_output_param_id("random_op_coverage"),
        clean_existing=True,
    )
    Path(gt_work_dir).mkdir(parents=True, exist_ok=True)
    Path(workload_dir).mkdir(parents=True, exist_ok=True)

    ground_truth_path = str(Path(gt_work_dir) / "ground_truth.json")
    workload_script_path = str(Path(gt_work_dir) / "coverage_workload.py")
    ground_truth_runner_script_path = str(
        Path(gt_work_dir) / "coverage_ground_truth_runner.py"
    )

    try:
        write_coverage_workload_artifacts(
            sampled,
            workload_script_path,
            ground_truth_runner_script_path,
        )

        # Run 1: torch.profiler ground truth (runner loads workload module)
        run_ground_truth_torch_profiler_subprocess(
            ground_truth_runner_script_path,
            workload_script_path,
            ground_truth_path,
            coverage_seed=seed,
            coverage_sample_budget=sample_budget,
        )
        with open(ground_truth_path) as f:
            ground_truth = json.load(f)

        # Run 2: rocprof-compute --torch-trace (profiled app is minimal workload)
        binary_handler_profile_rocprof_compute(
            {
                **COVERAGE_TEST_CONFIG,
                "coverage_workload": [
                    sys.executable,
                    workload_script_path,
                ],
            },
            workload_dir,
            ["--experimental", "--torch-trace", "--iteration-multiplexing"],
            check_success=False,
            app_name="coverage_workload",
        )

        roctx_kernels_map, roctx_marker_names = parse_roctx_markers(workload_dir)

        # Per-operator comparison
        failure_detail: List[Tuple[str, str]] = []
        passed = skipped = 0
        for op in sampled:
            outcome = compare_single_op(
                op,
                ground_truth,
                roctx_marker_names,
                roctx_kernels_map,
            )
            for line in outcome.log_lines:
                print(line)
            if outcome.status == "pass":
                passed += 1
            elif outcome.status == "fail":
                failure_detail.append((op.name, outcome.reason))
            else:
                skipped += 1

        print(
            f"\n  Summary: {len(sampled)} ops — "
            f"{passed} PASS, {len(failure_detail)} FAIL, {skipped} SKIP\n"
        )

        # TODO: tighten to ``assert not failure_detail`` once every sampled
        # operator reliably matches a ROCTX marker and its kernels. The
        # current assertion guards only against total regression
        # (zero successes).
        if failure_detail:
            warnings.warn(
                _multiline_coverage_failure_warning(
                    failure_detail,
                    max_ops=48,
                    seed=seed,
                    sample_budget=sample_budget,
                ),
                UserWarning,
                stacklevel=1,
            )
        assert passed > 0, (
            f"no operators PASSed ROCTX/kernel coverage "
            f"(sampled={len(sampled)}, FAIL={len(failure_detail)}, SKIP={skipped})"
        )
    finally:
        test_utils.clean_output_dir(
            COVERAGE_TEST_CONFIG["cleanup"],
            workload_dir,
        )
        test_utils.clean_output_dir(
            COVERAGE_TEST_CONFIG["cleanup"],
            gt_work_dir,
        )


# -- ROCTX marker CSV parsing (helpers for :func:`parse_roctx_markers`) --


def _with_correlation_id_standard_name(marker_df: pd.DataFrame) -> pd.DataFrame:
    if "Correlation_Id" not in marker_df.columns:
        return marker_df
    return marker_df.rename(columns={"Correlation_Id": "Correlation_ID"})


def _leaf_from_function_cell(func: object) -> str | None:
    if not isinstance(func, str):
        return None
    op_path = func.split(":#")[0] if ":#" in func else func
    if "/" in op_path:
        leaf = op_path.rsplit("/", 1)[-1].strip()
    else:
        leaf = op_path.strip()
    return leaf or None


def _collect_marker_ops_and_correlations(
    marker_df: pd.DataFrame,
) -> tuple[set[str], dict[str, set]]:
    marker_ops: set[str] = set()
    op_to_corr: dict[str, set] = {}
    func_col = marker_df.get("Function")
    corr_col = marker_df.get("Correlation_ID")
    if func_col is None:
        return marker_ops, op_to_corr

    for row_index, function_cell in enumerate(func_col):
        leaf = _leaf_from_function_cell(function_cell)
        if leaf is None:
            continue
        marker_ops.add(leaf)
        if corr_col is None:
            continue
        correlation_id = corr_col.iloc[row_index]
        if not pd.notna(correlation_id):
            continue
        op_to_corr.setdefault(leaf, set()).add(correlation_id)

    return marker_ops, op_to_corr


def _merge_kernel_names_for_correlation_ids(
    correlation_ids: set,
    correlation_id_to_kernels: dict,
) -> set[str]:
    kernels: set[str] = set()
    for correlation_id in correlation_ids:
        kernels |= correlation_id_to_kernels.get(correlation_id, set())
    return kernels


def _kernels_by_marker_leaf(
    op_to_corr: dict[str, set],
    counter_files: list[Path],
) -> dict[str, set[str]]:
    counter_df = pd.concat(
        [pd.read_csv(f) for f in counter_files],
        ignore_index=True,
    )
    counter_df = _with_correlation_id_standard_name(counter_df)
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
        kernels = _merge_kernel_names_for_correlation_ids(
            correlation_ids_for_leaf,
            correlation_id_to_kernels,
        )
        if kernels:
            op_to_kernels[leaf] = kernels
    return op_to_kernels
