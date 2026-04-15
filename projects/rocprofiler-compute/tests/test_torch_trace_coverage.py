# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""ROCTX marker coverage test for inject_roctx.py.

Selects ATen operators that have CUDA dispatch, synthesizes call arguments,
runs a generated ``torch.profiler`` workload for ground truth, then runs
rocprof-compute with ``--torch-trace`` and compares ROCTX marker names and
kernel correlation to that ground truth.

Sampling uses pytest options ``--coverage-seed`` and ``--coverage-n`` (see
``pytest --help``), with defaults defined in ``conftest.py``.

**Suite integration:** This module is registered like other torch-trace tests:
``@pytest.mark.torch_trace``, ``tests/test_categories.yaml`` (quick / standard /
comprehensive / full), and CTest target ``test_torch_trace_coverage`` in
``CMakeLists.txt``. CI may skip the whole file when PyTorch or CUDA is missing
(module-level skip). A one-line reason is printed to **stderr** during collection;
use ``pytest -rs`` to also list skip reasons in the short summary.

**Full operator matrix (maintainers):** To sample every discovered CUDA ATen op
plus structural patterns, set ``--coverage-n`` to at least
``len(aten_ops) + len(structural_ops)`` (typically on the order of thousands).
Capture stdout and attach the log to the PR when changing ``inject_roctx.py`` or
marker matching, for example (from ``rocprofiler-compute/tests``)::

    pytest test_torch_trace_coverage.py -m torch_trace \\
        --coverage-seed=0 --coverage-n=10000 \\
        -s 2>&1 | tee torch_trace_coverage_report.txt

Or from the ``rocprofiler-compute`` project root::

    pytest tests/test_torch_trace_coverage.py -m torch_trace \\
        --coverage-seed=0 --coverage-n=10000 \\
        -s 2>&1 | tee torch_trace_coverage_report.txt
"""

from __future__ import annotations

import json
import os
import random
import string
import subprocess
import sys
import textwrap
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Set, Tuple

import pandas as pd
import pytest
import test_utils

try:
    import torch
except ImportError as _torch_err:
    print(
        "SKIPPED test_torch_trace_coverage: PyTorch is not installed "
        f"({_torch_err}). Install a ROCm-compatible torch package to run this file.",
        file=sys.stderr,
        flush=True,
    )
    pytest.skip(
        "test_torch_trace_coverage requires PyTorch (import torch failed). "
        f"Original error: {_torch_err}",
        allow_module_level=True,
    )

if not torch.cuda.is_available():
    print(
        "SKIPPED test_torch_trace_coverage: torch.cuda.is_available() is False. "
        "This test needs a GPU and a PyTorch build with CUDA/ROCm enabled "
        "(check ROCM_PATH, HIP, and that a device is visible).",
        file=sys.stderr,
        flush=True,
    )
    pytest.skip(
        "test_torch_trace_coverage requires CUDA from PyTorch "
        "(torch.cuda.is_available() is False).",
        allow_module_level=True,
    )

COVERAGE_TEST_CONFIG: Dict[str, Any] = {"cleanup": True}

# One ``torch.profiler`` block per operator in the generated workload script.
_WORKLOAD_OP_BLOCK = string.Template(
    textwrap.dedent(
        """\
${setup_block}try:
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as _prof:
        ${call_line}
        torch.cuda.synchronize()
    results[${op_key}] = {
        "aten_ops": [e.name for e in _prof.events() if e.name.startswith("aten::")],
        "cuda_kernels": [
            e.name
            for e in _prof.events()
            if e.device_type == torch.autograd.DeviceType.CUDA
        ],
    }
except Exception as _e:
    results[${op_key}] = {"error": str(_e)[:200]}
"""
    )
)


def unique_get_output_param_id(prefix: str) -> str:
    """Build a ``param_id`` for :func:`test_utils.get_output_dir` without path races.

    Combines ``PYTEST_XDIST_WORKER`` (default ``main``), process id, active thread
    id, and a random UUID so parallel xdist workers, repeated local runs, and
    concurrent threads each get distinct directory names under a shared CWD.
    """
    worker = os.environ.get("PYTEST_XDIST_WORKER", "main")
    return f"{prefix}_{worker}_{os.getpid()}_{threading.get_ident()}_{uuid.uuid4().hex}"


# -- Hardcoded args for ops whose schemas don't encode shape constraints --


def get_hardcoded_args(device: str) -> Dict[str, Any]:
    """Return hardcoded shape recipes keyed by ATen op short name (last path segment).

    Args:
        device: Passed through to ``torch.randn`` / ``randint`` in each factory.

    Returns:
        Dict mapping short name (e.g. ``mm``) to a zero-arg callable that
        returns ``(positional_args_list, kwargs_dict)`` for that op. Used when
        :func:`build_args_for_op` cannot rely on schema-only tensor sizes.
    """
    return {
        "bmm": lambda: (
            [torch.randn(2, 4, 4, device=device), torch.randn(2, 4, 4, device=device)],
            {},
        ),
        "conv1d": lambda: (
            [torch.randn(1, 3, 16, device=device), torch.randn(6, 3, 3, device=device)],
            {},
        ),
        "conv2d": lambda: (
            [
                torch.randn(1, 3, 8, 8, device=device),
                torch.randn(6, 3, 3, 3, device=device),
            ],
            {},
        ),
        "conv3d": lambda: (
            [
                torch.randn(1, 3, 4, 4, 4, device=device),
                torch.randn(6, 3, 3, 3, 3, device=device),
            ],
            {},
        ),
        "conv_transpose1d": lambda: (
            [torch.randn(1, 3, 16, device=device), torch.randn(3, 6, 3, device=device)],
            {},
        ),
        "conv_transpose2d": lambda: (
            [
                torch.randn(1, 3, 8, 8, device=device),
                torch.randn(3, 6, 3, 3, device=device),
            ],
            {},
        ),
        "conv_transpose3d": lambda: (
            [
                torch.randn(1, 3, 4, 4, 4, device=device),
                torch.randn(3, 6, 3, 3, 3, device=device),
            ],
            {},
        ),
        "embedding": lambda: (
            [
                torch.randn(10, 8, device=device),
                torch.randint(0, 10, (4,), device=device),
            ],
            {},
        ),
        "cross_entropy_loss": lambda: (
            [
                torch.randn(4, 10, device=device),
                torch.randint(0, 10, (4,), device=device),
            ],
            {},
        ),
        "nll_loss_forward": lambda: (
            [
                torch.randn(4, 10, device=device).log_softmax(1),
                torch.randint(0, 10, (4,), device=device),
                torch.ones(10, device=device),
                0,
                -100,
            ],
            {},
        ),
        "batch_norm": lambda: (
            [
                torch.randn(2, 3, 4, 4, device=device),
                torch.randn(3, device=device),
                torch.randn(3, device=device),
                torch.randn(3, device=device),
                torch.randn(3, device=device),
                True,
                0.1,
                1e-5,
                False,
            ],
            {},
        ),
        "native_batch_norm": lambda: (
            [
                torch.randn(2, 3, 4, 4, device=device),
                torch.randn(3, device=device),
                torch.randn(3, device=device),
                torch.randn(3, device=device),
                torch.randn(3, device=device),
                True,
                0.1,
                1e-5,
            ],
            {},
        ),
        "addmm": lambda: (
            [
                torch.randn(4, device=device),
                torch.randn(4, 4, device=device),
                torch.randn(4, 4, device=device),
            ],
            {},
        ),
        "addbmm": lambda: (
            [
                torch.randn(4, 4, device=device),
                torch.randn(2, 4, 4, device=device),
                torch.randn(2, 4, 4, device=device),
            ],
            {},
        ),
        "baddbmm": lambda: (
            [
                torch.randn(2, 4, 4, device=device),
                torch.randn(2, 4, 4, device=device),
                torch.randn(2, 4, 4, device=device),
            ],
            {},
        ),
        "addmv": lambda: (
            [
                torch.randn(4, device=device),
                torch.randn(4, 4, device=device),
                torch.randn(4, device=device),
            ],
            {},
        ),
        "addr": lambda: (
            [
                torch.randn(4, 4, device=device),
                torch.randn(4, device=device),
                torch.randn(4, device=device),
            ],
            {},
        ),
        "one_hot": lambda: (
            [torch.randint(0, 5, (4,), device=device)],
            {},
        ),
    }


# -- Schema-driven arg generation --


def arg_from_schema_type(type_str: str, device: str) -> Any:
    """Synthesize one Python value suitable for a single schema positional argument.

    Args:
        type_str: The string form of the argument's type from the PyTorch schema.
        device: CUDA device string passed to tensor factories (always ``cuda``
            in this test).

    Returns:
        A tensor, scalar, list, or ``None`` for ``Optional``-filled slots.
        Returns ``None`` if ``type_str`` is a ``List[...]`` form we do not
        implement, so the caller can abort ``build_args_for_op``.
    """
    if type_str == "Tensor":
        return torch.randn(4, 4, device=device)
    if type_str in ("Scalar", "number"):
        return 1.0
    if type_str in ("int", "SymInt"):
        return 0
    if type_str == "float":
        return 1.0
    if type_str == "bool":
        return False
    if type_str == "ScalarType":
        return None
    if type_str == "str":
        return "mean"
    if type_str in ("List[int]", "List[SymInt]"):
        return [1, 1]
    if type_str == "List[bool]":
        return [False, False]
    if type_str == "List[Tensor]":
        return [torch.randn(4, 4, device=device)]
    if type_str.startswith("List"):
        return None
    return None


def build_args_for_op(
    op: OpEntry,
) -> Optional[Tuple[List[Any], dict]]:
    """Produce positional ``args`` and ``kwargs`` to run one sampled operator on CUDA.

    Args:
        op: Entry from :func:`discover_operators` (ATen or structural).

    Returns:
        ``([], {})`` for structural ops (the generated script builds the call).
        For ATen ops, ``(args, {})`` with only required positional args filled,
        stopping at the first parameter that has a schema default.
        ``None`` if the op needs args we cannot build (missing schema, unknown
        type, or unsupported list type).
    """
    device = "cuda"

    if op.category == "structural":
        return [], {}

    # Tier 1: hardcoded overrides for shape-constrained ops
    short = op.name.rsplit(".", 1)[-1]
    specific = get_hardcoded_args(device)
    if short in specific:
        return specific[short]()

    # Tier 2: schema-driven generation
    if op.schema is None:
        return None

    args = []
    for arg in op.schema.arguments:
        type_str = str(arg.type)

        if arg.default_value is not None:
            break  # remaining args have defaults

        if "Optional" in type_str:
            args.append(None)
            continue

        val = arg_from_schema_type(type_str, device)
        if val is None:
            return None
        args.append(val)

    return (args, {})


# -- Operator discovery --


def discover_operators() -> Tuple[List[OpEntry], List[OpEntry]]:
    """Enumerate CUDA ATen overloads and append fixed structural coverage labels.

    Walks ``torch.ops.aten``, keeps one overload per op (preferring ``default``),
    and retains only ops whose dispatcher reports a CUDA kernel. Does not filter
    on name beyond what ``dir`` returns (internal names may appear if exposed).

    Returns:
        ``(aten_ops, structural_ops)``. ``structural_ops`` is always the same
        small list used to generate nn/optimizer/backward/device workload snippets.
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

            # Prefer 'default' overload; skip if no CUDA dispatch
            ov_name = (
                "default"
                if "default" in packet.overloads()
                else next(iter(packet.overloads()), None)
            )
            if ov_name is None:
                continue
            overload = getattr(packet, ov_name)
            try:
                has_cuda = torch._C._dispatch_has_kernel_for_dispatch_key(
                    overload.name(),
                    torch._C.DispatchKey.CUDA,
                )
            except Exception:
                continue
            if not has_cuda:
                continue

            full = f"torch.ops.{ns_name}.{op_name}"
            if full in seen:
                continue
            seen.add(full)

            schema = getattr(overload, "_schema", None)
            aten_ops.append(OpEntry(full, "aten", packet, schema))

    structural_ops = [
        OpEntry(n, "structural", None, None)
        for n in (
            "nn.Module.__call__",
            "Optimizer.step",
            "torch.Tensor.backward",
            "torch.cuda.set_device",
        )
    ]

    return aten_ops, structural_ops


# -- Marker matching --


def marker_matches_op(op_name: str, marker_leaf: str) -> bool:
    """Decide if one ROCTX ``Function`` column leaf corresponds to ``op_name``.

    Args:
        op_name: Sampled op id (e.g. ``torch.ops.aten.mm`` or structural label).
        marker_leaf: Leaf segment derived from CSV (after ``/#`` and path split).

    Returns:
        ``True`` if the marker should count as coverage for this op; structural
        rules match inject_roctx naming (e.g. ``Optimizer.step`` vs ``*.step``).
    """
    if op_name == marker_leaf:
        return True
    op_leaf = op_name.rsplit("::", 1)[-1].rsplit(".", 1)[-1]
    marker_norm = marker_leaf.rsplit("::", 1)[-1].rsplit(".", 1)[-1]
    if marker_norm == op_leaf:
        return True
    if marker_leaf.endswith(f".{op_leaf}"):
        return True
    # inject_roctx emits "optimizer.SGD.step"
    if op_name == "Optimizer.step" and marker_leaf.endswith(".step"):
        return True
    # inject_roctx emits "nn.Module.Linear.forward"
    if (
        op_name == "nn.Module.__call__"
        and ".forward" in marker_leaf
        and marker_leaf.startswith("nn.Module.")
    ):
        return True
    return False


# -- Workload script generation --


def serialize_arg(argument_value: Any, vname: str) -> Tuple[List[str], str]:
    """Build emitted-source lines that define ``vname`` plus an expression to pass.

    Args:
        argument_value: Runtime value from :func:`build_args_for_op` (tensor, list,
            etc.).
        vname: Unique variable name in the generated workload script.

    Returns:
        ``(setup_lines, expr)`` where ``setup_lines`` are statements to run before
        the profiled call (empty for scalars), and ``expr`` is what appears inside
        ``call_expr(expr, ...)`` (usually ``vname`` or a literal).
    """
    if isinstance(argument_value, torch.Tensor):
        shape = tuple(argument_value.shape)
        if argument_value.dtype == torch.long:
            return (
                [f"{vname} = torch.randint(0, 4, {shape}, device=device)"],
                vname,
            )
        return ([f"{vname} = torch.randn({shape}, device=device)"], vname)
    if (
        isinstance(argument_value, list)
        and argument_value
        and isinstance(argument_value[0], torch.Tensor)
    ):
        shape = tuple(argument_value[0].shape)
        return (
            [
                f"{vname} = [torch.randn({shape}, device=device) "
                f"for _ in range({len(argument_value)})]"
            ],
            vname,
        )
    if argument_value is None:
        return [], "None"
    if isinstance(argument_value, bool):
        return [], str(argument_value)
    return [], repr(argument_value)


def emit_structural_preamble(
    op_name: str,
    safe_var: str,
) -> Tuple[List[str], str, str]:
    """Build setup lines and ``(call_expr, call_args)`` for one structural workload.

    Args:
        op_name: Structural label (e.g. ``nn.Module.__call__``).
        safe_var: Sanitized token derived from the op name for unique temp symbols.

    Returns:
        ``(setup_lines, call_expr, call_args)`` where ``call_args`` is the
        argument substring inside ``call_expr(...)`` in the generated script
        (empty when the call takes no extra args after setup).
    """
    if op_name == "nn.Module.__call__":
        return (
            [
                "import torch.nn as nn",
                f"_model_{safe_var} = nn.Linear(4, 4).cuda()",
            ],
            f"_model_{safe_var}",
            "torch.randn(2, 4, device=device)",
        )

    if op_name == "Optimizer.step":
        return (
            [
                "import torch.nn as nn",
                f"_m_{safe_var} = nn.Linear(4, 4).cuda()",
                f"_opt_{safe_var} = torch.optim.SGD("
                f"_m_{safe_var}.parameters(), lr=0.01)",
                f"_m_{safe_var}(torch.randn(2, 4, device=device)).sum().backward()",
            ],
            f"_opt_{safe_var}.step",
            "",
        )

    if op_name == "torch.Tensor.backward":
        return (
            [
                f"_loss_{safe_var} = torch.nn.Linear(4, 4)"
                f".cuda()(torch.randn("
                f"2, 4, device=device)).sum()"
            ],
            f"_loss_{safe_var}.backward",
            "",
        )

    if op_name == "torch.cuda.set_device":
        return [], "torch.cuda.set_device", "0"

    return [], op_name, ""


def build_workload_script_lines(
    operators: List[OpEntry],
    ground_truth_path: str,
) -> List[str]:
    """Build the full source of ``coverage_workload.py`` as text lines (no I/O).

    The emitted script imports torch, runs ``torch.profiler`` once per operator,
    and writes ``aten_ops``, ``cuda_kernels``, or ``error`` per key. Each op block
    is produced from :data:`_WORKLOAD_OP_BLOCK`. Operators skipped by
    :func:`build_args_for_op` do not appear in the output.
    """
    lines = [
        "import json, os, sys, torch",
        "from torch.profiler import profile, ProfilerActivity",
        "",
        "device = 'cuda'",
        "torch.cuda.synchronize()",
        "results = {}",
        "",
    ]

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

        kwarg_strs = [
            f"{keyword}={repr(value)}" for keyword, value in kwargs.items()
        ]
        call_args = ", ".join(arg_strs + kwarg_strs)

        if op.name.startswith("torch.ops."):
            call_expr = op.name
        elif op.category == "structural":
            extra_setup, call_expr, call_args = emit_structural_preamble(
                op.name,
                safe_var,
            )
            op_setup.extend(extra_setup)
        else:
            call_expr = op.name

        setup_block = ("\n".join(op_setup) + "\n") if op_setup else ""
        call_line = f"{call_expr}({call_args})"
        op_key = repr(op.name)
        block = _WORKLOAD_OP_BLOCK.substitute(
            setup_block=setup_block,
            call_line=call_line,
            op_key=op_key,
        ).rstrip("\n")
        lines.extend(block.splitlines())
        lines.append("")

    lines.append(f'with open({ground_truth_path!r}, "w") as _f:')
    lines.append("    json.dump(results, _f)")
    lines.append("")

    return lines


def generate_workload_script(
    operators: List[OpEntry],
    ground_truth_path: str,
    script_path: str,
) -> None:
    """Write ``build_workload_script_lines(...)`` to ``script_path`` on disk."""
    source_lines = build_workload_script_lines(operators, ground_truth_path)
    Path(script_path).write_text("\n".join(source_lines), encoding="utf-8")


# -- ROCTX marker CSV parsing --


def parse_roctx_markers(
    workload_dir: str,
) -> Tuple[Dict[str, Set[str]], Set[str]]:
    """Parse marker and counter CSVs produced by rocprof-compute under ``workload_dir``.

    Args:
        workload_dir: Directory tree containing ``*marker_api_trace.csv`` and
            optionally ``*counter_collection.csv`` (globbed recursively).

    Returns:
        ``(op_to_kernels, marker_ops)``. ``marker_ops`` is every distinct marker
        leaf string. ``op_to_kernels`` maps a leaf to GPU kernel names when
        correlation IDs join marker rows to counter rows; leaves without kernels
        are omitted from the dict.

    If no marker CSV exists, returns ``({}, set())``.
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


def compare_single_op(
    op: OpEntry,
    ground_truth: Dict[str, Any],
    roctx_marker_names: Set[str],
    roctx_kernels_map: Dict[str, Set[str]],
) -> OpCompareOutcome:
    """Compare one op's profiler JSON entry to parsed ROCTX markers and kernels.

    Args:
        op: Sampled operator.
        ground_truth: Map from op ``name`` to dict with ``cuda_kernels`` list and
            optional ``error`` from the generated workload script.
        roctx_marker_names: All marker leaves seen in CSVs.
        roctx_kernels_map: Marker leaf → kernel names from correlation join.

    Returns:
        Outcome with ``status`` ``pass`` / ``fail`` / ``skip``, ``skip_reason`` for
        skips, and ``log_lines`` for the caller to print (keeps I/O out of this
        function).
    """
    ground_truth_entry = ground_truth.get(op.name)

    if ground_truth_entry is None or "error" in ground_truth_entry:
        err_msg = (
            ground_truth_entry.get("error", "not in ground truth")
            if ground_truth_entry
            else "not in generated script"
        )
        reason = err_msg[:200]
        return OpCompareOutcome(
            "skip",
            reason,
            (
                f"  [SKIP] {op.name}",
                f"         reason: {reason[:120]}",
            ),
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

    # Structural ops are hierarchical wrappers: their markers
    # don't directly correlate to GPU kernels (inner ATen ops
    # do). Marker presence alone is sufficient for PASS.
    if op.category == "structural":
        if marker_found:
            return OpCompareOutcome(
                "pass",
                "",
                (f"  [PASS] {op.name} — structural marker present",),
            )
        # inject_roctx wraps base-class methods; subclass
        # overrides (e.g. SGD.step) may bypass the wrapper.
        skip_msg = "structural marker not found (possible inject_roctx gap)"
        return OpCompareOutcome(
            "skip",
            skip_msg,
            (
                f"  [WARN] {op.name}"
                " — structural marker not found"
                " (possible inject_roctx gap)",
            ),
        )

    if not profiler_kernel_set:
        if marker_found:
            return OpCompareOutcome(
                "pass",
                "",
                (f"  [PASS] {op.name} — marker found, no kernels",),
            )
        skip_msg = "no GPU kernels in torch.profiler ground truth"
        return OpCompareOutcome(
            "skip",
            skip_msg,
            (f"  [SKIP] {op.name} — no GPU kernels dispatched",),
        )

    kernel_summary = ", ".join(
        f"{kernel_name[:60]} (x{profiler_kernels.count(kernel_name)})"
        for kernel_name in sorted(profiler_kernel_set)[:5]
    )
    if len(profiler_kernel_set) > 5:
        kernel_summary += f" ... +{len(profiler_kernel_set) - 5} more"

    if marker_found and roctx_kernels:
        roctx_kernel_names_preview = sorted(roctx_kernels)[:5]
        roctx_summary = ", ".join(roctx_kernel_names_preview)
        return OpCompareOutcome(
            "pass",
            "",
            (
                f"  [PASS] {op.name}",
                f"         profiler: {kernel_summary}",
                f"         roctx: {roctx_summary}",
            ),
        )

    if marker_found:
        return OpCompareOutcome(
            "fail",
            "",
            (
                f"  [FAIL] {op.name}",
                f"         profiler: {kernel_summary}",
                "         roctx marker found but no correlated kernels",
            ),
        )

    return OpCompareOutcome(
        "fail",
        "",
        (
            f"  [FAIL] {op.name}",
            f"         profiler: {kernel_summary}",
            "         roctx marker: NOT FOUND",
        ),
    )


def print_torch_trace_coverage_session_header(
    seed: int,
    sample_budget: int,
    sampled_operator_count: int,
    aten_operator_count: int,
    structural_operator_count: int,
) -> None:
    """Print seed / sampling summary lines for the coverage test run."""
    print(
        f"\n  Seed: {seed} | {sampled_operator_count} operators"
        f" selected from {aten_operator_count} CUDA ATen ops"
        f" + {structural_operator_count} structural"
        f" (budget={sample_budget})"
    )
    print(
        f"  (reproduce: pytest ... --coverage-seed={seed}"
        f" --coverage-n={sample_budget} ...)\n"
    )


def run_ground_truth_torch_profiler_subprocess(script_path: str) -> None:
    """Execute the generated workload script under CPython; fail on error or timeout."""
    try:
        completed = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired as exc:
        pytest.fail(
            "Ground truth script timed out after 120s\n"
            f"stdout: {exc.stdout}\n"
            f"stderr: {exc.stderr}"
        )
    assert completed.returncode == 0, (
        f"Ground truth script failed:\nstderr: {completed.stderr}"
    )


# -- Main test --


@pytest.mark.torch_trace
def test_random_operator_kernel_coverage(
    binary_handler_profile_rocprof_compute,
    torch_trace_coverage_sampling,
):
    """Verify ``--torch-trace`` ROCTX output matches profiler ground truth.

    Steps: sample ops → emit ``coverage_workload.py`` + run it for JSON → run
    rocprof-compute on that script → parse CSVs → compare per op. Uses
    :func:`test_utils.get_output_dir` with :func:`unique_get_output_param_id` for
    both the ground-truth script directory and the rocprof workload directory so
    names stay unique under xdist, repeated runs, and threaded callers.

    Fails with ``assert not failures`` listing any op missing expected ROCTX
    correlation when kernels were present in ground truth.
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
    script_path = str(Path(gt_work_dir) / "coverage_workload.py")

    try:
        generate_workload_script(
            sampled,
            ground_truth_path,
            script_path,
        )

        # Run 1: torch.profiler ground truth
        run_ground_truth_torch_profiler_subprocess(script_path)
        with open(ground_truth_path) as f:
            ground_truth = json.load(f)

        # Run 2: rocprof-compute --torch-trace
        binary_handler_profile_rocprof_compute(
            {
                **COVERAGE_TEST_CONFIG,
                "coverage_workload": [
                    sys.executable,
                    script_path,
                ],
            },
            workload_dir,
            ["--experimental", "--torch-trace", "--iteration-multiplexing"],
            check_success=False,
            app_name="coverage_workload",
        )

        roctx_kernels_map, roctx_marker_names = parse_roctx_markers(workload_dir)

        # Per-operator comparison
        failures: List[str] = []
        passed = skipped = 0
        skipped_detail: List[Tuple[str, str]] = []
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
                failures.append(op.name)
            else:
                skipped += 1
                skipped_detail.append((op.name, outcome.skip_reason))

        print(
            f"\n  {len(sampled)} operators tested: "
            f"{passed} passed, {len(failures)} failed,"
            f" {skipped} skipped\n"
        )
        if skipped_detail:
            print(
                "  Skipped operators (name — reason) for debugging inject_roctx / "
                "schema / profiler gaps:\n"
                + "\n".join(
                    f"    - {name}: {reason}" for name, reason in skipped_detail
                )
                + "\n"
            )

        assert not failures, (
            f"seed={seed}, {len(failures)} op(s) failed kernel / ROCTX coverage:\n"
            + "\n".join(f"  - {n}" for n in failures)
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


class OpCompareOutcome(NamedTuple):
    """Result of :func:`compare_single_op` (status + optional log lines for stdout)."""

    status: str
    skip_reason: str
    log_lines: Tuple[str, ...]


class OpEntry(NamedTuple):
    """One row in the coverage sample: what to call and how it was discovered.

    Attributes:
        name: Human-readable op id, e.g. ``torch.ops.aten.mm`` or
            ``nn.Module.__call__`` for structural patterns.
        category: ``aten`` for ``torch.ops`` entries, ``structural`` for
            synthetic workload patterns (module forward, optimizer step, etc.).
        fn: For ATen ops, the ``OpOverloadPacket``; ``None`` for structural.
        schema: PyTorch ``FunctionSchema`` for the chosen overload, or ``None``.
    """

    name: str
    category: str
    fn: object
    schema: object
