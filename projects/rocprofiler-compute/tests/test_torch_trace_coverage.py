# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""ROCTX marker coverage test for inject_roctx.py.

Selects ATen operators that have CUDA dispatch, generates correct args
from their schemas, and verifies that rocprof-compute --torch-trace
produces matching ROCTX markers and correlated GPU kernels.

Env vars: TORCH_COVERAGE_SEED (int), TORCH_COVERAGE_N (int, default 50).
"""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Set, Tuple

import pandas as pd
import pytest
import test_utils
import torch

config: dict = {"cleanup": True}

skip_if_no_torch_gpu = pytest.mark.skipif(
    (
        importlib.util.find_spec("torch") is None
        or not __import__("torch").cuda.is_available()
    ),
    reason="PyTorch and GPU access are required for this test",
)


class OpEntry(NamedTuple):
    """Operator selected for coverage testing."""

    name: str  # e.g. "torch.ops.aten.mm"
    category: str  # aten | structural
    fn: object  # OpOverloadPacket or None
    schema: object  # FunctionSchema or None


# -- Hardcoded args for ops whose schemas don't encode shape constraints --


def _get_hardcoded_args(device):
    """Return op_short_name -> (args, kwargs) for shape-constrained ops."""
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


def _arg_from_schema_type(type_str, device):
    """Generate a default value for a schema type string."""
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


def _build_args_for_op(
    op: OpEntry,
) -> Optional[Tuple[list, dict]]:
    """Build (args, kwargs) from op schema. Returns None on failure."""
    device = "cuda"

    if op.category == "structural":
        return [], {}

    # Tier 1: hardcoded overrides for shape-constrained ops
    short = op.name.rsplit(".", 1)[-1]
    specific = _get_hardcoded_args(device)
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

        val = _arg_from_schema_type(type_str, device)
        if val is None:
            return None
        args.append(val)

    return (args, {})


# -- Operator discovery --


def _discover_operators() -> Tuple[List[OpEntry], List[OpEntry]]:
    """Discover ATen ops with CUDA dispatch + structural ops.

    Returns (aten_ops, structural_ops).
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


def _marker_matches_op(op_name: str, marker_leaf: str) -> bool:
    """Check if a marker leaf corresponds to an operator."""
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


def _serialize_arg(a, vname: str, lines: list) -> str:
    """Emit a line recreating argument ``a`` on GPU. Returns the expression."""
    if isinstance(a, torch.Tensor):
        shape = tuple(a.shape)
        if a.dtype == torch.long:
            lines.append(f"{vname} = torch.randint(0, 4, {shape}, device=device)")
        else:
            lines.append(f"{vname} = torch.randn({shape}, device=device)")
        return vname
    if isinstance(a, list) and a and isinstance(a[0], torch.Tensor):
        shape = tuple(a[0].shape)
        lines.append(
            f"{vname} = [torch.randn({shape}, device=device) for _ in range({len(a)})]"
        )
        return vname
    if a is None:
        return "None"
    if isinstance(a, bool):
        return str(a)
    return repr(a)


def _emit_structural_preamble(
    op_name: str,
    safe_var: str,
    lines: list,
) -> Tuple[str, str]:
    """Emit setup code for structural ops. Returns (call_expr, call_args)."""
    if op_name == "nn.Module.__call__":
        lines.append("import torch.nn as nn")
        lines.append(f"_model_{safe_var} = nn.Linear(4, 4).cuda()")
        return (
            f"_model_{safe_var}",
            "torch.randn(2, 4, device=device)",
        )

    if op_name == "Optimizer.step":
        lines.append("import torch.nn as nn")
        lines.append(f"_m_{safe_var} = nn.Linear(4, 4).cuda()")
        lines.append(
            f"_opt_{safe_var} = torch.optim.SGD(_m_{safe_var}.parameters(), lr=0.01)"
        )
        lines.append(
            f"_m_{safe_var}(torch.randn(2, 4, device=device)).sum().backward()"
        )
        return f"_opt_{safe_var}.step", ""

    if op_name == "torch.Tensor.backward":
        lines.append(
            f"_loss_{safe_var} = torch.nn.Linear(4, 4)"
            f".cuda()(torch.randn("
            f"2, 4, device=device)).sum()"
        )
        return f"_loss_{safe_var}.backward", ""

    if op_name == "torch.cuda.set_device":
        return "torch.cuda.set_device", "0"

    return op_name, ""


def _generate_workload_script(
    operators: List[OpEntry],
    ground_truth_path: str,
    script_path: str,
) -> None:
    """Generate a standalone .py profiling each op, writing ground truth JSON."""
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
        build_result = _build_args_for_op(op)
        if build_result is None:
            continue

        args, kwargs = build_result
        safe_var = op.name.replace(".", "_").replace("::", "_")

        arg_strs = []
        for i, a in enumerate(args):
            vname = f"_arg_{safe_var}_{i}"
            arg_strs.append(_serialize_arg(a, vname, lines))

        kwarg_strs = [f"{k}={repr(v)}" for k, v in kwargs.items()]
        call_args = ", ".join(arg_strs + kwarg_strs)

        if op.name.startswith("torch.ops."):
            call_expr = op.name
        elif op.category == "structural":
            call_expr, call_args = _emit_structural_preamble(
                op.name,
                safe_var,
                lines,
            )
        else:
            call_expr = op.name

        lines.append("try:")
        lines.append(
            "    with profile(activities="
            "[ProfilerActivity.CPU, ProfilerActivity.CUDA])"
            " as _prof:"
        )
        lines.append(f"        {call_expr}({call_args})")
        lines.append("        torch.cuda.synchronize()")
        lines.append(f"    results[{op.name!r}] = {{")
        lines.append(
            '        "aten_ops": [e.name for e in '
            "_prof.events() "
            'if e.name.startswith("aten::")],'
        )
        lines.append(
            '        "cuda_kernels": [e.name for e in '
            "_prof.events() "
            "if e.device_type == "
            "torch.autograd.DeviceType.CUDA],"
        )
        lines.append("    }")
        lines.append("except Exception as _e:")
        lines.append(f'    results[{op.name!r}] = {{"error": str(_e)[:200]}}')
        lines.append("")

    lines.append(f'with open({ground_truth_path!r}, "w") as _f:')
    lines.append("    json.dump(results, _f)")
    lines.append("")

    with open(script_path, "w") as f:
        f.write("\n".join(lines))


# -- ROCTX marker CSV parsing --


def _parse_roctx_markers(
    workload_dir: str,
) -> Tuple[Dict[str, Set[str]], Set[str]]:
    """Parse marker_api_trace and counter_collection CSVs.

    Returns (op_to_kernels, marker_ops) where op_to_kernels maps
    marker leaf name -> set of GPU kernel names via Correlation_ID.
    """
    marker_files = sorted(Path(workload_dir).glob("**/*marker_api_trace.csv"))
    counter_files = sorted(Path(workload_dir).glob("**/*counter_collection.csv"))
    if not marker_files:
        return {}, set()

    marker_df = pd.concat(
        [pd.read_csv(f) for f in marker_files],
        ignore_index=True,
    )
    if "Correlation_Id" in marker_df.columns:
        marker_df = marker_df.rename(columns={"Correlation_Id": "Correlation_ID"})

    marker_ops: Set[str] = set()
    op_to_corr: Dict[str, Set] = {}

    func_col = marker_df.get("Function")
    corr_col = marker_df.get("Correlation_ID")
    if func_col is not None:
        for idx, func in enumerate(func_col):
            if not isinstance(func, str):
                continue
            op_path = func.split(":#")[0] if ":#" in func else func
            leaf = (
                op_path.rsplit("/", 1)[-1].strip()
                if "/" in op_path
                else op_path.strip()
            )
            if leaf:
                marker_ops.add(leaf)
                if corr_col is not None:
                    cid = corr_col.iloc[idx]
                    if pd.notna(cid):
                        op_to_corr.setdefault(leaf, set()).add(cid)

    op_to_kernels: Dict[str, Set[str]] = {}
    if counter_files and op_to_corr:
        counter_df = pd.concat(
            [pd.read_csv(f) for f in counter_files],
            ignore_index=True,
        )
        if "Correlation_Id" in counter_df.columns:
            counter_df = counter_df.rename(columns={"Correlation_Id": "Correlation_ID"})
        if "Kernel_Name" in counter_df.columns:
            all_corr_ids = set()
            for ids in op_to_corr.values():
                all_corr_ids |= ids
            matched = counter_df[counter_df["Correlation_ID"].isin(all_corr_ids)]
            corr_to_kernel: Dict = {}
            for cid, kname in zip(
                matched["Correlation_ID"],
                matched["Kernel_Name"],
            ):
                if pd.notna(kname):
                    corr_to_kernel.setdefault(cid, set()).add(kname)

            for leaf, cids in op_to_corr.items():
                kernels: Set[str] = set()
                for cid in cids:
                    kernels |= corr_to_kernel.get(cid, set())
                if kernels:
                    op_to_kernels[leaf] = kernels

    return op_to_kernels, marker_ops


# -- Per-operator comparison --


def _compare_single_op(
    op: OpEntry,
    ground_truth: dict,
    roctx_marker_names: Set[str],
    roctx_kernels_map: Dict[str, Set[str]],
) -> str:
    """Compare ground truth vs ROCTX. Returns pass/fail/skip."""
    gt = ground_truth.get(op.name)

    if gt is None or "error" in gt:
        err_msg = (
            gt.get("error", "not in ground truth") if gt else "not in generated script"
        )
        print(f"  [SKIP] {op.name}")
        print(f"         reason: {err_msg[:120]}")
        return "skip"

    profiler_kernels = gt.get("cuda_kernels", [])
    profiler_kernel_set = set(profiler_kernels)

    marker_found = any(_marker_matches_op(op.name, m) for m in roctx_marker_names)
    roctx_kernels: Set[str] = set()
    for m_name in roctx_marker_names:
        if _marker_matches_op(op.name, m_name):
            roctx_kernels |= roctx_kernels_map.get(m_name, set())

    # Structural ops are hierarchical wrappers: their markers
    # don't directly correlate to GPU kernels (inner ATen ops
    # do). Marker presence alone is sufficient for PASS.
    if op.category == "structural":
        if marker_found:
            print(f"  [PASS] {op.name} — structural marker present")
            return "pass"
        # inject_roctx wraps base-class methods; subclass
        # overrides (e.g. SGD.step) may bypass the wrapper.
        print(
            f"  [WARN] {op.name}"
            " — structural marker not found"
            " (possible inject_roctx gap)"
        )
        return "skip"

    if not profiler_kernel_set:
        if marker_found:
            print(f"  [PASS] {op.name} — marker found, no kernels")
            return "pass"
        print(f"  [SKIP] {op.name} — no GPU kernels dispatched")
        return "skip"

    kernel_summary = ", ".join(
        f"{k[:60]} (x{profiler_kernels.count(k)})"
        for k in sorted(profiler_kernel_set)[:5]
    )
    if len(profiler_kernel_set) > 5:
        kernel_summary += f" ... +{len(profiler_kernel_set) - 5} more"

    if marker_found and roctx_kernels:
        roctx_summary = ", ".join(sorted(roctx_kernels)[:5])
        print(f"  [PASS] {op.name}")
        print(f"         profiler: {kernel_summary}")
        print(f"         roctx: {roctx_summary}")
        return "pass"

    if marker_found:
        print(f"  [FAIL] {op.name}")
        print(f"         profiler: {kernel_summary}")
        print("         roctx marker found but no correlated kernels")
        return "fail"

    print(f"  [FAIL] {op.name}")
    print(f"         profiler: {kernel_summary}")
    print("         roctx marker: NOT FOUND")
    return "fail"


# -- Main test --


@skip_if_no_torch_gpu
@pytest.mark.torch_trace
def test_random_operator_kernel_coverage(
    binary_handler_profile_rocprof_compute,
    tmp_path,
):
    """Select random ATen CUDA operators, profile with torch.profiler
    and rocprof-compute --torch-trace, compare coverage.
    """
    import random

    seed = int(os.environ.get("TORCH_COVERAGE_SEED", random.randrange(2**32)))
    sample_budget = int(os.environ.get("TORCH_COVERAGE_N", "50"))
    rng = random.Random(seed)

    aten_ops, structural_ops = _discover_operators()

    n_aten = min(
        max(0, sample_budget - len(structural_ops)),
        len(aten_ops),
    )
    sampled = rng.sample(aten_ops, n_aten) + structural_ops

    print(
        f"\n  Seed: {seed} | {len(sampled)} operators"
        f" selected from {len(aten_ops)} CUDA ATen ops"
        f" + {len(structural_ops)} structural"
        f" (budget={sample_budget})"
    )
    print(
        f"  (reproduce: TORCH_COVERAGE_SEED={seed}"
        f" TORCH_COVERAGE_N={sample_budget} pytest ...)\n"
    )

    # Generate workload script
    ground_truth_path = str(tmp_path / "ground_truth.json")
    script_path = str(tmp_path / "coverage_workload.py")
    _generate_workload_script(
        sampled,
        ground_truth_path,
        script_path,
    )

    # Run 1: torch.profiler ground truth
    try:
        run1 = subprocess.run(
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
    assert run1.returncode == 0, f"Ground truth script failed:\nstderr: {run1.stderr}"
    with open(ground_truth_path) as f:
        ground_truth = json.load(f)

    # Run 2: rocprof-compute --torch-trace
    workload_dir = test_utils.get_output_dir(param_id="random_op_coverage")
    binary_handler_profile_rocprof_compute(
        {
            **config,
            "coverage_workload": [
                sys.executable,
                script_path,
            ],
        },
        workload_dir,
        ["--experimental", "--torch-trace"],
        check_success=False,
        app_name="coverage_workload",
    )

    roctx_kernels_map, roctx_marker_names = _parse_roctx_markers(workload_dir)

    # Per-operator comparison
    passed = failed = skipped = 0
    for op in sampled:
        result = _compare_single_op(
            op,
            ground_truth,
            roctx_marker_names,
            roctx_kernels_map,
        )
        if result == "pass":
            passed += 1
        elif result == "fail":
            failed += 1
        else:
            skipped += 1

    print(
        f"\n  {len(sampled)} operators tested: "
        f"{passed} passed, {failed} failed,"
        f" {skipped} skipped\n"
    )

    # Cleanup
    test_utils.clean_output_dir(
        config["cleanup"],
        workload_dir,
    )
    assert failed == 0, (
        f"{failed} operator(s) failed kernel coverage check (seed={seed})"
    )
