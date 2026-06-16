# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""ROCTX marker coverage test for ``inject_roctx``.

Samples a random subset of ATen operators plus structural entry points,
runs ``torch.profiler`` as ground truth, runs ``rocprof-compute
--torch-trace``, and compares marker output per operator. Sampling is
controlled by ``--coverage-seed`` and ``--coverage-n``. Requires GPU.
"""

import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import common
import pytest
from conftest import require_torch

# Allow collection on CPU-only hosts.
try:
    import torch  # noqa: E402
except Exception:
    torch = None

COVERAGE_TEST_CONFIG: Dict[str, Any] = {"cleanup": True}

# Set default profiler
os.environ["ROCPROF"] = "rocprofiler-sdk"


@pytest.fixture
def torch_trace_coverage_sampling(request):
    """Return ``(seed, sample_budget)`` for the coverage test."""
    seed = request.config.getoption("--coverage-seed")
    n = request.config.getoption("--coverage-n")
    if n < 0:
        pytest.fail("--coverage-n must be non-negative")
    return seed, n


@pytest.mark.torch_trace
def test_random_operator_kernel_coverage(
    request,
    binary_handler_profile_rocprof_compute,
    torch_trace_coverage_sampling,
):
    """Verify ``--torch-trace`` ROCTX output matches ``torch.profiler`` per operator.

    Per-operator mismatches fail. Per-operator skips are allowed, but at
    least one operator must pass overall.
    """
    require_torch(gpu=True)
    from collections import Counter, defaultdict

    from torch_trace_coverage_utils import (
        C_TIER_BACKWARD_SENTINELS,
        categorize_skip_reason,
        compare_single_op,
        detect_cpp_tier_signature,
        discover_operators,
        format_cpp_tier_signature_report,
        format_missing_arg_builder_report,
        format_skip_breakdown_lines,
        multiline_coverage_failure_warning,
        parse_roctx_markers,
        print_torch_trace_coverage_session_header,
        run_ground_truth_torch_profiler_subprocess,
        unique_get_output_param_id,
        write_coverage_workload_artifacts,
    )

    seed, sample_budget = torch_trace_coverage_sampling
    rng = random.Random(seed)

    match_verbose = os.getenv("ROCPROF_OPERATOR_MATCH_VERBOSE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    aten_ops, structural_ops, excluded_aten_ops = discover_operators()

    # The budget caps the ATen sample only; structural entries are always included.
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
        len(excluded_aten_ops),
    )

    gt_work_dir = common.get_output_dir(
        param_id=unique_get_output_param_id("torch_trace_gt"),
        suffix="_tmp",
        clean_existing=True,
    )
    workload_dir = common.get_output_dir(
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

        # Ground-truth run via torch.profiler.
        run_ground_truth_torch_profiler_subprocess(
            ground_truth_runner_script_path,
            workload_script_path,
            ground_truth_path,
            coverage_seed=seed,
            coverage_sample_budget=sample_budget,
        )
        with open(ground_truth_path) as f:
            ground_truth = json.load(f)

        # rocprof-compute --torch-trace run on the same workload.
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

        # Sentinel check: the only proof C++ tier ran (PASS works on either).
        cpp_tier_active, cpp_tier_sentinel_counts = detect_cpp_tier_signature(
            workload_dir,
        )
        cpp_tier_backward_active = any(
            cpp_tier_sentinel_counts.get(name, 0) > 0
            for name in C_TIER_BACKWARD_SENTINELS
        )

        failure_detail: List[Tuple[str, str]] = []
        skip_categories: "Counter[str]" = Counter()
        skip_op_names: Dict[str, List[str]] = defaultdict(list)
        passed = skipped = 0
        for op in sampled:
            outcome = compare_single_op(
                op,
                ground_truth,
                roctx_marker_names,
                roctx_kernels_map,
                match_verbose=match_verbose,
            )
            for line in outcome.log_lines:
                print(line)
            if outcome.status == "pass":
                passed += 1
            elif outcome.status == "fail":
                failure_detail.append((op.name, outcome.reason))
            else:
                skipped += 1
                category = categorize_skip_reason(outcome.reason)
                skip_categories[category] += 1
                skip_op_names[category].append(op.name)

        print(
            f"\n  Summary: {len(sampled)} ops — "
            f"{passed} PASS, {len(failure_detail)} FAIL, {skipped} SKIP"
        )
        breakdown_lines = format_skip_breakdown_lines(
            dict(skip_categories),
            skip_op_names=dict(skip_op_names),
        )
        for line in breakdown_lines:
            print(line)
        print()

        arg_gap_ops = skip_op_names.get("argument_builder_gap") or []
        if arg_gap_ops:
            for line in format_missing_arg_builder_report(arg_gap_ops):
                print(line)
            print()

        for line in format_cpp_tier_signature_report(
            cpp_tier_active,
            cpp_tier_sentinel_counts,
        ):
            print(line)
        print()

        if not cpp_tier_active:
            pytest.fail(
                "torch_trace coverage ran without the C++ tier in the "
                "workload subprocess."
            )

        if failure_detail:
            for line in multiline_coverage_failure_warning(
                failure_detail,
                max_ops=48,
                seed=seed,
                sample_budget=sample_budget,
            ).splitlines():
                print(line)
            pytest.fail(
                f"{len(failure_detail)} sampled op(s) failed ROCTX "
                f"coverage (seed={seed}, budget={sample_budget}). "
                f"First: {failure_detail[:5]!r}. "
                f"Summary: {passed} PASS / {len(failure_detail)} FAIL "
                f"/ {skipped} SKIP. Re-run with pytest -s for per-op lines."
            )
        assert passed > 0, (
            f"no operators PASSed ROCTX/kernel coverage "
            f"(sampled={len(sampled)}, FAIL={len(failure_detail)}, SKIP={skipped})"
        )
        assert cpp_tier_backward_active, (
            "no autograd.engine:0 / autograd.bwd:0 sentinels; "
            f"sentinel counts: {cpp_tier_sentinel_counts}"
        )
    finally:
        common.clean_output_dir(
            COVERAGE_TEST_CONFIG["cleanup"],
            workload_dir,
        )
        common.clean_output_dir(
            COVERAGE_TEST_CONFIG["cleanup"],
            gt_work_dir,
        )


@pytest.mark.torch_trace
def test_function_apply_wrappers_idempotent(monkeypatch):
    """A grandchild ``Function`` subclass does not get a second ``apply`` wrapper."""
    require_torch()

    try:
        from utils import inject_roctx
    except SystemExit:
        pytest.skip("roctx bindings are unavailable in this environment")

    push_counter = {"count": 0}

    def _count_push(*_args, **_kwargs):
        push_counter["count"] += 1

    monkeypatch.setattr(inject_roctx, "_push_scope", _count_push)
    monkeypatch.setattr(inject_roctx, "_pop_scope", lambda: None)

    class Foo(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            return x + 1

        @staticmethod
        def backward(ctx, grad_out):
            return grad_out

    class Bar(Foo):
        pass

    assert inject_roctx.install_function_apply_wrappers() is True

    assert getattr(
        getattr(Foo.__dict__.get("apply"), "__func__", None),
        "_roctx_wrapped",
        False,
    )
    assert "apply" not in Bar.__dict__

    x = torch.tensor(1.0, requires_grad=True)
    y = Bar.apply(x)
    y.backward()

    assert push_counter["count"] == 1, "Bar.apply triggered more than one wrapper push"
