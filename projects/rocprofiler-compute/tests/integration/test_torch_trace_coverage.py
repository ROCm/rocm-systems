# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""ROCTX marker coverage for ``--torch-trace``.

``--coverage-n`` is the CUDA ``torch.ops`` sample budget (default 20).
Structural operators are always included as extras. Compares
``torch.profiler`` to the analyze call forest. Use ``--coverage-n 100``
for a nightly run. Requires a GPU.
"""

import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import common
import pytest

from tests.integration.common import require_torch

# Allow collection on CPU-only hosts.
try:
    import torch  # noqa: E402
except Exception:
    torch = None

COVERAGE_TEST_CONFIG: Dict[str, Any] = {"cleanup": True}

os.environ["ROCPROF"] = "rocprofiler-sdk"


@pytest.fixture
def torch_trace_coverage_sampling(request):
    seed = request.config.getoption("--coverage-seed")
    n = request.config.getoption("--coverage-n")
    if n < 0:
        pytest.fail("--coverage-n must be non-negative")
    return seed, n


@pytest.mark.torch_trace_coverage
def test_random_operator_kernel_coverage(
    request,
    binary_handler_profile_rocprof_compute,
    torch_trace_coverage_sampling,
):
    require_torch(gpu=True)
    from collections import Counter, defaultdict

    from torch_trace_coverage_utils import (
        call_trees_from_workload,
        categorize_skip_reason,
        compare_single_op,
        discover_operators,
        format_missing_arg_builder_report,
        format_skip_breakdown_lines,
        multiline_coverage_failure_warning,
        print_torch_trace_coverage_session_header,
        run_ground_truth_torch_profiler_subprocess,
        unique_output_param_id,
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
    aten_sample_count = min(sample_budget, len(aten_ops))
    sampled = rng.sample(aten_ops, aten_sample_count) + structural_ops

    print_torch_trace_coverage_session_header(
        seed,
        sample_budget,
        len(sampled),
        len(aten_ops),
        len(structural_ops),
        len(excluded_aten_ops),
    )

    ground_truth_dir = common.get_output_dir(
        param_id=unique_output_param_id("torch_trace_gt"),
        suffix="_tmp",
        clean_existing=True,
    )
    workload_dir = common.get_output_dir(
        param_id=unique_output_param_id("random_op_coverage"),
        clean_existing=True,
    )
    Path(ground_truth_dir).mkdir(parents=True, exist_ok=True)
    Path(workload_dir).mkdir(parents=True, exist_ok=True)

    ground_truth_path = str(Path(ground_truth_dir) / "ground_truth.json")
    workload_script_path = str(Path(ground_truth_dir) / "coverage_workload.py")
    ground_truth_runner_script_path = str(
        Path(ground_truth_dir) / "coverage_ground_truth_runner.py"
    )

    try:
        write_coverage_workload_artifacts(
            sampled,
            workload_script_path,
            ground_truth_runner_script_path,
        )

        run_ground_truth_torch_profiler_subprocess(
            ground_truth_runner_script_path,
            workload_script_path,
            ground_truth_path,
            coverage_seed=seed,
            coverage_sample_budget=sample_budget,
        )
        with open(ground_truth_path) as f:
            ground_truth = json.load(f)

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

        forest = call_trees_from_workload(workload_dir)

        failure_detail: List[Tuple[str, str]] = []
        skip_categories: "Counter[str]" = Counter()
        skip_op_names: Dict[str, List[str]] = defaultdict(list)
        passed = skipped = 0
        for op in sampled:
            outcome = compare_single_op(
                op,
                ground_truth,
                forest,
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

        if failure_detail:
            for line in multiline_coverage_failure_warning(
                failure_detail,
                max_ops=48,
                seed=seed,
                sample_budget=sample_budget,
            ).splitlines():
                print(line)
            pytest.fail(
                f"{len(failure_detail)} sampled op(s) failed analyze forest "
                f"coverage (seed={seed}, budget={sample_budget}). "
                f"First: {failure_detail[:5]!r}. "
                f"Summary: {passed} PASS / {len(failure_detail)} FAIL "
                f"/ {skipped} SKIP. Re-run with pytest -s for per-op lines."
            )
        assert passed > 0, (
            f"no operators PASSed forest coverage "
            f"(sampled={len(sampled)}, FAIL={len(failure_detail)}, SKIP={skipped})"
        )
    finally:
        common.clean_output_dir(
            COVERAGE_TEST_CONFIG["cleanup"],
            workload_dir,
        )
        common.clean_output_dir(
            COVERAGE_TEST_CONFIG["cleanup"],
            ground_truth_dir,
        )
