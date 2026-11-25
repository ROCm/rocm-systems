# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""ROCTX marker coverage for ``--trace-ml-api torch``.

``--coverage-n`` is the number of sampled operators (default 20), split
evenly between CUDA ``torch.ops`` and structural operators. Compares
``torch.profiler`` to the analyze call forest. Use ``--coverage-n 100``
for a nightly run. Requires a GPU.
"""

import json
import os
import random
import sys
from collections import Counter, defaultdict
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

try:
    import torch_trace_coverage_utils as coverage_utils
except ImportError:
    coverage_utils = None

COVERAGE_TEST_CONFIG: Dict[str, Any] = {"cleanup": True}

os.environ["ROCPROF"] = "rocprofiler-sdk"


@pytest.fixture
def torch_trace_coverage_sampling(request):
    seed = request.config.getoption("--coverage-seed")
    n = request.config.getoption("--coverage-n")
    if n < 0:
        pytest.fail("--coverage-n must be non-negative")
    return seed, n


def test_random_operator_kernel_coverage(
    request,
    tmp_path,
    binary_handler_profile_rocprof_compute,
    torch_trace_coverage_sampling,
):
    require_torch(gpu=True)
    seed, sample_budget = torch_trace_coverage_sampling
    rng = random.Random(seed)

    match_verbose = os.getenv("ROCPROF_OPERATOR_MATCH_VERBOSE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    aten_ops, structural_ops, excluded_aten_ops = coverage_utils.discover_operators()
    structural_sample_count = min(sample_budget // 2, len(structural_ops))
    aten_sample_count = min(sample_budget - structural_sample_count, len(aten_ops))
    sampled_aten = rng.sample(aten_ops, aten_sample_count)
    sampled = sampled_aten + rng.sample(structural_ops, structural_sample_count)
    sampled_aten_names = {op.name for op in sampled_aten}

    coverage_utils.print_torch_trace_coverage_session_header(
        seed,
        sample_budget,
        len(sampled),
        len(aten_ops),
        len(structural_ops),
        len(excluded_aten_ops),
    )

    workload_dir = common.get_output_dir(
        param_id=common.unique_output_param_id("random_op_coverage"),
        clean_existing=True,
    )
    Path(workload_dir).mkdir(parents=True, exist_ok=True)

    ground_truth_path = str(tmp_path / "ground_truth.json")
    workload_script_path = str(tmp_path / "coverage_workload.py")
    ground_truth_runner_script_path = str(tmp_path / "coverage_ground_truth_runner.py")

    try:
        coverage_utils.write_coverage_workload_artifacts(
            sampled,
            workload_script_path,
            ground_truth_runner_script_path,
        )

        coverage_utils.run_ground_truth_torch_profiler_subprocess(
            ground_truth_runner_script_path,
            workload_script_path,
            ground_truth_path,
            coverage_seed=seed,
            coverage_sample_budget=sample_budget,
        )
        with open(ground_truth_path, encoding="utf-8") as f:
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
            ["--experimental", "--trace-ml-api", "torch", "--iteration-multiplexing"],
            check_success=True,
            app_name="coverage_workload",
        )

        forest = coverage_utils.call_trees_from_workload(workload_dir)

        failure_detail: List[Tuple[str, str]] = []
        skip_categories: "Counter[str]" = Counter()
        skip_op_names: Dict[str, List[str]] = defaultdict(list)
        passed = skipped = aten_passed = 0
        for op in sampled:
            outcome = coverage_utils.compare_single_op(
                op,
                ground_truth,
                forest,
                match_verbose=match_verbose,
            )
            for line in outcome.log_lines:
                print(line)
            if outcome.status == "pass":
                passed += 1
                if op.name in sampled_aten_names:
                    aten_passed += 1
            elif outcome.status == "fail":
                failure_detail.append((op.name, outcome.reason))
            else:
                skipped += 1
                category = coverage_utils.categorize_skip_reason(outcome.reason)
                skip_categories[category] += 1
                skip_op_names[category].append(op.name)

        print(
            f"\n  Summary: {len(sampled)} ops - "
            f"{passed} PASS, {len(failure_detail)} FAIL, {skipped} SKIP"
        )
        breakdown_lines = coverage_utils.format_skip_breakdown_lines(
            dict(skip_categories),
            skip_op_names=dict(skip_op_names),
        )
        for line in breakdown_lines:
            print(line)
        print()

        arg_gap_ops = skip_op_names.get("argument_builder_gap") or []
        if arg_gap_ops:
            for line in coverage_utils.format_missing_arg_builder_report(arg_gap_ops):
                print(line)
            print()

        if failure_detail:
            for line in coverage_utils.multiline_coverage_failure_warning(
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
        assert aten_passed > 0, (
            f"no ATen operators PASSed forest coverage "
            f"(seed={seed}, sampled={len(sampled)}, PASS={passed}, "
            f"FAIL={len(failure_detail)}, SKIP={skipped})"
        )
    finally:
        common.clean_output_dir(
            COVERAGE_TEST_CONFIG["cleanup"],
            workload_dir,
        )
