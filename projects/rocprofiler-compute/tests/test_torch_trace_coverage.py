# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""ROCTX marker coverage test for inject_roctx.py.

Samples ATen operators and structural patterns (nn.Module forward,
Optimizer.step, autograd/compile/jit/distributed/cuda surfaces), generates
a workload + a ground-truth runner that profiles each op with
torch.profiler, runs rocprof-compute --torch-trace on the workload, then
compares ROCTX markers and kernel correlations against the ground truth.

Sampling controlled by --coverage-seed / --coverage-n (see conftest.py).
On ground-truth subprocess failure, the generated workload and runner are
copied to the pytest cwd as failed_torch_trace_coverage_{workload,runner}.py.

Torch-dependent helpers live in torch_trace_coverage_utils and are
imported lazily inside the test body, after require_torch_gpu has run.
"""

import json
import random
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest
import test_utils

COVERAGE_TEST_CONFIG: Dict[str, Any] = {"cleanup": True}


# -- Main test --


@pytest.fixture
def torch_trace_coverage_sampling(request):
    """Return (seed, sample_budget) for test_random_operator_kernel_coverage."""
    seed = request.config.getoption("--coverage-seed")
    n = request.config.getoption("--coverage-n")
    if n < 0:
        pytest.fail("--coverage-n must be non-negative")
    return seed, n


@pytest.mark.torch_trace
def test_random_operator_kernel_coverage(
    require_torch_gpu,
    request,
    binary_handler_profile_rocprof_compute,
    torch_trace_coverage_sampling,
):
    """Verify --torch-trace ROCTX output matches profiler ground truth.

    Steps: sample ops → emit workload + runner → run runner for JSON → run
    rocprof-compute on the workload → parse CSVs → compare per op. Per-op
    mismatches are reported (stdout + UserWarning) but do not
    individually fail the test item; the test fails only if no sampled
    operator passes (a regression guard while coverage gaps remain).
    """
    from torch_trace_coverage_utils import (
        compare_single_op,
        discover_operators,
        multiline_coverage_failure_warning,
        parse_roctx_markers,
        print_torch_trace_coverage_session_header,
        run_ground_truth_torch_profiler_subprocess,
        unique_get_output_param_id,
        write_coverage_workload_artifacts,
    )

    seed, sample_budget = torch_trace_coverage_sampling
    rng = random.Random(seed)

    aten_ops, structural_ops = discover_operators()

    # sample_budget caps only the ATen sample; every structural entry is
    # always included. When the budget is smaller than len(structural_ops)
    # the resulting sample size equals len(structural_ops); see the
    # --coverage-n help text in conftest.py.
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

        # TODO: tighten to assert not failure_detail once every sampled
        # operator reliably matches a ROCTX marker and its kernels. The
        # current assertion guards only against total regression
        # (zero successes).
        if failure_detail:
            warnings.warn(
                multiline_coverage_failure_warning(
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
