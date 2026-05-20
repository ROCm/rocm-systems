# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""ROCTX marker coverage test for inject_roctx.py.

Samples ATen + structural ops, builds a workload, runs torch.profiler
as ground truth, runs rocprof-compute --torch-trace, and compares
markers + kernel correlations per op. Sampling is controlled by
--coverage-seed / --coverage-n; the strict C++ tier check is on by
default and can be relaxed with --no-require-cpp-tier.

CI: runs in every test category (quick, standard, comprehensive, full)
    via test_categories.yaml; needs a GPU; ctest timeout 1800 s.
Coverage: included in the coverage XML.
Why: the only end-to-end check that the --torch-trace marker stream
    matches torch.profiler ground truth at the per-op level.
"""

import json
import random
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import common
import pytest

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
    mismatches are reported (stdout + UserWarning) and fail the test.
    SKIPs (builder gaps, ground-truth errors, etc.) do not fail.
    """
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

    from utils import inject_roctx_loader

    seed, sample_budget = torch_trace_coverage_sampling
    rng = random.Random(seed)

    # Strict C++ tier validation is ON BY DEFAULT; --no-require-cpp-tier
    # relaxes all three axes. --require-cpp-tier is for explicitness.
    explicit_require = bool(request.config.getoption("--require-cpp-tier"))
    explicit_no_require = bool(request.config.getoption("--no-require-cpp-tier"))
    if explicit_require and explicit_no_require:
        pytest.fail("Cannot pass both --require-cpp-tier and --no-require-cpp-tier.")
    require_cpp_tier = not explicit_no_require
    match_verbose = bool(
        request.config.getoption("--torch-trace-match-verbose"),
    )

    # Pre-flight: probe the loader in this Python env. The workload
    # subprocess shares the same JIT cache + source fingerprint, so a
    # parent-side miss is a near-certain predictor of subprocess
    # degradation; failing fast here avoids wasted workload time and
    # gives the cleanest diagnostic trail. The post-workload sentinel
    # scan below covers the second axis (did the subprocess actually
    # emit C++ markers?). Both axes are gated by require_cpp_tier.
    inject_roctx_loader.load()
    loader_tier = inject_roctx_loader.loaded_tier()
    _, loader_diagnostic_trail = inject_roctx_loader.consume_diagnostics()
    loader_cpp_tier_active = loader_tier in inject_roctx_loader.C_TIER_NAMES

    print(
        f"\n  Loader (this process): tier="
        f"{loader_tier if loader_tier else 'NONE (Python fallback)'}"
    )
    if loader_diagnostic_trail:
        trail_str = inject_roctx_loader.format_load_diagnostic_trail(
            loader_diagnostic_trail,
        )
        if trail_str:
            print(f"  Loader trail:\n{trail_str}")
    print()

    if not loader_cpp_tier_active:
        warnings.warn(
            "torch_trace coverage: loader returned the Python fallback "
            f"in the test runner's env (loaded_tier={loader_tier!r}); "
            "the subprocess will almost certainly fall through too. See "
            "the preceding [torch trace loader] trail for the cause.",
            UserWarning,
            stacklevel=1,
        )

    if require_cpp_tier and not loader_cpp_tier_active:
        trail_str = inject_roctx_loader.format_load_diagnostic_trail(
            loader_diagnostic_trail,
        )
        pytest.fail(
            "strict C++ tier (loader axis): pre-flight loader probe "
            f"returned loaded_tier={loader_tier!r} (Python fallback). "
            "Fix the .so build or pass --no-require-cpp-tier.\n"
            f"Loader trail:\n{trail_str or '  (no diagnostic lines captured)'}"
        )

    # discover_operators drops hardware-incompatible CUDA-only ATen ops;
    # the excluded count is reported in the session header.
    aten_ops, structural_ops, excluded_aten_ops = discover_operators()

    # sample_budget caps the ATen sample only; structural entries are
    # always included.
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

        # Sentinel-based C++ tier signature check on the raw marker
        # stream. compare_single_op() PASSes on either tier, so this
        # is the only proof the C++ tier actually ran.
        cpp_tier_active, cpp_tier_sentinel_counts = detect_cpp_tier_signature(
            workload_dir,
        )
        cpp_tier_backward_active = any(
            cpp_tier_sentinel_counts.get(name, 0) > 0
            for name in C_TIER_BACKWARD_SENTINELS
        )

        # Parent loaded the .so but the subprocess emitted no C++
        # sentinels -- usually a different sys.executable, install()
        # raised inside inject_roctx.py, or the workload bypassed it.
        loader_subprocess_mismatch = loader_cpp_tier_active and not cpp_tier_active

        # Per-operator comparison
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

        # Family Pareto for the dominant SKIP bucket: one row per
        # structural family (count + examples + recommended builder).
        arg_gap_ops = skip_op_names.get("argument_builder_gap") or []
        if arg_gap_ops:
            for line in format_missing_arg_builder_report(arg_gap_ops):
                print(line)
            print()

        # Reported separately because PASS doesn't require the C++ tier.
        for line in format_cpp_tier_signature_report(
            cpp_tier_active,
            cpp_tier_sentinel_counts,
        ):
            print(line)
        print()

        # Surfaces degradation even when strict mode is relaxed.
        if not cpp_tier_active:
            warnings.warn(
                "torch_trace coverage ran without the C++ tier in the "
                "workload subprocess: no aten:0 / aten.nested:0 / "
                "autograd.engine:0 / autograd.bwd:0 sentinels. PASS "
                "count above only proves markers + kernel correlation. "
                "See [torch trace loader]/[torch trace] WARNINGs for "
                "the cause.",
                UserWarning,
                stacklevel=1,
            )

        if loader_subprocess_mismatch:
            warnings.warn(
                f"torch_trace coverage: parent loaded the C++ tier "
                f"(loaded_tier={loader_tier!r}) but the subprocess "
                "emitted no sentinels. Check subprocess sys.executable, "
                "inject_roctx.py install() failures, and that the "
                "workload runs under rocprofv3.",
                UserWarning,
                stacklevel=1,
            )

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
        # Strict C++ tier: three axes (loader, forward sentinels,
        # backward sentinels). The loader axis is a defensive backstop
        # for the pre-flight fail-fast above.
        if require_cpp_tier:
            assert loader_cpp_tier_active, (
                "strict C++ tier (loader axis): "
                f"loaded_tier={loader_tier!r}. The pre-flight probe "
                "should have failed first. Pass --no-require-cpp-tier."
            )
            assert cpp_tier_active, (
                "strict C++ tier (subprocess forward axis): no "
                "aten:0 / aten.nested:0 sentinels in the marker stream. "
                f"Parent loader reported loaded_tier={loader_tier!r}, "
                "so the .so is buildable -- the subprocess just didn't "
                "use it. Pass --no-require-cpp-tier to relax."
            )
            assert cpp_tier_backward_active, (
                "strict C++ tier (subprocess backward axis): forward "
                "sentinels present but no autograd.engine:0 / "
                f"autograd.bwd:0. Sentinel counts: "
                f"{cpp_tier_sentinel_counts}. "
                "Pass --no-require-cpp-tier to relax."
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
