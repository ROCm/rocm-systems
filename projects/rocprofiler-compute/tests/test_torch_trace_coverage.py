# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""ROCTX marker coverage test for inject_roctx.py.

Samples ATen + structural ops, runs torch.profiler as ground truth,
runs rocprof-compute --torch-trace, compares per-op. Sampling via
--coverage-seed / --coverage-n; strict C++ tier checks with
skip-on-absence for local envs. Needs GPU.
"""

import json
import os
import random
import sys
import threading
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import common
import pytest
from conftest import require_torch

# Best-effort fallbacks so collection succeeds on CPU-only hosts.
try:
    import torch  # noqa: E402
except Exception:
    torch = None

try:
    from utils import inject_roctx_loader  # noqa: E402
except Exception:
    inject_roctx_loader = None

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
    request,
    binary_handler_profile_rocprof_compute,
    torch_trace_coverage_sampling,
):
    """Verify --torch-trace ROCTX output matches torch.profiler per op.

    Mismatches fail; per-op SKIPs (builder gaps etc.) don't; but at least
    one op must PASS (all-SKIP is a regression).
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

    from utils import inject_roctx_loader

    seed, sample_budget = torch_trace_coverage_sampling
    rng = random.Random(seed)

    # Enable with ROCPROF_OPERATOR_MATCH_VERBOSE=1 (or true/yes/on).
    match_verbose = os.getenv("ROCPROF_OPERATOR_MATCH_VERBOSE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    # Pre-flight loader probe; subprocess shares JIT cache + fingerprint.
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
            f"loader returned Python fallback (loaded_tier={loader_tier!r}); "
            "subprocess will likely fall through too.",
            UserWarning,
            stacklevel=1,
        )

    if not loader_cpp_tier_active:
        trail_str = inject_roctx_loader.format_load_diagnostic_trail(
            loader_diagnostic_trail,
        )
        pytest.skip(
            "Skipping torch_trace coverage: C++ tier unavailable in parent "
            f"loader probe (loaded_tier={loader_tier!r}).\n"
            f"Loader trail:\n{trail_str or '  (no diagnostic lines captured)'}"
        )

    aten_ops, structural_ops, excluded_aten_ops = discover_operators()

    # sample_budget caps ATen only; structural entries always included.
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

        # Sentinel check: the only proof C++ tier ran (PASS works on either).
        cpp_tier_active, cpp_tier_sentinel_counts = detect_cpp_tier_signature(
            workload_dir,
        )
        cpp_tier_backward_active = any(
            cpp_tier_sentinel_counts.get(name, 0) > 0
            for name in C_TIER_BACKWARD_SENTINELS
        )

        # Parent loaded .so but subprocess didn't: different sys.executable,
        # install() raised, or workload bypassed inject_roctx.py.
        loader_subprocess_mismatch = loader_cpp_tier_active and not cpp_tier_active

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
        # Strict C++ tier: subprocess marker stream must include both
        # forward and backward sentinel classes when the C++ tier is active.
        assert cpp_tier_active, (
            "strict C++ tier (subprocess forward axis): no "
            "aten:0 / aten.nested:0 sentinels in the marker stream. "
            f"Parent loader reported loaded_tier={loader_tier!r}, "
            "so the .so is buildable -- the subprocess just didn't "
            "use it."
        )
        assert cpp_tier_backward_active, (
            "strict C++ tier (subprocess backward axis): forward "
            "sentinels present but no autograd.engine:0 / "
            f"autograd.bwd:0. Sentinel counts: "
            f"{cpp_tier_sentinel_counts}."
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


# -----------------------------------
# C++ tier API contract tests
# -----------------------------------
# Direct .so contract tests for counters/capture/TLS behavior that the
# end-to-end --torch-trace integration test cannot observe.
# Wire-string shape is covered by test_random_operator_kernel_coverage above.


@pytest.fixture(scope="module")
def roctx_recordfn_so():
    """Load roctx_recordfn.so once for the cpp_tier_* tests below."""
    # Repeat skip checks here because module-scoped fixtures can resolve before
    # per-test require_torch(gpu=True) runs.
    if torch is None:
        pytest.skip("PyTorch is not installed")
    if not torch.cuda.is_available():
        pytest.skip("torch.cuda.is_available() is False")
    if inject_roctx_loader is None:
        pytest.skip("inject_roctx_loader import failed")
    mod = inject_roctx_loader.load()
    if mod is None:
        pytest.skip("roctx_recordfn.so unavailable (no toolchain/libtorch)")
    yield mod
    try:
        mod.uninstall()
    except Exception:
        pass


def _leaf_label(marker: str) -> str:
    """Return the LEAF context label (e.g. 'aten:0', 'autograd.bwd:0',
    'main.py:42') from a wire string built by build_marker_string()."""
    _, _, right = marker.partition(":#")
    last_ctx = right.split("/#")[-1]
    return last_ctx.partition("@")[2]


@pytest.mark.torch_trace
def test_cpp_tier_install_idempotent(roctx_recordfn_so):
    """install() is idempotent: two calls return the same handle and
    is_installed() remains true."""
    require_torch(gpu=True)
    mod = roctx_recordfn_so
    h1 = mod.install()
    h2 = mod.install()
    assert h1 == h2
    assert mod.is_installed()


@pytest.mark.torch_trace
def test_cpp_tier_push_pop_balance_under_forward(roctx_recordfn_so):
    """Pushes == pops + zero callback errors after a forward-only workload."""
    require_torch(gpu=True)
    mod = roctx_recordfn_so
    mod.install()
    before = mod.dump_stats()
    x = torch.randn(32, 32, device="cuda", requires_grad=False)
    mod.push_user_scope("test.scope.forward_only", "#1@test:1")
    try:
        y = x @ x
        del y
    finally:
        mod.pop_user_scope()
    after = mod.dump_stats()
    assert after["pushes"] > before["pushes"]
    assert after["pops"] == after["pushes"]
    assert after["callback_errors"] == 0


@pytest.mark.torch_trace
def test_cpp_tier_seqnr_correlation_across_worker_thread(roctx_recordfn_so):
    """fwd+bwd must save >= 1 snapshot, consume >= 1, with 0 errors."""
    require_torch(gpu=True)
    mod = roctx_recordfn_so
    mod.install()
    before = mod.dump_stats()

    x = torch.randn(64, 64, device="cuda", requires_grad=True)
    mod.push_user_scope("test.scope.SeqNrCheck.forward", "#1@test:42")
    try:
        y = (x @ x).sum()
    finally:
        mod.pop_user_scope()

    mod.push_user_scope("test.scope.SeqNrCheck.backward", "#1@test:43")
    try:
        y.backward()
    finally:
        mod.pop_user_scope()

    torch.cuda.synchronize()
    after = mod.dump_stats()

    delta_saved = after["snapshots_saved"] - before["snapshots_saved"]
    delta_consumed = after["snapshots_consumed"] - before["snapshots_consumed"]

    assert delta_saved > 0, "FUNCTION+seqNr save path is broken"
    assert delta_consumed > 0, (
        "BACKWARD_FUNCTION consume path is broken (TLS propagation, scope "
        "enrolment, or seqNr mismatch)"
    )
    assert after["callback_errors"] == 0
    assert after["pops"] == after["pushes"]


@pytest.mark.torch_trace
def test_cpp_tier_stats_pending_per_call_bounded(roctx_recordfn_so):
    """A small fwd+bwd should not leak more than a handful of snapshots."""
    # Use delta (not absolute) because module-scoped fixture tests share state.
    # Allow a small margin for autograd internals that do not backward.
    require_torch(gpu=True)
    mod = roctx_recordfn_so
    mod.install()
    before = mod.dump_stats()
    x = torch.randn(8, 8, device="cuda", requires_grad=True)
    y = (x * 2).sum()
    y.backward()
    torch.cuda.synchronize()
    after = mod.dump_stats()
    delta_pending = after["snapshots_pending"] - before["snapshots_pending"]
    assert delta_pending <= 4, (
        f"delta snapshots_pending={delta_pending} for a single fwd+bwd "
        f"-- snapshots leaking"
    )


@pytest.mark.torch_trace
def test_cpp_tier_new_leaf_labels_replace_legacy_dispatcher_label(roctx_recordfn_so):
    """Every leaf must carry one of the four C++-tier labels (aten:0,
    aten.nested:0, autograd.bwd:0, autograd.engine:0) or a USER_SCOPE
    file:line. The legacy 'dispatcher:0' sentinel must not appear."""
    require_torch(gpu=True)
    mod = roctx_recordfn_so
    mod.install()
    mod.start_capture()
    try:
        x = torch.randn(32, 32, device="cuda", requires_grad=True)
        # Forward outside USER_SCOPE: expect aten:0 / aten.nested:0 leaves.
        y = (x @ x).sum()
        # Forward inside USER_SCOPE: still uses aten.nested:0 once
        # the user scope has pushed a frame.
        mod.push_user_scope("test.nested", "#1@test:1")
        try:
            z = (x * 2).sum()
        finally:
            mod.pop_user_scope()
        # Backward exercises autograd.bwd:0 and autograd.engine:0.
        (y + z).backward()
        torch.cuda.synchronize()
        captured = mod.stop_capture()
    except Exception:
        mod.stop_capture()
        raise

    assert captured, "capture buffer is empty"

    expected_cpp_labels = {
        "aten:0",
        "aten.nested:0",
        "autograd.bwd:0",
        "autograd.engine:0",
    }
    seen = {_leaf_label(m) for m in captured}
    cpp_seen = seen & expected_cpp_labels

    assert "dispatcher:0" not in {label.split("@")[-1] for label in seen}, (
        f"legacy 'dispatcher:0' leaked into a marker (seen leaves: {seen})"
    )
    assert "aten:0" in cpp_seen, (
        f"no top-level aten:0 leaf observed -- label classifier broken "
        f"(seen leaves: {seen})"
    )
    assert "aten.nested:0" in cpp_seen, (
        f"no aten.nested:0 leaf observed -- nested ATen redispatch label "
        f"broken (seen leaves: {seen})"
    )
    assert "autograd.bwd:0" in cpp_seen, (
        f"no autograd.bwd:0 leaf observed -- BACKWARD_FUNCTION + seq>=0 "
        f"label broken (seen leaves: {seen})"
    )


@pytest.mark.torch_trace
def test_cpp_tier_user_scope_propagates_to_autograd_worker(roctx_recordfn_so):
    """USER_SCOPE pushed on the main thread must prefix BACKWARD records
    emitted on the autograd worker. This is the c10::ThreadLocalDebugInfo
    propagation channel; without it the worker would see an empty chain
    and the backward subtree would be ungrouped from its user scope."""
    require_torch(gpu=True)
    mod = roctx_recordfn_so
    mod.install()
    before = mod.dump_stats()
    mod.start_capture()
    try:
        mod.push_user_scope("test.user_scope.backward", "#1@test:42")
        try:
            x = torch.randn(64, 64, device="cuda", requires_grad=True)
            (x @ x).sum().backward()
        finally:
            mod.pop_user_scope()
        torch.cuda.synchronize()
        captured = mod.stop_capture()
    except Exception:
        mod.stop_capture()
        raise

    backward_markers = [
        m for m in captured if _leaf_label(m) in ("autograd.bwd:0", "autograd.engine:0")
    ]
    assert backward_markers, "no BACKWARD records captured"
    inherited = [
        m for m in backward_markers if m.startswith("test.user_scope.backward/")
    ]
    assert inherited, (
        "USER_SCOPE did not propagate to autograd worker via TLS DebugInfo. "
        f"Backward markers do not start with 'test.user_scope.backward/'. "
        f"Sample of 3: {backward_markers[:3]}"
    )

    after = mod.dump_stats()
    delta_inherits = after["user_scope_inherits"] - before["user_scope_inherits"]
    assert delta_inherits > 0, (
        "user_scope_inherits counter did not tick during this test -- "
        "apply_userscope_overlay is not firing on the worker thread"
    )


@pytest.mark.torch_trace
def test_cpp_tier_common_prefix_dedup_when_scope_wraps_fwd_and_bwd(roctx_recordfn_so):
    """A single USER_SCOPE that wraps BOTH forward and backward must not
    appear twice in the backward marker (snapshot already has it; the
    TLS overlay must skip the common prefix)."""
    require_torch(gpu=True)
    mod = roctx_recordfn_so
    mod.install()
    mod.start_capture()
    try:
        mod.push_user_scope("test.outer_step", "#1@test:7")
        try:
            x = torch.randn(48, 48, device="cuda", requires_grad=True)
            (x @ x).sum().backward()
        finally:
            mod.pop_user_scope()
        torch.cuda.synchronize()
        captured = mod.stop_capture()
    except Exception:
        mod.stop_capture()
        raise

    backward = [m for m in captured if _leaf_label(m) == "autograd.bwd:0"]
    assert backward, "no autograd.bwd:0 backward markers captured"

    for m in backward:
        marker_path = m.partition(":#")[0]
        segments = marker_path.split("/")
        # The wrapping scope should appear exactly once at the head.
        head_count = sum(1 for s in segments if s == "test.outer_step")
        assert head_count == 1, (
            f"USER_SCOPE 'test.outer_step' appears {head_count} times in a "
            f"backward marker -- common-prefix dedup in apply_userscope_overlay "
            f"is broken. Marker: {m!r}"
        )


def _tiny_train_step():
    """One fwd+bwd cycle used by the correlation stress tests."""
    x = torch.randn(128, 128, device="cuda", requires_grad=True)
    y = ((x @ x) + x).sum()
    y.backward()


@pytest.mark.torch_trace
def test_cpp_tier_correlation_under_many_steps(roctx_recordfn_so):
    """Most saved snapshots should be consumed (>=50%) and the soft cap
    must not fire on a workload this small."""
    require_torch(gpu=True)
    mod = roctx_recordfn_so
    mod.install()
    n_steps = 16
    before = mod.dump_stats()
    for _ in range(n_steps):
        _tiny_train_step()
    torch.cuda.synchronize()
    after = mod.dump_stats()

    delta_saved = after["snapshots_saved"] - before["snapshots_saved"]
    delta_consumed = after["snapshots_consumed"] - before["snapshots_consumed"]
    delta_dropped = after["snapshots_dropped"] - before["snapshots_dropped"]
    delta_errors = after["callback_errors"] - before["callback_errors"]

    assert delta_saved > 0
    assert delta_consumed >= int(0.5 * delta_saved), (
        f"only {delta_consumed}/{delta_saved} correlated -- worker callback "
        f"is not re-seeding the marker stack"
    )
    assert delta_errors == 0
    assert delta_dropped == 0, f"soft cap fired unexpectedly: {delta_dropped}"


@pytest.mark.torch_trace
def test_cpp_tier_detached_forward_does_not_explode(roctx_recordfn_so):
    """Detached forwards leak snapshots by design; the soft cap
    (10k/shard * 64 shards = 640k slots) must hold and the callback
    must not crash."""
    require_torch(gpu=True)
    mod = roctx_recordfn_so
    mod.install()
    before = mod.dump_stats()
    for _ in range(50):
        x = torch.randn(32, 32, device="cuda", requires_grad=True)
        y = (x @ x).sum().detach()
        del x, y
    torch.cuda.synchronize()
    after = mod.dump_stats()
    delta_saved = after["snapshots_saved"] - before["snapshots_saved"]
    delta_errors = after["callback_errors"] - before["callback_errors"]

    assert delta_saved > 0
    assert delta_errors == 0
    assert after["snapshots_pending"] < 640_000


@pytest.mark.torch_trace
def test_cpp_tier_concurrent_threads_no_callback_errors(roctx_recordfn_so):
    """Concurrent worker threads must emit C++-tier markers and keep
    callback/push-pop invariants intact."""
    require_torch(gpu=True)
    mod = roctx_recordfn_so
    mod.install()
    before = mod.dump_stats()
    mod.start_capture()

    n_workers = 4

    def worker(worker_id: int) -> None:
        scope = f"test.concurrent.worker{worker_id}"
        mod.push_user_scope(scope, f"#1@test_thread:{worker_id}")
        try:
            for _ in range(4):
                x = torch.randn(64, 64, device="cuda", requires_grad=True)
                (x @ x).sum().backward()
            torch.cuda.synchronize()
        finally:
            mod.pop_user_scope()

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_workers)]
    try:
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)
            assert not t.is_alive(), "worker thread deadlocked"
        captured = mod.stop_capture()
    except Exception:
        mod.stop_capture()
        raise

    cpp_labels = {"aten:0", "aten.nested:0", "autograd.bwd:0", "autograd.engine:0"}
    for worker_id in range(n_workers):
        scope = f"test.concurrent.worker{worker_id}/"
        per_worker = [
            m for m in captured if m.startswith(scope) and _leaf_label(m) in cpp_labels
        ]
        assert per_worker, (
            f"worker {worker_id} emitted no C++-tier markers (scope={scope!r}); "
            "global callback coverage is missing on at least one thread"
        )

    after = mod.dump_stats()
    assert (after["callback_errors"] - before["callback_errors"]) == 0
    assert after["pops"] == after["pushes"]


@pytest.mark.torch_trace
def test_cpp_tier_user_spawned_python_thread_emits_markers(roctx_recordfn_so):
    """A plain user-spawned Python thread running ATen forward ops should
    emit at least one captured C++-tier marker."""
    require_torch(gpu=True)
    mod = roctx_recordfn_so
    mod.install()
    mod.start_capture()

    def worker() -> None:
        mod.push_user_scope("test.user_thread", "#1@test_thread:plain")
        try:
            x = torch.randn(64, 64, device="cuda", requires_grad=False)
            y = x @ x
            del y
            torch.cuda.synchronize()
        finally:
            mod.pop_user_scope()

    t = threading.Thread(target=worker)
    try:
        t.start()
        t.join(timeout=60)
        assert not t.is_alive(), "user thread deadlocked"
        captured = mod.stop_capture()
    except Exception:
        mod.stop_capture()
        raise

    cpp_labels = {"aten:0", "aten.nested:0", "autograd.bwd:0", "autograd.engine:0"}
    cpp_markers = [
        m
        for m in captured
        if m.startswith("test.user_thread/") and _leaf_label(m) in cpp_labels
    ]
    assert cpp_markers, "no C++-tier markers captured from user-spawned Python thread"


@pytest.mark.torch_trace
def test_cpp_tier_function_apply_no_double_wrap_on_grandchild(
    monkeypatch,
):
    """A grandchild Function subclass should not get a second apply wrapper."""
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

    # With the MRO-aware guard, only Foo carries the wrapped staticmethod.
    assert getattr(Foo.__dict__.get("apply"), "_roctx_wrapped", False)
    assert "apply" not in Bar.__dict__

    x = torch.tensor(1.0, requires_grad=True)
    y = Bar.apply(x)
    y.backward()

    assert push_counter["count"] == 1, (
        "Bar.apply triggered more than one wrapper push; "
        "Function.apply was double-wrapped across the inheritance chain"
    )
