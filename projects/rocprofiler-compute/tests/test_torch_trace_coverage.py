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
import threading
import warnings
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

try:
    from utils import inject_roctx_loader  # noqa: E402
except Exception:
    inject_roctx_loader = None

COVERAGE_TEST_CONFIG: Dict[str, Any] = {"cleanup": True}


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

    from utils import inject_roctx_loader

    seed, sample_budget = torch_trace_coverage_sampling
    rng = random.Random(seed)

    match_verbose = os.getenv("ROCPROF_OPERATOR_MATCH_VERBOSE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    # Probe the loader; the subprocess shares its JIT cache and fingerprint.
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
            warnings.warn(
                "torch_trace coverage ran without the C++ tier in the "
                "workload subprocess.",
                UserWarning,
                stacklevel=1,
            )

        if loader_subprocess_mismatch:
            warnings.warn(
                f"torch_trace coverage: parent loaded the C++ tier "
                f"(loaded_tier={loader_tier!r}) but the subprocess "
                "emitted no sentinels.",
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
        assert cpp_tier_active, (
            "no aten:0 / aten.nested:0 sentinels in the marker stream "
            f"(parent loaded_tier={loader_tier!r})"
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


# C++ tier API contract tests for counters, capture, and TLS behaviour
# that the end-to-end integration test cannot observe directly.


@pytest.fixture(scope="module")
def roctx_recordfn_so():
    """Load the ``roctx_recordfn`` extension once for the C++ tier tests."""
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
    """Return the leaf-context label from a wire string."""
    _, _, right = marker.partition(":#")
    last_ctx = right.split("/#")[-1]
    return last_ctx.partition("@")[2]


@pytest.mark.torch_trace
def test_cpp_tier_install_idempotent(roctx_recordfn_so):
    """``install()`` is idempotent and ``is_installed()`` remains true."""
    require_torch(gpu=True)
    mod = roctx_recordfn_so
    h1 = mod.install()
    h2 = mod.install()
    assert h1 == h2
    assert mod.is_installed()


@pytest.mark.torch_trace
def test_cpp_tier_push_pop_balance_under_forward(roctx_recordfn_so):
    """Push and pop counts balance with zero callback errors after a forward."""
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
    """A forward+backward saves and consumes at least one snapshot."""
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

    assert delta_saved > 0, "FUNCTION+seqNr save path produced no snapshots"
    assert delta_consumed > 0, "BACKWARD_FUNCTION did not consume any snapshot"
    assert after["callback_errors"] == 0
    assert after["pops"] == after["pushes"]


@pytest.mark.torch_trace
def test_cpp_tier_stats_pending_per_call_bounded(roctx_recordfn_so):
    """A short forward+backward leaks at most a few pending snapshots."""
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
        f"delta snapshots_pending={delta_pending} (snapshot leak)"
    )


@pytest.mark.torch_trace
def test_cpp_tier_new_leaf_labels_replace_legacy_dispatcher_label(roctx_recordfn_so):
    """Every leaf carries a C++-tier label or a USER_SCOPE ``file:line``."""
    require_torch(gpu=True)
    mod = roctx_recordfn_so
    mod.install()
    mod.start_capture()
    try:
        x = torch.randn(32, 32, device="cuda", requires_grad=True)
        y = (x @ x).sum()
        mod.push_user_scope("test.nested", "#1@test:1")
        try:
            z = (x * 2).sum()
        finally:
            mod.pop_user_scope()
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
    assert "aten:0" in cpp_seen, f"no aten:0 leaf observed (seen: {seen})"
    assert "aten.nested:0" in cpp_seen, f"no aten.nested:0 leaf observed (seen: {seen})"
    assert "autograd.bwd:0" in cpp_seen, (
        f"no autograd.bwd:0 leaf observed (seen: {seen})"
    )


@pytest.mark.torch_trace
def test_cpp_tier_user_scope_propagates_to_autograd_worker(roctx_recordfn_so):
    """A main-thread USER_SCOPE prefixes backward records on the autograd worker."""
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
        "USER_SCOPE did not propagate to the autograd worker; "
        f"sample backward markers: {backward_markers[:3]}"
    )

    after = mod.dump_stats()
    delta_inherits = after["user_scope_inherits"] - before["user_scope_inherits"]
    assert delta_inherits > 0, "user_scope_inherits did not increment"


@pytest.mark.torch_trace
def test_cpp_tier_common_prefix_dedup_when_scope_wraps_fwd_and_bwd(roctx_recordfn_so):
    """A USER_SCOPE wrapping both forward and backward appears once per marker."""
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
        head_count = sum(1 for s in segments if s == "test.outer_step")
        assert head_count == 1, (
            f"USER_SCOPE 'test.outer_step' appears {head_count} times "
            f"in backward marker {m!r}"
        )


def _tiny_train_step():
    """One forward+backward cycle used by the correlation stress tests."""
    x = torch.randn(128, 128, device="cuda", requires_grad=True)
    y = ((x @ x) + x).sum()
    y.backward()


@pytest.mark.torch_trace
def test_cpp_tier_correlation_under_many_steps(roctx_recordfn_so):
    """At least half of saved snapshots are consumed and the soft cap does not fire."""
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
        f"only {delta_consumed}/{delta_saved} snapshots consumed"
    )
    assert delta_errors == 0
    assert delta_dropped == 0, f"soft cap fired unexpectedly: {delta_dropped}"


@pytest.mark.torch_trace
def test_cpp_tier_detached_forward_does_not_explode(roctx_recordfn_so):
    """Detached forwards stay under the per-shard soft cap and do not crash."""
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
    """Concurrent threads emit C++-tier markers and preserve push/pop balance."""
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
            f"worker {worker_id} emitted no C++-tier markers (scope={scope!r})"
        )

    after = mod.dump_stats()
    assert (after["callback_errors"] - before["callback_errors"]) == 0
    assert after["pops"] == after["pushes"]


@pytest.mark.torch_trace
def test_cpp_tier_user_spawned_python_thread_emits_markers(roctx_recordfn_so):
    """ATen forward ops on a user-spawned thread emit C++-tier markers."""
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
