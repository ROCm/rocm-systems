# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""End-to-end test for the C++ RecordFunction tier.

JIT-compiles roctx_recordfn.so, runs a small fwd+bwd workload, and asserts
the dump_stats() counters: push/pop balance, zero callback errors, and at
least one seqNr forward-to-backward correlation. Wire strings themselves
are validated by test_torch_trace_coverage.py.

CI: registered with ctest; not in test_categories.yaml so it runs only
    on the full ctest invocation. Needs a GPU; ctest timeout 600 s.
Coverage: not in the coverage XML (GPU-only, kept out of the coverage
    aggregation set in CMakeLists.txt).
Why: exercises the real JIT build, install/uninstall, USER_SCOPE
    propagation to the autograd worker, and the capture hook end-to-end
    -- the only place these run on a real GPU outside the coverage test.
"""

import importlib.util

import common  # noqa: F401  -- adds src/ to sys.path
import pytest

# Guard torch / loader imports so this module still loads on CPU-only
# hosts; each test below skips via the autouse _require_gpu fixture
# when torch or CUDA is unavailable. A module-level
# pytest.skip(..., allow_module_level=True) or top-level
# pytest.importorskip("torch") would collect zero items and make
# pytest exit with code 5 ("no tests collected"), which CTest reads
# as a test failure.
if importlib.util.find_spec("torch") is not None:
    import torch  # noqa: E402

    from utils import inject_roctx_loader  # noqa: E402


@pytest.fixture(autouse=True)
def _require_gpu(require_torch):
    require_torch(gpu=True)


@pytest.fixture(scope="module")
def loaded_module():
    # Duplicates the GPU check because pytest does not guarantee that
    # a function-scoped fixture resolves before a module-scoped one.
    # Without this, a CPU-only host where loaded_module happens to
    # resolve first would attempt to call into the loader (and
    # reference inject_roctx_loader, which is not bound on CPU-only
    # hosts) before _require_gpu got a chance to skip.
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")
    if not torch.cuda.is_available():
        pytest.skip("torch.cuda.is_available() is False")
    mod = inject_roctx_loader.load()
    if mod is None:
        pytest.skip("roctx_recordfn.so unavailable (no toolchain/libtorch)")
    yield mod
    try:
        mod.uninstall()
    except Exception:
        pass


def test_install_idempotent(loaded_module):
    h1 = loaded_module.install()
    h2 = loaded_module.install()
    assert h1 == h2
    assert loaded_module.is_installed()


def test_push_pop_balance_under_forward(loaded_module):
    loaded_module.install()
    before = loaded_module.dump_stats()
    x = torch.randn(32, 32, device="cuda", requires_grad=False)
    loaded_module.push_user_scope("test.scope.forward_only", "#1@test:1")
    try:
        y = x @ x
        del y
    finally:
        loaded_module.pop_user_scope()
    after = loaded_module.dump_stats()
    assert after["pushes"] > before["pushes"]
    assert after["pops"] == after["pushes"]
    assert after["callback_errors"] == 0


def test_seqnr_correlation_across_worker_thread(loaded_module):
    """fwd+bwd must save >= 1 snapshot, consume >= 1, with 0 errors."""
    loaded_module.install()
    before = loaded_module.dump_stats()

    x = torch.randn(64, 64, device="cuda", requires_grad=True)
    loaded_module.push_user_scope("test.scope.SeqNrCheck.forward", "#1@test:42")
    try:
        y = (x @ x).sum()
    finally:
        loaded_module.pop_user_scope()

    loaded_module.push_user_scope("test.scope.SeqNrCheck.backward", "#1@test:43")
    try:
        y.backward()
    finally:
        loaded_module.pop_user_scope()

    torch.cuda.synchronize()
    after = loaded_module.dump_stats()

    delta_saved = after["snapshots_saved"] - before["snapshots_saved"]
    delta_consumed = after["snapshots_consumed"] - before["snapshots_consumed"]

    assert delta_saved > 0, "FUNCTION+seqNr save path is broken"
    assert delta_consumed > 0, (
        "BACKWARD_FUNCTION consume path is broken (TLS propagation, scope "
        "enrolment, or seqNr mismatch)"
    )
    assert after["callback_errors"] == 0
    assert after["pops"] == after["pushes"]


def test_stats_pending_returns_to_zero(loaded_module):
    """After a small fwd+bwd, snapshots_pending should be near zero.
    Residue == forward ops whose backward never queued; <=4 absorbs the
    handful of autograd internals that legitimately don't backward.
    (saved - consumed > pending because same-seqNr overwrites also tick
    saved -- see the dump_stats contract in roctx_recordfn.cpp.)"""
    loaded_module.install()
    x = torch.randn(8, 8, device="cuda", requires_grad=True)
    y = (x * 2).sum()
    y.backward()
    torch.cuda.synchronize()
    stats = loaded_module.dump_stats()
    assert stats["snapshots_pending"] <= 4, (
        f"snapshots_pending={stats['snapshots_pending']} -- snapshots leaking"
    )


# Helpers for the marker-content tests below. start_capture/stop_capture
# are an opt-in test hook on the .so; they are off by default in
# production (one relaxed atomic load per emitted marker).


def _leaf_label(marker: str) -> str:
    """Return the LEAF context label (e.g. 'aten:0', 'autograd.bwd:0',
    'main.py:42') from a wire string built by build_marker_string()."""
    _, _, right = marker.partition(":#")
    last_ctx = right.split("/#")[-1]
    return last_ctx.partition("@")[2]


def test_new_leaf_labels_replace_legacy_dispatcher_label(loaded_module):
    """Every leaf must carry one of the four C++-tier labels (aten:0,
    aten.nested:0, autograd.bwd:0, autograd.engine:0) or a USER_SCOPE
    file:line. The legacy 'dispatcher:0' sentinel must not appear."""
    loaded_module.install()
    loaded_module.start_capture()
    try:
        x = torch.randn(32, 32, device="cuda", requires_grad=True)
        # Forward outside USER_SCOPE: expect aten:0 / aten.nested:0 leaves.
        y = (x @ x).sum()
        # Forward inside USER_SCOPE: still uses aten.nested:0 once
        # the user scope has pushed a frame.
        loaded_module.push_user_scope("test.nested", "#1@test:1")
        try:
            z = (x * 2).sum()
        finally:
            loaded_module.pop_user_scope()
        # Backward exercises autograd.bwd:0 and autograd.engine:0.
        (y + z).backward()
        torch.cuda.synchronize()
        captured = loaded_module.stop_capture()
    except Exception:
        loaded_module.stop_capture()
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


def test_user_scope_propagates_to_autograd_worker(loaded_module):
    """USER_SCOPE pushed on the main thread must prefix BACKWARD records
    emitted on the autograd worker. This is the c10::ThreadLocalDebugInfo
    propagation channel; without it the worker would see an empty chain
    and the backward subtree would be ungrouped from its user scope."""
    loaded_module.install()
    loaded_module.start_capture()
    try:
        loaded_module.push_user_scope("test.user_scope.backward", "#1@test:42")
        try:
            x = torch.randn(64, 64, device="cuda", requires_grad=True)
            (x @ x).sum().backward()
        finally:
            loaded_module.pop_user_scope()
        torch.cuda.synchronize()
        captured = loaded_module.stop_capture()
    except Exception:
        loaded_module.stop_capture()
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

    stats = loaded_module.dump_stats()
    assert stats["user_scope_inherits"] > 0, (
        "user_scope_inherits counter never ticked -- apply_userscope_overlay "
        "is not firing on the worker thread"
    )


def test_common_prefix_dedup_when_scope_wraps_fwd_and_bwd(loaded_module):
    """A single USER_SCOPE that wraps BOTH forward and backward must not
    appear twice in the backward marker (snapshot already has it; the
    TLS overlay must skip the common prefix)."""
    loaded_module.install()
    loaded_module.start_capture()
    try:
        loaded_module.push_user_scope("test.outer_step", "#1@test:7")
        try:
            x = torch.randn(48, 48, device="cuda", requires_grad=True)
            (x @ x).sum().backward()
        finally:
            loaded_module.pop_user_scope()
        torch.cuda.synchronize()
        captured = loaded_module.stop_capture()
    except Exception:
        loaded_module.stop_capture()
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


# Python-tier fallback test lives in test_inject_roctx_loader.py (no GPU).
