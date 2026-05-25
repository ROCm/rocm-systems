# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Stress tests for roctx_recordfn's seqNr forward-to-backward correlation.

Covers three conditions known to break similar tracers:
  * many matched fwd/bwd pairs across training steps;
  * detached graphs (forward snapshots whose backward never fires);
  * concurrent Python threads exercising the shard mutexes.

Counters from dump_stats() are the source of truth; wire strings are
covered by test_torch_trace_coverage.py.

CI: registered with ctest; not in test_categories.yaml so it runs only
    on the full ctest invocation. Needs a GPU; ctest timeout 600 s.
Coverage: not in the coverage XML (GPU-only, kept out of the coverage
    aggregation set in CMakeLists.txt).
Why: the coverage test asserts marker shape but not the correlation
    map's behaviour under load; these stressors catch leaks, drops,
    and shard-mutex regressions that wire-string checks can't see.
"""

import importlib.util
import threading

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
def mod():
    # Duplicates the GPU check because pytest does not guarantee that
    # a function-scoped fixture resolves before a module-scoped one.
    # See test_torch_trace_worker_thread.py for the full rationale.
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")
    if not torch.cuda.is_available():
        pytest.skip("torch.cuda.is_available() is False")
    m = inject_roctx_loader.load()
    if m is None:
        pytest.skip("roctx_recordfn.so unavailable")
    m.install()
    yield m
    try:
        m.uninstall()
    except Exception:
        pass


def _tiny_train_step():
    x = torch.randn(128, 128, device="cuda", requires_grad=True)
    y = ((x @ x) + x).sum()
    y.backward()


def test_correlation_under_many_steps(mod):
    """Most saved snapshots should be consumed (>=50%) and the soft cap
    must not fire on a workload this small."""
    N_STEPS = 16
    before = mod.dump_stats()
    for _ in range(N_STEPS):
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


def test_detached_forward_does_not_explode(mod):
    """Detached forwards leak snapshots by design; the soft cap
    (10k/shard * 64 shards = 640k slots) must hold and the callback
    must not crash."""
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


def test_concurrent_threads_no_callback_errors(mod):
    """Concurrent fwd+bwd on N Python threads: no callback errors, no
    deadlock, global push/pop balance preserved. Per-thread correlation
    may be lower because backward saves can race against same-seqNr
    consumes -- that is acceptable."""
    before = mod.dump_stats()

    def worker():
        for _ in range(4):
            x = torch.randn(64, 64, device="cuda", requires_grad=True)
            (x @ x).sum().backward()
        torch.cuda.synchronize()

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
        assert not t.is_alive(), "worker thread deadlocked"

    after = mod.dump_stats()
    assert (after["callback_errors"] - before["callback_errors"]) == 0
    assert after["pops"] == after["pushes"]
