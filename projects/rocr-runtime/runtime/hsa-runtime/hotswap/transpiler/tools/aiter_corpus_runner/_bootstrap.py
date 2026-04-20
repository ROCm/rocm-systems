"""Child-process bootstrap for the AITER corpus crash sweeper.

This module is imported by ``runner.py`` *inside* the child Python
process, before the user op-test script runs.  Its job is to:

  1. Set ``GPU_ARCHS`` *before* aiter is imported so AITER's loader
     picks the per-arch ``hsa/<gfx>/`` code-object directory we want
     to exercise (typically ``gfx950`` when running on a gfx942 box,
     so the resulting ``hipModuleLoad`` calls trip the hotswap hook
     and route through the Salmon transpiler).

  2. Monkey-patch ``aiter.test_common.checkAllclose`` so it actually
     raises on real failures instead of just logging.  The upstream
     implementation only ``logger.info``s a "failed!" line and
     returns the per-element mismatch percent (see test_common.py
     lines 274-307); a wrong transpiled kernel currently exits 0
     with no signal at all.  We preserve the original logging (so
     triage reads identically in stderr) and raise ``AssertionError``
     when the percent exceeds a configurable threshold, broken down
     by the ``msg=`` label the test passed (``"asm"``, ``"asm res"``,
     ``"res check"``, ...).  An ``atexit`` hook prints a per-label
     pass/fail summary so a single-script triage doesn't require
     parsing the whole stderr stream.

  3. Throttle ``aiter.test_common.perftest`` (and ``run_perftest``)
     to a single iteration so the 101-iter benchmark sweeps that
     decorate every AITER test don't dominate wall clock.  The
     correctness path is untouched; we just stop running the kernel
     a hundred times for a perf number we discard anyway.

Environment knobs the parent sets:

  AITER_CORPUS_FORCE_GFX     ``"gfx942"`` / ``"gfx950"`` / ...
                             Sets ``GPU_ARCHS`` so AITER picks the
                             matching ``hsa/<gfx>/`` directory.
                             Required.

  AITER_CORPUS_RNG_SEED      Integer seed fed to ``torch.manual_seed``
                             + ``torch.cuda.manual_seed_all`` +
                             ``random.seed`` + ``numpy.random.seed``
                             *before* the user op-test script runs.
                             Default ``"0"``.  Making this non-empty
                             means every mode (native / legacy /
                             salmon) sees identical random-input
                             tensors for a given script + args, which
                             is the invariant the cross-mode
                             comparison relies on: any per-label
                             ``checkAllclose`` delta between native
                             and salmon under this setup is
                             necessarily a transpilation signal, not
                             random noise from unseeded RNG.  Set
                             to the empty string to leave RNG alone
                             (older behaviour; only useful when
                             debugging RNG-sensitive numerical
                             issues in the kernel itself).

  AITER_CORPUS_STRICT_TOL    Float in ``[0.0, 1.0]``; mismatches
                             whose elementwise-fail-percent is
                             *strictly greater* than this raise
                             ``AssertionError``.  Default ``"0.01"``
                             matches AITER's own "warning vs failed"
                             cutoff (test_common.py:290).

  AITER_CORPUS_PERFTEST_ITERS  Integer; ignored on the correctness
                             path because we always run the kernel
                             exactly once (the ``perftest`` patch
                             discards iteration counts entirely).
                             Kept as an env knob so the runner can
                             surface it in its report header.

    AITER_CORPUS_SCRIPT        Absolute path to the user op-test
                               script to run via ``runpy``.  Required.

    AITER_CORPUS_SCRIPT_ARGS   shlex-quoted (single string) argv
                               items to append after the script path
                               on ``sys.argv``.  Optional.  Empty /
                               unset means the script sees no extra
                               args, i.e. its argparse defaults take
                               effect.

The bootstrap exits with the user script's natural exit status (or
the propagated exception's traceback).  Any failure inside the
bootstrap itself (env-var missing, monkey-patch failures) is loud
and non-zero — we never silently fall back to the un-patched path,
because that would defeat the entire point of the run.
"""
from __future__ import annotations

import atexit
import collections
import os
import runpy
import sys
import traceback
from typing import Any, Callable


def _require_env(name: str) -> str:
    val = os.environ.get(name, "").strip()
    if not val:
        print(
            f"[aiter_corpus_runner] required env var {name} is unset",
            file=sys.stderr,
            flush=True,
        )
        sys.exit(2)
    return val


def _set_gpu_archs(gfx: str) -> None:
    """Force AITER's ``GPU_ARCHS`` to the value the runner picked.
    Must run before any ``import aiter`` so the cached ``AITER_ASM_DIR``
    and the generated ``asm_<module>_configs.hpp`` (via codegen.py's
    read of ``AITER_GPU_ARCHS``) are built from this value.

    Two shapes appear in practice, all forwarded verbatim:

      ``"gfx942"``           single-arch run on a gfx942 host where
                             no transpile mode is enabled (spoof off).
      ``"gfx942;gfx950"``    the canonical runner default on a gfx942
                             host — kept identical across native,
                             legacy, and salmon modes so the on-disk
                             JIT cache (keyed by hipcc build flags)
                             is byte-identical between runs.  The
                             native arch ensures host-side fat
                             binaries actually run on the real
                             device; the source-ISA arch ensures
                             (a) codegen.py emits config rows keyed
                             on gfx950 for the asm lookup, and
                             (b) hipcc produces gfx950 device code,
                             which survives being fed to
                             ``hipModuleLoadData`` where Salmon
                             intercepts it.

    Note: ``GPU_ARCHS`` only steers the *Python-side* AITER paths
    (chip_info.get_gfx, JIT build target selection, asm-availability
    guards in tests).  The companion ``AITER_GPU_ARCHS`` — which
    ``hsa/codegen.py`` reads directly — is pre-seeded to the same
    value below so the codegen subprocess doesn't rely on import
    order (AITER normally derives it lazily inside
    ``chip_info.get_gfx_list()``; if our child process hasn't yet
    touched that code path when ``os.system(blob_gen_cmd)`` fires,
    codegen would silently emit host-arch-only rows).  The C++
    asm-kernel loader keys on ``hipGetDeviceProperties().gcnArchName``
    instead — that's why the runner also LD_PRELOADs
    ``libaiter_arch_spoof.so`` for transpile modes; see runner.py
    and arch_spoof.cpp for the full rationale.
    """
    os.environ["GPU_ARCHS"] = gfx
    # Pre-seed AITER_GPU_ARCHS from the same value so hsa/codegen.py
    # (invoked as a subprocess via ``os.system(blob_gen_cmd)`` from
    # aiter/jit/core.py) sees the full multi-arch list regardless of
    # whether anything in the parent process has yet called
    # ``chip_info.get_gfx_list()`` (which normally derives
    # AITER_GPU_ARCHS lazily from GPU_ARCHS).  Without this pre-seed,
    # the codegen subprocess inherits whatever the caller's shell
    # happened to export — almost always nothing — and emits config
    # rows for only the host arch, which causes the
    # ``get_heuristic_kernel_*: cannot get heuristic kernel!
    # arch_id:gfx950`` abort when the spoofed asm loader asks for the
    # foreign-ISA row.  Setting both env vars up front matches AITER's
    # post-get_gfx_list invariant and is the minimum change that
    # survives import ordering.
    os.environ["AITER_GPU_ARCHS"] = gfx
    spoof = os.environ.get("AITER_CORPUS_SPOOF_ARCH", "")
    if spoof:
        print(
            f"[aiter_corpus_runner] forced GPU_ARCHS={gfx!r} "
            f"(AITER_GPU_ARCHS pre-seeded identically); "
            f"hipGetDeviceProperties().gcnArchName spoofed to {spoof!r} "
            f"by libaiter_arch_spoof.so (AITER's C++ asm loader will "
            f"read hsa/{spoof}/*.co)",
            file=sys.stderr,
            flush=True,
        )
    else:
        print(
            f"[aiter_corpus_runner] forced GPU_ARCHS={gfx!r} "
            f"(AITER_GPU_ARCHS pre-seeded identically); "
            f"AITER_CORPUS_SPOOF_ARCH unset (C++ asm loader will use "
            f"the device's real gcnArchName)",
            file=sys.stderr,
            flush=True,
        )


def _label_key(msg: Any) -> str:
    """Normalize a ``checkAllclose(msg=...)`` argument into a short
    triage label.  AITER tests pass either a perf-style banner
    (``"[perf] dim: (128,8192), ..."``) or a path-distinguishing tag
    (``"asm"``, ``"asm res"``, ``"res check"``).  We keep the first
    40 chars of whichever was passed; that's enough to keep the
    distinct path tags separate while truncating the perf banners
    so they don't blow out the final summary."""
    if msg is None:
        return "<no msg>"
    s = str(msg).strip()
    if not s:
        return "<empty msg>"
    return s[:40]


def _patch_check_allclose(strict_tol: float) -> None:
    """Wrap ``aiter.test_common.checkAllclose`` so it (a) still logs
    via the upstream implementation (preserving stderr triage), and
    (b) raises ``AssertionError`` when the returned mismatch percent
    is *strictly greater* than ``strict_tol``.  Per-label pass/fail
    counts are accumulated in module-global state and dumped via
    ``atexit``.

    We patch by name on the module (not by re-binding the imported
    symbol) because the test scripts do ``from aiter.test_common
    import checkAllclose`` only *after* this bootstrap has already
    rebound the attribute; their import then gets the wrapper.  See
    aiter/test_common.py:274 for the upstream signature."""
    import aiter.test_common as tc

    original = tc.checkAllclose
    counts: dict[str, dict[str, int]] = collections.defaultdict(
        lambda: {"pass": 0, "fail": 0}
    )
    state = {"first_failure": None}

    def wrapper(a, b, rtol=1e-2, atol=1e-2, msg="",
                printNum=8, printLog=True):
        percent = original(
            a, b, rtol=rtol, atol=atol, msg=msg,
            printNum=printNum, printLog=printLog,
        )
        # The upstream returns either ``0`` (allclose) or the float
        # percent of mismatched elements.  Anything > strict_tol is
        # a real failure under our contract.
        #
        # We deliberately do NOT coerce odd return shapes (``None``,
        # strings, tuples) into a "pass".  If upstream ever changes
        # its contract we want to hear about it immediately, via an
        # honest ``TypeError`` that aborts the run, rather than
        # silently recording every comparison as 0% mismatched.
        # Same reason we refuse to paper over the ``percent is None``
        # case with ``or 0``: a None here is a contract bug, not a
        # pass.
        if percent is None:
            raise TypeError(
                "[aiter_corpus_runner] aiter.test_common.checkAllclose "
                "returned None — upstream contract says it returns a "
                "float mismatch percent (0 for allclose).  Refusing "
                "to treat None as a pass; investigate the upstream "
                "change before re-running the corpus sweep."
            )
        pct = float(percent)

        label = _label_key(msg)
        if pct > strict_tol:
            counts[label]["fail"] += 1
            if state["first_failure"] is None:
                state["first_failure"] = (label, pct, atol, rtol)
            raise AssertionError(
                f"[aiter_corpus_runner] checkAllclose label={label!r} "
                f"failed: {pct:.4%} of elements mismatched "
                f"(atol={atol}, rtol={rtol}, threshold={strict_tol})"
            )
        counts[label]["pass"] += 1
        return percent

    tc.checkAllclose = wrapper

    def _summary() -> None:
        if not counts:
            return
        total_pass = sum(c["pass"] for c in counts.values())
        total_fail = sum(c["fail"] for c in counts.values())
        print(
            f"[aiter_corpus_runner] checkAllclose: "
            f"{total_pass} pass / {total_fail} fail "
            f"(threshold={strict_tol})",
            file=sys.stderr,
            flush=True,
        )
        # Per-label breakdown so triage can immediately tell
        # torch-vs-asm mismatches apart from torch-vs-ck mismatches
        # in the same script.
        for label in sorted(counts):
            c = counts[label]
            mark = " FAIL" if c["fail"] else ""
            print(
                f"[aiter_corpus_runner]   "
                f"{c['pass']:>4} pass / {c['fail']:>4} fail  "
                f"label={label!r}{mark}",
                file=sys.stderr,
                flush=True,
            )
        if state["first_failure"] is not None:
            label, pct, atol, rtol = state["first_failure"]
            print(
                f"[aiter_corpus_runner] first failure: label={label!r} "
                f"percent={pct:.4%} atol={atol} rtol={rtol}",
                file=sys.stderr,
                flush=True,
            )

    atexit.register(_summary)

    print(
        f"[aiter_corpus_runner] patched aiter.test_common.checkAllclose "
        f"(raise on percent > {strict_tol})",
        file=sys.stderr,
        flush=True,
    )


# Sentinel value returned in place of the 101-iter average wall-clock
# us.  Must be (a) a plain float (some tests destructure as
# ``(data, avg) = run_perftest(...)`` then format ``avg``); (b)
# nonzero (several tests compute ``avg_a / avg_b - 1`` for an uplift
# percent, which divisions-by-zero out if we return 0.0); (c) not so
# small that downstream ``f"{avg:5.1%}"`` formatting overflows.  1.0
# satisfies all three and makes any uplift line render as "0.0%",
# which is correct: we ran the kernel exactly once, no perf signal
# was collected.
_PERFTEST_FAKE_AVG_US = 1.0


def _patch_perftest() -> None:
    """Replace ``aiter.test_common.perftest`` and ``run_perftest`` so
    the decorated callable is invoked exactly once and a fixed
    ``avg=_PERFTEST_FAKE_AVG_US`` is returned.  AITER tests
    typically destructure ``(data, avg) = run_torch(...)``; ``data``
    is the only thing the correctness path consumes.  Keeping
    ``avg`` as a plain float (not a pandas Series, as the unpatched
    ``get_trace_perf`` returns) matches the unpacking shape every
    op_test relies on."""
    import aiter.test_common as tc

    def perftest(num_iters: int = 101,
                 num_warmup: int = 2,
                 testGraph: bool = False,
                 num_rotate_args: int = 0,
                 needTrace: bool = False) -> Callable:
        def decorator(func: Callable) -> Callable:
            def wrapper(*args, **kwargs):
                data = func(*args, **kwargs)
                return data, _PERFTEST_FAKE_AVG_US
            return wrapper
        return decorator

    def run_perftest(func: Callable,
                     *args,
                     num_iters: int = 101,
                     num_warmup: int = 2,
                     testGraph: bool = False,
                     num_rotate_args: int = 0,
                     needTrace: bool = False,
                     **kwargs):
        return func(*args, **kwargs), _PERFTEST_FAKE_AVG_US

    tc.perftest = perftest
    tc.run_perftest = run_perftest

    print(
        f"[aiter_corpus_runner] patched aiter.test_common.perftest / "
        f"run_perftest (single-call, avg={_PERFTEST_FAKE_AVG_US})",
        file=sys.stderr,
        flush=True,
    )


def _seed_rng(seed_env: str) -> None:
    """Seed every RNG we know AITER's op-tests touch so the random
    inputs they build are deterministic across (native, legacy,
    salmon) invocations.  Called after monkey-patching but before
    ``runpy`` fires the user script.

    The seed itself comes from ``AITER_CORPUS_RNG_SEED`` (int).  An
    empty or unset value means "leave RNG alone" — we prefer the
    default of always seeding because the canonical asm-path
    smoke test (``test_moeTopkSoftmax.py``) uses ``torch.randn``
    at tight tolerances and unseeded inputs otherwise cause
    mode-dependent flakiness in sibling tests within the same
    script.  Seeding makes ``native`` either deterministically pass
    or deterministically fail a given label, which means a
    ``native != salmon`` verdict is always a real transpilation
    signal rather than RNG jitter.

    Importing torch here is safe: AITER's top-level ``import aiter``
    already imports torch, and the user op-test does the same — we
    just force the import a few lines earlier to seed it, which
    also conveniently warms up the CUDA context before the test's
    tensor allocations start."""
    if not seed_env:
        print(
            "[aiter_corpus_runner] RNG seeding disabled "
            "(AITER_CORPUS_RNG_SEED is empty); native/legacy/salmon "
            "numerical comparisons may be flaky",
            file=sys.stderr,
            flush=True,
        )
        return
    seed = int(seed_env)
    import random
    random.seed(seed)
    # numpy is an AITER hard dep — if it's missing the op-test would
    # fail on its own import line anyway, but we refuse to silently
    # skip numpy RNG seeding here: an ImportError at this point
    # means the environment is mis-configured and every downstream
    # AITER test will produce garbage.  Surface it now so the user
    # sees a clean "numpy is required" message instead of a confusing
    # numerical mismatch in the first test that happens to use
    # ``np.random``.
    import numpy as np
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        # ``manual_seed_all`` is the documented way to seed every
        # visible CUDA/HIP device; critically, it also reseeds the
        # default generator torch uses when a tensor is explicitly
        # placed on cuda via ``device="cuda"`` (which every AITER
        # op-test does at the top of the file via
        # ``torch.set_default_device("cuda")``).
        torch.cuda.manual_seed_all(seed)
    print(
        f"[aiter_corpus_runner] seeded torch/numpy/random with "
        f"{seed} (AITER_CORPUS_RNG_SEED); identical inputs across "
        f"native/legacy/salmon for this script + args",
        file=sys.stderr,
        flush=True,
    )


def _run_user_script(script: str) -> int:
    """Execute the script as if invoked directly so its
    ``if __name__ == '__main__':`` block fires (and any top-level
    test loops run too).  Make sure the script's own directory is
    on ``sys.path`` so it can import sibling modules.

    AITER's op_tests are mostly argparse-driven scripts that read
    their own ``sys.argv``; if we leave it pointing at the
    ``-m _bootstrap`` invocation, every test trips on an unknown
    arg.  Reset it to look exactly like ``python <script> [args]``
    would, where ``[args]`` is whatever the parent passed via the
    shlex-quoted ``AITER_CORPUS_SCRIPT_ARGS`` envelope."""
    import shlex
    script_dir = os.path.dirname(os.path.abspath(script))
    if script_dir and script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    extra = os.environ.get("AITER_CORPUS_SCRIPT_ARGS", "")
    extra_args = shlex.split(extra) if extra else []
    sys.argv = [script, *extra_args]

    try:
        runpy.run_path(script, run_name="__main__")
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else (
            0 if e.code is None else 1
        )
        return code
    except BaseException:
        traceback.print_exc()
        return 1
    return 0


def main() -> int:
    gfx = _require_env("AITER_CORPUS_FORCE_GFX")
    _set_gpu_archs(gfx)

    script = _require_env("AITER_CORPUS_SCRIPT")

    strict_tol = float(os.environ.get("AITER_CORPUS_STRICT_TOL", "0.01"))

    # Import + patch must happen *before* runpy fires the user
    # script.  Importing aiter at this point also gives the user a
    # clean stderr line if the install is broken (rather than a
    # confusing failure five seconds into the test).
    try:
        _patch_check_allclose(strict_tol)
        _patch_perftest()
    except BaseException:
        traceback.print_exc()
        print(
            "[aiter_corpus_runner] bootstrap patch failed — refusing "
            "to run the user script with an un-patched checkAllclose",
            file=sys.stderr,
            flush=True,
        )
        return 2

    # Seed *after* aiter has been imported-and-patched (both patches
    # above do ``import aiter.test_common``, which pulls in torch
    # transitively).  Seeding here means the user script's very
    # first ``torch.randn`` call sees a generator primed to the
    # runner-controlled seed.
    seed_env = os.environ.get("AITER_CORPUS_RNG_SEED", "0").strip()
    try:
        _seed_rng(seed_env)
    except BaseException:
        traceback.print_exc()
        print(
            "[aiter_corpus_runner] RNG seeding failed — refusing to "
            "run with non-deterministic inputs because the runner's "
            "cross-mode comparison relies on identical tensors",
            file=sys.stderr,
            flush=True,
        )
        return 2

    return _run_user_script(script)


if __name__ == "__main__":
    sys.exit(main())
