#!/usr/bin/env python3
"""Run AITER's own op_tests under each transpilation mode and report
which scripts crash, hang, or produce wrong results.

This is the AITER-corpus counterpart to ``triton_corpus_runner``: it
exercises the AOT-compiled kernels AITER ships under ``hsa/<gfx>/``
by forcing AITER's loader to pick the gfx950 binaries on a gfx942
box, which trips the ROCR hotswap hook and routes every
``hipModuleLoad`` through the Salmon transpiler.  The op-test scripts
already do ``checkAllclose(asm_output, torch_reference)`` themselves
— pytest's exit code (after we patch ``checkAllclose`` to actually
raise on failure) is the per-script verdict.

Per (script, mode) we record one of:

    PASS         exit 0
    FAIL         exit non-zero, normal termination (e.g. assertion
                 failure inside the script — the script ran far
                 enough to compare results).
    CRASH        terminated by signal (SIGSEGV / SIGABRT / SIGILL...)
    HANG         exceeded ``--timeout`` and was SIGKILLed.
    SPAWN_FAIL   the subprocess couldn't even start (missing python,
                 missing libsalmon_intercept, etc.).

The interesting comparisons are columnar: a script that PASSes under
``native`` (= AITER's own gfx942 .co set, no preload) but CRASHes /
HANGs / FAILs under ``legacy`` or ``salmon`` (= gfx950 .co set
forced through the hotswap path) is a real transpilation gap.  A
script that already FAILs under ``native`` is a problem with the
script itself (or the local AITER install) and the summary calls
those out separately so they don't pollute the salmon-coverage
signal.

The runner deliberately makes no attempt to parse the test scripts,
infer their kernels, or interpret their output — the per-script
verdict is exit-status plus the patched-``checkAllclose`` per-label
breakdown the bootstrap prints to stderr.  Adding a new script is
just dropping it under ``op_tests/``.

See ``README.md`` for the full setup recipe and the
``Reference & isolation`` section explaining why only AITER's asm
path goes through Salmon while the PyTorch and CK references
continue to run natively on gfx942 hardware.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import datetime
import functools
import json
import os
import queue
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
from typing import Dict, List, Optional, Set, Tuple


HERE = os.path.dirname(os.path.abspath(__file__))


# -- Child process registry + Ctrl-C handling ----------------------------
#
# We spawn every child with ``start_new_session=True`` so that a per-run
# 600s timeout can SIGKILL the child's whole session (hipcc + HSA
# background threads + any grandchildren).  Side effect: each child is
# in a *different* session than the runner, so a shell-level Ctrl-C on
# the runner does not propagate to the children.  Without the machinery
# below, Ctrl-C therefore leaks N orphan python processes (one per live
# worker) reparented to init — each still spinning in HSA wait loops
# or holding JIT-cache FileBaton locks.
#
# Fix: the parent tracks every live child's (pid, pgid) in a module-
# level registry protected by a lock, and installs SIGINT/SIGTERM
# handlers that:
#
#   * first signal → SIGTERM every live child session (graceful),
#     raise KeyboardInterrupt so main() unwinds;
#   * second signal (user impatient) → SIGKILL every survivor and
#     ``os._exit`` immediately, bypassing any worker-thread stalls.
#
# ``_register_child`` is called right after ``Popen`` returns; the
# pgid equals the pid because ``start_new_session=True`` makes the
# child a session leader.  ``_unregister_child`` runs in a
# ``finally`` inside ``_run_one`` so the registry is always consistent
# even if the pump threads or ``proc.wait`` raise.
_live_children_lock = threading.Lock()
_live_children: Dict[int, int] = {}  # pid -> pgid (== pid under start_new_session)
_shutdown_signals_received = 0
_shutdown_count_lock = threading.Lock()


def _register_child(pid: int) -> None:
    with _live_children_lock:
        _live_children[pid] = pid  # pgid == pid via start_new_session


def _unregister_child(pid: int) -> None:
    with _live_children_lock:
        _live_children.pop(pid, None)


def _kill_live_children(sig: int) -> int:
    """SIGnal every live child's process *group*.  Swallows
    ``ProcessLookupError`` (race: child already exited / was reaped by
    another thread).  Returns the number of pgids we actually signalled
    (excluding already-dead ones)."""
    with _live_children_lock:
        snapshot = list(_live_children.items())
    signalled = 0
    for pid, pgid in snapshot:
        try:
            os.killpg(pgid, sig)
            signalled += 1
        except ProcessLookupError:
            continue
    return signalled


def _shutdown_handler(signum: int, frame) -> None:
    """SIGINT / SIGTERM handler.  First signal is graceful (SIGTERM to
    every child session, then raise KeyboardInterrupt to let main()
    unwind through its finally blocks and emit a partial summary).
    Second signal escalates to SIGKILL + ``os._exit`` — covers the case
    where a child ignores SIGTERM because it's wedged inside an
    uninterruptible HSA / hipcc call."""
    global _shutdown_signals_received
    with _shutdown_count_lock:
        _shutdown_signals_received += 1
        escalating = _shutdown_signals_received > 1
    if escalating:
        n = _kill_live_children(signal.SIGKILL)
        sys.stderr.write(
            f"\n[aiter_corpus_runner] signal {signum} again: SIGKILLed "
            f"{n} surviving child session(s); exiting immediately.\n"
        )
        sys.stderr.flush()
        os._exit(128 + signum)
    n = _kill_live_children(signal.SIGTERM)
    sys.stderr.write(
        f"\n[aiter_corpus_runner] caught signal {signum}; SIGTERM'd "
        f"{n} live child session(s).  Press Ctrl-C again to SIGKILL "
        f"and exit immediately.\n"
    )
    sys.stderr.flush()
    raise KeyboardInterrupt()


def _install_shutdown_handlers() -> None:
    """Install our teardown handler for SIGINT and SIGTERM.  Must be
    called from the main thread before spawning any children."""
    signal.signal(signal.SIGINT, _shutdown_handler)
    signal.signal(signal.SIGTERM, _shutdown_handler)


@functools.lru_cache(maxsize=1)
def _detect_native_gfx() -> str:
    """Return the gfx arch of the live GPU as reported by rocminfo
    (e.g. ``"gfx942"``).  Cached for the runner's lifetime.

    This mirrors AITER's own ``chip_info._detect_native()`` so the
    arch string we feed into ``GPU_ARCHS`` is byte-identical to what
    AITER would auto-detect; anything else risks a mismatch in the
    ``AITER_GPU_ARCHS`` -> codegen.py -> cfg_* lookup chain.

    We detect at runner-process time (not child time) because every
    child needs the same arch, and shelling out to rocminfo once up
    front is cheap.  A clear error here is strictly better than
    letting AITER's ``rocminfo not found`` exception trip deep inside
    the child where the stderr only shows a truncated traceback.
    """
    rocminfo = None
    for candidate in ("rocminfo", "/opt/rocm/bin/rocminfo"):
        try:
            subprocess.check_output(
                [candidate, "--help"],
                stderr=subprocess.DEVNULL,
            )
            rocminfo = candidate
            break
        except (OSError, subprocess.CalledProcessError):
            continue
    if rocminfo is None:
        raise RuntimeError(
            "rocminfo not found in PATH or /opt/rocm/bin; cannot "
            "detect the native gfx arch.  Install rocminfo (part of "
            "the ROCm install) or override via --native-gfx."
        )
    try:
        out = subprocess.check_output(
            [rocminfo],
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"rocminfo ({rocminfo}) failed with exit {e.returncode}; "
            f"stderr:\n{e.stderr}"
        ) from e
    for line in out.splitlines():
        m = re.search(r"\b(gfx\w+)\b", line, re.IGNORECASE)
        if m:
            return m.group(1).lower()
    raise RuntimeError(
        "rocminfo ran but produced no gfx arch line; output was:\n"
        + out
    )


# The Triton runner's sibling rocm-7 venv works as-is for AITER too —
# both want a torch wheel built against ROCm 7.x, plus pytest + numpy.
# AITER's extra deps (pandas, einops, psutil, pybind11>=2.13,<3) are
# pip-installed into this venv on ``--setup``.
DEFAULT_TRITON_VENV = os.path.normpath(
    os.path.join(HERE, "..", "triton_corpus_runner", ".venv-rocm7")
)

# Same intercept shim as the Triton runner; built by
# ``compare_correctness/Makefile`` (``make libsalmon_intercept.so``).
DEFAULT_LIBSALMON = os.path.normpath(
    os.path.join(HERE, "..", "compare_correctness", "libsalmon_intercept.so")
)
# Salmon-enabled libhsa-runtime64.so.  Same path the Triton runner
# defaults to.
DEFAULT_LIBHSA = os.path.normpath(os.path.expanduser(
    "~/rocm-systems/projects/rocr-runtime/build/rocr/lib/libhsa-runtime64.so.1"
))
# Empty by default — the rocm-7 torch wheel ships a HIP runtime that
# tolerates the Salmon ROCR's multi-ISA agent enumeration directly.
DEFAULT_LIBAMDHIP = ""
# Local arch-spoofing shim that hooks hipGetDevicePropertiesR0000 /
# R0600 and rewrites gcnArchName to AITER_CORPUS_SPOOF_ARCH (only
# loaded for legacy/salmon modes).  Built by this directory's
# Makefile (``make libaiter_arch_spoof.so``).  Required because
# AITER's C++ asm loader keys off hipGetDeviceProperties().gcnArchName,
# *not* the GPU_ARCHS env var, when picking which hsa/<arch>/*.co
# binary to feed hipModuleLoadData — without spoofing, a gfx942
# device always reads hsa/gfx942/* and the gfx950 .co files Salmon
# is supposed to translate never get loaded at all.  See
# arch_spoof.cpp for the full rationale.
DEFAULT_LIBARCH_SPOOF = os.path.normpath(
    os.path.join(HERE, "libaiter_arch_spoof.so")
)

EMPTY_RULES = os.path.join(HERE, "_empty_rules.json")

# Auto-clone target.  ``--setup`` does ``git clone --recursive
# https://github.com/ROCm/aiter.git`` here on first run.
DEFAULT_AITER_ROOT = os.path.join(HERE, "_aiter")
DEFAULT_AITER_REPO = "https://github.com/ROCm/aiter.git"
# Persistent JIT-build cache for AITER's ``compile_ops`` decorator.
# Pinning this keeps hipcc compiles from being re-done on every run
# and keeps the artefacts out of the AITER source tree (so a re-clone
# doesn't lose them).
DEFAULT_JIT_CACHE = os.path.join(HERE, "_jit_cache")

# Foreign source ISA we want Salmon to translate.  This is used for
# two distinct-but-coupled things in the ``legacy`` / ``salmon``
# modes:
#
#   1. It's rewritten into ``hipGetDeviceProperties().gcnArchName``
#      by libaiter_arch_spoof.so so AITER's C++ asm loader reads
#      ``hsa/gfx950/<kernel>.co`` instead of
#      ``hsa/gfx942/<kernel>.co``.  The resulting ``hipModuleLoadData``
#      call then trips libsalmon_intercept.so → Salmon transpile
#      chain.
#
#   2. It's appended to the native gfx when building ``GPU_ARCHS``
#      for the transpile modes, so AITER's Python-side sees
#      ``GPU_ARCHS="gfx942;gfx950"``.  This has two effects that
#      together make the asm path actually hand us a gfx950 .co:
#        a) ``codegen.py`` keys off ``AITER_GPU_ARCHS`` to decide
#           which per-arch ``hsa/<gfx>/*.csv`` rows get emitted into
#           ``asm_<module>_configs.hpp`` — ``get_heuristic_kernel_*``
#           in the .cu files then filters this map by ``arch_id``
#           (the spoofed gfx950 string from (1) above).  Without a
#           gfx950 row in the map, the heuristic lookup aborts and
#           the asm .co is never loaded at all, so Salmon is never
#           asked to translate anything — observed directly on
#           test_moeTopkSoftmax.py where the first run with
#           ``GPU_ARCHS=native`` died with ``cannot get heuristic
#           kernel! arch_id:gfx950 ... dtype:bf16 ...``.
#        b) ``validate_and_update_archs()`` in ``aiter/jit/core.py``
#           forwards each entry as a separate ``--offload-arch=`` to
#           hipcc, so the non-asm host modules (module_moe_asm.so,
#           module_aiter_core.so, ...) are built as fat binaries
#           containing device code for both ISAs.  HIP picks the
#           gfx942 slice at launch time because our shim does
#           *not* intercept the internal HSA agent-info path HIP
#           uses to identify devices — only the public
#           ``hipGetDeviceProperties`` entry point that asm-path
#           dispatch reads.  Building single-arch gfx950 on gfx942
#           hardware crashed in ``hip::StatCO::getStatFunc`` on
#           first launch; adding gfx942 to the fat binary fixes
#           that without breaking the spoof.
#
# The native mode deliberately does *not* include gfx950 in
# GPU_ARCHS (and does not load the spoof shim either), so its
# ``cfg_topksoftmax`` map has only gfx942 entries and its host
# modules are single-arch gfx942 — which is exactly what a clean
# AITER-on-gfx942 baseline should look like.
DEFAULT_SOURCE_GFX = "gfx950"

# Curated single-GPU subset (chosen with the user) that exercises
# AITER's asm path without needing multi-GPU or out-of-tree fixtures.
# ``--all`` opts in to everything else.  ``test_moeTopkSoftmax.py``
# leads the list because it is the smallest end-to-end script that
# unambiguously routes through AITER's asm path (``topk_softmax_asm``)
# — i.e. the ``hipModuleLoadData`` chain Salmon needs to engage.
CURATED_SUBSET = [
    "test_moeTopkSoftmax.py",
    "test_layernorm2d.py",
    "test_layernorm2dFusedAddQuant.py",
    "test_rmsnorm2d.py",
    "test_rmsnorm2dFusedAddQuant.py",
    "test_pa.py",
    "test_pa_v1.py",
    "test_mla.py",
    "test_mha.py",
    "test_mha_varlen.py",
    "test_gemm_a16w16.py",
    "test_gemm_a8w8.py",
    "test_gemm_a4w4.py",
    "test_quant.py",
]


@dataclasses.dataclass
class ModeSpec:
    name: str
    description: str
    needs_preload: bool
    needs_ir_raiser: bool
    # Same opt-in strict tightening the Triton runner uses for the
    # salmon path: promotes known silent-miscompile sites
    # (MODE-register writes, implicitarg.ptr cross-arch lifts) to
    # honest refusals instead of warn-and-continue.  See
    # ``transpiler/pipeline.hpp::isStrictMode``.
    needs_strict_mode: bool = False
    # If set, libaiter_arch_spoof.so is added to LD_PRELOAD and
    # AITER_CORPUS_SPOOF_ARCH is set to the foreign ISA we want
    # AITER's C++ asm loader to read .co files for (gfx950 in
    # practice).  The Python-side ``GPU_ARCHS`` env var is *always*
    # left at ``native`` regardless of mode — see the comment on
    # DEFAULT_SOURCE_GFX above for why.
    needs_arch_spoof: bool = False


MODES: Dict[str, ModeSpec] = {
    "native": ModeSpec(
        name="native",
        description="AITER's gfx942 .co set, no LD_PRELOAD, no transpile",
        needs_preload=False,
        needs_ir_raiser=False,
        needs_arch_spoof=False,
    ),
    "legacy": ModeSpec(
        name="legacy",
        description="AITER's gfx950 .co set -> hotswap byte translator -> gfx942",
        needs_preload=True,
        needs_ir_raiser=False,
        needs_arch_spoof=True,
    ),
    "salmon": ModeSpec(
        name="salmon",
        description="AITER's gfx950 .co set -> hotswap Salmon IR raiser -> gfx942",
        needs_preload=True,
        needs_ir_raiser=True,
        needs_strict_mode=True,
        needs_arch_spoof=True,
    ),
}


@dataclasses.dataclass
class RunResult:
    script: str
    mode: str
    verdict: str           # PASS / FAIL / CRASH / HANG / SPAWN_FAIL
    detail: str            # short human-readable detail for the report
    exit_code: Optional[int]
    signal_name: Optional[str]
    elapsed_s: float
    stderr_tail: str


# -- subprocess plumbing ----------------------------------------------------


def _build_env(
    mode: ModeSpec,
    aiter_root: str,
    triton_venv: str,
    libsalmon: str,
    libhsa: str,
    libamdhip: str,
    libarch_spoof: str,
    spoof_arch: str,
    native_gfx: str,
    strict_tol: float,
    perftest_iters: int,
    jit_cache: str,
    visible_devices: Optional[str],
    rng_seed: str,
) -> Dict[str, str]:
    """Construct the environment for one child run.  Start from the
    parent env and selectively override; that keeps things like
    HOME / DISPLAY / ROCM_PATH that AITER may rely on."""
    env = dict(os.environ)

    # Make sure the bootstrap module is importable.
    env["PYTHONPATH"] = HERE + ":" + env.get("PYTHONPATH", "")

    # Prepend the AITER checkout so ``import aiter`` resolves to the
    # source tree we control.  Lazy ``compile_ops`` builds the C++
    # side on first use; we don't require ``setup.py develop`` to
    # have been run.
    env["PYTHONPATH"] = aiter_root + ":" + env["PYTHONPATH"]

    # Persistent JIT-build cache so AITER's per-op hipcc compiles
    # survive across runs and don't dominate wall clock after the
    # first sweep.  AITER's loader honours ``AITER_JIT_DIR`` (see
    # aiter/jit/core.py::get_user_jit_dir); pointing it at a runner-
    # owned directory also keeps the resulting *.so files out of the
    # checked-in source tree so a fresh ``--setup`` clone can't
    # accidentally pick up a previous run's gfx950-targeted cache
    # and crash on first import.
    os.makedirs(jit_cache, exist_ok=True)
    env["AITER_JIT_DIR"] = jit_cache

    # Avoid matplotlib trying to open a display on a tutorial-style
    # test that imports it for benchmark plots.
    env["MPLBACKEND"] = "Agg"

    # Defensive isolation: pin to a single GPU so any pathological
    # transpiled kernel that managed to wedge an SQ/CU only forces
    # an amdgpu queue reset on *one* device.  Same rationale as the
    # Triton runner.
    if visible_devices is not None:
        env["HIP_VISIBLE_DEVICES"] = visible_devices
    else:
        env.setdefault("HIP_VISIBLE_DEVICES", "0")

    if mode.needs_preload:
        for path, kind, hint in (
            (libhsa, "Salmon-enabled libhsa-runtime64.so",
             "build the Salmon ROCR tree first or override --libhsa"),
            (libsalmon, "libsalmon_intercept.so",
             "build compare_correctness first or override --libsalmon"),
            (libarch_spoof, "libaiter_arch_spoof.so",
             "run 'make' in tools/aiter_corpus_runner/ first, or "
             "override --libarch-spoof"),
        ):
            if not os.path.exists(path):
                raise RuntimeError(
                    f"{kind} not found at {path}; {hint}"
                )

        preload_chain: List[str] = []
        ldlib_dirs: List[str] = []
        if libamdhip:
            if not os.path.exists(libamdhip):
                raise RuntimeError(
                    f"--libamdhip {libamdhip!r} does not exist; pass an "
                    f"empty string to disable libamdhip preload"
                )
            preload_chain.append(libamdhip)
            ldlib_dirs.append(os.path.dirname(libamdhip))

        # arch-spoof first so its dlsym(RTLD_NEXT, ...) walks past
        # itself to find the real hipGetDevicePropertiesR0600 in
        # libamdhip64.  libhsa next so its rocr_salmon_patch_elf
        # symbol is in global scope when libsalmon_intercept's init
        # runs.  intercept last so its dlsym(RTLD_DEFAULT, ...)
        # lookup of rocr_salmon_patch_elf succeeds.
        preload_chain.extend([libarch_spoof, libhsa, libsalmon])
        ldlib_dirs.append(os.path.dirname(libhsa))

        env["LD_PRELOAD"] = ":".join(preload_chain)
        env["LD_LIBRARY_PATH"] = ":".join(
            ldlib_dirs + [env.get("LD_LIBRARY_PATH", "")]
        ).rstrip(":")
        env.setdefault("HSA_HOTSWAP_ISA_OVERRIDE", "gfx942")
        env.setdefault("HSA_HOTSWAP_RULES", EMPTY_RULES)
    else:
        # Native runs must be *clean* — no inherited preload from an
        # interactive shell that already had one set.
        env.pop("LD_PRELOAD", None)
        env.pop("HSA_HOTSWAP_ISA_OVERRIDE", None)
        env.pop("HSA_HOTSWAP_IR_RAISER", None)
        env.pop("HSA_HOTSWAP_RULES", None)

    if mode.needs_ir_raiser:
        env["HSA_HOTSWAP_IR_RAISER"] = "1"
    elif mode.needs_preload:
        env.pop("HSA_HOTSWAP_IR_RAISER", None)

    if mode.needs_strict_mode:
        env["HSA_SALMON_STRICT"] = "1"
    else:
        env.pop("HSA_SALMON_STRICT", None)

    # GPU_ARCHS — what AITER's Python layer sees (via
    # AITER_CORPUS_FORCE_GFX → bootstrap → ``GPU_ARCHS`` /
    # ``AITER_GPU_ARCHS``).  We deliberately use the SAME multi-arch
    # value for every mode so the on-disk JIT cache under
    # ``AITER_JIT_DIR`` (which AITER keys on a hash of build flags,
    # not on the mode we chose) is byte-identical across native,
    # legacy, and salmon runs.  Without this, running ``native`` first
    # would populate the cache with a single-arch
    # ``asm_topksoftmax_configs.hpp`` (only ``gfx942`` rows), and the
    # subsequent transpile runs — which share that cache — would
    # abort in ``get_heuristic_kernel_topksoftmax`` with
    # ``cannot get heuristic kernel! arch_id:gfx950`` the moment the
    # spoofed asm loader queries for the foreign-ISA row.  Always
    # emitting both rows costs a few extra kilobytes of hipcc
    # compilation (multi-``--offload-arch``) and zero runtime overhead
    # on gfx942 hardware, because the spoof-free native run still
    # dispatches via the real ``gcnArchName`` (= ``gfx942``).
    if spoof_arch and spoof_arch != native_gfx:
        env["AITER_CORPUS_FORCE_GFX"] = f"{native_gfx};{spoof_arch}"
    else:
        env["AITER_CORPUS_FORCE_GFX"] = native_gfx

    # Arch spoof.  Only the transpile modes get a spoofed
    # gcnArchName, because that's what tricks AITER's C++ asm loader
    # into reading the foreign-ISA .co files we want Salmon to
    # translate.  Native must see the *real* device arch so its
    # baseline reflects what the hardware can actually run without
    # transpilation.  We always wipe the variable before writing it
    # so an inherited shell value can't accidentally turn native
    # into a spoofed run.
    env.pop("AITER_CORPUS_SPOOF_ARCH", None)
    if mode.needs_arch_spoof and spoof_arch:
        env["AITER_CORPUS_SPOOF_ARCH"] = spoof_arch

    env["AITER_CORPUS_STRICT_TOL"] = f"{strict_tol}"
    env["AITER_CORPUS_PERFTEST_ITERS"] = f"{perftest_iters}"
    env["AITER_CORPUS_RNG_SEED"] = rng_seed

    return env


def _classify_exit(rc: Optional[int]) -> tuple[str, Optional[str]]:
    if rc is None:
        return "HANG", None
    if rc < 0:
        sig = -rc
        try:
            name = signal.Signals(sig).name
        except ValueError:
            name = f"signal {sig}"
        return "CRASH", name
    if rc == 0:
        return "PASS", None
    return "FAIL", None


def _tail(text: str, n: int = 16000) -> str:
    if len(text) <= n:
        return text
    return "...[truncated]\n" + text[-n:]


def _build_run_spec(
    script: str,
    mode: ModeSpec,
    aiter_root: str,
    triton_venv: str,
    libsalmon: str,
    libhsa: str,
    libamdhip: str,
    libarch_spoof: str,
    spoof_arch: str,
    native_gfx: str,
    strict_tol: float,
    perftest_iters: int,
    jit_cache: str,
    visible_devices: Optional[str],
    script_args: List[str],
    rng_seed: str,
) -> tuple[List[str], Dict[str, str]]:
    """Compute the argv + env for one (script, mode) run, without
    actually spawning anything.  Factored out of ``_run_one`` so that
    ``--print-command`` can emit a byte-identical reproduction recipe
    for a user who wants to re-run the child manually under gdb /
    strace / rocgdb / AMD_LOG_LEVEL=5, with zero risk of drift
    between the printed command and the one the runner uses."""
    env = _build_env(
        mode, aiter_root, triton_venv, libsalmon, libhsa, libamdhip,
        libarch_spoof, spoof_arch, native_gfx,
        strict_tol, perftest_iters, jit_cache, visible_devices,
        rng_seed,
    )
    env["AITER_CORPUS_SCRIPT"] = os.path.abspath(script)
    # ``script_args`` is forwarded to the user op-test as if it had
    # been invoked directly: ``python <script> arg1 arg2 ...``.  See
    # _bootstrap._run_user_script for how this is reconstructed onto
    # ``sys.argv``.  We use a shlex-quoted single-string envelope —
    # POSIX env values cannot contain NUL bytes, and shlex round-trips
    # cleanly for any whitespace / quoting an op-test arg might need.
    if script_args:
        env["AITER_CORPUS_SCRIPT_ARGS"] = shlex.join(script_args)
    else:
        env.pop("AITER_CORPUS_SCRIPT_ARGS", None)

    python = os.path.join(triton_venv, "bin", "python")
    if not os.path.exists(python):
        raise RuntimeError(
            f"venv python not found at {python}; "
            f"override --triton-venv if your venv lives elsewhere"
        )
    cmd = [python, "-m", "_bootstrap"]
    return cmd, env


def _format_print_command(
    script_label: str,
    mode_name: str,
    cmd: List[str],
    env: Dict[str, str],
    cwd: str,
) -> str:
    """Render (cmd, env) as a paste-ready multi-line shell command.
    Only the env vars that *differ* from the parent runner's
    environment are emitted — the parent env is inherited verbatim
    via ``dict(os.environ)`` in ``_build_env`` anyway, so printing
    every inherited DISPLAY / XDG_* / PATH would drown the actual
    runner-specific variables.  We additionally *include* any
    runner-specific variable that the parent happens to have set to
    the same value (e.g. AITER_CORPUS_FORCE_GFX would be inherited
    if the user set it manually), because those are load-bearing
    for reproducibility and a reader shouldn't have to know which
    env vars belong to which layer.

    Returns a string ready for stdout.  Uses shlex.quote for every
    value so it's safe to paste into bash even with shell
    metacharacters.
    """
    runner_keys_prefixes = (
        "AITER_", "HSA_", "HIP_", "GPU_ARCHS", "LD_PRELOAD",
        "LD_LIBRARY_PATH", "PYTHONPATH", "MPLBACKEND",
    )

    def _is_runner_owned(k: str) -> bool:
        if k == "GPU_ARCHS":
            return True
        return any(k.startswith(p) for p in runner_keys_prefixes)

    parent = os.environ
    interesting = {}
    for k, v in env.items():
        if _is_runner_owned(k) or parent.get(k) != v:
            interesting[k] = v

    lines = [
        f"# aiter_corpus_runner :: script={script_label!r} mode={mode_name!r}",
        f"#   — {len(interesting)} env var(s) different from the runner's "
        f"parent shell; inherit everything else",
        f"(cd {shlex.quote(cwd)} && \\",
    ]
    for k in sorted(interesting):
        lines.append(f"  {k}={shlex.quote(interesting[k])} \\")
    lines.append("  " + " ".join(shlex.quote(c) for c in cmd) + ")")
    return "\n".join(lines)


def _sanitize_log_component(s: str) -> str:
    """Make a path component safe for log filenames — AITER scripts
    can contain double underscores, nested dirs, unicode-shaped
    labels, etc.  Collapse to ``[A-Za-z0-9._-]``."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("._-") or "unnamed"


# -- Parallel dispatch ------------------------------------------------------
#
# The serial and parallel paths share ``_run_one`` — the differences
# below live entirely in how we *arrange* the (script, mode) jobs and
# which GPU each child gets.  Three primitives:
#
#   * ``_resolve_gpu_pool`` — turn ``--gpus`` (explicit) or the
#     parent shell's ``HIP_VISIBLE_DEVICES`` (implicit) into a list
#     of device IDs the workers can round-robin over.  No GPU
#     auto-discovery: we deliberately stay within whatever the user
#     already made visible to this shell, so running this tool on
#     a shared box never steps on someone else's GPU.
#
#   * ``_reap_orphan_build_locks`` — AITER serialises concurrent
#     @compile_ops builds of the same module via a FileBaton (empty
#     marker file created with O_CREAT|O_EXCL, unlinked by
#     ``baton.release()`` in a ``finally``).  SIGKILL / OOM / Ctrl-C
#     skip finally blocks, leaving the marker on disk; any future
#     concurrent build of that module then spins forever in
#     ``baton.wait()``.  We detect orphans at runner startup by
#     cross-referencing each ``build/lock_<md>`` file's inode against
#     every open FD on the host — a live owner still holds the FD,
#     a crashed owner does not.  If ``/proc`` inspection is
#     incomplete (permission denied on any pid's fd dir) we refuse
#     to reap: distinguishing "orphan" from "held by a process we
#     can't inspect" is not possible and stomping a live builder
#     corrupts its ``.so`` output.
#
#   * ``_run_parallel`` — a ``ThreadPoolExecutor`` fronting one
#     child-process worker per pool slot, plus a ``queue.Queue[str]``
#     of GPU IDs.  Each worker claims a GPU on entry, runs one child
#     to completion, releases the GPU.  We don't allow the pool
#     size to exceed the GPU pool: two workers on the same device
#     index would either OOM or thrash, and the speedup we're
#     after is strictly 1 GPU = 1 job.  Cold-cache concurrent
#     builds are correctly serialised by AITER's own FileBaton —
#     there is no runner-side "warmup" pass.
def _resolve_gpu_pool(arg_gpus: str, jobs: int) -> List[str]:
    """Produce the list of GPU IDs the parallel pool will draw from.

    Priority is:

      1. Explicit ``--gpus`` value (accepts ``0,1,2`` or ``0-3`` or
         ``0,2-5,7``).
      2. The parent shell's ``HIP_VISIBLE_DEVICES`` environment
         variable (same format).
      3. Default ``["0"]`` — only when ``--jobs == 1``.

    We deliberately do NOT scan the machine for GPUs.  The user
    said "use HIP_VISIBLE_DEVICES for discovery" and we honour that
    literally — on a shared box that's the only safe source of
    truth for "which GPUs is this shell allowed to touch".

    Raises ``RuntimeError`` on unparseable ranges, empty lists, or
    ``--jobs > 1`` with no GPU source configured.
    """
    raw = (arg_gpus or "").strip()
    source = "--gpus"
    if not raw:
        raw = os.environ.get("HIP_VISIBLE_DEVICES", "").strip()
        source = "HIP_VISIBLE_DEVICES"
    if not raw:
        if jobs > 1:
            raise RuntimeError(
                f"--jobs {jobs} needs a GPU list; pass --gpus "
                f"'0,1,2,...' (or a range like '0-7') or set "
                f"HIP_VISIBLE_DEVICES in this shell before invoking "
                f"the runner.  --jobs 1 (the default) falls back to "
                f"GPU 0 and needs no --gpus/HIP_VISIBLE_DEVICES."
            )
        return ["0"]

    gpus: List[str] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            lo_s, hi_s = token.split("-", 1)
            try:
                lo, hi = int(lo_s), int(hi_s)
            except ValueError as e:
                raise RuntimeError(
                    f"{source}: cannot parse range {token!r} "
                    f"(expected integer-integer)"
                ) from e
            if lo > hi:
                raise RuntimeError(
                    f"{source}: empty range {token!r} (lo > hi)"
                )
            gpus.extend(str(i) for i in range(lo, hi + 1))
        else:
            # Validate it's an int but keep the string form — HIP
            # cares about the textual value of the env var, not
            # Python's normalised int repr.
            try:
                int(token)
            except ValueError as e:
                raise RuntimeError(
                    f"{source}: {token!r} is not a valid GPU index"
                ) from e
            gpus.append(token)
    if not gpus:
        raise RuntimeError(
            f"{source}={raw!r} parsed to an empty GPU list"
        )
    # Preserve user-specified order; deduplicate only on exact
    # repeats (e.g., "--gpus 0,1,0" → ["0","1"]).
    seen = set()
    deduped = []
    for g in gpus:
        if g not in seen:
            seen.add(g)
            deduped.append(g)
    return deduped


def _run_one(
    script: str,
    mode: ModeSpec,
    aiter_root: str,
    triton_venv: str,
    libsalmon: str,
    libhsa: str,
    libamdhip: str,
    libarch_spoof: str,
    spoof_arch: str,
    native_gfx: str,
    strict_tol: float,
    perftest_iters: int,
    jit_cache: str,
    timeout_s: float,
    visible_devices: Optional[str],
    script_args: List[str],
    rng_seed: str,
    tee_stderr: bool,
    log_dir: Optional[str],
    script_label: str,
) -> RunResult:
    """Spawn one child process for one (script, mode) and wait.

    Streams the child's stdout+stderr via threaded pumps instead of
    ``proc.communicate()`` so we can tee to the parent terminal in
    real time (``tee_stderr``) and write the full unbuffered stream
    to a per-run log file (``log_dir``) without blowing memory or
    risking the child blocking on a full pipe buffer.  The in-memory
    tail we return in ``stderr_tail`` is kept bounded — see ``_tail``
    for the cutoff."""
    cmd, env = _build_run_spec(
        script, mode, aiter_root, triton_venv, libsalmon, libhsa,
        libamdhip, libarch_spoof, spoof_arch, native_gfx,
        strict_tol, perftest_iters, jit_cache, visible_devices,
        script_args, rng_seed,
    )

    # Per-run log file.  Open before spawn so a setup failure here
    # (bad log-dir path, no disk space) raises before we've started
    # anything expensive.  The filename is
    # <script>__<mode>__<ts>__<pid>.log with all non-safe chars
    # collapsed.  Timestamp is local time with microsecond precision
    # (``%Y%m%d-%H%M%S-%f``); the PID component disambiguates two
    # concurrent runner invocations on the same box.  Microseconds
    # matter in ``--jobs > 1`` mode — several worker threads share
    # the runner's PID and can open log files within the same
    # one-second bucket, so a ``%Y%m%d-%H%M%S`` stamp would collide.
    log_fh = None
    log_path: Optional[str] = None
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        fname = (
            f"{_sanitize_log_component(script_label)}"
            f"__{mode.name}__{ts}__pid{os.getpid()}.log"
        )
        log_path = os.path.join(log_dir, fname)
        log_fh = open(log_path, "wb", buffering=0)
        header = (
            f"# aiter_corpus_runner log\n"
            f"# script: {script}\n"
            f"# mode:   {mode.name}\n"
            f"# cmd:    {' '.join(shlex.quote(c) for c in cmd)}\n"
            f"# cwd:    {HERE}\n"
            f"# time:   {datetime.datetime.now().isoformat()}\n"
            f"# ---- child stdout+stderr (interleaved) ----\n"
        ).encode()
        log_fh.write(header)

    t0 = time.monotonic()
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=HERE,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            # New session so SIGKILL on timeout reaches grandchild
            # HIP / hipcc helpers too.
            start_new_session=True,
        )
    except OSError as e:
        if log_fh is not None:
            log_fh.write(f"spawn failed: {e}\n".encode())
            log_fh.close()
        return RunResult(
            script=script, mode=mode.name,
            verdict="SPAWN_FAIL",
            detail=f"could not exec child: {e}",
            exit_code=None, signal_name=None,
            elapsed_s=0.0, stderr_tail="",
        )

    # Child is alive.  Register before touching pipes so a signal
    # arriving mid-setup still finds this pgid in the registry and
    # can SIGTERM it via ``_kill_live_children``.  The matching
    # ``_unregister_child`` is just before the final ``return``; the
    # intervening code is exception-free by inspection.  If it ever
    # becomes otherwise (e.g. KeyboardInterrupt injected by our
    # shutdown handler), leaving a stale entry behind is safe: the
    # handler already SIGTERM'd the pgid, any follow-up ``killpg``
    # swallows ``ProcessLookupError``, and ``main`` SIGKILLs the
    # whole registry on exit as belt-and-braces.
    _register_child(proc.pid)

    # Pumps.  One thread per stream so a flood on one side (e.g.
    # hipcc emitting MB of compile output on stdout) can't starve
    # the other and mask a critical stderr line.  Each pump:
    #   * drains its stream into an in-memory bytearray (stderr
    #     only — the main-thread RunResult needs a tail of it),
    #   * writes every chunk immediately to the log file (if any),
    #   * writes stderr chunks immediately to sys.stderr.buffer
    #     when tee_stderr is on — critically, with the same prefix
    #     the per-run header announces, so multi-script runs can
    #     still be told apart in the merged live stream.
    #
    # Two subtle things this code has to get right:
    #
    #   1. Line-accurate prefixing across chunk boundaries.
    #      ``stream.read(4096)`` returns whatever the kernel has
    #      buffered, which has zero relationship to '\n'
    #      boundaries.  A naive ``chunk.splitlines()`` would
    #      split child line ``"abcdef\n"`` arriving in chunks
    #      ``b"abc"`` and ``b"def\n"`` into two prefixed lines
    #      ``[pfx] abc`` and ``[pfx] def`` — a correctness bug
    #      for any log consumer that searches by line.  We keep
    #      a per-pump ``pending`` bytearray that holds the
    #      trailing partial line across chunks, and only emit
    #      complete (newline-terminated) lines.  On EOF we flush
    #      any remaining partial as one last prefixed line.
    #
    #   2. Narrow exception handling.  The ``except Exception:
    #      pass`` idiom is a rule violation (we're explicitly
    #      asked never to silently swallow errors).  For tee we
    #      catch ``BrokenPipeError`` specifically — that's the
    #      only failure mode a correctly-sized parent stderr
    #      should ever produce, and it means the downstream
    #      pipe reader (``| head``, ``| tee``) closed.  On that
    #      signal we print a one-time notice and disable tee;
    #      anything else propagates.  For log-file writes we
    #      catch OSError, print it (once), close the handle,
    #      and NULL it out so we don't keep retrying against a
    #      broken FD.
    #
    # Memory note (same behavior as pre-refactor
    # ``proc.communicate()``): ``stderr_buf`` is unbounded.  A
    # child that emits multi-GB to stderr would exhaust the
    # runner's RAM.  In practice AITER op-tests emit a few
    # hundred KB max.  The log file (when enabled) writes
    # straight through and does not keep a memory copy.
    stderr_buf = bytearray()
    stderr_lock = threading.Lock()
    tee_prefix = (
        f"[{_sanitize_log_component(script_label)}::{mode.name}] "
    ).encode() if tee_stderr else b""

    # ``log_state`` and ``tee_state`` are mutable cells driven by
    # the pump threads — they let us disable a broken sink
    # exactly once across both pumps without hoisting the logic
    # out of the closure.  ``nonlocal`` doesn't apply because
    # ``log_fh`` is bound at outer-function scope; we use a list
    # cell instead of ``nonlocal`` for clarity.
    log_state = {"fh": log_fh, "disabled": False}
    tee_state = {"enabled": tee_stderr, "warned": False}
    sink_lock = threading.Lock()

    def _disable_log(reason: str) -> None:
        with sink_lock:
            if log_state["disabled"]:
                return
            log_state["disabled"] = True
            fh = log_state["fh"]
            log_state["fh"] = None
        sys.stderr.write(
            f"[aiter_corpus_runner] log-file write disabled "
            f"after error: {reason}\n"
        )
        sys.stderr.flush()
        if fh is not None:
            try:
                fh.close()
            except OSError:
                pass

    def _disable_tee(reason: str) -> None:
        with sink_lock:
            if not tee_state["enabled"]:
                return
            tee_state["enabled"] = False
            already = tee_state["warned"]
            tee_state["warned"] = True
        if not already:
            # Can't print this to sys.stderr — that's what just
            # broke.  Write direct to fd 2 via os.write; if
            # even that fails we let it raise (it won't in
            # practice because BrokenPipeError is a downstream
            # issue, fd 2 itself is still open).
            try:
                os.write(2, (
                    f"[aiter_corpus_runner] tee-stderr disabled "
                    f"for rest of run: {reason}\n"
                ).encode())
            except OSError:
                pass

    def _pump(stream, is_stderr: bool) -> None:
        pending = bytearray()  # partial trailing line across chunks
        while True:
            chunk = stream.read(4096)
            if not chunk:
                break
            if log_state["fh"] is not None:
                try:
                    log_state["fh"].write(chunk)
                except OSError as e:
                    _disable_log(str(e))
            if is_stderr:
                with stderr_lock:
                    stderr_buf.extend(chunk)
                if tee_state["enabled"]:
                    pending.extend(chunk)
                    nl = pending.rfind(b"\n")
                    if nl >= 0:
                        complete = bytes(pending[:nl + 1])
                        del pending[:nl + 1]
                        prefixed = b"".join(
                            tee_prefix + line + b"\n"
                            for line in complete.splitlines()
                        )
                        try:
                            sys.stderr.buffer.write(prefixed)
                            sys.stderr.buffer.flush()
                        except BrokenPipeError as e:
                            _disable_tee(str(e))
        # EOF — flush any trailing partial line so the user
        # doesn't lose the child's last line just because it
        # lacked a terminating newline.
        if is_stderr and tee_state["enabled"] and pending:
            try:
                sys.stderr.buffer.write(tee_prefix + bytes(pending) + b"\n")
                sys.stderr.buffer.flush()
            except BrokenPipeError as e:
                _disable_tee(str(e))

    t_err = threading.Thread(
        target=_pump, args=(proc.stderr, True), daemon=True,
    )
    t_out = threading.Thread(
        target=_pump, args=(proc.stdout, False), daemon=True,
    )
    t_err.start()
    t_out.start()

    timed_out = False
    try:
        proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass

    # Drain the pumps.  Short join timeout because by this point
    # the child is either exited (EOF on the pipes, pumps return)
    # or SIGKILLed (same).
    t_err.join(timeout=10)
    t_out.join(timeout=10)

    elapsed = time.monotonic() - t0

    with stderr_lock:
        stderr_bytes = bytes(stderr_buf)
    stderr = stderr_bytes.decode("utf-8", errors="replace")

    # Footer goes to whichever log handle the pumps left behind.
    # ``_disable_log`` may have already closed it and nulled the
    # cell on a write error; in that case there's nothing to do.
    # We take ``sink_lock`` so we can't race a pump thread that
    # is still trying to disable-and-close the same FD.
    with sink_lock:
        final_fh = log_state["fh"]
        log_state["fh"] = None
    if final_fh is not None:
        footer = (
            f"\n# ---- child terminated ----\n"
            f"# returncode: {proc.returncode!r}\n"
            f"# elapsed_s:  {elapsed:.3f}\n"
            f"# timed_out:  {timed_out}\n"
        ).encode()
        try:
            final_fh.write(footer)
        except OSError as e:
            sys.stderr.write(
                f"[aiter_corpus_runner] log-file footer write "
                f"failed: {e}\n"
            )
        finally:
            try:
                final_fh.close()
            except OSError:
                pass

    if timed_out:
        verdict, sig_name = "HANG", None
        detail = f"exceeded timeout of {timeout_s:.0f}s; SIGKILLed"
    else:
        verdict, sig_name = _classify_exit(proc.returncode)
        if verdict == "CRASH":
            detail = f"killed by {sig_name}"
        elif verdict == "FAIL":
            detail = f"exit {proc.returncode}"
        else:
            detail = "ok"
    if log_path is not None:
        detail = f"{detail}; log={log_path}"

    # Child is done — either exited, crashed, or was SIGKILLed on
    # timeout — and has been reaped by ``proc.wait``.  Drop it from
    # the registry so a later shutdown signal doesn't try to kill
    # a dead / recycled pgid.
    _unregister_child(proc.pid)

    return RunResult(
        script=script,
        mode=mode.name,
        verdict=verdict,
        detail=detail,
        exit_code=proc.returncode if not timed_out else None,
        signal_name=sig_name,
        elapsed_s=elapsed,
        stderr_tail=_tail(stderr),
    )


def _reap_orphan_build_locks(jit_cache: str) -> Tuple[List[str], List[str]]:
    """Remove FileBaton lock files left behind by crashed AITER builds.

    AITER's ``@compile_ops`` decorator serialises concurrent builds
    of the same module via ``aiter/jit/utils/file_baton.py``.  The
    baton creates an empty ``build/lock_<module_md5>`` file with
    ``O_CREAT|O_EXCL`` on acquire and unlinks it on release.
    Release runs in a ``finally`` block, so any exit path that
    skips finally blocks — SIGKILL, OOM-killer, kernel panic,
    Ctrl-C during certain syscalls — leaves the marker file in
    place.  Every subsequent concurrent build of that same module
    then spins forever in ``baton.wait()`` waiting for a release
    that will never come.

    We detect orphans by cross-referencing each lock file's inode
    against every open FD on the host via ``/proc/<pid>/fd``.  A
    live baton owner still holds the FD (``os.open`` + stash in
    ``self.fd``); a crashed owner's FD was closed by the kernel
    on process exit but the unlink never happened.

    Scope of the scan: we only inspect same-UID Python processes.
    AITER's ``mp_lock`` is only reached from ``@compile_ops`` in a
    Python import of AITER, so the FileBaton owner, if it exists,
    must be a Python interpreter (``/proc/<pid>/comm`` starts with
    ``"python"``) running as the current user.  Other processes on
    the host are out of scope:

    * other users' processes — cannot have opened an AITER-path
      baton, and their ``/proc/<pid>/fd`` is unreadable anyway;
    * same-UID non-Python processes — notably setuid-exec'd ssh
      daemons whose fd table the kernel hides from us even though
      the euid matches.  These are not Python, did not import
      AITER, and cannot hold an AITER baton.

    If a same-UID *Python* process's ``fd`` dir is un-scannable
    (should not happen on a stock kernel — indicates ptrace
    hardening or a setuid python wrapper) we refuse to reap.
    Deleting a live builder's lock would allow a second concurrent
    hipcc invocation to produce a corrupted ``.so``.  Per project
    rule: never silently fall back.

    Returns ``(reaped_paths, inspection_errors)``.  Callers should
    print both — the errors list surfaces inspection gaps that
    prevented cleanup, so a user running on a shared box knows
    why orphan locks weren't touched.
    """
    build_dir = os.path.join(jit_cache, "build")
    if not os.path.isdir(build_dir):
        return [], []

    try:
        lock_names = [n for n in os.listdir(build_dir) if n.startswith("lock_")]
    except OSError as e:
        return [], [f"listdir({build_dir}): {e}"]
    if not lock_names:
        return [], []

    # Resolve each candidate lock to its (st_dev, st_ino) so we can
    # match on inode rather than path — robust against someone
    # renaming or moving the file under us mid-scan.
    targets: Dict[Tuple[int, int], str] = {}
    for name in lock_names:
        p = os.path.join(build_dir, name)
        try:
            st = os.stat(p)
        except FileNotFoundError:
            continue
        targets[(st.st_dev, st.st_ino)] = p
    if not targets:
        return [], []

    # Walk every same-UID Python process's fd table.  Non-Python
    # processes cannot have imported AITER; different-UID processes
    # cannot have opened a file in our home dir through AITER's
    # code path.  A process that exited mid-scan is fine — its FDs
    # died with it.  Permission-denied on the fd dir of a same-UID
    # *python* process is a genuine inspection gap → fail closed.
    my_uid = os.getuid()
    errors: List[str] = []
    try:
        proc_entries = [e.path for e in os.scandir("/proc") if e.name.isdigit()]
    except OSError as e:
        return [], [f"scandir(/proc): {e}"]

    live_inodes: Set[Tuple[int, int]] = set()
    for ppath in proc_entries:
        try:
            proc_uid = os.stat(ppath).st_uid
        except FileNotFoundError:
            continue  # process exited mid-scan
        except PermissionError:
            # /proc/<pid> itself should be stat-able on a stock
            # kernel; PermissionError here indicates hidepid=2
            # which already hides the process from us entirely.
            # Conservative: ignore (any baton it holds has also
            # been hidden from us and there is nothing actionable).
            continue
        if proc_uid != my_uid:
            continue
        try:
            with open(os.path.join(ppath, "comm"), "r") as f:
                comm = f.read().strip()
        except (FileNotFoundError, PermissionError, OSError):
            continue  # exited or genuinely unreadable — treat as not-python
        if not comm.startswith("python"):
            continue  # cannot have imported AITER's FileBaton
        fd_dir = os.path.join(ppath, "fd")
        try:
            fd_names = os.listdir(fd_dir)
        except FileNotFoundError:
            continue  # process exited; its FDs are gone with it
        except PermissionError:
            errors.append(f"{fd_dir}: permission denied (comm={comm!r})")
            continue
        for fd in fd_names:
            try:
                st = os.stat(os.path.join(fd_dir, fd))
            except (FileNotFoundError, PermissionError):
                continue  # fd closed mid-scan, or symlink unreadable
            key = (st.st_dev, st.st_ino)
            if key in targets:
                live_inodes.add(key)

    if errors:
        # Fail closed: we can't prove these locks are orphan.
        return [], errors

    reaped: List[str] = []
    for key, path in targets.items():
        if key in live_inodes:
            continue
        try:
            os.unlink(path)
            reaped.append(path)
        except FileNotFoundError:
            continue
    return reaped, []


def _run_parallel(
    jobs_spec: List[Tuple[str, str]],
    gpu_pool: List[str],
    n_workers: int,
    args: argparse.Namespace,
    native_gfx: str,
    spoof_arch: str,
    log_dir: Optional[str],
    base_dirs: List[str],
) -> List[RunResult]:
    """Dispatch ``(script, mode)`` pairs across ``n_workers`` threads,
    each holding one GPU slot from ``gpu_pool`` at a time.

    The slot queue enforces a strict 1:1 worker-to-GPU binding: a
    worker blocks on ``gpu_queue.get()`` before spawning its child,
    and puts the GPU back after the child exits.  ``n_workers`` is
    already capped to ``len(gpu_pool)`` by ``main()`` so the queue
    starts full and no worker ever starves.

    Progress lines are printed under a ``threading.Lock`` so concurrent
    ``[run k/N] ...`` reports don't interleave mid-line.  The
    ``[script::mode]`` prefix of ``--tee-stderr`` keeps the
    per-child live output readable even when 8 streams interleave.

    Completion order is non-deterministic and intentional — the
    printed result order reflects who finishes first, which is
    often the most useful ordering during triage (a 60s HANG
    lands between two 5s PASSes).  ``_print_grid`` /
    ``_print_failures`` / ``_print_summary`` all re-sort their input
    by ``(script, mode)``, so the final report is still
    deterministic.
    """
    gpu_queue: queue.Queue = queue.Queue()
    for g in gpu_pool[:n_workers]:
        gpu_queue.put(g)

    print_lock = threading.Lock()
    n_total = len(jobs_spec)

    def _dispatch(job_idx: int, script: str, mode_name: str) -> RunResult:
        gpu = gpu_queue.get()
        try:
            mode = MODES[mode_name]
            label = _short_label(script, base_dirs)
            with print_lock:
                print(
                    f"[run {job_idx}/{n_total}] {label} :: {mode_name} "
                    f"... (gpu={gpu})",
                    file=sys.stderr, flush=True,
                )
            r = _run_one(
                script=script, mode=mode,
                aiter_root=args.aiter_root,
                triton_venv=args.triton_venv,
                libsalmon=args.libsalmon,
                libhsa=args.libhsa,
                libamdhip=args.libamdhip,
                libarch_spoof=args.libarch_spoof,
                spoof_arch=spoof_arch,
                native_gfx=native_gfx,
                strict_tol=args.strict_tolerance,
                perftest_iters=args.perftest_iters,
                jit_cache=args.jit_cache,
                timeout_s=args.timeout,
                visible_devices=gpu,
                script_args=args.script_arg,
                rng_seed=args.rng_seed,
                tee_stderr=args.tee_stderr,
                log_dir=log_dir,
                script_label=label,
            )
            with print_lock:
                print(
                    f"[verdict {job_idx}/{n_total}] {label} :: "
                    f"{mode_name} ... {r.verdict} "
                    f"({r.detail}, {r.elapsed_s:.1f}s, gpu={gpu})",
                    file=sys.stderr, flush=True,
                )
            return r
        finally:
            gpu_queue.put(gpu)

    results: List[RunResult] = []
    # Manual executor lifetime (not ``with``) so we can pass
    # ``cancel_futures=True`` on shutdown.  Critical on Ctrl-C: the
    # default ``__exit__`` calls ``shutdown(wait=True)`` which does
    # *not* cancel pending futures, so worker threads would keep
    # popping the executor queue and spawning brand-new children
    # even after our SIGINT handler has SIGTERM'd the in-flight
    # ones — exactly the orphan-process leak this registry exists
    # to prevent.
    pool = concurrent.futures.ThreadPoolExecutor(
        max_workers=n_workers,
        thread_name_prefix="aiter-worker",
    )
    futures = [
        pool.submit(_dispatch, i, s, m)
        for i, (s, m) in enumerate(jobs_spec, 1)
    ]
    try:
        # ``as_completed`` drains futures in finish-order; exceptions
        # (which would be a bug in ``_dispatch`` itself, not a child
        # failure — child failures surface as RunResult.verdict=CRASH)
        # propagate out of ``.result()`` and abort the sweep.
        for f in concurrent.futures.as_completed(futures):
            results.append(f.result())
    except KeyboardInterrupt:
        # Signal handler already SIGTERM'd every live child.  Cancel
        # the not-yet-started futures so workers stop dispatching new
        # work, then re-raise so ``main``'s outer handler can print
        # the interrupted banner and exit 130.
        for f in futures:
            f.cancel()
        raise
    finally:
        # ``wait=False`` lets us unwind immediately.  Python will
        # still wait for the worker *threads* to return before the
        # interpreter exits, but that's bounded: in-flight children
        # have been SIGTERM'd or completed, so each worker's
        # ``_run_one`` returns within a few seconds.
        pool.shutdown(wait=False, cancel_futures=True)
    return results


# -- discovery + setup ------------------------------------------------------


def _discover_scripts(
    aiter_root: str,
    explicit_scripts: List[str],
    explicit_dirs: List[str],
    use_all: bool,
) -> tuple[List[str], List[str]]:
    """Resolve the list of scripts to run plus the base dirs to use
    when computing short labels for the per-script grid.  Explicit
    ``--script`` wins; then ``--scripts-dir`` (repeatable); then a
    curated subset of ``aiter_root/op_tests/`` unless ``--all`` is
    set."""
    paths: List[str] = []
    seen: set[str] = set()

    def _add(p: str) -> None:
        ap = os.path.abspath(p)
        if ap in seen:
            return
        seen.add(ap)
        paths.append(ap)

    for s in explicit_scripts:
        if not os.path.isfile(s):
            raise RuntimeError(f"--script {s!r}: not a file")
        _add(s)

    base_dirs = list(explicit_dirs)

    op_tests_dir = os.path.join(aiter_root, "op_tests")
    if not paths and not base_dirs:
        if not os.path.isdir(op_tests_dir):
            raise RuntimeError(
                f"AITER op_tests dir not found at {op_tests_dir!r}; "
                f"run with --setup to clone, or pass --aiter-root"
            )
        base_dirs = [op_tests_dir]

    for d in base_dirs:
        if not os.path.isdir(d):
            raise RuntimeError(f"--scripts-dir {d!r}: not a directory")
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".py"):
                continue
            if fn.startswith("_") or fn == "__init__.py":
                continue
            if not fn.startswith("test_"):
                continue
            full = os.path.join(d, fn)
            if not paths and not use_all:
                # Only filter the auto-discovered set against the
                # curated subset; explicit --script paths are
                # respected as-is above.
                if fn not in CURATED_SUBSET:
                    continue
            _add(full)

    if not paths:
        raise RuntimeError(
            f"no scripts to run (use --all to include the full op_tests "
            f"corpus or --script PATH for a one-off)"
        )

    return paths, base_dirs


def _do_setup(aiter_root: str,
              triton_venv: str,
              repo: str) -> None:
    """One-shot: git clone AITER + pip install its Python deps into
    --triton-venv.  We deliberately do *not* run ``setup.py develop``
    because (a) it'd front-load the multi-minute C++ compile that
    ``compile_ops`` does lazily anyway, and (b) we put AITER on
    ``PYTHONPATH`` directly so the source-tree layout is enough.

    Idempotent: skips clone if the dir already has an aiter package,
    and skips pip install if the deps already import.
    """
    print(f"[setup] aiter_root = {aiter_root}", file=sys.stderr, flush=True)

    if not os.path.exists(os.path.join(aiter_root, "aiter", "__init__.py")):
        if os.path.exists(aiter_root) and os.listdir(aiter_root):
            raise RuntimeError(
                f"--aiter-root {aiter_root!r} exists, is non-empty, but "
                f"does not look like an AITER checkout (no aiter/__init__.py). "
                f"Remove or pass a different --aiter-root."
            )
        print(f"[setup] git clone --recursive {repo} {aiter_root}",
              file=sys.stderr, flush=True)
        subprocess.run(
            ["git", "clone", "--recursive", repo, aiter_root],
            check=True,
        )
    else:
        print(f"[setup] {aiter_root} already has an AITER checkout, "
              f"skipping clone", file=sys.stderr, flush=True)

    pip = os.path.join(triton_venv, "bin", "pip")
    if not os.path.exists(pip):
        raise RuntimeError(
            f"venv pip not found at {pip}; run the triton_corpus_runner "
            f"setup first (see its README) so the shared rocm-7 venv "
            f"exists, or pass --triton-venv to a venv you maintain."
        )

    requirements = os.path.join(aiter_root, "requirements.txt")
    if not os.path.isfile(requirements):
        raise RuntimeError(
            f"AITER requirements.txt missing at {requirements!r}"
        )
    print(f"[setup] {pip} install -r {requirements}",
          file=sys.stderr, flush=True)
    subprocess.run([pip, "install", "-r", requirements], check=True)

    print("[setup] done; AITER is on PYTHONPATH and its Python deps "
          "are installed.  C++ ops will compile lazily on first use "
          "via aiter's @compile_ops decorator (cached in --jit-cache).",
          file=sys.stderr, flush=True)


def _detect_aiter_root_problem(aiter_root: str) -> Optional[str]:
    """Return a human-readable message if ``aiter_root`` looks
    unusable; ``None`` if it looks like a valid AITER checkout."""
    if not os.path.isdir(aiter_root):
        return (f"--aiter-root {aiter_root!r} does not exist; run "
                f"with --setup to clone, or point it at an existing "
                f"AITER checkout.")
    if not os.path.isfile(os.path.join(aiter_root, "aiter", "__init__.py")):
        return (f"--aiter-root {aiter_root!r} does not contain an "
                f"AITER package (no aiter/__init__.py).  Pass --setup "
                f"to clone, or point at the right directory.")
    if not os.path.isdir(os.path.join(aiter_root, "op_tests")):
        return (f"--aiter-root {aiter_root!r} contains aiter but no "
                f"op_tests/ — looks like a partial install.")
    return None


# -- reporting --------------------------------------------------------------


_VERDICT_ORDER = ["PASS", "FAIL", "CRASH", "HANG", "SPAWN_FAIL"]


def _short_label(script: str, base_dirs: List[str]) -> str:
    ap = os.path.abspath(script)
    for d in base_dirs:
        d = os.path.abspath(d)
        if ap.startswith(d + os.sep):
            return os.path.relpath(ap, d)
    return os.path.basename(ap)


def _print_grid(results: List[RunResult], mode_names: List[str],
                base_dirs: List[str]) -> None:
    by_script: Dict[str, Dict[str, RunResult]] = {}
    for r in results:
        by_script.setdefault(r.script, {})[r.mode] = r

    label_w = max(
        (len(_short_label(s, base_dirs)) for s in by_script),
        default=10,
    )
    label_w = max(label_w, len("script"))
    cell_w = max(
        (len(f"{r.verdict} ({r.detail[:18]})") for r in results),
        default=10,
    )
    cell_w = min(max(cell_w, max(len(m) for m in mode_names)), 32)

    print()
    print("=== Per-script grid ===")
    header = "  " + "script".ljust(label_w)
    for m in mode_names:
        header += "   " + m.ljust(cell_w)
    print(header)
    print("  " + "-" * label_w + ("   " + "-" * cell_w) * len(mode_names))
    for script in sorted(by_script):
        row = "  " + _short_label(script, base_dirs).ljust(label_w)
        for m in mode_names:
            r = by_script[script].get(m)
            if r is None:
                cell = "—"
            elif r.verdict == "PASS":
                cell = "PASS"
            else:
                cell = f"{r.verdict} ({r.detail})"
            row += "   " + cell[:cell_w].ljust(cell_w)
        print(row)


def _print_failures(results: List[RunResult], mode_names: List[str],
                    base_dirs: List[str]) -> None:
    by_mode: Dict[str, List[RunResult]] = {m: [] for m in mode_names}
    for r in results:
        if r.verdict != "PASS":
            by_mode.setdefault(r.mode, []).append(r)

    if not any(by_mode.values()):
        return

    print()
    print("=== Failures ===")
    for m in mode_names:
        runs = by_mode.get(m, [])
        if not runs:
            continue
        print(f"\n{m}")
        for r in runs:
            print(
                f"  {_short_label(r.script, base_dirs)}"
                f"  {r.verdict}  ({r.detail}, {r.elapsed_s:.1f}s)"
            )
            if r.stderr_tail.strip():
                for line in r.stderr_tail.rstrip().splitlines()[-12:]:
                    print(f"      | {line}")


def _print_summary(results: List[RunResult], mode_names: List[str]) -> None:
    counts: Dict[str, Dict[str, int]] = {
        v: {m: 0 for m in mode_names} for v in _VERDICT_ORDER
    }
    for r in results:
        counts[r.verdict][r.mode] += 1

    n_scripts = len({r.script for r in results})
    print()
    print(f"=== Summary ({n_scripts} scripts × {len(mode_names)} modes) ===")
    label_w = max(len(v) for v in _VERDICT_ORDER)
    header = "  " + " " * label_w
    for m in mode_names:
        header += "   " + m.rjust(8)
    print(header)
    for v in _VERDICT_ORDER:
        row = "  " + v.ljust(label_w)
        for m in mode_names:
            row += "   " + str(counts[v][m]).rjust(8)
        print(row)

    by_script: Dict[str, Dict[str, str]] = {}
    for r in results:
        by_script.setdefault(r.script, {})[r.mode] = r.verdict

    if "native" in mode_names:
        native_ok = [s for s, m in by_script.items() if m.get("native") == "PASS"]
        for transp in ("legacy", "salmon"):
            if transp not in mode_names:
                continue
            broke = [s for s in native_ok if by_script[s].get(transp) != "PASS"]
            print(
                f"\n  {transp}: {len(native_ok) - len(broke)}/{len(native_ok)} "
                f"native-PASS scripts also pass under {transp}"
            )
            if broke:
                for s in sorted(broke):
                    print(f"    - {os.path.basename(s)} "
                          f"({by_script[s].get(transp, '—')})")


# -- CLI --------------------------------------------------------------------


def _parse_modes(arg: str) -> List[str]:
    if arg.strip().lower() in ("all", ""):
        return ["native", "legacy", "salmon"]
    out = []
    for tok in arg.split(","):
        tok = tok.strip()
        if tok not in MODES:
            raise argparse.ArgumentTypeError(
                f"unknown mode {tok!r}; valid: {sorted(MODES)}"
            )
        out.append(tok)
    if not out:
        raise argparse.ArgumentTypeError("at least one mode required")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--aiter-root", default=DEFAULT_AITER_ROOT,
        help=f"path to an AITER checkout (default {DEFAULT_AITER_ROOT}); "
             f"--setup will clone here on first run",
    )
    ap.add_argument(
        "--aiter-repo", default=DEFAULT_AITER_REPO,
        help=f"git URL for --setup (default {DEFAULT_AITER_REPO})",
    )
    ap.add_argument(
        "--setup", action="store_true",
        help="git clone AITER into --aiter-root and pip-install its "
             "Python deps into --triton-venv, then exit.  Idempotent.",
    )
    ap.add_argument(
        "--script", action="append", default=[],
        help="explicit op-test script to run (repeatable; bypasses "
             "the curated subset filter)",
    )
    ap.add_argument(
        "--scripts-dir", action="append", default=[],
        help="directory of test_*.py scripts to run, non-recursive "
             "(repeatable)",
    )
    ap.add_argument(
        "--all", action="store_true",
        help="when scripts are auto-discovered from op_tests/, "
             "include every test_*.py instead of just the curated "
             "single-GPU subset",
    )
    ap.add_argument(
        "--modes", default="all", type=_parse_modes,
        help="comma-separated subset of {native,legacy,salmon}; "
             "default 'all'",
    )
    ap.add_argument(
        "--source-gfx", default=DEFAULT_SOURCE_GFX,
        help=f"AITER ISA to force in legacy/salmon modes "
             f"(default {DEFAULT_SOURCE_GFX!r}); native always "
             f"auto-detects via rocminfo",
    )
    ap.add_argument(
        "--native-gfx", default="",
        help="override the auto-detected native gfx arch (the one "
             "rocminfo reports).  Empty default means call rocminfo "
             "once at runner startup and cache it.  Pass an explicit "
             "value only when rocminfo is unavailable or you're "
             "building on a machine without the target GPU.",
    )
    ap.add_argument(
        "--strict-tolerance", type=float, default=0.01,
        help="checkAllclose mismatch percent above which the patched "
             "wrapper raises (default 0.01, matching AITER's own "
             "warning-vs-failed cutoff)",
    )
    ap.add_argument(
        "--perftest-iters", type=int, default=1,
        help="iteration budget surfaced to the bootstrap (default 1); "
             "the patched perftest decorator always calls the kernel "
             "exactly once and returns avg=0.0, so this knob is "
             "informational",
    )
    ap.add_argument(
        "--timeout", type=float, default=600.0,
        help="per-child wall-clock timeout in seconds (default 600 — "
             "first run includes hipcc compile time for AITER's CK "
             "side, subsequent runs are much faster from --jit-cache)",
    )
    ap.add_argument(
        "--triton-venv", default=DEFAULT_TRITON_VENV,
        help=f"venv root with torch+rocm7 (default {DEFAULT_TRITON_VENV} "
             f"— shared with triton_corpus_runner)",
    )
    ap.add_argument(
        "--libsalmon", default=DEFAULT_LIBSALMON,
        help=f"path to libsalmon_intercept.so (default {DEFAULT_LIBSALMON})",
    )
    ap.add_argument(
        "--libhsa", default=DEFAULT_LIBHSA,
        help=f"path to the Salmon-enabled libhsa-runtime64.so.1 "
             f"(default {DEFAULT_LIBHSA})",
    )
    ap.add_argument(
        "--libamdhip", default=DEFAULT_LIBAMDHIP,
        help="optional system libamdhip64.so.* to LD_PRELOAD ahead of "
             "the venv's bundled HIP.  Empty (default) means trust the "
             "venv's HIP — correct for the rocm-7 nightly torch wheel.",
    )
    ap.add_argument(
        "--libarch-spoof", default=DEFAULT_LIBARCH_SPOOF,
        help=f"path to libaiter_arch_spoof.so (default "
             f"{DEFAULT_LIBARCH_SPOOF}); built by 'make' in this "
             f"directory.  Loaded only in legacy/salmon modes.",
    )
    ap.add_argument(
        "--spoof-arch", default="",
        help="value rewritten into hipGetDeviceProperties().gcnArchName "
             "for legacy/salmon modes.  Empty default ('') means use "
             "--source-gfx, which is what you want unless you're "
             "deliberately probing for a struct-layout regression in "
             "the HIP runtime.",
    )
    ap.add_argument(
        "--jit-cache", default=DEFAULT_JIT_CACHE,
        help=f"persistent JIT-build cache for AITER's @compile_ops "
             f"(default {DEFAULT_JIT_CACHE})",
    )
    ap.add_argument(
        "--gpu", default=None,
        help="value for HIP_VISIBLE_DEVICES in --jobs 1 (serial) mode "
             "— single index like '0', or comma list.  Default: '0' "
             "if HIP_VISIBLE_DEVICES is not already set; pass '' to "
             "clear it.  Ignored when --jobs > 1; use --gpus instead.",
    )
    ap.add_argument(
        "--jobs", type=int, default=1, metavar="N",
        help="run up to N (script, mode) jobs concurrently, one per "
             "GPU in the --gpus pool.  Default 1 (strictly serial).  "
             "Pool size is capped to len(--gpus).  AITER's own "
             "FileBaton-based mp_lock serialises concurrent "
             "@compile_ops builds of the same module, so a cold "
             "--jit-cache parallel run is correct — no runner-side "
             "warmup pass is needed.  Use this on a multi-GPU box "
             "to cut full-corpus sweep time from hours to minutes.",
    )
    ap.add_argument(
        "--gpus", default="",
        help="GPU pool for --jobs > 1 (comma-separated indices with "
             "optional ranges, e.g. '0,1,2,3' or '0-7' or "
             "'0,2-5,7').  Empty default falls back to the parent "
             "shell's HIP_VISIBLE_DEVICES env var.  If neither is "
             "set and --jobs > 1, the runner errors before doing "
             "any work.  No auto-discovery: we honour whatever the "
             "shell has declared visible so shared-box usage is safe.",
    )
    ap.add_argument(
        "--rng-seed", default="0",
        help="integer seed fed to torch / numpy / random in every "
             "child, *before* the op-test starts.  Default '0'.  "
             "Pass the empty string '' to leave RNG unseeded (older "
             "behaviour).  Seeding matters because AITER's op_tests "
             "build their random inputs with ``torch.randn(...)`` at "
             "top level and then compare at ``atol=1e-2 rtol=1e-2``; "
             "without a fixed seed, mode-dependent numerical noise in "
             "non-asm kernels can make the native-vs-salmon column "
             "look like a transpilation regression when it's really "
             "just RNG jitter.  The runner's whole cross-mode "
             "comparison presumes identical inputs, so this is on by "
             "default.",
    )
    ap.add_argument(
        "--script-arg", action="append", default=[],
        help="extra argv item to forward verbatim to every op-test "
             "script (repeatable).  Useful for shrinking AITER's "
             "default 17-token x N-config sweeps to a single "
             "configuration when verifying the asm path engages.  "
             "Example: --script-arg=-t --script-arg=8 --script-arg=-e "
             "--script-arg=128 cuts test_moeTopkSoftmax to one cell.",
    )
    ap.add_argument(
        "--json", default="",
        help="also write a machine-readable JSON record here",
    )
    ap.add_argument(
        "--tee-stderr", action="store_true",
        help="stream every child's stderr to this runner's stderr "
             "in real time (prefixed with [script::mode]).  Without "
             "this flag the runner waits until each child exits "
             "before surfacing any output, which makes it impossible "
             "to see *where* a hanging run stopped emitting.  On by "
             "default in interactive use only when you're triaging; "
             "add it for every run where you want live progress.",
    )
    ap.add_argument(
        "--log-dir", default="",
        help="write every child's complete unbuffered stdout+stderr "
             "to a per-run log file under this directory "
             "(<script>__<mode>__<timestamp>.log).  Empty default "
             "disables the feature.  Unlike the bounded stderr tail "
             "in the final report, these logs preserve the *entire* "
             "transcript — ideal for diffing a hung salmon run "
             "against a clean legacy one and spotting the divergent "
             "LoadCodeObject call.",
    )
    ap.add_argument(
        "--print-command", action="store_true",
        help="do not spawn any child; for every (script, mode) in "
             "the run matrix, print the exact env+argv the runner "
             "would have used, as a paste-ready bash block, then "
             "exit 0.  Useful for wrapping the child manually under "
             "gdb / rocgdb / strace / AMD_LOG_LEVEL=5 without having "
             "to reverse-engineer LD_PRELOAD ordering and the full "
             "AITER_CORPUS_* env set.",
    )
    args = ap.parse_args()

    if args.setup:
        try:
            _do_setup(args.aiter_root, args.triton_venv, args.aiter_repo)
        except subprocess.CalledProcessError as e:
            print(f"setup failed: {e}", file=sys.stderr)
            return 2
        except RuntimeError as e:
            print(f"setup failed: {e}", file=sys.stderr)
            return 2
        return 0

    aiter_problem = _detect_aiter_root_problem(args.aiter_root)
    if aiter_problem is not None:
        print(aiter_problem, file=sys.stderr)
        return 2

    try:
        scripts, base_dirs = _discover_scripts(
            args.aiter_root, args.script, args.scripts_dir, args.all,
        )
    except RuntimeError as e:
        print(f"discovery failed: {e}", file=sys.stderr)
        return 2

    spoof_arch = args.spoof_arch or args.source_gfx
    native_gfx = args.native_gfx.strip()
    if not native_gfx:
        try:
            native_gfx = _detect_native_gfx()
        except RuntimeError as e:
            print(f"native gfx detection failed: {e}", file=sys.stderr)
            return 2
    log_dir: Optional[str] = args.log_dir.strip() or None
    if log_dir is not None:
        log_dir = os.path.abspath(log_dir)

    if args.jobs < 1:
        print(
            f"--jobs must be >= 1 (got {args.jobs})",
            file=sys.stderr,
        )
        return 2
    try:
        gpu_pool = _resolve_gpu_pool(args.gpus, args.jobs)
    except RuntimeError as e:
        print(f"GPU pool resolution failed: {e}", file=sys.stderr)
        return 2
    jobs = args.jobs
    if jobs > len(gpu_pool):
        print(
            f"[aiter_corpus_runner] --jobs {jobs} exceeds GPU pool "
            f"size {len(gpu_pool)} ({','.join(gpu_pool)}); capping "
            f"to {len(gpu_pool)} (one worker per GPU, strictly).",
            file=sys.stderr,
        )
        jobs = len(gpu_pool)

    # Startup hygiene: reap any orphan FileBaton locks left in
    # ``_jit_cache/build/`` by previously-SIGKILLed AITER builds.
    # A lingering lock blocks any future concurrent build of the
    # same module in ``baton.wait()`` forever — see the comment on
    # ``_reap_orphan_build_locks`` for the why.  We do this
    # unconditionally (serial and parallel) because the bug also
    # bites sequential second runs of the same script.  Skipped
    # only when ``--print-command`` is set (dry-run, no cache work).
    if not args.print_command:
        reaped, inspection_errors = _reap_orphan_build_locks(args.jit_cache)
        for p in reaped:
            print(
                f"[jit-cache] reaped orphan FileBaton lock: {p}",
                file=sys.stderr,
            )
        if inspection_errors:
            # Surface every gap — we will not reap any lock when
            # inspection is incomplete, and the user needs to know
            # why.  This is deliberately not a warning-and-continue
            # pattern: it's an explicit refusal to silently fall back.
            print(
                "[jit-cache] orphan-lock reaper: refusing to reap any "
                "lock because /proc inspection is incomplete:",
                file=sys.stderr,
            )
            for err in inspection_errors:
                print(f"[jit-cache]   {err}", file=sys.stderr)

    # --print-command: no spawning, no work, just emit the exact
    # invocation the runner *would* use for each (script, mode)
    # pair in the current matrix.  We run the native-gfx +
    # spoof_arch resolution above so the printed block is a true
    # byte-identical reproduction of what the runner would run.
    if args.print_command:
        # --print-command is a "dry-run, print and exit" mode.
        # Silently ignoring --log-dir / --tee-stderr would be
        # confusing — the user asked for those sinks and none
        # will ever exist.  Warn loudly instead of swallowing.
        ignored = []
        if log_dir is not None:
            ignored.append("--log-dir")
        if args.tee_stderr:
            ignored.append("--tee-stderr")
        if ignored:
            print(
                f"[aiter_corpus_runner] note: {', '.join(ignored)} "
                f"is ignored under --print-command (no child is "
                f"actually spawned).",
                file=sys.stderr,
            )
        for s in scripts:
            for m in args.modes:
                mode = MODES[m]
                cmd, env = _build_run_spec(
                    script=s, mode=mode,
                    aiter_root=args.aiter_root,
                    triton_venv=args.triton_venv,
                    libsalmon=args.libsalmon,
                    libhsa=args.libhsa,
                    libamdhip=args.libamdhip,
                    libarch_spoof=args.libarch_spoof,
                    spoof_arch=spoof_arch,
                    native_gfx=native_gfx,
                    strict_tol=args.strict_tolerance,
                    perftest_iters=args.perftest_iters,
                    jit_cache=args.jit_cache,
                    visible_devices=args.gpu,
                    script_args=args.script_arg,
                    rng_seed=args.rng_seed,
                )
                # ``_build_run_spec`` has already set
                # ``AITER_CORPUS_SCRIPT`` to ``os.path.abspath(s)``;
                # don't re-assign it (dead write hides intent).
                block = _format_print_command(
                    script_label=_short_label(s, base_dirs),
                    mode_name=mode.name,
                    cmd=cmd, env=env, cwd=HERE,
                )
                print(block)
                print()
        return 0

    parallel_banner = (
        f"jobs={jobs} (serial)" if jobs == 1
        else f"jobs={jobs} × gpus=[{','.join(gpu_pool[:jobs])}]"
    )
    print(
        f"aiter corpus runner: {len(scripts)} script(s), "
        f"modes={args.modes}, native_gfx={native_gfx!r}, "
        f"source_gfx={args.source_gfx!r}, "
        f"spoof_arch={spoof_arch!r} (legacy/salmon only), "
        f"strict_tol={args.strict_tolerance}, "
        f"rng_seed={args.rng_seed!r}, "
        f"timeout={args.timeout:.0f}s, jit_cache={args.jit_cache}, "
        f"tee_stderr={args.tee_stderr}, "
        f"log_dir={log_dir or '<off>'}, "
        f"{parallel_banner}",
        file=sys.stderr,
    )
    for s in scripts:
        print(f"  - {_short_label(s, base_dirs)}", file=sys.stderr)
    print(file=sys.stderr)

    results: List[RunResult] = []

    # Full run matrix.  Concurrent @compile_ops builds of the same
    # module are serialised correctly by AITER's own FileBaton
    # (see ``aiter/jit/core.py::mp_lock`` + ``file_baton.py``), so
    # a cold ``--jit-cache`` parallel run is safe — the first
    # worker to enter the critical section builds, the others
    # spin-wait on the marker file and then dlopen the cached
    # artefact.  No runner-side warmup pass is needed.
    main_jobs: List[Tuple[str, str]] = [
        (s, m) for s in scripts for m in args.modes
    ]

    if jobs == 1:
        # Serial path — preserves the historic single-stream output
        # format for users / wrapper scripts that grep ``[run]`` /
        # ``[verdict]`` with no k/N prefix.
        for s, m in main_jobs:
            mode = MODES[m]
            print(f"[run] {_short_label(s, base_dirs)} :: {m} ... ",
                  end="", flush=True, file=sys.stderr)
            # With --tee-stderr the child's live output will appear
            # *after* this "[run] ..." line but *before* the verdict
            # line.  Flush a trailing newline now so the tee lands
            # on fresh rows instead of smearing onto the "..." —
            # readability > terseness during active triage.
            if args.tee_stderr:
                print("", file=sys.stderr, flush=True)
            r = _run_one(
                script=s, mode=mode,
                aiter_root=args.aiter_root,
                triton_venv=args.triton_venv,
                libsalmon=args.libsalmon,
                libhsa=args.libhsa,
                libamdhip=args.libamdhip,
                libarch_spoof=args.libarch_spoof,
                spoof_arch=spoof_arch,
                native_gfx=native_gfx,
                strict_tol=args.strict_tolerance,
                perftest_iters=args.perftest_iters,
                jit_cache=args.jit_cache,
                timeout_s=args.timeout,
                visible_devices=args.gpu,
                script_args=args.script_arg,
                rng_seed=args.rng_seed,
                tee_stderr=args.tee_stderr,
                log_dir=log_dir,
                script_label=_short_label(s, base_dirs),
            )
            results.append(r)
            # When tee was on, the "[run] ... " prefix is now long
            # scrolled off; re-print the script::mode so the verdict
            # line is self-contained.
            verdict_line = (
                f"[verdict] {_short_label(s, base_dirs)} :: {m} ... "
                f"{r.verdict} ({r.detail}, {r.elapsed_s:.1f}s)"
                if args.tee_stderr else
                f"{r.verdict} ({r.detail}, {r.elapsed_s:.1f}s)"
            )
            print(verdict_line, file=sys.stderr)
    else:
        parallel_results = _run_parallel(
            jobs_spec=main_jobs,
            gpu_pool=gpu_pool,
            n_workers=jobs,
            args=args, native_gfx=native_gfx, spoof_arch=spoof_arch,
            log_dir=log_dir, base_dirs=base_dirs,
        )
        results.extend(parallel_results)

    _print_grid(results, args.modes, base_dirs)
    _print_failures(results, args.modes, base_dirs)
    _print_summary(results, args.modes)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(
                {
                    "generated": datetime.datetime.now().isoformat(),
                    "modes": args.modes,
                    "native_gfx": native_gfx,
                    "source_gfx": args.source_gfx,
                    "spoof_arch": spoof_arch,
                    "strict_tolerance": args.strict_tolerance,
                    "rng_seed": args.rng_seed,
                    "timeout_s": args.timeout,
                    "aiter_root": args.aiter_root,
                    "jobs": jobs,
                    "gpu_pool": gpu_pool[:jobs],
                    "results": [dataclasses.asdict(r) for r in results],
                },
                f,
                indent=2,
            )
            f.write("\n")
        print(f"\nWrote machine-readable record to {args.json}",
              file=sys.stderr)

    # Exit code: 0 if every native run passed (the harness itself
    # worked); 1 if even native is broken (we can't trust anything).
    # Transpilation failures are a *finding*, not a harness error.
    if "native" in args.modes:
        native_ran = [r for r in results if r.mode == "native"]
        native_ok = [r for r in native_ran if r.verdict == "PASS"]
        if native_ran and not native_ok:
            return 1
    return 0


if __name__ == "__main__":
    # Install our SIGINT / SIGTERM handler before any child is
    # spawned.  The handler propagates the signal to every live
    # child's process group (see ``_shutdown_handler``), which is
    # essential because we spawn children with
    # ``start_new_session=True`` — so the shell's Ctrl-C does *not*
    # otherwise reach them and would leak orphan python processes
    # plus stale JIT-cache locks.
    _install_shutdown_handlers()
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        # Belt-and-braces cleanup.  Our SIGINT handler already
        # SIGTERM'd every live child session before raising
        # KeyboardInterrupt; most children will have drained and
        # unregistered themselves through ``_run_one``'s return
        # path by the time we arrive here, but there's a small
        # race where a worker thread may not have called
        # ``_unregister_child`` yet.  Anything *still* in the
        # registry either (a) just hasn't been reaped yet (harmless
        # — killpg on a dying pgid no-ops or ESRCHes) or (b) ignored
        # SIGTERM (wedged in an uninterruptible HSA / hipcc call),
        # in which case SIGKILL is the only way out.  Report both
        # cases the same way — the count is the upper bound of
        # "children that may still be alive".
        pending = _kill_live_children(signal.SIGKILL)
        if pending:
            sys.stderr.write(
                f"[aiter_corpus_runner] interrupted; escalated "
                f"SIGKILL to {pending} child session(s) that had "
                f"not yet exited.\n"
            )
        else:
            sys.stderr.write(
                "[aiter_corpus_runner] interrupted; all children "
                "already exited cleanly.\n"
            )
        sys.stderr.flush()
        sys.exit(130)
