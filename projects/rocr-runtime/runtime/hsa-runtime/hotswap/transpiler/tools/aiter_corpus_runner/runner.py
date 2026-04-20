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
import dataclasses
import datetime
import functools
import json
import os
import re
import signal
import subprocess
import sys
import time
from typing import Dict, List, Optional


HERE = os.path.dirname(os.path.abspath(__file__))


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
) -> RunResult:
    """Spawn one child process for one (script, mode) and wait."""
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
    import shlex
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
        return RunResult(
            script=script, mode=mode.name,
            verdict="SPAWN_FAIL",
            detail=f"could not exec child: {e}",
            exit_code=None, signal_name=None,
            elapsed_s=0.0, stderr_tail="",
        )

    timed_out = False
    try:
        _stdout_b, stderr_b = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            _stdout_b, stderr_b = proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            _stdout_b, stderr_b = b"", b""

    elapsed = time.monotonic() - t0
    stderr = (stderr_b or b"").decode("utf-8", errors="replace")

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
        help="value for HIP_VISIBLE_DEVICES (single index like '0', or "
             "comma list).  Default: '0' if HIP_VISIBLE_DEVICES is not "
             "already set; pass '' to clear it.",
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
    print(
        f"aiter corpus runner: {len(scripts)} script(s), "
        f"modes={args.modes}, native_gfx={native_gfx!r}, "
        f"source_gfx={args.source_gfx!r}, "
        f"spoof_arch={spoof_arch!r} (legacy/salmon only), "
        f"strict_tol={args.strict_tolerance}, "
        f"rng_seed={args.rng_seed!r}, "
        f"timeout={args.timeout:.0f}s, jit_cache={args.jit_cache}",
        file=sys.stderr,
    )
    for s in scripts:
        print(f"  - {_short_label(s, base_dirs)}", file=sys.stderr)
    print(file=sys.stderr)

    results: List[RunResult] = []
    for s in scripts:
        for m in args.modes:
            mode = MODES[m]
            print(f"[run] {_short_label(s, base_dirs)} :: {m} ... ",
                  end="", flush=True, file=sys.stderr)
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
            )
            results.append(r)
            print(f"{r.verdict} ({r.detail}, {r.elapsed_s:.1f}s)",
                  file=sys.stderr)

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
    sys.exit(main())
