#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Automation script that validates the ROCm Systems Profiler (rocprof-sys)
binaries shipped inside a nightly ROCm (TheRock) tarball against the latest
tests + examples from the ``develop`` branch of ROCm/rocm-systems.

What it does (in order), inside a reusable directory ``rocprofiler-systems-tests``
created in the *current working directory*. The directory is reused across runs so
work is incremental:

  1. Resolve the latest nightly ROCm multi-arch tarball from
     https://rocm.nightlies.amd.com/tarball-multi-arch/. It is downloaded and
     (re)extracted into ``<workdir>/rocm`` (ROCM_PATH, providing
     ``bin/rocprof-sys-*``) ONLY when a newer nightly is available than the one
     already extracted.
  2. Sparse-clone (or ``git fetch``+update) the ``develop`` branch of
     ROCm/rocm-systems, checking out only ``projects/rocprofiler-systems``.
  3. Create/reuse an isolated Python venv with the pytest test dependencies
     (from ``requirements.txt``).
  4. Build the example programs (standalone) and stage the pytest test-suite into
     the ROCm prefix so the suite runs in "install mode". The examples are rebuilt
     ONLY when the ROCm tree was replaced, the source changed, the artifacts are
     missing, or ``--force-rebuild`` is given.
  5. Run the pytest suite in install mode so every test exercises the
     ``rocprof-sys-*`` binaries that live in the downloaded tarball's ``bin/``
     folder (nothing from rocprof-sys itself is rebuilt).

Typical usage on a GPU test machine:

    python3 run-nightly-tarball-tests.py                 # latest multiarch nightly
    python3 run-nightly-tarball-tests.py --variant auto  # smallest per-GPU tarball
    python3 run-nightly-tarball-tests.py --tier quick    # fast smoke subset
    python3 run-nightly-tarball-tests.py --reruns 2      # retry flaky tests
    python3 run-nightly-tarball-tests.py --rocm-version 7.15.0a20260717
    python3 run-nightly-tarball-tests.py --pytest-args "-m gpu -k transpose"

Validate freshly-built binaries instead of the tarball's shipped ones (QA-style
build-from-source + test; uses the tarball only for ROCm runtime/toolchain):

    python3 run-nightly-tarball-tests.py --build-from-source

Air-gapped cluster (compute nodes have no network) - two phases sharing one
--work-dir on a shared filesystem:

    # Phase 1, on a networked node (e.g. the login node): fetch everything.
    python3 run-nightly-tarball-tests.py --work-dir /scratch/$USER/rocm-test \
        --variant auto --prepare-only

    # Phase 2, on the air-gapped GPU compute node: no network, build + run tests.
    python3 run-nightly-tarball-tests.py --work-dir /scratch/$USER/rocm-test \
        --offline --tier standard

Each run writes a detailed log, a summary log (with per-step timings, test counts,
failed-test names, and the rocprof-sys version under test), and, on failure, a
failures log listing each failing test with its output.

Environment variables:
    The script inherits your shell environment and forwards it to every git, build
    and pytest subprocess, so you can pre-set knobs before invoking it, e.g.:
      - http_proxy / https_proxy   : honored by the download, git and pip
      - PIP_INDEX_URL / PIP_NO_INDEX / PIP_FIND_LINKS : mirrored / offline pip
      - CC / CXX                   : compilers for the example/source builds
      - ROCPROFSYS_* / CMAKE_* / MAKEFLAGS : forwarded to the build/tests
    The script manages a few variables and will override yours:
      ROCM_PATH, ROCPROFSYS_CI, ROCPROFSYS_INSTALL_DIR / ROCPROFSYS_BUILD_DIR, and
      TMPDIR/TMP/TEMP (redirected into <work-dir>/tmp during tests).
    PATH and LD_LIBRARY_PATH are PREPENDED with the tarball's directories (your
    existing entries are preserved). ROCPROFSYS_TMPDIR and GIT_HTTP_LOW_SPEED_* are
    only set when you have not already set them.

Run ``--help`` for the full list of options.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

NIGHTLY_TARBALL_INDEX = "https://rocm.nightlies.amd.com/tarball-multi-arch/"
NIGHTLY_TARBALL_BASE = "https://rocm.nightlies.amd.com/tarball-multi-arch"
ROCM_SYSTEMS_REPO = "https://github.com/ROCm/rocm-systems.git"
PROJECT_SUBDIR = "projects/rocprofiler-systems"

# rocprof-sys binaries expected to live in the tarball's bin/ folder.
REQUIRED_ROCPROFSYS_BINARIES = [
    "rocprof-sys-run",
    "rocprof-sys-instrument",
    "rocprof-sys-sample",
    "rocprof-sys-avail",
    "rocprof-sys-causal",
]

# Tests known to be problematic under the TheRock install-mode flow. Mirrors
# tests/test_categories.yaml::_common_therock_regex_excludes. Applied as a
# pytest -k "not (...)" expression unless the caller overrides --pytest-args.
DEFAULT_DESELECT_KEYWORDS = [
    "transferbench",
    "fork",
    "openmp_target",
    "jacobi_usm",
    "jacobi_roctx",
    "jpeg_decode",
    "matrix_exponential",
    "scratch_memory",
    "selective_region",
    "shmem_pingpong",
    "video_decode",
]

# Curated fast smoke subset for `--tier quick` (pytest -k substrings). Approximates
# the quick tier in tests/test_categories.yaml (which is defined for CTest labels).
QUICK_KEYWORDS = [
    "transpose",
    "config",
    "cli",
    "avail",
    "presets",
    "roctx",
]

# Marker (label) categories excluded from the standard/quick tiers. Mirrors
# tests/test_categories.yaml::_common_therock_labels_exclude. Applied as a pytest
# marker expression "not X and not Y ...". We deliberately do NOT filter on the
# 'gpu' marker, so CPU-only install-mode tests (cli_help, config, presets, ...)
# are also validated against the tarball binaries.
LABEL_EXCLUDES = [
    "annotate",
    "mpi",
    "julia",
    "attach",
    "lulesh",
    "network",
    "overflow",
    "thread_limit",
]

# Map a detected GPU arch (gfxNNNN) to the smallest matching TheRock tarball
# variant. Used by `--variant auto`. Anything unmapped falls back to multiarch.
ARCH_TO_VARIANT = {
    "gfx900": "gfx900",
    "gfx906": "gfx906",
    "gfx908": "gfx908",
    "gfx90a": "gfx90a",
    "gfx942": "gfx94X-dcgpu",
    "gfx950": "gfx950-dcgpu",
    "gfx1010": "gfx101X-dgpu",
    "gfx1011": "gfx101X-dgpu",
    "gfx1012": "gfx101X-dgpu",
    "gfx1030": "gfx103X-all",
    "gfx1031": "gfx103X-all",
    "gfx1032": "gfx103X-all",
    "gfx1100": "gfx110X-all",
    "gfx1101": "gfx110X-all",
    "gfx1102": "gfx110X-all",
    "gfx1150": "gfx1150",
    "gfx1151": "gfx1151",
    "gfx1152": "gfx1152",
    "gfx1153": "gfx1153",
    "gfx1200": "gfx120X-all",
    "gfx1201": "gfx120X-all",
    "gfx1202": "gfx120X-all",
    "gfx1250": "gfx125X-dcgpu",
    "gfx1251": "gfx125X-dcgpu",
}

# minimum free disk space (GB) required before a tarball download
DEFAULT_MIN_FREE_GB = 40


# --------------------------------------------------------------------------- #
# Logging helpers
# --------------------------------------------------------------------------- #

_STEP = 0
_DETAIL_FH = None  # file handle for the detailed log (opened once workdir is known)
_SUMMARY_PATH = None  # Path for the summary log (set once workdir is known)
_FACTS: dict = {}  # accumulated run facts, also flushed to the summary on abort

_RUN_START = time.monotonic()
_STEP_TITLE = None  # title of the currently-open step (for timing)
_STEP_START = 0.0
_STEP_TIMES: list[tuple[str, float]] = []  # (title, seconds) for the summary


def _fmt_dur(secs: float) -> str:
    secs = int(round(secs))
    if secs < 60:
        return f"{secs}s"
    m, s = divmod(secs, 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s"


def _emit(text: str, *, end: str = "\n", stream=None) -> None:
    """Write ``text`` to the console and, if open, to the detailed log file."""
    out = stream or sys.stdout
    out.write(text + end)
    out.flush()
    if _DETAIL_FH is not None:
        _DETAIL_FH.write(text + end)
        _DETAIL_FH.flush()


def open_detailed_log(path: Path) -> None:
    """Open the detailed log file; all subsequent output is teed into it."""
    global _DETAIL_FH
    _DETAIL_FH = open(path, "a", encoding="utf-8")  # noqa: SIM115
    _DETAIL_FH.write(
        f"\n{'#' * 72}\n# rocprof-sys nightly tarball test run\n"
        f"# started: {_dt.datetime.now().isoformat(timespec='seconds')}\n"
        f"{'#' * 72}\n"
    )
    _DETAIL_FH.flush()


def log(msg: str) -> None:
    _emit(f"[nightly-test] {msg}")


def _close_step() -> None:
    """Record + print the duration of the step that is currently open."""
    global _STEP_TITLE
    if _STEP_TITLE is not None:
        dur = time.monotonic() - _STEP_START
        _STEP_TIMES.append((_STEP_TITLE, dur))
        _emit(f"[nightly-test] step completed in {_fmt_dur(dur)}")
        _STEP_TITLE = None


def step(title: str) -> None:
    global _STEP, _STEP_TITLE, _STEP_START
    _close_step()
    _STEP += 1
    _emit(f"\n{'=' * 72}\n[{_STEP}] {title}\n{'=' * 72}")
    _STEP_TITLE = title
    _STEP_START = time.monotonic()


def die(msg: str, code: int = 1) -> "None":
    _emit(f"\n[nightly-test][ERROR] {msg}", stream=sys.stderr)
    if _SUMMARY_PATH is not None:
        _FACTS.setdefault("result", "ABORTED")
        _FACTS["error"] = msg.splitlines()[0]
        try:
            write_summary(_SUMMARY_PATH, _FACTS)
        except Exception:  # noqa: BLE001
            pass
    sys.exit(code)


def run(cmd, *, cwd=None, env=None, check=True, quiet=False):
    """Run a subprocess, streaming (and teeing) its output. Returns returncode."""
    printable = " ".join(str(c) for c in cmd)
    if not quiet:
        log(f"$ {printable}" + (f"   (cwd={cwd})" if cwd else ""))
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    for line in proc.stdout:
        _emit(line, end="")
    proc.wait()
    if check and proc.returncode != 0:
        die(f"command failed (exit {proc.returncode}): {printable}", proc.returncode)
    return proc.returncode


# --------------------------------------------------------------------------- #
# Tarball resolution + download
# --------------------------------------------------------------------------- #


def _require_https(url: str) -> None:
    """Refuse anything but HTTPS (defense-in-depth against downgraded fetches)."""
    if not url.lower().startswith("https://"):
        die(f"refusing to fetch a non-HTTPS URL: {url}")


def _http_get_text(url: str, timeout: int = 60) -> str:
    _require_https(url)
    with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310
        return resp.read().decode("utf-8", errors="replace")


def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def fetch_published_sha256(url: str) -> tuple[str | None, str | None]:
    """Best-effort fetch of a published checksum next to ``url``.

    Returns ``(hex_digest, source_url)`` or ``(None, None)`` if none is published.
    """
    for ext in (".sha256", ".sha256sum", ".SHA256"):
        try:
            text = _http_get_text(url + ext, timeout=30)
        except Exception:  # noqa: BLE001
            continue
        m = re.search(r"\b[0-9a-fA-F]{64}\b", text)
        if m:
            return m.group(0).lower(), url + ext
    return None, None


def verify_tarball(
    url: str, tarball: Path, expected: str | None, require: bool, facts: dict
) -> None:
    """Verify the downloaded archive's SHA-256 before it is extracted/executed.

    Priority: an explicit ``--sha256`` value, else a checksum published next to the
    tarball. If neither is available, warn (or abort when ``require`` is set) since
    the extracted binaries are subsequently executed.
    """
    origin = "cli (--sha256)"
    if not expected:
        expected, origin = fetch_published_sha256(url)

    if not expected:
        msg = (
            "no SHA-256 checksum available for the downloaded tarball "
            "(none published alongside it and none passed via --sha256)"
        )
        if require:
            die(msg + "; aborting because --require-checksum was set.")
        log(
            "WARNING: " + msg + ". Skipping integrity verification. Pass --sha256 "
            "or --require-checksum to enforce."
        )
        facts["sha256_verified"] = "no (unavailable)"
        return

    log(f"Verifying tarball SHA-256 (source: {origin})...")
    actual = _sha256_file(tarball)
    if actual.lower() != expected.lower():
        die(
            "SHA-256 MISMATCH - the download is corrupt or has been tampered with:\n"
            f"       expected: {expected}\n"
            f"       actual:   {actual}"
        )
    log(f"SHA-256 verified OK: {actual}")
    facts["sha256_verified"] = "yes"
    facts["sha256"] = actual


def resolve_tarball(variant: str, version: str | None) -> tuple[str, str]:
    """Return (filename, url) of the nightly dist tarball to download.

    ``variant`` is e.g. ``multiarch`` or ``gfx94X-dcgpu``. The regex deliberately
    anchors a digit right after ``<variant>-`` so the separate ``<variant>-tests-``
    tarballs are never matched.
    """
    log(f"Fetching nightly tarball index: {NIGHTLY_TARBALL_INDEX}")
    try:
        html = _http_get_text(NIGHTLY_TARBALL_INDEX)
    except Exception as exc:  # noqa: BLE001
        die(f"could not fetch tarball index: {exc}")

    pattern = re.compile(
        r"therock-dist-linux-"
        + re.escape(variant)
        + r"-((\d+\.\d+\.\d+)a(\d{8}))\.tar\.gz"
    )

    # candidates: list of (date_int, version_str, filename)
    candidates = {}
    for m in pattern.finditer(html):
        filename = m.group(0)
        full_version = m.group(1)  # e.g. 7.15.0a20260717
        date_int = int(m.group(3))  # e.g. 20260717
        candidates[filename] = (date_int, full_version)

    if not candidates:
        die(
            f"no nightly tarballs found for variant '{variant}'.\n"
            f"       Check available variants at {NIGHTLY_TARBALL_INDEX}\n"
            f"       (e.g. multiarch, gfx94X-dcgpu, gfx90a, gfx110X-all, ...)."
        )

    if version:
        # match either the full 'X.Y.ZaYYYYMMDD' or just the trailing date
        matches = [
            fn for fn, (_d, ver) in candidates.items() if version in fn or ver == version
        ]
        if not matches:
            die(
                f"requested version '{version}' not found for variant '{variant}'.\n"
                f"       Browse {NIGHTLY_TARBALL_INDEX} for valid values."
            )
        filename = sorted(matches)[-1]
    else:
        filename = max(candidates, key=lambda fn: candidates[fn][0])

    url = f"{NIGHTLY_TARBALL_BASE}/{filename}"
    return filename, url


def download_file(url: str, dest: Path) -> None:
    """Download ``url`` to ``dest`` with resume support (prefers wget/curl)."""
    _require_https(url)
    if dest.exists() and dest.stat().st_size > 0:
        log(f"Tarball already present, skipping download: {dest}")
        return

    tmp = dest.with_suffix(dest.suffix + ".part")
    wget = shutil.which("wget")
    curl = shutil.which("curl")
    if wget:
        # Live in-place progress bar (nicer than the default "dot" meter). Under a
        # non-TTY log (cron/sbatch) this bar is redrawn via carriage returns, so the
        # captured output is a single messy '\r'-laden line - acceptable trade-off.
        cmd = [
            wget,
            "--continue",
            "--tries=3",
            "-q",
            "--show-progress",
            "--progress=bar:force:noscroll",
            "-O",
            str(tmp),
            url,
        ]
    elif curl:
        # curl's default meter already updates in place (no per-line spam).
        cmd = [curl, "-fL", "--retry", "3", "-C", "-", "-o", str(tmp), url]
    else:
        die(
            "neither 'wget' nor 'curl' is available to download the tarball. "
            "Install one, or pre-stage the tarball with --prepare-only on a "
            "networked node and run with --offline."
        )

    # Run the downloader with inherited stdio so its in-place progress bar renders
    # live on the console, instead of teeing every progress line into the logs.
    log(f"$ {' '.join(cmd)}")
    rc = subprocess.run(cmd).returncode  # noqa: S603
    if rc != 0:
        die(f"download failed (exit {rc}): {url}", rc)
    tmp.rename(dest)


def extract_tarball(tarball: Path, rocm_dir: Path) -> None:
    if rocm_dir.exists():
        shutil.rmtree(rocm_dir)
    rocm_dir.mkdir(parents=True, exist_ok=True)
    log(f"Extracting {tarball.name} -> {rocm_dir} (this can take a while)")
    # Prefer the system tar (much faster for multi-GB archives). GNU tar strips
    # leading '/' and rejects '..' members, so it is safe against tarbombs.
    tar = shutil.which("tar")
    if tar:
        run([tar, "-xf", str(tarball), "-C", str(rocm_dir)])
    else:
        with tarfile.open(tarball) as tf:
            # PEP 706 'data' filter (Python 3.8.17+/3.9.17+/3.10.12+/3.12+) blocks
            # path traversal, absolute paths and unsafe links.
            tf.extractall(rocm_dir, filter="data")  # noqa: S202


# marker file recording which tarball is currently extracted into rocm_dir
def _rocm_marker(workdir: Path) -> Path:
    return workdir / ".rocm-version"


def current_rocm_tarball(workdir: Path) -> str | None:
    marker = _rocm_marker(workdir)
    return marker.read_text().strip() if marker.is_file() else None


def set_rocm_tarball(workdir: Path, filename: str) -> None:
    _rocm_marker(workdir).write_text(filename + "\n")


def cleanup_downloaded_tarballs(workdir: Path) -> None:
    """Delete downloaded dist tarballs (and .part files) to reclaim disk space.

    Called after a successful extraction, since the extracted ``rocm/`` tree is all
    we need going forward.
    """
    for tb in workdir.glob("therock-dist-*.tar.gz*"):
        log(f"Removing extracted tarball to reclaim space: {tb.name}")
        tb.unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
# Environment for building / running against the tarball
# --------------------------------------------------------------------------- #


def make_rocm_env(base_env: dict, rocm_dir: Path) -> dict:
    env = dict(base_env)
    env["ROCM_PATH"] = str(rocm_dir)
    env["ROCPROFSYS_CI"] = "ON"

    # Make git abort quickly instead of hanging forever when a compute node has no
    # egress (the common cluster case). Abort if <1 KB/s for 30s.
    env.setdefault("GIT_HTTP_LOW_SPEED_LIMIT", "1000")
    env.setdefault("GIT_HTTP_LOW_SPEED_TIME", "30")

    bins = [str(rocm_dir / "bin"), str(rocm_dir / "llvm" / "bin")]
    libs = [
        str(rocm_dir / "lib"),
        str(rocm_dir / "lib64"),
        str(rocm_dir / "llvm" / "lib"),
    ]
    env["PATH"] = os.pathsep.join(bins + [env.get("PATH", "")]).rstrip(os.pathsep)
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        libs + [env.get("LD_LIBRARY_PATH", "")]
    ).rstrip(os.pathsep)
    return env


def apply_hpc_cray_settings(args, facts: dict) -> list[str]:
    """Bundle Cray-MPICH-on-gfx90a HPC specifics.

    Returns extra ``-D`` CMake args (MPI compile/link flags for the example build)
    and mutates ``args`` / ``os.environ`` in place:
      - injects Cray MPICH + XPMEM + GTL include/link flags so the tarball clang
        can build the GPU-aware MPI examples (e.g. hpc/jacobi-hip),
      - defaults the tarball variant to gfx90a,
      - skips examples that don't build against the tarball (jpegdecode/videodecode),
      - enables GPU-aware MPI at runtime (MPICH_GPU_SUPPORT_ENABLED=1).

    Requires the Cray MPICH module to be loaded (``module load cray-mpich``).
    """
    log("Applying HPC Cray MPICH (gfx90a) settings...")

    mpich_dir = os.environ.get("MPICH_DIR") or os.environ.get("CRAY_MPICH_DIR")
    if not mpich_dir:
        die(
            "--hpc-cray requires Cray MPICH in the environment but neither "
            "MPICH_DIR nor CRAY_MPICH_DIR is set.\n"
            "       Run 'module load cray-mpich' (and the gfx90a accel module) first."
        )
    xpmem = os.environ.get("CRAY_XPMEM_POST_LINK_OPTS", "")
    gtl_dir = os.environ.get("PE_MPICH_GTL_DIR_amd_gfx90a", "")
    gtl_libs = os.environ.get("PE_MPICH_GTL_LIBS_amd_gfx90a", "")
    if not gtl_libs:
        log(
            "WARNING: PE_MPICH_GTL_LIBS_amd_gfx90a is empty; GPU-aware MPI link may "
            "fail. Ensure the craype-accel-amd-gfx90a module is loaded."
        )

    # compile: MPI headers ; link: MPI + XPMEM + GPU-transport (GTL)
    cxx_flags = f"-I{mpich_dir}/include"
    link_flags = " ".join(
        p
        for p in [f"-L{mpich_dir}/lib", "-lmpi", xpmem, "-lxpmem", gtl_dir, gtl_libs]
        if p
    )
    extra = [f"-DCMAKE_CXX_FLAGS={cxx_flags}", f"-DCMAKE_EXE_LINKER_FLAGS={link_flags}"]

    # runtime: resolve libmpi/GTL and enable GPU-aware transfers
    lib_dirs = [f"{mpich_dir}/lib"]
    if gtl_dir.startswith("-L"):
        lib_dirs.append(gtl_dir[2:])
    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(
        lib_dirs + [os.environ.get("LD_LIBRARY_PATH", "")]
    ).rstrip(os.pathsep)
    os.environ.setdefault("MPICH_GPU_SUPPORT_ENABLED", "1")

    # default to the gfx90a tarball; honor an explicit --variant
    if args.variant == "multiarch":
        args.variant = "gfx90a"

    # skip examples that don't build against the tarball in this environment
    for ex in ("jpegdecode", "videodecode"):
        if ex not in args.disable_examples.split(","):
            args.disable_examples = (
                f"{args.disable_examples},{ex}" if args.disable_examples else ex
            )

    facts["hpc_cray"] = "on"
    facts["mpich_dir"] = mpich_dir
    log(f"HPC Cray: MPICH_DIR={mpich_dir}")
    log(f"HPC Cray: variant={args.variant}, disabled_examples={args.disable_examples}")
    return extra


# --------------------------------------------------------------------------- #
# Preflight / environment discovery
# --------------------------------------------------------------------------- #


def _tool_version(exe: str, env: dict | None = None) -> str:
    try:
        r = subprocess.run(
            [exe, "--version"], capture_output=True, text=True, timeout=15, env=env
        )
        lines = [ln.strip() for ln in (r.stdout + r.stderr).splitlines() if ln.strip()]
        # skip log-noise lines like "[hh:mm:ss][P:..][file] ... [error] ..." that
        # some rocprof-sys tools emit before the version banner
        clean = [ln for ln in lines if not ln.startswith("[") and "Exception" not in ln]
        for ln in clean:
            if re.search(r"\d+\.\d+\.\d+", ln) or "version" in ln.lower():
                return ln
        if clean:
            return clean[0]
        if lines:
            return lines[0]
    except Exception:  # noqa: BLE001
        pass
    return "unknown"


def detect_gpu_archs(rocm_dir: Path | None, env: dict | None = None) -> list[str]:
    """Return distinct gfx targets reported by rocminfo (empty if none/no GPU)."""
    search = []
    if rocm_dir is not None:
        search.append(rocm_dir / "bin" / "rocminfo")
    which = shutil.which("rocminfo", path=(env or os.environ).get("PATH"))
    if which:
        search.append(Path(which))
    rocminfo = next((p for p in search if Path(p).exists()), None)
    if rocminfo is None:
        return []
    try:
        out = subprocess.run(
            [str(rocminfo)], capture_output=True, text=True, timeout=30, env=env
        ).stdout
    except Exception:  # noqa: BLE001
        return []
    archs: list[str] = []
    for a in re.findall(r"gfx[0-9a-fA-F]+", out):
        if a != "gfx000" and a not in archs:
            archs.append(a)
    return archs


def resolve_variant(variant: str, archs: list[str]) -> str:
    """Map ``--variant auto`` to a per-family tarball based on detected GPU."""
    if variant != "auto":
        return variant
    for a in archs:
        if a in ARCH_TO_VARIANT:
            log(f"--variant auto: detected {a} -> tarball variant '{ARCH_TO_VARIANT[a]}'")
            return ARCH_TO_VARIANT[a]
    log(
        "WARNING: --variant auto could not map a detected GPU arch "
        f"({', '.join(archs) or 'none'}); falling back to 'multiarch'."
    )
    return "multiarch"


def _trust_problems(path: Path) -> list[str]:
    """Report ownership/permission issues that would let another local user plant
    code we later execute (venv python, staged tests, tarball binaries)."""
    problems = []
    try:
        st = path.stat()
    except FileNotFoundError:
        return problems
    if st.st_uid not in (os.getuid(), 0):
        problems.append(f"{path} is owned by uid {st.st_uid}, not you ({os.getuid()})")
    if st.st_mode & 0o0002:
        problems.append(f"{path} is world-writable")
    return problems


def preflight(args, workdir: Path, rocm_dir: Path) -> list[str]:
    """Fail fast on missing tools/space/network; return detected GPU archs."""
    step("Preflight checks")

    # workdir trust: we execute code from here (venv, staged tests, tarball bins),
    # so refuse a reused dir another user could have tampered with.
    trust_issues = []
    for p in (workdir, rocm_dir, workdir / "venv"):
        trust_issues += _trust_problems(p)
    if trust_issues:
        die(
            "untrusted working directory (could allow code injection):\n"
            + "\n".join(f"       - {i}" for i in trust_issues)
            + "\n       Use a private --work-dir you own, or remove the directory."
        )

    # required + informational tools
    for tool in ("git", "cmake"):
        path = shutil.which(tool)
        if not path:
            die(
                f"required tool '{tool}' not found on PATH. Install it (or pass "
                f"--install-system-deps) and re-run."
            )
        log(f"{tool:8s}: {_tool_version(tool)}")
    log(f"python  : {sys.version.split()[0]} ({sys.executable})")
    cxx = shutil.which("g++") or shutil.which("c++")
    if cxx:
        log(f"c++     : {_tool_version(cxx)}")
    else:
        log(
            "WARNING: no system C++ compiler (g++/c++) found; example build may "
            "rely solely on the tarball toolchain."
        )

    # disk space
    free_gb = shutil.disk_usage(workdir).free / (1024**3)
    rocm_present = (rocm_dir / "bin").is_dir()
    log(
        f"free disk: {free_gb:.1f} GB at {workdir} (min recommended: "
        f"{args.min_free_gb} GB)"
    )
    if free_gb < args.min_free_gb:
        if not rocm_present and not args.skip_download:
            die(
                f"insufficient free disk space ({free_gb:.1f} GB < "
                f"{args.min_free_gb} GB) for the tarball download/extract. "
                f"Free space or lower --min-free-gb."
            )
        log("WARNING: free disk space is below the recommended minimum.")

    # network reachability (only matters if we may download)
    if not args.skip_download:
        try:
            _http_get_text(NIGHTLY_TARBALL_INDEX, timeout=20)
            log(f"network : reachable ({NIGHTLY_TARBALL_INDEX})")
        except Exception as exc:  # noqa: BLE001
            die(
                f"cannot reach the nightly tarball index ({NIGHTLY_TARBALL_INDEX}): "
                f"{exc}. Use --skip-download to reuse an existing ROCm tree offline."
            )

    # GPU visibility. When the tarball is already extracted, run its rocminfo with
    # a proper ROCm env so it can find its own libraries (otherwise it reports none).
    det_env = make_rocm_env(os.environ.copy(), rocm_dir) if rocm_present else None
    archs = detect_gpu_archs(rocm_dir if rocm_present else None, det_env)
    if archs:
        log(f"GPU     : {', '.join(archs)}")
    elif not args.skip_tests and not args.prepare_only:
        log(
            "WARNING: no GPU detected via rocminfo. Most GPU tests will "
            "fail/skip. Continue anyway..."
        )
    return archs


def report_under_test(rocm_dir: Path, env: dict, facts: dict) -> None:
    """Log the manifest + rocprof-sys version being validated."""
    manifest = rocm_dir / "share" / "therock" / "therock_manifest.json"
    if manifest.is_file():
        log(f"TheRock manifest: {manifest}")
        try:
            data = json.loads(manifest.read_text())
            _emit(json.dumps(data, indent=2))
        except Exception:  # noqa: BLE001
            _emit(manifest.read_text())

    avail = rocm_dir / "bin" / "rocprof-sys-avail"
    version = _tool_version(str(avail), env) if avail.exists() else "unknown"
    facts["rocprofsys_version"] = version
    log(f"rocprof-sys binaries under test: {rocm_dir / 'bin'}")
    log(f"rocprof-sys version : {version}")


def smoke_check_rocprofsys(rocm_dir: Path, env: dict) -> None:
    """Fail fast if the shipped rocprof-sys binaries can't even start here.

    Runs ``rocprof-sys-avail --version``; if it is killed by a signal (e.g. SIGILL,
    SIGSEGV, SIGABRT) the binaries are incompatible with this machine and every test
    would fail, so abort early with a clear, generic message.
    """
    avail = rocm_dir / "bin" / "rocprof-sys-avail"
    if not avail.exists():
        return
    try:
        rc = subprocess.run(
            [str(avail), "--version"],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=60,
        ).returncode
    except subprocess.TimeoutExpired:
        log(
            "WARNING: 'rocprof-sys-avail --version' timed out during smoke check; "
            "continuing."
        )
        return
    # killed by a signal: negative returncode (direct exec) or 128+signal convention
    if rc < 0 or rc >= 128:
        sig = -rc if rc < 0 else rc - 128
        die(
            "the rocprof-sys binaries in the tarball crash on this system: "
            f"'rocprof-sys-avail --version' was killed by signal {sig} (exit {rc}).\n"
            "       The shipped binaries appear incompatible with this machine (a "
            "common cause is a CPU instruction-set / build mismatch). Every test "
            "would fail, so aborting now.\n"
            "       Investigate with: "
            f"gdb -q --batch -ex run -ex 'x/i $pc' --args {avail}"
        )
    if rc != 0:
        log(
            f"WARNING: 'rocprof-sys-avail --version' exited {rc} during smoke check "
            "(continuing)."
        )


def capture_rocm_sanity(rocm_dir: Path, env: dict, out_path: Path, facts: dict) -> None:
    """Write a ROCm environment sanity report (tool versions, GPU info) to a file."""
    checks = [
        ("hipcc --version", [str(rocm_dir / "bin" / "hipcc"), "--version"]),
        (
            "rocm-smi --showproductname",
            [str(rocm_dir / "bin" / "rocm-smi"), "--showproductname"],
        ),
        ("amd-smi version", [str(rocm_dir / "bin" / "amd-smi"), "version"]),
        ("rocminfo", [str(rocm_dir / "bin" / "rocminfo")]),
    ]
    lines = [f"ROCm sanity report ({rocm_dir})", "=" * 60]
    for title, cmd in checks:
        lines.append(f"\n### {title}")
        if not Path(cmd[0]).exists():
            lines.append(f"(skipped: {cmd[0]} not found)")
            continue
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=60, env=env)
            out = (r.stdout or "") + (r.stderr or "")
            # rocminfo is long - keep only the first ~120 lines
            if title == "rocminfo":
                out = "\n".join(out.splitlines()[:120])
            lines.append(out.rstrip())
        except Exception as exc:  # noqa: BLE001
            lines.append(f"(error: {exc})")
    out_path.write_text("\n".join(lines) + "\n")
    facts["rocm_sanity_log"] = str(out_path)
    log(f"ROCm sanity report -> {out_path}")


# --------------------------------------------------------------------------- #
# Steps
# --------------------------------------------------------------------------- #


def _git_rev(repo_dir: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    ).stdout.strip()


def _update_submodules(repo_dir: Path, env: dict) -> None:
    """Init/update the project's git submodules (needed for --build-from-source)."""
    log("Updating git submodules for projects/rocprofiler-systems (recursive)...")
    # Path-limited first (faster; avoids other monorepo projects), then fall back.
    rc = run(
        [
            "git",
            "-C",
            str(repo_dir),
            "submodule",
            "update",
            "--init",
            "--recursive",
            "--",
            PROJECT_SUBDIR,
        ],
        env=env,
        check=False,
    )
    if rc != 0:
        run(
            ["git", "-C", str(repo_dir), "submodule", "update", "--init", "--recursive"],
            env=env,
        )


def sync_source(
    workdir: Path,
    branch: str,
    env: dict,
    no_fetch: bool = False,
    submodules: bool = False,
) -> tuple[Path, bool]:
    """Sparse-clone (or update) rocm-systems.

    Returns ``(source_dir, source_changed)`` where ``source_changed`` is True when
    a fresh clone happened or the branch HEAD moved since the last run.

    When ``no_fetch`` is set (offline mode), an existing checkout is reused as-is
    with no network access; if none exists the run aborts with guidance. When
    ``submodules`` is set (source builds), submodules are initialized (online) or
    verified present (offline).
    """
    repo_dir = workdir / "rocm-systems"
    src_dir = repo_dir / PROJECT_SUBDIR

    if no_fetch:
        if not (src_dir / "CMakeLists.txt").is_file():
            die(
                "offline mode (--offline) but no source checkout at "
                f"{src_dir}.\n"
                "       Run once with --prepare-only on a networked node (e.g. the "
                "login node) first."
            )
        rev = _git_rev(repo_dir)
        log(f"Offline: reusing existing checkout at {rev[:10]} (no fetch).")
        if submodules and not (src_dir / "external").is_dir():
            die(
                "offline --build-from-source but submodules are not present at "
                f"{src_dir / 'external'}.\n"
                "       Run --prepare-only --build-from-source on a networked node "
                "first."
            )
        return src_dir, False

    if (src_dir / "CMakeLists.txt").is_file():
        old_rev = _git_rev(repo_dir)
        log(f"Existing checkout at {old_rev[:10]}; fetching latest '{branch}'...")
        run(["git", "-C", str(repo_dir), "fetch", "--prune", "origin", branch], env=env)
        run(["git", "-C", str(repo_dir), "checkout", branch], env=env, check=False)
        run(["git", "-C", str(repo_dir), "reset", "--hard", f"origin/{branch}"], env=env)
        new_rev = _git_rev(repo_dir)
        changed = old_rev != new_rev
        if changed:
            log(f"Source updated: {old_rev[:10]} -> {new_rev[:10]}")
        else:
            log(f"Source already up to date at {new_rev[:10]}")
        if submodules:
            _update_submodules(repo_dir, env)
        return src_dir, changed

    run(
        [
            "git",
            "clone",
            "--filter=blob:none",
            "--sparse",
            "--branch",
            branch,
            ROCM_SYSTEMS_REPO,
            str(repo_dir),
        ],
        env=env,
    )
    run(["git", "-C", str(repo_dir), "sparse-checkout", "set", PROJECT_SUBDIR], env=env)

    if not (src_dir / "CMakeLists.txt").is_file():
        die(f"sparse checkout did not produce expected source at {src_dir}")

    if submodules:
        _update_submodules(repo_dir, env)

    log(f"Checked out {branch} @ {_git_rev(repo_dir)[:10]}")
    return src_dir, True


def make_venv(workdir: Path, src_dir: Path, skip_install: bool = False) -> Path:
    """Create a venv and install requirements.txt. Returns the venv python path.

    When ``skip_install`` is set (offline mode), an existing venv is reused without
    any pip network access; if none exists the run aborts with guidance.
    """
    venv_dir = workdir / "venv"
    py = venv_dir / "bin" / "python"

    if skip_install:
        if not py.exists():
            die(
                "offline mode (--offline) but no venv at "
                f"{venv_dir}.\n"
                "       Run once with --prepare-only on a networked node (e.g. the "
                "login node) first."
            )
        log(f"Offline: reusing existing venv at {venv_dir} (no pip install).")
        return py

    if not py.exists():
        run([sys.executable, "-m", "venv", "--system-site-packages", str(venv_dir)])
    run([str(py), "-m", "pip", "install", "--upgrade", "pip", "wheel"], quiet=True)

    req = src_dir / "requirements.txt"
    if req.is_file():
        run([str(py), "-m", "pip", "install", "-r", str(req)])
    else:
        log(f"WARNING: {req} not found; installing pytest directly")
        run(
            [
                str(py),
                "-m",
                "pip",
                "install",
                "pytest",
                "pytest-subtests",
                "PyYAML",
                "numpy",
            ]
        )
    # extra plugins used by the runner: real per-test timeouts + flaky retries
    run(
        [str(py), "-m", "pip", "install", "pytest-timeout", "pytest-rerunfailures"],
        check=False,
    )
    return py


def capture_pip_freeze(venv_py: Path, out_path: Path, facts: dict) -> None:
    """Record the exact resolved venv package versions (reproducibility manifest)."""
    try:
        r = subprocess.run(
            [str(venv_py), "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        out_path.write_text(r.stdout)
        facts["pip_freeze_log"] = str(out_path)
        log(f"pip freeze -> {out_path}")
    except Exception as exc:  # noqa: BLE001
        log(f"WARNING: could not capture pip freeze: {exc}")


def install_system_deps() -> None:
    """Best-effort install of build/runtime system packages (needs sudo/root)."""
    apt = shutil.which("apt-get")
    if not apt:
        log("apt-get not found; skipping system-dependency installation.")
        return
    prefix = [] if os.geteuid() == 0 else (["sudo"] if shutil.which("sudo") else [])
    if not prefix and os.geteuid() != 0:
        log("Not root and sudo unavailable; skipping system-dependency installation.")
        return
    pkgs = ["build-essential", "cmake", "libopenmpi-dev", "git"]
    run(prefix + [apt, "update"], check=False)
    run(prefix + [apt, "install", "-y", *pkgs], check=False)


def build_from_source(
    src_dir: Path,
    workdir: Path,
    rocm_dir: Path,
    env: dict,
    venv_py: Path,
    jobs: int,
    use_mpi: bool,
    disable_examples: list[str] | None = None,
    extra_cmake_args: list[str] | None = None,
) -> Path:
    """Configure + build the full rocprofiler-systems project from source.

    Builds the rocprof-sys binaries, examples and test suite in-tree (like CI/QA),
    using the ROCm runtime + toolchain from the extracted tarball. Returns the
    build directory (its binaries are validated in pytest 'build mode').
    """
    build_dir = workdir / "build-from-source"
    cmake = shutil.which("cmake") or "cmake"

    # The in-build pytest CTest-generation step needs pytest + python from our venv,
    # so put the venv on PATH and point CMake's Python discovery at it.
    build_env = dict(env)
    build_env["PATH"] = os.pathsep.join(
        [str(venv_py.parent), build_env.get("PATH", "")]
    ).rstrip(os.pathsep)

    cfg = [
        cmake,
        "-S",
        str(src_dir),
        "-B",
        str(build_dir),
        "-DCMAKE_BUILD_TYPE=RelWithDebInfo",
        f"-DCMAKE_PREFIX_PATH={rocm_dir}",
        f"-DCMAKE_INSTALL_PREFIX={workdir / 'install-from-source'}",
        f"-DPython3_EXECUTABLE={venv_py}",
        "-DROCPROFSYS_BUILD_DYNINST=ON",
        "-DROCPROFSYS_BUILD_TBB=ON",
        "-DROCPROFSYS_BUILD_ELFUTILS=ON",
        "-DROCPROFSYS_BUILD_LIBIBERTY=ON",
        "-DROCPROFSYS_BUILD_CI=ON",
        "-DROCPROFSYS_BUILD_EXAMPLES=ON",
        "-DROCPROFSYS_BUILD_TESTING=ON",
        "-DROCPROFSYS_USE_PYTHON=ON",
        "-DROCPROFSYS_MAX_THREADS=64",
        f"-DROCPROFSYS_USE_MPI={'ON' if use_mpi else 'OFF'}",
    ]
    if disable_examples:
        cfg.append("-DROCPROFSYS_DISABLE_EXAMPLES=" + ";".join(disable_examples))
    if extra_cmake_args:
        cfg += extra_cmake_args
    log("Configuring full source build (this pulls/builds dyninst, elfutils, etc.)")
    run(cfg, env=build_env)
    log(f"Building rocprofiler-systems from source with {jobs} jobs (can take ~1-2h)")
    run([cmake, "--build", str(build_dir), "--parallel", str(jobs)], env=build_env)
    return build_dir


def build_examples(
    src_dir: Path,
    workdir: Path,
    rocm_dir: Path,
    env: dict,
    jobs: int,
    use_mpi: bool,
    disable_examples: list[str],
    extra_cmake_args: list[str] | None = None,
) -> None:
    """Configure/build the standalone examples and install into the ROCm prefix."""
    build_dir = workdir / "build-examples"
    cmake = shutil.which("cmake") or "cmake"
    cfg = [
        cmake,
        "-S",
        str(src_dir / "examples"),
        "-B",
        str(build_dir),
        "-DCMAKE_BUILD_TYPE=RelWithDebInfo",
        f"-DCMAKE_PREFIX_PATH={rocm_dir}",
        f"-DCMAKE_INSTALL_PREFIX={rocm_dir}",
        "-DROCPROFSYS_INSTALL_EXAMPLES=ON",
        f"-DROCPROFSYS_USE_MPI={'ON' if use_mpi else 'OFF'}",
    ]
    if disable_examples:
        # semicolon-separated CMake list; skips add_subdirectory() for these
        cfg.append("-DROCPROFSYS_DISABLE_EXAMPLES=" + ";".join(disable_examples))
        log("Skipping examples: " + ", ".join(disable_examples))
    if extra_cmake_args:
        cfg += extra_cmake_args
    run(cfg, env=env)
    run([cmake, "--build", str(build_dir), "--parallel", str(jobs)], env=env)
    run([cmake, "--install", str(build_dir)], env=env)

    examples_out = rocm_dir / "share" / "rocprofiler-systems" / "examples"
    n = len(list(examples_out.glob("*"))) if examples_out.is_dir() else 0
    log(f"Installed {n} example artifact(s) into {examples_out}")


def stage_tests(src_dir: Path, rocm_dir: Path, env: dict) -> Path:
    """Copy the pytest suite + helpers into the ROCm prefix (install-mode layout).

    Mirrors tests/CMakeLists.txt + tests/pytest/CMakeLists.txt copy rules.
    Returns the installed pytest directory.
    """
    tests_src = src_dir / "tests"
    tests_dst = rocm_dir / "share" / "rocprofiler-systems" / "tests"
    pytest_dst = tests_dst / "pytest"
    (pytest_dst / "rocprofsys").mkdir(parents=True, exist_ok=True)

    # pytest package + conftest + test_*.py
    pytest_src = tests_src / "pytest"
    for item in pytest_src.glob("test_*.py"):
        shutil.copy2(item, pytest_dst / item.name)
    shutil.copy2(pytest_src / "conftest.py", pytest_dst / "conftest.py")
    for item in (pytest_src / "rocprofsys").glob("*.py"):
        shutil.copy2(item, pytest_dst / "rocprofsys" / item.name)

    # top-level test helpers / validators / scripts
    top_level_files = [
        "check_amd_smi_metrics.py",
        "validate-causal-json.py",
        "validate-perfetto-proto.py",
        "validate-rocpd.py",
        "validate-timemory-json.py",
        "validate-unified-memory.py",
        "get_default_nic.sh",
        "generate_papi_nic_events.sh",
        "run_rocprofiler_systems.py",
        "test_categories.yaml",
        "README.md",
        "run_if_shmem_ok.sh",
        "shmem_validation_check.sh",
    ]
    for name in top_level_files:
        srcf = tests_src / name
        if srcf.is_file():
            shutil.copy2(srcf, tests_dst / name)

    # requirements.txt (some helpers reference it)
    req = src_dir / "requirements.txt"
    if req.is_file():
        shutil.copy2(req, tests_dst / "requirements.txt")

    # rocpd validation rule directory
    rules_src = tests_src / "rocpd-validation-rules"
    if rules_src.is_dir():
        rules_dst = tests_dst / "rocpd-validation-rules"
        if rules_dst.exists():
            shutil.rmtree(rules_dst)
        shutil.copytree(rules_src, rules_dst)

    # capability-check helper (standalone C++; needed by several tests)
    _build_capchk(tests_src, tests_dst, env)

    log(f"Staged test suite into {tests_dst}")
    return pytest_dst


def _build_capchk(tests_src: Path, tests_dst: Path, env: dict) -> None:
    capchk_cpp = tests_src / "rocprof-sys-capchk.cpp"
    if not capchk_cpp.is_file():
        return
    cxx = (
        env.get("CXX")
        or shutil.which("amdclang++", path=env.get("PATH"))
        or shutil.which("g++")
        or shutil.which("c++")
    )
    if not cxx:
        log("WARNING: no C++ compiler found; skipping rocprof-sys-capchk build.")
        return
    out = tests_dst / "rocprof-sys-capchk"
    run(
        [cxx, "-std=c++17", "-O2", str(capchk_cpp), "-o", str(out)],
        env=env,
        check=False,
    )


def tier_selection_args(tier: str) -> list[str]:
    """Return the pytest -m/-k selection args for a named tier.

    No 'gpu' marker filter is applied, so CPU-only install-mode tests (CLI help,
    config, presets, ...) run alongside the GPU tests. Known-problematic marker
    categories are excluded via ``-m "not ..."`` and known-flaky test names via
    ``-k "not (...)"`` (see LABEL_EXCLUDES / DEFAULT_DESELECT_KEYWORDS).
    """
    deselect = " or ".join(DEFAULT_DESELECT_KEYWORDS)
    label_excl = " and ".join(f"not {m}" for m in LABEL_EXCLUDES)
    if tier == "quick":
        include = " or ".join(QUICK_KEYWORDS)
        return ["-m", label_excl, "-k", f"({include}) and not ({deselect})"]
    if tier == "full":
        # everything that can run; only known-flaky test names are excluded
        return ["-k", f"not ({deselect})"]
    # standard (default): label-category excludes + known-flaky name excludes
    return ["-m", label_excl, "-k", f"not ({deselect})"]


def run_tests(
    venv_py: Path,
    pytest_dir: Path,
    rocm_dir: Path,
    env: dict,
    extra_pytest_args: str | None,
    tier: str,
    reruns: int,
    reruns_delay: int,
    rerun_failed: int,
    workdir: Path,
    facts: dict,
    mode: str = "install",
    test_root: Path | None = None,
) -> tuple[int, Path]:
    """Run the pytest suite against either the tarball binaries or a source build.

    ``mode`` is 'install' (test the tarball's shipped binaries; ``test_root`` is the
    ROCm prefix) or 'build' (test freshly-built binaries; ``test_root`` is the build
    dir). ROCm runtime always comes from ``rocm_dir``.

    Returns ``(returncode, final_junit_path)`` where the final JUnit reflects the
    last (re)run, so callers report counts/failures from the final state.
    """
    test_env = dict(env)
    test_env["ROCM_PATH"] = str(rocm_dir)
    test_env.setdefault("ROCPROFSYS_CI", "ON")
    root = str(test_root or rocm_dir)
    if mode == "build":
        # build mode: conftest discovers the build-tree binaries via this var.
        test_env["ROCPROFSYS_BUILD_DIR"] = root
    else:
        # install mode: point the suite at the tarball prefix (bin/rocprof-sys-*).
        test_env["ROCPROFSYS_INSTALL_DIR"] = root

    # Use a private temp dir inside the work dir. This keeps per-test output out
    # of the shared /tmp AND avoids the perfetto trace_processor collision: its
    # shell is extracted to "<tempdir>/trace_processor_python_api" (a fixed name),
    # so on a shared box a copy owned by another user makes chmod fail with EPERM.
    tmpdir = workdir / "tmp"
    tmpdir.mkdir(parents=True, exist_ok=True)
    for var in ("TMPDIR", "TMP", "TEMP"):
        test_env[var] = str(tmpdir)
    test_env.setdefault("ROCPROFSYS_TMPDIR", str(tmpdir / "rocprofsys"))
    # drop any stale perfetto shell we own so it is re-extracted under tmpdir
    stale = tmpdir / "trace_processor_python_api"
    if stale.exists():
        stale.unlink(missing_ok=True)
    log(f"Test TMPDIR: {tmpdir}")

    base = [str(venv_py), "-m", "pytest", str(pytest_dir), "-v", "-rA"]
    # A separate failed-rerun pass (--rerun-failed) needs pytest's last-failed
    # cache, so only disable the cache provider when that feature is off.
    if rerun_failed > 0:
        base += ["-o", f"cache_dir={workdir / '.pytest_cache'}"]
    else:
        base += ["-p", "no:cacheprovider"]

    selection = (
        extra_pytest_args.split() if extra_pytest_args else tier_selection_args(tier)
    )
    if reruns > 0:
        selection += ["--reruns", str(reruns), "--reruns-delay", str(reruns_delay)]

    junit = workdir / "pytest-results.xml"
    log(
        "Test selection: "
        + ("(custom) " + extra_pytest_args if extra_pytest_args else "tier=" + tier)
        + (f", reruns={reruns}" if reruns > 0 else "")
        + (f", rerun-failed={rerun_failed}" if rerun_failed > 0 else "")
    )
    log(f"JUnit results -> {junit}")
    cmd = base + selection + [f"--junitxml={junit}"]
    rc = run([str(c) for c in cmd], cwd=str(pytest_dir), env=test_env, check=False)

    final_junit = junit
    # Rerun only the failed tests, up to N times, each into its own JUnit artifact.
    for attempt in range(1, rerun_failed + 1):
        if rc == 0:
            break
        rerun_junit = workdir / f"pytest-rerun-{attempt}.xml"
        step(f"Re-running failed tests (attempt {attempt}/{rerun_failed})")
        cmd = base + ["--last-failed", f"--junitxml={rerun_junit}"]
        rc = run([str(c) for c in cmd], cwd=str(pytest_dir), env=test_env, check=False)
        final_junit = rerun_junit
        facts["rerun_failed_attempts"] = attempt
        if rc == 0:
            log(f"All previously-failed tests passed on rerun attempt {attempt} (flaky).")

    return rc, final_junit


def collect_failed_tests(junit: Path) -> list[tuple[str, str, str]]:
    """Return [(test_id, message, detail_text)] for failures/errors in the JUnit XML."""
    if not junit.is_file():
        return []
    try:
        import xml.etree.ElementTree as ET

        root = ET.parse(junit).getroot()
    except Exception:  # noqa: BLE001
        return []
    failed = []
    for tc in root.iter("testcase"):
        problems = tc.findall("failure") + tc.findall("error")
        if not problems:
            continue
        cls = tc.get("classname", "")
        name = tc.get("name", "")
        test_id = f"{cls}::{name}" if cls else name
        msg = (problems[0].get("message") or "").strip()
        detail = (problems[0].text or "").strip()
        failed.append((test_id, msg, detail))
    return failed


def write_failures_log(path: Path, failed: list[tuple[str, str, str]]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"Failed / errored tests: {len(failed)}\n\n")
        for test_id, msg, detail in failed:
            fh.write("=" * 72 + "\n" + test_id + "\n" + "-" * 72 + "\n")
            if msg:
                fh.write(f"message: {msg}\n")
            if detail:
                fh.write(detail + "\n")
            fh.write("\n")


def clean_previous_logs(workdir: Path, keep_stamp: str | None = None) -> None:
    """Delete log/report artifacts from previous runs (keeps rocm/, source, venv,
    and build dirs). Files matching the current run's ``keep_stamp`` are preserved."""
    patterns = [
        "detailed-*.log",
        "summary-*.log",
        "RESULT-*.md",
        "failures-*.log",
        "rocm-sanity-*.log",
        "pip-freeze-*.txt",
        "logs-*.tar.gz",
        "pytest-rerun-*.xml",
        "pytest-results.xml",
        "latest-*.txt",
    ]
    removed = 0
    for pattern in patterns:
        for f in workdir.glob(pattern):
            if keep_stamp and keep_stamp in f.name:
                continue
            try:
                f.unlink()
                removed += 1
            except OSError:
                pass
    log(f"--clean: removed {removed} log/report artifact(s) from previous runs")


def parse_junit(junit: Path) -> dict | None:
    """Return aggregate pytest counts from the JUnit XML, or None if unavailable."""
    if not junit.is_file():
        return None
    try:
        import xml.etree.ElementTree as ET

        root = ET.parse(junit).getroot()
        suites = root.findall("testsuite") or ([root] if root.tag == "testsuite" else [])
        agg = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0, "time": 0.0}
        for s in suites:
            for k in ("tests", "failures", "errors", "skipped"):
                agg[k] += int(s.get(k, 0) or 0)
            agg["time"] += float(s.get("time", 0) or 0)
        agg["passed"] = agg["tests"] - agg["failures"] - agg["errors"] - agg["skipped"]
        return agg
    except Exception as exc:  # noqa: BLE001
        log(f"WARNING: could not parse {junit}: {exc}")
        return None


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--variant",
        default="multiarch",
        help="Tarball GPU variant (default: multiarch). Use 'auto' to pick the "
        "smallest per-family tarball for the detected GPU. Examples: "
        "multiarch, gfx94X-dcgpu (MI300), gfx950-dcgpu, gfx90a, gfx110X-all.",
    )
    p.add_argument(
        "--rocm-version",
        default=None,
        help="Specific nightly version to use (e.g. 7.15.0a20260717 or 20260717). "
        "Default: the latest available for the variant.",
    )
    p.add_argument(
        "--sha256",
        default=None,
        help="Expected SHA-256 of the ROCm tarball. When set, the download is "
        "verified against it before extraction (recommended for pinned runs).",
    )
    p.add_argument(
        "--require-checksum",
        action="store_true",
        help="Abort if no SHA-256 is available (via --sha256 or published next to "
        "the tarball) instead of only warning.",
    )
    p.add_argument(
        "--branch",
        default="develop",
        help="rocm-systems branch to test from (default: develop).",
    )
    p.add_argument(
        "--work-dir",
        default=None,
        help="Override the working directory (default: " "./rocprofiler-systems-tests).",
    )
    p.add_argument(
        "--jobs",
        "-j",
        type=int,
        default=os.cpu_count() or 8,
        help="Parallel build jobs (default: nproc).",
    )
    p.add_argument(
        "--build-mpi-examples",
        action="store_true",
        help="Build MPI-enabled examples (requires libopenmpi-dev). Off by default.",
    )
    p.add_argument(
        "--disable-examples",
        default="lulesh",
        help="Comma-separated example directories to skip building (default: "
        "lulesh, which vendors Kokkos and requires C++20 in the standalone "
        "build). Pass '' to build everything.",
    )
    p.add_argument(
        "--install-system-deps",
        action="store_true",
        help="Best-effort apt-get install of build/runtime deps (needs sudo/root).",
    )
    p.add_argument(
        "--tier",
        default="standard",
        choices=("quick", "standard", "full"),
        help="Test tier to run (default: standard). 'quick' is a fast smoke subset, "
        "'standard' runs the suite minus known-flaky tests and excluded label "
        "categories (mpi, network, ...), 'full' runs everything that can run. "
        "GPU and CPU-only install-mode tests are both included. Ignored when "
        "--pytest-args is given.",
    )
    p.add_argument(
        "--reruns",
        type=int,
        default=0,
        help="Re-run each failing test up to N times before marking it failed "
        "(needs pytest-rerunfailures; default: 0).",
    )
    p.add_argument(
        "--reruns-delay",
        type=int,
        default=5,
        help="Seconds to wait between reruns (default: 5).",
    )
    p.add_argument(
        "--rerun-failed",
        type=int,
        default=0,
        help="After the main run, re-run only the failed tests up to N times, each "
        "into its own JUnit artifact (pytest-rerun-<n>.xml). Distinguishes flaky "
        "tests from hard failures (default: 0).",
    )
    p.add_argument(
        "--min-free-gb",
        type=int,
        default=DEFAULT_MIN_FREE_GB,
        help=f"Minimum free disk space (GB) required before a download "
        f"(default: {DEFAULT_MIN_FREE_GB}).",
    )
    p.add_argument(
        "--pytest-args",
        default=None,
        help="Override the pytest selection/args entirely (quoted). Takes precedence "
        "over --tier.",
    )
    p.add_argument(
        "--skip-download",
        action="store_true",
        help="Reuse the already-extracted ROCm tree in the work dir; never check "
        "for a newer nightly.",
    )
    p.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Force rebuilding the examples even when the source and ROCm tarball "
        "are unchanged.",
    )
    p.add_argument(
        "--skip-tests",
        action="store_true",
        help="Do everything except run the pytest suite.",
    )
    p.add_argument(
        "--clean",
        action="store_true",
        help="Delete log/report artifacts from previous runs (detailed/summary/"
        "RESULT/failures/sanity/pip-freeze/junit/archives/latest-*) before starting. "
        "Preserves the extracted ROCm tree, source checkout, venv, and build dirs.",
    )
    p.add_argument(
        "--build-from-source",
        action="store_true",
        help="Instead of testing the tarball's shipped binaries, build the full "
        "rocprofiler-systems from source (using the tarball for ROCm runtime + "
        "toolchain) and validate the freshly-built build-tree binaries. Pulls "
        "submodules and takes ~1-2h. Mirrors the QA build+CTest flow.",
    )
    p.add_argument(
        "--hpc-cray",
        action="store_true",
        help="Bundle Cray-MPICH-on-gfx90a HPC specifics: inject Cray MPICH + XPMEM + "
        "GTL build flags (so GPU-aware MPI examples like hpc/jacobi-hip build with "
        "the tarball clang), default the tarball variant to gfx90a, skip examples "
        "that don't build on the tarball (jpegdecode/videodecode), and enable "
        "GPU-aware MPI at runtime. Requires 'module load cray-mpich'.",
    )
    # ---- offline / two-phase (air-gapped compute node) workflow ----------- #
    p.add_argument(
        "--prepare-only",
        action="store_true",
        help="Phase 1 (run on a networked node, e.g. the login node): download + "
        "extract the tarball, clone the source, and create the venv, then STOP "
        "before building/testing. Nothing GPU-specific is done.",
    )
    p.add_argument(
        "--offline",
        action="store_true",
        help="Phase 2 (run on an air-gapped GPU compute node): do no network I/O at "
        "all (no tarball download, no git fetch, no pip). Reuses the "
        "tarball/source/venv staged by an earlier --prepare-only run.",
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    # --offline is the single air-gapped switch: it drives the internal "reuse the
    # already-staged tarball/source/venv without any network access" behavior.
    args.skip_clone = args.offline
    args.skip_pip = args.offline
    if args.offline:
        args.skip_download = True

    if args.work_dir:
        workdir = Path(args.work_dir).resolve()
    else:
        workdir = Path.cwd() / "rocprofiler-systems-tests"
    workdir.mkdir(parents=True, exist_ok=True)
    rocm_dir = workdir / "rocm"

    global _SUMMARY_PATH
    started = _dt.datetime.now()
    run_stamp = started.strftime("%Y%m%d-%H%M%S")
    detail_log = workdir / f"detailed-{run_stamp}.log"
    summary_log = workdir / f"summary-{run_stamp}.log"
    _SUMMARY_PATH = summary_log
    open_detailed_log(detail_log)

    if args.clean:
        clean_previous_logs(workdir, keep_stamp=run_stamp)

    facts = _FACTS
    facts.update(
        {
            "started": started.isoformat(timespec="seconds"),
            "command": " ".join(sys.argv),
            "host": socket.gethostname(),
            "work_dir": str(workdir),
            "rocm_prefix": str(rocm_dir),
            "variant": args.variant,
            "branch": args.branch,
            "detailed_log": str(detail_log),
        }
    )
    # scheduler / GPU-visibility context (useful on clusters); only record what's set
    sched = {
        k: os.environ[k]
        for k in (
            "SLURM_JOB_ID",
            "SLURM_JOB_NODELIST",
            "SLURM_JOB_GPUS",
            "ROCR_VISIBLE_DEVICES",
            "HIP_VISIBLE_DEVICES",
        )
        if os.environ.get(k)
    }
    if sched:
        facts["scheduler"] = "; ".join(f"{k}={v}" for k, v in sched.items())

    log(f"Working directory: {workdir}")
    log(f"ROCm prefix (ROCM_PATH): {rocm_dir}")
    log(f"Detailed log: {detail_log}")
    log(f"Summary log:  {summary_log}")

    # HPC Cray MPICH bundle: mutates args (variant/disable_examples) + os.environ
    # and returns extra CMake args (MPI compile/link flags) for the example/source
    # build.
    hpc_cray_cmake_args: list[str] = []
    if args.hpc_cray:
        hpc_cray_cmake_args = apply_hpc_cray_settings(args, facts)

    # ---- 0. Preflight ----------------------------------------------------- #
    archs = preflight(args, workdir, rocm_dir)
    facts["gpu_arch"] = ", ".join(archs) or "none"
    variant = resolve_variant(args.variant, archs)
    facts["variant"] = variant
    facts["tier"] = args.tier
    if args.reruns > 0:
        facts["reruns"] = args.reruns

    facts["mode"] = (
        "prepare-only" if args.prepare_only else ("offline" if args.offline else "normal")
    )

    # ---- 1. Resolve + download + extract the nightly ROCm tarball ---------- #
    step("Download + extract nightly ROCm tarball")
    rocm_updated = False
    current = current_rocm_tarball(workdir)
    if args.skip_download and not (rocm_dir / "bin").is_dir():
        die(
            "--skip-download/--offline was given but there is no extracted ROCm "
            f"tree at {rocm_dir}.\n"
            "       Run once with --prepare-only on a networked node first."
        )
    if args.skip_download and (rocm_dir / "bin").is_dir():
        log(f"--skip-download: reusing existing ROCm tree ({current or 'unknown'}).")
        facts["tarball"] = current or "unknown"
    else:
        filename, url = resolve_tarball(variant, args.rocm_version)
        facts["tarball"] = filename
        facts["tarball_url"] = url
        vm = re.search(r"-((\d+\.\d+\.\d+)a\d{8})\.tar\.gz$", filename)
        facts["rocm_version"] = vm.group(1) if vm else "unknown"
        if (rocm_dir / "bin").is_dir() and current == filename:
            log(f"ROCm tarball already current ({filename}); reusing extracted tree.")
        else:
            log(f"Selected tarball: {filename}")
            log(f"URL: {url}")
            if current and current != filename:
                log(f"Newer nightly available: {current} -> {filename}")
            tarball = workdir / filename
            download_file(url, tarball)
            verify_tarball(url, tarball, args.sha256, args.require_checksum, facts)
            extract_tarball(tarball, rocm_dir)
            set_rocm_tarball(workdir, filename)
            # the archive is now extracted into rocm_dir; drop every downloaded
            # tarball (including this one) to reclaim the multi-GB of disk space.
            cleanup_downloaded_tarballs(workdir)
            rocm_updated = True
    facts["rocm_updated"] = rocm_updated

    # sanity: the profiler binaries must be present in the tarball's bin/
    missing = [
        b for b in REQUIRED_ROCPROFSYS_BINARIES if not (rocm_dir / "bin" / b).exists()
    ]
    if missing:
        die(
            "the downloaded tarball does not contain the expected rocprof-sys "
            f"binaries in {rocm_dir / 'bin'}: {', '.join(missing)}.\n"
            "       Make sure you are using a full dist tarball (not the "
            "'-tests-' variant)."
        )
    log(
        "Verified rocprof-sys binaries present in tarball bin/: "
        + ", ".join(REQUIRED_ROCPROFSYS_BINARIES)
    )

    env = make_rocm_env(os.environ.copy(), rocm_dir)
    report_under_test(rocm_dir, env, facts)
    capture_rocm_sanity(rocm_dir, env, workdir / f"rocm-sanity-{run_stamp}.log", facts)

    # Fail fast if the tarball binaries can't start on this machine. Skipped when
    # building from source (those tarball binaries aren't what's tested) and during
    # prepare-only (which may run on a different node than the tests).
    if not args.build_from_source and not args.prepare_only:
        smoke_check_rocprofsys(rocm_dir, env)

    facts["validation"] = (
        "build-from-source" if args.build_from_source else "tarball-install"
    )

    # ---- 2. Sparse-clone / update develop --------------------------------- #
    step("Sparse-clone / update rocm-systems (projects/rocprofiler-systems)")
    src_dir, source_changed = sync_source(
        workdir,
        args.branch,
        env,
        no_fetch=args.skip_clone,
        submodules=args.build_from_source,
    )
    repo_dir = workdir / "rocm-systems"
    facts["git_revision"] = _git_rev(repo_dir) or "unknown"
    facts["git_subject"] = (
        subprocess.run(
            ["git", "-C", str(repo_dir), "log", "-1", "--format=%s"],
            capture_output=True,
            text=True,
        ).stdout.strip()
        or "unknown"
    )
    facts["source_changed"] = source_changed

    # ---- 3. Required installations for the tests -------------------------- #
    step("Install test dependencies")
    if args.install_system_deps:
        install_system_deps()
    venv_py = make_venv(workdir, src_dir, skip_install=args.skip_pip)
    capture_pip_freeze(venv_py, workdir / f"pip-freeze-{run_stamp}.txt", facts)

    # ---- Phase 1 stop: prepared for a later offline compute-node run ------ #
    if args.prepare_only:
        step("Prepare-only complete (--prepare-only)")
        log("Network prep done. The tarball, source, and venv are staged under:")
        log(f"  {workdir}")
        log("Now run the tests on the (air-gapped) GPU compute node with:")
        log(
            f"  python3 {Path(sys.argv[0]).name} --work-dir {workdir} --offline "
            f"--tier {args.tier}"
        )
        facts["result"] = "PREPARED (--prepare-only)"
        write_summary(summary_log, facts)
        return 0

    # ---- 4. Build (source or examples) + prepare the test tree ------------ #
    if args.build_from_source:
        step("Build rocprofiler-systems from source")
        _src_disable = [e.strip() for e in args.disable_examples.split(",") if e.strip()]
        build_dir = build_from_source(
            src_dir,
            workdir,
            rocm_dir,
            env,
            venv_py,
            args.jobs,
            args.build_mpi_examples,
            disable_examples=_src_disable,
            extra_cmake_args=hpc_cray_cmake_args,
        )
        pytest_dir = build_dir / "share" / "rocprofiler-systems" / "tests" / "pytest"
        if not (pytest_dir / "conftest.py").is_file():
            die(f"source build did not produce a pytest tree at {pytest_dir}")
        test_mode = "build"
        test_root = build_dir
        facts["build_dir"] = str(build_dir)
        # report the freshly-built rocprof-sys version (that's what's under test)
        built_avail = build_dir / "bin" / "rocprof-sys-avail"
        if built_avail.exists():
            facts["rocprofsys_version"] = _tool_version(str(built_avail), env)
        log(f"Testing build-tree binaries in: {build_dir / 'bin'}")
    else:
        step("Build examples + stage test suite into the ROCm prefix")
        disable_examples = [
            e.strip() for e in args.disable_examples.split(",") if e.strip()
        ]
        facts["disabled_examples"] = ", ".join(disable_examples) or "(none)"

        pytest_dir = rocm_dir / "share" / "rocprofiler-systems" / "tests" / "pytest"
        examples_out = rocm_dir / "share" / "rocprofiler-systems" / "examples"
        examples_present = examples_out.is_dir() and any(examples_out.iterdir())

        # Rebuild only when the inputs changed: fresh/updated ROCm tree (examples
        # were installed into the tree that was just replaced), updated source,
        # missing example artifacts, or an explicit --force-rebuild.
        need_build = (
            args.force_rebuild or rocm_updated or source_changed or not examples_present
        )
        if need_build:
            reasons = []
            if args.force_rebuild:
                reasons.append("--force-rebuild")
            if rocm_updated:
                reasons.append("rocm updated")
            if source_changed:
                reasons.append("source changed")
            if not examples_present:
                reasons.append("examples missing")
            log("Building examples (" + ", ".join(reasons) + ")")
            build_examples(
                src_dir,
                workdir,
                rocm_dir,
                env,
                args.jobs,
                args.build_mpi_examples,
                disable_examples,
                extra_cmake_args=hpc_cray_cmake_args,
            )
        else:
            log(f"Examples up to date ({facts['git_revision'][:10]}); skipping rebuild.")

        # Staging the pytest tree is cheap; refresh it whenever we rebuilt, the ROCm
        # tree changed, or the suite isn't staged yet.
        if need_build or not (pytest_dir / "conftest.py").is_file():
            pytest_dir = stage_tests(src_dir, rocm_dir, env)
        else:
            log("Test suite already staged; skipping.")

        test_mode = "install"
        test_root = rocm_dir
        facts["examples_rebuilt"] = need_build
        facts["examples_installed"] = (
            len(list(examples_out.glob("*"))) if examples_out.is_dir() else 0
        )

    # ---- 5. Run the tests ------------------------------------------------- #
    if args.skip_tests:
        step("Skipping test execution (--skip-tests)")
        var = "ROCPROFSYS_BUILD_DIR" if test_mode == "build" else "ROCPROFSYS_INSTALL_DIR"
        log(f"Everything is staged under: {test_root}")
        log("To run manually:")
        log(f"  {var}={test_root} ROCM_PATH={rocm_dir} \\")
        log(f"  {venv_py} -m pytest {pytest_dir} -v")
        facts["result"] = "SKIPPED (--skip-tests)"
        write_summary(summary_log, facts)
        return 0

    step(
        f"Run pytest suite in {test_mode} mode "
        f"({'build-tree' if test_mode == 'build' else 'tarball'} rocprof-sys binaries)"
    )
    rc, junit = run_tests(
        venv_py,
        pytest_dir,
        rocm_dir,
        env,
        args.pytest_args,
        args.tier,
        args.reruns,
        args.reruns_delay,
        args.rerun_failed,
        workdir,
        facts,
        mode=test_mode,
        test_root=test_root,
    )

    counts = parse_junit(junit)
    facts["pytest_exit"] = rc
    facts["result"] = "PASS" if rc == 0 else "FAIL"
    if counts:
        facts["tests_total"] = counts["tests"]
        facts["passed"] = counts["passed"]
        facts["failed"] = counts["failures"]
        facts["errors"] = counts["errors"]
        facts["skipped"] = counts["skipped"]
        facts["duration_sec"] = round(counts["time"], 1)

    failed = collect_failed_tests(junit)
    if failed:
        failures_log = workdir / f"failures-{run_stamp}.log"
        write_failures_log(failures_log, failed)
        facts["failures_log"] = str(failures_log)
        facts["_failed_tests"] = [tid for tid, _, _ in failed]

    step("Summary")
    write_summary(summary_log, facts)
    return rc


def write_summary(summary_log: Path, facts: dict) -> None:
    """Write the concise summary to a file and echo it to the console/detailed log."""
    facts["finished"] = _dt.datetime.now().isoformat(timespec="seconds")
    order = [
        "result",
        "mode",
        "validation",
        "error",
        "pytest_exit",
        "tests_total",
        "passed",
        "failed",
        "errors",
        "skipped",
        "duration_sec",
        "tier",
        "reruns",
        "rerun_failed_attempts",
        "hpc_cray",
        "mpich_dir",
        "variant",
        "gpu_arch",
        "host",
        "scheduler",
        "tarball",
        "rocm_version",
        "rocprofsys_version",
        "sha256_verified",
        "sha256",
        "rocm_updated",
        "tarball_url",
        "branch",
        "git_revision",
        "git_subject",
        "source_changed",
        "examples_rebuilt",
        "examples_installed",
        "disabled_examples",
        "build_dir",
        "work_dir",
        "rocm_prefix",
        "detailed_log",
        "rocm_sanity_log",
        "pip_freeze_log",
        "failures_log",
        "started",
        "finished",
        "command",
    ]
    lines = ["rocprof-sys nightly tarball test - SUMMARY", "=" * 44]
    for key in order:
        if key in facts:
            lines.append(f"{key:22s}: {facts[key]}")

    failed_tests = facts.get("_failed_tests") or []
    if failed_tests:
        lines.append("")
        lines.append(f"failed tests ({len(failed_tests)}):")
        for name in failed_tests[:50]:
            lines.append(f"  - {name}")
        if len(failed_tests) > 50:
            lines.append(f"  ... and {len(failed_tests) - 50} more (see failures log)")

    if _STEP_TIMES:
        lines.append("")
        lines.append("step timings:")
        for title, dur in _STEP_TIMES:
            lines.append(f"  {_fmt_dur(dur):>9}  {title}")
        lines.append(f"  {_fmt_dur(time.monotonic() - _RUN_START):>9}  TOTAL wall-clock")

    text = "\n".join(lines)

    with open(summary_log, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")

    _emit("\n" + text)
    _emit(f"\n[nightly-test] Summary written to: {summary_log}")

    # Gap 8: human-readable Markdown result. Gap 7: archive logs + latest pointers.
    write_result_md(summary_log, facts)
    archive_logs(summary_log, facts)


def write_result_md(summary_log: Path, facts: dict) -> None:
    """Write a Markdown RESULT file (mirrors the QA RESULT_*.md) next to the logs."""
    workdir = summary_log.parent
    stamp = summary_log.stem.replace("summary-", "")
    md = workdir / f"RESULT-{stamp}.md"

    def g(key, default="-"):
        return facts.get(key, default)

    lines = [
        "# rocprofiler-systems tarball validation",
        "",
        f"- Result: **{g('result')}**",
        f"- Mode: {g('mode')}",
        f"- Finished (UTC-local): {g('finished')}",
        f"- Host: {g('host')}",
        f"- Scheduler: {g('scheduler')}",
        f"- Work dir: {g('work_dir')}",
        "",
        "## Under test",
        f"- Tarball: {g('tarball')}",
        f"- ROCm version: {g('rocm_version')}",
        f"- rocprof-sys version: {g('rocprofsys_version')}",
        f"- SHA-256 verified: {g('sha256_verified')}",
        f"- Source branch: {g('branch')}",
        f"- Source commit: {g('git_revision')}  {g('git_subject', '')}",
        "",
        "## Test results",
        f"- pytest exit: {g('pytest_exit')}",
        f"- total / passed / failed / errors / skipped: "
        f"{g('tests_total')} / {g('passed')} / {g('failed')} / "
        f"{g('errors')} / {g('skipped')}",
        f"- tier: {g('tier')}",
        f"- reruns (inline): {g('reruns', 0)}",
        f"- failed-rerun attempts: {g('rerun_failed_attempts', 0)}",
    ]
    failed_tests = facts.get("_failed_tests") or []
    if failed_tests:
        lines += ["", "## Failed tests"]
        lines += [f"- {t}" for t in failed_tests[:100]]
        if len(failed_tests) > 100:
            lines.append(f"- ... and {len(failed_tests) - 100} more")

    if _STEP_TIMES:
        lines += ["", "## Stage timings"]
        lines += [f"- {title}: {_fmt_dur(dur)}" for title, dur in _STEP_TIMES]

    lines += [
        "",
        "## Logs",
        f"- Detailed: {g('detailed_log')}",
        f"- Summary: {summary_log}",
        f"- ROCm sanity: {g('rocm_sanity_log')}",
        f"- pip freeze: {g('pip_freeze_log')}",
        f"- Failures: {g('failures_log')}",
    ]
    md.write_text("\n".join(lines) + "\n")
    facts["result_md"] = str(md)
    log(f"RESULT written to: {md}")


def archive_logs(summary_log: Path, facts: dict) -> None:
    """Bundle this run's log artifacts into a tar.gz and update 'latest' pointers."""
    workdir = summary_log.parent
    stamp = summary_log.stem.replace("summary-", "")
    members = [
        workdir / f"detailed-{stamp}.log",
        workdir / f"summary-{stamp}.log",
        workdir / f"RESULT-{stamp}.md",
        workdir / f"rocm-sanity-{stamp}.log",
        workdir / f"pip-freeze-{stamp}.txt",
        workdir / f"failures-{stamp}.log",
        workdir / "pytest-results.xml",
    ] + sorted(workdir.glob("pytest-rerun-*.xml"))

    archive = workdir / f"logs-{stamp}.tar.gz"
    try:
        with tarfile.open(archive, "w:gz") as tf:
            for m in members:
                if m.is_file():
                    tf.add(m, arcname=m.name)
    except Exception as exc:  # noqa: BLE001
        log(f"WARNING: could not create log archive: {exc}")
        return

    # 'latest' pointers so automation/QA can find the newest run at a fixed path.
    (workdir / "latest-summary.txt").write_text(str(summary_log) + "\n")
    (workdir / "latest-archive.txt").write_text(str(archive) + "\n")
    if facts.get("result_md"):
        (workdir / "latest-result.txt").write_text(str(facts["result_md"]) + "\n")
    facts["log_archive"] = str(archive)
    log(f"Log archive: {archive}")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        die("interrupted", 130)
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        import traceback

        _emit(traceback.format_exc(), stream=sys.stderr)
        die(f"unexpected error: {exc}")
