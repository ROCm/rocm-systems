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

Each run writes a detailed log, a summary log (with per-step timings, test counts,
failed-test names, and the rocprof-sys version under test), and, on failure, a
failures log listing each failing test with its output.

Run ``--help`` for the full list of options.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import shutil
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
    interactive = sys.stdout.isatty()
    if wget:
        # Avoid the noisy default "dot" meter. On a TTY show a single in-place
        # updating bar; in non-interactive logs stay quiet (one line at the end).
        progress = (
            ["-q", "--show-progress", "--progress=bar:force:noscroll"]
            if interactive
            else ["-nv"]
        )
        cmd = [wget, "--continue", "--tries=3", *progress, "-O", str(tmp), url]
    elif curl:
        # curl's default meter already updates in place (no per-line spam).
        cmd = [curl, "-fL", "--retry", "3", "-C", "-", "-o", str(tmp), url]
    else:
        log("wget/curl not found; falling back to urllib (no resume).")
        with urllib.request.urlopen(url, timeout=120) as resp, open(
            tmp, "wb"
        ) as fh:  # noqa: S310
            shutil.copyfileobj(resp, fh)
        tmp.rename(dest)
        return

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
            try:
                # PEP 706 'data' filter (Python 3.8.17+/3.9.17+/3.10.12+/3.12+)
                # blocks path traversal, absolute paths and unsafe links.
                tf.extractall(rocm_dir, filter="data")  # noqa: S202
            except TypeError:
                _safe_extractall(tf, rocm_dir)


def _within(base: Path, target: str) -> bool:
    base_r = os.path.realpath(base)
    target_r = os.path.realpath(target)
    return target_r == base_r or target_r.startswith(base_r + os.sep)


def _safe_extractall(tf: tarfile.TarFile, dest: Path) -> None:
    """Path-traversal-safe extraction fallback for old Python without PEP 706."""
    for member in tf.getmembers():
        member_path = os.path.join(str(dest), member.name)
        if not _within(dest, member_path):
            die(f"unsafe path in archive (path traversal blocked): {member.name}")
        if member.issym() or member.islnk():
            link_target = os.path.join(os.path.dirname(member_path), member.linkname)
            if not _within(dest, link_target):
                die(f"unsafe link in archive blocked: {member.name} -> {member.linkname}")
    tf.extractall(dest)  # noqa: S202


# marker file recording which tarball is currently extracted into rocm_dir
def _rocm_marker(workdir: Path) -> Path:
    return workdir / ".rocm-version"


def current_rocm_tarball(workdir: Path) -> str | None:
    marker = _rocm_marker(workdir)
    return marker.read_text().strip() if marker.is_file() else None


def set_rocm_tarball(workdir: Path, filename: str) -> None:
    _rocm_marker(workdir).write_text(filename + "\n")


def cleanup_old_tarballs(workdir: Path, keep: str | None) -> None:
    """Delete downloaded dist tarballs to save space.

    Pass ``keep=None`` to remove every downloaded archive (used after a successful
    extraction, since the extracted ``rocm/`` tree is all we need), or a filename
    to preserve just that one.
    """
    preserve = {keep, f"{keep}.part"} if keep else set()
    for tb in workdir.glob("therock-dist-*.tar.gz*"):
        if tb.name not in preserve:
            log(f"Removing extracted tarball to reclaim space: {tb.name}")
            tb.unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
# Environment for building / running against the tarball
# --------------------------------------------------------------------------- #


def make_rocm_env(base_env: dict, rocm_dir: Path) -> dict:
    env = dict(base_env)
    env["ROCM_PATH"] = str(rocm_dir)
    env["ROCPROFSYS_CI"] = "ON"

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
    elif not args.skip_tests:
        log(
            "WARNING: no GPU detected via rocminfo. The default test selection is "
            "GPU-only ('-m gpu') and will mostly fail/skip. Continue anyway..."
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


# --------------------------------------------------------------------------- #
# Steps
# --------------------------------------------------------------------------- #


def _git_rev(repo_dir: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    ).stdout.strip()


def sync_source(workdir: Path, branch: str, env: dict) -> tuple[Path, bool]:
    """Sparse-clone (or update) rocm-systems.

    Returns ``(source_dir, source_changed)`` where ``source_changed`` is True when
    a fresh clone happened or the branch HEAD moved since the last run.
    """
    repo_dir = workdir / "rocm-systems"
    src_dir = repo_dir / PROJECT_SUBDIR

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

    log(f"Checked out {branch} @ {_git_rev(repo_dir)[:10]}")
    return src_dir, True


def make_venv(workdir: Path, src_dir: Path) -> Path:
    """Create a venv and install requirements.txt. Returns the venv python path."""
    venv_dir = workdir / "venv"
    py = venv_dir / "bin" / "python"
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


def build_examples(
    src_dir: Path,
    workdir: Path,
    rocm_dir: Path,
    env: dict,
    jobs: int,
    use_mpi: bool,
    disable_examples: list[str],
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
    """Return the pytest -m/-k selection args for a named tier."""
    deselect = " or ".join(DEFAULT_DESELECT_KEYWORDS)
    if tier == "quick":
        include = " or ".join(QUICK_KEYWORDS)
        return ["-m", "gpu", "-k", f"({include}) and not ({deselect})"]
    if tier == "full":
        return ["-m", "gpu"]
    # standard (default)
    return ["-m", "gpu", "-k", f"not ({deselect})"]


def run_tests(
    venv_py: Path,
    pytest_dir: Path,
    rocm_dir: Path,
    env: dict,
    extra_pytest_args: str | None,
    tier: str,
    reruns: int,
    reruns_delay: int,
    workdir: Path,
) -> int:
    """Run the pytest suite in install mode against the tarball binaries."""
    test_env = dict(env)
    # install mode: point the suite at the tarball prefix (bin/rocprof-sys-*).
    test_env["ROCPROFSYS_INSTALL_DIR"] = str(rocm_dir)
    test_env["ROCM_PATH"] = str(rocm_dir)
    test_env.setdefault("ROCPROFSYS_CI", "ON")

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

    cmd = [
        str(venv_py),
        "-m",
        "pytest",
        str(pytest_dir),
        "-p",
        "no:cacheprovider",
        "-v",
        "-rA",  # summary of all outcomes at the end (also lands in the log)
    ]
    if extra_pytest_args:
        cmd += extra_pytest_args.split()
    else:
        cmd += tier_selection_args(tier)

    if reruns > 0:
        cmd += ["--reruns", str(reruns), "--reruns-delay", str(reruns_delay)]

    junit = workdir / "pytest-results.xml"
    cmd += [f"--junitxml={junit}"]

    log(
        f"Test selection: {'(custom) ' + extra_pytest_args if extra_pytest_args else 'tier=' + tier}"
        + (f", reruns={reruns}" if reruns > 0 else "")
    )
    log(f"JUnit results -> {junit}")
    rc = run([str(c) for c in cmd], cwd=str(pytest_dir), env=test_env, check=False)
    return rc


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
        help="Test tier to run (default: standard). 'quick' is a fast smoke subset "
        "(~min), 'standard' is the full GPU suite minus known-flaky tests, "
        "'full' runs everything GPU. Ignored when --pytest-args is given.",
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
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

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

    facts = _FACTS
    facts.update(
        {
            "started": started.isoformat(timespec="seconds"),
            "command": " ".join(sys.argv),
            "work_dir": str(workdir),
            "rocm_prefix": str(rocm_dir),
            "variant": args.variant,
            "branch": args.branch,
            "detailed_log": str(detail_log),
        }
    )

    log(f"Working directory: {workdir}")
    log(f"ROCm prefix (ROCM_PATH): {rocm_dir}")
    log(f"Detailed log: {detail_log}")
    log(f"Summary log:  {summary_log}")

    # ---- 0. Preflight ----------------------------------------------------- #
    archs = preflight(args, workdir, rocm_dir)
    facts["gpu_arch"] = ", ".join(archs) or "none"
    variant = resolve_variant(args.variant, archs)
    facts["variant"] = variant
    facts["tier"] = args.tier
    if args.reruns > 0:
        facts["reruns"] = args.reruns

    # ---- 1. Resolve + download + extract the nightly ROCm tarball ---------- #
    step("Download + extract nightly ROCm tarball")
    rocm_updated = False
    current = current_rocm_tarball(workdir)
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
            cleanup_old_tarballs(workdir, keep=None)
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

    # ---- 2. Sparse-clone / update develop --------------------------------- #
    step("Sparse-clone / update rocm-systems (projects/rocprofiler-systems)")
    src_dir, source_changed = sync_source(workdir, args.branch, env)
    facts["git_revision"] = _git_rev(workdir / "rocm-systems") or "unknown"
    facts["source_changed"] = source_changed

    # ---- 3. Required installations for the tests -------------------------- #
    step("Install test dependencies")
    if args.install_system_deps:
        install_system_deps()
    venv_py = make_venv(workdir, src_dir)

    # ---- 4. Build examples + stage tests into the ROCm prefix ------------- #
    step("Build examples + stage test suite into the ROCm prefix")
    disable_examples = [e.strip() for e in args.disable_examples.split(",") if e.strip()]
    facts["disabled_examples"] = ", ".join(disable_examples) or "(none)"

    pytest_dir = rocm_dir / "share" / "rocprofiler-systems" / "tests" / "pytest"
    examples_out = rocm_dir / "share" / "rocprofiler-systems" / "examples"
    examples_present = examples_out.is_dir() and any(examples_out.iterdir())

    # Rebuild only when the inputs changed: fresh/updated ROCm tree (examples were
    # installed into the tree that was just replaced), updated source, missing
    # example artifacts, or an explicit --force-rebuild.
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
        )
    else:
        log(f"Examples up to date ({facts['git_revision'][:10]}); skipping rebuild.")

    # Staging the pytest tree is cheap; refresh it whenever we rebuilt, the ROCm
    # tree changed, or the suite isn't staged yet.
    if need_build or not (pytest_dir / "conftest.py").is_file():
        pytest_dir = stage_tests(src_dir, rocm_dir, env)
    else:
        log("Test suite already staged; skipping.")

    facts["examples_rebuilt"] = need_build
    facts["examples_installed"] = (
        len(list(examples_out.glob("*"))) if examples_out.is_dir() else 0
    )

    # ---- 5. Run the tests against the tarball binaries ------------------- #
    if args.skip_tests:
        step("Skipping test execution (--skip-tests)")
        log(f"Everything is staged under: {rocm_dir}")
        log("To run manually:")
        log(f"  ROCPROFSYS_INSTALL_DIR={rocm_dir} ROCM_PATH={rocm_dir} \\")
        log(f"  {venv_py} -m pytest {pytest_dir} -m gpu -v")
        facts["result"] = "SKIPPED (--skip-tests)"
        write_summary(summary_log, facts)
        return 0

    step("Run pytest suite in install mode (against tarball rocprof-sys binaries)")
    rc = run_tests(
        venv_py,
        pytest_dir,
        rocm_dir,
        env,
        args.pytest_args,
        args.tier,
        args.reruns,
        args.reruns_delay,
        workdir,
    )

    junit = workdir / "pytest-results.xml"
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
        "variant",
        "gpu_arch",
        "tarball",
        "rocm_version",
        "rocprofsys_version",
        "sha256_verified",
        "sha256",
        "rocm_updated",
        "tarball_url",
        "branch",
        "git_revision",
        "source_changed",
        "examples_rebuilt",
        "examples_installed",
        "disabled_examples",
        "work_dir",
        "rocm_prefix",
        "detailed_log",
        "failures_log",
        "started",
        "finished",
        "command",
    ]
    lines = ["rocprof-sys nightly tarball test - SUMMARY", "=" * 44]
    for key in order:
        if key in facts:
            lines.append(f"{key:20s}: {facts[key]}")

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
