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
    python3 run-nightly-tarball-tests.py --variant gfx90a  # smallest per-GPU tarball
    python3 run-nightly-tarball-tests.py --tier quick    # fast smoke subset
    python3 run-nightly-tarball-tests.py --tier full --run-labels mpi
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
    The script inherits your shell environment and forwards it.
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
import platform
import re
import shlex
import shutil
import socket
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path
from typing import NoReturn

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

# Tier names defined by tests/test_categories.yaml, ordered narrowest first.
# Mirrors tests/pytest/conftest.py::TIER_ORDER.
TIER_ORDER = ["quick", "standard", "comprehensive", "full"]

# Path of the tier definitions relative to the rocprofiler-systems source dir.
TEST_CATEGORIES_REL = Path("tests") / "test_categories.yaml"

# Fallback tarball variant when no per-GPU one can be resolved.
MULTIARCH_VARIANT = "multiarch"

# minimum free disk space (GB) required before a tarball download
DEFAULT_MIN_FREE_GB = 40

# Pinned perfetto trace_processor_shell (mirrors tests/CMakeLists.txt defaults).
TRACE_PROCESSOR_SHELL_URL = (
    "https://commondatastorage.googleapis.com/perfetto-luci-artifacts/"
    "v47.0/linux-amd64/trace_processor_shell"
)
TRACE_PROCESSOR_SHELL_SHA256 = (
    "832425c3c7934904d1e0ec1721beb51423de7dbcf399a899973f2b6b464603fa"
)


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


def die(msg: str, code: int = 1) -> NoReturn:
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
    # context manager so the stdout pipe is closed deterministically, not at GC
    with subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    ) as proc:
        for line in proc.stdout:
            _emit(line, end="")
        rc = proc.wait()
    if check and rc != 0:
        die(f"command failed (exit {rc}): {printable}", rc)
    return rc


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


_INDEX_HTML: str | None = None


def fetch_tarball_index(timeout: int = 60) -> str:
    """Return the nightly index HTML, fetching it at most once per run.

    Both the preflight reachability check and the tarball resolution need this
    page, and it does not change mid-run, so the second caller reuses the first
    one's copy instead of making another request.
    """
    global _INDEX_HTML
    if _INDEX_HTML is None:
        _INDEX_HTML = _http_get_text(NIGHTLY_TARBALL_INDEX, timeout=timeout)
    return _INDEX_HTML


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


_DIST_TARBALL_RE = re.compile(
    r"therock-dist-linux-(?P<variant>.+)-(?P<version>\d+\.\d+\.\d+a\d{8})\.tar\.gz\Z"
)


def parse_dist_tarball(filename: str) -> tuple[str, str] | None:
    """Split a dist tarball filename into (variant, ROCm version), or None.

    Lets a run that reuses an already-extracted tree (--offline / --skip-download)
    report the variant and version it is *actually* testing, rather than echoing back
    what was requested on the command line.
    """
    m = _DIST_TARBALL_RE.match(filename)
    return (m.group("variant"), m.group("version")) if m else None


def index_dist_variants(html: str) -> list[str]:
    """Return the dist tarball variants present in the index, sorted.

    Only the ``therock-dist-linux-<variant>-<version>.tar.gz`` families are
    reported; the parallel ``<variant>-tests-`` tarballs (sample data, not
    redistributables) are skipped.
    """
    variants = set()
    for m in re.finditer(
        r"therock-dist-linux-(.+?)-\d+\.\d+\.\d+a\d{8}\.tar\.gz",
        html,
    ):
        variant = m.group(1)
        if not variant.endswith("-tests"):
            variants.add(variant)
    return sorted(variants)


def _variant_family_pattern(variant: str) -> re.Pattern | None:
    """Compile the GPU-arch family a tarball variant covers, or None if not a family.

    TheRock names per-family tarballs with an upper-case ``X`` standing in for the
    last digit of the arch, plus a market suffix: ``gfx94X-dcgpu`` covers gfx940 /
    gfx941 / gfx942, ``gfx103X-all`` covers gfx1030 / gfx1031 / ... The arch token
    is alphanumeric, so only the ``X`` needs substituting.
    """
    token = variant.split("-", 1)[0]
    if not token.startswith("gfx"):
        return None
    return re.compile(token.replace("X", "[0-9a-f]") + r"\Z")


def redirect_variant(
    requested: str, available: list[str], fallback: str | None = None
) -> str:
    """Resolve a requested variant to one the index actually publishes.

    A specific arch is accepted where only its family ships, e.g. ``gfx942`` ->
    ``gfx94X-dcgpu``, which is how users refer to their GPU (and what rocminfo
    reports) even though no ``gfx942`` tarball exists.
    """
    if requested in available:
        return requested
    matches = [
        v for v in available if (p := _variant_family_pattern(v)) and p.match(requested)
    ]
    if len(matches) == 1:
        log(f"Variant '{requested}' ships as '{matches[0]}'; using that tarball.")
        return matches[0]
    if len(matches) > 1:
        die(
            f"variant '{requested}' is ambiguous: it matches "
            f"{', '.join(matches)}.\n"
            "       Pass one of those to --variant explicitly."
        )
    if fallback and fallback in available:
        log(
            f"WARNING: no tarball covers '{requested}'; falling back to " f"'{fallback}'."
        )
        return fallback
    die(
        f"no nightly tarball variant matches '{requested}'.\n"
        f"       Available variants: {', '.join(available)}\n"
        f"       Browse {NIGHTLY_TARBALL_INDEX} for the full listing."
    )


def resolve_tarball(
    variant: str, version: str | None, fallback: str | None = None
) -> tuple[str, str, str]:
    """Return (filename, url, variant) of the nightly dist tarball to download.

    ``variant`` is e.g. ``multiarch``, ``gfx94X-dcgpu``, or a specific arch such
    as ``gfx942`` that is redirected to the family tarball that ships it. The
    returned variant is the one actually resolved against the index.

    The filename regex deliberately anchors a digit right after ``<variant>-`` so
    the separate ``<variant>-tests-`` tarballs are never matched.
    """
    log(f"Reading nightly tarball index: {NIGHTLY_TARBALL_INDEX}")
    try:
        html = fetch_tarball_index()
    except Exception as exc:  # noqa: BLE001
        die(f"could not fetch tarball index: {exc}")

    available = index_dist_variants(html)
    if not available:
        die(
            f"could not parse any dist tarball from {NIGHTLY_TARBALL_INDEX}; the "
            "index layout may have changed."
        )
    variant = redirect_variant(variant, available, fallback)

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
            f"       Available variants: {', '.join(available)}\n"
            f"       Check {NIGHTLY_TARBALL_INDEX}"
        )

    if version:
        matches = [
            fn
            for fn, (date_int, ver) in candidates.items()
            if _rocm_version_matches(version, ver, date_int)
        ]
        if not matches:
            die(
                f"requested version '{version}' not found for variant '{variant}'.\n"
                f"       Browse {NIGHTLY_TARBALL_INDEX} for valid values."
            )
        filename = max(matches, key=lambda fn: candidates[fn][0])
    else:
        filename = max(candidates, key=lambda fn: candidates[fn][0])

    url = f"{NIGHTLY_TARBALL_BASE}/{filename}"
    return filename, url, variant


def _rocm_version_matches(requested: str, full_version: str, date_int: int) -> bool:
    """Return True when ``requested`` exactly selects ``full_version`` (X.Y.ZaYYYYMMDD).

    Accepts the full nightly string (``7.15.0a20260717``), the trailing build date
    (``20260717``), or the ROCm semver prefix without the date (``7.15.0``). Rejects
    loose substring matches such as ``0`` or ``10.1`` matching unrelated tarballs.
    """
    if requested == full_version:
        return True
    if requested.isdigit() and len(requested) == 8 and int(requested) == date_int:
        return True
    semver = re.fullmatch(r"(\d+\.\d+\.\d+)", requested)
    if semver and full_version.startswith(semver.group(1) + "a"):
        return True
    return False


def _looks_like_gzip(path: Path) -> bool:
    """Return True when ``path`` begins with the gzip magic header."""
    try:
        with open(path, "rb") as fh:
            return fh.read(2) == b"\x1f\x8b"
    except OSError:
        return False


def _cached_download_is_valid(
    path: Path,
    url: str,
    *,
    expected_sha256: str | None,
    require_checksum: bool,
    is_gzip: bool,
) -> bool:
    """Return True when an on-disk download can be reused without re-fetching."""
    if not path.is_file() or path.stat().st_size == 0:
        return False
    if is_gzip and not _looks_like_gzip(path):
        return False
    digest = expected_sha256
    origin = "cli (--sha256)"
    if not digest:
        digest, origin = fetch_published_sha256(url)
    if digest:
        actual = _sha256_file(path)
        if actual.lower() != digest.lower():
            log(
                f"Cached file checksum mismatch (source: {origin}); "
                f"will re-download: {path.name}"
            )
            return False
        return True
    if require_checksum:
        return False
    if is_gzip:
        return True
    return False


def download_file(
    url: str,
    dest: Path,
    *,
    expected_sha256: str | None = None,
    require_checksum: bool = False,
) -> None:
    """Download ``url`` to ``dest`` with resume support (prefers wget/curl)."""
    _require_https(url)
    if _cached_download_is_valid(
        dest,
        url,
        expected_sha256=expected_sha256,
        require_checksum=require_checksum,
        is_gzip=True,
    ):
        log(f"Tarball already present and verified, skipping download: {dest}")
        return
    if dest.exists():
        log(f"Removing invalid or unverified tarball: {dest.name}")
        dest.unlink(missing_ok=True)

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


# --------------------------------------------------------------------------- #
# Preflight / environment discovery
# --------------------------------------------------------------------------- #


def _run_version(
    exe: str, env: dict | None = None, timeout: int = 15
) -> tuple[int | None, str]:
    """Run ``<exe> --version`` once; return (exit status, parsed version line).

    The status is None when the command could not be run at all (missing, timed
    out). It is returned alongside the string because the rocprof-sys smoke check
    needs the status while the summary needs the version, and one probe serves both.
    """
    try:
        r = subprocess.run(
            [exe, "--version"], capture_output=True, text=True, timeout=timeout, env=env
        )
    except Exception:  # noqa: BLE001
        return None, "unknown"
    lines = [ln.strip() for ln in (r.stdout + r.stderr).splitlines() if ln.strip()]
    # skip log-noise lines like "[hh:mm:ss][P:..][file] ... [error] ..." that
    # some rocprof-sys tools emit before the version banner
    clean = [ln for ln in lines if not ln.startswith("[") and "Exception" not in ln]
    for ln in clean:
        if re.search(r"\d+\.\d+\.\d+", ln) or "version" in ln.lower():
            return r.returncode, ln
    if clean:
        return r.returncode, clean[0]
    if lines:
        return r.returncode, lines[0]
    return r.returncode, "unknown"


def _tool_version(exe: str, env: dict | None = None) -> str:
    return _run_version(exe, env)[1]


# Guards against ROCm's generic targets (gfx9-4-generic): without the length floor
# and the lookahead they truncate to a bogus "gfx9" that names no tarball variant.
_GFX_TARGET_RE = re.compile(r"gfx[0-9a-f]{3,}(?![0-9a-z-])")


_ROCMINFO_OUT: dict[str, str] = {}


def run_rocminfo(exe: str, env: dict | None = None) -> str:
    """Run ``rocminfo`` and return its stdout ("" on failure), once per executable.

    Preflight runs it to discover the GPU archs and the sanity report wants the same
    dump; keyed on the executable so the cache is only reused for an identical run
    (preflight may fall back to a system rocminfo before the tarball is extracted).
    """
    if exe not in _ROCMINFO_OUT:
        try:
            _ROCMINFO_OUT[exe] = subprocess.run(
                [exe], capture_output=True, text=True, timeout=30, env=env
            ).stdout
        except Exception:  # noqa: BLE001
            _ROCMINFO_OUT[exe] = ""
    return _ROCMINFO_OUT[exe]


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
    out = run_rocminfo(str(rocminfo), env)
    archs: list[str] = []
    for a in re.findall(_GFX_TARGET_RE, out):
        if a != "gfx000" and a not in archs:
            archs.append(a)
    return archs


def resolve_variant(variant: str, archs: list[str]) -> str:
    """Map ``--variant auto`` to the detected GPU arch.

    The arch is returned as-is (e.g. ``gfx942``); ``redirect_variant`` later maps
    it onto whichever family tarball the index actually publishes, so no
    hand-maintained arch->variant table is needed here.
    """
    if variant != "auto":
        return variant
    if archs:
        log(f"--variant auto: detected {archs[0]}")
        return archs[0]
    log(
        "WARNING: --variant auto detected no GPU arch; falling back to "
        f"'{MULTIARCH_VARIANT}'."
    )
    return MULTIARCH_VARIANT


def _trust_problems(path: Path) -> tuple[list[str], list[str]]:
    """Split ownership/permission problems for *path* into (fatal, advisory).

    Code is executed out of the work dir (venv python, staged pytest tree, tarball
    binaries), so a directory *any* local user can write to is refused outright.
    Foreign ownership is only advisory: reusing a tree prepared by a project
    account or a teammate is a legitimate shared-cluster workflow, and the owner
    is trusted by the person who chose that --work-dir.
    """
    fatal: list[str] = []
    advisory: list[str] = []
    try:
        st = path.stat()
    except FileNotFoundError:
        return fatal, advisory
    if st.st_mode & 0o0002:
        fatal.append(f"{path} is world-writable")
    if st.st_uid not in (os.getuid(), 0):
        advisory.append(f"{path} is owned by uid {st.st_uid}, not you ({os.getuid()})")
    return fatal, advisory


def preflight(args, workdir: Path, rocm_dir: Path) -> list[str]:
    """Fail fast on missing tools/space/network; return detected GPU archs."""
    step("Preflight checks")

    # workdir trust: we execute code from here (venv, staged tests, tarball bins).
    fatal: list[str] = []
    advisory: list[str] = []
    for p in (workdir, rocm_dir, workdir / "venv"):
        f, a = _trust_problems(p)
        fatal += f
        advisory += a
    if fatal:
        die(
            "unsafe working directory (any local user could plant code that we "
            "then execute):\n"
            + "\n".join(f"       - {i}" for i in fatal)
            + "\n       Drop world-write permission (chmod o-w) or use a "
            "--work-dir you own."
        )
    for issue in advisory:
        log(
            f"WARNING: {issue}; continuing, but its owner is trusted to have "
            "prepared the tarball, venv and tests you are about to execute."
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
            # cached for resolve_tarball(), which needs the same page
            fetch_tarball_index(timeout=20)
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


def report_under_test(rocm_dir: Path, env: dict, facts: dict) -> int | None:
    """Log the manifest + rocprof-sys version being validated.

    Returns the exit status of the single ``rocprof-sys-avail --version`` probe (None
    if it could not be run) so ``smoke_check_rocprofsys`` can judge the binaries from
    the same invocation instead of launching them a second time.
    """
    manifest = rocm_dir / "share" / "therock" / "therock_manifest.json"
    if manifest.is_file():
        log(f"TheRock manifest: {manifest}")
        try:
            data = json.loads(manifest.read_text())
            _emit(json.dumps(data, indent=2))
        except Exception:  # noqa: BLE001
            _emit(manifest.read_text())

    avail = rocm_dir / "bin" / "rocprof-sys-avail"
    if avail.exists():
        # generous timeout: this is the first touch of a large binary, possibly on a
        # cold shared filesystem, and its status drives the smoke check below
        rc, version = _run_version(str(avail), env, timeout=60)
    else:
        rc, version = None, "unknown"
    facts["rocprofsys_version"] = version
    log(f"rocprof-sys binaries under test: {rocm_dir / 'bin'}")
    log(f"rocprof-sys version : {version}")
    return rc


def smoke_check_rocprofsys(rocm_dir: Path, version_rc: int | None) -> None:
    """Fail fast if the shipped rocprof-sys binaries can't even start here.

    Judges the ``rocprof-sys-avail --version`` probe already run by
    ``report_under_test``: if it was killed by a signal (e.g. SIGILL, SIGSEGV,
    SIGABRT) the binaries are incompatible with this machine and every test would
    fail, so abort early with a clear, generic message.
    """
    avail = rocm_dir / "bin" / "rocprof-sys-avail"
    if not avail.exists():
        return
    rc = version_rc
    if rc is None:
        log(
            "WARNING: 'rocprof-sys-avail --version' could not be run (or timed out); "
            "skipping the smoke check and continuing."
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
        if title == "rocminfo":
            # reuse preflight's dump of this same binary; keep only the first ~120
            # lines, the rest is per-agent detail that bloats the report
            out = run_rocminfo(cmd[0], env)
            head = "\n".join(out.splitlines()[:120]).rstrip()
            lines.append(head or "(no output: rocminfo failed or timed out)")
            continue
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=60, env=env)
            lines.append(((r.stdout or "") + (r.stderr or "")).rstrip())
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
    force: bool = False,
) -> tuple[Path, bool]:
    """Sparse-clone (or update) rocm-systems.

    Returns ``(source_dir, source_changed)`` where ``source_changed`` is True when
    a fresh clone happened or the branch HEAD moved since the last run.

    When ``no_fetch`` is set (offline mode), an existing checkout is reused as-is
    with no network access; if none exists the run aborts with guidance. When
    ``submodules`` is set (source builds), submodules are initialized (online) or
    verified present (offline). When ``force`` is set, a successful fetch is
    always treated as a source change so examples/tests are rebuilt/refreshed.
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
        changed = old_rev != new_rev or force
        if changed and force and old_rev == new_rev:
            log(
                f"--force-sync: source still at {new_rev[:10]}, "
                "refreshing staged build/test artifacts anyway."
            )
        elif changed:
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


def mpi_available(workdir: Path, env: dict, extra_cmake_args: list[str] | None) -> bool:
    """Probe whether ``find_package(MPI)`` succeeds with this toolchain."""
    probe = workdir / ".mpi-probe"
    # never reuse the cache: stale results would ignore newly passed MPI hints
    shutil.rmtree(probe, ignore_errors=True)
    probe.mkdir(parents=True, exist_ok=True)
    (probe / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.25)\n"
        "project(rocprofsys_mpi_probe LANGUAGES C CXX)\n"
        "find_package(MPI)\n"
        "if(NOT (MPI_C_FOUND AND MPI_CXX_FOUND))\n"
        '    message(FATAL_ERROR "MPI_C/MPI_CXX not found")\n'
        "endif()\n",
        encoding="utf-8",
    )
    cmd = [
        shutil.which("cmake") or "cmake",
        "-S",
        str(probe),
        "-B",
        str(probe / "build"),
    ] + (extra_cmake_args or [])
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)
    except (OSError, subprocess.SubprocessError):
        return False
    return r.returncode == 0


def detect_mpi(workdir: Path, env: dict, cmake_args: list[str], facts: dict) -> bool:
    """Probe MPI, record the outcome, and explain it (see ``mpi_available``).

    Called immediately before a CMake configure rather than once up front, so a run
    that reuses up-to-date examples does not pay for a probe it cannot act on.
    """
    step("Detect MPI support")
    use_mpi = mpi_available(workdir, env, cmake_args)
    facts["mpi"] = "available" if use_mpi else "unavailable (find_package(MPI) failed)"
    if use_mpi:
        log("find_package(MPI) succeeded; MPI is enabled for the build below.")
        log("MPI tests stay excluded until you pass --run-labels mpi.")
    else:
        log(
            "find_package(MPI) failed; MPI is disabled, so MPI tests would skip as "
            "'binary not found'. For a vendor MPI with no mpicc wrapper, pass hints "
            "via --cmake-arg (see --help)."
        )
    return use_mpi


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


def _download_binary(
    url: str,
    dest: Path,
    *,
    expected_sha256: str,
    skip_download: bool,
) -> bool:
    """Download a pinned binary into ``dest``, or reuse a verified cached copy.

    Returns True when ``dest`` is present and executable afterward.
    """
    _require_https(url)
    if dest.is_file() and os.access(dest, os.X_OK):
        if _sha256_file(dest).lower() == expected_sha256.lower():
            return True
        if skip_download:
            log(
                f"WARNING: cached {dest.name} checksum mismatch; "
                "Perfetto validation may fail offline."
            )
            return True
        log(f"Removing invalid cached binary: {dest.name}")
        dest.unlink(missing_ok=True)

    if skip_download:
        return False

    tmp = dest.with_suffix(dest.suffix + ".part")
    wget = shutil.which("wget")
    curl = shutil.which("curl")
    if wget:
        cmd = [wget, "--tries=3", "-q", "-O", str(tmp), url]
    elif curl:
        cmd = [curl, "-fL", "--retry", "3", "-o", str(tmp), url]
    else:
        log(
            "WARNING: neither wget nor curl available; "
            "cannot download trace_processor_shell."
        )
        return False

    log(f"$ {' '.join(cmd)}")
    rc = subprocess.run(cmd).returncode  # noqa: S603
    if rc != 0:
        tmp.unlink(missing_ok=True)
        log(f"WARNING: download failed (exit {rc}): {url}")
        return False

    actual = _sha256_file(tmp)
    if actual.lower() != expected_sha256.lower():
        tmp.unlink(missing_ok=True)
        die(
            "trace_processor_shell SHA-256 mismatch:\n"
            f"       expected: {expected_sha256}\n"
            f"       actual:   {actual}"
        )
    tmp.rename(dest)
    dest.chmod(0o755)
    return True


def ensure_trace_processor_shell(workdir: Path, skip_download: bool) -> Path | None:
    """Cache the pinned perfetto trace_processor_shell under ``workdir``.

    Mirrors the binary staged by tests/CMakeLists.txt so Perfetto validation works
    on air-gapped compute nodes after a networked ``--prepare-only`` run.
    """
    if platform.machine() not in ("x86_64", "AMD64"):
        log(
            "WARNING: trace_processor_shell is built for x86-64 only; "
            f"skipping on {platform.machine()}."
        )
        return None

    dest = workdir / "trace_processor_shell"
    if _download_binary(
        TRACE_PROCESSOR_SHELL_URL,
        dest,
        expected_sha256=TRACE_PROCESSOR_SHELL_SHA256,
        skip_download=skip_download,
    ):
        log(f"trace_processor_shell ready: {dest}")
        return dest

    if skip_download:
        log(
            "WARNING: trace_processor_shell is not cached; Perfetto validation "
            "may fail offline."
        )
    return None


def stage_tests(
    src_dir: Path,
    rocm_dir: Path,
    workdir: Path,
    env: dict,
    *,
    skip_download: bool = False,
) -> Path:
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

    shell = ensure_trace_processor_shell(workdir, skip_download)
    if shell is not None:
        staged_shell = tests_dst / "trace_processor_shell"
        shutil.copy2(shell, staged_shell)
        staged_shell.chmod(0o755)
        log(f"Installed trace_processor_shell into {tests_dst}")

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


def _yaml_to_obj(yaml_path: Path, venv_py: Path):
    """Parse a YAML file, preferring this interpreter and falling back to the venv.

    The script itself may run under a bare system/cray python without PyYAML,
    while the test venv always has it (requirements.txt pins PyYAML>=5.1), so
    shell out to the venv interpreter and exchange the result as JSON.
    """
    try:
        import yaml  # noqa: PLC0415

        return yaml.safe_load(yaml_path.read_text())
    except ImportError:
        pass
    r = subprocess.run(
        [
            str(venv_py),
            "-c",
            "import json,sys,yaml; json.dump(yaml.safe_load(open(sys.argv[1]).read()), "
            "sys.stdout)",
            str(yaml_path),
        ],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        die(f"failed to parse {yaml_path} with {venv_py}:\n{r.stderr.strip()}")
    return json.loads(r.stdout)


def _flatten(values) -> list[str]:
    """One-level flatten, so a YAML alias item (``- *anchor``) expands in place.

    Mirrors tests/pytest/conftest.py::_load_test_categories.
    """
    flat: list[str] = []
    for v in values or []:
        flat.extend(v if isinstance(v, list) else [v])
    return [str(v) for v in flat]


# A YAML pattern is usable as a pytest -k term only if it is a plain substring,
# optionally with a trailing ".*". Both CTest's -R/-E (re.search) and pytest's -k
# match on containment, so "fork.*" and "fork" select the same tests.
_TRAILING_ANY = ".*"
_REGEX_METACHARS = set(".^$*+?{}[]()|\\")


def _pattern_to_k_term(pattern: str, source: str) -> str | None:
    """Translate one YAML name regex into a pytest ``-k`` term.

    Returns None for a match-everything pattern (an empty axis in CTest terms).
    Dies on anything that is not a literal, rather than silently mis-selecting
    tests: the tier definitions have always been literal + optional ".*", and a
    real regex needs a deliberate decision here.
    """
    term = pattern
    while term.endswith(_TRAILING_ANY):
        term = term[: -len(_TRAILING_ANY)]
    if not term:
        return None
    leftover = _REGEX_METACHARS & set(term)
    if leftover:
        die(
            f"cannot translate {source} pattern {pattern!r} into a pytest -k term: "
            f"unsupported regex metacharacter(s) {''.join(sorted(leftover))}.\n"
            f"       {TEST_CATEGORIES_REL} is consumed here as literal substrings "
            "(see _pattern_to_k_term). Either keep the pattern literal or teach "
            "this script how to translate it."
        )
    return term


def _name_axis_terms(patterns, source: str, *, veto_match_all: bool) -> list[str]:
    """Translate a name axis (regex_includes / regex_excludes) into ``-k`` terms.

    A match-everything pattern collapses an include axis to a pass-through (as an
    empty ``-R`` does in CTest) and is rejected on an exclude axis, where it would
    deselect the whole suite.
    """
    terms: list[str] = []
    for pattern in patterns:
        term = _pattern_to_k_term(pattern, source)
        if term is None:
            if veto_match_all:
                die(
                    f"{TEST_CATEGORIES_REL}: {source} pattern {pattern!r} matches "
                    "every test."
                )
            return []
        terms.append(term)
    return terms


def _label_axis_terms(patterns, source: str) -> list[str]:
    """Translate a label axis (label_includes / label_excludes) into ``-m`` terms."""
    return [t for t in (_pattern_to_k_term(p, source) for p in patterns) if t]


def load_test_categories(src_dir: Path, venv_py: Path) -> dict:
    """Load the tier definitions from tests/test_categories.yaml.

    Returns ``{tier: {"includes": [...], "excludes": [...], "label_includes":
    [...], "label_excludes": [...]}}`` with every axis flattened to plain
    pytest ``-k``/``-m`` terms.

    tests/test_categories.yaml is the single source of truth for tier policy,
    shared with CTest (tests/pytest/conftest.py turns the same axes into CTest
    LABELS at generate time). Parsing it here keeps this script from drifting
    from `ctest -L <tier>`.
    """
    yaml_path = src_dir / TEST_CATEGORIES_REL
    if not yaml_path.is_file():
        die(
            f"tier definitions not found at {yaml_path}.\n"
            "       The checked-out source tree looks incomplete; re-run without "
            "--offline (or with --force-sync) to refresh it."
        )
    data = _yaml_to_obj(yaml_path, venv_py) or {}
    categories = data.get("test_categories") or {}
    missing = [t for t in TIER_ORDER if not categories.get(t)]
    if missing:
        die(f"{yaml_path} defines no {', '.join(missing)} tier(s).")

    tiers: dict = {}
    for tier in TIER_ORDER:
        cfg = categories.get(tier) or {}
        tiers[tier] = {
            "includes": _name_axis_terms(
                _flatten(cfg.get("regex_includes")),
                f"{tier}.regex_includes",
                veto_match_all=False,
            ),
            "excludes": _name_axis_terms(
                _flatten(cfg.get("regex_excludes")),
                f"{tier}.regex_excludes",
                veto_match_all=True,
            ),
            "label_includes": _label_axis_terms(
                _flatten(cfg.get("label_includes")), f"{tier}.label_includes"
            ),
            "label_excludes": _label_axis_terms(
                _flatten(cfg.get("label_excludes")), f"{tier}.label_excludes"
            ),
        }
    return tiers


def drop_label_excludes(tiers: dict, labels: list[str]) -> list[str]:
    """Stop excluding *labels* in every tier; returns the labels actually dropped.

    ``label_excludes`` in the YAML is unconditional (e.g. 'mpi' is excluded even
    by the full tier, because TheRock CI has no MPI runtime), so re-enabling a
    category has to be an explicit local override rather than a tier choice.
    """
    dropped = []
    for label in labels:
        hit = False
        for axes in tiers.values():
            if label in axes["label_excludes"]:
                axes["label_excludes"].remove(label)
                hit = True
        if hit:
            dropped.append(label)
        else:
            log(
                f"WARNING: --run-labels {label}: no tier excludes that label; "
                "nothing to re-enable."
            )
    return dropped


def tier_selection_args(tier: str, tiers: dict) -> list[str]:
    """Return the pytest -m/-k selection args for a named tier.

    The YAML axes map onto pytest the same way they map onto CTest, and both
    match on containment, so the translation is direct:
      * ``regex_includes`` (-R)  -> ``-k "(a or b)"``
      * ``regex_excludes`` (-E)  -> ``-k "not (a or b)"``
      * ``label_includes`` (-L)  -> ``-m "(a or b)"``
      * ``label_excludes`` (-LE) -> ``-m "not a and not b"``
    An empty axis is a pass-through, exactly as in CTest.

    No 'gpu' marker filter is applied, so CPU-only install-mode tests (CLI help,
    config, presets, ...) run alongside the GPU tests. The marker names come from
    pytest markers, and the name terms match the standardized (hyphenated) test
    names that conftest.py registers as extra -k keywords.
    """
    axes = tiers[tier]
    k_parts = []
    if axes["includes"]:
        k_parts.append("(" + " or ".join(axes["includes"]) + ")")
    if axes["excludes"]:
        k_parts.append("not (" + " or ".join(axes["excludes"]) + ")")
    m_parts = []
    if axes["label_includes"]:
        m_parts.append("(" + " or ".join(axes["label_includes"]) + ")")
    m_parts += [f"not {label}" for label in axes["label_excludes"]]

    args = []
    if m_parts:
        args += ["-m", " and ".join(m_parts)]
    if k_parts:
        args += ["-k", " and ".join(k_parts)]
    return args


def run_tests(
    venv_py: Path,
    pytest_dir: Path,
    rocm_dir: Path,
    env: dict,
    extra_pytest_args: str | None,
    tier: str,
    tiers: dict,
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

    if extra_pytest_args:
        # shlex, not split(): marker expressions are one argument containing spaces,
        # e.g. --pytest-args "-m 'hpc and mpi'"
        selection = shlex.split(extra_pytest_args)
    else:
        selection = tier_selection_args(tier, tiers)
        log(f"Tier '{tier}' filters: " + " ".join(shlex.quote(a) for a in selection))
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
        "smallest per-family tarball for the detected GPU. A specific arch is "
        "accepted where only its family ships, e.g. 'gfx942' resolves to the "
        "gfx94X-dcgpu tarball. Examples: multiarch, gfx942, gfx94X-dcgpu (MI300), "
        "gfx950-dcgpu, gfx90a, gfx110X-all.",
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
        choices=tuple(TIER_ORDER),
        help="Test tier to run (default: standard). Tiers are read from "
        f"{TEST_CATEGORIES_REL} in the checked-out source, the same definitions "
        "'ctest -L <tier>' uses: 'quick' is a fast smoke subset, 'standard' is the "
        "PR tier, 'comprehensive' the nightly tier, and 'full' everything that can "
        "run. GPU and CPU-only install-mode tests are both included. Ignored when "
        "--pytest-args is given.",
    )
    p.add_argument(
        "--run-labels",
        default="",
        help="Comma-separated marker labels to stop excluding, e.g. 'mpi' to run the "
        f"MPI tests. {TEST_CATEGORIES_REL} excludes some categories (mpi, annotate, "
        "julia, ...) from every tier including 'full', so re-enabling one is an "
        "explicit override. Ignored when --pytest-args is given.",
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
        "--force-sync",
        action="store_true",
        help="Force a git fetch/reset of the source checkout and treat the source "
        "as changed so examples/tests are rebuilt/refreshed. Requires network "
        "access; cannot be combined with --offline.",
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
        "--cmake-arg",
        action="append",
        default=[],
        dest="cmake_args",
        metavar="ARG",
        help="Extra argument forwarded verbatim to the example / source CMake "
        "configure step. Repeatable. Must use the '=' form, since the value starts "
        "with a dash. Use it for site-specific needs, e.g. pointing "
        "find_package(MPI) at a vendor MPI that ships no mpicc wrapper for CMake to "
        "interrogate (Cray MPICH): --cmake-arg=-DMPI_C_LIB_NAMES=mpi "
        "--cmake-arg=-DMPI_mpi_LIBRARY=$MPICH_DIR/lib/libmpi.so. These come last on "
        "the CMake command line, so they also override what the script picked, e.g. "
        "--cmake-arg=-DROCPROFSYS_USE_MPI=OFF to build without MPI on a host that "
        "has it.",
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
    if args.force_sync and args.offline:
        die("--force-sync requires network access and cannot be combined with --offline.")
    if args.force_sync:
        args.skip_clone = False

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
            # shlex.join, not " ".join: keeps quoting so the recorded command can be
            # pasted back verbatim (--pytest-args takes one space-containing value).
            "command": shlex.join(sys.argv),
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

    if args.cmake_args:
        facts["cmake_args"] = " ".join(args.cmake_args)
        log("Extra CMake args: " + " ".join(shlex.quote(a) for a in args.cmake_args))

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
        # report what is staged, not what --variant/--rocm-version asked for: no
        # download happens here, so the extracted tree is the thing under test
        parsed = parse_dist_tarball(current) if current else None
        if parsed:
            facts["variant"], facts["rocm_version"] = parsed
    else:
        # --variant auto is a best-effort hint, so let it degrade to multiarch;
        # an explicitly requested variant must resolve or abort.
        filename, url, variant = resolve_tarball(
            variant,
            args.rocm_version,
            fallback=MULTIARCH_VARIANT if args.variant == "auto" else None,
        )
        facts["variant"] = variant
        facts["tarball"] = filename
        facts["tarball_url"] = url
        parsed = parse_dist_tarball(filename)
        facts["rocm_version"] = parsed[1] if parsed else "unknown"
        if (rocm_dir / "bin").is_dir() and current == filename:
            log(f"ROCm tarball already current ({filename}); reusing extracted tree.")
        else:
            log(f"Selected tarball: {filename}")
            log(f"URL: {url}")
            if current and current != filename:
                log(f"Newer nightly available: {current} -> {filename}")
            tarball = workdir / filename
            download_file(
                url,
                tarball,
                expected_sha256=args.sha256,
                require_checksum=args.require_checksum,
            )
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
    version_rc = report_under_test(rocm_dir, env, facts)
    capture_rocm_sanity(rocm_dir, env, workdir / f"rocm-sanity-{run_stamp}.log", facts)

    # Fail fast if the tarball binaries can't start on this machine, judged from the
    # --version probe above. Skipped when building from source (those tarball binaries
    # aren't what's tested) and during prepare-only (which may run on another node).
    if not args.build_from_source and not args.prepare_only:
        smoke_check_rocprofsys(rocm_dir, version_rc)

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
        force=args.force_sync,
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

    # Resolve the tier now (not just before pytest) so a bad/unreadable tier
    # definition fails before the build instead of after it.
    tiers = load_test_categories(src_dir, venv_py)
    dropped = drop_label_excludes(
        tiers, [x.strip() for x in args.run_labels.split(",") if x.strip()]
    )
    if dropped:
        facts["run_labels"] = ", ".join(dropped)
    if not args.pytest_args:
        log(
            f"Tier '{args.tier}' from {TEST_CATEGORIES_REL}: "
            + " ".join(shlex.quote(a) for a in tier_selection_args(args.tier, tiers))
        )

    # ---- Phase 1 stop: prepared for a later offline compute-node run ------ #
    if args.prepare_only:
        shell = ensure_trace_processor_shell(workdir, skip_download=False)
        if shell is None and platform.machine() in ("x86_64", "AMD64"):
            die(
                "failed to stage trace_processor_shell during --prepare-only.\n"
                "       Install wget or curl and re-run so offline Perfetto tests "
                "have a cached binary."
            )
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
    # MPI is built whenever the machine supports it and --tier/--run-labels decide
    # what runs, so it is probed just before each configure below - never during
    # --prepare-only, whose login node has a different MPI environment than the
    # compute node that does the build.
    if args.build_from_source:
        use_mpi = detect_mpi(workdir, env, args.cmake_args, facts)
        step("Build rocprofiler-systems from source")
        _src_disable = [e.strip() for e in args.disable_examples.split(",") if e.strip()]
        build_dir = build_from_source(
            src_dir,
            workdir,
            rocm_dir,
            env,
            venv_py,
            args.jobs,
            use_mpi,
            disable_examples=_src_disable,
            extra_cmake_args=args.cmake_args,
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
            use_mpi = detect_mpi(workdir, env, args.cmake_args, facts)
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
                use_mpi,
                disable_examples,
                extra_cmake_args=args.cmake_args,
            )
        else:
            # nothing will be configured, so the MPI probe is skipped: the examples
            # on disk already carry whatever MPI decision their build made.
            facts["mpi"] = "not probed (examples up to date)"
            log(f"Examples up to date ({facts['git_revision'][:10]}); skipping rebuild.")

        # Staging the pytest tree is cheap; refresh it whenever we rebuilt, the ROCm
        # tree changed, or the suite isn't staged yet.
        if need_build or not (pytest_dir / "conftest.py").is_file():
            pytest_dir = stage_tests(
                src_dir, rocm_dir, workdir, env, skip_download=args.skip_download
            )
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
        tiers,
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
        "run_labels",
        "mpi",
        "reruns",
        "rerun_failed_attempts",
        "cmake_args",
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
