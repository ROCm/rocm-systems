# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Detect the GPU arch and run host preflight checks."""

from __future__ import annotations

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


from .command import die, emit, log, step
from .constants import MULTIARCH_VARIANT, NIGHTLY_TARBALL_INDEX
from .environment import _foreign_rocm_roots, _is_under, _resolved, make_rocm_env
from .tarball import fetch_tarball_index


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


def _gfx_from_kfd_version(version: int) -> str | None:
    """Decode KFD ``gfx_target_version`` (major*10000 + minor*100 + step).

    gfx942 is 90402, gfx90a is 90010 (step 10 prints as hex ``a``), gfx1100 is
    110000. CPU nodes report 0 and are skipped.
    """
    if version <= 0:
        return None
    major = version // 10000
    minor = (version // 100) % 100
    step = version % 100
    if major <= 0:
        return None
    return f"gfx{major}{minor}{step:x}"


def _kfd_gpu_archs() -> list[str]:
    """GPU archs from the kernel topology. Independent of which rocminfo is on PATH."""
    root = Path("/sys/class/kfd/kfd/topology/nodes")
    if not root.is_dir():
        return []
    archs: list[str] = []
    for prop in sorted(root.glob("*/properties")):
        try:
            text = prop.read_text(errors="replace")
        except OSError:
            continue
        match = re.search(r"^gfx_target_version\s+(\d+)", text, re.M)
        if not match:
            continue
        arch = _gfx_from_kfd_version(int(match.group(1)))
        if arch and arch != "gfx000" and arch not in archs:
            archs.append(arch)
    return archs


def _rocminfo_archs(rocm_dir: Path | None, env: dict | None) -> list[str]:
    """Archs from rocminfo, skipping a leftover host ROCm binary when another exists."""
    probe_env = env or os.environ
    foreign = _foreign_rocm_roots(dict(probe_env), rocm_dir or Path("/nonexistent"))
    search: list[Path] = []
    if rocm_dir is not None:
        search.append(rocm_dir / "bin" / "rocminfo")
    which = shutil.which("rocminfo", path=probe_env.get("PATH"))
    if which:
        search.append(Path(which))
    existing = [p for p in search if p.exists()]
    if not existing:
        return []

    def _foreign_binary(path: Path) -> bool:
        resolved = _resolved(path)
        return resolved is not None and any(_is_under(resolved, root) for root in foreign)

    preferred = [p for p in existing if not _foreign_binary(p)]
    rocminfo = (preferred or existing)[0]
    if not preferred:
        log(
            "WARNING: the only rocminfo on PATH belongs to a leftover ROCm/TheRock "
            f"install ({rocminfo}). --variant auto may not match this GPU."
        )
    out = run_rocminfo(str(rocminfo), env)
    archs: list[str] = []
    for arch in re.findall(_GFX_TARGET_RE, out):
        if arch != "gfx000" and arch not in archs:
            archs.append(arch)
    return archs


def detect_gpu_archs(
    rocm_dir: Path | None, env: dict | None = None
) -> tuple[list[str], str]:
    """Return (gfx targets, source).

    KFD is preferred. A host TheRock left on PATH makes ``rocminfo`` the wrong
    binary (or a binary that only prints a generic target), so ``--variant auto``
    cannot trust PATH order.
    """
    kfd = _kfd_gpu_archs()
    if kfd:
        return kfd, "kfd"
    return _rocminfo_archs(rocm_dir, env), "rocminfo"


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

    # GPU visibility. KFD does not depend on a host rocminfo. When the tarball is
    # already extracted and KFD is unavailable, run its rocminfo with a ROCm env
    # that is not the leftover host install.
    det_env = make_rocm_env(os.environ.copy(), rocm_dir) if rocm_present else None
    archs, gpu_source = detect_gpu_archs(rocm_dir if rocm_present else None, det_env)
    if archs:
        log(f"GPU     : {', '.join(archs)} ({gpu_source})")
    elif not args.skip_tests and not args.prepare_only:
        log(
            "WARNING: no GPU detected via KFD or rocminfo. Most GPU tests will "
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
            emit(json.dumps(data, indent=2))
        except Exception:  # noqa: BLE001
            emit(manifest.read_text())

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
