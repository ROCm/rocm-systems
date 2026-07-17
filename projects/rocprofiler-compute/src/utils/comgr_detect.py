# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Detection of the double comgr condition.

Two different ``libamd_comgr`` libraries loaded into one process cause the
profiling run to abort. This module resolves the comgr required by the
rocprofiler-sdk tool and the comgr supplied by the workload, logs both, and
reports a mismatch. It also identifies the abort in captured workload output.
"""

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from utils.logger import console_debug, console_log, console_warning
from utils.utils_common import resolve_rocm_library_path

COMGR_LIB_STEM = "libamd_comgr.so"

# Substrings emitted by LLVM when two comgr libraries register the same
# command-line option.
_DOUBLE_COMGR_SIGNATURES = (
    "registered more than once",
    "inconsistency in registered CommandLine options",
)

_PYTHON_EXE_RE = re.compile(r"^python[0-9.]*$")

# Timeout in seconds for external commands.
_SUBPROCESS_TIMEOUT_SEC = 20


def output_indicates_double_comgr(output: str) -> bool:
    """Return True if the captured output contains a double comgr abort signature."""
    return any(signature in output for signature in _DOUBLE_COMGR_SIGNATURES)


def double_comgr_error_message(
    tool_comgr: Optional[Path],
    workload_comgrs: list[Path],
) -> str:
    """Return the error message for a double comgr abort."""
    lines = [
        "Detected two different 'libamd_comgr' libraries loaded in the same process.",
    ]
    if tool_comgr is not None:
        lines.append(f"  Profiler tool comgr : {tool_comgr}")
    for workload_comgr in workload_comgrs:
        lines.append(f"  Workload comgr      : {workload_comgr}")
    lines.append(
        "Align the workload's bundled ROCm with the profiler's ROCm (matching "
        "'libamd_comgr' major), or run the workload against a single ROCm stack."
    )
    return "\n".join(lines)


def _read_needed_comgr_soname(elf_path: Path) -> Optional[str]:
    """Return the ``libamd_comgr.so.<major>`` in the ELF's DT_NEEDED entries, if any."""
    readelf = shutil.which("readelf")
    if readelf is None:
        console_debug("comgr", "readelf not found; cannot read DT_NEEDED entries")
        return None
    try:
        completed = subprocess.run(
            [readelf, "-d", str(elf_path)],
            capture_output=True,
            text=True,
            timeout=_SUBPROCESS_TIMEOUT_SEC,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as err:
        console_debug("comgr", f"readelf failed on {elf_path}: {err}")
        return None
    for line in completed.stdout.splitlines():
        if "(NEEDED)" not in line or COMGR_LIB_STEM not in line:
            continue
        match = re.search(r"\[(" + re.escape(COMGR_LIB_STEM) + r"[0-9.]*)\]", line)
        if match:
            return match.group(1)
    return None


def resolve_tool_comgr(tool_path: Optional[str]) -> Optional[Path]:
    """Resolve the comgr required by the rocprofiler-sdk tool.

    Reads the tool's DT_NEEDED comgr soname and resolves it under the active
    ROCm root.
    """
    if not tool_path:
        return None
    soname = _read_needed_comgr_soname(Path(tool_path))
    if soname is None:
        return None
    rocm_root = Path(os.getenv("ROCM_PATH", "/opt/rocm"))
    resolved = resolve_rocm_library_path(str(rocm_root / "lib" / soname))
    if resolved and Path(resolved).exists():
        return Path(resolved)
    return None


def _glob_comgr_in_dir(directory: Path) -> list[Path]:
    """Return ``libamd_comgr.so*`` files directly under ``directory``."""
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.glob(f"{COMGR_LIB_STEM}*") if p.is_file())


def find_workload_comgr_dynamic(env: dict[str, str]) -> list[Path]:
    """Return workload comgr candidates from the workload's runtime environment.

    Scans ``LD_LIBRARY_PATH`` directories and ``LD_PRELOAD`` entries.
    """
    candidates: list[Path] = []
    for directory in env.get("LD_LIBRARY_PATH", "").split(os.pathsep):
        if directory:
            candidates.extend(_glob_comgr_in_dir(Path(directory)))
    for entry in env.get("LD_PRELOAD", "").replace(os.pathsep, " ").split():
        entry_path = Path(entry)
        if COMGR_LIB_STEM in entry_path.name and entry_path.is_file():
            candidates.append(entry_path)
    return _dedupe_paths(candidates)


def _resolve_interpreter(app_cmd: list[str]) -> Optional[Path]:
    """Return the Python interpreter for the workload command, if any."""
    if not app_cmd:
        return None
    executable = shutil.which(app_cmd[0])
    if not executable:
        return None
    resolved = Path(executable).resolve()
    if _PYTHON_EXE_RE.match(resolved.name):
        return resolved
    return None


def _interpreter_site_packages(interpreter: Path) -> list[Path]:
    """Return the interpreter's site-packages directories."""
    code = (
        "import site, json\n"
        "dirs = list(getattr(site, 'getsitepackages', lambda: [])())\n"
        "user = getattr(site, 'getusersitepackages', lambda: '')()\n"
        "if user:\n"
        "    dirs.append(user)\n"
        "print(json.dumps(dirs))\n"
    )
    try:
        completed = subprocess.run(
            [str(interpreter), "-c", code],
            capture_output=True,
            text=True,
            timeout=_SUBPROCESS_TIMEOUT_SEC,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as err:
        console_debug("comgr", f"interpreter introspection failed: {err}")
        return []
    if completed.returncode != 0:
        return []
    try:
        return [Path(directory) for directory in json.loads(completed.stdout.strip())]
    except (ValueError, TypeError):
        return []


def _elf_runpath_dirs(elf_path: Path) -> list[Path]:
    """Return the RPATH/RUNPATH directories of an ELF, with $ORIGIN expanded."""
    readelf = shutil.which("readelf")
    if readelf is None:
        return []
    try:
        completed = subprocess.run(
            [readelf, "-d", str(elf_path)],
            capture_output=True,
            text=True,
            timeout=_SUBPROCESS_TIMEOUT_SEC,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    origin = str(elf_path.resolve().parent)
    dirs: list[Path] = []
    for line in completed.stdout.splitlines():
        if "(RPATH)" not in line and "(RUNPATH)" not in line:
            continue
        match = re.search(r"\[(.*)\]", line)
        if not match:
            continue
        for raw_dir in match.group(1).split(os.pathsep):
            expanded = raw_dir.replace("$ORIGIN", origin).replace("${ORIGIN}", origin)
            if expanded:
                dirs.append(Path(expanded))
    return dirs


def find_workload_comgr_static(app_cmd: list[str]) -> list[Path]:
    """Return workload comgr candidates from static inspection.

    Checks the interpreter's site-packages and the RPATH/RUNPATH of the
    workload executable.
    """
    candidates: list[Path] = []
    interpreter = _resolve_interpreter(app_cmd)
    if interpreter is not None:
        for site_dir in _interpreter_site_packages(interpreter):
            if site_dir.is_dir():
                candidates.extend(
                    p for p in site_dir.rglob(f"{COMGR_LIB_STEM}*") if p.is_file()
                )
    if app_cmd:
        executable = shutil.which(app_cmd[0])
        if executable:
            for runpath_dir in _elf_runpath_dirs(Path(executable)):
                candidates.extend(_glob_comgr_in_dir(runpath_dir))
    return _dedupe_paths(candidates)


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    """Return paths de-duplicated by real path, preserving order."""
    seen: set[str] = set()
    unique: list[Path] = []
    for path in paths:
        key = _real_path(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def _real_path(path: Path) -> str:
    """Return the canonical real path with symlinks resolved."""
    try:
        return os.path.realpath(path)
    except OSError:
        return str(path)


def detect_and_log_double_comgr(
    app_cmd: list[str],
    tool_path: Optional[str],
    env: dict[str, str],
) -> None:
    """Detect and log the double comgr condition before the workload runs."""
    try:
        tool_comgr = resolve_tool_comgr(tool_path)
        if tool_comgr is not None:
            console_log("comgr", f"Profiler tool comgr (#1): {tool_comgr}")
        else:
            console_debug("comgr", "could not resolve profiler tool comgr")

        workload_comgrs = find_workload_comgr_dynamic(env)
        if workload_comgrs:
            console_log(
                "comgr",
                f"Workload comgr from dynamic env (#2): "
                f"{', '.join(str(p) for p in workload_comgrs)}",
            )
        else:
            workload_comgrs = find_workload_comgr_static(app_cmd)
            if workload_comgrs:
                console_log(
                    "comgr",
                    f"Workload comgr from static check (#2): "
                    f"{', '.join(str(p) for p in workload_comgrs)}",
                )
            else:
                console_debug("comgr", "no workload comgr found")

        _log_comgr_mismatch(tool_comgr, workload_comgrs)
    except Exception as err:  # noqa: BLE001
        console_debug("comgr", f"detection skipped: {err}")


def _log_comgr_mismatch(
    tool_comgr: Optional[Path],
    workload_comgrs: list[Path],
) -> None:
    """Warn when the tool comgr and a workload comgr are different objects."""
    if tool_comgr is None or not workload_comgrs:
        return
    tool_real = _real_path(tool_comgr)
    conflicting = [p for p in workload_comgrs if _real_path(p) != tool_real]
    if conflicting:
        console_warning(
            "comgr",
            "Double comgr detected: the workload provides a different "
            "'libamd_comgr' than the profiler tool. The run may abort.\n"
            f"  Profiler tool comgr : {tool_comgr}\n"
            + "\n".join(f"  Workload comgr      : {p}" for p in conflicting),
        )
    else:
        console_debug("comgr", "single comgr; workload matches tool")


def tool_path_from_preload(ld_preload: Optional[str]) -> Optional[str]:
    """Return the rocprofiler-sdk tool path from an ``LD_PRELOAD`` string, if present."""
    if not ld_preload:
        return None
    for entry in ld_preload.replace(os.pathsep, " ").split():
        if "rocprofiler-sdk-tool" in Path(entry).name:
            return entry
    return None
