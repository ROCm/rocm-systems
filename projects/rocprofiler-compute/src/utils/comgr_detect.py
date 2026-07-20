# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Detection of the double comgr condition.

Two different ``libamd_comgr`` libraries loaded into one process abort the
profiling run. This module resolves the comgr required by the rocprofiler-sdk
tool and the comgr supplied by the workload, reports a mismatch, selects a
single comgr to preload, and recognizes the abort in captured output.
"""

import ast
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

# Timeout in seconds for the import-probe subprocess.
_IMPORT_PROBE_TIMEOUT_SEC = 90

# Prefix marking the import probe's result line in its stdout.
_IMPORT_PROBE_MARKER = "__COMGR_PROBE_RESULT__:"


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
    exe_path = Path(executable)
    if _PYTHON_EXE_RE.match(exe_path.name) or _PYTHON_EXE_RE.match(
        Path(os.path.realpath(executable)).name
    ):
        return exe_path
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


def _workload_script_path(app_cmd: list[str]) -> Optional[Path]:
    """Return the ``.py`` script file from a python workload command, if any.

    Returns ``None`` for the module form (``python -m pkg``).
    """
    for arg in app_cmd[1:]:
        if arg == "-m":
            return None
        if arg.startswith("-"):
            continue
        candidate = Path(arg)
        if candidate.suffix == ".py" and candidate.is_file():
            return candidate
        return None
    return None


def _python_import_names(script_path: Path) -> list[str]:
    """Return the top-level package names imported in ``script_path``.

    Parses the source without executing it. Relative imports are skipped.
    """
    try:
        source = script_path.read_text(encoding="utf-8", errors="replace")
    except OSError as err:
        console_debug("comgr", f"could not read workload script: {err}")
        return []
    try:
        tree = ast.parse(source)
    except SyntaxError as err:
        console_debug("comgr", f"could not parse workload script: {err}")
        return []
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name:
                    names.append(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                names.append(node.module.split(".")[0])
    return list(dict.fromkeys(name for name in names if name))


# Probe program: import the given modules, then report every ``libamd_comgr``
# mapped into the process via /proc/self/maps.
_IMPORT_PROBE_SOURCE = f"""
import importlib, json, sys

names = json.loads(sys.argv[1])
for name in names:
    try:
        importlib.import_module(name)
    except BaseException:
        pass

found = []
try:
    with open("/proc/self/maps") as maps:
        for line in maps:
            fields = line.split()
            if len(fields) < 6:
                continue
            path = fields[-1]
            if "{COMGR_LIB_STEM}" in path and path not in found:
                found.append(path)
except OSError:
    pass

print("{_IMPORT_PROBE_MARKER}" + json.dumps(found))
"""


def find_workload_comgr_by_imports(
    app_cmd: list[str],
    env: dict[str, str],
) -> list[Path]:
    """Return the comgr libraries loaded by the workload's imports.

    Python workloads only. Parses the workload script's imports, imports those
    modules in a separate subprocess using the workload's interpreter and
    environment, and reports each ``libamd_comgr`` mapped into that process.
    """
    interpreter = _resolve_interpreter(app_cmd)
    if interpreter is None:
        return []
    script = _workload_script_path(app_cmd)
    if script is None:
        return []
    import_names = _python_import_names(script)
    if not import_names:
        return []
    try:
        completed = subprocess.run(
            [str(interpreter), "-c", _IMPORT_PROBE_SOURCE, json.dumps(import_names)],
            capture_output=True,
            text=True,
            timeout=_IMPORT_PROBE_TIMEOUT_SEC,
            check=False,
            cwd=str(script.resolve().parent),
            env=env,
        )
    except (OSError, subprocess.SubprocessError) as err:
        console_debug("comgr", f"import probe failed: {err}")
        return []
    for line in completed.stdout.splitlines():
        if not line.startswith(_IMPORT_PROBE_MARKER):
            continue
        try:
            paths = json.loads(line[len(_IMPORT_PROBE_MARKER) :])
        except (ValueError, TypeError):
            return []
        return _dedupe_paths([Path(p) for p in paths])
    return []


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


def dedupe_comgr_paths(paths: list[Path]) -> list[Path]:
    """De-duplicate comgr paths by real path, preserving order."""
    return _dedupe_paths(paths)


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
) -> Optional[Path]:
    """Detect and log the double comgr condition before the workload runs.

    Returns the workload's ``libamd_comgr`` to preload when it differs from the
    profiler tool's comgr, so that both share a single comgr; otherwise ``None``.
    """
    try:
        tool_comgr = resolve_tool_comgr(tool_path)
        if tool_comgr is not None:
            console_log("comgr", f"Profiler tool comgr: {tool_comgr}")
        else:
            console_debug("comgr", "could not resolve profiler tool comgr")

        dynamic_comgrs = find_workload_comgr_dynamic(env)
        if dynamic_comgrs:
            console_log(
                "comgr",
                f"Workload comgr (dynamic env): "
                f"{', '.join(str(p) for p in dynamic_comgrs)}",
            )

        import_comgrs = find_workload_comgr_by_imports(app_cmd, env)
        if import_comgrs:
            console_log(
                "comgr",
                f"Workload comgr (imports): {', '.join(str(p) for p in import_comgrs)}",
            )

        static_comgrs = find_workload_comgr_static(app_cmd)
        if static_comgrs:
            console_log(
                "comgr",
                f"Workload comgr (static): {', '.join(str(p) for p in static_comgrs)}",
            )

        workload_comgrs = _dedupe_paths(dynamic_comgrs + import_comgrs + static_comgrs)
        if not workload_comgrs:
            console_debug("comgr", "no workload comgr found")

        conflicting = _log_comgr_mismatch(tool_comgr, workload_comgrs)
        if conflicting and tool_comgr is not None:
            return conflicting[0]
    except Exception as err:  # noqa: BLE001
        console_debug("comgr", f"detection skipped: {err}")
    return None


def _log_comgr_mismatch(
    tool_comgr: Optional[Path],
    workload_comgrs: list[Path],
) -> list[Path]:
    """Warn on and return the workload comgrs that differ from the tool comgr."""
    if tool_comgr is None or not workload_comgrs:
        return []
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
    return conflicting


def tool_path_from_preload(ld_preload: Optional[str]) -> Optional[str]:
    """Return the rocprofiler-sdk tool path from ``LD_PRELOAD``, if present."""
    if not ld_preload:
        return None
    for entry in ld_preload.replace(os.pathsep, " ").split():
        if "rocprofiler-sdk-tool" in Path(entry).name:
            return entry
    return None
