# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Pre-flight detection of conflicting ROCm stacks.

When a workload bundles its own ROCm, two copies of a library can load into one
process. Two ``libamd_comgr`` libraries cause an LLVM command-line option
conflict, and two ``librocprofiler-register`` or ``librocprofiler-sdk``
libraries cause rocprofiler registration to fail (error 16).

This module resolves the ROCm stack used by the rocprofiler-sdk tool and the
stack supplied by the workload, reports both, and returns a per-backend
decision: force a single ``libamd_comgr`` through ``LD_PRELOAD`` when the soname
majors match, redirect the rocprofv3 backend to the workload's own ``rocprofv3``
when the workload supplies a distinct ``librocprofiler-sdk``, or stop early when
the stacks cannot be reconciled.
"""

import ast
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from utils.logger import console_debug, console_log, console_warning
from utils.utils_common import resolve_rocm_library_path
from utils.utils_exceptions import IncompatibleRocmStackError

COMGR_LIB_STEM = "libamd_comgr.so"
ROCPROFILER_REGISTER_LIB_STEM = "librocprofiler-register.so"
ROCPROFILER_SDK_LIB_STEM = "librocprofiler-sdk.so"

# Launcher shipped inside the workload's ROCm stack.
_ROCPROFV3_BINARY = "rocprofv3"

# Libraries compared between the profiler and workload stacks.
_STACK_LIB_STEMS = (
    COMGR_LIB_STEM,
    ROCPROFILER_REGISTER_LIB_STEM,
    ROCPROFILER_SDK_LIB_STEM,
)

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
_IMPORT_PROBE_MARKER = "__STACK_PROBE_RESULT__:"


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


def _rocprofiler_mismatch_message(
    lib_stem: str,
    tool_lib: Path,
    workload_lib: Path,
) -> str:
    """Return the error message for a conflicting rocprofiler library."""
    library = lib_stem[: -len(".so")] if lib_stem.endswith(".so") else lib_stem
    return "\n".join([
        f"Incompatible ROCm stacks: the workload provides a different "
        f"'{library}' than the profiler tool.",
        f"  Profiler tool : {tool_lib}",
        f"  Workload      : {workload_lib}",
        "Two rocprofiler stacks in one process cause rocprofiler registration to "
        "fail (error 16). Run the workload against the profiler's ROCm, or install "
        "a matching ROCm for the profiler.",
    ])


def _read_needed_soname(elf_path: Path, lib_stem: str) -> Optional[str]:
    """Return the ``<lib_stem>.<version>`` in the ELF's DT_NEEDED entries, if any."""
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
        if "(NEEDED)" not in line or lib_stem not in line:
            continue
        match = re.search(r"\[(" + re.escape(lib_stem) + r"[0-9.]*)\]", line)
        if match:
            return match.group(1)
    return None


def _resolve_tool_comgr(tool_path: Optional[str]) -> Optional[Path]:
    """Resolve the comgr required by the rocprofiler-sdk tool.

    Reads the tool's DT_NEEDED comgr soname and resolves it under the active
    ROCm root.
    """
    if not tool_path:
        return None
    soname = _read_needed_soname(Path(tool_path), COMGR_LIB_STEM)
    if soname is None:
        return None
    rocm_root = Path(os.getenv("ROCM_PATH", "/opt/rocm"))
    resolved = resolve_rocm_library_path(str(rocm_root / "lib" / soname))
    if resolved and Path(resolved).exists():
        return Path(resolved)
    return None


def _resolve_tool_stack_lib(tool_path: Optional[str], lib_stem: str) -> Optional[Path]:
    """Resolve a stack library from the profiler's ROCm.

    Prefers the tool's DT_NEEDED soname and otherwise searches the tool
    directory and the active ROCm library directory.
    """
    search_dirs: list[Path] = []
    if tool_path:
        search_dirs.append(Path(os.path.realpath(tool_path)).parent)
    search_dirs.append(Path(os.getenv("ROCM_PATH", "/opt/rocm")) / "lib")

    soname = _read_needed_soname(Path(tool_path), lib_stem) if tool_path else None
    if soname:
        for directory in search_dirs:
            resolved = resolve_rocm_library_path(str(directory / soname))
            if resolved and Path(resolved).exists():
                return Path(resolved)

    for directory in search_dirs:
        hits = _glob_lib_in_dir(directory, lib_stem)
        versioned = sorted(
            (h for h in hits if _soname_version(h, lib_stem)),
            key=lambda h: len(h.name),
            reverse=True,
        )
        if versioned:
            return versioned[0]
        if hits:
            return hits[0]
    return None


def _glob_lib_in_dir(directory: Path, lib_stem: str) -> list[Path]:
    """Return ``<lib_stem>*`` files directly under ``directory``."""
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.glob(f"{lib_stem}*") if p.is_file())


def _find_workload_libs_dynamic(
    env: dict[str, str],
    stems: tuple[str, ...],
) -> dict[str, list[Path]]:
    """Return stack-library candidates from the workload's runtime environment.

    Scans ``LD_LIBRARY_PATH`` directories and ``LD_PRELOAD`` entries.
    """
    result: dict[str, list[Path]] = {stem: [] for stem in stems}
    for directory in env.get("LD_LIBRARY_PATH", "").split(os.pathsep):
        if not directory:
            continue
        for stem in stems:
            result[stem].extend(_glob_lib_in_dir(Path(directory), stem))
    for entry in env.get("LD_PRELOAD", "").replace(os.pathsep, " ").split():
        entry_path = Path(entry)
        if not entry_path.is_file():
            continue
        for stem in stems:
            if stem in entry_path.name:
                result[stem].append(entry_path)
    return {stem: _dedupe_paths(paths) for stem, paths in result.items()}


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


# Probe program: import the given modules, then report every mapped library
# whose path contains one of the given stems, read from /proc/self/maps.
_IMPORT_PROBE_SOURCE = f"""
import importlib, json, sys

names = json.loads(sys.argv[1])
stems = json.loads(sys.argv[2])
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
            if any(stem in path for stem in stems) and path not in found:
                found.append(path)
except OSError:
    pass

print("{_IMPORT_PROBE_MARKER}" + json.dumps(found))
"""


def _run_import_probe(
    app_cmd: list[str],
    env: dict[str, str],
    stems: tuple[str, ...],
) -> list[Path]:
    """Return the stack libraries loaded by the workload's imports.

    Python workloads only. Parses the workload script's imports, imports those
    modules in a separate subprocess using the workload's interpreter and
    environment, and reports each mapped library matching ``stems``.
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
            [
                str(interpreter),
                "-c",
                _IMPORT_PROBE_SOURCE,
                json.dumps(import_names),
                json.dumps(list(stems)),
            ],
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


def _find_workload_libs_static(
    app_cmd: list[str],
    stems: tuple[str, ...],
) -> dict[str, list[Path]]:
    """Return stack-library candidates from static inspection.

    Checks the interpreter's site-packages and the RPATH/RUNPATH of the
    workload executable.
    """
    result: dict[str, list[Path]] = {stem: [] for stem in stems}
    interpreter = _resolve_interpreter(app_cmd)
    if interpreter is not None:
        for site_dir in _interpreter_site_packages(interpreter):
            if not site_dir.is_dir():
                continue
            for stem in stems:
                result[stem].extend(
                    p for p in site_dir.rglob(f"{stem}*") if p.is_file()
                )
    if app_cmd:
        executable = shutil.which(app_cmd[0])
        if executable:
            for runpath_dir in _elf_runpath_dirs(Path(executable)):
                for stem in stems:
                    result[stem].extend(_glob_lib_in_dir(runpath_dir, stem))
    return {stem: _dedupe_paths(paths) for stem, paths in result.items()}


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


def _under_directory(path: Path, directory: Path) -> bool:
    """Return True if ``path`` resolves to a location under ``directory``."""
    try:
        Path(_real_path(path)).relative_to(directory)
        return True
    except ValueError:
        return False


def _resolve_tool_stack(
    tool_path: Optional[str],
    stems: tuple[str, ...],
) -> dict[str, Optional[Path]]:
    """Resolve the profiler tool's copy of each stack library."""
    resolved: dict[str, Optional[Path]] = {}
    for stem in stems:
        if stem == COMGR_LIB_STEM:
            resolved[stem] = _resolve_tool_comgr(tool_path)
        else:
            resolved[stem] = _resolve_tool_stack_lib(tool_path, stem)
    return resolved


def _discover_workload_stack(
    app_cmd: list[str],
    env: dict[str, str],
    stems: tuple[str, ...],
) -> dict[str, list[Path]]:
    """Resolve the workload's copy of each stack library.

    Combines the runtime environment, an import probe of the workload's
    modules, and static inspection of site-packages and RPATH/RUNPATH.
    Libraries under the profiler's ROCm root are excluded.
    """
    rocm_root = Path(os.path.realpath(os.getenv("ROCM_PATH", "/opt/rocm")))
    dynamic = _find_workload_libs_dynamic(env, stems)
    probed = _run_import_probe(app_cmd, env, stems)
    static = _find_workload_libs_static(app_cmd, stems)
    result: dict[str, list[Path]] = {}
    for stem in stems:
        imports = [p for p in probed if stem in p.name]
        candidates = _dedupe_paths(dynamic[stem] + imports + static[stem])
        result[stem] = [p for p in candidates if not _under_directory(p, rocm_root)]
    return result


def _report_stacks(
    tool_stack: dict[str, Optional[Path]],
    workload_stack: dict[str, list[Path]],
) -> None:
    """Log the resolved profiler and workload ROCm stacks."""
    lines = [
        "ROCm libraries found in the profiler tool and in the workload:",
        "Profiler tool:",
    ]
    for stem in _STACK_LIB_STEMS:
        tool_lib = tool_stack.get(stem)
        lines.append(f"  {stem} : {tool_lib if tool_lib is not None else 'not found'}")
    lines.append("Workload:")
    for stem in _STACK_LIB_STEMS:
        workload_libs = workload_stack.get(stem, [])
        value = (
            ", ".join(str(p) for p in workload_libs) if workload_libs else "not found"
        )
        lines.append(f"  {stem} : {value}")
    console_log("stack", "\n".join(lines))


@dataclass(frozen=True)
class StackResolution:
    """Resolved profiler-tool and workload copies of each ROCm stack library."""

    tool_stack: dict[str, Optional[Path]]
    workload_stack: dict[str, list[Path]]

    def tool(self, lib_stem: str) -> Optional[Path]:
        """Return the profiler tool's copy of ``lib_stem``, if resolved."""
        return self.tool_stack.get(lib_stem)

    def workload(self, lib_stem: str) -> list[Path]:
        """Return the workload's copies of ``lib_stem``."""
        return self.workload_stack.get(lib_stem, [])


def resolve_rocm_stacks(
    app_cmd: list[str],
    tool_path: Optional[str],
    env: dict[str, str],
) -> Optional[StackResolution]:
    """Resolve and report the profiler and workload ROCm stacks once.

    Returns ``None`` when the stacks cannot be resolved.
    """
    # Backend decisions compare the workload stack against the tool stack;
    # without a tool path there is nothing to resolve.
    if not tool_path:
        console_debug(
            "stack", "no rocprofiler-sdk tool path; skipping ROCm stack pre-flight"
        )
        return None
    try:
        tool_stack = _resolve_tool_stack(tool_path, _STACK_LIB_STEMS)
        workload_stack = _discover_workload_stack(app_cmd, env, _STACK_LIB_STEMS)
    except Exception as err:  # noqa: BLE001
        console_debug("stack", f"stack resolution skipped: {err}")
        return None
    _report_stacks(tool_stack, workload_stack)
    return StackResolution(tool_stack, workload_stack)


@dataclass(frozen=True)
class Rocprofv3Launch:
    """How to launch the rocprofv3 backend on a single ROCm stack.

    ``rocprofv3`` is the workload stack's launcher to run instead of the default,
    and ``forced_comgr`` is the ``libamd_comgr`` to prepend to ``LD_PRELOAD``. At
    most one is set.
    """

    rocprofv3: Optional[str] = None
    forced_comgr: Optional[str] = None


def comgr_to_force(resolution: StackResolution) -> Optional[Path]:
    """Return the workload ``libamd_comgr`` to preload, or ``None``.

    Raises :class:`IncompatibleRocmStackError` when the workload provides a
    conflicting rocprofiler library.
    """
    _raise_on_rocprofiler_mismatch(resolution.tool_stack, resolution.workload_stack)
    tool_comgr = resolution.tool(COMGR_LIB_STEM)
    forceable = _log_comgr_mismatch(tool_comgr, resolution.workload(COMGR_LIB_STEM))
    if forceable:
        return forceable[0]
    return None


def _soname_major(path: Path, lib_stem: str) -> Optional[int]:
    """Return the soname major version of a stack library file, if present."""
    for name in (path.name, Path(_real_path(path)).name):
        match = re.search(re.escape(lib_stem) + r"\.([0-9]+)", name)
        if match:
            return int(match.group(1))
    return None


def _soname_version(path: Path, lib_stem: str) -> Optional[str]:
    """Return the full soname version of a stack library file, if present."""
    for name in (path.name, Path(_real_path(path)).name):
        match = re.search(re.escape(lib_stem) + r"\.([0-9.]+)", name)
        if match:
            return match.group(1).strip(".")
    return None


def _log_comgr_mismatch(
    tool_comgr: Optional[Path],
    workload_comgrs: list[Path],
) -> list[Path]:
    """Warn on comgr mismatches and return the workload comgrs that can be
    forced onto a single library via ``LD_PRELOAD``.

    Only workload comgrs whose soname major matches the tool comgr are
    returned; those with a different major are reported but not returned.
    """
    if tool_comgr is None or not workload_comgrs:
        return []
    tool_real = _real_path(tool_comgr)
    conflicting = [p for p in workload_comgrs if _real_path(p) != tool_real]
    if not conflicting:
        console_debug("comgr", "single comgr; workload matches tool")
        return []

    tool_major = _soname_major(tool_comgr, COMGR_LIB_STEM)
    forceable = [
        p
        for p in conflicting
        if tool_major is None
        or _soname_major(p, COMGR_LIB_STEM) is None
        or _soname_major(p, COMGR_LIB_STEM) == tool_major
    ]
    incompatible = [p for p in conflicting if p not in forceable]

    console_warning(
        "comgr",
        "Double comgr detected: the workload provides a different "
        "'libamd_comgr' than the profiler tool.\n"
        f"  Profiler tool comgr : {tool_comgr}\n"
        + "\n".join(f"  Workload comgr      : {p}" for p in conflicting),
    )
    if incompatible:
        console_warning(
            "comgr",
            "The workload comgr major differs from the profiler tool comgr "
            "major; a single comgr cannot be forced and the run may abort. "
            "Align the workload's bundled ROCm with the profiler's ROCm "
            "(matching 'libamd_comgr' major), or run the workload against a "
            "single ROCm stack.",
        )
    return forceable


def _rocprofiler_conflict(
    tool_lib: Optional[Path],
    workload_libs: list[Path],
    lib_stem: str,
) -> Optional[Path]:
    """Return a workload rocprofiler library whose soname major differs from
    the tool's.

    A workload library at the tool's real path, or one whose soname major
    cannot be read, is not treated as a conflict.
    """
    if tool_lib is None:
        return None
    tool_major = _soname_major(tool_lib, lib_stem)
    if tool_major is None:
        return None
    tool_real = _real_path(tool_lib)
    for workload_lib in workload_libs:
        if _real_path(workload_lib) == tool_real:
            continue
        workload_major = _soname_major(workload_lib, lib_stem)
        if workload_major is not None and workload_major != tool_major:
            return workload_lib
    return None


def _raise_on_rocprofiler_mismatch(
    tool_stack: dict[str, Optional[Path]],
    workload_stack: dict[str, list[Path]],
) -> None:
    """Raise when the workload provides a conflicting rocprofiler library."""
    for stem in (ROCPROFILER_REGISTER_LIB_STEM, ROCPROFILER_SDK_LIB_STEM):
        conflict = _rocprofiler_conflict(
            tool_stack.get(stem), workload_stack.get(stem, []), stem
        )
        if conflict is not None:
            raise IncompatibleRocmStackError(
                _rocprofiler_mismatch_message(stem, tool_stack[stem], conflict)
            )


def _missing_workload_rocprofv3_message(
    tool_sdk: Path,
    workload_sdk: Path,
) -> str:
    """Return the error message when the workload sdk has no matching rocprofv3."""
    return "\n".join([
        "The workload supplies its own 'librocprofiler-sdk', but its ROCm "
        "stack has no matching 'rocprofv3'.",
        f"  Profiler tool : {tool_sdk}",
        f"  Workload      : {workload_sdk}",
        "Two rocprofiler-sdk libraries in one process cause rocprofiler "
        "registration to fail (error 16). Use the rocprofiler-sdk backend "
        "(set ROCPROF=rocprofiler-sdk).",
    ])


def _distinct_workload_lib(
    tool_lib: Optional[Path],
    workload_libs: list[Path],
) -> Optional[Path]:
    """Return a workload library whose real path differs from the tool's."""
    if tool_lib is None:
        return None
    tool_real = _real_path(tool_lib)
    for lib in workload_libs:
        if _real_path(lib) != tool_real:
            return lib
    return None


def _workload_stack_rocprofv3(workload_sdk: Path) -> Optional[Path]:
    """Return the ``rocprofv3`` shipped alongside the workload's sdk, if any.

    The launcher resides at ``<root>/bin/rocprofv3`` where ``<root>/lib``
    contains the sdk.
    """
    parents = list(Path(_real_path(workload_sdk)).parents)
    for root in parents:
        if (root / "lib") not in parents:
            continue
        candidate = root / "bin" / _ROCPROFV3_BINARY
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    return None


def _soname_majors_differ(tool_lib: Path, workload_lib: Path, lib_stem: str) -> bool:
    """Return True when both soname majors are known and differ."""
    tool_major = _soname_major(tool_lib, lib_stem)
    workload_major = _soname_major(workload_lib, lib_stem)
    return (
        tool_major is not None
        and workload_major is not None
        and tool_major != workload_major
    )


def plan_rocprofv3(resolution: StackResolution) -> Rocprofv3Launch:
    """Return the rocprofv3-backend launch decision for ``resolution``.

    Returns a :class:`Rocprofv3Launch` that either redirects to the workload's
    own ``rocprofv3`` (when the workload provides a distinct ``librocprofiler-sdk``
    with a matching soname major and a usable launcher) or forces a single
    ``libamd_comgr``.

    Raises :class:`IncompatibleRocmStackError` when the workload provides a
    distinct ``librocprofiler-sdk`` whose soname major differs or that has no
    usable ``rocprofv3``, or a conflicting ``librocprofiler-register``.
    """
    tool_sdk = resolution.tool(ROCPROFILER_SDK_LIB_STEM)
    workload_sdk = _distinct_workload_lib(
        tool_sdk, resolution.workload(ROCPROFILER_SDK_LIB_STEM)
    )
    if workload_sdk is not None:
        if _soname_majors_differ(tool_sdk, workload_sdk, ROCPROFILER_SDK_LIB_STEM):
            raise IncompatibleRocmStackError(
                _rocprofiler_mismatch_message(
                    ROCPROFILER_SDK_LIB_STEM, tool_sdk, workload_sdk
                )
            )
        rocprofv3 = _workload_stack_rocprofv3(workload_sdk)
        if rocprofv3 is not None:
            console_warning(
                "stack",
                "Two ROCm stacks detected. To stay on a single stack, profiling "
                f"will use the workload's own rocprofv3: {rocprofv3}",
            )
            return Rocprofv3Launch(rocprofv3=str(rocprofv3))
        raise IncompatibleRocmStackError(
            _missing_workload_rocprofv3_message(tool_sdk, workload_sdk)
        )

    forced = comgr_to_force(resolution)
    return Rocprofv3Launch(forced_comgr=str(forced) if forced is not None else None)
