# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Detection of a workload that supplies its own ROCm alongside the profiler's.

Two copies of ``libamd_comgr`` in one process abort the run. Two copies of a
rocprofiler library cannot both register with rocprofiler.

A conflict is a warning, and profiling continues. A copy that holds the
profiler's own library is not a second ROCm, so it is not a conflict and
produces no warning.

Any second copy, conflicting or not, turns the rocprofv3 signal handlers off
for the run, so that an abort ends the process. That change is reported.
``run_prof`` explains a failed run from the findings this check returns.
"""

from __future__ import annotations

import filecmp
import os
import re
import shutil
import signal
import subprocess
import sys
import textwrap
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from utils.logger import console_debug, console_error, console_log, console_warning
from utils.utils_common import resolve_rocm_library_path

COMGR_LIB_NAME = "libamd_comgr.so"
ROCPROFILER_REGISTER_LIB_NAME = "librocprofiler-register.so"
ROCPROFILER_SDK_LIB_NAME = "librocprofiler-sdk.so"

# Set to 0 to stop rocprofv3 from replacing the workload's signal handlers.
ROCPROF_SIGNAL_HANDLERS_ENV = "ROCPROF_SIGNAL_HANDLERS"

# Libraries compared between the profiler's ROCm and the workload's.
_STACK_LIB_NAMES = (
    COMGR_LIB_NAME,
    ROCPROFILER_REGISTER_LIB_NAME,
    ROCPROFILER_SDK_LIB_NAME,
)

# Libraries that register with rocprofiler.
_REGISTRATION_SENSITIVE_LIB_NAMES = (
    ROCPROFILER_REGISTER_LIB_NAME,
    ROCPROFILER_SDK_LIB_NAME,
)

# Library directories of a ROCm installation.
_ROCM_LIB_DIR_NAMES = ("lib", "lib64")

# The profiler's ROCm installation when ROCM_PATH is not set.
_DEFAULT_ROCM_PATH = "/opt/rocm"

# Directory of compiled Python, which holds no libraries.
_PYCACHE_DIR_NAME = "__pycache__"

# Fragments of the messages LLVM emits when two libraries containing it are
# loaded into one process.
_LLVM_CONFLICT_SIGNATURES = (
    "registered more than once",
    "inconsistency in registered CommandLine options",
)

# Fragment of the message the rocprofiler registry emits when it refuses a
# second library.
_REGISTRATION_CONFLICT_SIGNATURES = ("ROCPROFILER_REGISTER_LIBRARY is already set to",)

# The signal rocprofv3 reports in the message its handler emits. Both conflicts
# end the run through abort, so any other signal has another cause.
_CAUGHT_SIGNAL_RE = re.compile(r"rocprofv3 caught signal (\d+)")

_PYTHON_EXECUTABLE_RE = re.compile(r"^python[0-9.]*$")

# Tokens the loader expands in RPATH and RUNPATH entries.
_DYNAMIC_TOKEN_RE = re.compile(r"\$\{(ORIGIN|LIB|PLATFORM)\}|\$(ORIGIN|LIB|PLATFORM)\b")

# Bytes read from a script when looking for its shebang line.
_SHEBANG_READ_LIMIT = 512

# Prints the site-packages directories of the interpreter that runs it.
_SITE_PACKAGES_SCRIPT = """
import site
directories = []
for getter in ("getsitepackages", "getusersitepackages"):
    try:
        value = getattr(site, getter)()
    except Exception:
        continue
    directories.extend([value] if isinstance(value, str) else list(value))
print("\\n".join(directories))
"""

_SUBPROCESS_TIMEOUT_SEC = 20

# The logger prefixes only the first line with the log level; the remaining
# lines carry this indent.
_BODY_INDENT = "  "

# Message wrap width, leaving room for the indent within 80 columns.
_WRAP_WIDTH = 76

_HEADLINE = "The workload supplies its own ROCm alongside the profiler's."

# How two copies of each library fail, shown beside the paths that name them.
_COMGR_NOTE = "two copies in one process abort the run"
_ROCPROFILER_NOTE = "two copies cannot both register with rocprofiler"

_LLVM_CONFLICT_CAUSE = textwrap.fill(
    "Two copies of libamd_comgr are the usual cause, one supplied by the "
    "workload and one by the profiler.",
    width=_WRAP_WIDTH,
)

_REGISTRATION_CONFLICT_CAUSE = textwrap.fill(
    "Two copies of a rocprofiler library are the usual cause, one supplied by "
    "the workload and one by the profiler.",
    width=_WRAP_WIDTH,
)

_POSSIBLE_CONFLICT_CAUSE = textwrap.fill(
    "The workload and the profiler each supply a copy of the same ROCm "
    "library, which can abort the run.",
    width=_WRAP_WIDTH,
)

# How each path was found. The loader searches LD_LIBRARY_PATH, LD_PRELOAD,
# RPATH and RUNPATH; it does not search site-packages.
_PROFILER_LABEL = "profiler"
_SEARCH_PATH_LABEL = "workload, search path"
_INSTALLED_LABEL = "workload, site-packages"

_INSTALLED_COPY_NOTE = textwrap.fill(
    "A copy in site-packages is installed rather than found on a search path. "
    "It is loaded only if the workload adds its directory to one.",
    width=_WRAP_WIDTH - len(_BODY_INDENT),
)

# How a conflict is resolved. Each remedy carries the condition it depends on:
# the profiler loads its tool from ROCM_PATH, and preloading merges two copies
# only when one satisfies both references to the library.
_SECOND_COPY_RESOLUTION = textwrap.fill(
    "Point ROCM_PATH at the workload's ROCm, if that installation also "
    "provides the rocprofiler-sdk tool. Or set LD_PRELOAD to the workload's "
    "copy of the conflicting library, if it carries the same soname as the "
    "profiler's copy.",
    width=_WRAP_WIDTH,
)

# What the run does about a second copy, whether or not it conflicts.
_SIGNAL_HANDLERS_OFF = textwrap.fill(
    f"Setting {ROCPROF_SIGNAL_HANDLERS_ENV}=0 so that an abort ends the run "
    "instead of leaving it unresponsive. Profiling continues.",
    width=_WRAP_WIDTH,
)

_SIGNAL_HANDLERS_KEPT = textwrap.fill(
    f"{ROCPROF_SIGNAL_HANDLERS_ENV} is already set and is left unchanged. "
    "Profiling continues.",
    width=_WRAP_WIDTH,
)


def check_single_rocm_stack(
    workload_cmd: list[str],
    env: dict[str, str],
    tool_path: Optional[str] = None,
) -> StackFindings:
    """Warn when the workload and the profiler each supply the same ROCm library.

    ``tool_path`` is the rocprofiler-sdk tool the profiler loads. Profiling
    continues, and the findings are returned for the report that follows a
    failed run.
    """
    try:
        findings = _find_stacks(workload_cmd, env, tool_path)
        if not findings.conflicts:
            console_debug(
                "stack", "no conflicting copy found for the libraries compared"
            )
            return findings
        message = _conflict_message(
            list(findings.conflicts), ROCPROF_SIGNAL_HANDLERS_ENV in env
        )
    except Exception:
        console_debug("stack", f"ROCm stack check failed:\n{traceback.format_exc()}")
        return StackFindings()
    console_warning("stack", message)
    return findings


def disable_rocprof_signal_handlers(
    findings: Optional[StackFindings],
    env: dict[str, str],
) -> None:
    """Turn the rocprofv3 signal handlers off in ``env``.

    rocprofv3 replaces the workload's signal handlers with its own, and an
    abort can re-enter them instead of ending the process. The handlers are
    left on when the workload supplies no second copy of a compared library,
    and when ``env`` already sets ``ROCPROF_SIGNAL_HANDLERS``.
    """
    if findings is None or not findings.has_second_copy():
        return
    if ROCPROF_SIGNAL_HANDLERS_ENV in env:
        console_debug(
            "stack", f"{ROCPROF_SIGNAL_HANDLERS_ENV} is already set; not changed"
        )
        return
    if findings.conflicts:
        # The warning naming the conflicts has already reported this.
        console_debug("stack", f"setting {ROCPROF_SIGNAL_HANDLERS_ENV}=0")
    else:
        console_log(
            "stack",
            "the workload supplies a copy of the profiler's ROCm; setting "
            f"{ROCPROF_SIGNAL_HANDLERS_ENV}=0 so that an abort ends the run",
        )
    env[ROCPROF_SIGNAL_HANDLERS_ENV] = "0"


def explain_failed_run(
    output: str,
    findings: Optional[StackFindings] = None,
) -> None:
    """Report the library conflict that can account for a failed run.

    ``findings`` come from the check run before profiling. The profiler's
    output identifies the conflict where it reports one. Where it does not,
    the copies the check found are named as a possible cause only if the run
    was aborted, since a run that ended any other way has another cause. A
    failure while reporting is suppressed.
    """
    try:
        reported = False
        if output_indicates_llvm_conflict(output):
            console_error(_llvm_conflict_message(findings), exit=False)
            reported = True
        if output_indicates_registration_conflict(output):
            console_error(_registration_conflict_message(findings), exit=False)
            reported = True
        if (
            not reported
            and output_indicates_abort(output)
            and findings is not None
            and findings.conflicts
        ):
            console_error(
                _possible_conflict_message(list(findings.conflicts)), exit=False
            )
    except Exception:
        console_debug("stack", f"conflict report failed:\n{traceback.format_exc()}")


def output_indicates_llvm_conflict(output: str) -> bool:
    """Return True if ``output`` reports two libraries containing LLVM."""
    return any(signature in output for signature in _LLVM_CONFLICT_SIGNATURES)


def output_indicates_registration_conflict(output: str) -> bool:
    """Return True if ``output`` reports a rocprofiler library failing to register."""
    return any(signature in output for signature in _REGISTRATION_CONFLICT_SIGNATURES)


def output_indicates_abort(output: str) -> bool:
    """Return True if ``output`` reports the run ending in an abort."""
    match = _CAUGHT_SIGNAL_RE.search(output)
    return match is not None and int(match.group(1)) == signal.SIGABRT


def _llvm_conflict_message(findings: Optional[StackFindings]) -> str:
    """Return the message for a conflict reported in the profiler's output."""
    conflicts = _comgr_conflicts(findings)
    return _build_message(
        "Two libraries containing LLVM in one process aborted the run.\n"
        + _LLVM_CONFLICT_CAUSE,
        _stacks_section(conflicts) or _profiler_comgr_section(findings),
        _SECOND_COPY_RESOLUTION,
    )


def _registration_conflict_message(findings: Optional[StackFindings]) -> str:
    """Return the message for a registration failure in the profiler's output."""
    conflicts = _rocprofiler_conflicts(findings)
    return _build_message(
        "A rocprofiler library could not register, and the run failed.\n"
        + _REGISTRATION_CONFLICT_CAUSE,
        _stacks_section(conflicts),
        _SECOND_COPY_RESOLUTION,
    )


def _possible_conflict_message(conflicts: list[StackConflict]) -> str:
    """Return the message for an abort the profiler's output does not explain."""
    return _build_message(
        "The run was aborted, and the profiler's output does not name a cause.\n"
        + _POSSIBLE_CONFLICT_CAUSE,
        _stacks_section(conflicts),
        _SECOND_COPY_RESOLUTION,
    )


def _rocprofiler_conflicts(
    findings: Optional[StackFindings],
) -> list[StackConflict]:
    """Return the conflicts in the libraries that register with rocprofiler."""
    if findings is None:
        return []
    return [
        conflict
        for conflict in findings.conflicts
        if conflict.lib_name in _REGISTRATION_SENSITIVE_LIB_NAMES
    ]


def _comgr_conflicts(findings: Optional[StackFindings]) -> list[StackConflict]:
    """Return the conflicts in the library that carries LLVM."""
    if findings is None:
        return []
    return [
        conflict
        for conflict in findings.conflicts
        if conflict.lib_name == COMGR_LIB_NAME
    ]


def _profiler_comgr_section(findings: Optional[StackFindings]) -> str:
    """Return the section naming the profiler's comgr."""
    if findings is None or findings.profiler_comgr is None:
        return ""
    return "\n".join([
        f"The check found only the profiler's {_display_name(COMGR_LIB_NAME)}",
        f"  {findings.profiler_comgr}",
    ])


def _conflict_message(conflicts: list[StackConflict], handlers_kept: bool) -> str:
    """Return the warning naming the conflicting libraries.

    ``handlers_kept`` reports that the environment already sets
    ``ROCPROF_SIGNAL_HANDLERS``, which the run leaves unchanged.
    """
    return _build_message(
        _HEADLINE,
        _stacks_section(conflicts),
        _SIGNAL_HANDLERS_KEPT if handlers_kept else _SIGNAL_HANDLERS_OFF,
    )


def _stacks_section(conflicts: list[StackConflict]) -> str:
    """Return the section naming the profiler's and the workload's libraries."""
    if not conflicts:
        return ""
    return _conflict_lines(
        "Libraries supplied by both the workload and the profiler", conflicts
    )


def _display_name(lib_name: str) -> str:
    """Return ``lib_name`` without its file extension."""
    return Path(lib_name).stem


def _conflict_lines(heading: str, conflicts: list[StackConflict]) -> str:
    """Return ``heading`` above the paths of each conflict.

    Each path carries the evidence that found it, since a copy in
    site-packages is weaker evidence than one the loader searches for.
    """
    width = _label_width(conflicts)
    lines = [heading]
    for conflict in conflicts:
        name = _display_name(conflict.lib_name)
        lines.append(f"  {name} ({_failure_note(conflict)})")
        lines.append(f"    {_PROFILER_LABEL:<{width}} : {conflict.profiler_lib}")
        for workload_lib in conflict.workload_libs:
            label = _workload_label(conflict, workload_lib)
            lines.append(f"    {label:<{width}} : {workload_lib}")
    if any(conflict.installed_libs for conflict in conflicts):
        lines.append("")
        lines.append(textwrap.indent(_INSTALLED_COPY_NOTE, _BODY_INDENT))
    return "\n".join(lines)


def _workload_label(conflict: StackConflict, workload_lib: Path) -> str:
    """Return the label naming how ``workload_lib`` was found."""
    if workload_lib in conflict.installed_libs:
        return _INSTALLED_LABEL
    return _SEARCH_PATH_LABEL


def _label_width(conflicts: list[StackConflict]) -> int:
    """Return the width that aligns every label the conflicts use."""
    labels = {_PROFILER_LABEL, _SEARCH_PATH_LABEL}
    if any(conflict.installed_libs for conflict in conflicts):
        labels.add(_INSTALLED_LABEL)
    return max(len(label) for label in labels)


def _failure_note(conflict: StackConflict) -> str:
    """Return how a second copy of ``conflict``'s library fails."""
    if conflict.lib_name == COMGR_LIB_NAME:
        return _COMGR_NOTE
    return _ROCPROFILER_NOTE


def _build_message(headline: str, *sections: str) -> str:
    """Return the headline above the sections.

    Each section is indented and separated from the next by a blank line. Any
    headline lines after the first are indented with the sections.
    """
    first_line, _, rest = headline.partition("\n")
    lead = first_line
    if rest:
        lead = f"{first_line}\n{textwrap.indent(rest, _BODY_INDENT)}"
    body = "\n\n".join(section for section in sections if section)
    return f"{lead}\n\n{textwrap.indent(body, _BODY_INDENT)}"


def _find_stacks(
    workload_cmd: list[str],
    env: dict[str, str],
    tool_path: Optional[str] = None,
) -> StackFindings:
    """Return the profiler's comgr and the libraries the workload also supplies."""
    rocm_root = _profiler_rocm_root(env, tool_path)
    profiler_libs: dict[str, Path] = {}
    for lib_name in _STACK_LIB_NAMES:
        profiler_lib = _profiler_lib(lib_name, rocm_root)
        if profiler_lib is None:
            console_debug("stack", f"no {lib_name} under {rocm_root}; not compared")
            continue
        profiler_libs[lib_name] = profiler_lib
    if not profiler_libs:
        console_warning(
            "stack",
            f"the profiler's ROCm was not found under {rocm_root}; "
            "the workload's ROCm was not compared with it",
        )
        return StackFindings()

    lib_names = list(profiler_libs)
    candidates = _workload_candidates(workload_cmd, env, lib_names)
    preloaded = {_real_path(path) for path in _preloaded_libs(env, lib_names)}
    conflicts: list[StackConflict] = []
    identical_copies: list[Path] = []
    for lib_name, profiler_lib in profiler_libs.items():
        reachable = candidates.by_lib_name[lib_name]
        identical_copies.extend(
            _identical_copies(reachable, profiler_lib, rocm_root, preloaded)
        )
        workload_libs = _workload_libs(reachable, profiler_lib, rocm_root, preloaded)
        if not workload_libs:
            continue
        conflicts.append(
            StackConflict(
                lib_name,
                profiler_lib,
                tuple(workload_libs),
                installed_libs=tuple(
                    path
                    for path in workload_libs
                    if _real_path(path) not in candidates.on_search_path
                ),
            )
        )
    return StackFindings(
        profiler_libs.get(COMGR_LIB_NAME), tuple(conflicts), tuple(identical_copies)
    )


def _workload_candidates(
    workload_cmd: list[str],
    env: dict[str, str],
    lib_names: list[str],
) -> _Candidates:
    """Return every copy of ``lib_names`` the workload can reach, keyed by name."""
    searched = (
        _libs_on_library_path(env, lib_names)
        + _preloaded_libs(env, lib_names)
        + _libs_in_runpath(workload_cmd, env, lib_names)
    )
    installed = _libs_in_site_packages(workload_cmd, env, lib_names)
    found: dict[str, list[Path]] = {lib_name: [] for lib_name in lib_names}
    for path in searched + installed:
        lib_name = _matching_lib_name(path.name, lib_names)
        if lib_name is not None:
            found[lib_name].append(path)
    return _Candidates(found, {_real_path(path) for path in searched})


def _matching_lib_name(file_name: str, lib_names: list[str]) -> Optional[str]:
    """Return the name in ``lib_names`` that ``file_name`` is a copy of, if any."""
    for lib_name in lib_names:
        if file_name.startswith(lib_name):
            return lib_name
    return None


def _profiler_rocm_root(env: dict[str, str], tool_path: Optional[str] = None) -> Path:
    """Return the ROCm installation the profiler runs against.

    The rocprofiler-sdk tool at ``tool_path`` is loaded into the workload, so
    the installation holding it takes precedence over ROCM_PATH. An empty
    ROCM_PATH is treated as unset.
    """
    installation = _installation_root(Path(tool_path)) if tool_path else None
    if installation is not None:
        return installation
    rocm_path = env.get("ROCM_PATH", "").strip() or _DEFAULT_ROCM_PATH
    return Path(_real_path(Path(rocm_path)))


def _installation_root(lib_path: Path) -> Optional[Path]:
    """Return the installation root above the library directory holding ``lib_path``."""
    for parent in Path(_real_path(lib_path)).parents:
        if parent.name in _ROCM_LIB_DIR_NAMES:
            return parent.parent
    return None


def _profiler_lib(lib_name: str, rocm_root: Path) -> Optional[Path]:
    """Return the profiler's copy of ``lib_name`` under ``rocm_root``."""
    for dir_name in _ROCM_LIB_DIR_NAMES:
        resolved = resolve_rocm_library_path(str(rocm_root / dir_name / lib_name))
        if resolved and Path(resolved).is_file():
            return Path(_real_path(Path(resolved)))
    return None


def _workload_libs(
    candidates: list[Path],
    profiler_lib: Path,
    rocm_root: Path,
    preloaded: set[str],
) -> list[Path]:
    """Return the workload's copies of the library, excluding the profiler's.

    A copy under the profiler's ROCm root, or one that resolves to the same
    file or holds identical contents, belongs to the same stack. A copy named
    in LD_PRELOAD is excluded, since the workload and the profiler both load
    it.
    """
    return [
        path
        for path in _unique_paths(candidates)
        if not _under_directory(path, rocm_root)
        and not _same_library(path, profiler_lib)
        and _real_path(path) not in preloaded
    ]


def _identical_copies(
    candidates: list[Path],
    profiler_lib: Path,
    rocm_root: Path,
    preloaded: set[str],
) -> list[Path]:
    """Return the workload's copies that hold the profiler's library."""
    return [
        path
        for path in _unique_paths(candidates)
        if not _under_directory(path, rocm_root)
        and _same_library(path, profiler_lib)
        and _real_path(path) not in preloaded
    ]


def _libs_on_library_path(env: dict[str, str], lib_names: list[str]) -> list[Path]:
    """Return the ``lib_names`` copies in the directories on LD_LIBRARY_PATH."""
    found: list[Path] = []
    for directory in env.get("LD_LIBRARY_PATH", "").split(os.pathsep):
        if directory:
            found.extend(_libs_in_directory(Path(directory), lib_names))
    return found


def _preloaded_libs(env: dict[str, str], lib_names: list[str]) -> list[Path]:
    """Return the ``lib_names`` copies named on LD_PRELOAD."""
    found: list[Path] = []
    for entry in env.get("LD_PRELOAD", "").replace(os.pathsep, " ").split():
        entry_path = Path(entry)
        if _matching_lib_name(entry_path.name, lib_names) and entry_path.is_file():
            found.append(entry_path)
    return found


def _libs_in_site_packages(
    workload_cmd: list[str],
    env: dict[str, str],
    lib_names: list[str],
) -> list[Path]:
    """Return ``lib_names`` copies shipped in a Python workload's site-packages."""
    interpreter = _workload_interpreter(workload_cmd, env)
    if interpreter is None:
        return []
    found: list[Path] = []
    for site_dir in _unique_paths(_site_packages(interpreter, env)):
        if not site_dir.is_dir():
            continue
        found.extend(_libs_under_directory(site_dir, lib_names))
    return found


def _libs_under_directory(directory: Path, lib_names: list[str]) -> list[Path]:
    """Return the ``lib_names`` copies anywhere under ``directory``."""

    def report_unreadable_directory(err: OSError) -> None:
        console_debug("stack", f"stopped reading {err.filename}: {err}")

    found: list[Path] = []
    for parent, dir_names, file_names in os.walk(
        directory, onerror=report_unreadable_directory
    ):
        dir_names[:] = [name for name in dir_names if name != _PYCACHE_DIR_NAME]
        for file_name in file_names:
            if _matching_lib_name(file_name, lib_names) is None:
                continue
            path = Path(parent) / file_name
            if path.is_file():
                found.append(path)
    return found


def _libs_in_runpath(
    workload_cmd: list[str],
    env: dict[str, str],
    lib_names: list[str],
) -> list[Path]:
    """Return ``lib_names`` copies in the workload executable's RPATH or RUNPATH."""
    if not workload_cmd:
        return []
    executable = _which(workload_cmd[0], env)
    if executable is None:
        return []
    found: list[Path] = []
    for directory in _runpath_dirs(executable, env):
        found.extend(_libs_in_directory(directory, lib_names))
    return found


def _libs_in_directory(directory: Path, lib_names: list[str]) -> list[Path]:
    """Return the ``lib_names`` copies, versioned or not, directly in ``directory``."""
    try:
        entries = list(directory.iterdir())
    except OSError:
        return []
    return sorted(
        path
        for path in entries
        if _matching_lib_name(path.name, lib_names) and path.is_file()
    )


def _workload_interpreter(
    workload_cmd: list[str],
    env: dict[str, str],
) -> Optional[Path]:
    """Return the workload's Python interpreter, or None if it is not Python."""
    if not workload_cmd:
        return None
    executable = _which(workload_cmd[0], env)
    if executable is None:
        return None
    if _is_python_executable(executable):
        return executable
    return _shebang_interpreter(executable, env) or _sibling_interpreter(executable)


def _is_python_executable(executable: Path) -> bool:
    """Return True if ``executable`` is named like a Python interpreter."""
    names = (executable.name, Path(_real_path(executable)).name)
    return any(_PYTHON_EXECUTABLE_RE.match(name) for name in names)


def _shebang_interpreter(script: Path, env: dict[str, str]) -> Optional[Path]:
    """Return the Python interpreter named on ``script``'s shebang line.

    A console script installed by pip names its virtual environment's
    interpreter this way.
    """
    name = _shebang_interpreter_name(script)
    if name is None:
        return None
    candidate = Path(name)
    interpreter = candidate if candidate.is_absolute() else _which(name, env)
    if interpreter is None or not _is_python_executable(interpreter):
        return None
    return interpreter if interpreter.is_file() else None


def _shebang_interpreter_name(script: Path) -> Optional[str]:
    """Return the interpreter named on ``script``'s first line, if it has one."""
    try:
        with open(script, "rb") as handle:
            first_line = handle.readline(_SHEBANG_READ_LIMIT)
    except OSError:
        return None
    if not first_line.startswith(b"#!"):
        return None
    words = first_line[2:].decode("utf-8", errors="replace").split()
    if words and Path(words[0]).name == "env":
        words = [word for word in words[1:] if not word.startswith("-")]
    return words[0] if words else None


def _sibling_interpreter(executable: Path) -> Optional[Path]:
    """Return the ``python`` beside ``executable`` in a virtual environment."""
    sibling = executable.parent / "python"
    return sibling if sibling.is_file() else None


def _site_packages(interpreter: Path, env: dict[str, str]) -> list[Path]:
    """Return the interpreter's site-packages and per-user site directories."""
    output = _command_output([str(interpreter), "-c", _SITE_PACKAGES_SCRIPT], env)
    if output is None:
        return []
    return [Path(line) for line in output.splitlines() if line.strip()]


def _runpath_dirs(elf_path: Path, env: dict[str, str]) -> list[Path]:
    """Return the RPATH and RUNPATH directories of an ELF."""
    readelf = _which("readelf", env)
    if readelf is None:
        console_debug("stack", "readelf not found; skipping RUNPATH inspection")
        return []
    output = _command_output([str(readelf), "-d", str(elf_path)], env)
    if output is None:
        return []
    return _parse_runpath_entries(output, str(elf_path.resolve().parent))


def _parse_runpath_entries(readelf_output: str, origin: str) -> list[Path]:
    """Return the directories listed in ``readelf -d`` RPATH and RUNPATH entries."""
    dirs: list[Path] = []
    for line in readelf_output.splitlines():
        if "(RPATH)" in line or "(RUNPATH)" in line:
            dirs.extend(_runpath_entry_dirs(line, origin))
    return dirs


def _runpath_entry_dirs(line: str, origin: str) -> list[Path]:
    """Return the directories in one bracketed ``readelf -d`` entry."""
    match = re.search(r"\[(.*)\]", line)
    if match is None:
        return []
    expanded = (
        _expand_dynamic_tokens(raw_dir, origin)
        for raw_dir in match.group(1).split(os.pathsep)
    )
    return [Path(directory) for directory in expanded if directory]


def _expand_dynamic_tokens(raw_dir: str, origin: str) -> str:
    """Return ``raw_dir`` with the loader's dynamic string tokens replaced."""
    token_values = {
        "ORIGIN": origin,
        "LIB": "lib64" if sys.maxsize > 2**32 else "lib",
        "PLATFORM": os.uname().machine,
    }
    return _DYNAMIC_TOKEN_RE.sub(
        lambda match: token_values[match.group(1) or match.group(2)], raw_dir
    )


def _which(name: str, env: dict[str, str]) -> Optional[Path]:
    """Return the executable ``name`` resolves to on the environment's PATH."""
    found = shutil.which(name, path=env.get("PATH", os.defpath))
    return Path(found) if found else None


def _command_output(command: list[str], env: dict[str, str]) -> Optional[str]:
    """Return the standard output of ``command``, or None if it does not succeed."""
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            timeout=_SUBPROCESS_TIMEOUT_SEC,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as err:
        console_debug("stack", f"{command[0]} failed: {err}")
        return None
    return completed.stdout if completed.returncode == 0 else None


def _unique_paths(paths: list[Path]) -> list[Path]:
    """Return ``paths`` de-duplicated by real path, preserving order."""
    seen: set[str] = set()
    unique: list[Path] = []
    for path in paths:
        key = _real_path(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def _real_path(path: Path) -> str:
    """Return the canonical path with symlinks resolved."""
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


def _same_library(first: Path, second: Path) -> bool:
    """Return True if both paths refer to the same shared library.

    The paths match when they resolve to the same file, are hard links to it,
    or hold identical contents.
    """
    real_first, real_second = _real_path(first), _real_path(second)
    if real_first == real_second:
        return True
    try:
        if os.path.samestat(Path(real_first).stat(), Path(real_second).stat()):
            return True
        return filecmp.cmp(real_first, real_second, shallow=False)
    except OSError:
        return False


@dataclass(frozen=True)
class _Candidates:
    """Copies of the compared libraries the workload can reach.

    ``on_search_path`` holds the resolved paths of the copies found on a
    directory the loader searches. The rest were found in site-packages, which
    the loader does not search.
    """

    by_lib_name: dict[str, list[Path]]
    on_search_path: set[str]


@dataclass(frozen=True)
class StackConflict:
    """A ROCm library that the workload supplies in addition to the profiler.

    ``installed_libs`` are the copies in ``workload_libs`` that were found in
    site-packages rather than on a directory the loader searches.
    """

    lib_name: str
    profiler_lib: Path
    workload_libs: tuple[Path, ...]
    installed_libs: tuple[Path, ...] = ()


@dataclass(frozen=True)
class StackFindings:
    """What the check found before profiling started.

    ``identical_copies`` hold the profiler's own library and do not conflict.
    """

    profiler_comgr: Optional[Path] = None
    conflicts: tuple[StackConflict, ...] = ()
    identical_copies: tuple[Path, ...] = ()

    def has_second_copy(self) -> bool:
        """Return whether the workload supplies a copy of a compared library."""
        return bool(self.conflicts or self.identical_copies)
