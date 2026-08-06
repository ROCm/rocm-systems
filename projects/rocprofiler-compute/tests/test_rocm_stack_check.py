# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Tests for the ROCm stack check run before profiling.

A workload that brings its own ROCm is warned about, whether it arrives on a
search path or in the site-packages of the interpreter a launcher script names,
and the warning states how the copies fail and which of the two found each one.
A single ROCm is not reported, whether it is spread over several directories,
already preloaded, or the one holding the rocprofiler-sdk tool. RUNPATH tokens
are expanded, an unreadable directory does not end the search, and a check that
fails does not stop profiling. Any second copy turns the rocprofv3 signal
handlers off and says so once, unless the environment already sets them. A
failed run is explained from the profiler's output, whether LLVM or the
rocprofiler registry reported the conflict, and from the findings when an abort
names no cause. A failure that is not an abort is left to the profiler's own
output.
"""

import errno
import os
import sys
from pathlib import Path

import pytest

from utils import rocm_stack_check as stack_check
from utils.rocm_stack_check import (
    check_single_rocm_stack,
    disable_rocprof_signal_handlers,
)

_WORKLOAD_CMD = ["python3", "simple_net.py"]

# The directory $LIB expands to on this architecture.
_LIB_DIR = "lib64" if sys.maxsize > 2**32 else "lib"

# Captured from a run that loaded a wheel's comgr alongside the profiler's. The
# post-run check matches on these strings, which are emitted by LLVM rather than
# by ROCm.
_ABORT_OUTPUT = (
    "   INFO    |-> [rocprofiler-sdk] : CommandLine Error: Option "
    "'spirv-expand-step' registered more than once!\n"
    "   INFO    |-> [rocprofiler-sdk] LLVM ERROR: inconsistency in registered "
    "CommandLine options\n"
)

# Captured from a run that loaded a second rocprofiler-sdk. The registry emits
# this message, naming both paths.
_REGISTRATION_ABORT_OUTPUT = (
    "   |-> [rocprofiler-sdk] F0808 18:59:56.020553 246 registration.cpp:230] "
    "ROCPROFILER_REGISTER_LIBRARY is already set to "
    "'/venv/_rocm_sdk_core/lib/librocprofiler-sdk.so.1' (resolves to "
    "'/venv/_rocm_sdk_core/lib/librocprofiler-sdk.so.1'), not overriding with "
    "'/venv/_rocm_sdk_devel/lib/librocprofiler-sdk.so.1'\n"
)


def _caught_signal_output(signal_number: int) -> str:
    """Return the line rocprofv3's handler emits for ``signal_number``."""
    return (
        "   |-> [rocprofiler-sdk] W0809 13:41:09.848003 633 tool.cpp:4136] "
        "[PPID=1][PID=633][TID=633][rocprofv3_error_signal_handler] "
        f"rocprofv3 caught signal {signal_number}...\n"
    )


_SIGABRT_OUTPUT = _caught_signal_output(6)
_SIGSEGV_OUTPUT = _caught_signal_output(11)


def _make_lib(directory: Path, name: str, contents: bytes = b"\x7fELF") -> Path:
    """Create a placeholder shared library and return its path."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_bytes(contents)
    return path


def _workload_env(profiler_lib: Path, **entries: str) -> dict:
    """Return a workload environment whose ROCM_PATH holds ``profiler_lib``."""
    return {"ROCM_PATH": str(profiler_lib.parent.parent), **entries}


def _warning(warnings: list, *args, **kwargs) -> str:
    """Return the single warning ``check_single_rocm_stack`` emits."""
    check_single_rocm_stack(*args, **kwargs)
    assert len(warnings) == 1
    return warnings[0]


@pytest.fixture
def emitted_warnings(monkeypatch) -> list:
    """Collect the warnings the check emits."""
    collected = []
    monkeypatch.setattr(
        stack_check,
        "console_warning",
        lambda *argv: collected.append(str(argv[-1])),
    )
    return collected


@pytest.fixture
def emitted_logs(monkeypatch) -> list:
    """Collect the messages the check logs."""
    collected = []
    monkeypatch.setattr(
        stack_check,
        "console_log",
        lambda *argv, **kwargs: collected.append(str(argv[-1])),
    )
    return collected


@pytest.fixture
def profiler_comgr(tmp_path: Path, monkeypatch) -> Path:
    """The profiler's ROCm, holding the comgr the workload is compared against."""
    comgr = _make_lib(tmp_path / "opt/rocm/lib", "libamd_comgr.so.3", b"profiler-comgr")
    # Detection finds nothing unless a test supplies a workload library.
    monkeypatch.setattr(stack_check, "_site_packages", lambda interpreter, env: [])
    monkeypatch.setattr(stack_check, "_runpath_dirs", lambda elf_path, env: [])
    return comgr


def test_a_wheel_that_bundles_rocm_is_warned_about(
    profiler_comgr: Path, tmp_path: Path, emitted_warnings: list, monkeypatch
) -> None:
    site_packages = tmp_path / "venv/site-packages"
    bundled = _make_lib(site_packages / "torch/lib", "libamd_comgr.so.3", b"wheel")
    monkeypatch.setattr(
        stack_check, "_site_packages", lambda interpreter, env: [site_packages]
    )

    message = _warning(emitted_warnings, _WORKLOAD_CMD, _workload_env(profiler_comgr))

    assert str(profiler_comgr) in message
    assert str(bundled) in message
    assert "abort the run" in message
    # The remedy appears in the report that follows a failed run.
    assert "ROCM_PATH" not in message
    assert "ROCPROF_SIGNAL_HANDLERS=0" in message
    # site-packages is not searched by the loader, so the copy may not load.
    assert "workload, site-packages" in message
    assert "loaded only if the workload adds its directory" in message


def test_a_launcher_script_is_followed_to_its_interpreter(
    profiler_comgr: Path, tmp_path: Path, emitted_warnings: list, monkeypatch
) -> None:
    # A console script such as torchrun names its interpreter on a shebang
    # line, and that interpreter's site-packages hold the workload's ROCm.
    venv_bin = tmp_path / "venv/bin"
    interpreter = _make_lib(venv_bin, "python3")
    launcher = venv_bin / "torchrun"
    launcher.write_text(f"#!{interpreter}\n", encoding="utf-8")
    launcher.chmod(0o755)
    site_packages = tmp_path / "venv/site-packages"
    bundled = _make_lib(site_packages / "torch/lib", "libamd_comgr.so.3", b"wheel")
    monkeypatch.setattr(
        stack_check, "_site_packages", lambda interpreter, env: [site_packages]
    )

    message = _warning(
        emitted_warnings,
        ["torchrun", "train.py"],
        _workload_env(profiler_comgr, PATH=str(venv_bin)),
    )

    assert str(bundled) in message


def test_an_unreadable_directory_does_not_hide_a_conflict(
    profiler_comgr: Path, tmp_path: Path, emitted_warnings: list, monkeypatch
) -> None:
    # A search path routinely names directories the user cannot read. Mode bits
    # do not restrict uid 0, so the error is raised rather than provoked.
    unreadable = tmp_path / "unreadable"
    unreadable.mkdir()
    bundled = _make_lib(tmp_path / "wheel/lib", "libamd_comgr.so.3", b"wheel")
    readable_iterdir = Path.iterdir

    def iterdir(self: Path):
        if self == unreadable:
            raise PermissionError(errno.EACCES, "Permission denied", str(self))
        return readable_iterdir(self)

    monkeypatch.setattr(Path, "iterdir", iterdir)
    search_path = os.pathsep.join([str(unreadable), str(bundled.parent)])

    message = _warning(
        emitted_warnings,
        _WORKLOAD_CMD,
        _workload_env(profiler_comgr, LD_LIBRARY_PATH=search_path),
    )

    assert str(bundled) in message


def test_a_copy_the_loader_searches_for_is_told_apart_from_an_installed_one(
    profiler_comgr: Path, tmp_path: Path, emitted_warnings: list
) -> None:
    # LD_LIBRARY_PATH is searched, so this copy loads rather than merely
    # being installed, and the caveat about site-packages does not apply.
    on_search_path = _make_lib(tmp_path / "wheel/lib", "libamd_comgr.so.3", b"wheel")

    message = _warning(
        emitted_warnings,
        _WORKLOAD_CMD,
        _workload_env(profiler_comgr, LD_LIBRARY_PATH=str(on_search_path.parent)),
    )

    assert f"workload, search path : {on_search_path}" in message
    assert "site-packages" not in message


def test_the_profilers_own_directory_is_not_reported(
    profiler_comgr: Path, emitted_warnings: list
) -> None:
    # A search path may name the ROCm the profiler itself came from.
    check_single_rocm_stack(
        _WORKLOAD_CMD,
        _workload_env(profiler_comgr, LD_LIBRARY_PATH=str(profiler_comgr.parent)),
    )

    assert emitted_warnings == []


def test_a_preloaded_copy_is_not_reported(
    profiler_comgr: Path, tmp_path: Path, emitted_warnings: list
) -> None:
    # Preloading one copy is the resolution the warning recommends.
    preloaded = _make_lib(tmp_path / "wheel/lib", "libamd_comgr.so.3", b"wheel")

    check_single_rocm_stack(
        _WORKLOAD_CMD,
        _workload_env(profiler_comgr, LD_PRELOAD=str(preloaded)),
    )

    assert emitted_warnings == []


def test_the_sdk_tool_path_names_the_profilers_rocm(
    profiler_comgr: Path, tmp_path: Path, emitted_warnings: list
) -> None:
    # The tool the profiler loads into the workload can come from a ROCm other
    # than the one on ROCM_PATH.
    tool_root = tmp_path / "therock"
    tool = tool_root / "lib/rocprofiler-sdk/librocprofiler-sdk-tool.so"
    _make_lib(tool.parent, tool.name)
    tool_comgr = _make_lib(tool_root / "lib", "libamd_comgr.so.3", b"tool-comgr")

    check_single_rocm_stack(
        _WORKLOAD_CMD,
        _workload_env(profiler_comgr, LD_LIBRARY_PATH=str(tool_comgr.parent)),
        str(tool),
    )

    assert emitted_warnings == []


@pytest.mark.parametrize(
    "entry, expected",
    [
        ("$ORIGIN/../lib", "/app/bin/../lib"),
        ("${ORIGIN}/../lib", "/app/bin/../lib"),
        ("/opt/rocm/$LIB", f"/opt/rocm/{_LIB_DIR}"),
        ("/opt/rocm/lib/$PLATFORM", f"/opt/rocm/lib/{os.uname().machine}"),
    ],
    ids=["origin", "braced_origin", "lib", "platform"],
)
def test_runpath_tokens_are_expanded(entry: str, expected: str) -> None:
    readelf_output = (
        f" 0x000000000000001d (RUNPATH)            Library runpath: [{entry}]\n"
    )

    dirs = stack_check._parse_runpath_entries(readelf_output, "/app/bin")

    assert dirs == [Path(expected)]


@pytest.mark.parametrize(
    "profiler_lib_name, workload_lib_name",
    [
        ("librocprofiler-sdk.so.1.0.0", "librocprofiler-sdk.so.1.0.0"),
        ("librocprofiler-sdk.so.1.0.0", "librocprofiler-sdk.so.2.0.0"),
        ("librocprofiler-register.so.0.4.0", "librocprofiler-register.so.0.4.0"),
    ],
    ids=["same_version", "different_version", "register_library"],
)
def test_a_second_rocprofiler_copy_warns_how_it_fails(
    profiler_lib_name: str,
    workload_lib_name: str,
    tmp_path: Path,
    emitted_warnings: list,
    monkeypatch,
) -> None:
    # The registry holds one rocprofiler library and refuses a second one,
    # whatever version each carries.
    profiler_lib = _make_lib(
        tmp_path / "opt/rocm/lib", profiler_lib_name, b"profiler-sdk"
    )
    monkeypatch.setattr(stack_check, "_site_packages", lambda interpreter, env: [])
    monkeypatch.setattr(stack_check, "_runpath_dirs", lambda elf_path, env: [])
    wheel = _make_lib(tmp_path / "wheel/lib", workload_lib_name, b"wheel-sdk")
    env = _workload_env(profiler_lib, LD_LIBRARY_PATH=str(wheel.parent))

    message = _warning(emitted_warnings, _WORKLOAD_CMD, env)

    assert str(wheel) in message
    assert "cannot both register" in message


@pytest.mark.parametrize("layout", ["identical_copy", "hard_link", "symlink"])
def test_a_split_rocm_installation_is_not_reported(
    layout: str, tmp_path: Path, emitted_warnings: list, monkeypatch
) -> None:
    # TheRock splits one ROCm across _rocm_sdk_devel and _rocm_sdk_core, each
    # holding its own copy of the rocprofiler libraries.
    profiler_lib = _make_lib(
        tmp_path / "devel/lib", "librocprofiler-sdk.so.1", b"profiler-sdk"
    )
    workload_lib_dir = tmp_path / "core/lib"
    workload_lib_dir.mkdir(parents=True)
    workload_lib = workload_lib_dir / "librocprofiler-sdk.so.1"
    if layout == "identical_copy":
        workload_lib.write_bytes(profiler_lib.read_bytes())
    elif layout == "hard_link":
        os.link(profiler_lib, workload_lib)
    else:
        workload_lib.symlink_to(profiler_lib)
    monkeypatch.setattr(stack_check, "_site_packages", lambda interpreter, env: [])
    monkeypatch.setattr(stack_check, "_runpath_dirs", lambda elf_path, env: [])

    findings = check_single_rocm_stack(
        _WORKLOAD_CMD,
        _workload_env(profiler_lib, LD_LIBRARY_PATH=str(workload_lib_dir)),
    )

    assert findings.conflicts == ()
    assert emitted_warnings == []
    # A symlink names the profiler's own file, so only a separate file counts
    # as a second copy.
    assert findings.identical_copies == (() if layout == "symlink" else (workload_lib,))


def test_the_warning_reports_a_signal_handler_setting_it_leaves_alone(
    profiler_comgr: Path, tmp_path: Path, emitted_warnings: list, monkeypatch
) -> None:
    site_packages = tmp_path / "venv/site-packages"
    _make_lib(site_packages / "torch/lib", "libamd_comgr.so.3", b"wheel")
    monkeypatch.setattr(
        stack_check, "_site_packages", lambda interpreter, env: [site_packages]
    )
    env = _workload_env(profiler_comgr, ROCPROF_SIGNAL_HANDLERS="1")

    message = _warning(emitted_warnings, _WORKLOAD_CMD, env)

    assert "ROCPROF_SIGNAL_HANDLERS is already set" in message
    assert "ROCPROF_SIGNAL_HANDLERS=0" not in message


def test_a_second_copy_turns_the_signal_handlers_off() -> None:
    copy = Path("/venv/lib/libamd_comgr.so.3")
    conflicting = stack_check.StackFindings(
        conflicts=(
            stack_check.StackConflict(
                "libamd_comgr.so", Path("/opt/rocm/lib/libamd_comgr.so.3"), (copy,)
            ),
        )
    )
    identical = stack_check.StackFindings(identical_copies=(copy,))

    for findings in (conflicting, identical):
        env = {}
        disable_rocprof_signal_handlers(findings, env)
        assert env == {"ROCPROF_SIGNAL_HANDLERS": "0"}


def test_a_copy_that_is_not_a_conflict_still_reports_the_handler_change(
    emitted_logs: list,
) -> None:
    findings = stack_check.StackFindings(
        identical_copies=(Path("/venv/lib/libamd_comgr.so.3"),)
    )

    disable_rocprof_signal_handlers(findings, {})

    assert "ROCPROF_SIGNAL_HANDLERS=0" in emitted_logs[-1]


def test_a_conflict_reports_the_handler_change_once(emitted_logs: list) -> None:
    # The warning naming the conflict has already said the handlers go off.
    findings = stack_check.StackFindings(
        conflicts=(
            stack_check.StackConflict(
                "libamd_comgr.so",
                Path("/opt/rocm/lib/libamd_comgr.so.3"),
                (Path("/venv/lib/libamd_comgr.so.3"),),
            ),
        )
    )

    disable_rocprof_signal_handlers(findings, {})

    assert emitted_logs == []


def test_the_signal_handlers_are_left_on_without_a_second_copy() -> None:
    env = {}

    disable_rocprof_signal_handlers(stack_check.StackFindings(), env)
    disable_rocprof_signal_handlers(None, env)

    assert env == {}


def test_a_signal_handler_setting_in_the_environment_is_kept() -> None:
    findings = stack_check.StackFindings(
        identical_copies=(Path("/venv/lib/libamd_comgr.so.3"),)
    )
    env = {"ROCPROF_SIGNAL_HANDLERS": "1"}

    disable_rocprof_signal_handlers(findings, env)

    assert env == {"ROCPROF_SIGNAL_HANDLERS": "1"}


def test_a_check_that_fails_does_not_stop_profiling(monkeypatch) -> None:
    def failing_check(*args, **kwargs):
        raise RuntimeError("detection failed")

    monkeypatch.setattr(stack_check, "_find_stacks", failing_check)

    assert check_single_rocm_stack(_WORKLOAD_CMD, {}).conflicts == ()


def test_an_abort_from_two_llvm_libraries_is_explained(
    profiler_comgr: Path, monkeypatch
) -> None:
    # The check compares what is on disk, so it can miss a copy the workload
    # only reaches once it runs, leaving the profiler's copy the only one to
    # name.
    reported = []
    monkeypatch.setattr(
        stack_check, "console_error", lambda message, **kwargs: reported.append(message)
    )
    findings = stack_check.StackFindings(profiler_comgr, ())

    stack_check.explain_failed_run(_ABORT_OUTPUT, findings)

    assert str(profiler_comgr) in reported[0]
    assert "found only the profiler's" in reported[0]
    assert "ROCM_PATH" in reported[0]


def test_a_registration_failure_is_explained(tmp_path: Path, monkeypatch) -> None:
    # The registry reports its refusal in its own words, and the message
    # repeats the paths the check named before the run.
    reported = []
    monkeypatch.setattr(
        stack_check, "console_error", lambda message, **kwargs: reported.append(message)
    )
    profiler_lib = tmp_path / "opt/rocm/lib/librocprofiler-sdk.so.1.3.2"
    workload_lib = tmp_path / "wheel/lib/librocprofiler-sdk.so.2.0.0"
    findings = stack_check.StackFindings(
        None,
        (
            stack_check.StackConflict(
                "librocprofiler-sdk.so",
                profiler_lib,
                (workload_lib,),
            ),
        ),
    )

    stack_check.explain_failed_run(_REGISTRATION_ABORT_OUTPUT, findings)

    assert len(reported) == 1
    assert "could not register" in reported[0]
    assert str(workload_lib) in reported[0]
    assert "ROCM_PATH" in reported[0]


def _comgr_findings(tmp_path: Path) -> "stack_check.StackFindings":
    """Return findings holding one conflicting pair of comgr copies."""
    return stack_check.StackFindings(
        None,
        (
            stack_check.StackConflict(
                "libamd_comgr.so",
                tmp_path / "opt/rocm/lib/libamd_comgr.so.3",
                (tmp_path / "wheel/lib/libamd_comgr.so.3",),
            ),
        ),
    )


def test_an_unexplained_abort_names_the_copies(tmp_path: Path, monkeypatch) -> None:
    # A conflict can abort the run without reaching the profiler's output.
    reported = []
    monkeypatch.setattr(
        stack_check, "console_error", lambda message, **kwargs: reported.append(message)
    )
    findings = _comgr_findings(tmp_path)

    stack_check.explain_failed_run(_SIGABRT_OUTPUT, findings)

    assert "does not name a cause" in reported[0]
    assert str(findings.conflicts[0].workload_libs[0]) in reported[0]
    # Each remedy states the condition it depends on, since neither works
    # unconditionally.
    assert "ROCM_PATH" in reported[0]
    assert "rocprofiler-sdk tool" in reported[0]
    assert "same soname" in reported[0]


@pytest.mark.parametrize(
    "output",
    ["Segmentation fault", _SIGSEGV_OUTPUT, "ValueError: no such metric"],
    ids=["no_signal_reported", "another_signal", "workload_error"],
)
def test_a_failure_that_is_not_an_abort_is_not_blamed_on_the_copies(
    output: str, tmp_path: Path, monkeypatch
) -> None:
    # The copies are named only for the abort the conflicts produce, so an
    # unrelated failure is left to the profiler's own output.
    reported = []
    monkeypatch.setattr(
        stack_check, "console_error", lambda message, **kwargs: reported.append(message)
    )

    stack_check.explain_failed_run(output, _comgr_findings(tmp_path))

    assert reported == []


def test_a_failure_without_findings_is_left_unexplained(monkeypatch) -> None:
    reported = []
    monkeypatch.setattr(
        stack_check, "console_error", lambda message, **kwargs: reported.append(message)
    )

    stack_check.explain_failed_run("Segmentation fault")
    stack_check.explain_failed_run("Segmentation fault", stack_check.StackFindings())

    assert reported == []


def test_an_abort_without_findings_names_no_copies(monkeypatch) -> None:
    # Attaching to a running process skips the pre-run check, so there is
    # nothing to name.
    reported = []
    monkeypatch.setattr(
        stack_check, "console_error", lambda message, **kwargs: reported.append(message)
    )

    stack_check.explain_failed_run(_ABORT_OUTPUT)

    assert "libamd_comgr" in reported[0]
    assert "The profiler's" not in reported[0]
