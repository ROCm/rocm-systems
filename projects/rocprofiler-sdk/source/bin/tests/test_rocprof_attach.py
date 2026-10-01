#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""GPU-free unit tests for the rocprof-attach launcher.

The target process is simulated with a fake procfs tree (PROC_DIR) whose <pid>/root points at a
fake filesystem, so no real process is inspected or attached to.
"""

import glob
import os
import re
import sys

import pytest

PID = 123

# ctest fails any test whose output matches this (cmake/rocprofiler_options.cmake)
CTEST_FAIL_REGEX = re.compile(
    r"threw an exception|Permission denied|failed with error code|Subprocess aborted|"
    r"Failed to resolve rocprofiler-sdk shared library path"
)

OTHER_LIBDIR = "/opt/rocm-other/lib"
SELF_LIBDIR = "/opt/rocm-self/lib"
SELF_TOOL = f"{SELF_LIBDIR}/rocprofiler-sdk/librocprofiler-sdk-tool.so"


class _FakeFunction:
    def __init__(self):
        self.calls = []
        self.restype = None
        self.argtypes = None

    def __call__(self, *args):
        self.calls.append(args)
        return 0


class _FakeAttachLibrary:
    def __init__(self):
        self.rocattach_attach = _FakeFunction()
        self.rocattach_attach_tree = _FakeFunction()
        self.rocattach_detach = _FakeFunction()
        self.rocattach_detach_tree = _FakeFunction()


class FakeTarget:
    """A fake /proc/<pid> plus the filesystem the target process sees."""

    def __init__(self, tmp_path, pid=PID):
        self.pid = pid
        self.proc = tmp_path / "proc"
        self.fs = tmp_path / "fs"
        self.pid_dir = self.proc / f"{pid}"
        (self.pid_dir / "task" / f"{pid}").mkdir(parents=True)
        (self.pid_dir / "task" / f"{pid}" / "comm").write_bytes(b"app\n")
        self.fs.mkdir()
        (self.pid_dir / "root").symlink_to(self.fs)
        self.maps = []

    def host(self, path):
        return self.fs / path.lstrip("/")

    def add_file(self, path, content=b"\x7fELF"):
        dst = self.host(path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(content)
        return path

    def add_link(self, path, target):
        dst = self.host(path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.symlink_to(target)
        return path

    def add_thread(self, tid, name):
        (self.pid_dir / "task" / f"{tid}").mkdir()
        (self.pid_dir / "task" / f"{tid}" / "comm").write_bytes(name.encode() + b"\n")

    def map(self, path, deleted=False):
        if isinstance(path, str):
            path = path.encode()
        suffix = b" (deleted)" if deleted else b""
        self.maps.append(
            b"7f0000000000-7f0000001000 r-xp 00000000 08:01 1234       " + path + suffix
        )
        (self.pid_dir / "maps").write_bytes(b"\n".join(self.maps) + b"\n")

    def unreadable_root(self):
        (self.pid_dir / "root").unlink()
        (self.pid_dir / "root").symlink_to(self.proc / "missing")

    def add_install(self, libdir, version="1.4.1", marker=True, tool=True):
        """Adds a rocprofiler-sdk installation to the target filesystem."""
        soversion = version.split(".")[0]
        content = b"\x7fELF rocp-bg-attach" if marker else b"\x7fELF"
        attach = self.add_file(
            f"{libdir}/librocprofiler-sdk-attach.so.{version}", content
        )
        self.add_link(
            f"{libdir}/librocprofiler-sdk-attach.so.{soversion}",
            f"librocprofiler-sdk-attach.so.{version}",
        )
        core = self.add_file(f"{libdir}/librocprofiler-sdk.so.{version}")
        if tool:
            tooldir = f"{libdir}/rocprofiler-sdk"
            self.add_file(f"{tooldir}/librocprofiler-sdk-tool.so.{version}")
            self.add_link(
                f"{tooldir}/librocprofiler-sdk-tool.so.{soversion}",
                f"librocprofiler-sdk-tool.so.{version}",
            )
            self.add_link(
                f"{tooldir}/librocprofiler-sdk-tool.so",
                f"librocprofiler-sdk-tool.so.{soversion}",
            )
        return (attach, core)

    def run_sdk(self, libdir, version="1.4.1", **kwargs):
        """Adds an installation and maps its libraries into the target."""
        attach, core = self.add_install(libdir, version, **kwargs)
        self.map(attach)
        self.map(core)


@pytest.fixture
def target(rocprof_attach, monkeypatch, tmp_path):
    monkeypatch.setattr(rocprof_attach, "PROC_DIR", str(tmp_path / "proc"))
    monkeypatch.delenv("ROCP_TOOL_LIBRARIES", raising=False)
    monkeypatch.delenv("ROCPROF_ATTACH_TOOL_LIBRARY", raising=False)
    return FakeTarget(tmp_path)


@pytest.fixture
def fake_rocattach(rocprof_attach, monkeypatch):
    loaded_paths = []
    fake_library = _FakeAttachLibrary()

    def load_library(path):
        loaded_paths.append(path)
        return fake_library

    monkeypatch.setattr(rocprof_attach.ctypes, "CDLL", load_library)
    monkeypatch.setattr(rocprof_attach.signal, "signal", lambda *_: None)
    monkeypatch.setattr(rocprof_attach.time, "sleep", lambda *_: None)
    fake_library.loaded_paths = loaded_paths
    return fake_library


def _output(capsys):
    """Returns (stdout, stderr) and checks that ctest would not flag them as a failure."""
    out, err = capsys.readouterr()
    assert not CTEST_FAIL_REGEX.search(out + err), out + err
    return (out, err)


def _main(rocprof_attach, *args):
    return rocprof_attach.main(
        ["--attach", f"{PID}", "--attach-duration-msec", "0", *args]
    )


# ------------------------------------------------------------------------------------------------
# launcher
# ------------------------------------------------------------------------------------------------


def test_default_attach_library_uses_runtime_soname(
    rocprof_attach, target, fake_rocattach, tmp_path, capsys
):
    tool_library = tmp_path / "tool.so"
    tool_library.touch()
    target.add_thread(PID + 1, "rocp-bg-attach")

    result = _main(
        rocprof_attach,
        "--attach-tool-library",
        str(tool_library),
        "--attach-children=false",
    )

    expected_library = os.path.join(
        rocprof_attach.ROCM_DIR,
        "lib",
        f"librocprofiler-sdk-rocattach.so.{rocprof_attach.ROCPROFILER_SDK_SOVERSION}",
    )
    assert result == 0
    assert rocprof_attach.ROCPROFILER_SDK_SOVERSION.isdigit()
    assert fake_rocattach.loaded_paths == [expected_library]
    assert fake_rocattach.rocattach_attach.calls == [(PID,)]
    assert fake_rocattach.rocattach_detach.calls == [(PID,)]
    assert not fake_rocattach.rocattach_attach_tree.calls
    assert not fake_rocattach.rocattach_detach_tree.calls
    _output(capsys)


def test_default_tool_library_matches_rocprofv3(rocprof_attach, rocprofv3):
    """With no target installation found, the tool path is the one rocprofv3 resolves."""
    rocm_dir = os.path.dirname(os.path.dirname(os.path.realpath(rocprofv3.__file__)))
    assert rocm_dir == rocprof_attach.ROCM_DIR
    expected = rocprofv3.resolve_library_path(
        f"{rocm_dir}/lib/rocprofiler-sdk/librocprofiler-sdk-tool.so", {}
    )
    assert rocprof_attach.default_tool_library() == expected


def test_same_install_keeps_rocprofv3_tool_string(
    rocprof_attach, rocprofv3, target, capsys
):
    """A target running this installation gets the exact string earlier releases passed.

    The target's root is the real filesystem and it maps this installation's attach library, so
    the selection must reproduce rocprofv3's string byte-for-byte (the target refuses a reattach
    whose tool library string differs from the first attachment's).
    """
    attach_libs = sorted(
        glob.glob(f"{rocprof_attach.ROCM_DIR}/lib/librocprofiler-sdk-attach.so.*.*.*")
    )
    if not attach_libs:
        pytest.skip("this installation has no rocprofiler-sdk attach library")

    (target.pid_dir / "root").unlink()
    (target.pid_dir / "root").symlink_to("/")
    target.map(os.path.realpath(attach_libs[-1]))
    target.add_thread(PID + 1, "rocp-bg-attach")

    sdk = rocprof_attach.find_target_sdk(PID)
    path, source = rocprof_attach.select_tool_library(PID, sdk)

    expected = rocprofv3.resolve_library_path(
        f"{rocprof_attach.ROCM_DIR}/lib/rocprofiler-sdk/librocprofiler-sdk-tool.so", {}
    )
    assert (path, source) == (expected, "target")
    assert rocprof_attach.is_same_install(PID, sdk)

    rocprof_attach.warn_option_support(PID, sdk, source)
    _, err = _output(capsys)
    assert "WARNING" not in err


# ------------------------------------------------------------------------------------------------
# tool library selection
# ------------------------------------------------------------------------------------------------


def test_uses_target_installation_tool(rocprof_attach, target, capsys):
    target.run_sdk(OTHER_LIBDIR)
    target.add_install(SELF_LIBDIR, "1.5.0")

    sdk = rocprof_attach.find_target_sdk(PID)
    assert sdk.libdir == OTHER_LIBDIR
    assert sdk.version == "1.4.1"

    path, source = rocprof_attach.select_tool_library(PID, sdk, SELF_TOOL)
    assert (path, source) == (
        f"{OTHER_LIBDIR}/rocprofiler-sdk/librocprofiler-sdk-tool.so",
        "target",
    )
    out, err = _output(capsys)
    assert "tool library of the target's rocprofiler-sdk v1.4.1 installation" in out
    assert "WARNING" not in err


def test_uses_target_tool_soname_when_unversioned_link_missing(
    rocprof_attach, target, capsys
):
    target.run_sdk(OTHER_LIBDIR)
    target.host(f"{OTHER_LIBDIR}/rocprofiler-sdk/librocprofiler-sdk-tool.so").unlink()

    sdk = rocprof_attach.find_target_sdk(PID)
    path, source = rocprof_attach.select_tool_library(PID, sdk, SELF_TOOL)
    assert (path, source) == (
        f"{OTHER_LIBDIR}/rocprofiler-sdk/librocprofiler-sdk-tool.so.1",
        "target",
    )
    _output(capsys)


def test_unverifiable_target_filesystem_uses_target_tool(rocprof_attach, target, capsys):
    target.run_sdk(OTHER_LIBDIR)
    target.unreadable_root()

    sdk = rocprof_attach.find_target_sdk(PID)
    path, source = rocprof_attach.select_tool_library(PID, sdk, SELF_TOOL)
    assert (path, source) == (
        f"{OTHER_LIBDIR}/rocprofiler-sdk/librocprofiler-sdk-tool.so",
        "target",
    )
    _, err = _output(capsys)
    assert "could not verify" in err


@pytest.mark.parametrize(
    "files, expected",
    [
        (["1.4.1", "1.3.0", "1.10.0"], "1.4.1"),  # the attach library's version
        (["1.3.0", "1.10.0", "1.9.2"], "1.10.0"),  # highest version otherwise
    ],
)
def test_uses_versioned_target_tool_when_links_missing(
    rocprof_attach, target, capsys, files, expected
):
    target.run_sdk(OTHER_LIBDIR, tool=False)
    tooldir = f"{OTHER_LIBDIR}/rocprofiler-sdk"
    for itr in files:
        target.add_file(f"{tooldir}/librocprofiler-sdk-tool.so.{itr}")
    # dangling versioned links are not candidates
    target.add_link(f"{tooldir}/librocprofiler-sdk-tool.so.9.9.9", "missing")

    sdk = rocprof_attach.find_target_sdk(PID)
    path, source = rocprof_attach.select_tool_library(PID, sdk, SELF_TOOL)
    assert (path, source) == (
        f"{tooldir}/librocprofiler-sdk-tool.so.{expected}",
        "target",
    )
    _, err = _output(capsys)
    assert "links are missing" in err


def test_same_library_keeps_fallback_string(rocprof_attach, target, capsys):
    """When the target's tool is the attacher's tool, the attacher's string is kept."""
    target.run_sdk(OTHER_LIBDIR)
    fallback = f"{OTHER_LIBDIR}/../lib/rocprofiler-sdk/librocprofiler-sdk-tool.so"

    sdk = rocprof_attach.find_target_sdk(PID)
    path, source = rocprof_attach.select_tool_library(PID, sdk, fallback)
    assert (path, source) == (fallback, "target")
    _output(capsys)


def test_missing_target_tool_falls_back(rocprof_attach, target, capsys):
    target.run_sdk(OTHER_LIBDIR, tool=False)
    target.add_install(SELF_LIBDIR, "1.5.0")

    sdk = rocprof_attach.find_target_sdk(PID)
    path, source = rocprof_attach.select_tool_library(PID, sdk, SELF_TOOL)
    assert (path, source) == (SELF_TOOL, "fallback")
    _, err = _output(capsys)
    assert f"installation in {OTHER_LIBDIR}" in err
    assert f"using {SELF_TOOL} instead" in err

    rocprof_attach.warn_option_support(PID, sdk, source)
    _, err = _output(capsys)
    assert "skipped with warnings" in err
    assert "thread trace is disabled" in err


def test_target_without_sdk_uses_fallback(rocprof_attach, target, capsys):
    target.add_install(SELF_LIBDIR, "1.5.0")
    target.map(target.add_file("/usr/lib/libc.so.6"))

    sdk = rocprof_attach.find_target_sdk(PID)
    assert sdk.maps_readable and sdk.libdir is None
    path, source = rocprof_attach.select_tool_library(PID, sdk, SELF_TOOL)
    assert (path, source) == (SELF_TOOL, "fallback")

    rocprof_attach.warn_option_support(PID, sdk, source)
    out, err = _output(capsys)
    assert "has not loaded rocprofiler-sdk" in out
    assert "WARNING" not in err


def test_unreadable_maps_uses_fallback(rocprof_attach, target, capsys):
    target.add_install(SELF_LIBDIR, "1.5.0")

    sdk = rocprof_attach.find_target_sdk(PID)
    assert not sdk.maps_readable
    path, source = rocprof_attach.select_tool_library(PID, sdk, SELF_TOOL)
    assert (path, source) == (SELF_TOOL, "fallback")

    rocprof_attach.warn_option_support(PID, sdk, source)
    _, err = _output(capsys)
    assert "could not read the memory maps" in err
    assert "different rocprofiler-sdk" not in err


def test_no_tool_library_is_an_error(rocprof_attach, target, fake_rocattach, capsys):
    target.run_sdk(OTHER_LIBDIR, tool=False)
    target.add_thread(PID + 1, "rocp-bg-attach")

    result = _main(
        rocprof_attach,
        "--attach-children=false",
        "--fallback-tool-library",
        SELF_TOOL,
    )
    assert result == 1
    assert not fake_rocattach.loaded_paths
    assert "ROCPROF_ATTACH_TOOL_LIBRARY" not in os.environ
    _, err = _output(capsys)
    assert "ERROR: no tool library was found" in err


def test_different_install_warns_about_option_support(rocprof_attach, target, capsys):
    target.run_sdk(OTHER_LIBDIR)

    sdk = rocprof_attach.find_target_sdk(PID)
    assert not rocprof_attach.is_same_install(PID, sdk)
    rocprof_attach.warn_option_support(PID, sdk, "target")
    _, err = _output(capsys)
    assert "ignores options it does not know" in err
    assert "--sys-trace" in err


def test_mixed_sdk_directories_prefer_attach_library(rocprof_attach, target, capsys):
    attach, _ = target.add_install(OTHER_LIBDIR)
    _, core = target.add_install(SELF_LIBDIR, "1.5.0")
    target.map(core)
    target.map(attach)

    sdk = rocprof_attach.find_target_sdk(PID)
    assert sdk.libdir == OTHER_LIBDIR
    _, err = _output(capsys)
    assert "more than one directory" in err


def test_non_utf8_maps(rocprof_attach, target, capsys):
    target.map(b"/opt/caf\xe9/lib/libfoo.so")
    target.run_sdk(OTHER_LIBDIR)

    sdk = rocprof_attach.find_target_sdk(PID)
    assert sdk.maps_readable
    assert sdk.libdir == OTHER_LIBDIR
    _output(capsys)


def test_loaded_tool_warning(rocprof_attach, target, capsys):
    target.run_sdk(OTHER_LIBDIR)
    earlier = target.add_file("/opt/custom/libmytool.so")
    target.map(f"{OTHER_LIBDIR}/rocprofiler-sdk/librocprofiler-sdk-tool.so.1.4.1")
    sdk = rocprof_attach.find_target_sdk(PID)

    # the tool that is already loaded: no warning
    selected = f"{OTHER_LIBDIR}/rocprofiler-sdk/librocprofiler-sdk-tool.so"
    rocprof_attach.warn_loaded_tool(PID, sdk, selected)
    _, err = _output(capsys)
    assert "WARNING" not in err

    rocprof_attach.warn_loaded_tool(PID, sdk, earlier)
    _, err = _output(capsys)
    assert "already has" in err


# ------------------------------------------------------------------------------------------------
# attachability
# ------------------------------------------------------------------------------------------------


def test_attach_thread_present(rocprof_attach, target, capsys):
    target.run_sdk(OTHER_LIBDIR, marker=False)
    target.add_thread(PID + 1, "rocp-bg-attach")

    sdk = rocprof_attach.find_target_sdk(PID)
    rocprof_attach.check_attachable(PID, sdk, attach_children=False)
    _, err = _output(capsys)
    assert "WARNING" not in err


def test_attach_library_without_thread_support_is_an_error(
    rocprof_attach, target, fake_rocattach, capsys
):
    target.run_sdk(OTHER_LIBDIR, version="1.1.0", marker=False)

    result = _main(rocprof_attach, "--attach-children=false")
    assert result == 1
    assert not fake_rocattach.loaded_paths
    _, err = _output(capsys)
    assert "predates support for attaching" in err
    assert "/opt/rocm-other/bin" in err
    assert "v1.1.0" in err


def test_attach_library_without_thread_support_in_tree_mode(
    rocprof_attach, target, capsys
):
    target.run_sdk(OTHER_LIBDIR, marker=False)

    sdk = rocprof_attach.find_target_sdk(PID)
    rocprof_attach.check_attachable(PID, sdk, attach_children=True)
    _, err = _output(capsys)
    assert "descendant processes will be attached" in err


def test_attach_thread_not_started(rocprof_attach, target, capsys):
    target.run_sdk(OTHER_LIBDIR)

    sdk = rocprof_attach.find_target_sdk(PID)
    rocprof_attach.check_attachable(PID, sdk, attach_children=False)
    _, err = _output(capsys)
    assert "unless it starts first" in err


def test_attach_library_not_loaded(rocprof_attach, target, capsys):
    target.map(target.add_file("/usr/lib/libc.so.6"))

    sdk = rocprof_attach.find_target_sdk(PID)
    rocprof_attach.check_attachable(PID, sdk, attach_children=False)
    _, err = _output(capsys)
    assert "ROCP_TOOL_ATTACH=1" in err


@pytest.mark.parametrize("state", ["deleted", "unreadable"])
def test_attach_library_not_inspectable(rocprof_attach, target, capsys, state):
    attach, core = target.add_install(OTHER_LIBDIR, marker=False)
    target.map(attach, deleted=(state == "deleted"))
    target.map(core)
    if state == "unreadable":
        target.host(attach).unlink()

    sdk = rocprof_attach.find_target_sdk(PID)
    rocprof_attach.check_attachable(PID, sdk, attach_children=False)
    _output(capsys)


# ------------------------------------------------------------------------------------------------
# environment and user tool libraries
# ------------------------------------------------------------------------------------------------


def test_user_tool_library_is_used_verbatim(
    rocprof_attach, target, fake_rocattach, capsys
):
    target.run_sdk(OTHER_LIBDIR)
    target.add_thread(PID + 1, "rocp-bg-attach")
    user = "/opt/custom/libmissing.so:librelative.so"

    result = _main(rocprof_attach, "--attach-children=false", "-t", user)
    assert result == 0
    assert os.environ["ROCPROF_ATTACH_TOOL_LIBRARY"] == user
    assert fake_rocattach.rocattach_attach.calls == [(PID,)]
    out, err = _output(capsys)
    assert "'/opt/custom/libmissing.so' is not visible" in err
    assert "'librelative.so' will be resolved by the target's dynamic loader" in out


def test_rocp_tool_libraries_is_ignored(
    rocprof_attach, target, fake_rocattach, monkeypatch, capsys
):
    target.run_sdk(OTHER_LIBDIR)
    target.add_thread(PID + 1, "rocp-bg-attach")
    monkeypatch.setenv("ROCP_TOOL_LIBRARIES", "/opt/custom/libother.so")

    result = _main(rocprof_attach, "--attach-children=false")
    assert result == 0
    assert "ROCP_TOOL_LIBRARIES" not in os.environ
    assert (
        os.environ["ROCPROF_ATTACH_TOOL_LIBRARY"]
        == f"{OTHER_LIBDIR}/rocprofiler-sdk/librocprofiler-sdk-tool.so"
    )
    _, err = _output(capsys)
    assert "ignoring ROCP_TOOL_LIBRARIES=/opt/custom/libother.so" in err


def test_attach_tree_uses_root_selection(rocprof_attach, target, fake_rocattach, capsys):
    target.run_sdk(OTHER_LIBDIR)
    target.add_thread(PID + 1, "rocp-bg-attach")

    result = _main(rocprof_attach, "--fallback-tool-library", SELF_TOOL)
    assert result == 0
    assert (
        os.environ["ROCPROF_ATTACH_TOOL_LIBRARY"]
        == f"{OTHER_LIBDIR}/rocprofiler-sdk/librocprofiler-sdk-tool.so"
    )
    assert fake_rocattach.rocattach_attach_tree.calls == [(PID,)]
    assert fake_rocattach.rocattach_detach_tree.calls == [(PID,)]
    _output(capsys)


if __name__ == "__main__":
    sys.exit(pytest.main(["-x", __file__] + sys.argv[1:]))
