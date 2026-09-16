#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
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

"""Fixtures for the rocprofv3-doctor unit and integration tests.

The unit tests here never touch the real system: they drive checks through a
``FakeAccessor`` that answers from a synthetic description of a machine. That
is what lets the whole suite run in CI with no GPU, no ROCm install, and no
particular kernel configuration.
"""

import os
import sys

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--doctor-path",
        action="store",
        default=None,
        help="path to the rocprofv3-doctor executable under test",
    )
    parser.addoption(
        "--rocprofv3-path",
        action="store",
        default=None,
        help="path to the rocprofv3 executable under test",
    )
    parser.addoption(
        "--rocprofv3-package-dir",
        action="store",
        default=None,
        help="directory containing the rocprofv3 Python package",
    )


def _repo_root():
    # tests/rocprofv3-doctor/conftest.py -> repo root
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


@pytest.fixture(scope="session", autouse=True)
def rocprofv3_package(request):
    """Make the rocprofv3 package importable, preferring the built copy."""
    candidates = []
    configured = request.config.getoption("--rocprofv3-package-dir")
    if configured:
        candidates.append(configured)
    candidates.append(os.path.join(_repo_root(), "source", "lib", "python"))

    for candidate in candidates:
        if os.path.isdir(candidate) and candidate not in sys.path:
            sys.path.insert(0, candidate)
    return candidates


@pytest.fixture
def doctor_path(request):
    """Path to the rocprofv3-doctor script, or the source-tree copy."""
    configured = request.config.getoption("--doctor-path")
    if configured:
        return configured
    return os.path.join(_repo_root(), "source", "bin", "rocprofv3-doctor.py")


@pytest.fixture
def rocprofv3_path(request):
    """Path to the rocprofv3 script, or the source-tree copy."""
    configured = request.config.getoption("--rocprofv3-path")
    if configured:
        return configured
    return os.path.join(_repo_root(), "source", "bin", "rocprofv3.py")


class FakeAccessor(object):
    """A SystemAccessor stand-in driven by a synthetic machine description.

    Every method mirrors the real ``SystemAccessor`` signature but answers from
    the dicts passed to the constructor, so a check can be exercised against a
    machine state that would be impossible or destructive to produce for real.
    """

    def __init__(
        self,
        rocm_root="/fake/rocm",
        files=None,
        dirs=None,
        file_contents=None,
        env=None,
        runs=None,
        groups=None,
        group_gids=None,
        group_members=None,
        username="tester",
        uid=1000,
        cap_eff=0,
        free_bytes=100 * 1024 * 1024 * 1024,
        writable_dirs=None,
        modules=None,
        cannot_write=None,
        tool_version=None,
        python_version=(3, 8, 10),
        sqlite_ok=True,
        pid=4242,
        links=None,
        module_dirs=None,
        rocm_root_source=None,
    ):
        self._pid = pid
        self.rocm_root = rocm_root
        self.rocm_root_source = rocm_root_source
        self.calls = []
        self.temp_dirs = []
        self.removed_trees = []
        # symlinks, as {link path: absolute target}; a link may name a file or
        # a directory, and links may chain
        self._links = dict(links or {})
        # find_module_dir answers, as {module name: package directory}
        self._module_dirs = dict(module_dirs or {})
        self.tool_version = tool_version
        self._files = set(files or [])
        self._dirs = set(dirs or [])
        self._file_contents = dict(file_contents or {})
        self._env = dict(env or {})
        self._runs = dict(runs or {})
        self._groups = list(groups or [])
        self._group_gids = dict(group_gids or {})
        self._group_members = dict(group_members or {})
        self._username = username
        self._uid = uid
        self._cap_eff = cap_eff
        self._free_bytes = free_bytes
        self._writable_dirs = writable_dirs
        self._cannot_write = set(cannot_write or [])
        self._modules = dict(modules or {})
        self._python_version = python_version
        self._sqlite_ok = sqlite_ok
        # every path with contents is implicitly an existing file
        for path in self._file_contents:
            self._files.add(path)

    # -- filesystem ---------------------------------------------------
    def path_exists(self, path):
        path = self.realpath(path)
        return path in self._files or path in self._dirs

    def path_is_dir(self, path):
        return self.realpath(path) in self._dirs

    def path_readable(self, path):
        return self.path_exists(path) and path not in self._cannot_write

    def path_writable(self, path):
        return self.path_exists(path) and path not in self._cannot_write

    def path_executable(self, path):
        return self.path_exists(path) and path not in self._cannot_write

    def path_is_link(self, path):
        return os.path.normpath(path) in self._links

    def readlink(self, path):
        return self._links.get(os.path.normpath(path), "")

    def realpath(self, path):
        # resolve the longest linked prefix repeatedly, like os.path.realpath
        path = os.path.normpath(path)
        for _ in range(32):
            parts = path.split(os.sep)
            for index in range(len(parts), 0, -1):
                prefix = os.sep.join(parts[:index]) or os.sep
                if prefix in self._links:
                    rest = parts[index:]
                    path = os.path.normpath(os.path.join(self._links[prefix], *rest))
                    break
            else:
                return path
        return path

    def abspath(self, path):
        return os.path.normpath(path if os.path.isabs(path) else "/cwd/" + path)

    def path_within(self, path, root):
        resolved_path = self.realpath(path)
        resolved_root = self.realpath(root)
        if resolved_path == resolved_root:
            return True
        return resolved_path.startswith(resolved_root.rstrip(os.sep) + os.sep)

    def dirname(self, path):
        return os.path.dirname(path)

    def basename(self, path):
        return os.path.basename(path)

    def join(self, *parts):
        return os.path.join(*parts)

    def glob(self, pattern):
        import fnmatch as _fnmatch

        # Like glob.glob: a wildcard never crosses a "/", and a literal
        # directory part is followed through symlinks.
        directory, base = os.path.split(pattern)
        if not any(char in directory for char in "*?["):
            real_dir = self.realpath(directory)
            matches = []
            for path in sorted(self._files | self._dirs | set(self._links)):
                if os.path.dirname(path) == real_dir and _fnmatch.fnmatch(
                    os.path.basename(path), base
                ):
                    matches.append(os.path.join(directory, os.path.basename(path)))
            return sorted(set(matches))

        depth = pattern.count(os.sep)
        matches = []
        for path in sorted(self._files | self._dirs):
            if path.count(os.sep) == depth and _fnmatch.fnmatch(path, pattern):
                matches.append(path)
        return matches

    def read_file(self, path):
        return self._file_contents.get(path)

    def touch_probe(self, path):
        if self._writable_dirs is not None:
            return path in self._writable_dirs
        return path not in self._cannot_write

    def free_bytes(self, path):
        return self._free_bytes

    def make_temp_dir(self, prefix="rocprofv3-doctor-"):
        path = "/tmp/{}fake".format(prefix)
        self._dirs.add(path)
        self.temp_dirs.append(path)
        return path

    def remove_tree(self, path):
        self._dirs.discard(path)
        self.removed_trees.append(path)

    # -- environment --------------------------------------------------
    def getenv(self, key, default=None):
        return self._env.get(key, default)

    def environ_items(self):
        return sorted(self._env.items())

    # -- subprocess ---------------------------------------------------
    def run(self, cmd, timeout=10, env=None):
        self.calls.append((list(cmd), dict(env or {})))
        for key in (tuple(cmd), " ".join(cmd), cmd[-1] if cmd else ""):
            if key in self._runs:
                return self._runs[key]
        for key, value in self._runs.items():
            if isinstance(key, str) and any(key in part for part in cmd):
                return value
        from rocprofv3.doctor_env import RUN_SPAWN_FAILED

        return (RUN_SPAWN_FAILED, "", "not configured in FakeAccessor")

    # -- user / group -------------------------------------------------
    def getpid(self):
        return self._pid

    def getuid(self):
        return self._uid

    def getgid(self):
        return self._uid

    def getgroups(self):
        return list(self._groups)

    def get_group_gid(self, group_name):
        return self._group_gids.get(group_name)

    def get_group_members(self, group_name):
        return self._group_members.get(group_name)

    def get_username(self):
        return self._username

    # -- capabilities -------------------------------------------------
    def get_cap_eff(self):
        return self._cap_eff

    # -- python -------------------------------------------------------
    def python_version(self):
        return self._python_version

    def python_executable(self):
        return "/usr/bin/python3"

    def can_import(self, module_name):
        if module_name in self._modules:
            value = self._modules[module_name]
            if value is None:
                return (False, "No module named {!r}".format(module_name))
            return (True, value)
        return (False, "No module named {!r}".format(module_name))

    def module_file(self, module_name):
        if module_name in self._modules and self._modules[module_name] is not None:
            return "/fake/site-packages/{}/__init__.py".format(module_name)
        return None

    def find_module_dir(self, module_name):
        return self._module_dirs.get(module_name)

    def sqlite_usable(self):
        return self._sqlite_ok

    # -- library resolution -------------------------------------------
    def ldconfig_entries(self):
        return {}

    def which(self, name):
        searched = []
        for directory in (self.getenv("PATH", "") or "").split(os.pathsep):
            if not directory:
                continue
            candidate = os.path.join(directory, name)
            searched.append(candidate)
            if self.path_exists(candidate) and self.path_executable(candidate):
                return (candidate, searched)
        return (None, searched)

    def library_search_dirs(self):
        dirs = []
        for entry in (self.getenv("LD_LIBRARY_PATH", "") or "").split(":"):
            if entry:
                dirs.append(entry)
        for suffix in ("lib", "lib64", "lib/rocprofiler-sdk"):
            dirs.append(os.path.join(self.rocm_root, suffix))
        return dirs

    def find_library(self, name):
        searched = []
        for directory in self.library_search_dirs():
            searched.append(directory)
            candidate = os.path.join(directory, name)
            if self.path_exists(candidate):
                return (candidate, searched)
        return (None, searched)


def make_healthy_accessor(**overrides):
    """A FakeAccessor describing a fully working ROCm machine."""
    root = "/fake/rocm"
    files = [
        root + "/lib/librocprofiler-sdk.so",
        root + "/lib/librocprofiler-sdk-roctx.so",
        root + "/lib/libhsa-runtime64.so",
        root + "/lib/libamdhip64.so",
        root + "/lib/libhsa-amd-aqlprofile64.so",
        root + "/lib/librocprofiler-register.so",
        root + "/lib/librocprofiler-sdk-rocattach.so",
        root + "/lib/rocprofiler-sdk/librocprofiler-sdk-tool.so",
        root + "/lib/rocprofiler-sdk/librocprofv3-list-avail.so",
        root + "/lib/rocprofiler-sdk/librocprofiler-sdk-tool-kokkosp.so",
        root + "/lib/rocprofiler-sdk/librocprof-trace-decoder.so",
        root + "/share/rocprofiler-sdk/basic_counters.xml",
        root + "/bin/rocprofv3",
        root + "/bin/rocprofv3-avail",
        root + "/bin/rocprof-attach",
        root + "/bin/rocpd",
        "/dev/kfd",
        "/dev/dri/renderD128",
        "/sys/class/kfd/kfd/topology/nodes/0/properties",
        "/sys/class/kfd/kfd/topology/nodes/1/properties",
    ]
    dirs = [
        root,
        root + "/lib",
        root + "/bin",
        root + "/share/rocprofiler-sdk",
        "/sys/class/kfd/kfd/topology/nodes/0",
        "/sys/class/kfd/kfd/topology/nodes/1",
        "/tmp",
        "/sys/module/amdgpu",
    ]
    file_contents = {
        "/sys/module/amdgpu/version": "6.16.13\n",
        "/proc/sys/kernel/perf_event_paranoid": "2\n",
        "/sys/class/kfd/kfd/topology/nodes/0/properties": (
            "cpu_cores_count 24\nsimd_count 0\ngfx_target_version 0\n"
        ),
        "/sys/class/kfd/kfd/topology/nodes/1/properties": (
            "cpu_cores_count 0\nsimd_count 256\ngfx_target_version 90000\n"
        ),
    }
    kwargs = dict(
        rocm_root=root,
        files=files,
        dirs=dirs,
        file_contents=file_contents,
        groups=[44, 109],
        group_gids={"render": 109, "video": 44},
        group_members={"render": ["tester"], "video": ["tester"]},
        modules={
            "sqlite3": "",
            "ctypes": "",
            "importlib": "",
            "yaml": "6.0",
            "pandas": "2.0.0",
            "otf2": "3.0",
            "rocprofv3": "",
        },
        env={},
    )
    # every library loads (runtime.libraries-load); a test's own runs= adds to
    # this rather than replacing it
    runs = {"import ctypes": (0, "", "")}
    runs.update(overrides.pop("runs", None) or {})
    kwargs["runs"] = runs
    kwargs.update(overrides)
    return FakeAccessor(**kwargs)
