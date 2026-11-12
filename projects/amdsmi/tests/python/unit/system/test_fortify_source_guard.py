#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Guard that ``-D_FORTIFY_SOURCE`` is only passed to optimized builds.

glibc warns when the macro is defined without optimization, and the -Werror
AI NIC library then fails to build. Runs cmake_modules/help_package.cmake in a
throwaway project, so no GPU or full amdsmi configure is needed.
"""

import functools
import os
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[4]
HELP_PACKAGE = REPO_ROOT / "cmake_modules" / "help_package.cmake"

FORTIFY_MACRO = "_FORTIFY_SOURCE"
FORTIFY_FLAG = "-D" + FORTIFY_MACRO + "=2"
STACK_PROTECTOR_FLAG = "-fstack-protector-all"
CAST_ALIGN_FLAG = "-Wcast-align"
OPTIMIZED_LEVELS = ("-O", "-O1", "-O2", "-O3", "-O4", "-Os", "-Og", "-Oz", "-Ofast")
UNOPTIMIZED_LEVEL = "-O0"
TAB_SEPARATED_OPTIMIZED_CXXFLAGS = "-g\t-O2\t-pipe"
RELEASE_FLAGS_ENDING_IN_O2 = "-DCMAKE_CXX_FLAGS_RELEASE=-O2"
OPTIMIZED_BUILD_TYPES = ("release", "MinSizeRel")
CUSTOM_BUILD_TYPE = "Custom"
CMAKE_TIMEOUT_SECONDS = 120

# Clang only: on GCC the -Wtrampolines set in generic_package() overwrites
# CMAKE_CXX_FLAGS in PARENT_SCOPE and drops the fortify flag.
CLANG_NAMES = ("amdclang++", "clang++")
KEPT_ENV_VARS = ("PATH", "HOME")
EXECUTABLE_MODE = 0o755

PROBE_CMAKELISTS = """\
cmake_minimum_required(VERSION 3.14)
project(fortify_probe LANGUAGES CXX)
list(APPEND CMAKE_MODULE_PATH "${HELP_PACKAGE_DIR}")
include(help_package)
generic_package()
message(STATUS "AMDSMI_PROBE_FLAGS=[${CMAKE_CXX_FLAGS}]")
"""
COMPILER_CMAKELISTS = """\
cmake_minimum_required(VERSION 3.14)
project(compiler_probe LANGUAGES CXX)
"""
PROBE_FLAGS_RE = re.compile(r"AMDSMI_PROBE_FLAGS=\[(.*)\]")


def _probe_env(compiler: str) -> dict:
    # Ambient CMAKE_*, CXXFLAGS or generator variables would change the flags under test.
    env = {name: os.environ[name] for name in KEPT_ENV_VARS if name in os.environ}
    env["CXX"] = compiler
    return env


def _run_cmake(cmakelists: str, *, env: dict, defines: list) -> subprocess.CompletedProcess:
    with tempfile.TemporaryDirectory() as tmp:
        src_dir = Path(tmp) / "src"
        src_dir.mkdir()
        (src_dir / "CMakeLists.txt").write_text(cmakelists, encoding="utf-8")
        cmd = ["cmake", "-S", str(src_dir), "-B", str(Path(tmp) / "build")] + defines
        # text= needs Python 3.7, so keep universal_newlines for 3.6.
        return subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            env=env,
            timeout=CMAKE_TIMEOUT_SECONDS,
        )


def _compiler_works(compiler: str) -> bool:
    result = _run_cmake(COMPILER_CMAKELISTS, env=_probe_env(compiler), defines=[])
    return result.returncode == 0


@functools.lru_cache(maxsize=None)
def _find_clang() -> str:
    for name in CLANG_NAMES:
        path = shutil.which(name)
        if path and _compiler_works(path):
            return path
    return ""


def _require_clang() -> str:
    """Return a working clang, skipping when none is installed and failing when one is broken."""
    candidates = [path for path in (shutil.which(name) for name in CLANG_NAMES) if path]
    if not candidates:
        raise unittest.SkipTest("needs amdclang++ or clang++ to see the fortify flag")
    clang = _find_clang()
    if not clang:
        raise AssertionError("no working compiler among: " + ", ".join(candidates))
    return clang


def _write_broken_compiler(path: Path) -> None:
    path.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    path.chmod(EXECUTABLE_MODE)


def _configure_flags(*, build_type: str, cxxflags: str = "", extra_defines: tuple = ()) -> str:
    """Return CMAKE_CXX_FLAGS after generic_package() for one configure."""
    env = _probe_env(_find_clang())
    if cxxflags:
        env["CXXFLAGS"] = cxxflags

    defines = [f"-DHELP_PACKAGE_DIR={HELP_PACKAGE.parent}"]
    if build_type:
        defines.append(f"-DCMAKE_BUILD_TYPE={build_type}")
    defines.extend(extra_defines)
    result = _run_cmake(PROBE_CMAKELISTS, env=env, defines=defines)

    match = PROBE_FLAGS_RE.search(result.stdout)
    if result.returncode != 0 or match is None:
        raise RuntimeError("probe configure failed:\n" + result.stdout)
    return match.group(1)


class FortifySourceGateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # Installed layout does not ship cmake_modules/.
        if not HELP_PACKAGE.is_file():
            raise unittest.SkipTest("cmake_modules/help_package.cmake not found")
        if shutil.which("cmake") is None:
            raise unittest.SkipTest("cmake not found")
        _require_clang()

    def test_debug_build_omits_fortify_source(self) -> None:
        self.assertNotIn(FORTIFY_MACRO, _configure_flags(build_type="Debug"))

    def test_empty_build_type_omits_fortify_source(self) -> None:
        self.assertNotIn(FORTIFY_MACRO, _configure_flags(build_type=""))

    def test_release_build_keeps_fortify_source(self) -> None:
        self.assertIn(FORTIFY_FLAG, _configure_flags(build_type="Release"))

    def test_relwithdebinfo_build_keeps_fortify_source(self) -> None:
        self.assertIn(FORTIFY_FLAG, _configure_flags(build_type="RelWithDebInfo"))

    def test_optimized_build_type_spellings_keep_fortify_source(self) -> None:
        for build_type in OPTIMIZED_BUILD_TYPES:
            with self.subTest(build_type=build_type):
                self.assertIn(FORTIFY_FLAG, _configure_flags(build_type=build_type))

    def test_custom_build_type_omits_fortify_source(self) -> None:
        self.assertNotIn(FORTIFY_MACRO, _configure_flags(build_type=CUSTOM_BUILD_TYPE))

    def test_optimized_cxxflags_keep_fortify_source_without_build_type(self) -> None:
        for level in OPTIMIZED_LEVELS:
            with self.subTest(level=level):
                flags = _configure_flags(build_type="", cxxflags=level)
                self.assertIn(FORTIFY_FLAG, flags)

    def test_optimization_flag_ending_build_type_flags_keeps_fortify_source(self) -> None:
        flags = _configure_flags(build_type="Release", extra_defines=(RELEASE_FLAGS_ENDING_IN_O2,))
        self.assertIn(FORTIFY_FLAG, flags)

    def test_tab_separated_cxxflags_keep_fortify_source_without_build_type(self) -> None:
        flags = _configure_flags(build_type="", cxxflags=TAB_SEPARATED_OPTIMIZED_CXXFLAGS)
        self.assertIn(FORTIFY_FLAG, flags)

    def test_unoptimized_cxxflags_omit_fortify_source_without_build_type(self) -> None:
        flags = _configure_flags(build_type="", cxxflags=UNOPTIMIZED_LEVEL)
        self.assertNotIn(FORTIFY_MACRO, flags)

    def test_debug_build_keeps_stack_protector(self) -> None:
        self.assertIn(STACK_PROTECTOR_FLAG, _configure_flags(build_type="Debug"))

    def test_debug_build_keeps_cast_align_warning(self) -> None:
        self.assertIn(CAST_ALIGN_FLAG, _configure_flags(build_type="Debug"))

    def test_ambient_cmake_build_type_does_not_leak_into_probe(self) -> None:
        with mock.patch.dict(os.environ, {"CMAKE_BUILD_TYPE": "Release"}):
            flags = _configure_flags(build_type="")
        self.assertNotIn(FORTIFY_MACRO, flags)


class CompilerProbeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if shutil.which("cmake") is None:
            raise unittest.SkipTest("cmake not found")

    def test_compiler_that_fails_to_configure_is_not_working(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            broken = Path(tmp) / "clang++"
            _write_broken_compiler(broken)
            self.assertFalse(_compiler_works(str(broken)))

    def test_broken_clang_on_path_fails_instead_of_skipping(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            for name in CLANG_NAMES:
                _write_broken_compiler(Path(tmp) / name)
            with mock.patch.dict(os.environ, {"PATH": tmp + os.pathsep + os.environ["PATH"]}):
                _find_clang.cache_clear()
                self.addCleanup(_find_clang.cache_clear)
                with self.assertRaises(AssertionError):
                    _require_clang()

    def test_no_clang_on_path_skips(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, {"PATH": tmp}):
                _find_clang.cache_clear()
                self.addCleanup(_find_clang.cache_clear)
                with self.assertRaises(unittest.SkipTest):
                    _require_clang()
