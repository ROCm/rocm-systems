#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Configure-only regression for installed SDK test preload environments.

Run with Python and CMake >= 3.24; no ROCm installation or GPU is required.
The production execute-test helper and conversion-script CMakeLists are used.
Only unrelated package targets and pytest validation registration are stubbed.
"""

import json
from pathlib import Path
import subprocess
import tempfile
import unittest


TESTS = Path(__file__).resolve().parents[1]


class PreloadEnvironmentTest(unittest.TestCase):
    def test_assignment_valued_callsites_use_environment(self):
        for path in TESTS.rglob("CMakeLists.txt"):
            source = path.read_text()
            if '"ROCPROF_PRELOAD=" PRELOAD_ENV' in source:
                with self.subTest(path=str(path.relative_to(TESTS))):
                    self.assertNotRegex(source, r'\bPRELOAD\s+"\$\{PRELOAD_ENV\}"')

    def test_generated_conversion_environments(self):
        common = (TESTS / "common/CMakeLists.txt").read_text()
        start = common.index("function(rocprofiler_set_integration_test_name ")
        end = common.index("function(rocprofiler_add_integration_validate_test ")
        production_helpers = common[start:end]
        for preload in ("", "/runtime/libclang_rt.asan.so",
                        "/runtime/libclang_rt.asan.so:/simulator/librocjitsu.so"):
            with self.subTest(preload=preload), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root / "helpers.cmake").write_text(production_helpers)
                # find_package() succeeds without loading or compiling SDK code.
                (root / "rocprofiler-sdk-config.cmake").write_text("")
                cmake = r'''
cmake_minimum_required(VERSION 3.24)
project(preload_environment_regression LANGUAGES CXX)
enable_testing()
set(rocprofiler-sdk_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
foreach(name vector-ops rocprofiler-sdk::rocprofv3 rocprofiler-sdk::convert-counters-collection-format)
    add_executable(${name} IMPORTED GLOBAL)
    set_property(TARGET ${name} PROPERTY IMPORTED_LOCATION "${CMAKE_COMMAND}")
endforeach()
add_library(rocprofiler-sdk::rocprofiler-sdk-shared-library SHARED IMPORTED GLOBAL)
set_property(TARGET rocprofiler-sdk::rocprofiler-sdk-shared-library PROPERTY IMPORTED_LOCATION "/sdk/lib/librocprofiler-sdk.so")
function(rocprofiler_configure_pytest_files)
endfunction()
function(rocprofiler_add_integration_validate_test)
endfunction()
include("${CMAKE_CURRENT_SOURCE_DIR}/helpers.cmake")
set(ROCPROFILER_MEMCHECK_PRELOAD_ENV_VALUE "@PRELOAD@")
if(ROCPROFILER_MEMCHECK_PRELOAD_ENV_VALUE)
    set(ROCPROFILER_MEMCHECK_PRELOAD_ENV "LD_PRELOAD=${ROCPROFILER_MEMCHECK_PRELOAD_ENV_VALUE}")
endif()
add_subdirectory("@CONVERSION@" conversion)
# Existing callsites can supply other assignments via a second ENVIRONMENT.
rocprofiler_add_integration_execute_test(repeated-environment
    COMMAND "${CMAKE_COMMAND}" -E true DEPENDS vector-ops
    ENVIRONMENT "ROCPROF_PRELOAD=@PRELOAD@"
    ENVIRONMENT "PYTHONPATH=/installed/python")
'''
                cmake = cmake.replace("@PRELOAD@", preload).replace(
                    "@CONVERSION@", (TESTS / "rocprofv3/conversion-script").as_posix()
                )
                (root / "CMakeLists.txt").write_text(cmake)
                configured = subprocess.run(
                    ["cmake", "-S", str(root), "-B", str(root / "build")],
                    capture_output=True, text=True,
                )
                self.assertEqual(configured.returncode, 0, configured.stdout + configured.stderr)
                listing = subprocess.check_output(
                    ["ctest", "--test-dir", str(root / "build"), "--show-only=json-v1"],
                    text=True,
                )
                tests = json.loads(listing)["tests"]
                self.assertEqual(len(tests), 4)
                for test in tests:
                    props = {entry["name"]: entry["value"] for entry in test["properties"]}
                    env = dict(item.split("=", 1) for item in props["ENVIRONMENT"] if item)
                    if preload:
                        self.assertEqual(env["LD_PRELOAD"], preload)
                        self.assertEqual(env["ROCPROF_PRELOAD"], preload)
                        self.assertNotIn("=", env["LD_PRELOAD"])
                    else:
                        self.assertNotIn("LD_PRELOAD", env)
                    if test["name"].endswith("repeated-environment"):
                        self.assertEqual(env["PYTHONPATH"], "/installed/python")


if __name__ == "__main__":
    unittest.main()
