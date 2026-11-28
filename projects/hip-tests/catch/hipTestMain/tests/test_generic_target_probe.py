#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Exercise the real generic-target CMake helper without a HIP installation."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

TARGETS = [
    "gfx9-generic",
    "gfx9-4-generic:sramecc+:xnack-",
    "gfx9-4-generic:sramecc-:xnack-",
    "gfx9-4-generic:sramecc+:xnack+",
    "gfx9-4-generic:sramecc-:xnack+",
    "gfx10-1-generic",
    "gfx10-3-generic",
    "gfx11-generic",
    "gfx11-7-generic",
    "gfx12-generic",
    "gfx12-5-generic",
]
RETAINED = ["gfx9-4-generic:sramecc+:xnack-", "gfx11-generic"]


class GenericTargetProbeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.compiler = self.root / "compiler.py"
        self.compiler.write_text(
            "import json, pathlib, sys\n"
            "log = pathlib.Path(__file__).with_suffix('.log')\n"
            "with log.open('a', encoding='utf-8') as stream:\n"
            "    stream.write(json.dumps(sys.argv[1:]) + '\\n')\n"
            "target = next(a.split('=', 1)[1] for a in sys.argv if a.startswith('--offload-arch='))\n"
            f"if '--reject-all' not in sys.argv and target in {RETAINED!r}:\n"
            "    sys.exit(0)\n"
            "print('rejected ' + target, file=sys.stderr)\n"
            "sys.exit(1)\n",
            encoding="utf-8",
        )
        helper = (
            Path(__file__).resolve().parents[2]
            / "cmake/hip-tests-generic-targets.cmake"
        )
        self.preamble = (
            "cmake_minimum_required(VERSION 3.20)\n"
            "project(GenericTargetProbe NONE)\n"
            f"set(CMAKE_HIP_COMPILER [=[{Path(sys.executable).as_posix()}]=])\n"
            f"set(HIP_DEVICE_BUILD_FLAGS_NO_ARCH [=[{self.compiler.as_posix()}]=])\n"
            f"include([=[{helper.as_posix()}]=])\n"
        )

    def configure(self, body):
        (self.root / "CMakeLists.txt").write_text(
            self.preamble + body, encoding="utf-8"
        )
        return subprocess.run(
            ["cmake", "-S", str(self.root), "-B", str(self.root / "build")],
            capture_output=True,
            text=True,
            check=False,
        )

    def invocations(self):
        return [
            json.loads(line)
            for line in self.compiler.with_suffix(".log")
            .read_text(encoding="utf-8")
            .splitlines()
        ]

    def test_partial_support_is_shared_but_rechecked_when_inputs_change(self):
        for child in ("compiler", "module"):
            directory = self.root / child
            directory.mkdir()
            (directory / "CMakeLists.txt").write_text(
                "hip_tests_get_generic_target_options(options targets)\n"
                'file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/inventory.txt" "${targets}\\n${options}\\n")\n',
                encoding="utf-8",
            )
        python_path = Path(sys.executable)
        alternate_compiler = f"{python_path.parent.as_posix()}/./{python_path.name}"
        body = (
            "add_subdirectory(compiler)\nadd_subdirectory(module)\n"
            "list(APPEND HIP_DEVICE_BUILD_FLAGS_NO_ARCH --changed-flags)\n"
            "hip_tests_get_generic_target_options(options targets)\n"
            "hip_tests_get_generic_target_options(options targets)\n"
            "set(HIP_PATH_OPT --hip-path=another-sdk)\n"
            "hip_tests_get_generic_target_options(options targets)\n"
            f"set(CMAKE_HIP_COMPILER [=[{alternate_compiler}]=])\n"
            "hip_tests_get_generic_target_options(options targets)\n"
            "hip_tests_get_generic_target_options(options targets)\n"
        )
        result = self.configure(body)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        calls = self.invocations()
        self.assertEqual(len(calls), 4 * len(TARGETS))
        for offset in range(0, len(calls), len(TARGETS)):
            self.assertEqual(
                [
                    next(
                        a.split("=", 1)[1]
                        for a in args
                        if a.startswith("--offload-arch=")
                    )
                    for args in calls[offset : offset + len(TARGETS)]
                ],
                TARGETS,
            )
        expected = [
            ";".join(RETAINED),
            ";".join(f"--offload-arch={t}" for t in RETAINED),
        ]
        for child in ("compiler", "module"):
            inventory = self.root / "build" / child / "inventory.txt"
            self.assertEqual(
                inventory.read_text(encoding="utf-8").splitlines(), expected
            )
        result = self.configure(body)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(len(self.invocations()), 8 * len(TARGETS))

    def test_no_supported_targets_reports_every_failure(self):
        result = self.configure(
            "list(APPEND HIP_DEVICE_BUILD_FLAGS_NO_ARCH --reject-all)\n"
            "hip_tests_get_generic_target_options(options targets)\n"
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(len(self.invocations()), len(TARGETS))
        for target in TARGETS:
            self.assertIn(f"rejected {target}", result.stderr)


if __name__ == "__main__":
    unittest.main()
