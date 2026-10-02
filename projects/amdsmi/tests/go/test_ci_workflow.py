# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import re
import unittest
from pathlib import Path


class WorkflowTests(unittest.TestCase):
    def test_go_job_selects_toolchains_and_runs_host_checks(self) -> None:
        repository = Path(__file__).resolve().parents[4]
        workflow = (repository / ".github/workflows/amdsmi-build.yml").read_text(encoding="utf-8")
        match = re.search(r"^  go-tests:\n(.*?)(?=^  [\w-]+:|\Z)", workflow, re.M | re.S)
        self.assertIsNotNone(match, "isolated Go test job is missing")
        job = match.group(1)
        self.assertIn("runs-on: ubuntu-24.04", job)
        for forbidden in ("self-hosted", "--privileged", "container:", "continue-on-error:"):
            self.assertNotIn(forbidden, job)
        self.assertIn("go: ['1.20.14', '1.24.1']", job)
        pin = "actions/setup-go@40f1582b2485089dde7abd97c1529aa768e1baff"
        precedent = (repository / ".github/workflows/amdsmi-dme-ci.yml").read_text(encoding="utf-8")
        self.assertIn(pin, precedent)
        self.assertIn(pin, job)
        self.assertIn("go-version: ${{ matrix.go }}", job)
        self.assertIn("cache: false", job)
        self.assertIn("working-directory: projects/amdsmi", job)
        for package in (
            "build-essential",
            "cmake",
            "pkg-config",
            "libdrm-dev",
            "libssl-dev",
            "libnl-3-dev",
            "libnl-genl-3-dev",
            "libmnl-dev",
            "python3",
        ):
            self.assertIn(package, job)
        self.assertIn("python3 -B -m unittest discover -s tests/go -p 'test_*.py' -v", job)
        self.assertIn("python3 -B tests/go/test_api_contract.py", job)
        for option in ("", " --race", " --checkptr", " --vet", " --build-example"):
            self.assertRegex(job, r"(?m)^ +python3 -B tests/go/run_tests\.py" + option + r"$")
        modern = re.search(r"if:.*matrix.go == '1.24.1'.*\n +run: \|\n((?: {10}[^\n]*\n)+)", job)
        self.assertIsNotNone(modern, "modern Go safety checks are missing")
        self.assertIn("run_tests.py --cgocheck2", modern.group(1))
        self.assertIn("run_tests.py --asan", modern.group(1))
        for option in (
            "ENABLE_ESMI_LIB=OFF",
            "BUILD_SHARED_LIBS=ON",
            "BUILD_CLI=OFF",
            "BUILD_TESTS=OFF",
            "ENABLE_LDCONFIG=OFF",
        ):
            self.assertIn("-D" + option, job)
        self.assertIn('cmake --build "$BUILD_DIR"', job)
        self.assertIn('--include-dir "$PWD/include"', job)
        self.assertIn('--library-dir "$BUILD_DIR/src"', job)
        self.assertNotIn("$BUILD_DIR/lib", job)
        for option in ("--run '^TestNativeVersion$'", "--vet", "--build-example"):
            self.assertIn('"${native[@]}" ' + option, job)
        self.assertIn('tests/go/test_install.py --build-dir "$BUILD_DIR"', job)


if __name__ == "__main__":
    unittest.main()
