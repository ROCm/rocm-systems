#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Exercise the real Python staging rules without building the GPU library."""

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(shutil.which("cmake") and shutil.which("make"), "CMake and make required")
class TestPythonPackageStaging(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.interface = self.source / "py-interface"
        self.interface.mkdir(parents=True)
        project = Path(__file__).resolve().parents[2]
        shutil.copy2(project / "py-interface/CMakeLists.txt", self.interface / "CMakeLists.txt")
        self.inputs = {
            "__init__.py": "from .amdsmi_interface import query\n",
            "amdsmi_interface.py": "def query(): return 1\n",
            "amdsmi_exception.py": "class Error(Exception): pass\n",
            "amdsmi_interface_utils.py": "VALUE = 1\n",
            "amdsmi_wrapper.py": "VALUE = 1\n",
            "README.md": "Package description\n",
            "setup.py": "NAME = 'amdsmi'\n",
            "pyproject.toml.in": "[project]\nname = 'amdsmi'\nversion = '1.0'\n",
            "_version.py.in": "__version__ = '1.0'\n",
        }
        for name, content in self.inputs.items():
            (self.interface / name).write_text(content, encoding="utf-8")
        (self.source / "LICENSE").write_text("Test license\n", encoding="utf-8")
        (self.source / "CMakeLists.txt").write_text(
            "cmake_minimum_required(VERSION 3.20)\n"
            "project(staging NONE)\n"
            "set(AMD_SMI amd_smi)\n"
            "add_custom_target(amd_smi)\n"
            'file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/src")\n'
            'file(WRITE "${PROJECT_BINARY_DIR}/src/libamd_smi.so" "fixture")\n'
            'set(AMDSMI_SYSTEM_PYTHON_SITELIB "${PROJECT_BINARY_DIR}/site" CACHE PATH "")\n'
            "set(SHARE_INSTALL_PREFIX share/amd_smi)\n"
            "add_subdirectory(py-interface)\n",
            encoding="utf-8",
        )
        self.build = self.root / "build"
        self.package = self.build / "py-interface/python_package"
        self._run("cmake", "-S", str(self.source), "-B", str(self.build), "-G", "Unix Makefiles")
        self._build()

    def _run(self, *command, **kwargs):
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            **kwargs,
        )
        self.assertEqual(result.returncode, 0, result.stdout)
        return result.stdout

    def _build(self):
        self._run("cmake", "--build", str(self.build), "--target", "python_package")

    def _edit(self, source, destination, text):
        source.write_text(text, encoding="utf-8")
        # Force an older output timestamp instead of depending on filesystem clock resolution.
        old = source.stat().st_mtime - 10
        os.utime(str(destination), (old, old))

    def test_incremental_api_rename_refreshes_exports_and_implementation(self):
        for name, content in {
            "amdsmi_interface.py": "def renamed_query(): return 2\n",
            "__init__.py": "from .amdsmi_interface import renamed_query\n",
        }.items():
            self._edit(self.interface / name, self.package / "amdsmi" / name, content)
        self._build()
        env = dict(os.environ, PYTHONPATH=str(self.package), PYTHONDONTWRITEBYTECODE="1")
        self._run(
            sys.executable,
            "-c",
            "import amdsmi; assert amdsmi.renamed_query() == 2",
            cwd=str(self.root),
            env=env,
        )

    def test_each_copied_source_triggers_refresh(self):
        for name in (
            "__init__.py",
            "amdsmi_interface.py",
            "amdsmi_exception.py",
            "amdsmi_interface_utils.py",
            "README.md",
            "setup.py",
            "LICENSE",
        ):
            with self.subTest(source=name):
                source = self.source / name if name == "LICENSE" else self.interface / name
                destination = (
                    self.package / name if name == "setup.py" else self.package / "amdsmi" / name
                )
                changed = source.read_text(encoding="utf-8") + "\n# Updated source\n"
                self._edit(source, destination, changed)
                self._build()
                self.assertEqual(destination.read_text(encoding="utf-8"), changed)


if __name__ == "__main__":
    unittest.main()
