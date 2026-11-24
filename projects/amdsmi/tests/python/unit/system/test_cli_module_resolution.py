#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""amd-smi must import the amdsmi package it was installed with.

$ROCM_PATH and $ROCM_HOME can name a different ROCm install on a multi-install host.
"""

import os
import shutil
import subprocess
import sys
import tempfile
import unittest

from common.common import cli_search_order, find_cli_dir, load_cli_module

_CLI_DIR = find_cli_dir(*cli_search_order(os.path.dirname(os.path.abspath(__file__))))

# Reports which copy was imported, then exits before amdsmi_init initializes any device.
_STUB_PACKAGE = "import os\nprint(os.path.realpath(__file__))\nraise SystemExit(0)\n"


def _make_rocm_root(root):
    """Create a stub ``<root>/share/amd_smi/amdsmi`` package; return its resolved ``__init__.py``."""
    package = os.path.join(root, "share", "amd_smi", "amdsmi")
    os.makedirs(package)
    init = os.path.join(package, "__init__.py")
    with open(init, "w", encoding="utf-8") as handle:
        handle.write(_STUB_PACKAGE)
    return os.path.realpath(init)


class TestCliModuleResolution(unittest.TestCase):
    def setUp(self):
        if not _CLI_DIR:
            self.skipTest("amd-smi CLI directory not found")
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp_dir = tmp.name
        self.own_root = os.path.join(tmp.name, "own")
        # Copied, not symlinked: amdsmi_init resolves symlinks to find its install root.
        self.cli_dir = os.path.join(self.own_root, "libexec", "amdsmi_cli")
        os.makedirs(self.cli_dir)
        for name in ("amdsmi_init.py", "amdsmi_cli_exceptions.py"):
            shutil.copy(os.path.join(_CLI_DIR, name), self.cli_dir)
        self.own_stub = _make_rocm_root(self.own_root)
        self.other_root = os.path.join(tmp.name, "other")
        _make_rocm_root(self.other_root)

    def _run_cli_init(self, env_var, *python_flags):
        env = {k: v for k, v in os.environ.items() if k not in ("ROCM_PATH", "ROCM_HOME")}
        env[env_var] = self.other_root
        child = f"import sys; sys.path.insert(0, {self.cli_dir!r}); import amdsmi_init"
        return subprocess.run(
            [sys.executable, *python_flags, "-c", child],
            cwd=self.tmp_dir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )

    def test_rocm_env_vars_do_not_override_own_install(self):
        for env_var in ("ROCM_PATH", "ROCM_HOME"):
            with self.subTest(env_var=env_var):
                result = self._run_cli_init(env_var)
                self.assertEqual(result.returncode, 0, msg=result.stderr)
                self.assertEqual(result.stdout.strip(), self.own_stub)

    def test_rocm_env_vars_are_not_a_fallback_when_own_copy_is_missing(self):
        shutil.rmtree(os.path.join(self.own_root, "share"))
        exc_path = os.path.join(self.cli_dir, "amdsmi_cli_exceptions.py")
        exit_codes = load_cli_module("amdsmi_cli_exceptions", exc_path).AmdSmiExitCode
        for env_var in ("ROCM_PATH", "ROCM_HOME"):
            with self.subTest(env_var=env_var):
                # -I -S hide every other amdsmi, so only the env var could supply one.
                result = self._run_cli_init(env_var, "-I", "-S")
                output = result.stdout + result.stderr
                self.assertEqual(result.returncode, exit_codes.IMPORT_ERROR, msg=output)
