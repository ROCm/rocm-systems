#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Exercise wait-checking and CPU-budget overrides through the native launcher."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

# Inspect the launcher's handoff in the child before runtime cleanup removes it.
PROBE = """
import json, os, sys
from pathlib import Path
invocation = Path(os.environ['ROCJITSU_INVOCATION_DIR'])
config = (invocation / 'effective_config.json' if sys.argv[1] == 'daemon'
          else Path((invocation / 'config_path').read_text().strip()))
print(json.dumps({'path': str(config), 'config': json.loads(config.read_text())}))
"""


class WaitCheckingCliTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.environ.get("RJ_ROCJITSU"):
            raise unittest.SkipTest("RJ_ROCJITSU is not set")
        cls.cli = os.environ["RJ_ROCJITSU"]
        cls.template = Path(os.environ["RJ_SIM_CONFIG"])
        cls.dbt_config = Path(os.environ["RJ_DBT_CONFIG"])

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="rj-wait-cli-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "source.json"
        self.env = dict(os.environ)
        for key in ("LD_PRELOAD", "ROCJITSU_INVOCATION_DIR", "ROCJITSU_SOCKET_PATH"):
            self.env.pop(key, None)
        self.env["ROCJITSU_RUNTIME_DIR"] = str(self.root / "runtime")
        self.write_source("all")

    def write_source(self, mode):
        config = json.loads(self.template.read_text())
        config.update(wait_checking=mode, cpu_thread_budget=6)
        self.source.write_text(json.dumps(config, indent=2) + "\n")

    def run_cli(self, *args, config=None):
        before = self.source.read_bytes()
        result = subprocess.run(
            [self.cli, "--config", str(config or self.source), *args],
            env=self.env,
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
        self.assertEqual(self.source.read_bytes(), before)
        return result

    def check_handoff(self, flags, mode, budget=6, daemon=False):
        result = self.run_cli(
            *flags,
            *(["--daemon"] if daemon else []),
            "--",
            sys.executable,
            "-c",
            PROBE,
            "daemon" if daemon else "local",
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        handoff = json.loads(result.stdout)
        self.assertEqual(handoff["config"]["wait_checking"], mode)
        self.assertEqual(handoff["config"]["cpu_thread_budget"], budget)
        self.assertEqual(
            Path(handoff["path"]).name,
            "effective_config.json" if flags or daemon else self.source.name,
        )

    def test_modes_alone_and_combined_with_cpu_budget(self):
        for mode in ("on", "off", "all"):
            self.write_source("all" if mode == "off" else "off")
            for assigned in (False, True):
                wait = (
                    [f"--wait-checking={mode}"]
                    if assigned
                    else ["--wait-checking", mode]
                )
                budget = (
                    ["--cpu-thread-budget=4"]
                    if assigned
                    else ["--cpu-thread-budget", "4"]
                )
                for order in ("wait-only", "budget-first", "wait-first"):
                    with self.subTest(mode=mode, assigned=assigned, order=order):
                        flags = (
                            wait
                            if order == "wait-only"
                            else (
                                budget + wait
                                if order == "budget-first"
                                else wait + budget
                            )
                        )
                        self.check_handoff(
                            flags, mode, 6 if order == "wait-only" else 4
                        )

    def test_omission_preserves_json_with_and_without_cpu_budget(self):
        self.check_handoff([], "all")
        self.check_handoff(["--cpu-thread-budget=4"], "all", 4)

    def test_daemon_modes(self):
        for mode in ("on", "off", "all"):
            with self.subTest(mode=mode):
                self.check_handoff([f"--wait-checking={mode}"], mode, daemon=True)

    def test_last_mode_wins(self):
        self.check_handoff(["--wait-checking=off", "--wait-checking", "on"], "on")

    def test_rejects_missing_or_invalid_values(self):
        for flags in (
            ["--wait-checking"],
            ["--wait-checking="],
            ["--wait-checking=warn"],
            ["--wait-checking=ALL"],
            ["--wait-checking", "--", "true"],
        ):
            with self.subTest(flags=flags):
                result = self.run_cli(*flags)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("--wait-checking requires on, off, or all", result.stderr)

    def test_rejects_attach_and_external_dbt_config(self):
        for flags, config, message in (
            (["--attach"], self.source, "cannot be combined with --attach"),
            (
                [],
                self.dbt_config,
                "cannot be combined with a dbt_guest simulator_config",
            ),
        ):
            with self.subTest(config=config, flags=flags):
                result = self.run_cli(
                    "--wait-checking=off", *flags, "--", "true", config=config
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("--wait-checking " + message, result.stderr)


if __name__ == "__main__":
    unittest.main()
