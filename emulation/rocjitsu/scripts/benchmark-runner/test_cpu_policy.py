# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CPU policy regression tests using temporary sysfs files, never host settings."""

from contextlib import ExitStack
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import benchmark_run
import cpu_policy
import host_setup

SCRIPT_DIR = Path(__file__).resolve().parent


class CpuPolicyTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.put("online", "0-127")
        for cpu in range(128):
            physical = cpu % 64
            self.put(
                f"cpu{cpu}/topology/thread_siblings_list",
                f"{physical},{physical + 64}",
            )
        for cpu in range(64):
            for name, value in {
                "related_cpus": f"{cpu} {cpu + 64}",
                "scaling_min_freq": "4000000",
                "scaling_max_freq": "5000000",
                "scaling_governor": "powersave",
                "energy_performance_preference": "balance_power",
                "boost": "1",
            }.items():
                self.put(f"cpufreq/policy{cpu}/{name}", value)
        for cpu in cpu_policy.POLICY_CPUS:
            for state, latency in enumerate((1, 9, 10, 100)):
                self.put(f"cpu{cpu}/cpuidle/state{state}/latency", str(latency))
                self.put(f"cpu{cpu}/cpuidle/state{state}/disable", "0")

    def put(self, relative, value):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value + "\n")

    def test_apply_and_readback(self):
        host_setup.apply(self.root)
        cpu_policy.check_policy(self.root)
        for name, expected in {
            "scaling_min_freq": "3200000",
            "scaling_max_freq": "3200000",
            "scaling_governor": "performance",
            "energy_performance_preference": "performance",
            "boost": "0",
        }.items():
            self.assertEqual(
                cpu_policy.read(self.root / f"cpufreq/policy16/{name}"), expected
            )
        self.assertEqual(cpu_policy.read(self.root / "cpufreq/policy0/boost"), "1")
        for state, expected in enumerate(("0", "0", "1", "1")):
            self.assertEqual(
                cpu_policy.read(self.root / f"cpu16/cpuidle/state{state}/disable"),
                expected,
            )

    def test_setup_rejects_ignored_write(self):
        write = host_setup.write

        def ignore_boost(path, value):
            if path != self.root / "cpufreq/policy63/boost":
                write(path, value)

        with patch.object(host_setup, "write", side_effect=ignore_boost):
            with self.assertRaisesRegex(ValueError, "boost"):
                host_setup.apply(self.root)

    def test_global_boost_fallback(self):
        (self.root / "cpufreq/policy63/boost").unlink()
        self.put("cpufreq/boost", "1")
        host_setup.apply(self.root)
        self.assertEqual(cpu_policy.read(self.root / "cpufreq/boost"), "0")
        cpu_policy.check_policy(self.root)
        self.put("cpufreq/boost", "1")
        with self.assertRaises(ValueError):
            cpu_policy.check_policy(self.root)

    def assert_preflight_rejects(self):
        before = {
            path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()
        }
        with self.assertRaises((ValueError, OSError)):
            host_setup.apply(self.root)
        self.assertEqual(
            before,
            {
                path: path.read_bytes()
                for path in self.root.rglob("*")
                if path.is_file()
            },
        )

    def test_missing_all_policies(self):
        shutil.rmtree(self.root / "cpufreq")
        self.assert_preflight_rejects()

    def test_missing_last_cpu_pair(self):
        shutil.rmtree(self.root / "cpufreq/policy63")
        self.assert_preflight_rejects()

    def test_wrong_sibling_topology(self):
        self.put("cpu127/topology/thread_siblings_list", "62,127")
        self.assert_preflight_rejects()

    def test_offline_cpu(self):
        self.put("online", "0-126")
        self.assert_preflight_rejects()

    def test_policy_overlaps_housekeeping(self):
        self.put("cpufreq/policy63/related_cpus", "0 63 127")
        self.assert_preflight_rejects()

    def test_missing_late_control(self):
        (self.root / "cpu127/cpuidle/state3/disable").unlink()
        self.assert_preflight_rejects()

    def test_missing_idle_states(self):
        shutil.rmtree(self.root / "cpu127/cpuidle")
        self.assert_preflight_rejects()

    def test_each_policy_field_drift(self):
        host_setup.apply(self.root)
        for relative, drift in {
            "cpufreq/policy63/scaling_min_freq": "2500000",
            "cpufreq/policy63/scaling_max_freq": "4000000",
            "cpufreq/policy63/scaling_governor": "powersave",
            "cpufreq/policy63/energy_performance_preference": "balance_power",
            "cpufreq/policy63/boost": "1",
            "cpu127/cpuidle/state2/disable": "0",
        }.items():
            with self.subTest(control=relative):
                original = cpu_policy.read(self.root / relative)
                self.put(relative, drift)
                with self.assertRaisesRegex(ValueError, Path(relative).name):
                    cpu_policy.check_policy(self.root)
                self.put(relative, original)

    def test_wrapper_rejects_drift_during_child(self):
        host_setup.apply(self.root)
        with ExitStack() as stack:
            stack.enter_context(patch.object(benchmark_run, "CPU_ROOT", self.root))
            stack.enter_context(patch.object(benchmark_run, "check_cpus"))
            stack.enter_context(
                patch.object(
                    benchmark_run, "check_scheduler", return_value=("bpfland_test", 1)
                )
            )
            stack.enter_context(
                patch.object(benchmark_run.os, "geteuid", return_value=1000)
            )
            stack.enter_context(
                patch.object(
                    benchmark_run.os,
                    "sched_getaffinity",
                    return_value=cpu_policy.BENCHMARK_CPUS,
                )
            )
            stack.enter_context(patch.object(benchmark_run.os, "sched_setaffinity"))
            child = Mock()
            child.wait.return_value = 0
            launch = stack.enter_context(
                patch.object(benchmark_run.subprocess, "Popen", return_value=child)
            )
            self.assertEqual(benchmark_run.run(["true"]), 0)
            child.wait.side_effect = lambda: (
                self.put("cpufreq/policy63/boost", "1") or 0
            )
            with self.assertRaises(ValueError):
                benchmark_run.run(["true"])
            launch.reset_mock()
            with self.assertRaises(ValueError):
                benchmark_run.run(["true"])
            launch.assert_not_called()

    def test_script_imports(self):
        for script in ("host_setup.py", "benchmark_run.py"):
            with self.subTest(script=script):
                result = subprocess.run(
                    [sys.executable, str(SCRIPT_DIR / script), "--help"],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
