# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CPU policy regression tests using temporary sysfs files, never host settings."""

from contextlib import ExitStack
import importlib.util
import os
from pathlib import Path
import select
import shutil
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

SCRIPT_DIR = Path(__file__).resolve().parents[2] / "scripts" / "benchmark-runner"


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPT_DIR / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


with patch.dict(sys.modules):
    cpu_policy = load_script("cpu_policy")
    host_setup = load_script("host_setup")
    benchmark_run = load_script("benchmark_run")


class CpuPolicyTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.put("online", cpu_policy.format_cpus(cpu_policy.ONLINE_CPUS))
        for cpu in sorted(cpu_policy.ONLINE_CPUS):
            physical = cpu % cpu_policy.PHYSICAL_CORE_COUNT
            self.put(
                f"cpu{cpu}/topology/thread_siblings_list",
                f"{physical},{physical + cpu_policy.PHYSICAL_CORE_COUNT}",
            )
        for cpu in range(cpu_policy.PHYSICAL_CORE_COUNT):
            self.put(f"cpu{cpu}/acpi_cppc/nominal_freq", "3200")
            for name, value in {
                "related_cpus": f"{cpu} {cpu + cpu_policy.PHYSICAL_CORE_COUNT}",
                "cpuinfo_min_freq": "1000000",
                "cpuinfo_max_freq": "6000000",
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

    def assert_preflight_rejects(self, message: str):
        before = {
            path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()
        }
        with self.assertRaisesRegex((ValueError, OSError), message):
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
        self.assert_preflight_rejects("cpufreq policies missing CPUs 16-63,80-127")

    def test_missing_last_cpu_pair(self):
        shutil.rmtree(self.root / "cpufreq/policy63")
        self.assert_preflight_rejects("cpufreq policies missing CPUs")

    def test_wrong_sibling_topology(self):
        self.put("cpu127/topology/thread_siblings_list", "62,127")
        self.assert_preflight_rejects("expected SMT siblings")

    def test_offline_cpu(self):
        self.put("online", "0-126")
        self.assert_preflight_rejects("online CPUs 0-127")

    def test_policy_overlaps_housekeeping(self):
        self.put("cpufreq/policy63/related_cpus", "0 63 127")
        self.assert_preflight_rejects("housekeeping CPUs")

    def test_missing_late_control(self):
        (self.root / "cpu127/cpuidle/state3/disable").unlink()
        self.assert_preflight_rejects("disable")

    def test_missing_idle_states(self):
        shutil.rmtree(self.root / "cpu127/cpuidle")
        self.assert_preflight_rejects("idle-state controls unavailable")

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

    def test_nominal_frequency_per_policy(self):
        self.put("cpu16/acpi_cppc/nominal_freq", "2500")
        host_setup.apply(self.root)
        for cpu, expected in ((16, "2500000"), (17, "3200000")):
            for control in ("scaling_min_freq", "scaling_max_freq"):
                self.assertEqual(
                    cpu_policy.read(self.root / f"cpufreq/policy{cpu}/{control}"),
                    expected,
                )
        cpu_policy.check_policy(self.root)

    def test_one_policy_per_logical_cpu(self):
        for cpu in sorted(cpu_policy.ONLINE_CPUS):
            if cpu >= cpu_policy.PHYSICAL_CORE_COUNT:
                shutil.copytree(
                    self.root / f"cpufreq/policy{cpu - cpu_policy.PHYSICAL_CORE_COUNT}",
                    self.root / f"cpufreq/policy{cpu}",
                )
                self.put(f"cpu{cpu}/acpi_cppc/nominal_freq", "3200")
            self.put(f"cpufreq/policy{cpu}/related_cpus", str(cpu))
        host_setup.apply(self.root)
        cpu_policy.check_policy(self.root)
        self.assertEqual(
            cpu_policy.read(self.root / "cpufreq/policy127/scaling_max_freq"), "3200000"
        )

    def test_invalid_nominal_frequency(self):
        for value, message in (
            ("0", "positive"),
            ("-1", "positive"),
            ("invalid", "invalid literal"),
            ("7000", "outside"),
        ):
            with self.subTest(value=value):
                self.put("cpu63/acpi_cppc/nominal_freq", value)
                self.assert_preflight_rejects(message)

    def test_missing_nominal_frequency(self):
        (self.root / "cpu63/acpi_cppc/nominal_freq").unlink()
        self.assert_preflight_rejects("nominal_freq")

    def test_frequency_write_order(self):
        original_write = host_setup.write

        def check_write(path, value):
            if path.name == "scaling_min_freq":
                self.assertLessEqual(
                    int(value), int(cpu_policy.read(path.with_name("scaling_max_freq")))
                )
            if path.name == "scaling_max_freq":
                self.assertGreaterEqual(
                    int(value), int(cpu_policy.read(path.with_name("scaling_min_freq")))
                )
            original_write(path, value)

        with patch.object(host_setup, "write", side_effect=check_write):
            for frequency in ("2500", "5000", "3200"):
                self.put("cpu16/acpi_cppc/nominal_freq", frequency)
                host_setup.apply(self.root)

    def test_cpu_range_format(self):
        for cpus, expected in (([], ""), ([4], "4"), ([3, 1, 2, 1, 8, 9], "1-3,8-9")):
            with self.subTest(cpus=cpus):
                self.assertEqual(cpu_policy.format_cpus(cpus), expected)

    def test_script_imports_ignore_python_environment(self):
        self.put("shadow/argparse.py", "raise RuntimeError('shadow argparse loaded')")
        environment = dict(
            os.environ,
            PYTHONPATH=str(self.root / "shadow"),
            PYTHONHOME=str(self.root / "missing"),
        )
        for script in ("host_setup.py", "benchmark_run.py"):
            result = subprocess.run(
                [sys.executable, "-B", "-E", "-s", str(SCRIPT_DIR / script), "--help"],
                env=environment,
                cwd=self.root,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_script_imports_ignore_user_site(self):
        version = f"python{sys.version_info.major}.{sys.version_info.minor}"
        marker = self.root / "user_site_loaded"
        self.put(
            f"home/.local/lib/{version}/site-packages/usercustomize.py",
            f"from pathlib import Path; Path({str(marker)!r}).touch()",
        )
        environment = dict(os.environ, HOME=str(self.root / "home"))
        for name in ("PYTHONUSERBASE", "PYTHONNOUSERSITE", "PYTHONHOME", "PYTHONPATH"):
            environment.pop(name, None)
        # A venv may already disable user-site imports; exercise the base Python.
        python = getattr(sys, "_base_executable", sys.executable)
        subprocess.run([python, "-B", "-E", "-c", "pass"], env=environment, check=True)
        self.assertTrue(marker.exists(), "baseline must execute usercustomize")
        marker.unlink()
        for script in ("host_setup.py", "benchmark_run.py"):
            result = subprocess.run(
                [python, "-B", "-E", "-s", str(SCRIPT_DIR / script), "--help"],
                env=environment,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(marker.exists())

    def test_script_imports(self):
        for script in ("host_setup.py", "benchmark_run.py"):
            with self.subTest(script=script):
                result = subprocess.run(
                    [
                        sys.executable,
                        "-B",
                        "-E",
                        "-s",
                        str(SCRIPT_DIR / script),
                        "--help",
                    ],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)


class SchedulerTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        (self.root / 'root').mkdir()
        (self.root / 'enable_seq').write_text('7')
        self.set_scheduler('enabled', 'bpfland_test')
        scheduler_root = patch.object(benchmark_run, 'SCHED_ROOT', self.root)
        scheduler_root.start()
        self.addCleanup(scheduler_root.stop)

    def set_scheduler(self, state: str, name: str):
        (self.root / 'state').write_text(state)
        (self.root / 'root/ops').write_text(name)

    def test_readiness_accepts_bpfland(self):
        host_setup.wait_ready(self.root)
        self.assertEqual(benchmark_run.check_scheduler(), ('bpfland_test', 7))

    def test_readiness_rejects_wrong_scheduler_or_state(self):
        for state, name in (
            ('enabled', 'scx_lavd_something'),
            ('disabled', 'bpfland_test'),
        ):
            with self.subTest(state=state, name=name):
                self.set_scheduler(state, name)
                with patch.object(
                    host_setup.time,
                    'monotonic',
                    side_effect=(0, host_setup.SCHEDULER_READY_TIMEOUT_SECONDS + 1),
                ):
                    with self.assertRaisesRegex(
                        ValueError, 'Bpfland did not become ready'
                    ):
                        host_setup.wait_ready(self.root)
                with self.assertRaisesRegex(RuntimeError, 'Bpfland must be enabled'):
                    benchmark_run.check_scheduler()

    def test_readiness_waits_for_bpfland(self):
        self.set_scheduler('enabled', 'scx_lavd_something')
        with patch.object(
            host_setup.time,
            'monotonic',
            side_effect=(0, 0, host_setup.SCHEDULER_READY_TIMEOUT_SECONDS + 1),
        ):
            with patch.object(
                host_setup.time,
                'sleep',
                side_effect=lambda _: self.set_scheduler('enabled', 'bpfland_test'),
            ) as sleep:
                host_setup.wait_ready(self.root)
                sleep.assert_called_once_with(host_setup.SCHEDULER_READY_POLL_SECONDS)
        self.assertEqual(benchmark_run.check_scheduler(), ('bpfland_test', 7))

    def test_readiness_waits_for_scheduler_files(self):
        (self.root / 'root/ops').unlink()
        with patch.object(
            host_setup.time,
            'monotonic',
            side_effect=(0, 0, host_setup.SCHEDULER_READY_TIMEOUT_SECONDS + 1),
        ):
            with patch.object(
                host_setup.time,
                'sleep',
                side_effect=lambda _: self.set_scheduler('enabled', 'bpfland_test'),
            ) as sleep:
                host_setup.wait_ready(self.root)
                sleep.assert_called_once_with(host_setup.SCHEDULER_READY_POLL_SECONDS)


class CgroupTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.cgroup = self.root / "cgroup"
        self.proc = self.root / "proc_cgroup"
        for group, cpus in (
            ("benchmark.slice", "0-63"),
            ("system.slice", "0-15,64-79"),
        ):
            path = self.cgroup / group
            path.mkdir(parents=True)
            (path / "cpuset.cpus.effective").write_text(cpus)
            (path / "cgroup.events").write_text("populated 1\n")
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch.object(benchmark_run, "CGROUP_ROOT", self.cgroup))
        stack.enter_context(patch.object(benchmark_run, "PROC_CGROUP", self.proc))

    def membership(self, unit: str):
        path = self.cgroup / "benchmark.slice" / unit
        path.mkdir()
        (path / "cpuset.cpus.effective").write_text("0-63")
        self.proc.write_text(f"0::/benchmark.slice/{unit}\n")
        return path

    def test_runner_and_docker_scope(self):
        for unit in ("actions.runner.example.service", "docker-example.scope"):
            with self.subTest(unit=unit):
                self.membership(unit)
                benchmark_run.check_cpus()

    def test_missing_runner_cpuset(self):
        path = self.membership("actions.runner.example.service")
        (path / "cpuset.cpus.effective").unlink()
        with self.assertRaises(FileNotFoundError):
            benchmark_run.check_cpus()

    def test_misplaced_container(self):
        self.proc.write_text("0::/system.slice/docker-example.scope\n")
        with self.assertRaisesRegex(RuntimeError, "under benchmark.slice"):
            benchmark_run.check_cpus()

    def test_child_cpu_range_diagnostic(self):
        path = self.membership("docker-example.scope")
        (path / "cpuset.cpus.effective").write_text("0-15")
        with self.assertRaisesRegex(RuntimeError, "allow 16-63 within CPUs 0-63"):
            benchmark_run.check_cpus()


class ProcessGroupTest(unittest.TestCase):
    # Exercise real processes without requiring host policy or setarch permission.
    launcher = """
import runpy
import subprocess
import sys

wrapper = runpy.run_path(sys.argv[1])['benchmark_run']
wrapper.check_cpus = lambda: None
wrapper.check_policy = lambda root: None
wrapper.check_scheduler = lambda: ('bpfland_test', 1)
wrapper.os.sched_getaffinity = lambda pid: wrapper.BENCHMARK_CPUS
wrapper.os.sched_setaffinity = lambda pid, cpus: None
popen = subprocess.Popen
wrapper.subprocess.Popen = lambda argv, **kwargs: popen(argv[3:], **kwargs)
sys.exit(wrapper.run([sys.executable, '-c', sys.argv[2]]))
"""
    payload = """
import os
import subprocess
import sys
import time

child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
print(os.getpid(), child.pid, flush=True)
time.sleep(60)
"""

    def check_group_cancellation(self, signum: int):
        process = subprocess.Popen(
            [sys.executable, '-B', '-c', self.launcher, __file__, self.payload],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        descendants = []
        try:
            self.assertTrue(select.select([process.stdout], [], [], 10)[0])
            descendants = [int(pid) for pid in process.stdout.readline().split()]
            self.assertEqual(len(descendants), 2)
            for pid in descendants:
                self.assertEqual(os.getpgid(pid), process.pid)
            # The harness owns cancellation of the wrapper and its descendants.
            os.killpg(process.pid, signum)
            process.communicate(timeout=5)
            self.assertEqual(process.returncode, -signum)
            # Both descendants must close inherited pipes for communicate to return.
        finally:
            for pid in [process.pid, *descendants]:
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            process.communicate(timeout=5)

    def test_group_cancellation_kills_workload_and_descendants(self):
        for signum in (signal.SIGKILL, signal.SIGTERM):
            with self.subTest(signal=signum):
                self.check_group_cancellation(signum)


if __name__ == "__main__":
    unittest.main()
