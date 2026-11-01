# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Integration tests against the real launcher; no GPU framework required."""

import atexit
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest import mock

try:
    import cloudpickle
except ImportError:
    print(
        "Install the Rocjitsu Python package (including cloudpickle) to run these tests"
    )
    sys.exit(77)

import rocjitsu

CONFIG = Path(__file__).resolve().parents[2] / "configs/gfx1201_r9700.json"


def simulated_targets(gpu_id=8716):
    descriptor = os.open("/dev/kfd", os.O_RDWR)
    try:
        assert "librocjitsu.so" in Path("/proc/self/maps").read_text()
        # Keep the KFD connection alive while querying the daemon's topology.
        return sorted(
            int(line.split()[1])
            for path in Path("/sys/class/kfd/kfd/topology/nodes").glob("*/properties")
            # Daemon mode can expose host GPUs alongside the configured device.
            if int(path.with_name("gpu_id").read_text()) == gpu_id
            for line in path.read_text().splitlines()
            if line.startswith("gfx_target_version ") and int(line.split()[1]) != 0
        )
    finally:
        os.close(descriptor)


class PythonInstallationTest(unittest.TestCase):
    @contextmanager
    def installation(self):
        native = rocjitsu.diagnostics()["executable"]
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory)
            (prefix / "bin").mkdir()
            (prefix / "bin/rocjitsu").symlink_to(native)
            shutil.copytree(CONFIG.parent, prefix / "share/rocjitsu/configs")
            with (
                mock.patch.object(sys, "prefix", directory),
                mock.patch.dict(os.environ, {"PATH": directory}),
            ):
                yield prefix

    def test_presets_match_installed_json_filenames(self):
        expected = tuple(sorted(path.stem for path in CONFIG.parent.glob("*.json")))
        with self.installation() as prefix:
            self.assertEqual(rocjitsu.config.names(), expected)
            self.assertEqual(
                tuple(dir(rocjitsu.config)), tuple(sorted((*expected, "names")))
            )
            for name in expected:
                self.assertEqual(
                    getattr(rocjitsu.config, name),
                    prefix / "share/rocjitsu/configs" / (name + ".json"),
                )
            with self.assertRaises(AttributeError):
                rocjitsu.config.mi455x

    def test_wheel_runtime_precedes_prefix_and_path(self):
        from rocjitsu import _launch

        with self.installation() as prefix:
            package = prefix / "site-packages/_rocjitsu"
            (package / "bin").mkdir(parents=True)
            (package / "bin/rocjitsu").symlink_to(prefix / "bin/rocjitsu")
            shutil.copytree(CONFIG.parent, package / "share/rocjitsu/configs")
            metadata = SimpleNamespace(
                THIS_PACKAGE_ENTRY=SimpleNamespace(
                    get_py_package=lambda: SimpleNamespace(
                        origin=str(package / "__init__.py")
                    )
                )
            )
            with mock.patch.dict(sys.modules, {"rocjitsu._dist_info": metadata}):
                self.assertEqual(_launch.launcher(), str(package / "bin/rocjitsu"))
                self.assertEqual(
                    rocjitsu.config.gfx1250_mi455x,
                    package / "share/rocjitsu/configs/gfx1250_mi455x.json",
                )
                with rocjitsu.Session(rocjitsu.config.gfx1250_mi455x) as session:
                    self.assertEqual(session.command[0], str(package / "bin/rocjitsu"))
                    self.assertEqual(session.run(lambda: 42), 42)
                (package / "bin/rocjitsu").unlink()
                with self.assertRaisesRegex(
                    FileNotFoundError, "missing its native launcher"
                ):
                    _launch.launcher()

    def test_session_defaults_come_from_installation_without_path(self):
        with self.installation() as prefix:
            with rocjitsu.Session(rocjitsu.config.gfx1250_mi455x) as session:
                self.assertEqual(session.command[0], str(prefix / "bin/rocjitsu"))
                self.assertEqual(session.run(lambda: sys.executable), sys.executable)
                self.assertEqual(session.run(lambda: 6 * 7), 42)
            for name in ("gfx1201_r9700", "gfx1201_r9700.json"):
                with self.subTest(name=name), rocjitsu.Session(name) as session:
                    self.assertEqual(session.run(lambda: 42), 42)

    def test_cli_uses_installed_configs(self):
        with self.installation():
            self.assertIn(
                "thread_budget_table",
                rocjitsu.diagnostics("gfx1201_r9700", cpu_thread_budget=1),
            )

    def test_existing_file_takes_precedence_over_preset_name(self):
        with self.installation() as prefix:
            local = prefix / "gfx1250_mi455x"
            shutil.copyfile(CONFIG, local)
            code = f"""import rocjitsu
with rocjitsu.Session('gfx1250_mi455x') as session:
    assert session.command[2] == {str(local)!r}
    assert session.run(lambda: 42) == 42
"""
            result = subprocess.run(
                [sys.executable, "-c", code],
                cwd=prefix,
                env={**os.environ, "PATH": str(prefix / "bin")},
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_missing_installed_configs_has_actionable_error(self):
        with self.installation() as prefix:
            shutil.rmtree(prefix / "share/rocjitsu/configs")
            with self.assertRaisesRegex(
                FileNotFoundError, "install the native runtime"
            ):
                rocjitsu.config.names()
            self.assertFalse(hasattr(rocjitsu.config, "gfx1201_r9700"))
            self.assertIsNone(getattr(rocjitsu.config, "missing", None))
            self.assertIn("names", dir(rocjitsu.config))


class PythonApiTest(unittest.TestCase):
    def test_run_closure_and_keyword_arguments(self):
        offset = 7
        pid, result = rocjitsu.run(
            lambda x, y: (os.getpid(), x + y + offset), 3, y=4, config=CONFIG
        )
        self.assertNotEqual(pid, os.getpid())
        self.assertEqual(result, 14)

    def test_callable_parameter_names_are_workload_keywords(self):
        def workload(*, function, self):
            return function + self

        self.assertEqual(
            rocjitsu.run(workload, function=19, self=23, config=CONFIG), 42
        )
        with rocjitsu.Session(CONFIG) as session:
            self.assertEqual(session.run(workload, function=19, self=23), 42)

    def test_session_state_and_exception_recovery(self):
        with rocjitsu.Session(CONFIG) as session:
            session.run(lambda: os.environ.update(ROCJITSU_TEST_VALUE="persistent"))
            self.assertEqual(
                session.run(lambda: os.environ["ROCJITSU_TEST_VALUE"]), "persistent"
            )
            self.assertEqual(session.run(os.getpid), session.pid)
            with self.assertRaisesRegex(
                rocjitsu.RemoteError, "ZeroDivisionError"
            ) as caught:
                session.run(lambda: 1 / 0)
            self.assertIn("test_api.py", caught.exception.remote_traceback)
            self.assertEqual(session.run(lambda: 42), 42)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            session.run(lambda: 1)
        self.assertNotIn("ROCJITSU_TEST_VALUE", os.environ)

    def test_timeout_closes_session(self):
        with rocjitsu.Session(CONFIG) as session:
            with self.assertRaises(TimeoutError):
                session.run(time.sleep, 30, timeout=0.1)
            with self.assertRaisesRegex(RuntimeError, "closed"):
                session.run(lambda: 0)
            with self.assertRaises(ProcessLookupError):
                os.kill(session.pid, 0)

    def test_startup_timeout_has_diagnostics_and_reaps_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory, "slow-python")
            pid_path = Path(directory, "pid")
            script.write_text(
                '#!/bin/sh\necho "$$" > "$API_PID_FILE"\necho startup-marker >&2\nsleep 60\n'
            )
            script.chmod(0o700)
            with self.assertRaises(rocjitsu.WorkerError) as caught:
                rocjitsu.run(
                    lambda: 1,
                    config=CONFIG,
                    python=script,
                    env={"API_PID_FILE": str(pid_path)},
                    startup_timeout=0.3,
                    capture_output=True,
                )
            self.assertIn("startup timed out", str(caught.exception))
            self.assertIn("startup-marker", caught.exception.stderr)
            with self.assertRaises(ProcessLookupError):
                os.kill(int(pid_path.read_text()), 0)

    def assert_process_terminated(self, pid):
        # Reaping the supervisor does not wait for every descendant to finish
        # handling the group kill. Observe termination before asserting it.
        deadline = time.monotonic() + 5
        stat = Path(f"/proc/{pid}/stat")
        while True:
            try:
                state = stat.read_text().split(")", 1)[1].split()[0]
            except (FileNotFoundError, ProcessLookupError):
                return
            if state in ("Z", "X"):
                return
            if time.monotonic() >= deadline:
                self.fail(f"Process {pid} did not terminate; last state was {state}")
            time.sleep(0.01)

    def test_timeout_terminates_worker_descendants(self):
        with rocjitsu.Session(CONFIG) as session:
            pid = session.run(
                lambda: subprocess.Popen(
                    [sys.executable, "-c", "import time; time.sleep(60)"]
                ).pid
            )
            with self.assertRaises(TimeoutError):
                session.run(time.sleep, 30, timeout=0.1)
            self.assert_process_terminated(pid)

    def check_forked_worker_crash(self, expected_status=23):
        def fork_and_crash():
            if os.fork() == 0:
                time.sleep(60)
                os._exit(0)
            os._exit(23)

        with rocjitsu.Session(CONFIG, capture_output=True) as session:
            errors = []

            def call():
                try:
                    session.run(fork_and_crash)
                except Exception as error:
                    errors.append(error)

            thread = threading.Thread(target=call)
            thread.start()
            thread.join(timeout=5)
            hung = thread.is_alive()
            if hung:
                session.close()
                thread.join(timeout=5)
            self.assertFalse(hung, "worker death was hidden by an inherited socket")
            self.assertIsInstance(errors[0], rocjitsu.WorkerError)
            self.assertEqual(errors[0].returncode, expected_status)
            self.assertTrue(session.closed)

    def test_forked_helper_does_not_hide_worker_crash(self):
        self.check_forked_worker_crash()

    def test_ignored_sigchld_does_not_hide_worker_crash(self):
        previous = signal.signal(signal.SIGCHLD, signal.SIG_IGN)
        try:
            self.check_forked_worker_crash()
        finally:
            signal.signal(signal.SIGCHLD, previous)

    def check_delayed_close(self, *, ignore_sigchld=False, external_reaper=False):
        previous = signal.getsignal(signal.SIGCHLD)
        if ignore_sigchld:
            signal.signal(signal.SIGCHLD, signal.SIG_IGN)
        try:
            with rocjitsu.Session(CONFIG, capture_output=True) as session:

                def fork_helper():
                    pid = os.fork()
                    if pid == 0:
                        time.sleep(60)
                        os._exit(0)
                    return pid

                descendant = session.run(fork_helper)
                reaped = []
                if external_reaper:
                    # The host may reap every direct child. The group leader
                    # must stay alive until cleanup, despite worker exit.
                    def reap():
                        try:
                            reaped.append(os.waitpid(session._process.pid, 0))
                        except ChildProcessError:
                            # Session.close() may win the race to reap its child.
                            pass

                    reaper = threading.Thread(target=reap)
                    reaper.start()
                os.kill(session.pid, signal.SIGKILL)
                self.assertTrue(session._worker_exited.wait(timeout=5))
                self.assertFalse(session.closed)
                self.assertEqual(session.returncode, -signal.SIGKILL)
                # Delay cleanup after worker death. The descendant still owns
                # the inherited RPC descriptor, while its group anchor is live.
                time.sleep(0.1)
                self.assertNotIn(
                    Path(f"/proc/{descendant}/stat")
                    .read_text()
                    .split(")", 1)[1]
                    .split()[0],
                    ("Z", "X"),
                )
                with mock.patch(
                    "os.killpg",
                    side_effect=AssertionError(
                        "caller must never signal a stored numeric process group"
                    ),
                ):
                    session.close()
                if external_reaper:
                    reaper.join(timeout=5)
                    self.assertFalse(reaper.is_alive())
                    if reaped:
                        self.assertEqual(reaped[0][0], session._process.pid)
                self.assert_process_terminated(descendant)
        finally:
            signal.signal(signal.SIGCHLD, previous)

    def test_delayed_close_with_ignored_sigchld_cleans_descendants(self):
        self.check_delayed_close(ignore_sigchld=True)

    def test_delayed_close_with_external_reaper_cleans_descendants(self):
        self.check_delayed_close(external_reaper=True)

    def test_installed_package_does_not_inject_parent_site(self):
        with tempfile.TemporaryDirectory() as directory:
            site = Path(directory, "site-packages")
            site.mkdir()
            shutil.copytree(
                Path(rocjitsu.__file__).parent,
                site / "rocjitsu",
                ignore=shutil.ignore_patterns("__pycache__"),
            )
            (site / "parent_only_framework.py").write_text("value = 42\n")
            code = f'''import sys, importlib.util
sys.path.insert(0, {str(site)!r})
import rocjitsu
assert rocjitsu.__file__.startswith({str(site)!r})
assert importlib.util.find_spec('parent_only_framework') is not None
assert not rocjitsu.run(lambda: importlib.util.find_spec('parent_only_framework') is not None,
                       config={str(CONFIG)!r})
'''
            result = subprocess.run(
                [sys.executable, "-c", code], text=True, capture_output=True, timeout=30
            )
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_worker_exit(self):
        with rocjitsu.Session(CONFIG) as session:
            with self.assertRaisesRegex(rocjitsu.WorkerError, "status 23"):
                session.run(os._exit, 23)

    def test_bad_config_and_missing_launcher(self):
        with self.assertRaises(rocjitsu.WorkerError):
            rocjitsu.Session({"not_a_config_field": 1})
        with tempfile.TemporaryDirectory() as directory:
            with (
                mock.patch.object(sys, "prefix", directory),
                mock.patch.dict(os.environ, {"PATH": directory}),
                self.assertRaises(FileNotFoundError),
            ):
                rocjitsu.Session(CONFIG)

    def test_large_result(self):
        self.assertEqual(len(rocjitsu.run(lambda: b"x" * 2**20, config=CONFIG)), 2**20)

    def test_cpu_autograd_rejection_preserves_session(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is not installed")
        with rocjitsu.Session(CONFIG) as session:
            with self.assertRaisesRegex(TypeError, "Detach CPU tensors"):
                session.run(lambda value: value, torch.ones(2, requires_grad=True))
            self.assertEqual(session.run(lambda: 42), 42)
            with self.assertRaisesRegex(
                rocjitsu.RemoteError, "Detach CPU tensors"
            ) as error:
                session.run(lambda: torch.ones(2, requires_grad=True))
            self.assertEqual(error.exception.kind, "TypeError")
            self.assertEqual(session.run(lambda: 42), 42)

    def test_worker_keeps_its_own_installed_packages(self):
        import importlib.util

        with tempfile.TemporaryDirectory() as directory:
            site = Path(directory) / "site-packages"
            site.mkdir()
            (site / "only_in_parent_environment.py").write_text("value = 42\n")
            sys.path.insert(0, str(site))
            try:
                self.assertIsNotNone(
                    importlib.util.find_spec("only_in_parent_environment")
                )
                found = rocjitsu.run(
                    lambda: importlib.util.find_spec("only_in_parent_environment")
                    is not None,
                    config=CONFIG,
                    python=sys.executable,
                )
                self.assertFalse(found)
            finally:
                sys.path.remove(str(site))

    def test_worker_import_environment_requires_explicit_overrides(self):
        import importlib.util

        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "caller_environment_module.py").write_text("value = 42\n")

            def found():
                return importlib.util.find_spec("caller_environment_module") is not None

            with mock.patch.dict(os.environ, {"PYTHONPATH": directory}):
                self.assertFalse(
                    rocjitsu.run(found, config=CONFIG, python=sys.executable)
                )
                self.assertTrue(
                    rocjitsu.run(found, config=CONFIG, env={"PYTHONPATH": directory})
                )
            with mock.patch.dict(
                os.environ,
                {
                    "PYTHONHOME": directory,
                    "PYTHONSTARTUP": str(Path(directory, "startup.py")),
                },
            ):
                self.assertEqual(
                    rocjitsu.run(
                        lambda: (
                            os.environ.get("PYTHONHOME"),
                            os.environ.get("PYTHONSTARTUP"),
                        ),
                        config=CONFIG,
                    ),
                    (None, None),
                )

    def test_worker_environment_can_unset_inherited_values(self):
        with mock.patch.dict(os.environ, {"HIP_VISIBLE_DEVICES": "1"}):
            self.assertIsNone(
                rocjitsu.run(
                    lambda: os.environ.get("HIP_VISIBLE_DEVICES"),
                    config=CONFIG,
                    env={"HIP_VISIBLE_DEVICES": None},
                )
            )
            self.assertEqual(os.environ["HIP_VISIBLE_DEVICES"], "1")

    def test_callable_error_drains_pytorch_work(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is not installed")
        if not torch.version.hip:
            self.skipTest("ROCm-enabled PyTorch is not installed")

        def queue_and_fail():
            x = torch.ones((512, 512), device="cuda")
            for _ in range(8):
                torch.mm(x, x)
            event = torch.cuda.Event()
            event.record()
            sys.modules["__main__"].queued_event = event
            raise ValueError("queued-work-failure")

        with rocjitsu.Session(CONFIG) as session:
            with self.assertRaisesRegex(rocjitsu.RemoteError, "queued-work-failure"):
                session.run(queue_and_fail)
            self.assertTrue(
                session.run(lambda: sys.modules["__main__"].queued_event.query())
            )

            def fail_with_sync_error():
                synchronize = torch.cuda.synchronize

                def fail_sync(device):
                    torch.cuda.synchronize = synchronize
                    synchronize(device)
                    raise RuntimeError("synchronization-failure")

                torch.cuda.synchronize = fail_sync
                raise ValueError("callable-failure")

            with self.assertRaises(rocjitsu.RemoteError) as caught:
                session.run(fail_with_sync_error)
            self.assertEqual(caught.exception.kind, "ValueError")
            self.assertEqual(caught.exception.message, "callable-failure")
            self.assertIn("synchronization-failure", caught.exception.remote_traceback)
            self.assertEqual(session.run(lambda: 42), 42)

    def test_worker_does_not_inherit_trailing_caller_search_paths(self):
        import importlib.util

        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "only_on_parent_path.py").write_text("value = 42\n")
            sys.path.append(directory)
            try:
                self.assertIsNotNone(importlib.util.find_spec("only_on_parent_path"))
                self.assertFalse(
                    rocjitsu.run(
                        lambda: importlib.util.find_spec("only_on_parent_path")
                        is not None,
                        config=CONFIG,
                    )
                )
            finally:
                sys.path.remove(directory)

    def test_forked_child_cannot_use_parent_session(self):
        config = json.loads(CONFIG.read_text())
        config["cpu_thread_budget"] = 1
        with rocjitsu.Session(config) as session:
            pid = os.fork()
            if pid == 0:
                try:
                    session.run(lambda: 1)
                except RuntimeError:
                    session.close()
                    atexit._run_exitfuncs()
                    os._exit(0)
                os._exit(1)
            self.assertEqual(os.waitpid(pid, 0)[1], 0)
            self.assertEqual(session.run(lambda: 9), 9)
            # GPU initialization reads the config for the first time here,
            # after the forked child's normal exit callbacks have run.
            self.assertEqual(session.run(simulated_targets), [120001])

    def test_daemon_and_attach_modes(self):
        with tempfile.TemporaryDirectory() as directory:
            env = {"ROCJITSU_RUNTIME_DIR": directory}
            with rocjitsu.Session(
                CONFIG, mode="daemon", cpu_thread_budget=1, env=env, capture_output=True
            ) as session:
                self.assertEqual(session.run(simulated_targets), [120001])
                self.assertEqual(session.run(os.getpid), session.pid)
            launcher = rocjitsu.diagnostics()["executable"]
            with subprocess.Popen(
                [
                    launcher,
                    "--daemon",
                    "--config",
                    str(CONFIG.with_name("gfx1250_mi455x.json")),
                    "--cpu-thread-budget",
                    "1",
                ],
                env={**os.environ, **env},
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                start_new_session=True,
            ) as daemon:
                try:
                    deadline = time.monotonic() + 15
                    while (
                        not Path(directory, "daemon.sock").exists()
                        and time.monotonic() < deadline
                    ):
                        time.sleep(0.01)
                    self.assertTrue(Path(directory, "daemon.sock").exists())
                    for _ in range(2):
                        with rocjitsu.Session(
                            CONFIG, mode="attach", env=env, capture_output=True
                        ) as session:
                            # Attach uses the daemon's target, not the client config.
                            self.assertEqual(
                                session.run(simulated_targets, 1250), [120500]
                            )
                        self.assertIsNone(daemon.poll())
                finally:
                    os.killpg(daemon.pid, signal.SIGKILL)
                    daemon.communicate(timeout=10)

    def test_cli_diagnostics_and_budget(self):
        original = CONFIG.read_bytes()
        result = rocjitsu.diagnostics(CONFIG, cpu_thread_budget=2)
        self.assertIn("native_version", result)
        self.assertIn("Configured |", result["thread_budget_table"])
        self.assertIsInstance(result["vfio_user"], bool)
        self.assertEqual(rocjitsu.cli(["--help"]).returncode, 0)
        with self.assertRaises(subprocess.CalledProcessError) as caught:
            rocjitsu.cli(["--not-a-real-option"])
        self.assertIn("unknown option", caught.exception.stderr)
        self.assertEqual(CONFIG.read_bytes(), original)

    def test_launch_environment_directory_and_budget(self):
        def effective():
            path = Path(os.environ["ROCJITSU_INVOCATION_DIR"], "config_path")
            config_path = path.read_text().splitlines()[0]
            return (
                os.getcwd(),
                os.environ["API_TEST"],
                json.loads(Path(config_path).read_text())["cpu_thread_budget"],
            )

        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(
                rocjitsu.run(
                    effective,
                    config=CONFIG,
                    cpu_thread_budget=2,
                    cwd=directory,
                    env={
                        "API_TEST": "worker",
                        "ROCJITSU_INVOCATION_DIR": "/stale/path",
                    },
                ),
                (directory, "worker", 2),
            )
        for value in (-1, 2**32, True, 1.5):
            with self.assertRaises(ValueError):
                rocjitsu.Session(CONFIG, cpu_thread_budget=value)
        with self.assertRaises(ValueError):
            rocjitsu.Session(CONFIG, mode="attach", cpu_thread_budget=2)

    def test_crash_diagnostics_and_bounded_output(self):
        def crash():
            os.write(1, b"x" * (2**20) + b"stdout-end")
            os.write(2, b"y" * (2**20) + b"stderr-end")
            os._exit(23)

        with rocjitsu.Session(CONFIG, capture_output=True, log_limit=4096) as session:
            with self.assertRaises(rocjitsu.WorkerError) as caught:
                session.run(crash, timeout=30)
            error = caught.exception
            self.assertEqual(error.returncode, 23)
            self.assertEqual(error.command, session.command)
            self.assertTrue(error.stdout.endswith("stdout-end"))
            self.assertTrue(error.stderr.endswith("stderr-end"))
            self.assertLessEqual(len(error.stderr), 4096)
            self.assertTrue(session.closed)
        with self.assertRaises(rocjitsu.WorkerError) as caught:
            rocjitsu.Session({"vm": {"arch": "invalid_arch"}}, capture_output=True)
        self.assertIn("arch", caught.exception.stderr)

    def test_close_cancels_active_call(self):
        with tempfile.TemporaryDirectory() as directory:
            started = Path(directory, "started")
            errors = []

            def workload():
                started.touch()
                time.sleep(60)

            with rocjitsu.Session(CONFIG) as session:

                def call():
                    try:
                        session.run(workload)
                    except Exception as error:
                        errors.append(error)

                thread = threading.Thread(target=call)
                thread.start()
                deadline = time.monotonic() + 10
                while not started.exists() and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertTrue(started.exists())
                session.close()
                thread.join(timeout=10)
                self.assertFalse(thread.is_alive())
                self.assertIsInstance(errors[0], rocjitsu.WorkerError)
                self.assertTrue(session.closed)
                session.close()

    def test_request_deserialization_error_is_recoverable(self):
        class MissingModule:
            def __reduce__(self):
                return (__import__, ("rocjitsu_deliberately_missing_module",))

        with rocjitsu.Session(CONFIG) as session:
            with self.assertRaisesRegex(rocjitsu.RemoteError, "ModuleNotFoundError"):
                session.run(lambda value: value, MissingModule())
            self.assertEqual(session.run(lambda: 17), 17)

    def test_result_deserialization_error_preserves_worker_state(self):
        def worker_value(directory):
            import importlib

            sys.path.insert(0, directory)
            module = importlib.import_module("rocjitsu_worker_only_value")
            module.value = 42
            return module.Value()

        def worker_state():
            import rocjitsu_worker_only_value

            return os.getpid(), rocjitsu_worker_only_value.value

        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "rocjitsu_worker_only_value.py").write_text(
                "class Value:\n    pass\n"
            )
            with rocjitsu.Session(CONFIG) as session:
                pid = session.pid
                with self.assertRaises(ModuleNotFoundError) as caught:
                    session.run(worker_value, directory)
                self.assertEqual(caught.exception.name, "rocjitsu_worker_only_value")
                self.assertFalse(session.closed)
                self.assertEqual(session.run(worker_state), (pid, 42))


if __name__ == "__main__":
    unittest.main()
