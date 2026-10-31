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
            self.assertEqual(tuple(dir(rocjitsu.config)), expected)
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

    def test_cli_names_and_activation_paths_use_installed_configs(self):
        with self.installation() as prefix:
            self.assertIn(
                "thread_budget_table",
                rocjitsu.diagnostics("gfx1201_r9700", cpu_thread_budget=1),
            )
            # A fresh bootstrap has its real prefix; use the installed preset's
            # path while checking global activation with the same native binary.
            code = f"import rocjitsu; rocjitsu.enable({str(rocjitsu.config.gfx1201_r9700)!r}); assert rocjitsu.is_enabled()"
            result = subprocess.run(
                [sys.executable, "-m", "rocjitsu", "-c", code],
                env={**os.environ, "PATH": str(prefix / "bin")},
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

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
    def bootstrap(self, code):
        return subprocess.run(
            [sys.executable, "-m", "rocjitsu", "-c", code],
            text=True,
            capture_output=True,
            timeout=30,
        )

    def test_run_closure_and_keyword_arguments(self):
        offset = 7
        pid, result = rocjitsu.run(
            lambda x, y: (os.getpid(), x + y + offset), 3, y=4, config=CONFIG
        )
        self.assertNotEqual(pid, os.getpid())
        self.assertEqual(result, 14)

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

    def test_timeout_terminates_worker_descendants(self):
        with rocjitsu.Session(CONFIG) as session:
            pid = session.run(
                lambda: subprocess.Popen(
                    [sys.executable, "-c", "import time; time.sleep(60)"]
                ).pid
            )
            with self.assertRaises(TimeoutError):
                session.run(time.sleep, 30, timeout=0.1)
            stat = Path(f"/proc/{pid}/stat")
            try:
                state = stat.read_text().split(")", 1)[1].split()[0]
            except (FileNotFoundError, ProcessLookupError):
                # The system reaper may remove the child while /proc is read.
                pass
            else:
                self.assertEqual(state, "Z")

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
                self.assertNotEqual(
                    Path(f"/proc/{descendant}/stat")
                    .read_text()
                    .split(")", 1)[1]
                    .split()[0],
                    "Z",
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
                stat = Path(f"/proc/{descendant}/stat")
                try:
                    state = stat.read_text().split(")", 1)[1].split()[0]
                except (FileNotFoundError, ProcessLookupError):
                    pass
                else:
                    self.assertEqual(state, "Z")
        finally:
            signal.signal(signal.SIGCHLD, previous)

    def test_delayed_close_with_ignored_sigchld_cleans_descendants(self):
        self.check_delayed_close(ignore_sigchld=True)

    def test_delayed_close_with_external_reaper_cleans_descendants(self):
        self.check_delayed_close(external_reaper=True)

    def test_pre_activation_directory_relative_metadata(self):
        code = f'''import ctypes, errno, os, rocjitsu
fd = os.open('/sys', os.O_RDONLY | os.O_DIRECTORY)
try:
    for operation in (
        lambda: os.stat('class/drm', dir_fd=fd),
        lambda: os.stat('class/drm', dir_fd=fd, follow_symlinks=False),
        lambda: os.readlink('class/drm/card0', dir_fd=fd),
    ):
        try:
            operation()
        except OSError as error:
            assert error.errno == errno.ENODEV, error
        else:
            raise AssertionError('directory-relative discovery reached host')
    assert not os.access('class/drm', os.F_OK, dir_fd=fd)
finally:
    os.close(fd)
libc = ctypes.CDLL(None, use_errno=True)
buf = ctypes.create_string_buffer(512)
for name in ('fstatat', 'fstatat64'):
    fn = getattr(libc, name)
    fn.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p, ctypes.c_int]
    assert fn(-100, b'/dev/dri/renderD199', buf, 0) == -1
    assert ctypes.get_errno() == errno.ENODEV
libc.statx.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_uint, ctypes.c_void_p]
assert libc.statx(-100, b'/dev/dri/renderD199', 0, 0x7ff, buf) == -1
assert ctypes.get_errno() == errno.ENODEV
try:
    rocjitsu.enable({str(CONFIG)!r})
except RuntimeError as error:
    assert 'discovery' in str(error), str(error)
else:
    raise AssertionError('activation after directory-relative discovery accepted')
'''
        result = self.bootstrap(code)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("call enable(config)", result.stderr)

    def test_pre_activation_pci_drm_metadata(self):
        # Deliberately nonexistent BDF/minor: failure must be ENODEV before any
        # host lookup, and the attempted discovery must still freeze activation.
        code = f'''import errno, os, rocjitsu
for path in ('/sys/bus/pci/devices/ffff:ff:1f.7/drm/card199/device/vendor',
             '/sys/devices/pci0000:ff/ffff:ff:1f.7/drm/renderD199/device/vendor'):
    try:
        os.open(path, os.O_RDONLY)
    except OSError as error:
        assert error.errno == errno.ENODEV, error
    else:
        raise AssertionError(path)
try:
    rocjitsu.enable({str(CONFIG)!r})
except RuntimeError as error:
    assert 'discovery' in str(error), str(error)
else:
    raise AssertionError('activation after PCI DRM discovery accepted')
'''
        result = self.bootstrap(code)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_global_activation_with_spawn_and_forkserver(self):
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory, "spawn_test.py")
            script.write_text(
                f'''import json, multiprocessing as mp, os, pathlib, stat, sys
import rocjitsu
config = json.loads(pathlib.Path({str(CONFIG.with_name('gfx1250_mi455x.json'))!r}).read_text())
rocjitsu.enable(config, cpu_thread_budget=1)

def child(queue):
    assert rocjitsu.is_enabled()
    rocjitsu.enable(config, cpu_thread_budget=1)
    fd = os.open('/dev/kfd', os.O_RDWR)
    assert not stat.S_ISCHR(os.fstat(fd).st_mode)
    os.close(fd)
    versions = {{int(line.split()[1])
        for path in pathlib.Path('/sys/class/kfd/kfd/topology/nodes').glob('*/properties')
        for line in path.read_text().splitlines() if line.startswith('gfx_target_version ')}}
    assert versions == {{0, 120500}}, versions
    try:
        rocjitsu.enable(config, cpu_thread_budget=2)
    except RuntimeError:
        pass
    else:
        raise AssertionError('changed inherited config accepted')
    queue.put('child-ok')

if __name__ == '__main__':
    context = mp.get_context(sys.argv[1])
    queue = context.Queue()
    process = context.Process(target=child, args=(queue,))
    process.start()
    process.join(15)
    if process.is_alive():
        process.kill()
        process.join()
        raise AssertionError('spawn child timed out')
    assert process.exitcode == 0, process.exitcode
    assert queue.get(timeout=5) == 'child-ok'
    queue.close()
    print('spawn-ok')
'''
            )
            for method in ("spawn", "forkserver"):
                with self.subTest(method=method):
                    result = subprocess.run(
                        [sys.executable, "-m", "rocjitsu", str(script), method],
                        text=True,
                        capture_output=True,
                        timeout=30,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn("spawn-ok", result.stdout)

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

    def test_enable_cannot_replace_session_launch_configuration(self):
        for mode in ("local", "daemon"):
            with self.subTest(mode=mode), rocjitsu.Session(
                CONFIG, mode=mode, cpu_thread_budget=1, capture_output=True
            ) as session:
                with self.assertRaisesRegex(
                    rocjitsu.RemoteError, "native launch configuration"
                ):
                    session.run(rocjitsu.enable, CONFIG)
                session.run(lambda: os.close(os.open("/dev/kfd", os.O_RDWR)))

    def test_bootstrap_preserves_python_verbose_option(self):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "rocjitsu",
                "-v",
                "-c",
                "print('python-verbose-ran')",
            ],
            text=True,
            capture_output=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("python-verbose-ran", result.stdout)

    def test_pre_activation_refuses_device_and_metadata_paths(self):
        code = f'''import ctypes, errno, os, rocjitsu
libc = ctypes.CDLL(None, use_errno=True)
libc.fopen.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
libc.fopen.restype = ctypes.c_void_p
for path in ('/dev/kfd', '/dev/dri/renderD199', '/sys/class/drm/card0/device/vendor', '/sys/class/kfd/kfd/topology/nodes'):
    for operation in (lambda: os.open(path, os.O_RDONLY), lambda: os.stat(path), lambda: os.lstat(path)):
        try:
            operation()
        except OSError as error:
            assert error.errno == errno.ENODEV, (path, error)
        else:
            raise AssertionError(path)
    assert not libc.fopen(os.fsencode(path), b'r'), path
    assert ctypes.get_errno() == errno.ENODEV
    assert not os.access(path, os.F_OK)
for path in ('/dev/dri', '/sys/class/drm', '/sys/class/kfd'):
    try:
        os.listdir(path)
    except OSError as error:
        assert error.errno == errno.ENODEV, (path, error)
    else:
        raise AssertionError(path)
# Dirfd-relative device opens are refused too; opening /dev does not touch GPUs.
fd = os.open('/dev', os.O_RDONLY | os.O_DIRECTORY)
try:
    os.open('dri/renderD199', os.O_RDONLY, dir_fd=fd)
except OSError as error:
    assert error.errno == errno.ENODEV, error
else:
    raise AssertionError('relative DRM open accepted')
finally:
    os.close(fd)
try:
    rocjitsu.enable({str(CONFIG)!r})
except RuntimeError as error:
    assert 'discovery' in str(error), str(error)
else:
    raise AssertionError('activation after discovery accepted')
'''
        result = self.bootstrap(code)
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

    def test_enable_rejects_late_dlopen(self):
        launcher = Path(rocjitsu.diagnostics()["executable"]).resolve()
        candidates = (
            launcher.parent / "../../librocjitsu.so",
            launcher.parent / "../lib/librocjitsu.so",
            launcher.parent / "../lib64/librocjitsu.so",
        )
        library = next(path.resolve() for path in candidates if path.exists())
        code = f'''import ctypes, rocjitsu
library = ctypes.CDLL({str(library)!r}, mode=ctypes.RTLD_GLOBAL)
try:
    rocjitsu.enable({str(CONFIG)!r})
except RuntimeError as error:
    assert 'preloaded' in str(error)
else:
    raise AssertionError('late dlopen accepted')
'''
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0, result.stderr)

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
            session.run(lambda: os.close(os.open("/dev/kfd", os.O_RDWR)))

    def test_daemon_and_attach_modes(self):
        with tempfile.TemporaryDirectory() as directory:
            env = {"ROCJITSU_RUNTIME_DIR": directory}
            with rocjitsu.Session(
                CONFIG, mode="daemon", cpu_thread_budget=1, env=env, capture_output=True
            ) as session:
                session.run(lambda: os.close(os.open("/dev/kfd", os.O_RDWR)))
                self.assertEqual(session.run(os.getpid), session.pid)
            launcher = rocjitsu.diagnostics()["executable"]
            with subprocess.Popen(
                [
                    launcher,
                    "--daemon",
                    "--config",
                    str(CONFIG),
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
                    for _ in range(2):
                        with rocjitsu.Session(
                            CONFIG, mode="attach", env=env, capture_output=True
                        ) as session:
                            with self.assertRaisesRegex(
                                rocjitsu.RemoteError, "native launch configuration"
                            ):
                                session.run(rocjitsu.enable, CONFIG)
                            session.run(
                                lambda: os.close(os.open("/dev/kfd", os.O_RDWR))
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

    def test_preload_only_rejects_configuration_and_daemon_options(self):
        for option in (
            "--cpu-thread-budget=0",
            "--wait-checking=off",
            "--node=local",
            "--daemon",
            "--attach",
        ):
            with self.subTest(option=option):
                result = rocjitsu.cli(
                    ["--preload-only", option, "--", sys.executable, "-c", "pass"],
                    check=False,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("no configuration options", result.stderr)

    def test_bootstrap_native_flags(self):
        code = "import os; print(os.getpid()); os.close(os.open('/dev/kfd', os.O_RDWR))"
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "rocjitsu",
                "--config",
                str(CONFIG),
                "--cpu-thread-budget",
                "1",
                "--",
                "-c",
                code,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        stdout, stderr = process.communicate(timeout=30)
        self.assertEqual(process.returncode, 0, stderr)
        self.assertEqual(int(stdout.strip()), process.pid)

    def test_enable_snapshot_budget_and_native_diagnostic(self):
        code = f'''import rocjitsu, json, os, pathlib, tempfile
try:
    rocjitsu.enable({{'vm': {{'arch': 'invalid_arch'}}}})
except RuntimeError as error:
    assert 'arch' in str(error), str(error)
else:
    raise AssertionError('invalid config accepted')
with tempfile.TemporaryDirectory() as directory:
    path = pathlib.Path(directory, 'input.json')
    path.write_bytes(pathlib.Path({str(CONFIG)!r}).read_bytes())
    rocjitsu.enable(path, cpu_thread_budget=1)
    rocjitsu.enable(path, cpu_thread_budget=1)
    path.unlink()
handoff = pathlib.Path(os.environ['ROCJITSU_INVOCATION_DIR'], 'config_path')
effective = pathlib.Path(handoff.read_text().strip())
assert json.loads(effective.read_text())['cpu_thread_budget'] == 1
os.close(os.open('/dev/kfd', os.O_RDWR))
'''
        result = self.bootstrap(code)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_enable_requires_preloading(self):
        code = f"import rocjitsu; rocjitsu.enable({str(CONFIG)!r})"
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("python -m rocjitsu", result.stderr)

    def test_enable_idempotence_and_config_change(self):
        code = f'''import rocjitsu, json, os, pathlib, tempfile
config = json.load(open({str(CONFIG)!r}))
pid = os.getpid()
rocjitsu.enable(config)
rocjitsu.enable(config)
rocjitsu.enable({str(CONFIG)!r})
with tempfile.TemporaryDirectory() as directory:
    relaxed = pathlib.Path(directory, 'relaxed.json')
    relaxed.write_text('// native JSON comment\\n' + json.dumps(config))
    rocjitsu.enable(relaxed)
os.close(os.open('/dev/kfd', os.O_RDWR))
rocjitsu.enable(config)
assert rocjitsu.is_enabled() and pid == os.getpid()
config['cpu_thread_budget'] = 3
try:
    rocjitsu.enable(config)
except RuntimeError as error:
    assert 'different configuration' in str(error)
else:
    raise AssertionError('configuration change accepted')
'''
        result = self.bootstrap(code)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_enable_rejects_prior_discovery(self):
        code = f'''import rocjitsu, os
try:
    os.open('/dev/kfd', os.O_RDWR)
except OSError:
    pass
try:
    rocjitsu.enable({str(CONFIG)!r})
except RuntimeError as error:
    assert 'discovery' in str(error), str(error)
else:
    raise AssertionError('late activation accepted')
'''
        result = self.bootstrap(code)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_enable_bad_config_can_be_retried(self):
        code = f'''import rocjitsu
cases = [
    ({{'not_a_config_field': 1}}, ''),
    ({{'dbt_guest': {{'guest_device': {{'num_sdma_engines': 1}}}}}},
     'dbt_guest.guest_device.num_sdma_queues_per_engine'),
    ({{'vm': {{'arch': 'rdna4', 'gpu': {{'device': {{'num_sdma_engines': 1}}}}}}}},
     'vm.gpu.device.num_sdma_queues_per_engine'),
]
for config, diagnostic in cases:
    try:
        rocjitsu.enable(config)
    except RuntimeError as error:
        assert diagnostic in str(error), str(error)
    else:
        raise AssertionError('bad config accepted')
    assert not rocjitsu.is_enabled()
rocjitsu.enable({str(CONFIG)!r})
assert rocjitsu.is_enabled()
'''
        result = self.bootstrap(code)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_failed_non_gpu_lookup_preserves_activation(self):
        code = f'''import errno, os, tempfile, rocjitsu
directory = tempfile.mkdtemp()
os.chdir(directory)
os.rmdir(directory)
try:
    os.stat('not-a-gpu-file')
except OSError as error:
    assert error.errno == errno.ENOENT, error
else:
    raise AssertionError('missing file unexpectedly exists')
os.chdir('/')
try:
    os.stat('not-a-gpu-file', dir_fd=-1)
except OSError as error:
    assert error.errno == errno.EBADF, error
else:
    raise AssertionError('invalid directory descriptor accepted')
rocjitsu.enable({str(CONFIG)!r})
assert rocjitsu.is_enabled()
'''
        result = self.bootstrap(code)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_exec_child_after_enabling_parent_exit_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            result_path = Path(directory, "child-result")
            child_errors = Path(directory, "child-stderr")
            Path(directory, "config_path").write_text(str(CONFIG))
            child_code = f'''import errno, os, pathlib, sys, time
parent = int(sys.argv[1])
deadline = time.monotonic() + 10
while os.getppid() == parent and time.monotonic() < deadline:
    time.sleep(0.01)
assert os.getppid() != parent, 'enabling parent did not exit'
try:
    fd = os.open('/dev/kfd', os.O_RDWR)
except OSError as error:
    assert error.errno == errno.ENODEV, error
else:
    os.close(fd)
    raise AssertionError('child reused another config after enabling parent exit')
pathlib.Path({str(result_path)!r}).write_text('failed-closed')
'''
            code = f'''import os, rocjitsu, subprocess, sys
rocjitsu.enable({str(CONFIG.with_name('gfx1250_mi455x.json'))!r}, cpu_thread_budget=1)
with open({str(child_errors)!r}, 'w') as errors:
    subprocess.Popen([sys.executable, '-c', {child_code!r}, str(os.getpid())],
                     env=dict(os.environ, ROCJITSU_RUNTIME_DIR={directory!r}),
                     stdout=subprocess.DEVNULL, stderr=errors)
'''
            parent = self.bootstrap(code)
            self.assertEqual(parent.returncode, 0, parent.stderr)
            deadline = time.monotonic() + 10
            while not result_path.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertTrue(result_path.exists(), child_errors.read_text())
            self.assertEqual(result_path.read_text(), "failed-closed")

    def test_enabled_config_reaches_exec_children(self):
        code = f'''import rocjitsu, json, subprocess, sys
config = json.load(open({str(CONFIG.with_name('gfx1250_mi455x.json'))!r}))
config['cpu_thread_budget'] = 1
rocjitsu.enable(config)
subprocess.run([sys.executable, '-c', """
import os, pathlib, stat
fd = os.open('/dev/kfd', os.O_RDWR)
assert not stat.S_ISCHR(os.fstat(fd).st_mode)
os.close(fd)
versions = {{int(line.split()[1])
    for path in pathlib.Path('/sys/class/kfd/kfd/topology/nodes').glob('*/properties')
    for line in path.read_text().splitlines() if line.startswith('gfx_target_version ')}}
assert versions == {{0, 120500}}, versions
"""], check=True)
'''
        result = self.bootstrap(code)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_inherited_activation_does_not_use_another_default_config(self):
        code = f'''import errno, os, pathlib, rocjitsu, subprocess, sys, tempfile
rocjitsu.enable({str(CONFIG.with_name('gfx1250_mi455x.json'))!r}, cpu_thread_budget=1)
pathlib.Path(os.environ['ROCJITSU_INVOCATION_DIR'], 'config_path').unlink()
with tempfile.TemporaryDirectory() as directory:
    pathlib.Path(directory, 'config_path').write_text({str(CONFIG)!r})
    environment = dict(os.environ, ROCJITSU_RUNTIME_DIR=directory)
    subprocess.run([sys.executable, '-c', """
import errno, os
try:
    descriptor = os.open('/dev/kfd', os.O_RDWR)
except OSError as error:
    assert error.errno == errno.ENODEV, error
else:
    os.close(descriptor)
    raise AssertionError('inherited activation used another default config')
"""], env=environment, check=True)
'''
        result = self.bootstrap(code)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_activated_directory_relative_metadata_uses_simulated_topology(self):
        code = f'''import ctypes, os, pathlib, rocjitsu
rocjitsu.enable({str(CONFIG)!r}, cpu_thread_budget=1)
path = '/sys/class/kfd/kfd/topology/nodes/1/properties'
expected = os.stat(path)
assert expected.st_size > 0
root = os.open('/', os.O_RDONLY | os.O_DIRECTORY)
try:
    for name in (path, path.lstrip('/')):
        observed = os.stat(name, dir_fd=root, follow_symlinks=False)
        assert (observed.st_dev, observed.st_ino) == (expected.st_dev, expected.st_ino)
        assert os.access(name, os.R_OK, dir_fd=root, follow_symlinks=False)
    # CPU-only hosts have /dev but may have no /dev/dri directory.
    device_directory = os.open('/dev', os.O_RDONLY | os.O_DIRECTORY)
    try:
        expected_drm = os.stat('/dev/dri/renderD128')
        observed_drm = os.stat('dri/renderD128', dir_fd=device_directory)
        assert (observed_drm.st_mode, observed_drm.st_rdev) == (expected_drm.st_mode, expected_drm.st_rdev)
    finally:
        os.close(device_directory)
    directory = os.open('/sys/dev/char', os.O_RDONLY | os.O_DIRECTORY)
    try:
        link = '/sys/dev/char/226:128'
        def read_link(name, **options):
            try:
                return ('target', os.readlink(name, **options))
            except OSError as error:
                return ('errno', error.errno)
        assert read_link('226:128', dir_fd=directory) == read_link(link)
    finally:
        os.close(directory)

    # statx has a Linux-defined layout independent of libc's struct stat aliases.
    class Statx(ctypes.Structure):
        _fields_ = [('mask', ctypes.c_uint), ('blksize', ctypes.c_uint),
                    ('attributes', ctypes.c_uint64), ('nlink', ctypes.c_uint),
                    ('uid', ctypes.c_uint), ('gid', ctypes.c_uint),
                    ('mode', ctypes.c_uint16), ('spare', ctypes.c_uint16),
                    ('ino', ctypes.c_uint64), ('rest', ctypes.c_byte * 216)]
    libc = ctypes.CDLL(None, use_errno=True)
    libc.statx.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
                          ctypes.c_uint, ctypes.POINTER(Statx)]
    metadata = Statx()
    assert ctypes.sizeof(metadata) == 256
    assert libc.statx(root, path.lstrip('/').encode(), 0, 0x7ff,
                      ctypes.byref(metadata)) == 0, ctypes.get_errno()
    assert metadata.ino == expected.st_ino
    assert metadata.mode == expected.st_mode
    drm = os.open('/dev/dri/renderD128', os.O_RDWR)
    try:
        expected_drm = os.fstat(drm)
        assert libc.statx(drm, b'', 0x1000, 0x7ff,
                          ctypes.byref(metadata)) == 0, ctypes.get_errno()
        assert metadata.mode == expected_drm.st_mode
    finally:
        os.close(drm)
finally:
    os.close(root)
'''
        result = self.bootstrap(code)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
