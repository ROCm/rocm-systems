# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Configuration ownership and native launcher options shared by the public APIs."""

from collections.abc import Mapping
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile

INVOCATION_DIR_ENV = "ROCJITSU_INVOCATION_DIR"
DAEMON_PID_ENV = "ROCJITSU_DAEMON_PID"
PROGRAMMATIC_ENV = "ROCJITSU_PROGRAMMATIC"


def launcher(executable=None):
    if sys.platform != "linux":
        raise RuntimeError("The rocjitsu execution API requires Linux")
    name = os.fspath(executable or os.environ.get("ROCJITSU_EXECUTABLE", "rocjitsu"))
    path = shutil.which(name)
    if path is None:
        raise FileNotFoundError(
            "Cannot find rocjitsu; set ROCJITSU_EXECUTABLE or executable="
        )
    return str(Path(path).absolute())


def positive_timeout(value, name, *, optional=False):
    if optional and value is None:
        return
    if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")


def budget_option(value):
    if value is None:
        return []
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 2**32:
        raise ValueError("cpu_thread_budget must be an integer from 0 to 4294967295")
    return ["--cpu-thread-budget", str(value)]


def launch_options(mode, cpu_thread_budget):
    if mode not in ("local", "daemon", "attach"):
        raise ValueError("mode must be 'local', 'daemon', or 'attach'")
    budget = budget_option(cpu_thread_budget)
    if mode == "attach" and budget:
        raise ValueError("cpu_thread_budget cannot be combined with mode='attach'")
    return ([] if mode == "local" else [f"--{mode}"]) + budget


class Config:
    def __init__(self, config):
        self._owner = os.getpid()
        self.directory = None
        if isinstance(config, Mapping):
            text = json.dumps(dict(config), allow_nan=False)
            self.prepare_directory()
            self.path = str(Path(self.directory) / "config.json")
            try:
                Path(self.path).write_text(text, encoding="utf-8")
            except BaseException:
                self.close()
                raise
        else:
            # Keep path configs in place: native DBT resolves sibling configs
            # relative to this path. The caller owns the source file's lifetime.
            self.path = str(Path(config).resolve(strict=True))
            if not Path(self.path).is_file():
                raise ValueError("config must name a file")

    def prepare_directory(self):
        if self.directory is None:
            self.directory = tempfile.mkdtemp(prefix="rocjitsu-python-")
        return self.directory

    def close(self):
        if self.directory is not None and self._owner == os.getpid():
            shutil.rmtree(self.directory, ignore_errors=True)
            self.directory = None


def environment(env=None):
    result = os.environ.copy()
    result.update(env or {})
    # Each invocation gets its own handoff. Never reuse a parent's VM/daemon.
    result.pop(INVOCATION_DIR_ENV, None)
    result.pop(DAEMON_PID_ENV, None)
    result.pop(PROGRAMMATIC_ENV, None)
    # The chosen interpreter must have this package installed. Adding the
    # caller's package parent here would inject its entire site-packages into
    # a different ROCm environment, replacing that worker's framework stack.
    return result


def cli(
    args, *, config=None, executable=None, env=None, cwd=None, timeout=30, check=True
):
    """Run a native CLI command and return subprocess.CompletedProcess[str].

    Arguments are separate tokens, never shell text. This is also the escape
    hatch for standalone modes such as VFIO. Output is captured; timeout follows
    subprocess.run semantics. Prefer Session for Python callable execution.
    """
    if isinstance(args, (str, bytes)):
        raise TypeError("args must be a sequence of argument tokens, not shell text")
    positive_timeout(timeout, "timeout", optional=True)
    owned = Config(config) if config is not None else None
    try:
        command = [launcher(executable)]
        if owned is not None:
            command += ["--config", owned.path]
        command += [os.fspath(arg) for arg in args]
        with subprocess.Popen(
            command,
            env=environment(env),
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
            start_new_session=True,
        ) as process:
            try:
                stdout, stderr = process.communicate(timeout=timeout)
            except BaseException:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.communicate()
                raise
            result = subprocess.CompletedProcess(
                command, process.returncode, stdout, stderr
            )
            if check:
                result.check_returncode()
            return result
    finally:
        if owned is not None:
            owned.close()


def diagnostics(
    config=None, *, cpu_thread_budget=None, executable=None, env=None, timeout=30
):
    """Query the native version, VFIO capability and optional thread allocation.

    No GPU runtime is initialized. CLI failures other than an unavailable VFIO
    capability raise CalledProcessError, including the native stdout/stderr.
    """
    budget = budget_option(cpu_thread_budget)
    if config is None and budget:
        raise ValueError("cpu_thread_budget requires config")
    path = launcher(executable)
    options = dict(executable=path, env=env, timeout=timeout)
    result = {
        "executable": path,
        "python": sys.executable,
        "python_version": sys.version.split()[0],
        "native_version": cli(["--version"], **options).stdout.strip(),
    }
    vfio = cli(["--check-vfio-user"], check=False, **options)
    result["vfio_user"] = vfio.returncode == 0
    result["vfio_diagnostic"] = (vfio.stdout + vfio.stderr).strip()
    if config is not None:
        result["thread_budget_table"] = cli(
            ["--thread-budget-table", *budget], config=config, **options
        ).stdout
    return result
