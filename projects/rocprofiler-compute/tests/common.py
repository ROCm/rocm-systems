# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

import inspect
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from threading import Thread
from unittest.mock import Mock

ROOT = os.path.dirname(os.path.dirname(__file__))
src_candidate = os.path.join(ROOT, "src")
SRC = src_candidate if os.path.isdir(src_candidate) else ROOT
if SRC not in sys.path:
    sys.path.insert(0, SRC)

# Imported after sys.path is extended, since it lives under src/.
from utils import csv_compression  # noqa: E402

SUPPORTED_ARCHS = {
    "gfx908": {"mi100": ["MI100"]},
    "gfx90a": {"mi200": ["MI210", "MI250", "MI250X"]},
    "gfx940": {"mi300": ["MI300A_A0"]},
    "gfx941": {"mi300": ["MI300X_A0"]},
    "gfx942": {"mi300": ["MI300A_A1", "MI300X_A1"]},
    "gfx950": {"mi350": ["MI350"]},
    "gfx1150": {"rdna35_point_1": ["RDNA35_POINT_1"]},
    "gfx1151": {"rdna35_halo": ["RDNA35_HALO"]},
    "gfx1152": {"rdna35_point_2": ["RDNA35_POINT_2"]},
    "gfx1153": {"rdna35_gorgon_point": ["RDNA35_GORGON_POINT"]},
    "gfx1250": {"gfx1250_series": ["gfx1250"]},
}


def check_resource_allocation():
    """Check if CTEST resource allocation is enabled for parallel testing and set
    HIP_VISIBLE_DEVICES variable accordingly with assigned gpu index.
    """

    if "CTEST_RESOURCE_GROUP_COUNT" not in os.environ:
        return

    if "CTEST_RESOURCE_GROUP_0_GPUS" in os.environ:
        resource = os.environ["CTEST_RESOURCE_GROUP_0_GPUS"]
        # extract assigned gpu id from env var: example format -> 'id:0,slots:1'
        for item in resource.split(","):
            key, value = item.split(":")
            if key == "id":
                os.environ["HIP_VISIBLE_DEVICES"] = value
                return

    return


def check_file_pattern(pattern, file_path):
    """Check if the given pattern exists in the file.

    Opened through the compression boundary so a compressed results file is
    searched by its contents, the same as a plain one.
    """
    content = ""
    with csv_compression.open_csv_read(file_path) as f:
        content = f.read()
    return len(re.findall(pattern, content)) != 0


def get_output_dir(suffix="_output", clean_existing=True, param_id=None):
    """
    Provides a unique output directory based on the name of the calling test function
    with a suffix applied. For parametrized tests, pass param_id to ensure unique
    directory names and avoid NFS conflicts.

    Args:
        suffix (str, optional): suffix to append to output_dir.
            Defaults to "_output".
        clean_existing (bool, optional): Whether to remove existing directory if exists.
            Defaults to True.
        param_id (str, optional): Unique identifier for parametrized tests.
            When provided, appended to the directory name to ensure uniqueness.
            Defaults to None.
    """

    func_name = inspect.stack()[1].function

    param_suffix = ""
    if param_id:
        param_suffix = "_" + re.sub(r"[^\w\-]", "_", str(param_id))

    output_dir = func_name + param_suffix + suffix
    if clean_existing:
        if Path(output_dir).exists():
            shutil.rmtree(output_dir)
    return output_dir


def clean_output_dir(cleanup, output_dir):
    """Remove output directory generated from rocprofiler-compute execution

    Args:
        cleanup (boolean): flag to enable/disable directory cleanup
        output_dir (string): name of directory to remove
    """
    if cleanup:
        if Path(output_dir).exists():
            try:
                shutil.rmtree(output_dir)
            except OSError:
                print(
                    "WARNING: shutil.rmdir(output_dir): directory may not be empty..."
                )
    return


def read_binary_file_tree(root: Path) -> dict[Path, bytes]:
    """Return binary contents keyed by each file's path relative to root."""
    return {
        file_path.relative_to(root): file_path.read_bytes()
        for file_path in root.rglob("*")
        if file_path.is_file()
    }


def strip_ansi(s: str) -> str:
    ansi_escape = re.compile(r"\x1B[@-_][0-?]*[ -/]*[@-~]")
    return ansi_escape.sub("", s)


def _tee(pipe, sink, out) -> None:
    """Echo each line from pipe to sink while accumulating it in out."""
    with pipe:
        for line in pipe:
            print(line, end="", file=sink, flush=True)
            out.append(line)


def run_subprocess(
    command, capture_output=False, stream=False
) -> subprocess.CompletedProcess:
    """Run command in text mode and return a CompletedProcess.

    capture_output: capture stdout and stderr onto the returned object.
    stream: echo output line by line as the child produces it (requires
        capture_output); otherwise captured output is printed once at the end.
    """
    if not capture_output:
        return subprocess.run(command, text=True)

    if not stream:
        # Capture everything, then echo it in one shot after the child exits.
        process = subprocess.run(command, text=True, capture_output=True)
        if process.stdout:
            print(process.stdout, end="")
        if process.stderr:
            print(process.stderr, end="", file=sys.stderr)
        return process

    # Read each pipe on its own thread; reading serially can deadlock if one
    # fills its buffer while we block on the other.
    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    out_buf, err_buf = [], []
    # Tee to the real fds, not sys.stdout/stderr, which pytest's capsys swaps
    # for in-memory buffers that never reach the terminal.
    with os.fdopen(os.dup(1), "w", closefd=True) as real_out, os.fdopen(
        os.dup(2), "w", closefd=True
    ) as real_err:
        readers = [
            Thread(target=_tee, args=(proc.stdout, real_out, out_buf)),
            Thread(target=_tee, args=(proc.stderr, real_err, err_buf)),
        ]
        for r in readers:
            r.start()
        for r in readers:
            r.join()
        proc.wait()
    return subprocess.CompletedProcess(
        command, proc.returncode, "".join(out_buf), "".join(err_buf)
    )


def patch_console(monkeypatch, module, *names, **overrides):
    """Patch ``module.console_<name>`` with a Mock for each name; return {name: Mock}.

    Pass ``name=callable`` to substitute a specific mock (e.g. a record-and-raise
    stub for the console_error exit path).
    """
    mocks = {}
    for name in names:
        mock = overrides.get(name, Mock())
        monkeypatch.setattr(f"{module}.console_{name}", mock)
        mocks[name] = mock
    return mocks
