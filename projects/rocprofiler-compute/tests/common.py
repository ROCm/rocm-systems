# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

import copy
import gzip
import inspect
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import uuid
from pathlib import Path
from threading import Thread
from typing import Any, Dict, Set
from unittest.mock import Mock

import config
from memory_chart.loader import Layouts, PanelConfigs
from utils import csv_compression, metric_library

ROOT = os.path.dirname(os.path.dirname(__file__))
src_candidate = os.path.join(ROOT, "src")
SRC = src_candidate if os.path.isdir(src_candidate) else ROOT
if SRC not in sys.path:
    sys.path.insert(0, SRC)

# Arch chain of write_metric_library_family().
METRIC_LIBRARY_FAMILY_PARENTS = {
    "gfx_base": None,
    "gfx_c1": "gfx_base",
    "gfx_c2": "gfx_c1",
    "gfx_c3": "gfx_c2",
    "gfx_c4": "gfx_c3",
}

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


def memory_chart_archs() -> list[str]:
    """Analysis-config archs that have a memory chart panel."""
    return sorted(
        path.name
        for path in config.analysis_configs_dir.iterdir()
        if path.is_dir() and PanelConfigs.for_arch(path.name)
    )


def memory_chart_layout_data(arch: str) -> dict:
    """The parsed layout file of an arch's memory chart."""
    return json.loads(Layouts.for_arch(arch).path.read_text(encoding="utf-8"))


def with_metric_change(data: dict, metric: str, **fields) -> dict:
    """A copy of layout *data* with *fields* set on every item and arrow of *metric*."""
    data = copy.deepcopy(data)
    items = [item for block in data["blocks"] for item in block.get("content", [])]
    for entry in (*items, *data["arrows"]):
        if entry["metric"] == metric:
            entry.update(fields)
    return data


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

    Callers pass compressed counter artifacts as well as plain files such as
    sysinfo.csv and profiling_config.yaml, so the reader follows the name.
    """
    if str(file_path).endswith(".gz"):
        opener = gzip.open(file_path, "rt", encoding="utf-8")
    else:
        opener = open(file_path, encoding="utf-8")
    with opener as f:
        content = f.read()
    return len(re.findall(pattern, content)) != 0


def write_gzip_csv(path, text):
    """Write text to a gzip CSV through the interface the source uses."""
    with csv_compression.open_gzip_csv_write(path) as f:
        f.write(text)
    return Path(path)


def write_result_csv(workload_dir, text, pass_index=0):
    """Write a compressed rocpd counter artifact into workload_dir."""
    return write_gzip_csv(workload_dir / f"results_pmc_perf_{pass_index}.csv.gz", text)


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


def unique_output_param_id(prefix):
    """Return a get_output_dir param_id that no other pytest worker or run uses."""
    worker = os.environ.get("PYTEST_XDIST_WORKER", "main")
    return f"{prefix}_{worker}_{os.getpid()}_{threading.get_ident()}_{uuid.uuid4().hex}"


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


_KERNEL_SUFFIX_RE = re.compile(r"(?:\s*(?:\[clone \.[^\]]+\]|\.kd))+\s*$")


def normalize_kernel_name(name: str) -> str:
    """Return ``name`` without trailing ``.kd`` or ``[clone ...]`` suffixes."""
    return _KERNEL_SUFFIX_RE.sub("", name).strip()


def normalize_kernel_names(names: Set[str]) -> Set[str]:
    """Return the normalized form of each name in ``names``."""
    return {normalize_kernel_name(name) for name in names}


def write_text_file(path: Path, text: str) -> Path:
    """Write a text file, creating its directory, and return its path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def write_metric_library_family(root: Path) -> Dict[str, Path]:
    """Write a synthetic family: a base and a four-deep chain of compute.yaml files.

    gfx_base defines three metrics; each child inherits from its parent in
    METRIC_LIBRARY_FAMILY_PARENTS and changes one thing: gfx_c1 a description, gfx_c2
    adds a metric, gfx_c3 removes one, gfx_c4 changes a formula and removes a field.

    Returns:
        Each arch's file, keyed by arch.
    """
    yaml_by_arch = {
        "gfx_base": """\
compute.ipc:
  name: IPC
  formula: SUM(SQ_INSTS_VALU) / SUM(SQ_BUSY_CYCLES)
  unit: Instructions per Cycle
  description: ""
compute.salu_util:
  name: SALU Utilization
  formula: 100 * SUM(SQ_ACTIVE_INST_SCA) / SUM($GRBM_GUI_ACTIVE_PER_XCD)
  unit: Percent
  description: How busy the SALU was.
compute.valu_util:
  name: VALU Utilization
  formula: 100 * SUM(SQ_ACTIVE_INST_VALU) / SUM($GRBM_GUI_ACTIVE_PER_XCD)
  unit: Percent
  description: How busy the VALU was.
  peak: 100
""",
        "gfx_c1": """\
compute.salu_util:
  description: How often the SALU was busy.
""",
        "gfx_c2": """\
compute.mfma_flops_f8:
  name: MFMA FLOPs (F8)
  formula: 512 * SUM(SQ_INSTS_VALU_MFMA_MOPS_F8) / SUM($denom)
  unit: FLOP
  description: F8 MFMA operations per normUnit.
""",
        "gfx_c3": """\
compute.ipc: !remove
""",
        "gfx_c4": """\
compute.valu_util:
  formula: 100 * SUM(SQ_ACTIVE_INST_VALU) / SUM($GRBM_GUI_ACTIVE_PER_XCD * 2)
  peak: !remove
""",
    }
    files = {}
    for arch, text in yaml_by_arch.items():
        parent = METRIC_LIBRARY_FAMILY_PARENTS[arch]
        inherit = f"inherit: ../{parent}/compute.yaml\n" if parent else ""
        files[arch] = write_text_file(root / arch / "compute.yaml", inherit + text)
    return files


def use_metric_library_family(monkeypatch: Any, root: Path) -> Dict[str, Path]:
    """Write the synthetic family under root and make it the only metric library chain.

    Returns:
        Each arch's file, keyed by arch.
    """
    monkeypatch.setattr(
        metric_library, "METRIC_LIBRARY_PARENT_ARCH", METRIC_LIBRARY_FAMILY_PARENTS
    )
    monkeypatch.setattr(metric_library.MetricLibrary, "_cache", {})
    return write_metric_library_family(root)
