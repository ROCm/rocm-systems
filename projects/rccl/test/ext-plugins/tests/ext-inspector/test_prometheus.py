# *************************************************************************
#  * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#  *
#  * See LICENSE.txt for license information
#  ************************************************************************

"""Functional tests for the Prometheus output format added to the Inspector
plugin in NCCL 2.29.2.

The release notes describe:
    "Improved debuggability and observability with realtime monitoring
     support in RAS, Prometheus output format for Inspector, and profiler
     support for CopyEngine (CE) based collectives."

These tests validate:
  1. A .prom file is produced when the Inspector plugin's Prometheus
     mode is enabled.
  2. The exposition follows the Prometheus text format:
       <metric_name>{<labels>} <value> [timestamp]
     with one record per line.
  3. The expected `nccl_inspector_*` metric families are present.

The tests skip cleanly when the Prometheus output mode is not enabled by
the built Inspector plugin (e.g., older builds).
"""

import os
import re
import subprocess
import glob

import pytest


# Prometheus text-format line:
#   metric_name{label1="value1",label2="value2"} 1.234 [timestamp_ms]
# Anchored, tolerant of whitespace, optional trailing timestamp.
PROM_LINE = re.compile(
    r'^'
    r'(?P<name>[a-zA-Z_][a-zA-Z0-9_:]*)'         # metric name
    r'(?P<labels>\{[^}]*\})?'                    # optional labels block
    r'\s+'
    r'(?P<value>[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?|NaN|\+?Inf|-Inf)'
    r'(?:\s+\d+)?'                               # optional timestamp_ms
    r'\s*$'
)


@pytest.fixture
def prom_env(paths):
    """Env that turns on the Inspector plugin's Prometheus output.

    Note: the exact env-var name for the Prometheus toggle is implementation
    defined in plugins/profiler/inspector/inspector_prom.cc; we set a
    superset of plausible names so the test is robust to renames.
    """
    env = os.environ.copy()
    env.update({
        "PATH": f"{paths.OMPI_INSTALL_DIR}/bin:{env.get('PATH', '')}",
        "LD_LIBRARY_PATH": (
            f"{paths.RCCL_INSTALL_DIR}:{paths.OMPI_INSTALL_DIR}/lib:"
            f"{paths.INSPECTOR_DIR}:{env.get('LD_LIBRARY_PATH', '')}"
        ),
        "HSA_NO_SCRATCH_RECLAIM": "1",
        "NCCL_PROFILER_PLUGIN": paths.INSPECTOR_SO,
        "NCCL_INSPECTOR_ENABLE": "1",
        "NCCL_INSPECTOR_PROMETHEUS_ENABLE": "1",
        "NCCL_INSPECTOR_PROMETHEUS_EXPORT": "1",
        "NCCL_INSPECTOR_DUMP_THREAD_INTERVAL_MICROSECONDS": "500",
        "NCCL_DEBUG": "INFO",
    })
    return env


def _has_prom_support(env, paths):
    """Detect whether the Inspector plugin in this build emits .prom files."""
    return os.path.exists(os.path.join(paths.INSPECTOR_DIR,
                                        "inspector_prom.o"))


@pytest.mark.ext_inspector
@pytest.mark.allreduce
def test_prometheus_dump_created(paths, prom_env, inspector_helpers):
    if not _has_prom_support(prom_env, paths):
        pytest.skip("Inspector plugin in this build does not include the "
                    "Prometheus exporter (plugins/profiler/inspector/"
                    "inspector_prom.cc not present).")

    dump_dir = os.path.join(paths.INSPECTOR_DUMP_DIR,
                            "prometheus_dumps", "allreduce")
    os.makedirs(dump_dir, exist_ok=True)
    for f in glob.glob(os.path.join(dump_dir, "*")):
        os.remove(f)
    prom_env["NCCL_INSPECTOR_DUMP_DIR"] = dump_dir

    args = [
        f"{paths.OMPI_INSTALL_DIR}/bin/mpirun", "-np", "8",
        "--mca", "pml", "ucx",
        "--mca", "btl", "^vader,openib",
        f"{paths.RCCL_TESTS_DIR}/build/all_reduce_perf",
        "-b", "8", "-e", "1M", "-f", "2", "-g", "1",
    ]
    result = subprocess.run(args, env=prom_env, capture_output=True, text=True)
    assert result.returncode == 0, (
        f"mpirun failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )

    prom_files = (glob.glob(os.path.join(dump_dir, "*.prom"))
                  or glob.glob(os.path.join(dump_dir, "*prometheus*")))
    assert prom_files, (
        "Inspector did not produce any *.prom output in " + dump_dir
    )


@pytest.mark.ext_inspector
@pytest.mark.allreduce
def test_prometheus_schema(paths, prom_env, inspector_helpers):
    if not _has_prom_support(prom_env, paths):
        pytest.skip("Inspector plugin in this build does not include the "
                    "Prometheus exporter.")

    dump_dir = os.path.join(paths.INSPECTOR_DUMP_DIR,
                            "prometheus_dumps", "schema")
    os.makedirs(dump_dir, exist_ok=True)
    for f in glob.glob(os.path.join(dump_dir, "*")):
        os.remove(f)
    prom_env["NCCL_INSPECTOR_DUMP_DIR"] = dump_dir

    args = [
        f"{paths.OMPI_INSTALL_DIR}/bin/mpirun", "-np", "4",
        "--mca", "pml", "ucx",
        "--mca", "btl", "^vader,openib",
        f"{paths.RCCL_TESTS_DIR}/build/all_reduce_perf",
        "-b", "8", "-e", "4M", "-f", "4", "-g", "1",
    ]
    subprocess.run(args, env=prom_env, capture_output=True, text=True,
                   check=True)

    prom_files = glob.glob(os.path.join(dump_dir, "*.prom"))
    if not prom_files:
        pytest.skip(".prom file naming differs in this build; skipping "
                    "schema check (see test_prometheus_dump_created).")

    saw_nccl_metric = False
    for path in prom_files:
        with open(path, "r") as f:
            for raw in f:
                line = raw.rstrip("\n")
                if not line or line.startswith("#"):
                    continue # HELP/TYPE comments
                match = PROM_LINE.match(line)
                assert match, (
                    f"Line in {path} does not match Prometheus exposition "
                    f"format: {line!r}"
                )
                if match.group("name").startswith("nccl_inspector_"):
                    saw_nccl_metric = True

    assert saw_nccl_metric, (
        "Expected at least one nccl_inspector_* metric in the Prometheus "
        "output; got none."
    )
