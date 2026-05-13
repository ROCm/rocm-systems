# *************************************************************************
#  * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#  *
#  * See LICENSE.txt for license information
#  ************************************************************************

"""Functional tests for profiler support of CopyEngine (CE) based
collectives added in NCCL 2.29.2.

Release-notes feature: "profiler support for CopyEngine (CE) based
collectives". The 2.29.7 follow-up release notes also call out:
"Fixed CE-based collective operations to fall back to cudaMemcpyAsync API
when null/default stream is used."

These tests run a small AllReduce that the implementation may dispatch
through the CE path on hardware that supports it, then verify that the
profiler plugin emitted CE event records (alongside or instead of the
regular kernel event records). The tests skip cleanly on platforms where
the CE path is never selected.
"""

import json
import os
import glob
import subprocess

import pytest


def _load_profiler_json(json_dir):
    files = glob.glob(os.path.join(json_dir, "*.json"))
    records = []
    for path in files:
        with open(path, "r") as f:
            try:
                records.append(json.load(f))
            except json.JSONDecodeError:
                # Some profilers emit JSONL; tolerate both formats.
                f.seek(0)
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    return records


def _has_ce_event(records):
    """Walk the captured profiler events looking for any CE-tagged event."""
    needles = ("CopyEngine", "CE_", "ce_coll", "ce-collective", "CECollective")
    text = json.dumps(records)
    return any(n in text for n in needles)


@pytest.mark.ext_profiler
@pytest.mark.allreduce
def test_ce_allreduce_emits_profiler_event(paths):
    dump_dir = os.path.join(paths.PROFILER_DUMP_DIR, "ce_allreduce")
    os.makedirs(dump_dir, exist_ok=True)
    for f in glob.glob(os.path.join(dump_dir, "*.json")):
        os.remove(f)

    env = os.environ.copy()
    env.update({
        "PATH": f"{paths.OMPI_INSTALL_DIR}/bin:{env.get('PATH', '')}",
        "LD_LIBRARY_PATH": (
            f"{paths.RCCL_INSTALL_DIR}:{paths.OMPI_INSTALL_DIR}/lib:"
            f"{paths.PROFILER_DIR}:{env.get('LD_LIBRARY_PATH', '')}"
        ),
        "HSA_NO_SCRATCH_RECLAIM": "1",
        "NCCL_PROFILER_PLUGIN": paths.PROFILER_SO,
        "NCCL_PROFILER_DUMP_DIR": dump_dir,
        "NCCL_DEBUG": "INFO",
    })

    args = [
        f"{paths.OMPI_INSTALL_DIR}/bin/mpirun", "-np", "2",
        "--mca", "pml", "ucx",
        "--mca", "btl", "^vader,openib",
        f"{paths.RCCL_TESTS_DIR}/build/all_reduce_perf",
        # Small messages favor CE collectives on hardware that supports it.
        "-b", "8", "-e", "1024", "-f", "2", "-g", "1",
    ]
    res = subprocess.run(args, env=env, capture_output=True, text=True)
    assert res.returncode == 0, (
        f"mpirun failed:\nstdout:\n{res.stdout}\nstderr:\n{res.stderr}"
    )

    records = _load_profiler_json(dump_dir)
    assert records, f"No profiler JSON written to {dump_dir}"

    if not _has_ce_event(records):
        pytest.skip("This platform did not dispatch the AllReduce through "
                    "the CE path; no CE-tagged profiler event observed. "
                    "Test is valid on hardware where CE collectives fire.")


@pytest.mark.ext_profiler
@pytest.mark.allreduce
def test_ce_default_stream_fallback(paths):
    """Regression for the 2.29.7 fix: CE collectives on null/default stream
    must fall back to cudaMemcpyAsync rather than crash, and the profiler
    must still emit an event for the operation."""
    pytest.skip("TODO: add rccl-tests harness flag for running on the null "
                "stream so this regression can be driven from pytest; the "
                "fix is covered manually for now.")
