# *************************************************************************
#  * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#  *
#  * See LICENSE.txt for license information
#  ************************************************************************

import os
import subprocess
import pytest
import glob
import json


def _sendrecv_env(paths, dump_dir, extra=None):
    env = os.environ.copy()
    env.update({
        "PATH": f"{paths.OMPI_INSTALL_DIR}/bin:{env.get('PATH', '')}",
        "LD_LIBRARY_PATH": f"{paths.RCCL_INSTALL_DIR}:{paths.OMPI_INSTALL_DIR}/lib:{paths.INSPECTOR_DIR}:{env.get('LD_LIBRARY_PATH', '')}",
        "HSA_NO_SCRATCH_RECLAIM": "1",
        "NCCL_PROFILER_PLUGIN": paths.INSPECTOR_SO,
        "NCCL_INSPECTOR_ENABLE": "1",
        "NCCL_INSPECTOR_DUMP_DIR": dump_dir,
        "NCCL_DEBUG": "INFO",
    })
    if extra:
        env.update(extra)
    return env


def _run_sendrecv(paths, env, log_name):
    args = [
        f"{paths.OMPI_INSTALL_DIR}/bin/mpirun", "-np", "8",
        "--mca", "pml", "ucx",
        "--mca", "btl", "^vader,openib",
        f"{paths.RCCL_TESTS_DIR}/build/sendrecv_perf",
        "-b", "1M",
        "-e", "8M",
        "-f", "2",
        "-g", "1",
    ]

    log_dir = os.path.join(paths.LOGDIR, "sendrecv_ext_inspector_test_logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, log_name)
    with open(log_file, "w") as logfile:
        result = subprocess.run(
            args,
            env=env,
            stdout=logfile,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
    return result, log_file


def _fresh_dump_dir(paths, name):
    dump_dir = os.path.join(paths.INSPECTOR_DUMP_DIR, "sendrecv_inspector_dumps", name)
    os.makedirs(dump_dir, exist_ok=True)
    for f in glob.glob(os.path.join(dump_dir, "*.log")):
        os.remove(f)
    return dump_dir


@pytest.mark.ext_inspector
@pytest.mark.sendrecv
def test_single_node_p2p_records(paths, inspector_helpers):
    """Point-to-point operations are reported with GPU kernel timing.

    Kernel-channel state is keyed by channel id while p2p channel ids are spread across
    the p2p channel space, so a lookup that assumes a packed range finds nothing, falls
    back to CPU timing and is then dropped by NCCL_INSPECTOR_REQUIRE_KERNEL_TIMING.
    The requirement is left at its default here so that regression resurfaces as an
    empty dump.
    """

    dump_dir = _fresh_dump_dir(paths, "single_node")
    env = _sendrecv_env(paths, dump_dir, {
        "NCCL_INSPECTOR_DUMP_THREAD_INTERVAL_MICROSECONDS": "500",
        # DDA bypasses the profiler interface entirely, so keep dispatch instrumented.
        "RCCL_DDA_ENABLE": "0",
    })

    result, log_file = _run_sendrecv(paths, env, "single_node_p2p.log")
    assert result.returncode == 0, f"SendRecv inspector test failed, see {log_file}"

    assert paths.check_event_in_log(log_file, "NCCL Inspector Environment Variables"), \
        f"Inspector plugin should have printed environment variables. Check {log_file}"

    dump_files = glob.glob(os.path.join(dump_dir, "*.log"))
    assert len(dump_files) == 8, \
        f"Should have 8 inspector dump files (one per rank), found {len(dump_files)}: {dump_files}"

    total_p2p = 0
    for dump_file in dump_files:
        n = inspector_helpers.count_inspector_p2p_records(dump_file)
        total_p2p += n
        with open(dump_file, 'r') as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line or "p2p_perf" not in line:
                    continue
                is_valid, record, message = inspector_helpers.validate_inspector_p2p_line(line)
                assert is_valid, \
                    f"P2P record at line {lineno} in {dump_file} is invalid: {message}"

                p2p = record["p2p_perf"]
                assert p2p["p2p"] in ("Send", "Recv"), \
                    f"Unexpected p2p type '{p2p['p2p']}' at line {lineno} in {dump_file}"
                assert p2p["p2p_timing_source"] == "kernel_gpu", \
                    (f"P2P record at line {lineno} in {dump_file} should carry GPU kernel timing, "
                     f"got '{p2p['p2p_timing_source']}'")
                assert p2p["p2p_exec_time_us"] > 0, \
                    f"P2P record at line {lineno} in {dump_file} should have a non-zero exec time"
                assert record["header"]["n_ranks"] == 8

    assert total_p2p > 0, \
        f"SendRecv should produce p2p records across the ranks, found none in {dump_dir}"


@pytest.mark.ext_inspector
@pytest.mark.sendrecv
def test_single_node_dump_at_teardown(paths, inspector_helpers):
    """Output is written when no periodic dump interval is configured.

    NCCL_INSPECTOR_DUMP_THREAD_INTERVAL_MICROSECONDS is left unset so the default (-1)
    applies: no dump thread runs and the only output comes from communicator teardown.
    """

    dump_dir = _fresh_dump_dir(paths, "teardown")
    env = _sendrecv_env(paths, dump_dir, {"RCCL_DDA_ENABLE": "0"})
    env.pop("NCCL_INSPECTOR_DUMP_THREAD_INTERVAL_MICROSECONDS", None)

    result, log_file = _run_sendrecv(paths, env, "single_node_teardown.log")
    assert result.returncode == 0, f"SendRecv inspector teardown test failed, see {log_file}"

    dump_files = glob.glob(os.path.join(dump_dir, "*.log"))
    assert len(dump_files) == 8, \
        f"Should have 8 inspector dump files (one per rank), found {len(dump_files)}: {dump_files}"

    for dump_file in dump_files:
        assert os.path.getsize(dump_file) > 0, \
            (f"{dump_file} is empty: with no periodic interval the teardown dump is the only "
             f"output, so an empty file means it never ran")
        records = inspector_helpers.count_inspector_p2p_records(dump_file)
        assert records > 0, \
            f"{dump_file} should hold p2p records written at teardown, found {records}"
