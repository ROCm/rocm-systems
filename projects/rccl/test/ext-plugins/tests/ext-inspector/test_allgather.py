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

@pytest.mark.ext_inspector
@pytest.mark.allgather
def test_single_node(paths, inspector_helpers):
    """Validate dump file creation, JSONL schema, and topology fields."""

    dump_dir = os.path.join(paths.INSPECTOR_DUMP_DIR, "allgather_inspector_dumps", "single_node")
    os.makedirs(dump_dir, exist_ok=True)

    # Remove any existing dump files
    for f in glob.glob(os.path.join(dump_dir, "*.log")):
        os.remove(f)

    env = os.environ.copy()
    env.update({
        "PATH": f"{paths.OMPI_INSTALL_DIR}/bin:{env.get('PATH', '')}",
        "LD_LIBRARY_PATH": f"{paths.RCCL_INSTALL_DIR}:{paths.OMPI_INSTALL_DIR}/lib:{paths.INSPECTOR_DIR}:{env.get('LD_LIBRARY_PATH', '')}",
        "HSA_NO_SCRATCH_RECLAIM": "1",
        "NCCL_PROFILER_PLUGIN": paths.INSPECTOR_SO,
        "NCCL_INSPECTOR_ENABLE": "1",
        "NCCL_INSPECTOR_DUMP_THREAD_INTERVAL_MICROSECONDS": "500",
        "NCCL_INSPECTOR_DUMP_DIR": dump_dir,
        "NCCL_DEBUG": "INFO",
    })

    args = [
        f"{paths.OMPI_INSTALL_DIR}/bin/mpirun", "-np", "8",
        "--mca", "pml", "ucx",
        "--mca", "btl", "^vader,openib",
        f"{paths.RCCL_TESTS_DIR}/build/all_gather_perf",
        "-b", "8",
        "-e", "128M",
        "-f", "2",
        "-g", "1",
    ]

    log_dir = os.path.join(paths.LOGDIR, "allgather_ext_inspector_test_logs")
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, "single_node.log")
    with open(log_file, "w") as logfile:
        result = subprocess.run(
            args,
            env=env,
            stdout=logfile,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )

    assert result.returncode == 0, f"AllGather inspector test failed, see {log_file}"

    # Verify inspector plugin initialized
    assert paths.check_event_in_log(log_file, "NCCL Inspector Environment Variables"), \
        f"Inspector plugin should have printed environment variables. Check {log_file}"

    # Verify profiler/plugin init message
    assert paths.check_event_in_log(log_file, "PROFILER/Plugin: init"), \
        f"Inspector plugin should have initialized. Check {log_file}"

    # Verify inspector dump files were created (one per rank)
    dump_files = glob.glob(os.path.join(dump_dir, "*.log"))
    assert len(dump_files) == 8, \
        f"Should have 8 inspector dump files (one per rank), found {len(dump_files)}: {dump_files}"

    # Validate each dump file
    for dump_file in dump_files:
        is_valid, num_records, errors = inspector_helpers.validate_inspector_log_file(dump_file)
        assert is_valid, \
            f"Inspector dump file {dump_file} validation failed: {errors}"
        assert num_records > 0, \
            f"Inspector dump file {dump_file} should have records, found {num_records}"

        # Verify all records are AllGather with correct topology
        with open(dump_file, 'r') as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                assert record["coll_perf"]["coll"] == "AllGather", \
                    f"Record at line {lineno} in {dump_file} should be AllGather, got '{record['coll_perf']['coll']}'"
                assert record["header"]["n_ranks"] == 8, \
                    f"Record at line {lineno} in {dump_file} should have n_ranks=8, got {record['header']['n_ranks']}"
                assert record["header"]["nnodes"] == 1, \
                    f"Record at line {lineno} in {dump_file} should have nnodes=1, got {record['header']['nnodes']}"

@pytest.mark.ext_inspector
@pytest.mark.allgather
def test_single_node_verbose(paths, inspector_helpers):
    """Validate dump file creation, JSONL schema, topology fields, and verbose event trace fields."""

    dump_dir = os.path.join(paths.INSPECTOR_DUMP_DIR, "allgather_inspector_dumps", "single_node_verbose")
    os.makedirs(dump_dir, exist_ok=True)

    # Remove any existing dump files
    for f in glob.glob(os.path.join(dump_dir, "*.log")):
        os.remove(f)

    env = os.environ.copy()
    env.update({
        "PATH": f"{paths.OMPI_INSTALL_DIR}/bin:{env.get('PATH', '')}",
        "LD_LIBRARY_PATH": f"{paths.RCCL_INSTALL_DIR}:{paths.OMPI_INSTALL_DIR}/lib:{paths.INSPECTOR_DIR}:{env.get('LD_LIBRARY_PATH', '')}",
        "HSA_NO_SCRATCH_RECLAIM": "1",
        "NCCL_PROFILER_PLUGIN": paths.INSPECTOR_SO,
        "NCCL_INSPECTOR_ENABLE": "1",
        "NCCL_INSPECTOR_DUMP_THREAD_INTERVAL_MICROSECONDS": "500",
        "NCCL_INSPECTOR_DUMP_DIR": dump_dir,
        "NCCL_INSPECTOR_DUMP_VERBOSE": "1",
        "NCCL_DEBUG": "INFO",
    })

    args = [
        f"{paths.OMPI_INSTALL_DIR}/bin/mpirun", "-np", "8",
        "--mca", "pml", "ucx",
        "--mca", "btl", "^vader,openib",
        f"{paths.RCCL_TESTS_DIR}/build/all_gather_perf",
        "-b", "8",
        "-e", "128M",
        "-f", "2",
        "-g", "1",
    ]

    log_dir = os.path.join(paths.LOGDIR, "allgather_ext_inspector_test_logs")
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, "single_node_verbose.log")
    with open(log_file, "w") as logfile:
        result = subprocess.run(
            args,
            env=env,
            stdout=logfile,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )

    assert result.returncode == 0, f"AllGather inspector verbose test failed, see {log_file}"

    # Verify inspector plugin initialized
    assert paths.check_event_in_log(log_file, "NCCL Inspector Environment Variables"), \
        f"Inspector plugin should have printed environment variables. Check {log_file}"

    # Verify dump files were created (one per rank)
    dump_files = glob.glob(os.path.join(dump_dir, "*.log"))
    assert len(dump_files) == 8, \
        f"Should have 8 inspector dump files (one per rank), found {len(dump_files)}: {dump_files}"

    # Validate each dump file including verbose fields
    for dump_file in dump_files:
        is_valid, num_records, errors = inspector_helpers.validate_inspector_log_file(dump_file)
        assert is_valid, \
            f"Inspector dump file {dump_file} validation failed: {errors}"
        assert num_records > 0, \
            f"Inspector dump file {dump_file} should have records, found {num_records}"

        # Verify verbose event trace fields are present
        with open(dump_file, 'r') as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)

                # Validate standard fields
                assert record["coll_perf"]["coll"] == "AllGather", \
                    f"Record at line {lineno} in {dump_file} should be AllGather"
                assert record["header"]["n_ranks"] == 8
                assert record["header"]["nnodes"] == 1

                # Validate verbose event trace fields
                verbose_valid, verbose_msg = inspector_helpers.validate_inspector_verbose_record(record)
                assert verbose_valid, \
                    f"Verbose validation failed at line {lineno} in {dump_file}: {verbose_msg}"

@pytest.mark.ext_inspector
@pytest.mark.allgather
@pytest.mark.multinode
def test_multinode(paths, inspector_helpers):
    """Validate dump file creation, JSONL schema, and multi-node topology fields."""

    # Get available nodes using the shared function
    nodelist = paths.get_available_nodes()

    # Skip test if no nodes available (SLURM not available) or less than 2 nodes
    if len(nodelist) == 0:
        pytest.skip("No nodes available")
    elif len(nodelist) < 2:
        pytest.skip(f"Multinode test requires at least 2 nodes, but only {len(nodelist)} available: {nodelist}")

    # Check for common network interface across all nodes
    common_interface = paths.find_common_interface(nodelist)
    if common_interface is None:
        pytest.skip("Multinode test requires all nodes to have the same network interface (eth0 or eth1).")

    # Build host specification string (8 processes per node)
    host_spec = ",".join([f"{node}:8" for node in nodelist])
    total_processes = len(nodelist) * 8
    print(f"Using host specification: {host_spec}")

    dump_dir = os.path.join(paths.INSPECTOR_DUMP_DIR, "allgather_inspector_dumps", "multinode")
    os.makedirs(dump_dir, exist_ok=True)

    # Remove any existing dump files
    for f in glob.glob(os.path.join(dump_dir, "*.log")):
        os.remove(f)

    env = os.environ.copy()
    env.update({
        "PATH": f"{paths.OMPI_INSTALL_DIR}/bin:{env.get('PATH', '')}",
        "LD_LIBRARY_PATH": f"{paths.RCCL_INSTALL_DIR}:{paths.OMPI_INSTALL_DIR}/lib:{paths.INSPECTOR_DIR}:{env.get('LD_LIBRARY_PATH', '')}",
        "HSA_NO_SCRATCH_RECLAIM": "1",
        "NCCL_IGNORE_CPU_AFFINITY": "1",
        "NCCL_PROFILER_PLUGIN": paths.INSPECTOR_SO,
        "NCCL_INSPECTOR_ENABLE": "1",
        "NCCL_INSPECTOR_DUMP_THREAD_INTERVAL_MICROSECONDS": "500",
        "NCCL_INSPECTOR_DUMP_DIR": dump_dir,
        "NCCL_DEBUG": "INFO",
        "NCCL_SOCKET_IFNAME": common_interface,
        "NCCL_DMABUF_ENABLE": "1",
    })

    args = [
        f"{paths.OMPI_INSTALL_DIR}/bin/mpirun", "-np", f"{total_processes}",
        "--host", host_spec,
        "--mca", "pml", "ucx",
        "--mca", "btl", "^vader,openib",
        f"{paths.RCCL_TESTS_DIR}/build/all_gather_perf",
        "-b", "8",
        "-e", "128M",
        "-f", "2",
        "-g", "1",
    ]

    log_dir = os.path.join(paths.LOGDIR, "allgather_ext_inspector_test_logs")
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, "multinode.log")
    with open(log_file, "w") as logfile:
        result = subprocess.run(
            args,
            env=env,
            stdout=logfile,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )

    assert result.returncode == 0, f"Multi-node AllGather inspector test failed, see {log_file}"

    # Verify inspector plugin initialized
    assert paths.check_event_in_log(log_file, "NCCL Inspector Environment Variables"), \
        f"Inspector plugin should have printed environment variables. Check {log_file}"

    # Verify inspector dump files were created (one per rank)
    dump_files = glob.glob(os.path.join(dump_dir, "*.log"))
    assert len(dump_files) == total_processes, \
        f"Should have {total_processes} inspector dump files (one per rank), found {len(dump_files)}: {dump_files}"

    # Validate each dump file
    for dump_file in dump_files:
        is_valid, num_records, errors = inspector_helpers.validate_inspector_log_file(dump_file)
        assert is_valid, \
            f"Inspector dump file {dump_file} validation failed: {errors}"
        assert num_records > 0, \
            f"Inspector dump file {dump_file} should have records, found {num_records}"

        # Verify all records are AllGather with correct multi-node topology
        with open(dump_file, 'r') as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                assert record["coll_perf"]["coll"] == "AllGather", \
                    f"Record at line {lineno} in {dump_file} should be AllGather, got '{record['coll_perf']['coll']}'"
                assert record["header"]["n_ranks"] == total_processes, \
                    f"Record at line {lineno} in {dump_file} should have n_ranks={total_processes}, got {record['header']['n_ranks']}"
                assert record["header"]["nnodes"] == len(nodelist), \
                    f"Record at line {lineno} in {dump_file} should have nnodes={len(nodelist)}, got {record['header']['nnodes']}"

    # Verify dump files come from multiple hostnames
    hostnames = set()
    for dump_file in dump_files:
        with open(dump_file, 'r') as f:
            first_line = f.readline().strip()
            if first_line:
                record = json.loads(first_line)
                hostnames.add(record["metadata"]["hostname"])
    assert len(hostnames) >= 2, \
        f"Multi-node test should have dump files from at least 2 hostnames, found {len(hostnames)}: {hostnames}"

@pytest.mark.ext_inspector
@pytest.mark.allgather
@pytest.mark.multinode
def test_multinode_verbose(paths, inspector_helpers):
    """Validate dump file creation, JSONL schema, multi-node topology fields, and verbose event trace fields."""

    # Get available nodes using the shared function
    nodelist = paths.get_available_nodes()

    # Skip test if no nodes available (SLURM not available) or less than 2 nodes
    if len(nodelist) == 0:
        pytest.skip("No nodes available")
    elif len(nodelist) < 2:
        pytest.skip(f"Multinode test requires at least 2 nodes, but only {len(nodelist)} available: {nodelist}")

    # Check for common network interface across all nodes
    common_interface = paths.find_common_interface(nodelist)
    if common_interface is None:
        pytest.skip("Multinode test requires all nodes to have the same network interface (eth0 or eth1).")

    # Build host specification string (8 processes per node)
    host_spec = ",".join([f"{node}:8" for node in nodelist])
    total_processes = len(nodelist) * 8
    print(f"Using host specification: {host_spec}")

    dump_dir = os.path.join(paths.INSPECTOR_DUMP_DIR, "allgather_inspector_dumps", "multinode_verbose")
    os.makedirs(dump_dir, exist_ok=True)

    # Remove any existing dump files
    for f in glob.glob(os.path.join(dump_dir, "*.log")):
        os.remove(f)

    env = os.environ.copy()
    env.update({
        "PATH": f"{paths.OMPI_INSTALL_DIR}/bin:{env.get('PATH', '')}",
        "LD_LIBRARY_PATH": f"{paths.RCCL_INSTALL_DIR}:{paths.OMPI_INSTALL_DIR}/lib:{paths.INSPECTOR_DIR}:{env.get('LD_LIBRARY_PATH', '')}",
        "HSA_NO_SCRATCH_RECLAIM": "1",
        "NCCL_IGNORE_CPU_AFFINITY": "1",
        "NCCL_PROFILER_PLUGIN": paths.INSPECTOR_SO,
        "NCCL_INSPECTOR_ENABLE": "1",
        "NCCL_INSPECTOR_DUMP_THREAD_INTERVAL_MICROSECONDS": "500",
        "NCCL_INSPECTOR_DUMP_DIR": dump_dir,
        "NCCL_INSPECTOR_DUMP_VERBOSE": "1",
        "NCCL_DEBUG": "INFO",
        "NCCL_SOCKET_IFNAME": common_interface,
        "NCCL_DMABUF_ENABLE": "1",
    })

    args = [
        f"{paths.OMPI_INSTALL_DIR}/bin/mpirun", "-np", f"{total_processes}",
        "--host", host_spec,
        "--mca", "pml", "ucx",
        "--mca", "btl", "^vader,openib",
        f"{paths.RCCL_TESTS_DIR}/build/all_gather_perf",
        "-b", "8",
        "-e", "128M",
        "-f", "2",
        "-g", "1",
    ]

    log_dir = os.path.join(paths.LOGDIR, "allgather_ext_inspector_test_logs")
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, "multinode_verbose.log")
    with open(log_file, "w") as logfile:
        result = subprocess.run(
            args,
            env=env,
            stdout=logfile,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )

    assert result.returncode == 0, f"Multi-node AllGather inspector verbose test failed, see {log_file}"

    # Verify inspector plugin initialized
    assert paths.check_event_in_log(log_file, "NCCL Inspector Environment Variables"), \
        f"Inspector plugin should have printed environment variables. Check {log_file}"

    # Verify inspector dump files were created (one per rank)
    dump_files = glob.glob(os.path.join(dump_dir, "*.log"))
    assert len(dump_files) == total_processes, \
        f"Should have {total_processes} inspector dump files (one per rank), found {len(dump_files)}: {dump_files}"

    # Validate each dump file including verbose fields
    for dump_file in dump_files:
        is_valid, num_records, errors = inspector_helpers.validate_inspector_log_file(dump_file)
        assert is_valid, \
            f"Inspector dump file {dump_file} validation failed: {errors}"
        assert num_records > 0, \
            f"Inspector dump file {dump_file} should have records, found {num_records}"

        # Verify verbose event trace fields are present
        with open(dump_file, 'r') as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)

                # Validate standard fields
                assert record["coll_perf"]["coll"] == "AllGather", \
                    f"Record at line {lineno} in {dump_file} should be AllGather"
                assert record["header"]["n_ranks"] == total_processes
                assert record["header"]["nnodes"] == len(nodelist)

                # Validate verbose event trace fields
                verbose_valid, verbose_msg = inspector_helpers.validate_inspector_verbose_record(record)
                assert verbose_valid, \
                    f"Verbose validation failed at line {lineno} in {dump_file}: {verbose_msg}"

