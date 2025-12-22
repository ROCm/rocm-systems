# *************************************************************************
#  * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
#  *
#  * See LICENSE.txt for license information
#  ************************************************************************

import os
import subprocess
import glob
import pytest

@pytest.mark.ext_profiler
@pytest.mark.reducescatter
def test_profiler_initialization(paths):
    """Test profiler functionality with ReduceScatter operations."""
    
    dump_dir = os.path.join(paths.PROFILER_DUMP_DIR, "reducescatter_profiler_dumps")
    os.makedirs(dump_dir, exist_ok=True)

    dump_file_base = os.path.join(dump_dir, "profiler_initialization")
    
    # Remove any existing trace files
    trace_pattern = f"{dump_file_base}*.json"
    for f in glob.glob(trace_pattern):
        os.remove(f)
    
    env = os.environ.copy()
    env.update({
        "PATH": f"{paths.OMPI_INSTALL_DIR}/bin:{env.get('PATH', '')}",
        "LD_LIBRARY_PATH": f"{paths.RCCL_INSTALL_DIR}:{paths.OMPI_INSTALL_DIR}/lib:{paths.PROFILER_DIR}:{env.get('LD_LIBRARY_PATH', '')}",
        "HSA_NO_SCRATCH_RECLAIM": "1",
        "NCCL_PROFILER_PLUGIN": paths.PROFILER_SO,
        "NCCL_PROFILE_EVENT_MASK": "3",  # Group (1) + Coll (2) = 3
        "NCCL_PROFILE_DUMP_FILE": dump_file_base,
        "NCCL_DEBUG": "INFO",
    })
    
    args = [
        f"{paths.OMPI_INSTALL_DIR}/bin/mpirun", "-np", "4",
        "--mca", "pml", "ucx",
        "--mca", "btl", "^vader,openib",
        f"{paths.RCCL_TESTS_DIR}/build/reduce_scatter_perf",
        "-b", "1",
        "-e", "8M",
        "-f", "2",
        "-g", "1",
    ]
    
    log_dir = os.path.join(paths.LOGDIR, "reducescatter_ext_profiler_test_logs")
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, "profiler_initialization.log")
    with open(log_file, "w") as logfile:
        result = subprocess.run(
            args,
            env=env,
            stdout=logfile,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
    
    assert result.returncode == 0, f"ReduceScatter test failed, see {log_file}"
    
    # Verify plugin initialized
    assert paths.check_event_in_log(log_file, "PROFILER/Plugin: init"), \
        f"Plugin should have initialized. Check {log_file}"
    
    # Verify trace files were created (one per rank)
    trace_files = glob.glob(trace_pattern)
    assert len(trace_files) == 4, \
        f"Should have 4 trace files (one per rank), found {len(trace_files)}: {trace_files}"
    
    # Validate each trace file
    for trace_file in trace_files:
        is_valid, message = paths.validate_json_trace(trace_file)
        assert is_valid, f"Trace file {trace_file} validation failed: {message}"
        
        # Check for Group events
        group_events = paths.count_events_in_trace(trace_file, category="GROUP")
        assert group_events > 0, f"Should have Group events in {trace_file}"
        
        # Check for ReduceScatter collective events
        reducescatter_events = paths.count_events_in_trace(trace_file, event_name="ReduceScatter")
        assert reducescatter_events > 0, f"Should have ReduceScatter events in {trace_file}"


@pytest.mark.ext_profiler
@pytest.mark.reducescatter
def test_invalid_mask_value(paths):
    """Test profiler behavior with invalid event mask (0 = no events)"""
    
    dump_dir = os.path.join(paths.PROFILER_DUMP_DIR, "reducescatter_profiler_dumps")
    os.makedirs(dump_dir, exist_ok=True)

    dump_file_base = os.path.join(dump_dir, "invalid_mask_value_profiling")
    
    # Remove any existing trace files
    trace_pattern = f"{dump_file_base}*.json"
    for f in glob.glob(trace_pattern):
        os.remove(f)
    
    env = os.environ.copy()
    env.update({
        "PATH": f"{paths.OMPI_INSTALL_DIR}/bin:{env.get('PATH', '')}",
        "LD_LIBRARY_PATH": f"{paths.RCCL_INSTALL_DIR}:{paths.OMPI_INSTALL_DIR}/lib:{paths.PROFILER_DIR}:{env.get('LD_LIBRARY_PATH', '')}",
        "HSA_NO_SCRATCH_RECLAIM": "1",
        "NCCL_PROFILER_PLUGIN": paths.PROFILER_SO,
        "NCCL_PROFILE_EVENT_MASK": "0",  # Invalid: no events enabled
        "NCCL_PROFILE_DUMP_FILE": dump_file_base,
        "NCCL_DEBUG": "INFO",
    })
    
    args = [
        f"{paths.OMPI_INSTALL_DIR}/bin/mpirun", "-np", "4",
        "--mca", "pml", "ucx",
        "--mca", "btl", "^vader,openib",
        f"{paths.RCCL_TESTS_DIR}/build/reduce_scatter_perf",
        "-b", "1",
        "-e", "8M",
        "-f", "2",
        "-g", "1",
    ]
    
    log_dir = os.path.join(paths.LOGDIR, "reducescatter_ext_profiler_test_logs")
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, "invalid_mask_value_profiling.log")
    with open(log_file, "w") as logfile:
        result = subprocess.run(
            args,
            env=env,
            stdout=logfile,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
    
    assert result.returncode == 0, f"ReduceScatter test should still succeed even with invalid mask, see {log_file}"
    
    # Verify plugin initialized
    assert paths.check_event_in_log(log_file, "PROFILER/Plugin: init"), \
        f"Plugin should have initialized even with mask=0. Check {log_file}"
    
    # Verify trace files were created (one per rank)
    trace_files = glob.glob(trace_pattern)
    assert len(trace_files) == 4, \
        f"Should have 4 trace files (one per rank), found {len(trace_files)}: {trace_files}"
    
    # Validate each trace file - with mask=0, trace files should be nearly empty
    # They should contain valid JSON but no actual profiling events
    for trace_file in trace_files:
        is_valid, message = paths.validate_json_trace(trace_file)
        assert is_valid, f"Trace file {trace_file} should still be valid JSON: {message}"
        
        # With mask=0, there should be no Group or Collective events
        group_events = paths.count_events_in_trace(trace_file, category="GROUP")
        assert group_events == 0, f"Should have no Group events with mask=0 in {trace_file}, found {group_events}"
        
        reducescatter_events = paths.count_events_in_trace(trace_file, event_name="ReduceScatter")
        assert reducescatter_events == 0, f"Should have no ReduceScatter events with mask=0 in {trace_file}, found {reducescatter_events}"


@pytest.mark.ext_profiler
@pytest.mark.reducescatter
def test_single_node_detailed_profiling(paths):
    """Test profiler with single-node ReduceScatter using full event mask (255) across wide message range"""
    
    dump_dir = os.path.join(paths.PROFILER_DUMP_DIR, "reducescatter_profiler_dumps")
    os.makedirs(dump_dir, exist_ok=True)

    dump_file_base = os.path.join(dump_dir, "single_node_detailed_profiling")
    
    # Remove any existing trace files
    trace_pattern = f"{dump_file_base}*.json"
    for f in glob.glob(trace_pattern):
        os.remove(f)
    
    env = os.environ.copy()
    env.update({
        "PATH": f"{paths.OMPI_INSTALL_DIR}/bin:{env.get('PATH', '')}",
        "LD_LIBRARY_PATH": f"{paths.RCCL_INSTALL_DIR}:{paths.OMPI_INSTALL_DIR}/lib:{paths.PROFILER_DIR}:{env.get('LD_LIBRARY_PATH', '')}",
        "HSA_NO_SCRATCH_RECLAIM": "1",
        "NCCL_PROFILER_PLUGIN": paths.PROFILER_SO,
        "NCCL_PROFILE_EVENT_MASK": "255",  # All events: Group (1) + Coll (2) + P2P (4) + ProxyOp (8) + ProxyStep (16) + ProxyCtrl (32) + KernelCh (64) + NetPlugin (128) = 255
        "NCCL_PROFILE_DUMP_FILE": dump_file_base,
        "NCCL_DEBUG": "INFO",
    })
    
    args = [
        f"{paths.OMPI_INSTALL_DIR}/bin/mpirun", "-np", "8",
        "--bind-to", "none",
        "--mca", "pml", "ucx",
        "--mca", "btl", "^vader,openib",
        f"{paths.RCCL_TESTS_DIR}/build/reduce_scatter_perf",
        "-b", "8",        
        "-e", "128M",       
        "-f", "2",        
        "-g", "1",
    ]
    
    log_dir = os.path.join(paths.LOGDIR, "reducescatter_ext_profiler_test_logs")
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, "single_node_detailed_profiling.log")
    with open(log_file, "w") as logfile:
        result = subprocess.run(
            args,
            env=env,
            stdout=logfile,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
    
    assert result.returncode == 0, f"Single-node detailed ReduceScatter profiling test failed, see {log_file}"
    
    # Verify plugin initialized
    assert paths.check_event_in_log(log_file, "PROFILER/Plugin: init"), \
        f"Plugin should have initialized. Check {log_file}"
    
    # Verify trace files were created (one per rank)
    trace_files = glob.glob(trace_pattern)
    assert len(trace_files) == 8, \
        f"Should have 8 trace files (one per rank), found {len(trace_files)}: {trace_files}"
    
    # Validate each trace file
    for trace_file in trace_files:
        is_valid, message = paths.validate_json_trace(trace_file)
        assert is_valid, f"Trace file {trace_file} validation failed: {message}"
        
        # With NCCL_PROFILE_EVENT_MASK=255, we capture all event types
        # However, single-node behavior differs significantly from multi-node
        
        # Check for Group events
        group_events = paths.count_events_in_trace(trace_file, category="GROUP")
        assert group_events > 0, \
            f"Should have Group events in {trace_file}, found {group_events}"
        
        # Check for ReduceScatter events
        reducescatter_events = paths.count_events_in_trace(trace_file, event_name="ReduceScatter")
        assert reducescatter_events > 0, \
            f"Should have ReduceScatter events in {trace_file}, found {reducescatter_events}"
        
        # Verify GPU kernel channel events exist
        kernel_events = paths.count_events_in_trace(trace_file, category="GPU")
        assert kernel_events > 0, \
            f"Should have GPU (KernelCh) events in {trace_file}, found {kernel_events}"
        
        # Verify ProxyCtrl events exist
        proxy_ctrl_events = paths.count_events_in_trace(trace_file, category="PROXY")
        assert proxy_ctrl_events > 0, \
            f"Should have PROXY (ProxyCtrl) events in {trace_file}, found {proxy_ctrl_events}"
        
        append_events = paths.count_events_in_trace(trace_file, event_name="Append")
        sleep_events = paths.count_events_in_trace(trace_file, event_name="Sleep")
        assert append_events > 0 or sleep_events > 0, \
            f"Should have ProxyCtrl events in {trace_file}, found Append={append_events}, Sleep={sleep_events}"
        
        # We should NOT see ProxyOp network events (ScheduleSend/Recv, ProgressSend/Recv)
        schedule_send_events = paths.count_events_in_trace(trace_file, event_name="ScheduleSend")
        schedule_recv_events = paths.count_events_in_trace(trace_file, event_name="ScheduleRecv")
        assert schedule_send_events == 0, \
            f"Single-node should have NO ScheduleSend events (no network) in {trace_file}, found {schedule_send_events}"
        assert schedule_recv_events == 0, \
            f"Single-node should have NO ScheduleRecv events (no network) in {trace_file}, found {schedule_recv_events}"
        
        # Should also NOT see ProxyStep network events (RecvWait, SendWait, etc.)
        net_events = paths.count_events_in_trace(trace_file, category="NET")
        assert net_events == 0, \
            f"Single-node should have NO NET (ProxyStep) events in {trace_file}, found {net_events}"
        
        # Verify trace file exists and has content
        trace_file_size = os.path.getsize(trace_file)
        assert trace_file_size > 0, \
            f"Trace file {trace_file} is empty"


@pytest.mark.ext_profiler
@pytest.mark.reducescatter
def test_multinode_detailed_profiling(paths):
    """Test profiler with multi-node ReduceScatter operations using full event mask (255)"""
    
    # Get available nodes using the shared function
    nodelist = paths.get_available_nodes()
    
    # Skip test if no nodes available (SLURM not available) or less than 2 nodes
    if not nodelist:
        pytest.skip("Multinode test requires SLURM allocation")
    
    if len(nodelist) < 2:
        pytest.skip(f"Multinode test requires at least 2 nodes, found {len(nodelist)}: {nodelist}")
    
    # Check for common network interface across all nodes
    common_interface = paths.find_common_interface(nodelist)
    if common_interface is None:
        pytest.skip(f"Multinode test requires all nodes to have the same network interface (eth0 or eth1).")
    
    # Build host specification string (8 processes per node)
    host_spec = ",".join([f"{node}:8" for node in nodelist])
    total_processes = len(nodelist) * 8
    print(f"Using host specification: {host_spec}")
    
    dump_dir = os.path.join(paths.PROFILER_DUMP_DIR, "reducescatter_profiler_dumps")
    os.makedirs(dump_dir, exist_ok=True)

    dump_file_base = os.path.join(dump_dir, "multinode_detailed_profiling")
    
    # Remove any existing trace files
    trace_pattern = f"{dump_file_base}*.json"
    for f in glob.glob(trace_pattern):
        os.remove(f)
    
    env = os.environ.copy()
    env.update({
        "PATH": f"{paths.OMPI_INSTALL_DIR}/bin:{env.get('PATH', '')}",
        "LD_LIBRARY_PATH": f"{paths.RCCL_INSTALL_DIR}:{paths.OMPI_INSTALL_DIR}/lib:{paths.PROFILER_DIR}:{env.get('LD_LIBRARY_PATH', '')}",
        "HSA_NO_SCRATCH_RECLAIM": "1",
        "NCCL_IGNORE_CPU_AFFINITY": "1",
        "NCCL_PROFILER_PLUGIN": paths.PROFILER_SO,
        "NCCL_PROFILE_EVENT_MASK": "255",  # All events: Group (1) + Coll (2) + P2P (4) + ProxyOp (8) + ProxyStep (16) + ProxyCtrl (32) + KernelCh (64) + NetPlugin (128) = 255
        "NCCL_PROFILE_DUMP_FILE": dump_file_base,
        "NCCL_DEBUG": "INFO",
        "NCCL_SOCKET_IFNAME": common_interface,
        "NCCL_DMABUF_ENABLE": "1",
    })
    
    args = [
        f"{paths.OMPI_INSTALL_DIR}/bin/mpirun", "-np", f"{total_processes}",
        "--host", host_spec,
        "--mca", "pml", "ucx",
        "--mca", "btl", "^vader,openib",
        f"{paths.RCCL_TESTS_DIR}/build/reduce_scatter_perf",
        "-b", "8",
        "-e", "128M",
        "-f", "2",
        "-g", "1",
    ]
    
    log_dir = os.path.join(paths.LOGDIR, "reducescatter_ext_profiler_test_logs")
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, "multinode_detailed_profiling.log")
    with open(log_file, "w") as logfile:
        result = subprocess.run(
            args,
            env=env,
            stdout=logfile,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
    
    assert result.returncode == 0, f"Multi-node ReduceScatter profiling test failed, see {log_file}"
    
    # Verify plugin initialized
    assert paths.check_event_in_log(log_file, "PROFILER/Plugin: init"), \
        f"Plugin should have initialized. Check {log_file}"
    
    # Verify trace files were created (one per rank)
    trace_files = glob.glob(trace_pattern)
    assert len(trace_files) == total_processes, \
        f"Should have {total_processes} trace files (one per rank), found {len(trace_files)}: {trace_files}"
    
    # Validate each trace file
    for trace_file in trace_files:
        is_valid, message = paths.validate_json_trace(trace_file)
        assert is_valid, f"Trace file {trace_file} validation failed: {message}"
        
        # With NCCL_PROFILE_EVENT_MASK=255, we should capture all event types
        
        # Check for Group events (one per ReduceScatter call)
        group_events = paths.count_events_in_trace(trace_file, category="GROUP")
        assert group_events > 0, f"Should have Group events in {trace_file}, found {group_events}"
        
        # Check for ReduceScatter events
        reducescatter_events = paths.count_events_in_trace(trace_file, event_name="ReduceScatter")
        assert reducescatter_events > 0, \
            f"Should have ReduceScatter events in {trace_file}, found {reducescatter_events}"
        
        # For multi-node tests, verify ProxyOp events exist
        proxy_events = paths.count_events_in_trace(trace_file, category="PROXY")
        assert proxy_events > 0, \
            f"Should have Proxy events in {trace_file}, found {proxy_events}"
        
        # Check for Send and Recv operations
        schedule_send_events = paths.count_events_in_trace(trace_file, event_name="ScheduleSend")
        schedule_recv_events = paths.count_events_in_trace(trace_file, event_name="ScheduleRecv")
        assert schedule_send_events > 0 or schedule_recv_events > 0, \
            f"Should have ScheduleSend or ScheduleRecv events in {trace_file}, found Send={schedule_send_events}, Recv={schedule_recv_events}"
        
        # Verify NET (ProxyStep) events exist
        net_events = paths.count_events_in_trace(trace_file, category="NET")
        assert net_events > 0, \
            f"Should have NET events in {trace_file}, found {net_events}"
        
        # Verify GPU kernel channel events exist
        kernel_events = paths.count_events_in_trace(trace_file, category="GPU")
        assert kernel_events > 0, \
            f"Should have GPU (KernelCh) events in {trace_file}, found {kernel_events}"
        
        # Verify ProxyCtrl events exist
        append_events = paths.count_events_in_trace(trace_file, event_name="Append")
        sleep_events = paths.count_events_in_trace(trace_file, event_name="Sleep")
        assert append_events > 0 or sleep_events > 0, \
            f"Should have ProxyCtrl events in {trace_file}, found Append={append_events}, Sleep={sleep_events}"
        
        # Verify trace file exists and has content
        trace_file_size = os.path.getsize(trace_file)
        assert trace_file_size > 0, \
            f"Trace file {trace_file} is empty"

