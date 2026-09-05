# *************************************************************************
#  * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
#  *
#  * See LICENSE.txt for license information
#  ************************************************************************

import os
import subprocess
import pytest
import glob
import json

@pytest.mark.ext_profiler
@pytest.mark.allreduce
def test_profiler_initialization(paths):
    """Test profiler functionality with AllReduce operations."""
    
    dump_dir = os.path.join(paths.PROFILER_DUMP_DIR, "allreduce_profiler_dumps")
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
        f"{paths.RCCL_TESTS_DIR}/build/all_reduce_perf",
        "-b", "1",
        "-e", "8M",
        "-f", "2",
        "-g", "1",
    ]
    
    log_dir = os.path.join(paths.LOGDIR, "allreduce_ext_profiler_test_logs")
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
    
    assert result.returncode == 0, f"AllReduce profiling test failed, see {log_file}"
    
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
        
        # Check for Group API events
        group_events = paths.count_events_in_trace(trace_file, category="GROUP_API")
        assert group_events > 0, f"Should have Group API events in {trace_file}"
        
        # Check for AllReduce events
        allreduce_events = paths.count_events_in_trace(trace_file, event_name="AllReduce")
        assert allreduce_events > 0, f"Should have AllReduce events in {trace_file}"


@pytest.mark.ext_profiler
@pytest.mark.allreduce
def test_group_events_traced(paths):
    """GROUP spans reach the trace and do not duplicate the collectives they cover.

    A group event carries no parent, so a plugin that requires one drops it before it is
    ever pooled and the category never appears. The span is emitted on its own because the
    task events it covers are already written under the collective API entry; counting COLL
    against COLL_API catches a regression that walks the group queue again.
    """

    dump_dir = os.path.join(paths.PROFILER_DUMP_DIR, "allreduce_profiler_dumps")
    os.makedirs(dump_dir, exist_ok=True)

    dump_file_base = os.path.join(dump_dir, "group_events")

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
        # DDA launches its own kernels and bypasses the instrumented path, which would
        # leave the trace without COLL or GROUP entries whatever the mask asks for.
        "RCCL_DDA_ENABLE": "0",
        "NCCL_DEBUG": "INFO",
    })

    args = [
        f"{paths.OMPI_INSTALL_DIR}/bin/mpirun", "-np", "4",
        "--mca", "pml", "ucx",
        "--mca", "btl", "^vader,openib",
        f"{paths.RCCL_TESTS_DIR}/build/all_reduce_perf",
        "-b", "1M",
        "-e", "8M",
        "-f", "2",
        "-g", "1",
    ]

    log_dir = os.path.join(paths.LOGDIR, "allreduce_ext_profiler_test_logs")
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, "group_events.log")
    with open(log_file, "w") as logfile:
        result = subprocess.run(
            args,
            env=env,
            stdout=logfile,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )

    assert result.returncode == 0, f"AllReduce group event test failed, see {log_file}"

    trace_files = glob.glob(trace_pattern)
    assert len(trace_files) == 4, \
        f"Should have 4 trace files (one per rank), found {len(trace_files)}: {trace_files}"

    for trace_file in trace_files:
        is_valid, message = paths.validate_json_trace(trace_file)
        assert is_valid, f"Trace file {trace_file} validation failed: {message}"

        group_events = paths.count_events_in_trace(trace_file, category="GROUP")
        assert group_events > 0, \
            f"Should have GROUP events in {trace_file}, found {group_events}"
        assert group_events % 2 == 0, \
            f"Each GROUP span needs a begin and an end, found {group_events} entries in {trace_file}"

        coll_events = paths.count_events_in_trace(trace_file, category="COLL")
        coll_api_events = paths.count_events_in_trace(trace_file, category="COLL_API")
        assert coll_events == coll_api_events, \
            (f"{trace_file} holds {coll_events} COLL entries against {coll_api_events} COLL_API "
             f"entries; tracing group spans must not emit the collectives a second time")


@pytest.mark.ext_profiler
@pytest.mark.allreduce
def test_group_pool_is_recycled(paths):
    """Group slots return to the pool, so tracing survives to the end of a run.

    Nothing parents to a group event, so its slot is only freed when stopEvent drops
    the reference taken at start. Without that the pool never rotates and every group
    after the first groupPoolSize is dropped, leaving only groups from the very start
    of the run. Shrinking the pool makes that visible in a short run.

    The plugin dumps whatever the pools still hold at finalize, so the count of GROUP
    entries is always bounded by the pool size; what distinguishes a rotating pool is
    that the survivors are the newest groups rather than the first few.
    """

    pool_size = 4

    dump_dir = os.path.join(paths.PROFILER_DUMP_DIR, "allreduce_profiler_dumps")
    os.makedirs(dump_dir, exist_ok=True)

    dump_file_base = os.path.join(dump_dir, "group_pool")

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
        "NCCL_PROFILE_GROUP_POOL_SIZE": str(pool_size),
        "RCCL_DDA_ENABLE": "0",
        "NCCL_DEBUG": "INFO",
    })

    args = [
        f"{paths.OMPI_INSTALL_DIR}/bin/mpirun", "-np", "4",
        "--mca", "pml", "ucx",
        "--mca", "btl", "^vader,openib",
        f"{paths.RCCL_TESTS_DIR}/build/all_reduce_perf",
        "-b", "1M",
        "-e", "8M",
        "-f", "2",
        "-g", "1",
    ]

    log_dir = os.path.join(paths.LOGDIR, "allreduce_ext_profiler_test_logs")
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, "group_pool.log")
    with open(log_file, "w") as logfile:
        result = subprocess.run(
            args,
            env=env,
            stdout=logfile,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )

    assert result.returncode == 0, f"AllReduce group pool test failed, see {log_file}"

    trace_files = glob.glob(trace_pattern)
    assert len(trace_files) == 4, \
        f"Should have 4 trace files (one per rank), found {len(trace_files)}: {trace_files}"

    for trace_file in trace_files:
        is_valid, message = paths.validate_json_trace(trace_file)
        assert is_valid, f"Trace file {trace_file} validation failed: {message}"

        with open(trace_file) as f:
            events = json.load(f)

        group_ts = [e["ts"] for e in events if e.get("cat") == "GROUP" and e.get("ts", 0) > 0]
        all_ts = [e["ts"] for e in events if e.get("ts", 0) > 0]
        assert group_ts, f"Should have GROUP events in {trace_file}, found none"

        # A pool that never rotates keeps its first groupPoolSize occupants, so every
        # survivor sits at the very start of the trace instead of the end.
        first, last = min(all_ts), max(all_ts)
        cutoff = first + 0.5 * (last - first)
        assert max(group_ts) > cutoff, \
            (f"{trace_file} traced its newest group at {max(group_ts):.0f}, in the first half "
             f"of the {first:.0f}..{last:.0f} run, with a {pool_size}-slot pool; slots are not "
             f"being returned, so groups past the pool size were dropped")


@pytest.mark.ext_profiler
@pytest.mark.allreduce
def test_plugin_short_name_resolution(paths):
    """A bare plugin name resolves to the librccl- prefixed library.

    A value with no '/' and no 'lib' prefix or '.so' suffix is expanded to
    librccl-profiler-<name>.so and looked up on the loader path, so the plugin can be
    selected without spelling out a full path.
    """

    dump_dir = os.path.join(paths.PROFILER_DUMP_DIR, "allreduce_profiler_dumps")
    os.makedirs(dump_dir, exist_ok=True)

    dump_file_base = os.path.join(dump_dir, "short_name")

    trace_pattern = f"{dump_file_base}*.json"
    for f in glob.glob(trace_pattern):
        os.remove(f)

    env = os.environ.copy()
    env.update({
        "PATH": f"{paths.OMPI_INSTALL_DIR}/bin:{env.get('PATH', '')}",
        "LD_LIBRARY_PATH": f"{paths.RCCL_INSTALL_DIR}:{paths.OMPI_INSTALL_DIR}/lib:{paths.PROFILER_DIR}:{env.get('LD_LIBRARY_PATH', '')}",
        "HSA_NO_SCRATCH_RECLAIM": "1",
        # Bare name rather than a path: expanded to librccl-profiler-example.so.
        "NCCL_PROFILER_PLUGIN": "example",
        "NCCL_PROFILE_EVENT_MASK": "3",  # Group (1) + Coll (2) = 3
        "NCCL_PROFILE_DUMP_FILE": dump_file_base,
        "RCCL_DDA_ENABLE": "0",
        "NCCL_DEBUG": "INFO",
    })

    args = [
        f"{paths.OMPI_INSTALL_DIR}/bin/mpirun", "-np", "4",
        "--mca", "pml", "ucx",
        "--mca", "btl", "^vader,openib",
        f"{paths.RCCL_TESTS_DIR}/build/all_reduce_perf",
        "-b", "1M",
        "-e", "4M",
        "-f", "2",
        "-g", "1",
    ]

    log_dir = os.path.join(paths.LOGDIR, "allreduce_ext_profiler_test_logs")
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, "short_name.log")
    with open(log_file, "w") as logfile:
        result = subprocess.run(
            args,
            env=env,
            stdout=logfile,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )

    assert result.returncode == 0, f"Short plugin name test failed, see {log_file}"

    assert paths.check_event_in_log(log_file, "PROFILER/Plugin: init"), \
        f"Plugin given as a bare name should have been found and initialized. Check {log_file}"

    # Traces prove the resolved library is the example plugin rather than a silent no-op.
    trace_files = glob.glob(trace_pattern)
    assert len(trace_files) == 4, \
        f"Should have 4 trace files (one per rank), found {len(trace_files)}: {trace_files}"


@pytest.mark.ext_profiler
@pytest.mark.allreduce
def test_ce_events_traced(paths):
    """A Copy-Engine AllReduce reports its CE_COLL, CE_SYNC and CE_BATCH events.

    CE dispatch needs symmetric memory plus the CE CTA policy, and DDA would otherwise
    claim the collective before CE is considered. The run is skipped when the log shows
    CE was not selected, so an unsupported driver or arch does not read as a failure.
    """

    dump_dir = os.path.join(paths.PROFILER_DUMP_DIR, "allreduce_profiler_dumps")
    os.makedirs(dump_dir, exist_ok=True)

    dump_file_base = os.path.join(dump_dir, "ce_events")

    trace_pattern = f"{dump_file_base}*.json"
    for f in glob.glob(trace_pattern):
        os.remove(f)

    env = os.environ.copy()
    env.update({
        "PATH": f"{paths.OMPI_INSTALL_DIR}/bin:{env.get('PATH', '')}",
        "LD_LIBRARY_PATH": f"{paths.RCCL_INSTALL_DIR}:{paths.OMPI_INSTALL_DIR}/lib:{paths.PROFILER_DIR}:{env.get('LD_LIBRARY_PATH', '')}",
        "HSA_NO_SCRATCH_RECLAIM": "1",
        "NCCL_PROFILER_PLUGIN": paths.PROFILER_SO,
        "NCCL_PROFILE_EVENT_MASK": "28672",  # CeColl (4096) + CeSync (8192) + CeBatch (16384)
        "NCCL_PROFILE_DUMP_FILE": dump_file_base,
        # CE dispatch prerequisites.
        "RCCL_CE_ALLREDUCE": "1",
        "RCCL_FORCE_CE_ALLREDUCE": "1",
        "NCCL_CTA_POLICY": "2",
        "NCCL_LOCAL_REGISTER": "0",
        "NCCL_CUMEM_ENABLE": "1",
        # DDA is checked before CE, so leave it off or the collective never reaches CE.
        "RCCL_DDA_ENABLE": "0",
        "NCCL_DEBUG": "INFO",
        # The dispatch line below is logged under the COLL subsystem, which the default
        # mask leaves out.
        "NCCL_DEBUG_SUBSYS": "INIT,COLL",
    })

    args = [
        f"{paths.OMPI_INSTALL_DIR}/bin/mpirun", "-np", "8",
        "--mca", "pml", "ucx",
        "--mca", "btl", "^vader,openib",
        f"{paths.RCCL_TESTS_DIR}/build/all_reduce_perf",
        "-b", "1M",
        "-e", "8M",
        "-f", "2",
        "-g", "1",
    ]

    log_dir = os.path.join(paths.LOGDIR, "allreduce_ext_profiler_test_logs")
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, "ce_events.log")
    with open(log_file, "w") as logfile:
        result = subprocess.run(
            args,
            env=env,
            stdout=logfile,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )

    assert result.returncode == 0, f"CE AllReduce profiling test failed, see {log_file}"

    if not paths.check_event_in_log(log_file, "CE 2-shot AllReduce"):
        pytest.skip(f"CE AllReduce was not dispatched on this configuration, see {log_file}")

    trace_files = glob.glob(trace_pattern)
    assert len(trace_files) == 8, \
        f"Should have 8 trace files (one per rank), found {len(trace_files)}: {trace_files}"

    total_ce_coll = 0
    for trace_file in trace_files:
        is_valid, message = paths.validate_json_trace(trace_file)
        assert is_valid, f"Trace file {trace_file} validation failed: {message}"

        ce_coll = paths.count_events_in_trace(trace_file, category="CE_COLL")
        ce_batch = paths.count_events_in_trace(trace_file, category="CE_BATCH")
        ce_sync = paths.count_events_in_trace(trace_file, category="CE_SYNC")
        total_ce_coll += ce_coll

        assert ce_coll > 0, f"Should have CE_COLL events in {trace_file}"
        assert ce_coll % 2 == 0, \
            f"Each CE_COLL span needs a begin and an end, found {ce_coll} entries in {trace_file}"
        # The forced path hands its CeColl handle to ncclMemOpSync and ncclCeLaunchBatchOps,
        # so every CeColl must bring sync and batch children with it.
        assert ce_sync > 0, \
            f"Should have CE_SYNC events under {ce_coll // 2} CE_COLL spans in {trace_file}"
        assert ce_sync % 2 == 0, \
            f"Each CE_SYNC span needs a begin and an end, found {ce_sync} entries in {trace_file}"
        assert ce_batch > 0, \
            f"Should have CE_BATCH events under {ce_coll // 2} CE_COLL spans in {trace_file}"
        assert ce_batch % 2 == 0, \
            f"Each CE_BATCH span needs a begin and an end, found {ce_batch} entries in {trace_file}"

    assert total_ce_coll > 0, f"CE AllReduce should report CE_COLL events, found none in {dump_dir}"


@pytest.mark.ext_profiler
@pytest.mark.allreduce
def test_invalid_mask_value(paths):
    """Test profiler behavior with invalid event mask (0 = no events)"""
    
    dump_dir = os.path.join(paths.PROFILER_DUMP_DIR, "allreduce_profiler_dumps")
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
        f"{paths.RCCL_TESTS_DIR}/build/all_reduce_perf",
        "-b", "1",
        "-e", "8M",
        "-f", "2",
        "-g", "1",
    ]
    
    log_dir = os.path.join(paths.LOGDIR, "allreduce_ext_profiler_test_logs")
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
    
    assert result.returncode == 0, f"AllReduce test should still succeed even with invalid mask, see {log_file}"
    
    # Verify plugin initialized (it should still initialize)
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
        group_events = paths.count_events_in_trace(trace_file, category="GROUP_API")
        assert group_events == 0, f"Should have no Group API events with mask=0 in {trace_file}, found {group_events}"
        
        allreduce_events = paths.count_events_in_trace(trace_file, event_name="AllReduce")
        assert allreduce_events == 0, f"Should have no AllReduce events with mask=0 in {trace_file}, found {allreduce_events}"


@pytest.mark.ext_profiler
@pytest.mark.allreduce
def test_single_node_detailed_profiling(paths):
    """Test profiler with single-node AllReduce using full event mask (255) across wide message range"""
    
    dump_dir = os.path.join(paths.PROFILER_DUMP_DIR, "allreduce_profiler_dumps")
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
        "--mca", "pml", "ucx",
        "--mca", "btl", "^vader,openib",
        f"{paths.RCCL_TESTS_DIR}/build/all_reduce_perf",
        "-b", "8",        
        "-e", "128M",       
        "-f", "2",        
        "-g", "1",
    ]
    
    log_dir = os.path.join(paths.LOGDIR, "allreduce_ext_profiler_test_logs")
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
    
    assert result.returncode == 0, f"Single-node detailed AllReduce profiling test failed, see {log_file}"
    
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
        
        # Check for Group API events
        group_events = paths.count_events_in_trace(trace_file, category="GROUP_API")
        assert group_events > 0, \
            f"Should have Group API events in {trace_file}, found {group_events}"
        
        # Check for AllReduce events
        allreduce_events = paths.count_events_in_trace(trace_file, event_name="AllReduce")
        assert allreduce_events > 0, \
            f"Should have AllReduce events in {trace_file}, found {allreduce_events}"
        
        # With KernelCh enabled (bit 6), we should see GPU kernel channel events
        kernel_events = paths.count_events_in_trace(trace_file, category="GPU")
        assert kernel_events > 0, \
            f"Should have GPU (KernelCh) events in {trace_file}, found {kernel_events}"
        
        # With ProxyCtrl enabled (bit 5), we should see Append/Sleep events
        proxy_ctrl_events = paths.count_events_in_trace(trace_file, category="PROXY")
        assert proxy_ctrl_events > 0, \
            f"Should have PROXY (ProxyCtrl) events in {trace_file}, found {proxy_ctrl_events}"
        
        append_events = paths.count_events_in_trace(trace_file, event_name="Append")
        sleep_events = paths.count_events_in_trace(trace_file, event_name="Sleep")
        assert append_events > 0 or sleep_events > 0, \
            f"Should have ProxyCtrl events (Append or Sleep) in {trace_file}, found Append={append_events}, Sleep={sleep_events}"
        
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
@pytest.mark.allreduce
@pytest.mark.multinode
def test_multinode_detailed_profiling(paths):
    """Test profiler with multi-node AllReduce operations using full event mask (255)"""
    
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
        pytest.skip(f"Multinode test requires all nodes to have the same network interface (eth0 or eth1).")
    
    # Build host specification string (8 processes per node)
    host_spec = ",".join([f"{node}:8" for node in nodelist])
    total_processes = len(nodelist) * 8
    print(f"Using host specification: {host_spec}")
    
    dump_dir = os.path.join(paths.PROFILER_DUMP_DIR, "allreduce_profiler_dumps")
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
        f"{paths.RCCL_TESTS_DIR}/build/all_reduce_perf",
        "-b", "8",
        "-e", "128M",
        "-f", "2",
        "-g", "1",
    ]
    
    log_dir = os.path.join(paths.LOGDIR, "allreduce_ext_profiler_test_logs")
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
    
    assert result.returncode == 0, f"Multi-node AllReduce profiling test failed, see {log_file}"
    
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
        
        # Check for Group API events (one per AllReduce call)
        group_events = paths.count_events_in_trace(trace_file, category="GROUP_API")
        assert group_events > 0, f"Should have Group API events in {trace_file}, found {group_events}"
        
        # Check for AllReduce events
        allreduce_events = paths.count_events_in_trace(trace_file, event_name="AllReduce")
        assert allreduce_events > 0, \
            f"Should have AllReduce events in {trace_file}, found {allreduce_events}"
        
        # For multi-node tests, verify ProxyOp events exist
        proxy_events = paths.count_events_in_trace(trace_file, category="PROXY")
        assert proxy_events > 0, \
            f"Should have Proxy events in {trace_file}, found {proxy_events}"
        
        # With ProxyOp enabled (bit 3), check for Send and Recv operations
        schedule_send_events = paths.count_events_in_trace(trace_file, event_name="ScheduleSend")
        schedule_recv_events = paths.count_events_in_trace(trace_file, event_name="ScheduleRecv")
        assert schedule_send_events > 0, \
            f"Should have ScheduleSend events in {trace_file}, found {schedule_send_events}"
        assert schedule_recv_events > 0, \
            f"Should have ScheduleRecv events in {trace_file}, found {schedule_recv_events}"
        
        # With ProxyStep enabled (bit 4), verify network step events exist
        net_events = paths.count_events_in_trace(trace_file, category="NET")
        assert net_events > 0, \
            f"Should have NET events in {trace_file}, found {net_events}"
        
        # Check for specific ProxyStep events
        recv_wait_events = paths.count_events_in_trace(trace_file, event_name="RecvWait")
        send_wait_events = paths.count_events_in_trace(trace_file, event_name="SendWait")
        assert recv_wait_events > 0, \
            f"Should have RecvWait events in {trace_file}, found {recv_wait_events}"
        assert send_wait_events > 0, \
            f"Should have SendWait events in {trace_file}, found {send_wait_events}"
        
        # With KernelCh enabled (bit 6), we should see GPU kernel channel events
        kernel_events = paths.count_events_in_trace(trace_file, category="GPU")
        assert kernel_events > 0, \
            f"Should have GPU (KernelCh) events in {trace_file}, found {kernel_events}"
        
        # With ProxyCtrl enabled (bit 5), we should see Append/Sleep events
        append_events = paths.count_events_in_trace(trace_file, event_name="Append")
        sleep_events = paths.count_events_in_trace(trace_file, event_name="Sleep")
        assert append_events > 0 or sleep_events > 0, \
            f"Should have ProxyCtrl events (Append or Sleep) in {trace_file}, found Append={append_events}, Sleep={sleep_events}"
        
        # Verify trace file exists and has content
        trace_file_size = os.path.getsize(trace_file)
        assert trace_file_size > 0, \
            f"Trace file {trace_file} is empty"