# *************************************************************************
#  * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#  *
#  * See LICENSE.txt for license information
#  ************************************************************************

import os
import re
import subprocess
import pytest


@pytest.mark.ext_profiler
@pytest.mark.allreduce
@pytest.mark.multinode
def test_multinode_dump_is_complete(paths):
    """The proxy-trace dump reports every operation on its own line.

    The dump is written through the debug logger, which formats into a fixed 1KB buffer,
    so emitting it as one message truncates all but the first record. Proxy operations
    only exist for inter-node traffic, hence the multi-node requirement.
    """

    plugin_so = os.path.join(paths.RCCL_INSTALL_DIR, "plugins", "profiler", "proxytrace",
                             "librccl-profiler-proxytrace.so")
    if not os.path.exists(plugin_so):
        pytest.skip(f"proxytrace plugin is not built at {plugin_so}")

    nodelist = paths.get_available_nodes()
    if len(nodelist) == 0:
        pytest.skip("No nodes available")
    elif len(nodelist) < 2:
        pytest.skip(f"Multinode test requires at least 2 nodes, but only {len(nodelist)} available: {nodelist}")

    common_interface = paths.find_common_interface(nodelist)
    if common_interface is None:
        pytest.skip("Multinode test requires all nodes to have the same network interface (eth0 or eth1).")

    host_spec = ",".join([f"{node}:8" for node in nodelist])
    total_processes = len(nodelist) * 8

    env = os.environ.copy()
    env.update({
        "PATH": f"{paths.OMPI_INSTALL_DIR}/bin:{env.get('PATH', '')}",
        "LD_LIBRARY_PATH": f"{paths.RCCL_INSTALL_DIR}:{paths.OMPI_INSTALL_DIR}/lib:{env.get('LD_LIBRARY_PATH', '')}",
        "HSA_NO_SCRATCH_RECLAIM": "1",
        "NCCL_IGNORE_CPU_AFFINITY": "1",
        "NCCL_PROFILER_PLUGIN": plugin_so,
        "NCCL_SOCKET_IFNAME": common_interface,
        "NCCL_DMABUF_ENABLE": "1",
        "RCCL_DDA_ENABLE": "0",
        # The dump goes out at WARN through the debug logger, so debugging must be on.
        "NCCL_DEBUG": "WARN",
    })

    args = [
        f"{paths.OMPI_INSTALL_DIR}/bin/mpirun", "-np", f"{total_processes}",
        "--host", host_spec,
        "--mca", "pml", "ucx",
        "--mca", "btl", "^vader,openib",
        f"{paths.RCCL_TESTS_DIR}/build/all_reduce_perf",
        "-b", "1M",
        "-e", "8M",
        "-f", "2",
        "-g", "1",
    ]

    log_dir = os.path.join(paths.LOGDIR, "proxytrace_ext_profiler_test_logs")
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, "multinode_dump.log")
    with open(log_file, "w") as logfile:
        result = subprocess.run(
            args,
            env=env,
            stdout=logfile,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )

    assert result.returncode == 0, f"Multi-node proxytrace test failed, see {log_file}"

    with open(log_file, 'r') as f:
        log_content = f.read()

    assert "ncclProfilerProxyTraceDump" in log_content, \
        f"Proxy trace should have been dumped at communicator destroy, see {log_file}"

    header_lines = log_content.count("commDump for all active ops")
    assert header_lines > 0, f"Dump should carry its summary header, see {log_file}"

    # Each record names the channel and status of one proxy operation.
    record_re = re.compile(r"createT:\d+.*?chan:\d+.*?status:")
    op_lines = len(record_re.findall(log_content))
    assert op_lines > 0, \
        f"Dump should report proxy operations, found none in {log_file}"

    # A single logger call truncates at the 1KB buffer but still leaves the header and the
    # first few records intact, so a record count alone cannot tell the two apart. Counting
    # logger prefixes can: emitting one call per record stamps every record with its own
    # prefix, so prefixes outnumber records, while sending the whole dump as one message
    # produces a single prefix covering many records. Ranks write to a shared stream and
    # their lines interleave, so this counts over the whole file rather than per line.
    prefixes = log_content.count("ncclProfilerProxyTraceDump")
    assert prefixes > op_lines, \
        (f"{log_file} holds {op_lines} proxy-op records under only {prefixes} logger calls, "
         f"so records are sharing one message -- the dump is being emitted whole and truncated "
         f"rather than one line per operation")
