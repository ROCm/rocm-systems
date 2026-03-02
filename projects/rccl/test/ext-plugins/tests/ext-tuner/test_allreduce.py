# *************************************************************************
#  * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
#  *
#  * See LICENSE.txt for license information
#  ************************************************************************

import os
import subprocess
import pytest

@pytest.mark.ext_tuner
@pytest.mark.allreduce
def test_valid_config_with_wildcards(paths):
    """Test CSV plugin with wildcard values for matching configurations"""
    
    env = os.environ.copy()
    env.update({
        "PATH": f"{paths.OMPI_INSTALL_DIR}/bin:{env.get('PATH', '')}",
        "LD_LIBRARY_PATH": f"{paths.RCCL_INSTALL_DIR}:{paths.OMPI_INSTALL_DIR}/lib:{paths.PLUGIN_DIR}:{env.get('LD_LIBRARY_PATH', '')}",
        "HSA_NO_SCRATCH_RECLAIM": "1",
        "NCCL_TUNER_PLUGIN": paths.PLUGIN_SO,
        "NCCL_TUNER_CONFIG_FILE": paths.VALID_CONFIG_WITH_WILDCARDS,
        "NCCL_DEBUG": "INFO",
        "NCCL_DEBUG_SUBSYS": "TUNING",
    })

    args = [
        f"{paths.OMPI_INSTALL_DIR}/bin/mpirun", "-np", "4",
        "--mca", "pml", "ucx",
        "--mca", "btl", "^vader,openib",
        f"{paths.RCCL_TESTS_DIR}/build/all_reduce_perf",
        "-b", "8",
        "-e", "128M",
        "-f", "2",
        "-g", "1",
    ]

    allreduce_log_dir = os.path.join(paths.LOGDIR, "allreduce_csv_plugin_test_logs")
    os.makedirs(allreduce_log_dir, exist_ok=True)

    log_file = os.path.join(allreduce_log_dir, "test_allreduce_valid_config_with_wildcards.log")
    with open(log_file, "w") as logfile:
        rccl_test = subprocess.run(
            args,
            env=env,
            stdout=logfile,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )

    assert rccl_test.returncode == 0, f"CSV Plugin test failed, see {log_file}"
    
    # Read and validate log content
    with open(log_file, "r") as logfile:
        log_content = logfile.read()
    
    # Check that plugin loaded configurations
    assert "TUNER/ExamplePlugin: Loaded" in log_content and "tuning configurations" in log_content, \
        f"Plugin should have loaded configurations from {paths.VALID_CONFIG_WITH_WILDCARDS}"
    
    # Check that plugin applied valid configurations (test fails if no configs applied)
    plugin_applied = "TUNER/ExamplePlugin: Applied config for collType=" in log_content
    
    assert plugin_applied, \
        f"Plugin should have applied valid configurations, but none were applied. Check {log_file} for details"
    
@pytest.mark.ext_tuner
@pytest.mark.allreduce
def test_valid_config_without_wildcards(paths):
    """Test CSV plugin with specific values (no wildcards -1) for precise matching"""

    env = os.environ.copy()
    env.update({
        "PATH": f"{paths.OMPI_INSTALL_DIR}/bin:{env.get('PATH', '')}",
        "LD_LIBRARY_PATH": f"{paths.RCCL_INSTALL_DIR}:{paths.OMPI_INSTALL_DIR}/lib:{paths.PLUGIN_DIR}:{env.get('LD_LIBRARY_PATH', '')}",
        "HSA_NO_SCRATCH_RECLAIM": "1",
        "NCCL_TUNER_PLUGIN": paths.PLUGIN_SO,
        "NCCL_TUNER_CONFIG_FILE": paths.VALID_CONFIG_WITHOUT_WILDCARDS,
        "NCCL_DEBUG": "INFO",
        "NCCL_DEBUG_SUBSYS": "TUNING",
    })

    args = [
        f"{paths.OMPI_INSTALL_DIR}/bin/mpirun", "-np", "4",
        "--mca", "pml", "ucx",
        "--mca", "btl", "^vader,openib",
        f"{paths.RCCL_TESTS_DIR}/build/all_reduce_perf",
        "-b", "8",
        "-e", "128M",
        "-f", "2",
        "-g", "1",
    ]

    allreduce_log_dir = os.path.join(paths.LOGDIR, "allreduce_csv_plugin_test_logs")
    os.makedirs(allreduce_log_dir, exist_ok=True)

    log_file = os.path.join(allreduce_log_dir, "test_allreduce_valid_config_without_wildcards.log")
    with open(log_file, "w") as logfile:
        rccl_test = subprocess.run(
            args,
            env=env,
            stdout=logfile,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )

    assert rccl_test.returncode == 0, f"CSV Plugin test failed, see {log_file}"
    
    # Read and validate log content
    with open(log_file, "r") as logfile:
        log_content = logfile.read()
    
    # Check that plugin loaded configurations
    assert "TUNER/ExamplePlugin: Loaded" in log_content and "tuning configurations" in log_content, \
        f"Plugin should have loaded configurations from {paths.VALID_CONFIG_WITHOUT_WILDCARDS}"
    
    # With specific values, plugin should either apply matching configs or report no matches
    plugin_applied = "TUNER/ExamplePlugin: Applied config for collType=" in log_content
    
    # Test should fail if no config is applied - we expect specific configs to match
    assert plugin_applied, \
        f"Plugin should have applied at least one configuration from {paths.VALID_CONFIG_WITHOUT_WILDCARDS}. Check {log_file} for details"

@pytest.mark.ext_tuner
@pytest.mark.allreduce
def test_no_matching_config(paths):
    """Test CSV plugin behavior with no matching configurations"""

    env = os.environ.copy()
    env.update({
        "PATH": f"{paths.OMPI_INSTALL_DIR}/bin:{env.get('PATH', '')}",
        "LD_LIBRARY_PATH": f"{paths.RCCL_INSTALL_DIR}:{paths.OMPI_INSTALL_DIR}/lib:{paths.PLUGIN_DIR}:{env.get('LD_LIBRARY_PATH', '')}",
        "HSA_NO_SCRATCH_RECLAIM": "1",
        "NCCL_TUNER_PLUGIN": paths.PLUGIN_SO,
        "NCCL_TUNER_CONFIG_FILE": paths.NO_MATCHING_CONFIG,
        "NCCL_DEBUG": "INFO",
        "NCCL_DEBUG_SUBSYS": "TUNING",
    })

    args = [
        f"{paths.OMPI_INSTALL_DIR}/bin/mpirun", "-np", "8",
        "--bind-to", "none",
        "--mca", "pml", "ucx",
        "--mca", "btl", "^vader,openib",
        f"{paths.RCCL_TESTS_DIR}/build/all_reduce_perf",
        "-b", "8",
        "-e", "128M",
        "-f", "2",
        "-g", "1",
    ]

    allreduce_log_dir = os.path.join(paths.LOGDIR, "allreduce_csv_plugin_test_logs")
    os.makedirs(allreduce_log_dir, exist_ok=True)

    log_file = os.path.join(allreduce_log_dir, "test_allreduce_no_matching_config.log")
    with open(log_file, "w") as logfile:
        rccl_test = subprocess.run(
            args,
            env=env,
            stdout=logfile,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )

    assert rccl_test.returncode == 0, f"CSV Plugin test failed, see {log_file}"
    
    # Read and validate log content
    with open(log_file, "r") as logfile:
        log_content = logfile.read()
    
    # Check that plugin loaded configurations
    assert "TUNER/ExamplePlugin: Loaded" in log_content and "tuning configurations" in log_content, \
        f"Plugin should have loaded configurations from {paths.NO_MATCHING_CONFIG}"
    
    # Check that NO configurations were applied (they should not match the test environment)
    plugin_applied = "TUNER/ExamplePlugin: Applied config for collType=" in log_content
    
    assert not plugin_applied, \
        f"Plugin should NOT have applied any configurations from {paths.NO_MATCHING_CONFIG} as they don't match the test environment. Check {log_file} for details"

@pytest.mark.ext_tuner
@pytest.mark.allreduce
def test_incorrect_values_config(paths):
    """Test CSV plugin behavior with invalid/incorrect values in configuration"""

    env = os.environ.copy()
    env.update({
        "PATH": f"{paths.OMPI_INSTALL_DIR}/bin:{env.get('PATH', '')}",
        "LD_LIBRARY_PATH": f"{paths.RCCL_INSTALL_DIR}:{paths.OMPI_INSTALL_DIR}/lib:{paths.PLUGIN_DIR}:{env.get('LD_LIBRARY_PATH', '')}",
        "HSA_NO_SCRATCH_RECLAIM": "1",
        "NCCL_TUNER_PLUGIN": paths.PLUGIN_SO,
        "NCCL_TUNER_CONFIG_FILE": paths.INCORRECT_VALUES_CONFIG,
        "NCCL_DEBUG": "INFO",
        "NCCL_DEBUG_SUBSYS": "TUNING",
    })

    args = [
        f"{paths.OMPI_INSTALL_DIR}/bin/mpirun", "-np", "4",
        "--mca", "pml", "ucx",
        "--mca", "btl", "^vader,openib",
        f"{paths.RCCL_TESTS_DIR}/build/all_reduce_perf",
        "-b", "8",
        "-e", "128M",
        "-f", "2",
        "-g", "1",
    ]

    allreduce_log_dir = os.path.join(paths.LOGDIR, "allreduce_csv_plugin_test_logs")
    os.makedirs(allreduce_log_dir, exist_ok=True)

    log_file = os.path.join(allreduce_log_dir, "test_allreduce_incorrect_values_config.log")
    with open(log_file, "w") as logfile:
        rccl_test = subprocess.run(
            args,
            env=env,
            stdout=logfile,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )

    assert rccl_test.returncode == 0, f"CSV Plugin test failed, see {log_file}"
    
    # Read and validate log content
    with open(log_file, "r") as logfile:
        log_content = logfile.read()
    
    # Check that plugin loaded some configurations (plugin should handle invalid values gracefully)
    assert "TUNER/ExamplePlugin: Loaded" in log_content and "tuning configurations" in log_content, \
        f"Plugin should have loaded configurations from {paths.INCORRECT_VALUES_CONFIG}"
    
    # Plugin should still function despite invalid values (using defaults)
    # It might apply configs with default values or report no matches
    plugin_applied = "TUNER/ExamplePlugin: Applied config for collType=" in log_content
    
    assert plugin_applied, \
        "Plugin should either apply configurations (with defaults) or report no matches"

@pytest.mark.ext_tuner
@pytest.mark.allreduce
def test_unsupported_algo_proto_config(paths):
    """Test that plugin handles unsupported algorithm/protocol combinations"""

    env = os.environ.copy()
    env.update({
        "PATH": f"{paths.OMPI_INSTALL_DIR}/bin:{env.get('PATH', '')}",
        "LD_LIBRARY_PATH": f"{paths.RCCL_INSTALL_DIR}:{paths.OMPI_INSTALL_DIR}/lib:{paths.PLUGIN_DIR}:{env.get('LD_LIBRARY_PATH', '')}",
        "HSA_NO_SCRATCH_RECLAIM": "1",
        "NCCL_TUNER_PLUGIN": paths.PLUGIN_SO,
        "NCCL_TUNER_CONFIG_FILE": paths.UNSUPPORTED_ALGO_PROTO_CONFIG,
        "NCCL_DEBUG": "INFO",
        "NCCL_DEBUG_SUBSYS": "TUNING",
    })

    args = [
        f"{paths.OMPI_INSTALL_DIR}/bin/mpirun", "-np", "4",
        "--mca", "pml", "ucx",
        "--mca", "btl", "^vader,openib",
        f"{paths.RCCL_TESTS_DIR}/build/all_reduce_perf",
        "-b", "8",
        "-e", "128M",
        "-f", "2",
        "-g", "1",
    ]

    allreduce_log_dir = os.path.join(paths.LOGDIR, "allreduce_csv_plugin_test_logs")
    os.makedirs(allreduce_log_dir, exist_ok=True)

    log_file = os.path.join(allreduce_log_dir, "test_allreduce_unsupported_algo_proto.log")
    with open(log_file, "w") as logfile:
        rccl_test = subprocess.run(
            args,
            env=env,
            stdout=logfile,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )

    assert rccl_test.returncode == 0, f"CSV Plugin test failed, see {log_file}"
    
    # Read and validate log content
    with open(log_file, "r") as logfile:
        log_content = logfile.read()
    
    # Check that plugin loaded configurations
    assert "TUNER/ExamplePlugin: Loaded" in log_content and "tuning configurations" in log_content, \
        f"Plugin should have loaded configurations from {paths.UNSUPPORTED_ALGO_PROTO_CONFIG}"
    
    # Check for unsupported combinations - should see IGNORE or out of bounds messages
    ignored_combinations = "Algorithm/protocol combination" in log_content and "is marked as IGNORE" in log_content
    out_of_bounds = "out of bounds" in log_content
    
    assert ignored_combinations or out_of_bounds, \
        f"Plugin should report unsupported algorithm/protocol combinations as IGNORE or out of bounds. Check {log_file} for details"

@pytest.mark.ext_tuner
@pytest.mark.allreduce
def test_singlenode_config(paths):
    """Test CSV plugin with single-node configuration"""

    env = os.environ.copy()
    env.update({
        "PATH": f"{paths.OMPI_INSTALL_DIR}/bin:{env.get('PATH', '')}",
        "LD_LIBRARY_PATH": f"{paths.RCCL_INSTALL_DIR}:{paths.OMPI_INSTALL_DIR}/lib:{paths.PLUGIN_DIR}:{env.get('LD_LIBRARY_PATH', '')}",
        "HSA_NO_SCRATCH_RECLAIM": "1",
        "NCCL_TUNER_PLUGIN": paths.PLUGIN_SO,
        "NCCL_TUNER_CONFIG_FILE": paths.SINGLENODE_CONFIG,
        "NCCL_DEBUG": "INFO",
        "NCCL_DEBUG_SUBSYS": "TUNING",
    })

    args = [
        f"{paths.OMPI_INSTALL_DIR}/bin/mpirun", "-np", "8",
        "--bind-to", "none",
        "--mca", "pml", "ucx",
        "--mca", "btl", "^vader,openib",
        f"{paths.RCCL_TESTS_DIR}/build/all_reduce_perf",
        "-b", "8",
        "-e", "128M",
        "-f", "2",
        "-g", "1",
    ]

    allreduce_log_dir = os.path.join(paths.LOGDIR, "allreduce_csv_plugin_test_logs")
    os.makedirs(allreduce_log_dir, exist_ok=True)

    log_file = os.path.join(allreduce_log_dir, "test_allreduce_singlenode.log")
    with open(log_file, "w") as logfile:
        rccl_test = subprocess.run(
            args,
            env=env,
            stdout=logfile,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )

    assert rccl_test.returncode == 0, f"Single-node CSV Plugin test failed, see {log_file}"

    # Read and validate log content
    with open(log_file, "r") as logfile:
        log_content = logfile.read()
    
    # Check that plugin loaded configurations
    assert "TUNER/ExamplePlugin: Loaded" in log_content and "tuning configurations" in log_content, \
        f"Plugin should have loaded configurations from {paths.SINGLENODE_CONFIG}"

    # Check that configurations were applied for single-node setup
    plugin_applied = "TUNER/ExamplePlugin: Applied config for collType=" in log_content
    
    assert plugin_applied, \
        f"Plugin should have applied single-node configurations from {paths.SINGLENODE_CONFIG}. Check {log_file} for details"

@pytest.mark.ext_tuner
@pytest.mark.allreduce
@pytest.mark.multinode
def test_multinode_config(paths):
    """Test CSV plugin with multi-node configuration"""

    # Get available nodes using the shared function
    nodelist = paths.get_available_nodes()
    
    # Skip test if no nodes available (SLURM not available) or less than 2 nodes
    if len(nodelist) == 0:
        pytest.skip("No nodes available")
    elif len(nodelist) < 2:
        pytest.skip(f"Multinode allreduce test requires at least 2 nodes, but only {len(nodelist)} available: {nodelist}")

    # Check for common network interface across all nodes
    common_interface = paths.find_common_interface(nodelist)
    if common_interface is None:
        pytest.skip(f"Multinode allreduce test requires all nodes to have the same network interface (eth0 or eth1).")
    
    # Build host specification string (4 processes per node)
    host_spec = ",".join([f"{node}:8" for node in nodelist])
    total_processes = len(nodelist) * 8
    print(f"Using host specification: {host_spec}")

    env = os.environ.copy()
    env.update({
        "PATH": f"{paths.OMPI_INSTALL_DIR}/bin:{env.get('PATH', '')}",
        "LD_LIBRARY_PATH": f"{paths.RCCL_INSTALL_DIR}:{paths.OMPI_INSTALL_DIR}/lib:{paths.PLUGIN_DIR}:{env.get('LD_LIBRARY_PATH', '')}",
        "HSA_NO_SCRATCH_RECLAIM": "1",
        "NCCL_IGNORE_CPU_AFFINITY": "1",
        "NCCL_TUNER_PLUGIN": paths.PLUGIN_SO,
        "NCCL_TUNER_CONFIG_FILE": paths.MULTINODE_CONFIG,
        "NCCL_DEBUG": "INFO",
        "NCCL_DEBUG_SUBSYS": "TUNING",
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

    allreduce_log_dir = os.path.join(paths.LOGDIR, "allreduce_csv_plugin_test_logs")
    os.makedirs(allreduce_log_dir, exist_ok=True)

    log_file = os.path.join(allreduce_log_dir, "test_allreduce_multinode.log")
    with open(log_file, "w") as logfile:
        rccl_test = subprocess.run(
            args,
            env=env,
            stdout=logfile,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )

    assert rccl_test.returncode == 0, f"Multi-node CSV Plugin test failed, see {log_file}"
    
    # Read and validate log content
    with open(log_file, "r") as logfile:
        log_content = logfile.read()
    
    # Check that plugin loaded configurations
    assert "TUNER/ExamplePlugin: Loaded" in log_content and "tuning configurations" in log_content, \
        f"Plugin should have loaded configurations from {paths.MULTINODE_CONFIG}"
    
    # Check that configurations were applied for multi-node setup
    plugin_applied = "TUNER/ExamplePlugin: Applied config for collType=" in log_content
    
    assert plugin_applied, \
        f"Plugin should have applied multi-node configurations from {paths.MULTINODE_CONFIG}. Check {log_file} for details"

