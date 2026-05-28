# *************************************************************************
#  * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
#  *
#  * See LICENSE.txt for license information
#  ************************************************************************

import os
import pytest
import subprocess
import re
from types import SimpleNamespace

WORKDIR = os.getcwd()

RCCL_INSTALL_DIR = "path/to/rccl"
OMPI_INSTALL_DIR = "path/to/ompi/install"
RCCL_TESTS_DIR = "path/to/rccl-tests"

# Plugin Paths
PLUGIN_DIR = f"{RCCL_INSTALL_DIR}/ext-tuner/example"
PLUGIN_SO = f"{PLUGIN_DIR}/libnccl-tuner-example.so"

# CSV Configs 
VALID_CONFIG_WITH_WILDCARDS = os.path.join(WORKDIR, "assets/csv_confs/valid_config_with_wildcards.conf")
VALID_CONFIG_WITHOUT_WILDCARDS = os.path.join(WORKDIR, "assets/csv_confs/valid_config_without_wildcards.conf")
NO_MATCHING_CONFIG = os.path.join(WORKDIR, "assets/csv_confs/no_matching_config.conf")
INCORRECT_VALUES_CONFIG = os.path.join(WORKDIR, "assets/csv_confs/incorrect_values_config.conf")
UNSUPPORTED_ALGO_PROTO_CONFIG = os.path.join(WORKDIR, "assets/csv_confs/unsupported_algo_proto_config.conf")
SINGLENODE_CONFIG = os.path.join(WORKDIR, "assets/csv_confs/singlenode_config.conf")
MULTINODE_CONFIG = os.path.join(WORKDIR, "assets/csv_confs/multinode_config.conf")

LOGDIR = os.path.join(WORKDIR, "logs")
os.makedirs(LOGDIR, exist_ok=True)

# Helper Functions
def get_avg_bus_bandwidth(log_content: str):
    """Extract average bus bandwidth from RCCL test log"""
    pattern = r'#\s*Avg bus bandwidth\s*:\s*([\d.]+)'
    match = re.search(pattern, log_content, re.IGNORECASE)
    return float(match.group(1)) if match else None

def check_node_interface(node: str, interface: str) -> bool:
    """Check if a node has the specified interface with an IP address"""
    try:
        cmd = ["ssh", "-o", "ConnectTimeout=5", "-o", "StrictHostKeyChecking=no",
               node, f"ip addr show {interface} | grep 'inet ' | wc -l"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return result.returncode == 0 and int(result.stdout.strip()) > 0
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError):
        return False

def find_common_interface(nodelist):
    """Find a common network interface across all nodes"""
    interfaces_to_check = ["eth0", "eth1"]

    for interface in interfaces_to_check:
        all_nodes_have_interface = True
        for node in nodelist:
            if not check_node_interface(node, interface):
                all_nodes_have_interface = False
                break
        if all_nodes_have_interface:
            return interface
    return None

def get_available_nodes():
    """Get available nodes from SLURM environment"""
    try:
        # Get available nodes
        result = subprocess.run(
            ["scontrol", "show", "hostnames"], 
            capture_output=True, 
            text=True, 
            check=True
        )
        nodelist = result.stdout.strip().split('\n')
        nodelist = [node.strip() for node in nodelist if node.strip()]
        
        return nodelist
        
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []

# Pytest Fixture
@pytest.fixture(scope="session")
def paths():
    return SimpleNamespace(
        # Paths
        WORKDIR=WORKDIR,
        RCCL_INSTALL_DIR=RCCL_INSTALL_DIR,
        OMPI_INSTALL_DIR=OMPI_INSTALL_DIR,
        PLUGIN_DIR=PLUGIN_DIR,
        PLUGIN_SO=PLUGIN_SO,
        RCCL_TESTS_DIR=RCCL_TESTS_DIR,
        # CSV Configs
        VALID_CONFIG_WITH_WILDCARDS=VALID_CONFIG_WITH_WILDCARDS,
        VALID_CONFIG_WITHOUT_WILDCARDS=VALID_CONFIG_WITHOUT_WILDCARDS,
        NO_MATCHING_CONFIG=NO_MATCHING_CONFIG,
        INCORRECT_VALUES_CONFIG=INCORRECT_VALUES_CONFIG,
        UNSUPPORTED_ALGO_PROTO_CONFIG=UNSUPPORTED_ALGO_PROTO_CONFIG,
        SINGLENODE_CONFIG=SINGLENODE_CONFIG,
        MULTINODE_CONFIG=MULTINODE_CONFIG,
        LOGDIR=LOGDIR,
        # Helper Functions
        get_avg_bus_bandwidth=get_avg_bus_bandwidth,
        check_node_interface=check_node_interface,
        find_common_interface=find_common_interface,
        get_available_nodes=get_available_nodes,
    )