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
import json
import glob

WORKDIR = os.getcwd()

RCCL_INSTALL_DIR = "path/to/rccl"
OMPI_INSTALL_DIR = "path/to/ompi/install"
RCCL_TESTS_DIR = "path/to/rccl-tests"

# Ext-Tuner Paths
PLUGIN_DIR = f"{RCCL_INSTALL_DIR}/ext-tuner/example"
PLUGIN_SO = f"{PLUGIN_DIR}/libnccl-tuner-example.so"

# Ext-Profiler Paths
PROFILER_DIR = f"{RCCL_INSTALL_DIR}/ext-profiler/example"
PROFILER_SO = f"{PROFILER_DIR}/librccl-profiler.so"

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

PROFILER_DUMP_DIR = os.path.join(WORKDIR, "profiler_dumps")
os.makedirs(PROFILER_DUMP_DIR, exist_ok=True)

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

def validate_json_trace(trace_file):
    """Validate that a trace file is valid JSON and follows chrome trace format"""
    
    if not os.path.exists(trace_file):
        return False, "File does not exist"
    
    try:
        with open(trace_file, 'r') as f:
            data = json.load(f)
        
        # Basic validation for chrome trace format
        if not isinstance(data, list):
            return False, "Trace must be a JSON array"
        
        # Check for at least one valid event
        if len(data) > 1:  # More than just the closing empty object
            # Validate some events have required fields
            valid_events = 0
            for event in data:
                if isinstance(event, dict) and 'name' in event and 'ph' in event:
                    valid_events += 1
            
            if valid_events == 0:
                return False, "No valid trace events found"
        
        return True, f"Valid JSON trace with {len(data)} entries"
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {e}"
    except Exception as e:
        return False, f"Error reading file: {e}"

def check_event_in_log(log_file, event_string):
    """Check if a specific event string appears in the log file"""
    if not os.path.exists(log_file):
        return False
    
    with open(log_file, 'r') as f:
        content = f.read()
        return event_string in content

def count_events_in_trace(trace_file, event_name=None, category=None):
    """Count events in a trace file, optionally filtered by name or category"""
    
    if not os.path.exists(trace_file):
        return 0
    
    try:
        with open(trace_file, 'r') as f:
            data = json.load(f)
        
        count = 0
        for event in data:
            if not isinstance(event, dict):
                continue
            
            match = True
            if event_name and event.get('name') != event_name:
                match = False
            if category and event.get('cat') != category:
                match = False
            
            if match and 'name' in event:  # Valid event
                count += 1
        
        return count
    except:
        return 0

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
        PROFILER_DIR=PROFILER_DIR,
        PROFILER_SO=PROFILER_SO,
        # CSV Configs
        VALID_CONFIG_WITH_WILDCARDS=VALID_CONFIG_WITH_WILDCARDS,
        VALID_CONFIG_WITHOUT_WILDCARDS=VALID_CONFIG_WITHOUT_WILDCARDS,
        NO_MATCHING_CONFIG=NO_MATCHING_CONFIG,
        INCORRECT_VALUES_CONFIG=INCORRECT_VALUES_CONFIG,
        UNSUPPORTED_ALGO_PROTO_CONFIG=UNSUPPORTED_ALGO_PROTO_CONFIG,
        SINGLENODE_CONFIG=SINGLENODE_CONFIG,
        MULTINODE_CONFIG=MULTINODE_CONFIG,
        LOGDIR=LOGDIR,
        PROFILER_DUMP_DIR=PROFILER_DUMP_DIR,
        # Helper Functions for Ext-Tuner
        get_avg_bus_bandwidth=get_avg_bus_bandwidth,
        check_node_interface=check_node_interface,
        find_common_interface=find_common_interface,
        get_available_nodes=get_available_nodes,
        # Helper Functions for Ext-Profiler
        validate_json_trace=validate_json_trace,
        check_event_in_log=check_event_in_log,
        count_events_in_trace=count_events_in_trace,
    )

def pytest_runtest_setup(item):
    """Check plugin availability before running each test"""
    # Check for ext_tuner marker
    if item.get_closest_marker("ext_tuner"):
        if not os.path.exists(PLUGIN_SO):
            pytest.skip(f"Tuner plugin library not found at: {PLUGIN_SO}")
    
    # Check for ext_profiler marker
    if item.get_closest_marker("ext_profiler"):
        if not os.path.exists(PROFILER_SO):
            pytest.skip(f"Profiler plugin library not found at: {PROFILER_SO}")

@pytest.fixture(scope="session", autouse=True)
def clear_profiler_dump(request):
    """Automatically clear profiler dump folder once before ext_profiler tests"""
    # Check if any test in the session has ext_profiler marker
    has_profiler_tests = any(
        item.get_closest_marker("ext_profiler") 
        for item in request.session.items
    )
    
    if has_profiler_tests:
        # Clear all JSON files in the profiler dump directory (including subdirectories)
        pattern = os.path.join(PROFILER_DUMP_DIR, "**", "*.json")
        for trace_file in glob.glob(pattern, recursive=True):
            try:
                os.remove(trace_file)
            except OSError:
                pass  # Ignore errors if file doesn't exist or can't be removed
