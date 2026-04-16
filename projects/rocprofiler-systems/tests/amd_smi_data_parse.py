#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT


import os
import re
import shutil
import subprocess
from enum import Enum

# Mapping from amd-smi output metric names to rocpd database metric names
METRIC_MAPPINGS = {
    # GPU Activity
    "MM_ACTIVITY": "device_busy_mm",
    # Temperature
    "EDGE": "device_temp",
    # Power
    "SOCKET_POWER": "device_power",
    # Memory
    "TOTAL_VRAM": "device_memory_usage",
    # VCN/JPEG activity
    "VCN_ACTIVITY": "device_vcn_activity",
    "JPEG_ACTIVITY": "device_jpeg_activity",
    # XGMI
    "XGMI_LINK_SPEED": "device_xgmi_link_speed",
    "XGMI_LINK_WIDTH": "device_xgmi_link_width",
    "XGMI_READ_DATA": "device_xgmi_read_data",
    "XGMI_WRITE_DATA": "device_xgmi_write_data",
    # PCIe
    "PCIE_BANDWIDTH": "device_pcie_bandwidth_inst",
    "PCIE_BANDWIDTH_ACC": "device_pcie_bandwidth_acc",
    "PCIE_LINK_SPEED": "device_pcie_link_speed",
    "PCIE_LINK_WIDTH": "device_pcie_link_width",
}


# Enum to categorize metric status
class MetricStatus(Enum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"


# =============================================================================
# Utility functions
# =============================================================================


def find_amd_smi_executable():
    """
    Find the amd-smi executable, searching in standard ROCm paths.

    Returns:
        string or None: Path to amd-smi executable if found, None otherwise.
    """
    # Check if it's in PATH
    amd_smi_path = shutil.which("amd-smi")
    if amd_smi_path:
        return amd_smi_path

    # Search in standard ROCm paths
    search_paths = []

    # Check environment variables
    rocm_path_env = os.environ.get("ROCM_PATH")
    if rocm_path_env:
        search_paths.append(rocm_path_env)

    # Add standard ROCm installation paths
    search_paths.extend(
        [
            "/opt/rocm",
            "/opt/rocm/bin",
            "/usr/local/bin",
        ]
    )

    # Check versioned ROCm paths
    if os.path.exists("/opt"):
        for entry in os.listdir("/opt"):
            if entry.startswith("rocm-"):
                search_paths.append(f"/opt/{entry}")

    # Search for amd-smi in each path
    for base_path in search_paths:
        for suffix in ["bin", ""]:
            if suffix:
                amd_smi_path = os.path.join(base_path, suffix, "amd-smi")
            else:
                amd_smi_path = os.path.join(base_path, "amd-smi")

            if os.path.isfile(amd_smi_path) and os.access(amd_smi_path, os.X_OK):
                return amd_smi_path

    return None


def extract_metric_from_query(query):
    """
    Extract the metric name from a validation query string.

    Looks for patterns like:
        - WHERE info.name = 'device_temp'
        - WHERE info.name = "device_power"
        - WHERE info.name LIKE 'device_jpeg_activity_%'

    Args:
        query: SQL query string from validation rule.

    Returns:
        string or None: The metric name (base name without wildcards) if found, None otherwise.
    """
    # Pattern 1: info.name = 'metric_name' or info.name = "metric_name"
    pattern_equals = r"info\.name\s*=\s*['\"]([^'\"]+)['\"]"
    match = re.search(pattern_equals, query, re.IGNORECASE)
    if match:
        return match.group(1)

    # Pattern 2: info.name LIKE 'metric_name%' or info.name LIKE 'metric_name_%'
    pattern_like = r"info\.name\s+LIKE\s+['\"]([^'\"]+)['\"]"
    match = re.search(pattern_like, query, re.IGNORECASE)
    if match:
        metric_name = match.group(1)
        metric_name = re.sub(r"[_%]+$", "", metric_name)
        return metric_name

    return None


# =============================================================================
# AMD-SMI command execution and parsing functions
# =============================================================================


def get_amd_smi_metrics():
    """
    Run amd-smi metric command and return the raw CLI output.

    Returns:
        tuple: (success: bool, amd_smi_cmd_op: string, message: string)
            - success: True if command executed successfully
            - amd_smi_cmd_op: The raw stdout from 'amd-smi metric' command
            - message: Error description if failed, empty string otherwise
    """
    amd_smi_exe = find_amd_smi_executable()
    if amd_smi_exe is None:
        return (
            False,
            "",
            "amd-smi not found in PATH or standard ROCm paths (/opt/rocm, $ROCM_PATH)",
        )

    try:
        result = subprocess.run(
            [amd_smi_exe, "metric"], capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            return (True, result.stdout, "")
        else:
            error_msg = (
                f"amd-smi metric command failed with return code {result.returncode}"
            )
            if result.stderr:
                error_msg += f": {result.stderr}"
            return (False, "", error_msg)
    except subprocess.TimeoutExpired:
        return (False, "", "amd-smi metric command timed out after 30 seconds")
    except Exception as excp:
        return (False, "", f"Error running amd-smi metric: {excp}")


def parse_amd_smi_metrics(amd_smi_cmd_op):
    """
    Parse AMD-SMI metric output and extract supported metric names.

    Args:
        amd_smi_cmd_op: Raw string output from 'amd-smi metric' command.

    Returns:
        dict: Dictionary with Metric status enum keys containing sets of metric names.
    """
    metrics = {MetricStatus.SUPPORTED: set(), MetricStatus.UNSUPPORTED: set()}

    if not amd_smi_cmd_op:
        return metrics

    lines = amd_smi_cmd_op.strip().split("\n")

    for line in lines:
        line = line.strip()

        # Parse metric lines (format: "METRIC_NAME: value")
        metric_match = re.match(r"^([A-Z_0-9]+):\s*(.+)$", line)
        if metric_match:
            metric_name = metric_match.group(1)
            metric_value = metric_match.group(2).strip()

            # Skip N/A metrics - hardware doesn't support them
            if metric_value == "N/A" or metric_value.startswith("[N/A"):
                continue

            if metric_name in METRIC_MAPPINGS:
                db_metric_name = METRIC_MAPPINGS[metric_name]
                metrics[MetricStatus.SUPPORTED].add((metric_name, db_metric_name))
            else:
                metrics[MetricStatus.UNSUPPORTED].add(metric_name)

    # Print unique supported metrics
    if metrics[MetricStatus.SUPPORTED]:
        print(f"Supported metrics : {metrics[MetricStatus.SUPPORTED]}")

    # Print unique unsupported metrics
    if metrics[MetricStatus.UNSUPPORTED]:
        print(f"Unsupported metrics : {metrics[MetricStatus.UNSUPPORTED]}")

    return metrics


# =============================================================================
# API functions
# =============================================================================


def collect_supported_metrics():
    """
    Collect and return metrics dictionary from AMD-SMI.

    1. Finds and runs amd-smi executable
    2. Parses the output to extract supported and unsupported metrics
    3. Returns the metrics dictionary

    Returns:
        dict or None: Dictionary with MetricStatus keys, or None if collection failed.
    """
    # Run amd-smi metric command
    success, amd_smi_cmd_op, error_msg = get_amd_smi_metrics()

    if not success:
        print(f"Failed in get_amd_smi_metrics {error_msg}")
        return None

    # Parse the output
    metrics = parse_amd_smi_metrics(amd_smi_cmd_op)

    if not metrics[MetricStatus.SUPPORTED]:
        print("No supported metrics found in parse_amd_smi_metrics.")
        return None

    return metrics


def is_metric_supported(query, metrics):
    """
    Check if the metric in a query is supported by the hardware.
    1. Extracts the metric name from the SQL query
    2. Checks if that metric exists in the supported metrics set

    Args:
        query: SQL query string from validation rule.
        metrics: Dictionary with MetricStatus keys containing sets of supported and unsupported metric

    Returns:
        tuple: (is_supported: bool, metric_name: string or None)
            - is_supported: True if metric is supported
            - metric_name: The extracted metric name, or None if not found
    """
    if metrics is None:
        print("Metrics dictionary is empty, cannot process further.")
        return (False, None)

    metric_name = extract_metric_from_query(query)

    # Check if metric name in supported metrics
    supported_mapped_names = {mapped for _, mapped in metrics[MetricStatus.SUPPORTED]}
    is_supported = metric_name in supported_mapped_names
    return (is_supported, metric_name)
