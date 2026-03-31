#!/usr/bin/env python3

"""
AMD SMI XCP Partition Metrics Example

This example demonstrates how to:
1. Access XCP (Graphics Cluster Partition) metrics
2. Monitor partition utilization and throttling
3. Display partition-specific performance data
4. Handle XCP-enabled vs non-XCP systems

XCP (Graphics Cluster Partitioning) allows dividing GPU resources into
multiple partitions for multi-tenant workloads and resource isolation.

Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""

import sys
import os
import time

# Add the built Python package to the path
build_path = os.path.join(
    os.path.dirname(__file__), "..", "..", "build", "py-interface", "python_package"
)
if os.path.exists(build_path):
    sys.path.insert(0, build_path)

try:
    # Try importing from built package first
    import amdsmi
except ImportError:
    # Fall back to development imports
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    try:
        from amdsmi_interface import amdsmi_init, amdsmi_shut_down, amdsmi_get_processor_handles
        from amdsmi_interface import amdsmi_get_gpu_partition_metrics_info
        from amdsmi_exception import AmdSmiException

        # Create a fake amdsmi module-like object for compatibility
        class AmdsmiCompat:
            amdsmi_init = amdsmi_init
            amdsmi_shut_down = amdsmi_shut_down
            amdsmi_get_processor_handles = amdsmi_get_processor_handles
            amdsmi_get_gpu_partition_metrics_info = amdsmi_get_gpu_partition_metrics_info
            AmdSmiException = AmdSmiException

        amdsmi = AmdsmiCompat()
    except ImportError as e:
        print(
            f"Error: AMD SMI library not found. Please ensure amdsmi is installed or built. Details: {e}"
        )
        sys.exit(1)


def print_partition_metrics_info(gpu_handle, gpu_id):
    """Print XCP partition metrics information for a specific GPU."""

    print(f"\n=== GPU {gpu_id} XCP Partition Metrics ===")

    try:
        # Get partition-specific metrics
        partition_metrics = amdsmi.amdsmi_get_gpu_partition_metrics_info(gpu_handle)

        print(f"Raw partition metrics available: {len(partition_metrics)} fields")

        # Core utilization metrics
        print(f"\nUtilization Metrics:")
        gfx_util = partition_metrics.get("current_gfxclk_utilization", "N/A")
        umc_util = partition_metrics.get("current_uclk_utilization", "N/A")
        mm_activity = partition_metrics.get("average_mm_activity", "N/A")

        if gfx_util != "N/A":
            print(f"  Current GFXCLK Utilization: {gfx_util}%")
        if umc_util != "N/A":
            print(f"  Current UCLK Utilization: {umc_util}%")
        if mm_activity != "N/A":
            print(f"  Average MM Activity: {mm_activity}%")

        # Clock frequencies
        print(f"\nClock Frequencies:")
        current_gfxclk = partition_metrics.get("current_gfxclk", "N/A")
        current_uclk = partition_metrics.get("current_uclk", "N/A")
        current_socclk = partition_metrics.get("current_socclk", "N/A")

        if current_gfxclk != "N/A" and isinstance(current_gfxclk, (int, float)):
            gfxclk_mhz = current_gfxclk / 1000000  # Convert Hz to MHz
            print(f"  Current GFXCLK: {gfxclk_mhz:.0f} MHz")
        else:
            print(f"  Current GFXCLK: {current_gfxclk}")
        if current_uclk != "N/A" and isinstance(current_uclk, (int, float)):
            uclk_mhz = current_uclk / 1000000
            print(f"  Current UCLK: {uclk_mhz:.0f} MHz")
        else:
            print(f"  Current UCLK: {current_uclk}")
        if current_socclk != "N/A" and isinstance(current_socclk, (int, float)):
            socclk_mhz = current_socclk / 1000000
            print(f"  Current SOCCLK: {socclk_mhz:.0f} MHz")
        else:
            print(f"  Current SOCCLK: {current_socclk}")

        # Throttling information
        print(f"\nThrottling Status:")
        throttle_mask = partition_metrics.get("throttle_status_bitmask", 0)

        if throttle_mask == 0:
            print(f"  Status: No throttling active")
        else:
            print(f"  Status: Throttling detected (bitmask: 0x{throttle_mask:x})")

            # Decode common throttling reasons (bit positions may vary by GPU)
            throttle_reasons = []
            if throttle_mask & 0x1:
                throttle_reasons.append("Power Limit")
            if throttle_mask & 0x2:
                throttle_reasons.append("Thermal Limit")
            if throttle_mask & 0x4:
                throttle_reasons.append("Current Limit")
            if throttle_mask & 0x8:
                throttle_reasons.append("Voltage Limit")

            if throttle_reasons:
                print(f"  Possible Reasons: {', '.join(throttle_reasons)}")

        # Partition-specific information
        print(f"\nPartition Information:")
        num_partitions = partition_metrics.get("num_partition", "N/A")
        if num_partitions != "N/A":
            print(f"  Number of Partitions: {num_partitions}")

        # Per-partition utilization (if available)
        partition_keys = [
            key for key in partition_metrics.keys() if "per_" in key and "utilization" in key
        ]
        if partition_keys:
            print(f"  Per-Partition Utilization:")
            for key in partition_keys[:5]:  # Show first 5 to avoid clutter
                value = partition_metrics[key]
                if isinstance(value, (list, tuple)):
                    print(f"    {key}: {value}")
                else:
                    print(f"    {key}: {value}")

        # Additional metrics of interest
        print(f"\nAdditional Metrics:")
        for key, value in partition_metrics.items():
            if key not in [
                "current_gfxclk_utilization",
                "current_uclk_utilization",
                "average_mm_activity",
                "current_gfxclk",
                "current_uclk",
                "current_socclk",
                "throttle_status_bitmask",
                "num_partition",
            ] and not key.startswith("per_"):
                print(f"  {key}: {value}")

    except amdsmi.AmdSmiException as e:
        if "not supported" in str(e) or "not available" in str(e):
            print(f"  XCP partition metrics not supported on this GPU")
            print(f"  This may be because:")
            print(f"    - GPU does not support XCP partitioning")
            print(f"    - XCP is not enabled on this system")
            print(f"    - Driver version is too old")
        else:
            print(f"  Error getting partition metrics: {e}")


def monitor_partition_metrics(gpu_handles, duration_seconds=30, interval_seconds=2):
    """Monitor partition metrics over time."""

    print(f"\n=== Monitoring Partition Metrics for {duration_seconds} seconds ===")
    print(f"Sampling every {interval_seconds} seconds...")
    print(f"Press Ctrl+C to stop early\n")

    start_time = time.time()
    sample_count = 0

    try:
        while (time.time() - start_time) < duration_seconds:
            sample_count += 1
            current_time = time.time() - start_time

            print(f"Sample #{sample_count} at {current_time:.1f}s:")

            for i, gpu_handle in enumerate(gpu_handles):
                try:
                    partition_metrics = amdsmi.amdsmi_get_gpu_partition_metrics_info(gpu_handle)

                    gfx_util = partition_metrics.get("current_gfxclk_utilization", "N/A")
                    throttle_mask = partition_metrics.get("throttle_status_bitmask", 0)

                    throttle_status = "None" if throttle_mask == 0 else f"0x{throttle_mask:x}"

                    print(f"  GPU {i}: GFX={gfx_util}%, Throttle={throttle_status}")

                except amdsmi.AmdSmiException:
                    print(f"  GPU {i}: XCP metrics not available")

            print()  # Empty line between samples
            time.sleep(interval_seconds)

    except KeyboardInterrupt:
        print(f"\nMonitoring stopped by user after {sample_count} samples")


def main():
    """Main function to demonstrate XCP partition metrics."""

    print("AMD SMI XCP Partition Metrics Example")
    print("=" * 45)

    try:
        # Initialize AMD SMI
        ret = amdsmi.amdsmi_init()
        print(f"AMD SMI initialized successfully (return code: {ret})")

        # Get list of GPU devices
        gpu_handles = amdsmi.amdsmi_get_processor_handles()

        if not gpu_handles:
            print("No GPU devices found on the system.")
            return

        print(f"Found {len(gpu_handles)} GPU device(s)")

        # Display partition metrics for each GPU
        for i, gpu_handle in enumerate(gpu_handles):
            print_partition_metrics_info(gpu_handle, i)

        # Ask user if they want to monitor metrics over time
        print(f"\n" + "=" * 45)

        try:
            response = input("Monitor partition metrics over time? (y/N): ").strip().lower()
            if response in ["y", "yes"]:
                monitor_partition_metrics(gpu_handles)
        except (KeyboardInterrupt, EOFError):
            print("\nSkipping monitoring...")

        print(f"\n" + "=" * 45)
        print("XCP Partition Metrics Example Complete")

    except amdsmi.AmdSmiException as e:
        print(f"AMD SMI Error: {e}")
        return 1

    except Exception as e:
        print(f"Unexpected error: {e}")
        return 1

    finally:
        try:
            amdsmi.amdsmi_shut_down()
            print("AMD SMI shut down successfully")
        except:
            pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
