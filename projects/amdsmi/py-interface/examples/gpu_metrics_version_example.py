#!/usr/bin/env python3

"""
AMD SMI GPU Metrics Version Detection Example

This example demonstrates how to:
1. Detect GPU metrics version (v1.x, v1.8, v3.x)
2. Access version-specific metrics data
3. Handle different metrics structures across versions
4. Display backward compatibility information

Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""

import sys
import os

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
        from amdsmi_interface import amdsmi_get_gpu_metrics_header_info, amdsmi_get_gpu_metrics_info
        from amdsmi_exception import AmdSmiException

        # Create a fake amdsmi module-like object for compatibility
        class AmdsmiCompat:
            amdsmi_init = amdsmi_init
            amdsmi_shut_down = amdsmi_shut_down
            amdsmi_get_processor_handles = amdsmi_get_processor_handles
            amdsmi_get_gpu_metrics_header_info = amdsmi_get_gpu_metrics_header_info
            amdsmi_get_gpu_metrics_info = amdsmi_get_gpu_metrics_info
            AmdSmiException = AmdSmiException

        amdsmi = AmdsmiCompat()
    except ImportError as e:
        print(
            f"Error: AMD SMI library not found. Please ensure amdsmi is installed or built. Details: {e}"
        )
        sys.exit(1)


def print_gpu_metrics_version_info(gpu_handle, gpu_id):
    """Print detailed GPU metrics version information for a specific GPU."""

    print(f"\n=== GPU {gpu_id} Metrics Version Information ===")

    try:
        # Get GPU metrics header information
        metrics_header = amdsmi.amdsmi_get_gpu_metrics_header_info(gpu_handle)

        format_revision = metrics_header.get("format_revision", "Unknown")
        content_revision = metrics_header.get("content_revision", "Unknown")

        print(f"Format Revision: {format_revision}")
        print(f"Content Revision: {content_revision}")

        # Determine metrics version based on format revision
        if format_revision == 1:
            if content_revision >= 8:
                version_str = "v1.8+ (XCP partition support)"
            else:
                version_str = "v1.x (Legacy)"
        elif format_revision >= 3:
            version_str = "v3.x (Latest with enhanced XCP support)"
        else:
            version_str = f"Unknown (format: {format_revision}, content: {content_revision})"

        print(f"Detected Version: {version_str}")

        # Get and analyze metrics structure
        metrics_info = amdsmi.amdsmi_get_gpu_metrics_info(gpu_handle)

        # Check for key features by version
        print(f"\nFeature Support Analysis:")

        # Basic metrics (available in all versions)
        basic_metrics = [
            "average_gfx_activity",
            "average_umc_activity",
            "average_mm_activity",
            "current_gfxclk",
            "current_mclk",
            "current_socclk",
        ]

        available_basic = sum(1 for metric in basic_metrics if metric in metrics_info)
        print(f"  Basic Metrics: {available_basic}/{len(basic_metrics)} available")

        # XCP partition metrics (v1.8+ and v3.x)
        partition_metrics = [
            "current_gfxclk_utilization",
            "current_uclk_utilization",
            "throttle_status_bitmask",
        ]

        available_partition = sum(1 for metric in partition_metrics if metric in metrics_info)
        print(f"  XCP Partition Metrics: {available_partition}/{len(partition_metrics)} available")

        # Enhanced v3.x features
        v3_features = ["num_partition", "gfxclk_lock_status"]

        available_v3 = sum(1 for feature in v3_features if feature in metrics_info)
        print(f"  v3.x Enhanced Features: {available_v3}/{len(v3_features)} available")

        # Display sample metrics data
        print(f"\nSample Metrics Data:")

        if "average_gfx_activity" in metrics_info:
            print(f"  GFX Activity: {metrics_info['average_gfx_activity']}%")

        if "current_gfxclk" in metrics_info:
            current_gfxclk = metrics_info["current_gfxclk"]
            if current_gfxclk != "N/A" and isinstance(current_gfxclk, (int, float)):
                gfxclk_mhz = current_gfxclk / 1000000  # Convert Hz to MHz
                print(f"  Current GFX Clock: {gfxclk_mhz:.0f} MHz")
            else:
                print(f"  Current GFX Clock: {current_gfxclk}")

        if "current_gfxclk_utilization" in metrics_info:
            print(f"  Current GFXCLK Utilization: {metrics_info['current_gfxclk_utilization']}%")

        if "throttle_status_bitmask" in metrics_info:
            throttle_mask = metrics_info["throttle_status_bitmask"]
            if throttle_mask != 0:
                print(f"  Throttle Status: Active (0x{throttle_mask:x})")
            else:
                print(f"  Throttle Status: None")

    except amdsmi.AmdSmiException as e:
        print(f"Error getting GPU metrics for GPU {gpu_id}: {e}")


def main():
    """Main function to demonstrate GPU metrics version detection."""

    print("AMD SMI GPU Metrics Version Detection Example")
    print("=" * 50)

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

        # Process each GPU
        for i, gpu_handle in enumerate(gpu_handles):
            print_gpu_metrics_version_info(gpu_handle, i)

        print(f"\n" + "=" * 50)
        print("GPU Metrics Version Detection Complete")

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
