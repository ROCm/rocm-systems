#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""Standalone violation-status verification script.

Usage:
    python3 amd_smi_violation_example.py

Calls amdsmi_get_violation_status() for every GPU and prints all fields.
On older ASICs (Navi/MI1x/MI2x) where the violation API is not supported,
falls back to throttle_status via amdsmi_get_gpu_metrics_info().
N/A means the field is unsupported on this ASIC (max_uint sentinel).
MI3x+ (MI300X and newer) is required for full violation data.
"""

import amdsmi


def _active_str(value) -> str:
    """Convert active flag value to ACTIVE/NOT ACTIVE/N/A, matching CLI and C++ output."""
    if value == "N/A":
        return "N/A"
    return "ACTIVE" if value else "NOT ACTIVE"


def main() -> None:
    amdsmi.amdsmi_init()
    try:
        processors = amdsmi.amdsmi_get_processor_handles()
        if not processors:
            print("No processors found.")
            return

        for i, processor in enumerate(processors):
            print(f"\n{'=' * 60}")
            print(f"GPU {i} violation status")
            print(f"{'=' * 60}")

            try:
                v = amdsmi.amdsmi_get_violation_status(processor)
            except amdsmi.AmdSmiException as exc:
                # Navi/MI1x/MI2x: violation API not supported — fall back to gpu_metrics
                print(f"  amdsmi_get_violation_status failed: {exc}")
                print("  Falling back to throttle_status via amdsmi_get_gpu_metrics_info()...")
                try:
                    m = amdsmi.amdsmi_get_gpu_metrics_info(processor)
                    # throttle_status: same field the CLI uses for amd-smi metric --power
                    ts = m.get("throttle_status", "N/A")
                    if ts == "N/A":
                        print("  throttle_status: N/A")
                    elif ts:
                        print("  throttle_status: THROTTLED")
                    else:
                        print("  throttle_status: UNTHROTTLED")
                except amdsmi.AmdSmiException as metrics_exc:
                    print(f"  amdsmi_get_gpu_metrics_info also failed: {metrics_exc}")
                continue

            # -- Metadata --
            print(f"  reference_timestamp (us since epoch): {v['reference_timestamp']}")
            print(f"  violation_timestamp (ns):             {v['violation_timestamp']}")
            print(f"  acc_counter:                          {v['acc_counter']}")

            # -- Accumulated counters --
            print("\n  -- Accumulated counters --")
            print(f"  acc_prochot_thrm:  {v['acc_prochot_thrm']}")
            print(f"  acc_ppt_pwr:       {v['acc_ppt_pwr']}")  # PVIOL
            print(f"  acc_socket_thrm:   {v['acc_socket_thrm']}")  # TVIOL
            print(f"  acc_vr_thrm:       {v['acc_vr_thrm']}")
            print(f"  acc_hbm_thrm:      {v['acc_hbm_thrm']}")

            # -- Violation % (>0% = throttled) --
            print("\n  -- Violation % (>0% = throttled) --")
            print(f"  per_prochot_thrm (%): {v['per_prochot_thrm']}")
            print(f"  per_ppt_pwr (%):      {v['per_ppt_pwr']}")  # PVIOL
            print(f"  per_socket_thrm (%):  {v['per_socket_thrm']}")  # TVIOL
            print(f"  per_vr_thrm (%):      {v['per_vr_thrm']}")
            print(f"  per_hbm_thrm (%):     {v['per_hbm_thrm']}")

            # -- Active flags --
            print("\n  -- Active flags (ACTIVE / NOT ACTIVE / N/A=unsupported) --")
            print(f"  active_prochot_thrm: {_active_str(v['active_prochot_thrm'])}")
            print(f"  active_ppt_pwr:      {_active_str(v['active_ppt_pwr'])}")
            print(f"  active_socket_thrm:  {_active_str(v['active_socket_thrm'])}")
            print(f"  active_vr_thrm:      {_active_str(v['active_vr_thrm'])}")
            print(f"  active_hbm_thrm:     {_active_str(v['active_hbm_thrm'])}")

            # -- GPU metrics v1.8 per-XCP/XCC 2D arrays --
            # Each field is a list-of-lists indexed [xcp][xcc].
            # Skip fields where every entry is "N/A" to keep output clean on pre-v1.8 drivers.
            xcp_fields = [
                "acc_gfx_clk_below_host_limit_pwr",
                "acc_gfx_clk_below_host_limit_thm",
                "acc_low_utilization",
                "acc_gfx_clk_below_host_limit_total",
                "per_gfx_clk_below_host_limit_pwr",
                "per_gfx_clk_below_host_limit_thm",
                "per_low_utilization",
                "per_gfx_clk_below_host_limit_total",
                "active_gfx_clk_below_host_limit_pwr",
                "active_gfx_clk_below_host_limit_thm",
                "active_low_utilization",
                "active_gfx_clk_below_host_limit_total",
            ]

            any_xcp_printed = False
            for field in xcp_fields:
                rows = v[field]
                if all(val == "N/A" for row in rows for val in row):
                    continue
                if not any_xcp_printed:
                    print("\n  -- GPU metrics v1.8 per-XCP/XCC (N/A = unsupported) --")
                    any_xcp_printed = True
                print(f"  {field}:")
                for xcp_idx, row in enumerate(rows):
                    # Only print XCC rows that have at least one non-N/A value
                    if all(val == "N/A" for val in row):
                        continue
                    # Translate active flag rows to ACTIVE/NOT ACTIVE to match CLI/C++ output
                    if field.startswith("active_"):
                        display_row = [_active_str(val) for val in row]
                    else:
                        display_row = row
                    print(f"    XCP[{xcp_idx}]: {display_row}")

            if not any_xcp_printed:
                print("\n  -- GPU metrics v1.8 per-XCP/XCC: all N/A (pre-v1.8 driver) --")

    finally:
        amdsmi.amdsmi_shut_down()


if __name__ == "__main__":
    main()
