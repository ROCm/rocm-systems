##############################################################################
# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
##############################################################################

# flake8: noqa: E501

from __future__ import annotations

from typing import Any, Optional

# This is where we centralize node IDs and map them to the existing dict structure.

ROOT = "workload_bw_measurement"
NODE_HIGH_EA_L2_BP = "high_ea_l2_backpressure"
NODE_HIGH_TCR_TCP_BP = "high_tcr_tcp_backpressure"
NODE_VL1D_HIT_LOW = "vl1d_hit_low"
NODE_NON_TEMPORAL_SET = "non_temporal_set"
NODE_HIGH_TC_TA_BP = "high_tc_ta_backpressure"
NODE_HIGH_TA_VMEM_BP = "high_ta_vmem_backpressure"
NODE_HIGH_VL1D_SET_FULL_STALL = "high_vl1d_set_full_stall"
NODE_TAGRAM_HOTSPOTTING = "tagram_hotspotting"
NODE_UTCL1_STALL = "utcl1_stall"
NODE_UTCL1_LATENCY = "utcl1_latency"
NODE_L2_CHANNEL_HOTSPOTTING = "l2_channel_hotspotting"
NODE_L2_HIT_LOW = "l2_hit_low"
NODE_WRITE_TRAFFIC = "write_traffic"


def build_paths(tree: dict, prefix: Optional[list[str]] = None) -> dict[str, list[str]]:
    if prefix is None:
        prefix = []

    node_id = tree["node_id"]
    path = prefix + [node_id]

    paths = {node_id: path}

    for child in tree.get("children", []):
        paths.update(build_paths(child, path))

    return paths


TREE_DICT: dict[str, Any] = {
    "node_id": ROOT,
    "label": "Memory Bandwidth Measurement",
    "children": [
        {
            "node_id": NODE_HIGH_EA_L2_BP,
            "label": "High EA -> L2 Backpressure",
            "children": [
                {
                    "node_id": NODE_HIGH_TCR_TCP_BP,
                    "label": "High TCR -> TCP Backpressure",
                    "children": [
                        {
                            "node_id": NODE_VL1D_HIT_LOW,
                            "label": "vL1d Hit Low",
                            "children": [
                                {
                                    "node_id": NODE_NON_TEMPORAL_SET,
                                    "label": "Non-Temporal set",
                                    "children": [
                                        {
                                            "node_id": "nt1",
                                            "label": "Try non-temporal (NT)",
                                        },
                                        {
                                            "node_id": NODE_HIGH_TC_TA_BP,
                                            "label": "High TC -> TA Backpressure",
                                            "children": [
                                                {
                                                    "node_id": NODE_HIGH_TA_VMEM_BP,
                                                    "label": "High TA -> VMEM Backpressure",
                                                    "children": [
                                                        {
                                                            "node_id": "sw_opt1",
                                                            "label": "SW optimization",
                                                        },
                                                        {
                                                            "node_id": "bottleneck_workload",
                                                            "label": "Bottleneck: Workload",
                                                        },
                                                    ],
                                                },
                                                {
                                                    "node_id": NODE_HIGH_VL1D_SET_FULL_STALL,
                                                    "label": "High vL1d set full stall",
                                                    "children": [
                                                        {
                                                            "node_id": NODE_TAGRAM_HOTSPOTTING,
                                                            "label": "Tagram hotspotting",
                                                            "children": [
                                                                {
                                                                    "node_id": "bottleneck_tcp_cap",
                                                                    "label": "Bottleneck: TCP capacity",
                                                                },
                                                                {
                                                                    "node_id": "sw_opt2",
                                                                    "label": "SW optimization",
                                                                },
                                                                {
                                                                    "node_id": NODE_UTCL1_STALL,
                                                                    "label": "UTCL1 Stall",
                                                                    "children": [
                                                                        {
                                                                            "node_id": NODE_UTCL1_LATENCY,
                                                                            "label": "UTCL1 Latency",
                                                                            "children": [
                                                                                {
                                                                                    "node_id": "bottleneck_utcl2",
                                                                                    "label": "Bottleneck: UTCL2",
                                                                                }
                                                                            ],
                                                                        }
                                                                    ],
                                                                },
                                                            ],
                                                        }
                                                    ],
                                                },
                                            ],
                                        },
                                    ],
                                }
                            ],
                        },
                        {
                            "node_id": NODE_L2_CHANNEL_HOTSPOTTING,
                            "label": "L2 Channel hotspotting",
                            "children": [
                                {"node_id": "sw_opt3", "label": "SW optimization"},
                                {
                                    "node_id": NODE_L2_HIT_LOW,
                                    "label": "L2 Hit Low",
                                    "children": [
                                        {
                                            "node_id": "nt2",
                                            "label": "Try non-temporal (NT)",
                                        },
                                        {
                                            "node_id": NODE_WRITE_TRAFFIC,
                                            "label": "Write Traffic",
                                            "children": [
                                                {
                                                    "node_id": "bottleneck_l2_arb",
                                                    "label": "Bottleneck: L2 Arb",
                                                },
                                                {
                                                    "node_id": "unlikely",
                                                    "label": "Unlikely",
                                                },
                                            ],
                                        },
                                    ],
                                },
                            ],
                        },
                    ],
                },
            ],
        }
    ],
}
