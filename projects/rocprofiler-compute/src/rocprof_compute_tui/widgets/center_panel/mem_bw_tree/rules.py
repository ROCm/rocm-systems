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

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .predicates import (
    AlwaysTrue,
    AnyOf,
    Compare,
    Dominates,
    Predicate,
)

# Node IDs (stable)
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

# Metric IDs (canonical)
METRICS_HIGH_EA_L2_BP = (
    "4.1.10",
    "17.2.0",
    "17.2.4",
)

METRICS_HIGH_TCR_TCP_BP = ("16.3.20",)

METRICS_VL1D_HIT_LOW = ("16.3.5",)

METRICS_NON_TEMPORAL_SET = ()

METRICS_HIGH_TC_TA_BP = ("16.3.21",)

METRICS_HIGH_TA_VMEM_BP = (
    "15.1.1",
    "11.2.6",
)

METRICS_HIGH_VL1D_SET_FULL_STALL = ("16.3.22",)

METRICS_TAGRAM_HOTSPOTTING = (
    "16.3.10",
    "16.3.11",
    "16.3.12",
    "16.3.13",
)

METRICS_UTCL1_STALL = ("16.5.2",)

METRICS_UTCL1_LATENCY = ("16.6.7",)

METRICS_L2_CHANNEL_HOTSPOTTING = ("18",)

METRICS_L2_HIT_LOW = ("17.1.2",)

METRICS_WRITE_TRAFFIC = (
    "17.2.5",
    "17.2.6",
)


@dataclass(frozen=True)
class Rule:
    node_id: str
    metric_ids: Sequence[str]
    predicate: Predicate


RULES: list[Rule] = [
    # Rules references doc "AMD Instinct™ MI300 GPU Memory Bandwidth"
    Rule(
        #
        node_id=NODE_HIGH_EA_L2_BP,
        metric_ids=METRICS_HIGH_EA_L2_BP,
        predicate=Compare("17.2.0 + 17.2.4", "4.1.10 * 0.75", ">="),
    ),
    Rule(
        # "A high GL2-GL1 stall rate, e.g., over 10%,
        # could clearly indicate that GL2 is a potential bottleneck."
        node_id=NODE_HIGH_TCR_TCP_BP,
        metric_ids=METRICS_HIGH_TCR_TCP_BP,
        predicate=Compare("16.3.20", 10, "<"),
    ),
    Rule(
        node_id=NODE_NON_TEMPORAL_SET,
        metric_ids=METRICS_NON_TEMPORAL_SET,
        predicate=AlwaysTrue("Assuming workload is Non-Temporal."),
    ),
    Rule(
        # "On the TA-TCP interface, the key metric TCP_TA_ADDR_STALL_RATE
        # allows identifying whether TCP is a bottleneck"
        node_id=NODE_HIGH_TC_TA_BP,
        metric_ids=METRICS_HIGH_TC_TA_BP,
        predicate=Compare("16.3.21", 10, ">"),
    ),
    Rule(
        node_id=NODE_HIGH_VL1D_SET_FULL_STALL,
        metric_ids=METRICS_HIGH_VL1D_SET_FULL_STALL,
        predicate=Compare("16.3.22", 10, ">"),
    ),
    Rule(
        node_id=NODE_TAGRAM_HOTSPOTTING,
        metric_ids=METRICS_TAGRAM_HOTSPOTTING,
        predicate=AnyOf([
            Dominates("16.3.10", ["16.3.11, 16.3.12, 16.3.13"]),
            Dominates("16.3.11", ["16.3.10, 16.3.12, 16.3.13"]),
            Dominates("16.3.12", ["16.3.10, 16.3.11, 16.3.13"]),
            Dominates("16.3.13", ["16.3.10, 16.3.11, 16.3.12"]),
        ]),
    ),
]
