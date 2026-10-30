# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Shared call-tree records with standard-library dependencies only."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class KernelStats:
    """Aggregated kernel launch stats for one kernel name.

    min_duration_ns and max_duration_ns are None until a dispatch with a
    positive duration is observed. pmc_group_keys holds one
    process_rocpd_csv group key per launch of this name.
    """

    launches: int = 0
    total_duration_ns: float = 0.0
    min_duration_ns: Optional[float] = None
    max_duration_ns: Optional[float] = None
    kernel_id: Optional[int] = None
    pmc_group_keys: List[Tuple] = field(default_factory=list)


@dataclass
class CallTreeNode:
    """A node in the operator call tree.

    children is the list of child operator nodes. file_name, line_number,
    backend, start_timestamp, end_timestamp, thread_id, and
    launcher_thread_id are optional fields copied from a marker row when set.
    invocation_ids stores marker-start strings for this node. args_invocations
    maps each distinct operator-argument blob to the invocation ids that used
    it.

    Inclusive over this node plus all descendants:
      kernel_launches, total_duration_ms, min/max/mean dispatch stats.
    """

    name: str
    children: List["CallTreeNode"] = field(default_factory=list)
    kernels: Dict[str, KernelStats] = field(default_factory=dict)
    kernel_launches: int = 0
    total_duration_ms: float = 0.0
    invocation_ids: Set[str] = field(default_factory=set)
    args_invocations: Dict[str, Set[str]] = field(default_factory=dict)
    min_dispatch_ns: Optional[float] = None
    max_dispatch_ns: Optional[float] = None
    mean_dispatch_ns: Optional[float] = None
    file_name: Optional[str] = None
    line_number: Optional[int] = None
    backend: Optional[str] = None
    start_timestamp: Optional[float] = None
    end_timestamp: Optional[float] = None
    thread_id: Optional[str] = None
    launcher_thread_id: Optional[str] = None

    @property
    def call_count(self) -> int:
        return len(self.invocation_ids)

    @property
    def args_variants(self) -> List[Tuple[str, int]]:
        """Distinct operator-argument blobs with call counts, most frequent first."""
        return sorted(
            (
                (args_blob, len(recorded_invocation_ids))
                for args_blob, recorded_invocation_ids in self.args_invocations.items()
            ),
            key=lambda variant: (-variant[1], variant[0]),
        )


@dataclass
class NodeRollup:
    """Inclusive subtree aggregate returned by rollup_node_stats."""

    launches: int
    total_duration_ns: float
    min_dispatch_ns: Optional[float]
    max_dispatch_ns: Optional[float]


# Use the actual node type so other modules can resolve the forest annotation.
CallTreeForest = Dict[str, List[CallTreeNode]]
