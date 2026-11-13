# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Perfmon pass files and the counter slots each hardware block has."""

from typing import Dict, List, Optional, Set

from utils.utils_counter_defs import counter_to_block

_ACCUM_SUFFIX = "_ACCUM"
# Counter families that must not share a pass, mapped to the family they clash
# with. On gfx1250 the SP FLOP counters read zero when SP_VALU_* and SQ_VMEM_*
# share a pass, even with free SQ slots.
_INCOMPATIBLE_FAMILIES: Dict[str, str] = {
    "SP_VALU_": "SQ_VMEM_",
    "SQ_VMEM_": "SP_VALU_",
}


def slot_cost(counter: str) -> int:
    """Return the block slots a counter uses. An *_ACCUM counter uses 2, one
    more for its accumulation register. Any other counter uses 1."""
    return 2 if counter.split("[", 1)[0].endswith(_ACCUM_SUFFIX) else 1


def flat_counters_in_perfmon_file(counter_file: "CounterFile") -> List[str]:
    """Ordered list of PMC counter names assigned to one perfmon bucket file."""
    return [
        ctr
        for block_name in counter_file.blocks
        for ctr in counter_file.blocks[block_name].elements
    ]


def _incompatible_family(counter: str) -> Optional[str]:
    """Return the _INCOMPATIBLE_FAMILIES prefix that counter starts with."""
    return next(
        (prefix for prefix in _INCOMPATIBLE_FAMILIES if counter.startswith(prefix)),
        None,
    )


# Set with limited size
class LimitedSet:
    def __init__(self, maxsize: int) -> None:
        self.avail: int = maxsize
        self.elements: List[str] = []

    def add(self, element: str, cost: int = 1) -> bool:
        if element in self.elements:
            return True
        # Store all channels for a TCC channel counter in the same file
        if element.split("[")[0] in {elem.split("[")[0] for elem in self.elements}:
            self.elements.append(element)
            return True
        if self.avail >= cost:
            self.avail -= cost
            self.elements.append(element)
            return True
        return False

    def reserve(self, n: int) -> bool:
        if self.avail < n:
            return False
        self.avail -= n
        return True


# Represents a file that lists PMC counters. Number of counters for each
# block limited according to perfmon config.
class CounterFile:
    def __init__(self, name: str, perfmon_config: Dict[str, int]) -> None:
        self.name: str = name
        self.blocks: Dict[str, LimitedSet] = {
            block: LimitedSet(capacity) for block, capacity in perfmon_config.items()
        }
        # Families from _INCOMPATIBLE_FAMILIES that this pass already holds
        self._families: Set[str] = set()

    def add(self, counter: str) -> bool:
        family = _incompatible_family(counter)
        if family is not None and _INCOMPATIBLE_FAMILIES[family] in self._families:
            return False
        if not self.blocks[counter_to_block(counter)].add(counter, slot_cost(counter)):
            return False
        if family is not None:
            self._families.add(family)
        return True

    def reserve(self, counter: str, n: int) -> bool:
        return self.blocks[counter_to_block(counter)].reserve(n)
