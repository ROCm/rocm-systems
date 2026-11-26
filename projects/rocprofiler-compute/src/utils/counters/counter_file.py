# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Perfmon pass files and the counter slots each hardware block has."""

from typing import Dict, List, Set

from utils.utils_counter_defs import counter_to_block

_ACCUM_SUFFIX = "_ACCUM"


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


# Set with limited size
class LimitedSet:
    def __init__(self, maxsize: int) -> None:
        self.avail: int = maxsize
        # Ordered set: keeps the order counters are written in
        self.elements: Dict[str, None] = {}
        self._bases: Set[str] = set()

    def add(self, element: str, cost: int = 1) -> bool:
        if element in self.elements:
            return True
        # Store all channels for a TCC channel counter in the same file
        if element.split("[")[0] in self._bases:
            self.elements[element] = None
            return True
        if self.avail >= cost:
            self.avail -= cost
            self.elements[element] = None
            self._bases.add(element.split("[")[0])
            return True
        return False


# Represents a file that lists PMC counters. Number of counters for each
# block limited according to perfmon config.
class CounterFile:
    def __init__(self, name: str, perfmon_config: Dict[str, int]) -> None:
        self.name: str = name
        self.blocks: Dict[str, LimitedSet] = {
            block: LimitedSet(capacity) for block, capacity in perfmon_config.items()
        }

    def add(self, counter: str) -> bool:
        return self.blocks[counter_to_block(counter)].add(counter, slot_cost(counter))
