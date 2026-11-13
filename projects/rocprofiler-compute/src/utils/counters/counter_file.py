# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Perfmon pass files and the counter slots each hardware block has."""

from typing import Dict, List

from utils.utils_counter_defs import counter_to_block

_ACCUM_SUFFIX = "_ACCUM"


def slot_cost(counter: str) -> int:
    """Return the block slots a counter uses: 2 for an *_ACCUM counter, which
    also uses an accumulation register, and 1 for any other counter."""
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

    def add(self, counter: str) -> bool:
        return self.blocks[counter_to_block(counter)].add(counter, slot_cost(counter))

    def reserve(self, counter: str, n: int) -> bool:
        return self.blocks[counter_to_block(counter)].reserve(n)
