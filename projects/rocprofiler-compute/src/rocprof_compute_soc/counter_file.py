# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Perfmon bucket file types, independent of the SoC allocator."""

from __future__ import annotations

from utils.utils_counter_defs import counter_to_block, pmc_slot_cost


class LimitedSet:
    def __init__(self, maxsize: int) -> None:
        self.avail: int = maxsize
        self.elements: list[str] = []

    def add(self, element: str, cost: int = 1) -> bool:
        if element in self.elements:
            return True
        # Store all channels for a TCC channel counter in the same file
        if element.split("[")[0] in {elem.split("[")[0] for elem in self.elements}:
            self.elements.append(element)
            return True
        if cost < 0:
            cost = 0
        if cost == 0:
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


class CounterFile:
    """One perfmon pass file. Block sizes come from ``perfmon_config``."""

    def __init__(self, name: str, perfmon_config: dict[str, int]) -> None:
        self.name: str = name
        self.blocks: dict[str, LimitedSet] = {
            block: LimitedSet(capacity) for block, capacity in perfmon_config.items()
        }

    def add(self, counter: str) -> bool:
        block = counter_to_block(counter)
        cost = pmc_slot_cost(counter, present=self.blocks[block].elements)
        return self.blocks[block].add(counter, cost=cost)

    def reserve(self, counter: str, n: int) -> bool:
        return self.blocks[counter_to_block(counter)].reserve(n)


def flat_counters_in_perfmon_file(counter_file: CounterFile) -> list[str]:
    """Ordered list of PMC counter names assigned to one perfmon bucket file."""
    return [
        ctr
        for block_name in counter_file.blocks
        for ctr in counter_file.blocks[block_name].elements
    ]
