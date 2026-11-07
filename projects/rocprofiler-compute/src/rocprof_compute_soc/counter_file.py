# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Perfmon bucket file types, independent of the SoC allocator."""

from typing import Dict, List, Optional, Sequence, Set, Tuple

from utils.utils_counter_defs import counter_to_block, pmc_slot_cost

# Families that share one block budget but must not share a pass.
# gfx1250 charges SP_VALU_* and SQ_VMEM_* to SQ. SP FLOP counters return
# zero when those families share a pass, and free SQ slots do not prevent it.
_INCOMPATIBLE_PASS_PREFIXES: Tuple[Tuple[str, str], ...] = (("SP_VALU_", "SQ_VMEM_"),)


def _counter_base_name(counter: str) -> str:
    """Counter name without a channel suffix."""
    return counter.split("[", 1)[0]


def _has_family(names: Sequence[str], prefix: str) -> bool:
    return any(name.startswith(prefix) for name in names)


def _family_prefix(counter: str) -> Optional[str]:
    """Incompatible-family prefix of counter, if it belongs to one."""
    base = _counter_base_name(counter)
    for left, right in _INCOMPATIBLE_PASS_PREFIXES:
        if base.startswith(left):
            return left
        if base.startswith(right):
            return right
    return None


def _cannot_share_pass(existing: Sequence[str], incoming: str) -> bool:
    """Return whether incoming must not join existing in one pass."""
    bases = [_counter_base_name(name) for name in existing]
    incoming_base = _counter_base_name(incoming)
    for left, right in _INCOMPATIBLE_PASS_PREFIXES:
        has_left = incoming_base.startswith(left) or _has_family(bases, left)
        has_right = incoming_base.startswith(right) or _has_family(bases, right)
        if has_left and has_right:
            return True
    return False


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

    def __init__(self, name: str, perfmon_config: Dict[str, int]) -> None:
        self.name: str = name
        self.blocks: Dict[str, LimitedSet] = {
            block: LimitedSet(capacity) for block, capacity in perfmon_config.items()
        }
        # Prefixes of incompatible families already stored in this pass.
        self._seen_families: Set[str] = set()

    def add(self, counter: str) -> bool:
        block = counter_to_block(counter)
        if counter in self.blocks[block].elements:
            return True
        # The family check looks at the prefixes recorded below, not every
        # counter already in the pass.
        if self._seen_families and _cannot_share_pass(
            tuple(self._seen_families), counter
        ):
            return False
        cost = pmc_slot_cost(counter, present=self.blocks[block].elements)
        if not self.blocks[block].add(counter, cost=cost):
            return False
        prefix = _family_prefix(counter)
        if prefix is not None:
            self._seen_families.add(prefix)
        return True

    def reserve(self, counter: str, n: int) -> bool:
        return self.blocks[counter_to_block(counter)].reserve(n)


def flat_counters_in_perfmon_file(counter_file: CounterFile) -> List[str]:
    """Ordered list of PMC counter names assigned to one perfmon bucket file."""
    return [
        ctr
        for block_name in counter_file.blocks
        for ctr in counter_file.blocks[block_name].elements
    ]
