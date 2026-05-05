#!/usr/bin/env python3
"""
Counter Collection Optimizer for rocprofv3

Optimizes performance counter collection by grouping counters to minimize
application replays while respecting hardware block limits.

Uses First-Fit Decreasing (FFD) bin-packing algorithm to pack counters into
minimal number of collection passes.
"""

import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


# Hardware limits extracted from aqlprofile gfxip/*/block_info.h files
HARDWARE_LIMITS = {
    # GFX8 (Polaris, Fiji)
    "gfx800": {"SQ": 8, "TA": 2, "TCP": 4, "TCC": 4, "TCA": 4, "GRBM": 4, "SPI": 6},
    "gfx801": {"SQ": 8, "TA": 2, "TCP": 4, "TCC": 4, "TCA": 4, "GRBM": 4, "SPI": 6},
    "gfx802": {"SQ": 8, "TA": 2, "TCP": 4, "TCC": 4, "TCA": 4, "GRBM": 4, "SPI": 6},
    "gfx803": {"SQ": 8, "TA": 2, "TCP": 4, "TCC": 4, "TCA": 4, "GRBM": 4, "SPI": 6},
    "gfx805": {"SQ": 8, "TA": 2, "TCP": 4, "TCC": 4, "TCA": 4, "GRBM": 4, "SPI": 6},

    # GFX9 (Vega, Radeon VII)
    "gfx900": {"SQ": 8, "TA": 2, "TCP": 4, "TCC": 4, "CB": 4, "DB": 4, "SPI": 6, "GRBM": 4},
    "gfx902": {"SQ": 8, "TA": 2, "TCP": 4, "TCC": 4, "CB": 4, "DB": 4, "SPI": 6, "GRBM": 4},
    "gfx904": {"SQ": 8, "TA": 2, "TCP": 4, "TCC": 4, "CB": 4, "DB": 4, "SPI": 6, "GRBM": 4},
    "gfx906": {"SQ": 8, "TA": 2, "TCP": 4, "TCC": 4, "CB": 4, "DB": 4, "SPI": 6, "GRBM": 4},
    "gfx908": {"SQ": 8, "TA": 2, "TCP": 4, "TCC": 4, "CB": 4, "DB": 4, "SPI": 6, "GRBM": 4},
    "gfx90a": {"SQ": 8, "TA": 2, "TCP": 4, "TCC": 4, "CB": 4, "DB": 4, "SPI": 6, "GRBM": 4, "TD": 2, "TCA": 2},
    "gfx90c": {"SQ": 8, "TA": 2, "TCP": 4, "TCC": 4, "CB": 4, "DB": 4, "SPI": 6, "GRBM": 4},
    "gfx940": {"SQ": 8, "TA": 2, "TCP": 4, "TCC": 4, "CB": 4, "DB": 4, "SPI": 6, "GRBM": 4, "TD": 2, "TCA": 2},
    "gfx941": {"SQ": 8, "TA": 2, "TCP": 4, "TCC": 4, "CB": 4, "DB": 4, "SPI": 6, "GRBM": 4, "TD": 2, "TCA": 2},
    "gfx942": {"SQ": 8, "TA": 2, "TCP": 4, "TCC": 4, "CB": 4, "DB": 4, "SPI": 6, "GRBM": 4, "TD": 2, "TCA": 2},

    # GFX10 (RDNA, RDNA2)
    "gfx1010": {"SQ": 8, "TA": 2, "TCP": 4, "GL1C": 4, "GL2C": 4, "GCR": 4, "GRBM": 4},
    "gfx1011": {"SQ": 8, "TA": 2, "TCP": 4, "GL1C": 4, "GL2C": 4, "GCR": 4, "GRBM": 4},
    "gfx1012": {"SQ": 8, "TA": 2, "TCP": 4, "GL1C": 4, "GL2C": 4, "GCR": 4, "GRBM": 4},
    "gfx1013": {"SQ": 8, "TA": 2, "TCP": 4, "GL1C": 4, "GL2C": 4, "GCR": 4, "GRBM": 4},
    "gfx1030": {"SQ": 8, "TA": 2, "TCP": 4, "GL1C": 4, "GL2C": 4, "GCR": 4, "GRBM": 4},
    "gfx1031": {"SQ": 8, "TA": 2, "TCP": 4, "GL1C": 4, "GL2C": 4, "GCR": 4, "GRBM": 4},
    "gfx1032": {"SQ": 8, "TA": 2, "TCP": 4, "GL1C": 4, "GL2C": 4, "GCR": 4, "GRBM": 4},
    "gfx1033": {"SQ": 8, "TA": 2, "TCP": 4, "GL1C": 4, "GL2C": 4, "GCR": 4, "GRBM": 4},
    "gfx1034": {"SQ": 8, "TA": 2, "TCP": 4, "GL1C": 4, "GL2C": 4, "GCR": 4, "GRBM": 4},
    "gfx1035": {"SQ": 8, "TA": 2, "TCP": 4, "GL1C": 4, "GL2C": 4, "GCR": 4, "GRBM": 4},

    # GFX11 (RDNA3)
    "gfx1100": {"SQ": 8, "TA": 2, "TCP": 4, "GL1C": 4, "GL2C": 4, "GRBM": 4},
    "gfx1101": {"SQ": 8, "TA": 2, "TCP": 4, "GL1C": 4, "GL2C": 4, "GRBM": 4},
    "gfx1102": {"SQ": 8, "TA": 2, "TCP": 4, "GL1C": 4, "GL2C": 4, "GRBM": 4},
    "gfx1103": {"SQ": 8, "TA": 2, "TCP": 4, "GL1C": 4, "GL2C": 4, "GRBM": 4},

    # GFX12 (future RDNA4 / MI-series)
    "gfx1200": {"SQC": 16, "SQG": 8, "TCP": 4, "GL1C": 4, "GL2C": 4, "CHA": 4, "CHC": 4, "GRBM": 4},
    "gfx1201": {"SQC": 16, "SQG": 8, "TCP": 4, "GL1C": 4, "GL2C": 4, "CHA": 4, "CHC": 4, "GRBM": 4},
}


@dataclass
class BlockUsage:
    """Tracks counter usage for a hardware block."""
    used: int = 0
    max: int = 0

    @property
    def utilization(self) -> float:
        """Return utilization as 0.0-1.0."""
        return self.used / self.max if self.max > 0 else 0.0

    @property
    def available(self) -> int:
        """Return number of available counter slots."""
        return max(0, self.max - self.used)

    def can_add(self, count: int = 1) -> bool:
        """Check if we can add more counters."""
        return (self.used + count) <= self.max


@dataclass
class CounterGroup:
    """Represents a collection pass with counters and block usage."""
    counters: List[str] = field(default_factory=list)
    block_usage: Dict[str, BlockUsage] = field(default_factory=dict)

    def can_add_counter(self, counter: str, block: str) -> bool:
        """Check if counter can be added without exceeding block limit."""
        if block not in self.block_usage:
            return True  # No limit yet
        return self.block_usage[block].can_add()

    def add_counter(self, counter: str, block: str, block_limit: int):
        """Add counter to this group."""
        self.counters.append(counter)

        if block not in self.block_usage:
            self.block_usage[block] = BlockUsage(used=0, max=block_limit)

        self.block_usage[block].used += 1


@dataclass
class OptimizationReport:
    """Results from counter optimization."""
    architecture: str
    input_counters: int
    original_passes: int
    optimized_passes: int
    groups: List[CounterGroup]
    unknown_counters: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def reduction_percent(self) -> float:
        """Calculate percentage reduction in passes."""
        if self.original_passes == 0:
            return 0.0
        return (1 - self.optimized_passes / self.original_passes) * 100

    @property
    def passes_saved(self) -> int:
        """Number of passes saved."""
        return max(0, self.original_passes - self.optimized_passes)


class CounterOptimizer:
    """Optimizes counter collection to minimize application replays."""

    def __init__(self, arch: str, hardware_limits: Dict[str, int], counter_metadata: Dict[str, str]):
        """
        Initialize optimizer.

        Args:
            arch: GPU architecture (e.g., "gfx90a")
            hardware_limits: Dict of block → max concurrent counters
            counter_metadata: Dict of counter name → hardware block
        """
        self.arch = arch
        self.hardware_limits = hardware_limits
        self.counter_metadata = counter_metadata

    @classmethod
    def create(cls, arch: str, metadata_path: Optional[Path] = None) -> 'CounterOptimizer':
        """
        Factory method to create optimizer for given architecture.

        Args:
            arch: GPU architecture (e.g., "gfx90a")
            metadata_path: Optional path to counter_blocks.json

        Returns:
            CounterOptimizer instance
        """
        # Get hardware limits
        arch_normalized = normalize_arch(arch)
        limits = HARDWARE_LIMITS.get(arch_normalized)

        if limits is None:
            # Try generic arch family (gfx9, gfx10, etc.)
            generic = re.match(r'(gfx\d+)', arch_normalized)
            if generic:
                family = generic.group(1)
                # Use first matching variant as fallback
                for key in HARDWARE_LIMITS:
                    if key.startswith(family):
                        limits = HARDWARE_LIMITS[key]
                        break

        if limits is None:
            # Ultimate fallback: conservative limits
            limits = {"SQ": 4, "TA": 2, "TCP": 2, "TCC": 2, "GRBM": 2}
            print(f"Warning: Unknown architecture '{arch}', using conservative limits", file=sys.stderr)

        # Load counter metadata
        if metadata_path is None:
            metadata_path = Path(__file__).parent / "counter_blocks.json"

        try:
            with open(metadata_path, 'r') as f:
                all_metadata = json.load(f)

            # Try exact match, then generic family
            counter_map = all_metadata.get(arch_normalized)
            if counter_map is None:
                generic = re.match(r'(gfx\d+)', arch_normalized)
                if generic:
                    counter_map = all_metadata.get(generic.group(1))

            if counter_map is None:
                counter_map = {}
                print(f"Warning: No counter metadata for '{arch}'", file=sys.stderr)

        except FileNotFoundError:
            counter_map = {}
            print(f"Warning: Counter metadata file not found: {metadata_path}", file=sys.stderr)

        return cls(arch_normalized, limits, counter_map)

    def optimize(self, counters: List[str], original_passes: int = 1) -> Tuple[List[List[str]], OptimizationReport]:
        """
        Groups counters into minimal passes respecting hardware limits.

        Uses First-Fit Decreasing (FFD) bin-packing algorithm:
        1. Sort counters by block (most constrained first)
        2. For each counter, try to fit into existing group
        3. If no group fits, create new group

        Args:
            counters: List of counter names
            original_passes: Number of passes in original input (for reporting)

        Returns:
            Tuple of (optimized counter groups, optimization report)
        """
        # Track unknown counters
        unknown_counters = []
        counter_to_block = {}

        # Map counters to blocks
        for counter in counters:
            block = self.counter_metadata.get(counter)
            if block is None:
                unknown_counters.append(counter)
                # Try to infer block from counter name (e.g., SQ_WAVES → SQ)
                block_prefix = counter.split('_')[0]
                if block_prefix in self.hardware_limits:
                    block = block_prefix
                else:
                    # Skip unknown counters
                    continue

            counter_to_block[counter] = block

        # Count counters per block
        block_counts = defaultdict(int)
        for block in counter_to_block.values():
            block_counts[block] += 1

        # Sort counters by block constraint (most constrained first)
        # This improves packing efficiency
        def counter_sort_key(counter: str) -> Tuple[int, str]:
            block = counter_to_block.get(counter, "")
            if not block:
                return (999999, counter)  # Unknown counters last

            block_limit = self.hardware_limits.get(block, 999999)
            count = block_counts[block]

            # Sort by: ratio of usage/limit (descending), then block name
            ratio = count / block_limit if block_limit > 0 else 0
            return (-ratio, block, counter)

        sorted_counters = sorted([c for c in counters if c in counter_to_block],
                                key=counter_sort_key)

        # First-Fit Decreasing bin packing
        groups: List[CounterGroup] = []

        for counter in sorted_counters:
            block = counter_to_block[counter]
            block_limit = self.hardware_limits.get(block, 1)

            # Try to fit into existing group
            placed = False
            for group in groups:
                if group.can_add_counter(counter, block):
                    group.add_counter(counter, block, block_limit)
                    placed = True
                    break

            # Create new group if needed
            if not placed:
                new_group = CounterGroup()
                new_group.add_counter(counter, block, block_limit)
                groups.append(new_group)

        # Convert groups to counter lists
        optimized_groups = [group.counters for group in groups]

        # Build report
        report = OptimizationReport(
            architecture=self.arch,
            input_counters=len(counters),
            original_passes=original_passes,
            optimized_passes=len(groups),
            groups=groups,
            unknown_counters=unknown_counters
        )

        if unknown_counters:
            report.warnings.append(
                f"Unknown counters skipped: {', '.join(unknown_counters)}"
            )

        return optimized_groups, report


def normalize_arch(arch: str) -> str:
    """Normalize architecture name to standard format."""
    # Remove common prefixes/suffixes
    arch = arch.lower().strip()
    arch = re.sub(r'^(amd:)?', '', arch)  # Remove AMD: prefix if present

    # Ensure gfx prefix
    if not arch.startswith('gfx'):
        arch = 'gfx' + arch

    return arch


def detect_architecture(explicit_arch: Optional[str] = None) -> str:
    """
    Detect GPU architecture from system.

    Priority:
    1. Explicit argument
    2. ROCPROFILER_TARGET_ARCH env var
    3. Query rocminfo
    4. Fallback to gfx90a

    Args:
        explicit_arch: Explicitly specified architecture

    Returns:
        Normalized architecture string (e.g., "gfx90a")
    """
    if explicit_arch:
        return normalize_arch(explicit_arch)

    # Try environment variable
    env_arch = os.environ.get('ROCPROFILER_TARGET_ARCH')
    if env_arch:
        return normalize_arch(env_arch)

    # Try rocminfo
    try:
        result = subprocess.run(['rocminfo'], capture_output=True, text=True, timeout=5)
        # Look for gfx pattern (e.g., "gfx90a", "gfx1100")
        matches = re.findall(r'\bgfx\w+', result.stdout)
        if matches:
            # Return first match
            return normalize_arch(matches[0])
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # Fallback
    print("Warning: Could not detect GPU architecture, using gfx90a as fallback", file=sys.stderr)
    return "gfx90a"


if __name__ == '__main__':
    # Simple CLI for testing
    if len(sys.argv) < 2:
        print("Usage: counter_optimizer.py <counter1> <counter2> ...", file=sys.stderr)
        print("   or: counter_optimizer.py --arch gfx90a <counter1> <counter2> ...", file=sys.stderr)
        sys.exit(1)

    arch = None
    if '--arch' in sys.argv:
        idx = sys.argv.index('--arch')
        arch = sys.argv[idx + 1]
        counters = sys.argv[idx + 2:]
    else:
        counters = sys.argv[1:]

    arch = detect_architecture(arch)
    optimizer = CounterOptimizer.create(arch)

    groups, report = optimizer.optimize(counters, original_passes=1)

    print(f"Architecture: {report.architecture}")
    print(f"Input counters: {report.input_counters}")
    print(f"Optimized passes: {report.optimized_passes}")
    print()

    for i, group in enumerate(groups, 1):
        print(f"Pass {i}: {' '.join(group)}")
