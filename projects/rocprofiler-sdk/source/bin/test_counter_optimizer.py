#!/usr/bin/env python3
"""
Unit tests for counter_optimizer.py

Run with: python3 test_counter_optimizer.py
Or with pytest: pytest test_counter_optimizer.py -v
"""

import unittest
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from counter_optimizer import (
    CounterOptimizer,
    normalize_arch,
    detect_architecture,
    BlockUsage,
    CounterGroup,
)


class TestNormalizeArch(unittest.TestCase):
    """Test architecture name normalization."""

    def test_basic_normalization(self):
        self.assertEqual(normalize_arch("gfx90a"), "gfx90a")
        self.assertEqual(normalize_arch("GFX90A"), "gfx90a")
        self.assertEqual(normalize_arch("gfx1100"), "gfx1100")

    def test_prefix_removal(self):
        self.assertEqual(normalize_arch("amd:gfx90a"), "gfx90a")
        self.assertEqual(normalize_arch("AMD:gfx1100"), "gfx1100")

    def test_add_gfx_prefix(self):
        self.assertEqual(normalize_arch("90a"), "gfx90a")
        self.assertEqual(normalize_arch("1100"), "gfx1100")


class TestBlockUsage(unittest.TestCase):
    """Test BlockUsage class."""

    def test_utilization(self):
        usage = BlockUsage(used=4, max=8)
        self.assertEqual(usage.utilization, 0.5)
        self.assertEqual(usage.available, 4)
        self.assertTrue(usage.can_add(3))
        self.assertFalse(usage.can_add(5))

    def test_full_utilization(self):
        usage = BlockUsage(used=8, max=8)
        self.assertEqual(usage.utilization, 1.0)
        self.assertEqual(usage.available, 0)
        self.assertFalse(usage.can_add(1))

    def test_empty_utilization(self):
        usage = BlockUsage(used=0, max=8)
        self.assertEqual(usage.utilization, 0.0)
        self.assertEqual(usage.available, 8)
        self.assertTrue(usage.can_add(8))


class TestCounterGroup(unittest.TestCase):
    """Test CounterGroup class."""

    def test_add_counter(self):
        group = CounterGroup()
        group.add_counter("SQ_WAVES", "SQ", 8)

        self.assertEqual(len(group.counters), 1)
        self.assertEqual(group.counters[0], "SQ_WAVES")
        self.assertIn("SQ", group.block_usage)
        self.assertEqual(group.block_usage["SQ"].used, 1)
        self.assertEqual(group.block_usage["SQ"].max, 8)

    def test_can_add_counter(self):
        group = CounterGroup()

        # Add counters up to limit
        for i in range(8):
            self.assertTrue(group.can_add_counter(f"SQ_{i}", "SQ"))
            group.add_counter(f"SQ_{i}", "SQ", 8)

        # Should not be able to add more
        self.assertFalse(group.can_add_counter("SQ_9", "SQ"))

    def test_multi_block(self):
        group = CounterGroup()
        group.add_counter("SQ_WAVES", "SQ", 8)
        group.add_counter("TCP_STALL", "TCP", 4)

        self.assertEqual(len(group.counters), 2)
        self.assertEqual(len(group.block_usage), 2)
        self.assertEqual(group.block_usage["SQ"].used, 1)
        self.assertEqual(group.block_usage["TCP"].used, 1)


class TestCounterOptimizer(unittest.TestCase):
    """Test CounterOptimizer class."""

    def test_basic_optimization(self):
        """Test that optimizer reduces passes for simple case."""
        optimizer = CounterOptimizer(
            arch="gfx90a",
            hardware_limits={"SQ": 8},
            counter_metadata={f"SQ_A_{i}": "SQ" for i in range(10)},
        )

        # 10 SQ counters, max=8, should need 2 passes
        counters = [f"SQ_A_{i}" for i in range(10)]
        groups, report = optimizer.optimize(counters, original_passes=1)

        self.assertEqual(len(groups), 2)
        self.assertEqual(report.optimized_passes, 2)
        self.assertEqual(sum(len(g) for g in groups), 10)

    def test_multi_block_constraints(self):
        """Test multi-block constraint handling."""
        optimizer = CounterOptimizer(
            arch="gfx90a",
            hardware_limits={"SQ": 8, "TCP": 4},
            counter_metadata={
                **{f"SQ_A_{i}": "SQ" for i in range(5)},
                **{f"TCP_A_{i}": "TCP" for i in range(3)},
            },
        )

        # 5 SQ + 3 TCP should fit in one pass
        counters = [f"SQ_A_{i}" for i in range(5)] + [f"TCP_A_{i}" for i in range(3)]
        groups, report = optimizer.optimize(counters, original_passes=1)

        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0]), 8)
        self.assertEqual(report.optimized_passes, 1)

    def test_multi_block_overflow(self):
        """Test that overflow in one block creates new pass."""
        optimizer = CounterOptimizer(
            arch="gfx90a",
            hardware_limits={"SQ": 8, "TCP": 4},
            counter_metadata={
                **{f"SQ_A_{i}": "SQ" for i in range(10)},
                **{f"TCP_A_{i}": "TCP" for i in range(5)},
            },
        )

        # 10 SQ + 5 TCP will need multiple passes
        counters = [f"SQ_A_{i}" for i in range(10)] + [f"TCP_A_{i}" for i in range(5)]
        groups, report = optimizer.optimize(counters, original_passes=1)

        # Should create at least 2 passes (SQ overflow triggers it)
        self.assertGreaterEqual(len(groups), 2)
        self.assertEqual(sum(len(g) for g in groups), 15)

    def test_unknown_counters(self):
        """Test handling of unknown counters."""
        optimizer = CounterOptimizer(
            arch="gfx90a",
            hardware_limits={"SQ": 8},
            counter_metadata={"SQ_WAVES": "SQ"},
        )

        counters = ["SQ_WAVES", "UNKNOWN_COUNTER", "ANOTHER_UNKNOWN"]
        groups, report = optimizer.optimize(counters, original_passes=1)

        # Should skip unknown counters
        self.assertEqual(len(report.unknown_counters), 2)
        self.assertIn("UNKNOWN_COUNTER", report.unknown_counters)
        self.assertEqual(sum(len(g) for g in groups), 1)  # Only SQ_WAVES

    def test_empty_input(self):
        """Test handling of empty counter list."""
        optimizer = CounterOptimizer(
            arch="gfx90a",
            hardware_limits={"SQ": 8},
            counter_metadata={},
        )

        groups, report = optimizer.optimize([], original_passes=1)

        self.assertEqual(len(groups), 0)
        self.assertEqual(report.input_counters, 0)
        self.assertEqual(report.optimized_passes, 0)

    def test_report_statistics(self):
        """Test optimization report statistics."""
        optimizer = CounterOptimizer(
            arch="gfx90a",
            hardware_limits={"SQ": 8},
            counter_metadata={f"SQ_A_{i}": "SQ" for i in range(16)},
        )

        counters = [f"SQ_A_{i}" for i in range(16)]
        groups, report = optimizer.optimize(counters, original_passes=4)

        self.assertEqual(report.input_counters, 16)
        self.assertEqual(report.original_passes, 4)
        self.assertEqual(report.optimized_passes, 2)  # 16 counters / 8 per pass
        self.assertEqual(report.passes_saved, 2)
        self.assertEqual(report.reduction_percent, 50.0)

    def test_no_optimization_needed(self):
        """Test case where counters already fit in one pass."""
        optimizer = CounterOptimizer(
            arch="gfx90a",
            hardware_limits={"SQ": 8, "TCP": 4},
            counter_metadata={
                "SQ_WAVES": "SQ",
                "SQ_INSTS": "SQ",
                "TCP_STALL": "TCP",
            },
        )

        counters = ["SQ_WAVES", "SQ_INSTS", "TCP_STALL"]
        groups, report = optimizer.optimize(counters, original_passes=1)

        self.assertEqual(len(groups), 1)
        self.assertEqual(report.optimized_passes, 1)
        self.assertEqual(report.passes_saved, 0)

    def test_factory_method(self):
        """Test CounterOptimizer.create() factory method."""
        # Should not raise exception even with unknown arch
        optimizer = CounterOptimizer.create("gfx90a")

        self.assertIsNotNone(optimizer)
        self.assertEqual(optimizer.arch, "gfx90a")
        self.assertIsNotNone(optimizer.hardware_limits)
        self.assertIsInstance(optimizer.counter_metadata, dict)

    def test_realistic_gfx90a_scenario(self):
        """Test realistic counter collection scenario for gfx90a."""
        # Use real metadata for gfx90a
        try:
            optimizer = CounterOptimizer.create("gfx90a")
        except:
            self.skipTest("Counter metadata not available")

        # Real counter names from gfx90a
        counters = [
            "SQ_WAVES",
            "SQ_INSTS_VALU",
            "SQ_INSTS_SALU",
            "SQ_INSTS_VMEM_RD",
            "SQ_INSTS_VMEM_WR",
            "SQ_INSTS_SMEM",
            "SQ_INSTS_FLAT",
            "SQ_INSTS_LDS",
            "SQ_INSTS_GDS",
            "TCP_TCP_TA_DATA_STALL_CYCLES",
            "TA_TA_BUSY",
            "TCC_HIT",
            "TCC_MISS",
        ]

        groups, report = optimizer.optimize(counters, original_passes=1)

        # 9 SQ counters (max 8) should trigger 2 passes
        self.assertGreaterEqual(len(groups), 2)
        self.assertEqual(sum(len(g) for g in groups), len(counters))

        # Verify no counter was lost
        all_optimized = []
        for group in groups:
            all_optimized.extend(group)
        self.assertEqual(sorted(all_optimized), sorted(counters))


class TestDetectArchitecture(unittest.TestCase):
    """Test architecture detection."""

    def test_explicit_arch(self):
        """Test explicit architecture specification."""
        arch = detect_architecture("gfx90a")
        self.assertEqual(arch, "gfx90a")

    def test_normalization(self):
        """Test that detection normalizes input."""
        arch = detect_architecture("GFX1100")
        self.assertEqual(arch, "gfx1100")

    def test_fallback(self):
        """Test fallback when detection fails."""
        # Detection may or may not work depending on system
        arch = detect_architecture()
        self.assertTrue(arch.startswith("gfx"))


def run_tests():
    """Run all tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test cases
    suite.addTests(loader.loadTestsFromTestCase(TestNormalizeArch))
    suite.addTests(loader.loadTestsFromTestCase(TestBlockUsage))
    suite.addTests(loader.loadTestsFromTestCase(TestCounterGroup))
    suite.addTests(loader.loadTestsFromTestCase(TestCounterOptimizer))
    suite.addTests(loader.loadTestsFromTestCase(TestDetectArchitecture))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Return exit code
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_tests())
