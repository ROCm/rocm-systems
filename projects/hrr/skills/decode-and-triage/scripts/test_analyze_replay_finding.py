#!/usr/bin/env python3
"""Unit tests for analyze_replay_finding.py (stdlib unittest)."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import analyze_replay_finding as arf  # noqa: E402


class AnalyzeReplayFindingTests(unittest.TestCase):
    def test_read_only_page_fault_from_replay_log(self) -> None:
        text = """
[HRR progress] elapsed_s=10 seq=100 kernels=50 d2h_pass=1 d2h_fail=0 d2h_attempted=1 last="foo"
Memory access fault by GPU node-1 (Agent handle: 0x1) on address 0xdeadbeef.
Reason: Write access to a read-only page
:0:rocdevice.cpp:1: Memory Fault Error [host: x, GPU index: 0, faulting addr: 0xdeadbeef, kernel: Cijk_Test_MT128x192x128_SK3]
"""
        finding = arf.Finding(outcome="UNKNOWN", fault_class="unknown")
        arf.parse_text(text, "sample.log", finding)
        self.assertEqual(finding.outcome, "MAF")
        self.assertEqual(finding.fault_class, "read_only_page_fault")
        self.assertEqual(finding.fault_address, "0xdeadbeef")
        self.assertIn("Cijk_Test", finding.kernel_name or "")

    def test_version_mismatch_from_info(self) -> None:
        text = "[HRR] Version mismatch: file=3 reader=4\n"
        finding = arf.Finding(outcome="UNKNOWN", fault_class="unknown")
        arf.parse_text(text, "info", finding)
        self.assertEqual(finding.fault_class, "version_mismatch")
        self.assertEqual(finding.outcome, "ABORT")
        self.assertTrue(any("wire version 3" in n for n in finding.notes))

    def test_fatal_api_abort(self) -> None:
        text = "[HRR] Fatal: T138 Event 7352 (hipMemcpyWithStream) returned 1 (invalid argument) — aborting replay\n"
        finding = arf.Finding(outcome="UNKNOWN", fault_class="unknown")
        arf.parse_text(text, "replay.log", finding)
        self.assertEqual(finding.fault_class, "replay_fatal_api")
        self.assertEqual(finding.failing_call_index, 7352)
        self.assertEqual(finding.failing_api, "hipMemcpyWithStream")

    def test_maf_not_overwritten_by_later_info_only_parse(self) -> None:
        maf_log = (
            "Memory access fault by GPU node-1 on address 0x1. "
            "Reason: Write access to a read-only page\n"
        )
        info = "Complete: NO\nrecovered 100 events\n"
        finding = arf.Finding(outcome="UNKNOWN", fault_class="unknown")
        arf.parse_text(maf_log, "replay.log", finding)
        arf.parse_text(info, "archive (--info)", finding)
        self.assertEqual(finding.fault_class, "read_only_page_fault")
        self.assertEqual(finding.outcome, "MAF")

    def test_to_dict_json_serializable(self) -> None:
        finding = arf.Finding(outcome="PASS", fault_class="replay_pass")
        payload = json.dumps(finding.to_dict())
        self.assertIn("replay_pass", payload)


if __name__ == "__main__":
    unittest.main()
