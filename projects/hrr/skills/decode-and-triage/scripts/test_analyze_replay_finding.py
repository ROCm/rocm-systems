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

    def test_a_gpu_fault_outranks_the_signal_that_killed_the_process(self) -> None:
        """A memory fault aborts the process, so both markers are in the log."""
        text = (
            "Memory access fault by GPU node-2 (Agent handle: 0x1) on address "
            "0x7f91da000000. Reason: Unknown.\n"
            "[triage] replay killed by signal 6\n"
        )
        finding = arf.Finding(outcome="UNKNOWN", fault_class="unknown")
        arf.parse_text(text, "sample.log", finding)
        self.assertEqual(finding.outcome, "MAF")
        self.assertEqual(finding.fault_class, "illegal_memory_access")

    def test_to_dict_json_serializable(self) -> None:
        finding = arf.Finding(outcome="PASS", fault_class="replay_pass")
        payload = json.dumps(finding.to_dict())
        self.assertIn("replay_pass", payload)

    def test_memory_fault_outranks_the_abort_line(self) -> None:
        """A fault surfaced by --sync-after-event still reports as a fault.

        That path makes hrr_playback print its own abort line for the same
        fault the runtime reports, so both lines appear in one log.
        """
        text = (
            "Memory access fault by GPU node-2 (Agent handle: 0x1) on address "
            "0x7f91da000000. Reason: Write access to a read-only page\n"
            "[HRR] Fatal: GPU error after T0 Event 4210 (hipModuleLaunchKernel): "
            "4 (hipErrorLaunchFailure) — aborting\n"
        )
        finding = arf.Finding(outcome="UNKNOWN", fault_class="unknown")
        arf.parse_text(text, "replay.log", finding)
        self.assertEqual(finding.fault_class, "read_only_page_fault")
        self.assertEqual(finding.outcome, "MAF")


class RecordedCaptureTests(unittest.TestCase):
    """Checks against replay output recorded from a gfx950 host.

    The fixtures are unedited tool output, so a change in what hrr-playback
    prints shows up here as a failing test rather than as a silently degraded
    finding.
    """

    FIXTURES = SCRIPT_DIR.parent / "evals" / "fixtures"

    def _analyze(self, *names: str) -> arf.Finding:
        finding = arf.Finding(outcome="UNKNOWN", fault_class="unknown")
        for name in names:
            path = self.FIXTURES / name
            arf.parse_text(path.read_text(encoding="utf-8"), name, finding)
        return arf.finalize(finding)

    def test_clean_replay_implicates_no_kernel(self) -> None:
        finding = self._analyze("replay_pass.log", "info_pass.txt")
        self.assertEqual(finding.outcome, "PASS")
        self.assertEqual(finding.fault_class, "replay_pass")
        self.assertIsNone(finding.kernel_name)
        self.assertIsNone(finding.kernel_family)

    def test_clean_replay_carries_archive_totals(self) -> None:
        finding = self._analyze("replay_pass.log", "info_pass.txt")
        self.assertEqual(finding.archive_complete, "yes")
        self.assertEqual(finding.archive_events, 185469)
        self.assertEqual(finding.archive_kernels, 13233)
        self.assertEqual(finding.d2h_fail, 0)

    def test_truncated_kernel_names_are_ignored(self) -> None:
        """Every name in this capture's table is cut off by the column width."""
        finding = self._analyze("info_pass.txt")
        self.assertEqual(finding.archive_kernel_names, [])

    def test_single_kernel_archive_attributes_the_fault(self) -> None:
        finding = self._analyze("replay_memory_fault.log", "info_crash.txt")
        self.assertEqual(finding.outcome, "MAF")
        self.assertEqual(finding.fault_class, "illegal_memory_access")
        self.assertEqual(finding.fault_address, "0x7f91da000000")
        self.assertEqual(finding.kernel_name, "crash_oob")
        self.assertEqual(finding.archive_complete, "no")
        self.assertTrue(any("inferred from the archive" in n for n in finding.notes))

    def test_a_crashed_replay_implicates_no_kernel(self) -> None:
        """A signal death is a failed run, not a verdict on the workload.

        This archive holds exactly one kernel, so the inference that serves a
        real fault would otherwise hand that kernel over as the culprit for a
        crash of the replay process itself.
        """
        finding = arf.Finding(outcome="UNKNOWN", fault_class="unknown")
        arf.parse_text(
            "[HRR] Replaying 5 events\n[triage] replay killed by signal 11\n",
            "replay.log",
            finding,
        )
        arf.parse_text(
            (self.FIXTURES / "info_crash.txt").read_text(encoding="utf-8"),
            "info_crash.txt",
            finding,
        )
        finding = arf.finalize(finding)
        self.assertEqual(finding.outcome, "ABORT")
        self.assertEqual(finding.fault_class, "replay_crashed")
        self.assertEqual(finding.replay_signal, 11)
        self.assertIsNone(finding.kernel_name)
        self.assertTrue(any("killed by signal 11" in n for n in finding.notes))

    def test_short_memory_fault_form_still_names_the_kernel(self) -> None:
        """Some ROCm builds omit `host:` and start the bracket at `GPU index:`."""
        finding = self._analyze("replay_aten_chevron.log")
        self.assertEqual(finding.outcome, "MAF")
        self.assertEqual(finding.fault_address, "0x7f8c11a00000")
        self.assertIn("mul_kernel_cuda", finding.kernel_name or "")

    def test_aten_chevron_fault_carries_a_caveat_not_a_new_verdict(self) -> None:
        """A `<<<>>>` ATen fault can be genuine, so the verdict stands.

        Capture records the pointers such kernels pass inside by-value structs
        and replay translates them, so reclassifying would suppress a real
        finding. An archive predating that support has none recorded, which
        faults identically, so the ambiguity is flagged instead.
        """
        finding = self._analyze("replay_aten_chevron.log")
        self.assertEqual(finding.fault_class, "illegal_memory_access")
        self.assertTrue(
            any("hipLaunchByPtr" in n for n in finding.notes),
            f"expected the ATen caveat, got {finding.notes}",
        )

    def test_last_launch_attributes_the_failing_event(self) -> None:
        """Per-event lines are the only record when no Fatal line is written."""
        finding = self._analyze("replay_aten_chevron.log")
        self.assertEqual(finding.failing_call_index, 41310)
        self.assertEqual(finding.failing_api, "hipLaunchKernel")

    def test_queue_abort_is_not_a_hang(self) -> None:
        """Recorded from an out-of-bounds ATen gather replayed on gfx950.

        The queue aborts with a hardware exception. Reading that as a hang
        sends the reader after stalled work that never existed.
        """
        finding = self._analyze("replay_queue_abort_aten.log")
        self.assertNotEqual(finding.fault_class, "hang")
        self.assertEqual(finding.fault_class, "replay_aborted")
        self.assertEqual(finding.outcome, "ABORT")
        self.assertTrue(
            any("HSA_STATUS_ERROR_EXCEPTION" in n for n in finding.notes),
            f"expected the raw HSA status in the notes, got {finding.notes}",
        )

    def test_queue_abort_keeps_the_kernel_it_named(self) -> None:
        """That bracket carries a kernel but no faulting address."""
        finding = self._analyze("replay_queue_abort_aten.log")
        self.assertIn("indexSelectSmallIndex", finding.kernel_name or "")
        self.assertEqual(finding.kernel_family, "pytorch_kernel")
        self.assertTrue(
            any("hipLaunchByPtr" in n for n in finding.notes),
            f"expected the ATen caveat, got {finding.notes}",
        )

    def test_a_stopped_replay_reports_a_hang(self) -> None:
        """A hang prints nothing, so the runner's marker is the only signal."""
        finding = self._analyze("replay_hang_timeout.log")
        self.assertEqual(finding.outcome, "HANG")
        self.assertEqual(finding.fault_class, "hang")
        self.assertIsNone(finding.kernel_name)

    def test_progress_kernel_survives_the_info_pass(self) -> None:
        """The `--info` dump has no progress lines and must not erase them."""
        finding = arf.Finding(outcome="UNKNOWN", fault_class="unknown")
        arf.parse_text(
            '[HRR progress] elapsed_s=0.0 seq=4 kernels=1 d2h_pass=0 d2h_fail=0 '
            'd2h_attempted=0 last="crash_oob"\n',
            "replay.log",
            finding,
        )
        arf.parse_text(
            (self.FIXTURES / "info_crash.txt").read_text(encoding="utf-8"),
            "info",
            finding,
        )
        self.assertEqual(finding.last_progress_kernel, "crash_oob")

    def test_unreadable_archive_records_both_versions(self) -> None:
        finding = self._analyze("version_mismatch.log")
        self.assertEqual(finding.fault_class, "version_mismatch")
        self.assertEqual(finding.archive_format_version, 3)
        self.assertEqual(finding.reader_format_version, 4)
        self.assertIsNone(finding.kernel_name)


if __name__ == "__main__":
    unittest.main()
