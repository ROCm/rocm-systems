#!/usr/bin/env python3

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, replace
import io
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time
import unittest
from unittest import mock

import consan_validation as validation
import consan_llama_validation as llama_validation
from consan_coverage_gate import _COVERAGE_COUNT_FIELDS
from consan_validation_test_support import (
    RETIRED_COVERAGE_OUTPUT_PARSER_CONTRACT,
    RETIRED_TOPK_CODE_OBJECT_FINGERPRINT,
    temporary_root,
)


@dataclass(frozen=True)
class NativeGtestTargetExpectation:
    build_dir: str
    base: str
    matrix: str
    suite: str
    matrix_suite: str
    matrix_operation: str
    d128_block_oracle: str
    d128_block_fault_uses_oracle: bool = False


def moi_auto_report(
    reader: int,
    generation: int,
    *,
    diagnostics: int = 0,
    visible_records: int = 1,
    diagnostic_capacity: int = 1,
    code_object_fingerprint: str = RETIRED_TOPK_CODE_OBJECT_FINGERPRINT,
) -> str:
    return (
        f"ConSan MOI auto report reader={reader} addr=0x1000 bytes=4096 "
        f"generation={generation} "
        f"code_object={code_object_fingerprint} "
        f"diagnostics={diagnostics} visible_records={visible_records} "
        f"diagnostic_capacity={diagnostic_capacity} "
        "visible_barriers=0 visible_atomics=0 visible_fences=0 "
        "sampled_conflicts=0 sampled_immediate_conflicts=0"
    )


def complete_coverage_log(*extra_lines: str) -> str:
    counts = {name: 0 for name in _COVERAGE_COUNT_FIELDS}
    for name in (
        "access_discovered",
        "access_supported",
        "access_selected",
        "access_patched",
    ):
        counts[name] = 1
    coverage = " ".join(f"{name}={counts[name]}" for name in _COVERAGE_COUNT_FIELDS)
    return "\n".join(
        (
            "[rocjitsu-dbi-hooks] ConSan coverage reader=7 flavor=moi "
            "engine=record_replay analysis_complete=true expert_limit=false "
            f"{coverage}",
            "[rocjitsu-dbi-hooks] ConSan analysis verdict applicable=true "
            "analysis_complete=true static_complete=true dynamic_complete=true "
            "applicable_code_objects=1 incomplete_code_objects=0 "
            "access=1/1 barrier=0/0 atomic=0/0 fence=0/0 "
            "dynamic_incomplete=0 replay_unsupported_access=0 "
            "replay_unsupported_atomics=0 replay_unsupported_fences=0 "
            "replay_metadata_full=0",
            moi_auto_report(7, 1),
            *extra_lines,
        )
    )


def moi_auto_replay(
    reader: int,
    generation: int,
    diagnostics: int,
    *,
    conflict: bool | None = None,
    metadata_full: bool = False,
    capacity_exhausted: bool = False,
    diagnostic_capacity: int = 1,
    provenance_repaired: int = 0,
    provenance_unresolved: int = 0,
    code_object_fingerprint: str = RETIRED_TOPK_CODE_OBJECT_FINGERPRINT,
) -> str:
    if conflict is None:
        conflict = diagnostics != 0
    return (
        f"ConSan MOI auto replay reader={reader} generation={generation} "
        f"code_object={code_object_fingerprint} diagnostics={diagnostics} "
        f"conflict={'true' if conflict else 'false'} "
        f"metadata_full={'true' if metadata_full else 'false'} "
        "diagnostic_capacity_exhausted="
        f"{'true' if capacity_exhausted else 'false'} "
        f"diagnostic_capacity={diagnostic_capacity} "
        f"provenance_repaired={provenance_repaired} "
        f"provenance_unresolved={provenance_unresolved}"
    )


def moi_auto_replay_diagnostic(
    reader: int,
    report_generation: int,
    generation: int,
    index: int,
    *,
    code_object_fingerprint: str = RETIRED_TOPK_CODE_OBJECT_FINGERPRINT,
) -> str:
    return (
        f"ConSan MOI auto replay diagnostic reader={reader} "
        f"report_generation={report_generation} generation={generation} "
        f"code_object={code_object_fingerprint} index={index} kind=1 "
        "first_owner=1 second_owner=2 "
        "first_inst=0xfe96c second_inst=0xfe974 "
        "first_lds_known=true first_lds=[0,4) second_lds=[0,4) "
        "first_kind=2 second_kind=2"
    )


def assert_current_replay_log(test: unittest.TestCase, log_text: str) -> None:
    """Assert that test input already carries every producer-owned contract field."""
    for line in log_text.splitlines():
        if "ConSan MOI auto replay diagnostic reader=" in line:
            for field in ("report_generation=", "generation=", "code_object="):
                test.assertIn(field, line)
        elif (
            "ConSan MOI auto replay reader=" in line and " skipped " not in f" {line} "
        ):
            for field in (
                "generation=",
                "code_object=",
                "diagnostic_capacity=",
                "provenance_repaired=",
            ):
                test.assertIn(field, line)


class ConSanValidationTest(unittest.TestCase):
    def test_launcher_json_is_an_exact_argv_prefix(self) -> None:
        self.assertEqual(
            validation._launcher_from_json(
                '["env", "-u", "HSA_MODEL_LIB", "tool", "--"]'
            ),
            ["env", "-u", "HSA_MODEL_LIB", "tool", "--"],
        )
        self.assertEqual(validation._launcher_from_json(None), [])
        for malformed in ('"tool"', "[]", '["tool", ""]', "not-json"):
            with self.assertRaises(validation.ValidationError):
                validation._launcher_from_json(malformed)

    def test_run_process_timeout_contains_descendants(self) -> None:
        parent = (
            "import subprocess,sys,time; "
            "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
            "print(child.pid, flush=True); time.sleep(60)"
        )
        with temporary_root() as root:
            returncode, _, output = validation._run_process(
                [sys.executable, "-c", parent], os.environ.copy(), root / "run.log", 1
            )
        self.assertEqual(returncode, 124)
        self.assertIn("validation timeout after 1s", output)
        child_pid = int(output.splitlines()[0])
        for _ in range(20):
            try:
                state = (
                    Path(f"/proc/{child_pid}/stat")
                    .read_text(encoding="utf-8")
                    .split()[2]
                )
            # procfs can surface ESRCH if the task disappears during open.
            except (FileNotFoundError, ProcessLookupError):
                break
            if state == "Z":
                break
            time.sleep(0.05)
        else:
            self.fail(f"timed-out descendant {child_pid} remained runnable")

    def test_run_process_timeout_does_not_wait_for_an_escaped_output_pipe(
        self,
    ) -> None:
        parent = (
            "import subprocess,time; "
            "child=subprocess.Popen(['sleep','60'], start_new_session=True); "
            "print(child.pid, flush=True); time.sleep(60)"
        )
        child_pid = None
        try:
            with (
                temporary_root() as root,
                mock.patch.object(
                    validation,
                    "PROCESS_TERMINATION_GRACE_SECONDS",
                    0.1,
                ),
                mock.patch.object(
                    validation,
                    "PROCESS_OUTPUT_DRAIN_SECONDS",
                    0.1,
                ),
            ):
                started = time.monotonic()
                returncode, _, output = validation._run_process(
                    [sys.executable, "-c", parent],
                    os.environ.copy(),
                    root / "run.log",
                    1,
                )
                elapsed = time.monotonic() - started
            child_pid = int(output.splitlines()[0])
            self.assertEqual(returncode, 124)
            self.assertLess(elapsed, 2)
            self.assertIn("validation timeout after 1s", output)
        finally:
            if child_pid is not None:
                try:
                    os.killpg(child_pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

    def test_coverage_summary_rejects_static_analysis_incompleteness(self) -> None:
        counts = {name: 0 for name in _COVERAGE_COUNT_FIELDS}
        counts["access_discovered"] = 1
        counts["access_unsupported"] = 1
        coverage = " ".join(f"{name}={counts[name]}" for name in _COVERAGE_COUNT_FIELDS)
        log = "\n".join(
            (
                "[rocjitsu-dbi-hooks] ConSan coverage reader=1 flavor=moi engine=record_replay "
                f"analysis_complete=false expert_limit=false {coverage}",
                "[rocjitsu-dbi-hooks] ConSan coverage_site reader=1 kind=access "
                "disposition=unsupported reason=unsupported_mnemonic outcome=unsupported "
                "lowering_reason=semantic_unsupported resource_reason=none container=k "
                "scope=kernel text=0x0 mnemonic=ds_unknown",
                "[rocjitsu-dbi-hooks] ConSan analysis verdict applicable=true "
                "analysis_complete=false "
                "static_complete=false dynamic_complete=true applicable_code_objects=1 "
                "incomplete_code_objects=1 access=0/0 barrier=0/0 atomic=0/0 fence=0/0 "
                "visible_evidence=0 dynamic_incomplete=0 replay_unsupported_access=0 "
                "replay_unsupported_atomics=0 replay_unsupported_fences=0 "
                "replay_metadata_full=0",
            )
        )
        summary = validation._coverage_summary(log)
        self.assertFalse(summary["accepted"])
        self.assertIn("analysis incomplete", summary["reasons"])

    def test_coverage_summary_preserves_strict_load_rejection(self) -> None:
        summary = validation._coverage_summary(
            "[rocjitsu-dbi-hooks] ConSan load rejection reader=73 "
            "reason=transform-error status=4112 policy=strict action=terminate "
            "exit_code=92\n"
        )
        self.assertFalse(summary["accepted"])
        self.assertEqual(
            summary["error"], "ConSan rejected a code object before execution"
        )
        self.assertEqual(
            summary["load_rejection"],
            {
                "reader": "73",
                "reason": "transform-error",
                "status": "4112",
                "policy": "strict",
                "action": "terminate",
                "exit_code": "92",
            },
        )

    def test_coverage_output_diagnostic_inventory_accepts_declared_conflicts(
        self,
    ) -> None:
        contract = RETIRED_COVERAGE_OUTPUT_PARSER_CONTRACT
        summary = validation._coverage_output_diagnostic_summary(
            "\n".join(
                (
                    moi_auto_report(7, 1),
                    moi_auto_replay(7, 1, 2, provenance_repaired=2),
                    moi_auto_replay_diagnostic(7, 1, 91, 0),
                    moi_auto_replay_diagnostic(7, 1, 92, 1)
                    .replace(
                        "first_owner=1 second_owner=2", "first_owner=3 second_owner=4"
                    )
                    .replace("first_inst=0xfe96c", "first_inst=0xfe974")
                    .replace("second_inst=0xfe974", "second_inst=0xfe97c")
                    .replace("first_lds=[0,4)", "first_lds=[4,8)")
                    .replace("second_lds=[0,4)", "second_lds=[4,8)"),
                )
            ),
            contract,
        )

        self.assertTrue(summary["accepted"])
        self.assertEqual(summary["replay_count"], 2)
        self.assertEqual(summary["observed_signatures"], ["exact-lds-write-write"])
        self.assertEqual(summary["pre_replay_count"], 0)
        self.assertEqual(len(summary["records"]), 2)
        self.assertEqual(summary["records"][0]["first_lds"], "[0,4)")
        self.assertEqual(summary["records"][0]["second_lds"], "[0,4)")

    def test_coverage_output_diagnostic_inventory_rejects_unbounded_output(
        self,
    ) -> None:
        contract = RETIRED_COVERAGE_OUTPUT_PARSER_CONTRACT
        fixture = (
            Path(__file__).with_name("fixtures")
            / "gfx1201_topk_record_replay_current_diagnostics.log"
        ).read_text(encoding="utf-8")
        baseline = validation._coverage_output_diagnostic_summary(fixture, contract)
        self.assertTrue(baseline["accepted"], baseline["reasons"])
        lines = fixture.splitlines()
        pre_report = lines[0]
        replay_summary = lines[1]
        details = lines[2:]
        cases = {
            "unexpected kind": (
                fixture.replace("kind=1", "kind=2", 1),
                "unexpected diagnostics=metadata-full",
            ),
            "unknown numeric kind": (
                fixture.replace("kind=1", "kind=7", 1),
                "unexpected diagnostics=unknown-7",
            ),
            "missing kind": (
                fixture.replace("kind=1 ", "", 1),
                "unexpected diagnostics=malformed",
            ),
            "different access conflict": (
                fixture.replace("second_lds=[2688,2692)", "second_lds=[2692,2696)", 1),
                "unexpected diagnostics=access-conflict",
            ),
            "same-owner access conflict": (
                fixture.replace(
                    "first_owner=5 second_owner=2", "first_owner=5 second_owner=5", 1
                ),
                "unexpected diagnostics=access-conflict",
            ),
            "missing detail": (
                "\n".join((pre_report, replay_summary, *details[:-1])),
                "replay diagnostic inventory incomplete",
            ),
            "pre-replay diagnostic": (
                fixture.replace(
                    "diagnostics=0 visible_diagnostics",
                    "diagnostics=1 visible_diagnostics",
                    1,
                ),
                "pre-replay diagnostics=1",
            ),
            "missing replay summary": (
                "\n".join((pre_report, *details)),
                "missing replay summaries",
            ),
            "capacity exhausted": (
                fixture.replace(
                    "diagnostic_capacity_exhausted=false",
                    "diagnostic_capacity_exhausted=true",
                    1,
                ),
                "replay diagnostic capacity status invalid",
            ),
            "metadata full": (
                fixture.replace("metadata_full=false", "metadata_full=true", 1),
                "replay metadata status invalid",
            ),
            "unresolved provenance": (
                fixture.replace(
                    "provenance_unresolved=0", "provenance_unresolved=1", 1
                ),
                "replay provenance unresolved",
            ),
            "repaired provenance exceeds diagnostics": (
                fixture.replace("provenance_repaired=3", "provenance_repaired=4", 1),
                "replay provenance repaired exceeds diagnostics",
            ),
            "missing conflict status": (
                fixture.replace("conflict=true ", "", 1),
                "replay conflict status invalid",
            ),
            "malformed exact range": (
                fixture.replace("first_lds=[2688,2692)", "first_lds=bad", 1),
                "unexpected diagnostics=access-conflict",
            ),
            "outside instruction groups": (
                fixture.replace("first_inst=0xfe96c", "first_inst=0xfe95c", 1),
                "unexpected diagnostic sites",
            ),
            "spans qualified instruction groups": (
                fixture.replace("second_inst=0xfe974", "second_inst=0xfea70", 1),
                "unexpected diagnostic sites",
            ),
            "duplicate replay summary": (
                "\n".join((pre_report, replay_summary, replay_summary, *details)),
                "duplicate replay summary",
            ),
            "duplicate pre-replay summary": (
                "\n".join((pre_report, pre_report, replay_summary, *details)),
                "duplicate pre-replay summary",
            ),
            "malformed replay summary": (
                fixture.replace("diagnostics=3 ", "diagnostic_total=3 ", 1),
                "malformed replay diagnostic summary",
            ),
            "malformed explicit replay generation": (
                "\n".join(
                    (
                        pre_report,
                        replay_summary.replace("generation=3", "generation=bad", 1),
                        *details,
                    )
                ),
                "malformed replay diagnostic summary",
            ),
            "missing replay generation": (
                "\n".join(
                    (
                        pre_report,
                        replay_summary.replace("generation=3 ", "", 1),
                        *details,
                    )
                ),
                "malformed replay diagnostic summary",
            ),
            "malformed pre-replay summary": (
                fixture.replace("sampled_conflicts=0 ", "", 1),
                "malformed pre-replay diagnostic summary",
            ),
            "detail without summary": (
                "\n".join(
                    (
                        pre_report,
                        replay_summary,
                        *details,
                        details[0]
                        .replace("reader=1679495808", "reader=9", 1)
                        .replace("report_generation=3", "report_generation=2", 1),
                    )
                ),
                "replay diagnostic detail has no summary",
            ),
            "missing diagnostic generation": (
                fixture.replace(
                    "report_generation=3 generation=3", "report_generation=3", 1
                ),
                "replay diagnostic detail has malformed generation",
            ),
            "sampled conflict": (
                fixture.replace("sampled_conflicts=0", "sampled_conflicts=1", 1),
                "pre-replay sampled conflicts=1",
            ),
            "malformed explicit detail generation": (
                fixture.replace("report_generation=3", "report_generation=bad", 1),
                "replay diagnostic detail has malformed identity",
            ),
            "duplicate diagnostic index": (
                fixture.replace("index=1 kind=1", "index=0 kind=1", 1),
                "replay diagnostic indices invalid",
            ),
            "missing diagnostic index": (
                fixture.replace("index=0 kind=1", "kind=1", 1),
                "replay diagnostic indices invalid",
            ),
            "noncontiguous diagnostic indices": (
                fixture.replace("index=2 kind=1", "index=3 kind=1", 1),
                "replay diagnostic indices invalid",
            ),
            "runtime diagnostic capacity mismatch": (
                fixture.replace(
                    "diagnostic_capacity=18794",
                    "diagnostic_capacity=18793",
                    1,
                ),
                "replay diagnostic capacity mismatch",
            ),
            "code-object fingerprint drift": (
                fixture.replace(
                    RETIRED_TOPK_CODE_OBJECT_FINGERPRINT,
                    "fnv1a64:0000000000000000",
                ),
                "diagnostic code-object fingerprint does not match contract",
            ),
            "replay skipped": (
                "\n".join(
                    (
                        pre_report,
                        "ConSan MOI auto replay reader=1679495808 generation=3 "
                        f"code_object={RETIRED_TOPK_CODE_OBJECT_FINGERPRINT} skipped "
                        "required_shadow_entries=1048577 limit=1048576",
                    )
                ),
                "replay skipped",
            ),
        }
        for name, (log, reason) in cases.items():
            with self.subTest(name=name):
                summary = validation._coverage_output_diagnostic_summary(log, contract)
                self.assertFalse(summary["accepted"])
                self.assertTrue(
                    any(reason in item for item in summary["reasons"]),
                    summary["reasons"],
                )

    def test_coverage_output_diagnostic_inventory_is_accounted_per_reader(
        self,
    ) -> None:
        contract = RETIRED_COVERAGE_OUTPUT_PARSER_CONTRACT
        summary = validation._coverage_output_diagnostic_summary(
            "\n".join(
                (
                    moi_auto_report(7, 1),
                    moi_auto_report(8, 2),
                    moi_auto_replay(7, 1, 1),
                    moi_auto_replay(8, 2, 0),
                    moi_auto_replay_diagnostic(8, 2, 2, 0),
                )
            ),
            contract,
        )
        self.assertFalse(summary["accepted"])
        self.assertTrue(
            any(
                "reader=7,generation=1, reported=1, detailed=0" in reason
                for reason in summary["reasons"]
            ),
            summary["reasons"],
        )
        self.assertTrue(
            any(
                "reader=8,generation=2, reported=0, detailed=1" in reason
                for reason in summary["reasons"]
            ),
            summary["reasons"],
        )

    def test_coverage_output_contract_fingerprint_applies_only_to_diagnostics(
        self,
    ) -> None:
        contract = RETIRED_COVERAGE_OUTPUT_PARSER_CONTRACT
        clean_fingerprint = "fnv1a64:0000000000000001"
        summary = validation._coverage_output_diagnostic_summary(
            "\n".join(
                (
                    moi_auto_report(7, 1),
                    moi_auto_replay(7, 1, 1, provenance_repaired=1),
                    moi_auto_replay_diagnostic(7, 1, 91, 0),
                    moi_auto_report(8, 2, code_object_fingerprint=clean_fingerprint),
                    moi_auto_replay(
                        8,
                        2,
                        0,
                        code_object_fingerprint=clean_fingerprint,
                    ),
                )
            ),
            contract,
        )
        self.assertTrue(summary["accepted"], summary["reasons"])
        self.assertEqual(
            summary["observed_code_object_fingerprints"],
            [clean_fingerprint, RETIRED_TOPK_CODE_OBJECT_FINGERPRINT],
        )

    def test_coverage_output_requires_contracted_object_even_without_diagnostics(
        self,
    ) -> None:
        contract = RETIRED_COVERAGE_OUTPUT_PARSER_CONTRACT
        other_fingerprint = "fnv1a64:0000000000000001"
        summary = validation._coverage_output_diagnostic_summary(
            "\n".join(
                (
                    moi_auto_report(8, 2, code_object_fingerprint=other_fingerprint),
                    moi_auto_replay(
                        8,
                        2,
                        0,
                        code_object_fingerprint=other_fingerprint,
                    ),
                )
            ),
            contract,
        )
        self.assertFalse(summary["accepted"])
        self.assertIn(
            "contract code-object fingerprint missing from diagnostic summaries: "
            f"expected={RETIRED_TOPK_CODE_OBJECT_FINGERPRINT}",
            summary["reasons"],
        )

    def test_coverage_output_requires_replay_for_each_nonempty_report(
        self,
    ) -> None:
        contract = RETIRED_COVERAGE_OUTPUT_PARSER_CONTRACT
        summary = validation._coverage_output_diagnostic_summary(
            "\n".join(
                (
                    moi_auto_report(7, 1),
                    moi_auto_report(8, 2),
                    moi_auto_replay(7, 1, 0),
                )
            ),
            contract,
        )
        self.assertFalse(summary["accepted"])
        self.assertTrue(
            any(
                "missing replay summaries: reader=8,generation=2" in reason
                for reason in summary["reasons"]
            ),
            summary["reasons"],
        )

    def test_coverage_output_allows_zero_event_report_without_replay(
        self,
    ) -> None:
        contract = RETIRED_COVERAGE_OUTPUT_PARSER_CONTRACT
        summary = validation._coverage_output_diagnostic_summary(
            "\n".join(
                (
                    moi_auto_report(7, 1),
                    moi_auto_report(8, 2, visible_records=0),
                    moi_auto_replay(7, 1, 0),
                )
            ),
            contract,
        )
        self.assertTrue(summary["accepted"], summary["reasons"])

    def test_coverage_output_identity_includes_report_generation(self) -> None:
        contract = RETIRED_COVERAGE_OUTPUT_PARSER_CONTRACT
        summary = validation._coverage_output_diagnostic_summary(
            "\n".join(
                (
                    moi_auto_report(7, 1),
                    moi_auto_report(7, 2),
                    moi_auto_replay(7, 1, 0),
                    moi_auto_replay(7, 2, 0),
                )
            ),
            contract,
        )
        self.assertTrue(summary["accepted"], summary["reasons"])
        self.assertEqual(
            set(summary["readers"]),
            {"reader=7,generation=1", "reader=7,generation=2"},
        )

    def test_coverage_output_detail_uses_report_not_record_generation(self) -> None:
        contract = RETIRED_COVERAGE_OUTPUT_PARSER_CONTRACT
        summary = validation._coverage_output_diagnostic_summary(
            "\n".join(
                (
                    moi_auto_report(7, 1),
                    moi_auto_replay(7, 1, 1, provenance_repaired=1),
                    moi_auto_replay_diagnostic(7, 1, 99, 0),
                )
            ),
            contract,
        )
        self.assertTrue(summary["accepted"], summary["reasons"])
        self.assertEqual(summary["records"][0]["report_generation"], 1)
        self.assertEqual(summary["records"][0]["generation"], 99)

    def test_coverage_output_rejects_replay_without_matching_report(
        self,
    ) -> None:
        contract = RETIRED_COVERAGE_OUTPUT_PARSER_CONTRACT
        summary = validation._coverage_output_diagnostic_summary(
            "\n".join(
                (
                    moi_auto_report(7, 1),
                    moi_auto_replay(7, 1, 0),
                    moi_auto_replay(8, 2, 0),
                )
            ),
            contract,
        )
        self.assertFalse(summary["accepted"])
        self.assertTrue(
            any(
                "unexpected replay summaries: reader=8,generation=2" in reason
                for reason in summary["reasons"]
            ),
            summary["reasons"],
        )

    def test_coverage_output_diagnostic_inventory_enforces_maximum(self) -> None:
        contract = RETIRED_COVERAGE_OUTPUT_PARSER_CONTRACT
        reader_counts = ((7, 1, 3), (8, 2, 2))
        lines = []
        for reader, generation, count in reader_counts:
            lines.extend(
                (
                    moi_auto_report(reader, generation),
                    moi_auto_replay(
                        reader,
                        generation,
                        count,
                        provenance_repaired=count,
                    ),
                )
            )
            lines.extend(
                moi_auto_replay_diagnostic(reader, generation, generation, index)
                .replace("first_lds=[0,4)", f"first_lds=[{index},{index + 1})")
                .replace("second_lds=[0,4)", f"second_lds=[{index},{index + 1})")
                for index in range(count)
            )
        summary = validation._coverage_output_diagnostic_summary(
            "\n".join(lines),
            contract,
        )
        self.assertFalse(summary["accepted"])
        self.assertTrue(
            any("exceed declared maximum" in reason for reason in summary["reasons"]),
            summary["reasons"],
        )

    def test_coverage_output_rejects_hostile_report_count_without_expanding_it(
        self,
    ) -> None:
        contract = RETIRED_COVERAGE_OUTPUT_PARSER_CONTRACT
        summary = validation._coverage_output_diagnostic_summary(
            "\n".join(
                (
                    moi_auto_report(7, 1),
                    moi_auto_replay(7, 1, 1000000000000000000),
                )
            ),
            contract,
        )
        self.assertFalse(summary["accepted"])
        self.assertTrue(
            any("exceed declared maximum" in reason for reason in summary["reasons"]),
            summary["reasons"],
        )

    def test_coverage_output_current_summary_requires_diagnostic_capacity(
        self,
    ) -> None:
        contract = RETIRED_COVERAGE_OUTPUT_PARSER_CONTRACT
        summary = validation._coverage_output_diagnostic_summary(
            "\n".join(
                (
                    moi_auto_report(7, 1),
                    "ConSan MOI auto replay reader=7 generation=1 "
                    f"code_object={RETIRED_TOPK_CODE_OBJECT_FINGERPRINT} "
                    "diagnostics=0 conflict=false metadata_full=false "
                    "diagnostic_capacity_exhausted=false provenance_repaired=0 "
                    "provenance_unresolved=0",
                )
            ),
            contract,
        )
        self.assertFalse(summary["accepted"])
        self.assertTrue(
            any(
                "replay diagnostic capacity mismatch" in reason
                for reason in summary["reasons"]
            ),
            summary["reasons"],
        )

    def test_record_replay_parser_normalizes_producer_output(self) -> None:
        parsed = validation._parse_record_replay_diagnostic_output(
            "\n".join(
                (
                    moi_auto_report(7, 1),
                    moi_auto_replay(7, 1, 1, provenance_repaired=1),
                    moi_auto_replay_diagnostic(7, 1, 99, 0),
                )
            )
        )

        self.assertEqual(parsed.profile, "record-replay")
        self.assertEqual(parsed.structural_reasons, ())
        self.assertEqual(parsed.diagnostic_count, 1)
        self.assertEqual(parsed.sources[0].identity, "reader=7,generation=1")
        self.assertEqual(
            dict(parsed.sources[0].artifact_fields)["provenance_repaired"], 1
        )
        self.assertEqual(parsed.records[0].signature, "exact-lds-write-write")
        self.assertEqual(parsed.records[0].first_instruction, 0xFE96C)
        self.assertEqual(parsed.records[0].second_instruction, 0xFE974)
        self.assertEqual(parsed.records[0].first_instruction_raw, "0xfe96c")
        self.assertEqual(parsed.records[0].second_instruction_raw, "0xfe974")
        self.assertEqual(dict(parsed.records[0].artifact_fields)["generation"], 99)

    def test_diagnostic_output_dispatch_fails_closed(self) -> None:
        with self.assertRaisesRegex(
            validation.ValidationError,
            "no complete diagnostic-output parser for profile: sampled",
        ):
            validation._diagnostic_output_summary("", "sampled")
        with self.assertRaisesRegex(
            validation.ValidationError,
            "diagnostic-output contract profile mismatch",
        ):
            validation._diagnostic_output_summary(
                "",
                "record-replay",
                replace(
                    RETIRED_COVERAGE_OUTPUT_PARSER_CONTRACT,
                    profile="sampled",
                ),
            )

    def test_shared_diagnostic_evaluator_applies_clean_or_contract_policy(
        self,
    ) -> None:
        contract = RETIRED_COVERAGE_OUTPUT_PARSER_CONTRACT
        parsed = validation._parse_record_replay_diagnostic_output(
            "\n".join(
                (
                    moi_auto_report(7, 1),
                    moi_auto_replay(7, 1, 1, provenance_repaired=1),
                    moi_auto_replay_diagnostic(7, 1, 99, 0),
                )
            )
        )

        clean = validation._evaluate_diagnostic_output(
            parsed,
            validation.DiagnosticPolicy.clean(),
        )
        declared = validation._evaluate_diagnostic_output(
            parsed,
            validation.DiagnosticPolicy.from_contract(contract),
        )

        self.assertFalse(clean["accepted"])
        self.assertIn(
            "replay diagnostics exceed declared maximum: observed=1, maximum=0",
            clean["reasons"],
        )
        self.assertTrue(declared["accepted"], declared["reasons"])

    def test_shared_diagnostic_evaluator_owns_provenance_policy(self) -> None:
        parsed = validation._parse_record_replay_diagnostic_output(
            "\n".join(
                (
                    moi_auto_report(7, 1),
                    moi_auto_replay(7, 1, 0, provenance_unresolved=1),
                )
            )
        )

        self.assertIn(
            "replay provenance unresolved: reader=7,generation=1, count=1",
            parsed.structural_reasons,
        )
        summary = validation._evaluate_diagnostic_output(
            parsed,
            validation.DiagnosticPolicy.clean(),
        )
        self.assertFalse(summary["accepted"])
        self.assertIn(
            "replay provenance unresolved: reader=7,generation=1, count=1",
            summary["reasons"],
        )

    def test_ordinary_record_replay_validates_zero_diagnostic_structure(
        self,
    ) -> None:
        valid_log = complete_coverage_log(moi_auto_replay(7, 1, 0))
        valid = validation._coverage_summary(
            valid_log,
            profile="record-replay",
        )
        malformed = validation._coverage_summary(
            valid_log.replace("diagnostic_capacity=1 ", "", 1),
            profile="record-replay",
        )

        self.assertTrue(valid["accepted"], valid["reasons"])
        self.assertEqual(valid["diagnostics"]["policy"]["kind"], "clean")
        self.assertEqual(valid["diagnostics"]["replay_count"], 0)
        self.assertFalse(malformed["accepted"])
        self.assertTrue(
            any(
                "replay diagnostic capacity mismatch" in reason
                for reason in malformed["reasons"]
            ),
            malformed["reasons"],
        )

    def test_record_replay_capacity_is_clamped_to_visible_records(self) -> None:
        report = moi_auto_report(
            7,
            1,
            visible_records=3,
            diagnostic_capacity=18794,
        )
        valid_log = complete_coverage_log(
            moi_auto_replay(7, 1, 0, diagnostic_capacity=3)
        ).replace(moi_auto_report(7, 1), report)
        invalid_log = valid_log.replace(
            "diagnostic_capacity=3 ",
            "diagnostic_capacity=18794 ",
            1,
        )

        valid = validation._coverage_summary(valid_log, profile="record-replay")
        invalid = validation._coverage_summary(invalid_log, profile="record-replay")

        self.assertTrue(valid["accepted"], valid["reasons"])
        self.assertFalse(invalid["accepted"])
        self.assertTrue(
            any(
                "replay diagnostic capacity mismatch" in reason
                and "expected=3" in reason
                for reason in invalid["reasons"]
            ),
            invalid["reasons"],
        )

    def test_ordinary_record_replay_allows_an_empty_report_without_replay(
        self,
    ) -> None:
        summary = validation._coverage_summary(
            complete_coverage_log().replace("visible_records=1", "visible_records=0"),
            profile="record-replay",
        )

        self.assertTrue(summary["accepted"], summary["reasons"])
        self.assertEqual(summary["diagnostics"]["readers"], {})

    def test_ordinary_record_replay_rejects_replay_for_an_empty_report(
        self,
    ) -> None:
        summary = validation._coverage_summary(
            complete_coverage_log(moi_auto_replay(7, 1, 0)).replace(
                "visible_records=1", "visible_records=0"
            ),
            profile="record-replay",
        )

        self.assertFalse(summary["accepted"])
        self.assertIn(
            "unexpected replay summaries: reader=7,generation=1",
            summary["reasons"],
        )

    def test_record_replay_rejects_duplicate_summary_fields(self) -> None:
        log = complete_coverage_log(moi_auto_replay(7, 1, 0))
        for duplicated in (
            log.replace("diagnostics=0 ", "diagnostics=999 diagnostics=0 ", 1),
            log.replace(
                "diagnostics=0 conflict=false",
                "diagnostics=999 diagnostics=0 conflict=false",
                1,
            ),
        ):
            with self.subTest(line=duplicated.splitlines()[-1]):
                summary = validation._coverage_summary(
                    duplicated, profile="record-replay"
                )
                self.assertFalse(summary["accepted"])
                self.assertTrue(
                    any(
                        "duplicate field 'diagnostics'" in reason
                        for reason in summary["reasons"]
                    ),
                    summary["reasons"],
                )

    def test_record_replay_classifies_report_degradation_messages(self) -> None:
        messages = {
            "needs hsa_memory_copy for coarse-grained summary": (
                "pre-replay report requires hsa_memory_copy: reader=7"
            ),
            "hsa_memory_copy failed status=1": (
                "pre-replay hsa_memory_copy failed: reader=7, status=1"
            ),
            "has invalid header magic=0x0 abi=1 header_size=2": (
                "pre-replay report header invalid: reader=7"
            ),
            "has inconsistent ABI-v2 layout": (
                "pre-replay report layout inconsistent: reader=7"
            ),
        }
        for suffix, expected in messages.items():
            with self.subTest(suffix=suffix):
                parsed = validation._parse_record_replay_diagnostic_output(
                    f"ConSan MOI auto report reader=7 {suffix}"
                )
                self.assertIn(expected, parsed.structural_reasons)
                self.assertNotIn(
                    "malformed pre-replay diagnostic summary",
                    parsed.structural_reasons,
                )

    def test_record_replay_skip_is_one_precise_structural_failure(self) -> None:
        fixture = (
            Path(__file__).with_name("fixtures")
            / "gfx1201_topk_record_replay_current_runtime.log"
        ).read_text(encoding="utf-8")
        assert_current_replay_log(self, fixture)
        replay_line = next(
            line
            for line in fixture.splitlines()
            if "ConSan MOI auto replay reader=" in line
        )
        skipped = (
            "[rocjitsu-dbi-hooks] ConSan MOI auto replay "
            "reader=725954112 generation=3 "
            f"code_object={RETIRED_TOPK_CODE_OBJECT_FINGERPRINT} skipped "
            "required_shadow_entries=1048577 limit=1048576"
        )
        summary = validation._diagnostic_output_summary(
            fixture.replace(replay_line, skipped),
            "record-replay",
        )

        self.assertFalse(summary["accepted"])
        replay_reasons = [reason for reason in summary["reasons"] if "replay" in reason]
        self.assertEqual(
            replay_reasons,
            [
                "replay skipped: reader=725954112,generation=3, "
                "required_shadow_entries=1048577, limit=1048576"
            ],
        )

    def test_record_replay_preserves_malformed_instruction_tokens(self) -> None:
        parsed = validation._parse_record_replay_diagnostic_output(
            "\n".join(
                (
                    moi_auto_report(7, 1),
                    moi_auto_replay(7, 1, 1),
                    moi_auto_replay_diagnostic(7, 1, 1, 0).replace(
                        "first_inst=0xfe96c", "first_inst=not-an-offset"
                    ),
                )
            )
        )

        record = validation._diagnostic_record_result(parsed.records[0])
        self.assertIsNone(record["first_instruction"])
        self.assertEqual(record["first_instruction_raw"], "not-an-offset")

    def test_non_replay_profiles_do_not_consume_replay_grammar(self) -> None:
        summary = validation._coverage_summary(
            complete_coverage_log(),
            profile="sampled",
        )

        self.assertTrue(summary["accepted"], summary["reasons"])
        self.assertNotIn("diagnostics", summary)

    def test_coverage_summary_dispatches_through_diagnostic_registry(self) -> None:
        parsed = validation.ParsedDiagnosticOutput(
            profile="sampled",
            sources=(),
            records=(),
            diagnostic_count=0,
            pre_output_count=0,
            structural_reasons=(),
        )
        parser = mock.Mock(return_value=parsed)
        with mock.patch.dict(
            validation.DIAGNOSTIC_OUTPUT_PARSERS,
            {"sampled": parser},
            clear=False,
        ):
            summary = validation._coverage_summary(
                complete_coverage_log(),
                profile="sampled",
            )

        self.assertTrue(summary["accepted"], summary["reasons"])
        self.assertIn("diagnostics", summary)
        parser.assert_called_once()

    def test_diagnostic_policy_requires_qualified_sites(self) -> None:
        parsed = validation._parse_record_replay_diagnostic_output(
            "\n".join(
                (
                    moi_auto_report(7, 1),
                    moi_auto_replay(7, 1, 1),
                    moi_auto_replay_diagnostic(7, 1, 1, 0),
                )
            )
        )
        summary = validation._evaluate_diagnostic_output(
            parsed,
            validation.DiagnosticPolicy(
                diagnostics=("exact-lds-write-write",),
                max_diagnostics=1,
            ),
        )

        self.assertFalse(summary["accepted"])
        self.assertIn(
            "policy declares diagnostics without qualified instruction groups",
            summary["reasons"],
        )

    def test_coverage_summary_applies_clean_or_declared_diagnostic_policy(
        self,
    ) -> None:
        contract = RETIRED_COVERAGE_OUTPUT_PARSER_CONTRACT
        log = complete_coverage_log(
            moi_auto_replay(7, 1, 1),
            moi_auto_replay_diagnostic(7, 1, 1, 0).replace("kind=1", "kind=3", 1),
        )
        normal = validation._coverage_summary(log, profile="record-replay")
        coverage_output = validation._coverage_summary(
            log,
            profile="record-replay",
            coverage_output_contract=contract,
        )

        self.assertFalse(normal["accepted"])
        self.assertIn("diagnostics", normal)
        self.assertFalse(coverage_output["accepted"])
        self.assertEqual(
            coverage_output["diagnostics"]["observed_signatures"],
            ["barrier-divergence"],
        )
        incomplete = validation._coverage_summary(
            log.replace("static_complete=true", "static_complete=false"),
            profile="record-replay",
            coverage_output_contract=contract,
        )
        self.assertIn("static coverage incomplete", incomplete["reasons"])
        self.assertTrue(
            any("barrier-divergence" in reason for reason in incomplete["reasons"]),
            incomplete["reasons"],
        )

    def test_manifest_is_the_complete_north_star_matrix(self) -> None:
        manifest = validation._manifest("gfx1201")
        self.assertEqual(len(manifest["workloads"]), 20)
        self.assertEqual(
            [profile["id"] for profile in manifest["profiles"]],
            list(validation.PROFILE_IDS),
        )
        self.assertEqual(
            len({workload["id"] for workload in manifest["workloads"]}), 20
        )
        workloads = {workload["id"]: workload for workload in manifest["workloads"]}
        self.assertEqual(
            workloads["pytorch-torch-mode"]["targets"],
            ("gfx950", "gfx1250", "gfx1201"),
        )
        self.assertEqual(
            workloads["pytorch-torch-mode"]["coverage_output_contract"], None
        )
        self.assertEqual(workloads["pytorch-torch-mode"]["run_timeout_seconds"], 30)
        self.assertEqual(
            workloads["pytorch-rdna4-compiled-softmax"]["targets"], ("gfx1201",)
        )
        self.assertEqual(
            workloads["pytorch-rdna4-split-softmax"]["targets"], ("gfx1201",)
        )
        self.assertEqual(workloads["pytorch-rdna4-llm-topk"]["targets"], ("gfx1201",))
        self.assertEqual(
            workloads["pytorch-rdna4-llm-topk"]["fault_families"],
            ("barrier-drop",),
        )
        self.assertEqual(
            workloads["pytorch-rdna4-llm-topk"]["run_timeout_seconds"], 120
        )
        self.assertIsNone(
            workloads["pytorch-rdna4-llm-topk"]["coverage_output_contract"]
        )
        self.assertEqual(workloads["pytorch-rdna4-sdpa"]["run_timeout_seconds"], 30)
        self.assertEqual(workloads["pytorch-rdna4-sdpa"]["targets"], ("gfx1201",))
        self.assertEqual(
            workloads["pytorch-torch-histc"]["targets"],
            ("gfx950", "gfx1250", "gfx1201"),
        )
        self.assertEqual(
            workloads["llama-rdna4-mul-mat-vec-q"]["targets"], ("gfx1201",)
        )
        self.assertEqual(workloads["llama-rdna4-rms-norm"]["targets"], ("gfx1201",))

    def test_coverage_output_contract_rejects_unknown_profile(self) -> None:
        workload = replace(
            validation.WORKLOAD_BY_ID["pytorch-torch-mode"],
            coverage_output_contract=validation.CoverageOutputContract(
                profile="sampled",
                diagnostics=("exact-lds-write-write",),
                max_diagnostics=1,
                instruction_groups=((1, 2),),
                code_object_fingerprint=RETIRED_TOPK_CODE_OBJECT_FINGERPRINT,
                tracking_issue="bd-test",
                withhold_fault_qualification=True,
                fault_qualification_withheld_reason="test-only exception",
            ),
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "invalid coverage-output profile for pytorch-torch-mode: sampled",
        ):
            validation._validate_coverage_output_contract(workload)

    def test_coverage_output_contract_validates_every_bound(self) -> None:
        base = RETIRED_COVERAGE_OUTPUT_PARSER_CONTRACT
        cases = {
            "diagnostics": (
                replace(base, diagnostics=("test-only-conflict",)),
                "invalid coverage-output diagnostics",
            ),
            "empty diagnostics": (
                replace(base, diagnostics=()),
                "invalid coverage-output diagnostics",
            ),
            "maximum": (
                replace(base, max_diagnostics=0),
                "max_diagnostics must be positive",
            ),
            "policy limit": (
                replace(
                    base,
                    max_diagnostics=validation.MAX_COVERAGE_OUTPUT_DIAGNOSTICS + 1,
                ),
                "max_diagnostics exceeds policy limit",
            ),
            "groups": (
                replace(base, instruction_groups=((2, 2),)),
                "instruction_groups are invalid",
            ),
            "code object": (
                replace(base, code_object_fingerprint="nightly"),
                "code_object_fingerprint is invalid",
            ),
            "tracking": (
                replace(base, tracking_issue="none"),
                "needs a tracking bead",
            ),
            "withholding reason": (
                replace(base, fault_qualification_withheld_reason=""),
                "needs a fault-withholding reason",
            ),
            "unused withholding reason": (
                replace(base, withhold_fault_qualification=False),
                "has an unused fault-withholding reason",
            ),
        }
        for name, (contract, reason) in cases.items():
            with self.subTest(name=name):
                workload = replace(
                    validation.WORKLOAD_BY_ID["pytorch-rdna4-llm-topk"],
                    coverage_output_contract=contract,
                )
                with self.assertRaisesRegex(RuntimeError, reason):
                    validation._validate_coverage_output_contract(workload)

    def test_coverage_output_contract_keeps_unrelated_profile_faults(self) -> None:
        contract = RETIRED_COVERAGE_OUTPUT_PARSER_CONTRACT
        validation._validate_coverage_output_contract(
            replace(
                validation.WORKLOAD_BY_ID["pytorch-rdna4-llm-topk"],
                coverage_output_contract=contract,
            )
        )
        workload = replace(
            validation.WORKLOAD_BY_ID["pytorch-rdna4-llm-topk"],
            fault_families=(),
            coverage_output_contract=contract,
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "must not suppress unrelated profile fault qualification",
        ):
            validation._validate_coverage_output_contract(workload)

    def test_coverage_output_contract_can_retain_fault_qualification(self) -> None:
        base = RETIRED_COVERAGE_OUTPUT_PARSER_CONTRACT
        contract = replace(
            base,
            withhold_fault_qualification=False,
            fault_qualification_withheld_reason="",
        )
        workload = replace(
            validation.WORKLOAD_BY_ID["pytorch-rdna4-llm-topk"],
            coverage_output_contract=contract,
        )

        validation._validate_coverage_output_contract(workload)
        self.assertIsNone(
            validation._fault_qualification_contract_for_profile(
                workload, "record-replay"
            )
        )
        with mock.patch.dict(
            validation.WORKLOAD_BY_ID,
            {workload.id: workload},
        ):
            audit = validation._explain_contract(
                Path("/workspace"),
                "gfx1201",
                (workload.id,),
                validation.PROFILE_IDS,
                None,
                allow_reference=False,
            )
        self.assertEqual(
            audit["usability_audit"]["fault_qualification_exceptions"],
            [],
        )
        record_replay = next(
            expectation
            for expectation in audit["workloads"][0]["faults"][0][
                "profile_expectations"
            ]
            if expectation["profile"] == "record-replay"
        )
        self.assertEqual(record_replay["disposition"], "applicable")
        self.assertEqual(record_replay["detector"], "REVIEW_REQUIRED")

    def test_no_workload_declares_a_coverage_output_contract(self) -> None:
        contracts = [
            (workload.id, workload.coverage_output_contract.profile)
            for workload in validation.WORKLOADS
            if workload.coverage_output_contract is not None
        ]
        self.assertEqual(contracts, [])

    def test_workload_manifest_validates_every_declared_workload(self) -> None:
        invalid = replace(
            validation.WORKLOAD_BY_ID["pytorch-torch-mode"],
            coverage_output_contract=validation.CoverageOutputContract(
                profile="always",
                diagnostics=("exact-lds-write-write",),
                max_diagnostics=1,
                instruction_groups=((1, 2),),
                code_object_fingerprint=RETIRED_TOPK_CODE_OBJECT_FINGERPRINT,
                tracking_issue="bd-test",
                withhold_fault_qualification=True,
                fault_qualification_withheld_reason="test-only exception",
            ),
        )
        with (
            mock.patch.object(validation, "WORKLOADS", (invalid,)),
            self.assertRaisesRegex(
                RuntimeError,
                "invalid coverage-output profile for pytorch-torch-mode: always",
            ),
        ):
            validation._validate_workload_manifest()

    def test_gfx950_manifest_includes_portable_pytorch_workloads(self) -> None:
        workload_ids = {
            workload["id"] for workload in validation._manifest("gfx950")["workloads"]
        }
        self.assertTrue(
            {
                "pytorch-torch-mode",
                "pytorch-torch-topk",
                "pytorch-torch-sort",
                "pytorch-scatter-reduce",
                "pytorch-torch-histc",
                "pytorch-norm-softmax",
            }.issubset(workload_ids)
        )

    def test_text_manifest_filters_target_specific_workloads(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(validation.main(["--target", "gfx1201", "manifest"]), 0)
        text = output.getvalue()
        self.assertIn("pytorch-rdna4-compiled-softmax", text)
        self.assertIn("pytorch-rdna4-split-softmax", text)
        self.assertIn("pytorch-rdna4-llm-topk", text)
        self.assertIn("pytorch-torch-mode", text)
        self.assertIn("pytorch-rdna4-sdpa", text)
        self.assertIn("pytorch-torch-histc", text)
        self.assertIn("llama-rdna4-mul-mat-vec-q", text)
        self.assertIn("llama-rdna4-rms-norm", text)
        self.assertNotIn("pytorch-tdm-descriptor-add", text)
        self.assertNotIn("tensile-sk-mxf8gemm-explicit", text)

    def test_explain_all_matches_the_target_manifest_workload_set(self) -> None:
        for target in ("gfx942", "gfx1250"):
            with self.subTest(target=target), temporary_root() as workspace:
                output = io.StringIO()
                with (
                    mock.patch.dict(
                        os.environ,
                        {validation.WORKSPACE_ENV: str(workspace)},
                        clear=False,
                    ),
                    redirect_stdout(output),
                ):
                    self.assertEqual(
                        validation.main(["--target", target, "explain", "--json"]),
                        0,
                    )
                explained = json.loads(output.getvalue())
                self.assertEqual(
                    [workload["id"] for workload in explained["workloads"]],
                    [
                        workload["id"]
                        for workload in validation._manifest(target)["workloads"]
                    ],
                )

    def test_workload_selection_preserves_all_or_resolves_one_row(self) -> None:
        aggregate = validation._resolve_workload_selection(
            validation._parse_args(["--target", "gfx950", "doctor"]),
            allow_all=True,
        )
        concrete = validation._resolve_workload_selection(
            validation._parse_args(
                [
                    "--target",
                    "gfx950",
                    "doctor",
                    "--workload",
                    "d128-block",
                ]
            ),
            allow_all=True,
        )

        self.assertTrue(aggregate.is_all)
        self.assertIsNone(aggregate.workload)
        self.assertIsNone(aggregate.selected_ids())
        self.assertFalse(concrete.is_all)
        self.assertIs(concrete.workload, validation.WORKLOAD_BY_ID["d128-block"])
        self.assertEqual(concrete.selected_ids(), ("d128-block",))

        with self.assertRaisesRegex(
            validation.ValidationError,
            "doctor requires one concrete workload",
        ):
            validation._resolve_workload_selection(
                validation._parse_args(["--target", "gfx950", "doctor"]),
                allow_all=False,
            )
        with self.assertRaisesRegex(
            validation.ValidationError,
            "command requires one concrete workload",
        ):
            aggregate.require_workload()
        with self.assertRaisesRegex(
            validation.ValidationError,
            "gfx942 manifest excludes workload: pytorch-rdna4-compiled-softmax",
        ):
            validation._resolve_workload_selection(
                validation._parse_args(
                    [
                        "--target",
                        "gfx942",
                        "doctor",
                        "--workload",
                        "pytorch-rdna4-compiled-softmax",
                    ]
                ),
                allow_all=True,
            )

    def test_exact_key_validation_reports_missing_and_extra_ids(self) -> None:
        with self.assertRaisesRegex(
            RuntimeError,
            r"matrix mismatch: missing=\['second'\] extra=\['third'\]",
        ):
            validation._validate_exact_keys(
                "matrix",
                {"first": object(), "third": object()},
                ("first", "second"),
            )

    def test_cli_rejects_workload_excluded_by_target_manifest(self) -> None:
        with temporary_root() as workspace:
            commands = (
                ["doctor", "--workload", "pytorch-rdna4-compiled-softmax"],
                ["explain", "--workload", "pytorch-rdna4-compiled-softmax"],
                [
                    "run",
                    "--workload",
                    "pytorch-rdna4-compiled-softmax",
                    "--phase",
                    "clean",
                    "--artifact-root",
                    str(workspace / "run"),
                ],
                [
                    "inventory",
                    "--workload",
                    "pytorch-rdna4-compiled-softmax",
                    "--artifact-root",
                    str(workspace / "inventory"),
                ],
                [
                    "fault",
                    "--workload",
                    "pytorch-rdna4-compiled-softmax",
                    "--spec",
                    str(workspace / "fault.json"),
                    "--fault",
                    "fault-0",
                    "--artifact-root",
                    str(workspace / "fault"),
                ],
            )
            for command in commands:
                with self.subTest(command=command[0]):
                    error = io.StringIO()
                    with (
                        mock.patch.dict(
                            os.environ,
                            {validation.WORKSPACE_ENV: str(workspace)},
                            clear=False,
                        ),
                        redirect_stderr(error),
                    ):
                        self.assertEqual(
                            validation.main(["--target", "gfx942", *command]),
                            2,
                        )
                    self.assertIn(
                        "gfx942 manifest excludes workload: pytorch-rdna4-compiled-softmax",
                        error.getvalue(),
                    )

    def test_cli_reports_target_exclusion_before_missing_workspace(self) -> None:
        error = io.StringIO()
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            redirect_stderr(error),
        ):
            self.assertEqual(
                validation.main(
                    [
                        "--target",
                        "gfx942",
                        "doctor",
                        "--workload",
                        "pytorch-rdna4-compiled-softmax",
                    ]
                ),
                2,
            )
        self.assertIn(
            "gfx942 manifest excludes workload: pytorch-rdna4-compiled-softmax",
            error.getvalue(),
        )

    def test_direct_handlers_report_target_exclusion_before_workspace(self) -> None:
        cases = (
            (
                validation._run,
                [
                    "--target",
                    "gfx942",
                    "run",
                    "--workload",
                    "pytorch-rdna4-compiled-softmax",
                    "--phase",
                    "clean",
                    "--artifact-root",
                    "/unused",
                ],
            ),
            (
                validation._inventory,
                [
                    "--target",
                    "gfx942",
                    "inventory",
                    "--workload",
                    "pytorch-rdna4-compiled-softmax",
                    "--artifact-root",
                    "/unused",
                ],
            ),
            (
                validation._fault,
                [
                    "--target",
                    "gfx942",
                    "fault",
                    "--workload",
                    "pytorch-rdna4-compiled-softmax",
                    "--spec",
                    "/unused/fault.json",
                    "--fault",
                    "fault-0",
                    "--artifact-root",
                    "/unused",
                ],
            ),
        )
        with mock.patch.dict(os.environ, {}, clear=True):
            for handler, argv in cases:
                with self.subTest(command=argv[2]):
                    with self.assertRaisesRegex(
                        validation.ValidationError,
                        "gfx942 manifest excludes workload: "
                        "pytorch-rdna4-compiled-softmax",
                    ):
                        handler(validation._parse_args(argv))

    def test_gfx950_manifest_resolves_cdna4_native_workloads(self) -> None:
        manifest = validation._manifest("gfx950")
        workloads = {workload["id"]: workload for workload in manifest["workloads"]}
        self.assertEqual(
            workloads["d128-block"]["relative_path"],
            (
                "hip-moi-build-gfx950-tests/tests/"
                "hip_moi_instrumented_cdna4_d128_attention_block_test"
            ),
        )
        self.assertEqual(
            workloads["wmma-attention"]["clean_filter"],
            "HipMoiCdna4MfmaAttentionBlock.*",
        )
        self.assertEqual(
            workloads["d128-block"]["overhead_filter"],
            ("HipMoiCdna4D128AttentionBlock." "SampledFastContextMatchesHostReference"),
        )
        self.assertEqual(
            workloads["jakub-attention"]["relative_path"],
            (
                "hip-moi-build-gfx950-tests/tests/"
                "hip_moi_reference_cdna4_jakub_matmul"
            ),
        )
        self.assertEqual(
            workloads["streamk-arrival"]["fault_families"],
            ("atomic-weaken-order",),
        )
        self.assertEqual(
            workloads["pytorch-scatter-reduce"]["fault_families"],
            ("atomic-weaken-order",),
        )
        self.assertEqual(
            workloads["pytorch-torch-histc"]["fault_families"],
            ("barrier-drop", "atomic-weaken-order"),
        )
        native_spellings = json.dumps(
            [
                workloads[workload_id]
                for workload_id in validation.NATIVE_GTEST_WORKLOAD_OVERRIDES["gfx950"]
            ]
        )
        self.assertNotIn("rdna4", native_spellings.lower())

    def test_gfx942_manifest_resolves_cdna3_native_workloads(self) -> None:
        manifest = validation._manifest("gfx942")
        workloads = {workload["id"]: workload for workload in manifest["workloads"]}
        self.assertEqual(
            workloads["d128-block"]["relative_path"],
            (
                "hip-moi-build-gfx942-tests/tests/"
                "hip_moi_instrumented_cdna3_d128_attention_block_test"
            ),
        )
        self.assertEqual(
            workloads["wmma-attention"]["clean_filter"],
            "HipMoiCdna3MfmaAttentionBlock.*",
        )
        self.assertEqual(
            workloads["d128-block"]["overhead_filter"],
            ("HipMoiCdna3D128AttentionBlock." "SampledFastContextMatchesHostReference"),
        )
        self.assertEqual(
            workloads["jakub-attention"]["relative_path"],
            (
                "hip-moi-build-gfx942-tests/tests/"
                "hip_moi_reference_cdna3_jakub_matmul"
            ),
        )
        self.assertEqual(
            workloads["streamk-arrival"]["fault_families"],
            ("atomic-weaken-order",),
        )
        self.assertEqual(workloads["d128-pressure"]["overhead_processes"], 1)
        self.assertEqual(workloads["clip-bf16"]["overhead_processes"], 1)
        native_spellings = json.dumps(
            [
                workloads[workload_id]
                for workload_id in validation.NATIVE_GTEST_WORKLOAD_OVERRIDES["gfx942"]
            ]
        )
        self.assertNotIn("rdna4", native_spellings.lower())
        self.assertNotIn("cdna4", native_spellings.lower())

    def test_public_workload_path_accessor_uses_target_registry(self) -> None:
        self.assertEqual(
            validation.resolved_workload_relative_path(
                "gfx942",
                "d128-pressure",
            ),
            (
                "hip-moi-build-gfx942-tests/tests/"
                "hip_moi_instrumented_cdna3_d128_attention_pressure_test"
            ),
        )

    def test_gfx942_doctor_rejects_missing_resolved_cdna3_executable(self) -> None:
        with temporary_root() as workspace:
            (workspace / "hip-moi").mkdir()
            hook = (
                workspace / "rocjitsu-build/lib/rocjitsu/src/rocjitsu/hooks/"
                "librocjitsu_dbi_hooks.so"
            )
            hook.parent.mkdir(parents=True)
            hook.touch()
            with mock.patch.object(validation.shutil, "which", return_value="/tool"):
                doctor = validation._doctor(workspace, "gfx942", ("d128-block",))
        executable = doctor["paths"]["workload:d128-block:executable"]
        self.assertEqual(
            executable["path"],
            str(
                workspace / "hip-moi-build-gfx942-tests/tests/"
                "hip_moi_instrumented_cdna3_d128_attention_block_test"
            ),
        )
        self.assertFalse(executable["present"])
        self.assertEqual(
            {label for label, path in doctor["paths"].items() if not path["present"]},
            {"workload:d128-block:executable"},
        )
        self.assertTrue(all(doctor["tools"].values()))
        self.assertEqual(doctor["runtimes"], {})
        self.assertFalse(doctor["ok"])

    def test_gfx1250_manifest_resolves_target_native_workloads(self) -> None:
        manifest = validation._manifest("gfx1250")
        workloads = {workload["id"]: workload for workload in manifest["workloads"]}
        self.assertEqual(
            workloads["d128-block"]["relative_path"],
            (
                "hip-moi-build-gfx1250-tests/tests/"
                "hip_moi_instrumented_gfx1250_d128_attention_block_test"
            ),
        )
        self.assertEqual(
            workloads["wmma-attention"]["clean_filter"],
            "HipMoiGfx1250WmmaAttentionBlock.*",
        )
        self.assertEqual(
            workloads["d128-block"]["fault_filter"],
            (
                "HipMoiGfx1250D128AttentionBlock."
                "SampledFastContextMatchesHostReference"
            ),
        )
        self.assertEqual(
            workloads["jakub-attention"]["relative_path"],
            (
                "hip-moi-build-gfx1250-tests/tests/"
                "hip_moi_reference_gfx1250_jakub_matmul"
            ),
        )
        self.assertEqual(
            workloads["tp1-prefill"]["fault_families"],
            ("barrier-move",),
        )

    def test_native_gtest_routing_matrix_pins_executables_and_phase_filters(
        self,
    ) -> None:
        target_shapes = {
            "gfx1201": NativeGtestTargetExpectation(
                build_dir="hip-moi-build",
                base="rdna4",
                matrix="rdna4_wmma",
                suite="Rdna4",
                matrix_suite="Rdna4Wmma",
                matrix_operation="Wmma",
                d128_block_oracle="ExactContextMatchesHostReference",
            ),
            "gfx942": NativeGtestTargetExpectation(
                build_dir="hip-moi-build-gfx942-tests",
                base="cdna3",
                matrix="cdna3_mfma",
                suite="Cdna3",
                matrix_suite="Cdna3Mfma",
                matrix_operation="Mfma",
                d128_block_oracle="SampledFastContextMatchesHostReference",
            ),
            "gfx950": NativeGtestTargetExpectation(
                build_dir="hip-moi-build-gfx950-tests",
                base="cdna4",
                matrix="cdna4_mfma",
                suite="Cdna4",
                matrix_suite="Cdna4Mfma",
                matrix_operation="Mfma",
                d128_block_oracle="SampledFastContextMatchesHostReference",
            ),
            "gfx1250": NativeGtestTargetExpectation(
                build_dir="hip-moi-build-gfx1250-tests",
                base="gfx1250",
                matrix="gfx1250_wmma",
                suite="Gfx1250",
                matrix_suite="Gfx1250Wmma",
                matrix_operation="Wmma",
                d128_block_oracle="SampledFastContextMatchesHostReference",
                d128_block_fault_uses_oracle=True,
            ),
        }
        workload_stems = {
            "d128-block": ("base", "instrumented", "d128_attention_block_test"),
            "d128-pressure": (
                "base",
                "instrumented",
                "d128_attention_pressure_test",
            ),
            "wmma-attention": ("matrix", "instrumented", "attention_block_test"),
            "streamk-arrival": (
                "matrix",
                "instrumented",
                "streamk_arrival_counter_test",
            ),
            "tree-atomic-or": (
                "matrix",
                "instrumented",
                "streamk_tree_atomic_or_test",
            ),
            "jakub-attention": ("base", "reference", "jakub_matmul"),
        }
        self.assertEqual(
            tuple(workload_stems),
            validation.NATIVE_GTEST_WORKLOAD_IDS,
        )
        for target, shape in target_shapes.items():
            expected_paths = {}
            for workload_id, (family_kind, binary_kind, stem) in workload_stems.items():
                family = shape.base if family_kind == "base" else shape.matrix
                expected_paths[workload_id] = (
                    f"{shape.build_dir}/tests/hip_moi_{binary_kind}_{family}_{stem}"
                )
            d128_block_clean = f"HipMoi{shape.suite}D128AttentionBlock.*"
            d128_block_oracle = (
                f"HipMoi{shape.suite}D128AttentionBlock." f"{shape.d128_block_oracle}"
            )
            expected_filters = {
                "d128-block": (
                    d128_block_clean,
                    d128_block_oracle,
                    (
                        d128_block_oracle
                        if shape.d128_block_fault_uses_oracle
                        else d128_block_clean
                    ),
                ),
                "d128-pressure": (
                    f"HipMoi{shape.suite}D128AttentionPressure.*",
                    f"HipMoi{shape.suite}D128AttentionPressure."
                    "FullKvDoubleBufferedExactContextMatchesHostReference",
                    f"HipMoi{shape.suite}D128AttentionPressure.*",
                ),
                "wmma-attention": (
                    f"HipMoi{shape.matrix_suite}AttentionBlock.*",
                    f"HipMoi{shape.matrix_suite}AttentionBlock."
                    "ExactContextMatchesHostReference",
                    f"HipMoi{shape.matrix_suite}AttentionBlock.*",
                ),
                "streamk-arrival": (
                    f"HipMoi{shape.matrix_suite}StreamKArrivalCounter."
                    f"AcqRelFetchAddOrders{shape.matrix_operation}Partials",
                )
                * 3,
                "tree-atomic-or": (
                    f"HipMoi{shape.matrix_suite}StreamKTreeAtomicOr."
                    f"AcqRelBitmaskOrders{shape.matrix_operation}Partials",
                )
                * 3,
                "jakub-attention": (
                    f"SafeFp16Packed/Jakub{shape.suite}MatmulReference."
                    "MatchesHostReference/*",
                )
                * 3,
            }
            with self.subTest(target=target), temporary_root() as workspace:
                (workspace / "hip-moi").mkdir()
                hook = (
                    workspace / "rocjitsu-build/lib/rocjitsu/src/rocjitsu/hooks/"
                    "librocjitsu_dbi_hooks.so"
                )
                hook.parent.mkdir(parents=True)
                hook.touch()
                for relative_path in expected_paths.values():
                    executable = workspace / relative_path
                    executable.parent.mkdir(parents=True, exist_ok=True)
                    executable.touch()
                with mock.patch.object(
                    validation.shutil, "which", return_value="/tool"
                ):
                    doctor = validation._doctor(
                        workspace,
                        target,
                        tuple(expected_paths),
                    )

                self.assertTrue(doctor["ok"], doctor)
                for workload_id, relative_path in expected_paths.items():
                    executable = doctor["paths"][f"workload:{workload_id}:executable"]
                    self.assertEqual(
                        executable,
                        {
                            "path": str(workspace / relative_path),
                            "present": True,
                        },
                    )
                    clean_filter, overhead_filter, fault_filter = expected_filters[
                        workload_id
                    ]
                    for phase, expected_filter in (
                        ("clean", clean_filter),
                        ("overhead", overhead_filter),
                        ("fault", fault_filter),
                    ):
                        with self.subTest(
                            target=target,
                            workload=workload_id,
                            phase=phase,
                        ):
                            self.assertEqual(
                                validation._workload_command(
                                    workspace,
                                    target,
                                    validation.WORKLOAD_BY_ID[workload_id],
                                    phase,
                                    workspace / "unused.json",
                                ),
                                [
                                    str(workspace / relative_path),
                                    f"--gtest_filter={expected_filter}",
                                ],
                            )

    def test_gfx1250_doctor_reports_missing_target_native_jakub_artifact(
        self,
    ) -> None:
        with temporary_root() as workspace:
            (workspace / "hip-moi").mkdir()
            hook = (
                workspace / "rocjitsu-build/lib/rocjitsu/src/rocjitsu/hooks/"
                "librocjitsu_dbi_hooks.so"
            )
            hook.parent.mkdir(parents=True)
            hook.touch()
            with mock.patch.object(validation.shutil, "which", return_value="/tool"):
                doctor = validation._doctor(
                    workspace,
                    "gfx1250",
                    ("jakub-attention",),
                )

        self.assertFalse(doctor["ok"])
        self.assertEqual(
            doctor["paths"]["workload:jakub-attention:executable"],
            {
                "path": str(
                    workspace / "hip-moi-build-gfx1250-tests/tests/"
                    "hip_moi_reference_gfx1250_jakub_matmul"
                ),
                "present": False,
            },
        )

    def test_main_doctor_all_uses_target_filtered_workloads(self) -> None:
        result = {
            "ok": True,
            "workspace": "/workspace",
            "target": "gfx1201",
            "paths": {},
            "tools": {},
        }
        with (
            mock.patch.object(
                validation,
                "_workspace_from_environment",
                return_value=Path("/workspace"),
            ),
            mock.patch.object(validation, "_doctor", return_value=result) as doctor,
        ):
            self.assertEqual(validation.main(["--target", "gfx1201", "doctor"]), 0)
        doctor.assert_called_once_with(Path("/workspace"), "gfx1201", None)

    def test_workload_doctor_requires_only_selected_inputs_and_tools(self) -> None:
        with temporary_root() as workspace:
            with mock.patch.object(validation.shutil, "which", return_value="/tool"):
                doctor = validation._doctor(workspace, "gfx950", ("d128-block",))
        self.assertEqual(doctor["workloads"], ["d128-block"])
        self.assertIn("hip-moi", doctor["paths"])
        self.assertNotIn("hip-moi-build", doctor["paths"])
        self.assertIn("workload:d128-block:executable", doctor["paths"])
        self.assertNotIn("iree-test-suites", doctor["paths"])
        self.assertNotIn("iree-test-suites-build", doctor["paths"])
        self.assertEqual(doctor["tools"], {"rocminfo": "/tool"})

    def test_health_smoke_falls_back_to_ready_selected_workload(self) -> None:
        workload = validation.WORKLOAD_BY_ID["d128-block"]
        with temporary_root() as workspace:
            command = validation._health_smoke_command(
                workspace, "gfx950", workload, workspace / "health.json"
            )
        self.assertTrue(command[0].endswith("cdna4_d128_attention_block_test"))
        self.assertEqual(command[1], "--gtest_filter=HipMoiCdna4D128AttentionBlock.*")

    def test_profile_environment_scrubs_controls_and_relies_on_sync_defaults(
        self,
    ) -> None:
        workload = validation.WORKLOAD_BY_ID["streamk-arrival"]
        with mock.patch.dict(
            os.environ,
            {
                "RJ_CONSAN_MAX_PATCHES": "1",
                "RJ_CONSAN_TMP_VGPR": "99",
                "HSA_TOOLS_LIB": "/stale/hook.so",
                "HSA_TOOLS_ROCPROFILER_V1_TOOLS": "0",
            },
            clear=False,
        ):
            environment = validation._clean_environment(
                "record-replay", workload, Path("/new/hook.so")
            )
        self.assertNotIn("RJ_CONSAN_MAX_PATCHES", environment)
        self.assertNotIn("RJ_CONSAN_TMP_VGPR", environment)
        self.assertEqual(environment["HSA_TOOLS_LIB"], "/new/hook.so")
        self.assertNotIn("HSA_TOOLS_ROCPROFILER_V1_TOOLS", environment)
        self.assertEqual(environment["RJ_CONSAN_MODE"], "record-replay")
        self.assertEqual(environment["RJ_CONSAN_POLICY"], "strict")
        self.assertNotIn("RJ_CONSAN_FLAVOR", environment)
        self.assertNotIn("RJ_CONSAN_MOI_ENGINE", environment)
        self.assertNotIn("RJ_CONSAN_MOI_TRACK_BARRIERS", environment)
        self.assertNotIn("RJ_CONSAN_MOI_TRACK_ATOMICS", environment)
        self.assertEqual(
            validation.ORDINARY_MOI_RUNTIME_DEFAULTS,
            {
                "RJ_CONSAN_MOI_TRACK_BARRIERS": "1",
                "RJ_CONSAN_MOI_TRACK_ATOMICS": "1",
            },
        )

    def test_pytorch_profile_enables_environment_hsa_tool_loading(self) -> None:
        workload = validation.WORKLOAD_BY_ID["pytorch-torch-histc"]
        with mock.patch.dict(
            os.environ,
            {"HSA_TOOLS_ROCPROFILER_V1_TOOLS": "0"},
            clear=False,
        ):
            baseline = validation._clean_environment(None, workload, Path("/hook.so"))
            environment = validation._clean_environment(
                "record-replay", workload, Path("/hook.so")
            )
        self.assertNotIn("HSA_TOOLS_ROCPROFILER_V1_TOOLS", baseline)
        self.assertEqual(environment["HSA_TOOLS_ROCPROFILER_V1_TOOLS"], "1")
        setting = validation._audited_settings(environment)
        v1_tool = next(
            item for item in setting if item["name"] == "HSA_TOOLS_ROCPROFILER_V1_TOOLS"
        )
        self.assertEqual(v1_tool["category"], "runtime-plumbing")
        self.assertFalse(v1_tool["usability_exception"])

    def test_growth_policy_overrides_are_audited_as_workload_tuning(self) -> None:
        for name, value in (
            ("RJ_CONSAN_MAX_PATCHED_IMAGE_GROWTH_BYTES", "4096"),
            ("RJ_CONSAN_MAX_PATCHED_IMAGE_GROWTH_PERCENT", "37"),
            ("RJ_CONSAN_MAX_PROCESS_CONCURRENT_TRANSFORM_BYTES", "12288"),
            ("RJ_CONSAN_MAX_PROCESS_PATCHED_IMAGE_BYTES", "16384"),
            ("RJ_CONSAN_MAX_PROCESS_PATCHED_IMAGE_GROWTH_BYTES", "8192"),
        ):
            [setting] = validation._audited_settings({name: value})
            self.assertEqual(setting["category"], "workload-tuning")
            self.assertTrue(setting["usability_exception"])

    def test_supercollider_does_not_receive_moi_tracking_controls(self) -> None:
        workload = validation.WORKLOAD_BY_ID["streamk-arrival"]
        environment = validation._clean_environment(
            "supercollider", workload, Path("/hook.so")
        )
        self.assertNotIn("RJ_CONSAN_MOI_TRACK_BARRIERS", environment)
        self.assertNotIn("RJ_CONSAN_MOI_TRACK_ATOMICS", environment)

    def test_scatter_disables_strict_record_requirement_for_inapplicable_lds(
        self,
    ) -> None:
        workload = validation.WORKLOAD_BY_ID["pytorch-scatter-reduce"]
        for profile in ("record-replay", "inline-shadow"):
            environment = validation._clean_environment(
                profile, workload, Path("/hook.so")
            )
            self.assertEqual(environment["RJ_CONSAN_MOI_REQUIRE_RECORDS"], "0")

    def test_qwen_sampled_relies_on_the_standard_runtime_operating_point(self) -> None:
        qwen = validation.WORKLOAD_BY_ID["qwen-prefill"]
        tp1 = validation.WORKLOAD_BY_ID["tp1-prefill"]
        qwen_environment = validation._clean_environment(
            "sampled", qwen, Path("/hook.so")
        )
        tp1_environment = validation._clean_environment(
            "sampled", tp1, Path("/hook.so")
        )
        self.assertEqual(qwen_environment["RJ_CONSAN_MOI_REQUIRE_RECORDS"], "1")
        self.assertNotIn("RJ_CONSAN_MOI_RUNTIME_SAMPLE_STRIDE", qwen_environment)
        self.assertNotIn("RJ_CONSAN_MOI_RUNTIME_SAMPLE_OFFSET", qwen_environment)
        self.assertNotIn("RJ_CONSAN_MOI_RUNTIME_SAMPLE_STRIDE", tp1_environment)

    def test_qwen_gfx1250_overhead_uses_a_software_backend_median(self) -> None:
        qwen = validation.WORKLOAD_BY_ID["qwen-prefill"]
        output = Path("/tmp/qwen-overhead.json")
        gfx1250 = validation._workload_command(
            Path("/workspace"), "gfx1250", qwen, "overhead", output
        )
        gfx1201 = validation._workload_command(
            Path("/workspace"), "gfx1201", qwen, "overhead", output
        )
        self.assertIn("--benchmark_repetitions=1", gfx1250)
        self.assertIn("--benchmark_repetitions=10", gfx1201)

    def test_sharktank_gfx1250_overhead_uses_one_inner_repetition(self) -> None:
        tp1 = validation.WORKLOAD_BY_ID["tp1-prefill"]
        command = validation._workload_command(
            Path("/workspace"), "gfx1250", tp1, "overhead", Path("/unused")
        )
        self.assertEqual(command[command.index("--repetitions") + 1], "1")

    def test_sharktank_native_cdna_uses_configured_python_and_one_repetition(
        self,
    ) -> None:
        tp1 = validation.WORKLOAD_BY_ID["tp1-prefill"]
        with mock.patch.dict(
            os.environ,
            {validation.SHARKTANK_PYTHON_ENV: "/workspace/venv/bin/python"},
        ):
            for target in ("gfx942", "gfx950"):
                with self.subTest(target=target):
                    command = validation._workload_command(
                        Path("/workspace"), target, tp1, "overhead", Path("/unused")
                    )
                    self.assertEqual(command[0], "/workspace/venv/bin/python")
                    self.assertEqual(command[command.index("--repetitions") + 1], "1")

    def test_active_architectures_use_one_outer_overhead_process(self) -> None:
        workload = validation.WORKLOAD_BY_ID["d128-pressure"]
        self.assertEqual(
            validation._outer_repetitions("gfx942", "overhead", workload), 1
        )
        self.assertEqual(
            validation._outer_repetitions("gfx950", "overhead", workload), 1
        )
        self.assertEqual(
            validation._outer_repetitions("gfx1250", "overhead", workload), 1
        )
        self.assertEqual(
            validation._outer_repetitions("gfx1201", "overhead", workload),
            workload.overhead_processes,
        )

    def test_gfx942_explain_uses_the_effective_outer_process_count(self) -> None:
        audit = validation._explain_contract(
            Path("/workspace"),
            "gfx942",
            ("d128-pressure", "clip-bf16"),
            validation.PROFILE_IDS,
            None,
            allow_reference=False,
        )
        for workload in audit["workloads"]:
            with self.subTest(workload=workload["id"]):
                self.assertEqual(workload["overhead_processes"], 1)
                self.assertEqual(workload["commands"]["clean"]["processes"], 1)
                self.assertEqual(workload["commands"]["overhead"]["processes"], 1)
        d128 = next(
            workload
            for workload in audit["workloads"]
            if workload["id"] == "d128-pressure"
        )
        self.assertIn("cdna3", d128["relative_path"])

    def test_gfx950_explain_uses_effective_atomic_fault_families(self) -> None:
        audit = validation._explain_contract(
            Path("/workspace"),
            "gfx950",
            ("pytorch-scatter-reduce",),
            validation.PROFILE_IDS,
            None,
            allow_reference=False,
        )
        workload = audit["workloads"][0]
        self.assertEqual(workload["fault_families"], ("atomic-weaken-order",))
        self.assertEqual(
            [fault["family"] for fault in workload["faults"]],
            ["atomic-weaken-order"],
        )

    def test_native_cdna_rejects_a_scope_only_fault_contract(self) -> None:
        workload = replace(
            validation.WORKLOAD_BY_ID["pytorch-scatter-reduce"],
            fault_families=("atomic-weaken-scope",),
        )
        with self.assertRaisesRegex(
            validation.ValidationError,
            "gfx950 workload has no applicable fault family: pytorch-scatter-reduce",
        ):
            validation._fault_families("gfx950", workload)

    def test_active_architecture_qwen_overhead_uses_one_repetition(self) -> None:
        workload = validation.WORKLOAD_BY_ID["qwen-prefill"]
        for target in ("gfx942", "gfx950", "gfx1250"):
            with self.subTest(target=target):
                command = validation._workload_command(
                    Path("/workspace"),
                    target,
                    workload,
                    "overhead",
                    Path("/artifacts/benchmark.json"),
                )
                self.assertIn("--benchmark_repetitions=1", command)

    def test_pytorch_gfx950_overhead_uses_one_repetition(self) -> None:
        workload = validation.WORKLOAD_BY_ID["pytorch-torch-mode"]
        command = validation._workload_command(
            Path("/workspace"),
            "gfx950",
            workload,
            "overhead",
            Path("/unused"),
        )
        self.assertEqual(command[command.index("--repetitions") + 1], "1")

    def test_pytorch_native_overhead_isolates_repetitions_by_process(self) -> None:
        workloads = tuple(
            workload
            for workload in validation.WORKLOADS
            if workload.kind == "pytorch"
            and workload.targets is not None
            and "gfx1201" in workload.targets
            and workload.overhead_processes > 1
        )
        self.assertEqual(
            tuple(workload.id for workload in workloads),
            ("pytorch-rdna4-llm-topk",),
        )
        for workload in workloads:
            with self.subTest(workload=workload.id):
                self.assertEqual(
                    validation._outer_repetitions("gfx1201", "overhead", workload),
                    validation.PYTORCH_OVERHEAD_PROCESSES,
                )
                command = validation._workload_command(
                    Path("/workspace"),
                    "gfx1201",
                    workload,
                    "overhead",
                    Path("/unused"),
                )
                self.assertEqual(
                    command[command.index("--repetitions") + 1],
                    "1",
                )
                self.assertEqual(
                    validation._outer_repetitions("gfx1201", "clean", workload),
                    1,
                )

        small = validation.WORKLOAD_BY_ID["pytorch-scatter-reduce"]
        self.assertEqual(validation._outer_repetitions("gfx1201", "overhead", small), 1)
        command = validation._workload_command(
            Path("/workspace"),
            "gfx1201",
            small,
            "overhead",
            Path("/unused"),
        )
        self.assertEqual(command[command.index("--repetitions") + 1], "10")

    def test_topk_record_replay_is_a_strict_clean_and_overhead_gate(
        self,
    ) -> None:
        hook = Path("/workspace/hook.so")
        clean_gate = validation._clean_environment(
            "record-replay",
            validation.WORKLOAD_BY_ID["pytorch-torch-mode"],
            hook,
            "gfx1201",
        )
        topk_clean = validation._run_environment(
            "record-replay",
            validation.WORKLOAD_BY_ID["pytorch-rdna4-llm-topk"],
            hook,
            "gfx1201",
            "clean",
        )

        self.assertEqual(clean_gate["RJ_CONSAN_MOI_FORBID_DIAGNOSTICS"], "1")
        self.assertEqual(topk_clean["RJ_CONSAN_MOI_FORBID_DIAGNOSTICS"], "1")
        self.assertEqual(topk_clean["RJ_CONSAN_POLICY"], "strict")
        overhead = validation._run_environment(
            "record-replay",
            validation.WORKLOAD_BY_ID["pytorch-rdna4-llm-topk"],
            hook,
            "gfx1201",
            "overhead",
        )
        self.assertEqual(overhead["RJ_CONSAN_MOI_FORBID_DIAGNOSTICS"], "1")
        for profile in ("sampled", "inline-shadow"):
            with self.subTest(profile=profile):
                other_engine = validation._run_environment(
                    profile,
                    validation.WORKLOAD_BY_ID["pytorch-rdna4-llm-topk"],
                    hook,
                    "gfx1201",
                    "clean",
                )
                self.assertEqual(other_engine["RJ_CONSAN_MOI_FORBID_DIAGNOSTICS"], "1")

    def test_run_environment_rejects_unknown_phase(self) -> None:
        with self.assertRaisesRegex(
            validation.ValidationError, "unsupported validation phase: fault"
        ):
            validation._run_environment(
                "record-replay",
                validation.WORKLOAD_BY_ID["pytorch-rdna4-llm-topk"],
                Path("/workspace/hook.so"),
                "gfx1201",
                "fault",
            )

    def test_coverage_output_overhead_uses_the_same_diagnostic_contract(
        self,
    ) -> None:
        contract = RETIRED_COVERAGE_OUTPUT_PARSER_CONTRACT
        workload = replace(
            validation.WORKLOAD_BY_ID["pytorch-rdna4-llm-topk"],
            coverage_output_contract=contract,
        )
        with temporary_root() as root:
            artifact_root = root / "artifacts"
            hook = root / "hook.so"
            hook.write_bytes(b"hook")
            with (
                mock.patch.object(validation, "_hook_path", return_value=hook),
                mock.patch.object(
                    validation, "_workload_command", return_value=["/bin/true"]
                ),
                mock.patch.object(
                    validation,
                    "_run_process",
                    return_value=(0, 0.1, "runtime output"),
                ),
                mock.patch.object(
                    validation, "_coverage_summary", return_value={"accepted": True}
                ) as coverage_summary,
                mock.patch.object(
                    validation, "_json_medians", return_value={"topk": 1.0}
                ),
                mock.patch.object(validation, "_source_identities", return_value=[]),
            ):
                result = validation._run_profile(
                    root,
                    "gfx1201",
                    workload,
                    "record-replay",
                    "overhead",
                    artifact_root,
                    120,
                )

        self.assertTrue(result["accepted"])
        self.assertEqual(
            len(result["coverage_runs"]),
            validation.PYTORCH_OVERHEAD_PROCESSES,
        )
        self.assertEqual(
            coverage_summary.call_count,
            validation.PYTORCH_OVERHEAD_PROCESSES,
        )
        coverage_summary.assert_has_calls(
            [
                mock.call(
                    "runtime output",
                    profile="record-replay",
                    coverage_output_contract=contract,
                )
            ]
            * validation.PYTORCH_OVERHEAD_PROCESSES
        )

    def test_gtest_run_rejects_zero_or_unreported_test_count(self) -> None:
        workload = validation.WORKLOAD_BY_ID["d128-block"]
        cases = (
            (
                "matched",
                "[==========] Running 1 test from 1 test suite.\n",
                [1],
                True,
            ),
            (
                "zero",
                "[==========] Running 0 tests from 0 test suites.\n",
                [0],
                False,
            ),
            ("missing", "process exited successfully\n", [None], False),
        )
        for name, output, expected_counts, expected_accepted in cases:
            with self.subTest(name=name), temporary_root() as root:
                artifact_root = root / "artifacts"
                hook = root / "hook.so"
                hook.write_bytes(b"hook")
                with (
                    mock.patch.object(validation, "_hook_path", return_value=hook),
                    mock.patch.object(
                        validation,
                        "_workload_command",
                        return_value=["/bin/true"],
                    ),
                    mock.patch.object(
                        validation,
                        "_run_process",
                        return_value=(0, 0.1, output),
                    ),
                    mock.patch.object(
                        validation,
                        "_source_identities",
                        return_value=[],
                    ),
                ):
                    result = validation._run_profile(
                        root,
                        "gfx950",
                        workload,
                        None,
                        "clean",
                        artifact_root,
                        30,
                    )

            self.assertEqual(result["gtest_test_counts"], expected_counts)
            self.assertEqual(result["accepted"], expected_accepted)

    def test_ordinary_record_replay_run_persists_structural_verdict(self) -> None:
        workload = validation.WORKLOAD_BY_ID["pytorch-torch-mode"]
        valid_log = complete_coverage_log(moi_auto_replay(7, 1, 0))
        cases = (
            ("valid", valid_log, True),
            (
                "malformed",
                valid_log.replace("provenance_unresolved=0", "provenance_state=0"),
                False,
            ),
        )
        for name, log, expected in cases:
            with self.subTest(name=name), temporary_root() as root:
                artifact_root = root / "artifacts"
                hook = root / "hook.so"
                hook.write_bytes(b"hook")
                with (
                    mock.patch.object(validation, "_hook_path", return_value=hook),
                    mock.patch.object(
                        validation,
                        "_workload_command",
                        return_value=["/bin/true"],
                    ),
                    mock.patch.object(
                        validation,
                        "_run_process",
                        return_value=(0, 0.1, log),
                    ),
                    mock.patch.object(
                        validation,
                        "_source_identities",
                        return_value=[],
                    ),
                ):
                    result = validation._run_profile(
                        root,
                        "gfx1201",
                        workload,
                        "record-replay",
                        "clean",
                        artifact_root,
                        30,
                    )

                persisted = json.loads(
                    (
                        artifact_root
                        / workload.id
                        / "clean"
                        / "record-replay"
                        / "result.json"
                    ).read_text(encoding="utf-8")
                )
                self.assertEqual(result["accepted"], expected)
                self.assertEqual(persisted["accepted"], expected)
                self.assertEqual(
                    persisted["coverage"]["diagnostics"]["policy"]["kind"],
                    "clean",
                )
                if name == "malformed":
                    self.assertIn(
                        "replay provenance unresolved: "
                        "reader=7,generation=1, count=None",
                        persisted["coverage"]["diagnostics"]["reasons"],
                    )

    def test_coverage_output_profile_does_not_relax_not_detected_faults(
        self,
    ) -> None:
        environment = validation._fault_trial_environment(
            "record-replay",
            validation.WORKLOAD_BY_ID["pytorch-rdna4-llm-topk"],
            Path("/workspace/hook.so"),
            "gfx1201",
            {"environment": {}},
            {"detector": "not_detected"},
            {},
        )
        self.assertEqual(environment["RJ_CONSAN_MOI_FORBID_DIAGNOSTICS"], "1")

    def test_coverage_output_contract_separates_result_phase(self) -> None:
        coverage_output_workload = replace(
            validation.WORKLOAD_BY_ID["pytorch-rdna4-llm-topk"],
            coverage_output_contract=RETIRED_COVERAGE_OUTPUT_PARSER_CONTRACT,
        )
        self.assertEqual(
            validation._result_phase(
                "clean",
                "record-replay",
                coverage_output_workload,
            ),
            "coverage-output",
        )
        self.assertEqual(
            validation._result_phase(
                "clean",
                "sampled",
                coverage_output_workload,
            ),
            "clean",
        )
        self.assertEqual(
            validation._result_phase(
                "clean",
                "record-replay",
                validation.WORKLOAD_BY_ID["pytorch-torch-mode"],
            ),
            "clean",
        )
        self.assertEqual(
            validation._result_phase(
                "overhead",
                "record-replay",
                coverage_output_workload,
            ),
            "overhead",
        )
        self.assertEqual(
            validation._result_phase(
                "clean",
                "record-replay",
                validation.WORKLOAD_BY_ID["pytorch-rdna4-llm-topk"],
            ),
            "clean",
        )

    def test_coverage_output_result_references_workload_provenance(self) -> None:
        workload = replace(
            validation.WORKLOAD_BY_ID["pytorch-rdna4-llm-topk"],
            coverage_output_contract=RETIRED_COVERAGE_OUTPUT_PARSER_CONTRACT,
        )
        log = complete_coverage_log(moi_auto_replay(7, 1, 0))
        with temporary_root() as root:
            artifact_root = root / "artifacts"
            provenance = validation._workload_provenance_path(artifact_root, workload)
            provenance.parent.mkdir(parents=True)
            provenance.write_text("{}\n", encoding="utf-8")
            hook = root / "hook.so"
            hook.write_bytes(b"hook")
            with (
                mock.patch.object(validation, "_hook_path", return_value=hook),
                mock.patch.object(
                    validation, "_workload_command", return_value=["/bin/true"]
                ),
                mock.patch.object(
                    validation,
                    "_run_process",
                    return_value=(0, 0.1, log),
                ),
                mock.patch.object(validation, "_source_identities", return_value=[]),
            ):
                result = validation._run_profile(
                    root,
                    "gfx1201",
                    workload,
                    "record-replay",
                    "clean",
                    artifact_root,
                    120,
                )
            self.assertTrue(provenance.exists())
            persisted = json.loads(
                (
                    artifact_root
                    / workload.id
                    / "coverage-output"
                    / "record-replay"
                    / "result.json"
                ).read_text(encoding="utf-8")
            )

        self.assertEqual(result["phase"], "coverage-output")
        self.assertEqual(result["provenance"], str(provenance))
        self.assertTrue(result["accepted"])
        self.assertEqual(
            result["coverage"]["diagnostics"]["replay_count"],
            0,
        )
        self.assertEqual(
            persisted["coverage"]["diagnostics"],
            json.loads(json.dumps(result["coverage"]["diagnostics"])),
        )

    def test_workload_provenance_is_shared_only_when_inputs_match(self) -> None:
        workload = validation.WORKLOAD_BY_ID["pytorch-rdna4-llm-topk"]
        with temporary_root() as root:
            hook = root / "hook.so"
            workload_input = root / "input.py"
            hook.write_bytes(b"hook")
            workload_input.write_bytes(b"input")
            workload_root = root / "artifacts" / workload.id
            with (
                mock.patch.object(validation, "_hook_path", return_value=hook),
                mock.patch.object(
                    validation, "_input_files", return_value={"input": workload_input}
                ),
                mock.patch.object(validation, "_source_identities", return_value=[]),
                mock.patch.object(
                    validation,
                    "_manifest",
                    return_value={"schema_version": 1, "profiles": ("a", "b")},
                ),
            ):
                first = validation._write_provenance(
                    root, "gfx1201", workload, workload_root
                )
                second = validation._write_provenance(
                    root, "gfx1201", workload, workload_root
                )
                workload_input.write_bytes(b"changed")
                with self.assertRaisesRegex(
                    validation.ValidationError,
                    "provenance conflicts with existing artifact",
                ):
                    validation._write_provenance(
                        root, "gfx1201", workload, workload_root
                    )

        self.assertEqual(first, second)
        self.assertEqual(first, workload_root / "provenance.json")

    def test_topk_explain_reports_strict_clean_contract(self) -> None:
        audit = validation._explain_contract(
            Path("/workspace"),
            "gfx1201",
            ("pytorch-rdna4-llm-topk",),
            validation.PROFILE_IDS,
            None,
            allow_reference=False,
        )
        workload = audit["workloads"][0]
        record_replay = next(
            profile
            for profile in workload["profiles"]
            if profile["id"] == "record-replay"
        )
        settings = {
            setting["name"]: setting["value"] for setting in record_replay["settings"]
        }

        self.assertEqual(settings["RJ_CONSAN_MOI_FORBID_DIAGNOSTICS"], "1")
        self.assertEqual(record_replay["clean_result_phase"], "clean")
        self.assertEqual(
            record_replay["clean_artifact_root"],
            "$ARTIFACT_ROOT/pytorch-rdna4-llm-topk/clean/record-replay",
        )
        self.assertEqual(
            workload["commands"]["clean"]["profile_artifact_roots"]["record-replay"],
            "$ARTIFACT_ROOT/pytorch-rdna4-llm-topk/clean/record-replay",
        )
        self.assertEqual(
            workload["commands"]["clean"]["payload_argv"],
            validation._workload_command(
                Path("/workspace"),
                "gfx1201",
                validation.WORKLOAD_BY_ID["pytorch-rdna4-llm-topk"],
                "clean",
                Path(
                    "$ARTIFACT_ROOT/pytorch-rdna4-llm-topk/"
                    "clean/record-replay/benchmark-0.json"
                ),
            ),
        )
        self.assertEqual(
            audit["usability_audit"]["coverage_output_contracts"],
            [],
        )
        self.assertEqual(
            audit["usability_audit"]["fault_qualification_exceptions"],
            [],
        )
        self.assertEqual(len(workload["faults"]), 1)
        expectations = {
            item["profile"]: item
            for item in workload["faults"][0]["profile_expectations"]
        }
        for profile in validation.PROFILE_IDS:
            with self.subTest(profile=profile):
                self.assertEqual(
                    expectations[profile]["disposition"],
                    "applicable",
                )
                self.assertEqual(
                    expectations[profile]["detector"],
                    "REVIEW_REQUIRED",
                )

    def test_topk_explain_text_renders_profile_specific_commands(
        self,
    ) -> None:
        audit = validation._explain_contract(
            Path("/workspace"),
            "gfx1201",
            ("pytorch-rdna4-llm-topk",),
            validation.PROFILE_IDS,
            None,
            allow_reference=False,
        )
        output = io.StringIO()
        with redirect_stdout(output):
            validation._print_explain(audit)
        rendered = output.getvalue()
        self.assertIn("clean (1 process(es)):", rendered)
        self.assertIn(
            shlex.join(audit["workloads"][0]["commands"]["clean"]["payload_argv"]),
            rendered,
        )

    def test_coverage_output_contract_accepts_current_physical_runtime_fixture(
        self,
    ) -> None:
        contract = RETIRED_COVERAGE_OUTPUT_PARSER_CONTRACT
        fixture = (
            Path(__file__).with_name("fixtures")
            / "gfx1201_topk_record_replay_current_runtime.log"
        ).read_text(encoding="utf-8")
        assert_current_replay_log(self, fixture)

        summary = validation._coverage_output_diagnostic_summary(fixture, contract)
        self.assertTrue(summary["accepted"], summary["reasons"])
        self.assertEqual(summary["replay_count"], 0)
        self.assertEqual(summary["records"], [])
        self.assertEqual(
            set(summary["readers"]),
            {"reader=725954112,generation=3"},
        )

        drifted = fixture.replace(
            "diagnostics=0 conflict=false",
            "diagnostic_total=0 conflict=false",
            1,
        )
        drifted_summary = validation._coverage_output_diagnostic_summary(
            drifted, contract
        )
        self.assertFalse(drifted_summary["accepted"])
        self.assertIn(
            "malformed replay diagnostic summary",
            drifted_summary["reasons"],
        )

    def test_ordinary_record_replay_accepts_current_physical_runtime_fixture(
        self,
    ) -> None:
        fixture = (
            Path(__file__).with_name("fixtures")
            / "gfx1201_topk_record_replay_current_runtime.log"
        ).read_text(encoding="utf-8")
        assert_current_replay_log(self, fixture)

        summary = validation._diagnostic_output_summary(fixture, "record-replay")

        self.assertTrue(summary["accepted"], summary["reasons"])
        self.assertEqual(summary["diagnostic_count"], 0)
        self.assertEqual(summary["records"], [])

    def test_ordinary_record_replay_rejects_physical_diagnostic_fixture(
        self,
    ) -> None:
        fixture = (
            Path(__file__).with_name("fixtures")
            / "gfx1201_topk_record_replay_current_diagnostics.log"
        ).read_text(encoding="utf-8")
        assert_current_replay_log(self, fixture)

        summary = validation._diagnostic_output_summary(fixture, "record-replay")

        self.assertFalse(summary["accepted"])
        self.assertEqual(summary["diagnostic_count"], 3)
        self.assertIn(
            "unexpected diagnostics=exact-lds-write-write",
            summary["reasons"],
        )

    def test_coverage_output_contract_accepts_current_physical_diagnostic_fixture(
        self,
    ) -> None:
        contract = RETIRED_COVERAGE_OUTPUT_PARSER_CONTRACT
        fixture = (
            Path(__file__).with_name("fixtures")
            / "gfx1201_topk_record_replay_current_diagnostics.log"
        ).read_text(encoding="utf-8")
        assert_current_replay_log(self, fixture)

        summary = validation._coverage_output_diagnostic_summary(fixture, contract)

        self.assertTrue(summary["accepted"], summary["reasons"])
        self.assertEqual(summary["replay_count"], 3)
        self.assertEqual(
            [record["first_instruction"] for record in summary["records"]],
            ["0xfe96c", "0xfe96c", "0xfe9c4"],
        )
        self.assertEqual(
            [record["second_instruction"] for record in summary["records"]],
            ["0xfe974", "0xfe974", "0xfe9c4"],
        )
        self.assertEqual(
            [record["generation"] for record in summary["records"]],
            [3, 3, 3],
        )
        self.assertEqual(
            summary["observed_code_object_fingerprints"],
            [RETIRED_TOPK_CODE_OBJECT_FINGERPRINT],
        )
        self.assertEqual(
            summary["readers"]["reader=1679495808,generation=3"]["provenance_repaired"],
            3,
        )

    def test_coverage_output_contract_accepts_alternate_store_group_fixture(
        self,
    ) -> None:
        contract = RETIRED_COVERAGE_OUTPUT_PARSER_CONTRACT
        fixture = (
            Path(__file__).with_name("fixtures")
            / "gfx1201_topk_record_replay_alternate_group_diagnostics.log"
        ).read_text(encoding="utf-8")
        assert_current_replay_log(self, fixture)

        summary = validation._coverage_output_diagnostic_summary(fixture, contract)

        self.assertTrue(summary["accepted"], summary["reasons"])
        self.assertEqual(summary["replay_count"], 2)
        self.assertEqual(
            [
                (
                    record["first_instruction"],
                    record["second_instruction"],
                )
                for record in summary["records"]
            ],
            [("0xfea68", "0xfea70"), ("0xfea68", "0xfea70")],
        )
        self.assertEqual(
            summary["readers"]["reader=1142454128,generation=3"]["provenance_repaired"],
            2,
        )

    def test_native_cdna_scrubs_software_model_environment_without_changing_gfx1250(
        self,
    ) -> None:
        model_environment = {
            name: f"configured-{name}" for name in validation.SOFTWARE_MODEL_ENVIRONMENT
        }
        with mock.patch.dict(os.environ, model_environment, clear=False):
            native_cdna = {
                target: validation._clean_environment(
                    None,
                    validation.WORKLOAD_BY_ID[workload_id],
                    Path("/workspace/hook.so"),
                    target,
                )
                for target, workload_id in (
                    ("gfx942", "qwen-prefill"),
                    ("gfx950", "pytorch-torch-mode"),
                )
            }
            gfx1250 = validation._clean_environment(
                None,
                validation.WORKLOAD_BY_ID["pytorch-torch-mode"],
                Path("/workspace/hook.so"),
                "gfx1250",
            )
        for target, environment in native_cdna.items():
            with self.subTest(target=target):
                self.assertTrue(
                    validation.SOFTWARE_MODEL_ENVIRONMENT.isdisjoint(environment)
                )
        self.assertEqual(
            {name: gfx1250[name] for name in validation.SOFTWARE_MODEL_ENVIRONMENT},
            model_environment,
        )

    def test_cdna_atomics_only_admit_order_faults(self) -> None:
        for target in ("gfx942", "gfx950"):
            for workload_id in ("streamk-arrival", "tree-atomic-or"):
                with self.subTest(target=target, workload=workload_id):
                    workload = validation.WORKLOAD_BY_ID[workload_id]
                    self.assertEqual(
                        validation._fault_families(target, workload),
                        ("atomic-weaken-order",),
                    )
                    self.assertEqual(
                        validation._fault_families("gfx1201", workload),
                        ("atomic-weaken-order", "atomic-weaken-scope"),
                    )

    def test_pytorch_gfx1250_runs_both_variants_once(self) -> None:
        workload = validation.WORKLOAD_BY_ID["pytorch-tdm-descriptor-add"]
        with mock.patch.dict(
            os.environ,
            {validation.PYTORCH_PYTHON_ENV: "/workspace/venv/bin/python"},
        ):
            command = validation._workload_command(
                Path("/workspace"),
                "gfx1250",
                workload,
                "overhead",
                Path("/unused"),
            )
        self.assertEqual(command[0], "/workspace/venv/bin/python")
        self.assertEqual(command[command.index("--repetitions") + 1], "1")
        self.assertEqual(command[command.index("--workload") + 1], "tdm-descriptor-add")

    def test_pytorch_python_discovers_standard_workspace_environment(self) -> None:
        with temporary_root() as workspace:
            python = workspace / "consan-pytorch-venv" / "bin" / "python"
            python.parent.mkdir(parents=True)
            python.touch()
            with mock.patch.dict(
                os.environ, {validation.PYTORCH_PYTHON_ENV: ""}, clear=False
            ):
                self.assertEqual(validation._pytorch_python(workspace), python)

    def test_pytorch_python_explicit_environment_overrides_workspace(self) -> None:
        with mock.patch.dict(
            os.environ,
            {validation.PYTORCH_PYTHON_ENV: "/custom/venv/bin/python"},
            clear=False,
        ):
            self.assertEqual(
                validation._pytorch_python(Path("/workspace")),
                Path("/custom/venv/bin/python"),
            )

    def test_pytorch_doctor_rejects_interpreter_with_broken_imports(self) -> None:
        workload = validation.WORKLOAD_BY_ID["pytorch-rdna4-compiled-softmax"]
        with temporary_root() as workspace:
            python = workspace / "consan-pytorch-venv" / "bin" / "python"
            python.parent.mkdir(parents=True)
            python.touch()
            completed = subprocess.CompletedProcess(
                [str(python)], 1, stdout="", stderr="No module named torch"
            )
            with (
                mock.patch.object(validation.shutil, "which", return_value="/tool"),
                mock.patch.object(
                    validation.subprocess, "run", return_value=completed
                ) as run,
            ):
                doctor = validation._doctor(workspace, "gfx1201", (workload.id,))
        self.assertFalse(doctor["ok"])
        self.assertFalse(doctor["runtimes"]["pytorch"]["ok"])
        self.assertIn("No module named torch", doctor["runtimes"]["pytorch"]["detail"])
        self.assertEqual(run.call_args.args[0][0], str(python))
        environment = run.call_args.kwargs["env"]
        self.assertEqual(environment["HSA_TOOLS_ROCPROFILER_V1_TOOLS"], "1")
        self.assertEqual(
            environment["RJ_CONSAN_TEST_KERNEL_FILTER"],
            "__consan_pytorch_runtime_probe_never_matches__",
        )

    def test_pytorch_doctor_rejects_runtime_that_skips_consan_hook(self) -> None:
        workload = validation.WORKLOAD_BY_ID["pytorch-rdna4-compiled-softmax"]
        with temporary_root() as workspace:
            python = workspace / "consan-pytorch-venv" / "bin" / "python"
            hook = (
                workspace / "rocjitsu-build/lib/rocjitsu/src/rocjitsu/hooks/"
                "librocjitsu_dbi_hooks.so"
            )
            python.parent.mkdir(parents=True)
            hook.parent.mkdir(parents=True)
            python.touch()
            hook.touch()
            completed = subprocess.CompletedProcess(
                [str(python)],
                0,
                stdout=json.dumps(
                    {
                        "torch": "test",
                        "hip": "test",
                        "triton": "test",
                        "device": "test GPU",
                        "arch": "gfx1201",
                        "numeric_oracle": True,
                        "hook_loaded": False,
                    }
                ),
                stderr="",
            )
            with (
                mock.patch.object(validation.shutil, "which", return_value="/tool"),
                mock.patch.object(validation.subprocess, "run", return_value=completed),
            ):
                doctor = validation._doctor(workspace, "gfx1201", (workload.id,))
        runtime = doctor["runtimes"]["pytorch"]
        self.assertFalse(doctor["ok"])
        self.assertFalse(runtime["ok"])
        self.assertIn(
            "PyTorch HSA runtime did not load the ConSan hook", runtime["reasons"]
        )

    def test_pytorch_doctor_probes_canonical_hook_path(self) -> None:
        workload = validation.WORKLOAD_BY_ID["pytorch-rdna4-compiled-softmax"]
        with temporary_root() as workspace:
            python = workspace / "consan-pytorch-venv" / "bin" / "python"
            real_build = workspace / "build"
            hook_suffix = Path(
                "lib/rocjitsu/src/rocjitsu/hooks/librocjitsu_dbi_hooks.so"
            )
            hook = real_build / hook_suffix
            python.parent.mkdir(parents=True)
            hook.parent.mkdir(parents=True)
            python.touch()
            hook.touch()
            (workspace / "rocjitsu-build").symlink_to(
                real_build, target_is_directory=True
            )
            completed = subprocess.CompletedProcess(
                [str(python)],
                0,
                stdout=json.dumps(
                    {
                        "torch": "test",
                        "hip": "test",
                        "triton": "test",
                        "device": "test GPU",
                        "arch": "gfx1201",
                        "numeric_oracle": True,
                        "hook_loaded": True,
                    }
                ),
                stderr="",
            )
            with mock.patch.object(
                validation.subprocess, "run", return_value=completed
            ) as run:
                runtime = validation._pytorch_runtime_probe(
                    python,
                    workspace / "rocjitsu-build" / hook_suffix,
                    "gfx1201",
                    workload,
                )
        self.assertTrue(runtime["ok"])
        self.assertEqual(run.call_args.args[0][-1], str(hook.resolve()))

    def test_pytorch_only_doctor_uses_runtime_target_probe_without_rocminfo(
        self,
    ) -> None:
        workload = validation.WORKLOAD_BY_ID["pytorch-rdna4-compiled-softmax"]
        with temporary_root() as workspace:
            python = workspace / "consan-pytorch-venv" / "bin" / "python"
            hook = (
                workspace / "rocjitsu-build/lib/rocjitsu/src/rocjitsu/hooks/"
                "librocjitsu_dbi_hooks.so"
            )
            python.parent.mkdir(parents=True)
            hook.parent.mkdir(parents=True)
            python.touch()
            hook.touch()
            runtime = {
                "ok": True,
                "python": str(python),
                "detail": {
                    "arch": "gfx1201",
                    "numeric_oracle": True,
                    "hook_loaded": True,
                },
                "reasons": [],
            }
            with (
                mock.patch.object(validation.shutil, "which", return_value=None),
                mock.patch.object(
                    validation, "_pytorch_runtime_probe", return_value=runtime
                ),
            ):
                doctor = validation._doctor(workspace, "gfx1201", (workload.id,))
        self.assertTrue(doctor["ok"])
        self.assertEqual(doctor["tools"], {})

    def test_pytorch_cluster_workload_runs_once(self) -> None:
        workload = validation.WORKLOAD_BY_ID["pytorch-cluster-load-sync"]
        with mock.patch.dict(
            os.environ,
            {validation.PYTORCH_PYTHON_ENV: "/workspace/venv/bin/python"},
        ):
            command = validation._workload_command(
                Path("/workspace"),
                "gfx1250",
                workload,
                "clean",
                Path("/unused"),
            )
        self.assertEqual(command[command.index("--repetitions") + 1], "1")
        self.assertEqual(command[command.index("--workload") + 1], "cluster-load-sync")

    def test_pytorch_rdna4_softmax_uses_native_compiled_client(self) -> None:
        workload = validation.WORKLOAD_BY_ID["pytorch-rdna4-compiled-softmax"]
        with mock.patch.dict(
            os.environ,
            {validation.PYTORCH_PYTHON_ENV: "/workspace/venv/bin/python"},
        ):
            command = validation._workload_command(
                Path("/workspace"),
                "gfx1201",
                workload,
                "clean",
                Path("/unused"),
            )
        self.assertEqual(command[0], "/workspace/venv/bin/python")
        self.assertEqual(command[command.index("--repetitions") + 1], "1")
        self.assertEqual(
            command[command.index("--workload") + 1], "rdna4-compiled-softmax"
        )

    def test_pytorch_rdna4_split_softmax_uses_upstream_native_shape(self) -> None:
        workload = validation.WORKLOAD_BY_ID["pytorch-rdna4-split-softmax"]
        with mock.patch.dict(
            os.environ,
            {validation.PYTORCH_PYTHON_ENV: "/workspace/venv/bin/python"},
        ):
            command = validation._workload_command(
                Path("/workspace"),
                "gfx1201",
                workload,
                "clean",
                Path("/unused"),
            )
        self.assertEqual(command[0], "/workspace/venv/bin/python")
        self.assertEqual(command[command.index("--repetitions") + 1], "1")
        self.assertEqual(
            command[command.index("--workload") + 1], "rdna4-split-softmax"
        )

    def test_pytorch_rdna4_llm_topk_uses_native_client(self) -> None:
        workload = validation.WORKLOAD_BY_ID["pytorch-rdna4-llm-topk"]
        with mock.patch.dict(
            os.environ,
            {validation.PYTORCH_PYTHON_ENV: "/workspace/venv/bin/python"},
        ):
            command = validation._workload_command(
                Path("/workspace"),
                "gfx1201",
                workload,
                "clean",
                Path("/unused"),
            )
        self.assertEqual(command[0], "/workspace/venv/bin/python")
        self.assertEqual(command[command.index("--repetitions") + 1], "1")
        self.assertEqual(command[command.index("--workload") + 1], "rdna4-llm-topk")

    def test_pytorch_rdna4_sdpa_uses_native_client(self) -> None:
        workload = validation.WORKLOAD_BY_ID["pytorch-rdna4-sdpa"]
        with mock.patch.dict(
            os.environ,
            {validation.PYTORCH_PYTHON_ENV: "/workspace/venv/bin/python"},
        ):
            command = validation._workload_command(
                Path("/workspace"),
                "gfx1201",
                workload,
                "clean",
                Path("/unused"),
            )
        self.assertEqual(command[0], "/workspace/venv/bin/python")
        self.assertEqual(command[command.index("--repetitions") + 1], "1")
        self.assertEqual(command[command.index("--workload") + 1], "rdna4-sdpa")

    def test_llama_rdna4_command_uses_gpu_cpu_oracle_wrapper(self) -> None:
        workload = validation.WORKLOAD_BY_ID["llama-rdna4-rms-norm"]
        command = validation._workload_command(
            Path("/workspace"),
            "gfx1201",
            workload,
            "clean",
            Path("/artifacts/benchmark-0.json"),
        )
        self.assertTrue(command[1].endswith("consan_llama_validation.py"))
        self.assertEqual(command[command.index("--workload") + 1], "rms-norm")
        self.assertEqual(
            command[command.index("--executable") + 1],
            (
                "/workspace/rocjitsu-test-corpus-build/kernels/gfx1201/"
                "cases/llama.cpp/llama_cpp_rms_norm"
            ),
        )
        self.assertEqual(
            command[command.index("--output-dir") + 1],
            "/artifacts/benchmark-0-llama-work",
        )

    def test_native_matvec_uses_fault_sensitive_realistic_shape(self) -> None:
        self.assertEqual(llama_validation.WORKLOADS["mul-mat-vec-q"]["n_embd"], 1024)
        self.assertEqual(llama_validation.WORKLOADS["mul-mat-vec-q"]["n_tokens"], 1)
        self.assertEqual(
            llama_validation.WORKLOADS["mul-mat-vec-q"]["tolerance"], 2.0e-2
        )

    def test_llama_cpu_oracle_environment_scrubs_instrumentation(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "RJ_CONSAN_MODE": "inline-shadow",
                "HSA_TOOLS_LIB": "/hook.so",
                "HSA_TOOLS_ROCPROFILER_V1_TOOLS": "1",
                "HIP_TARGET": "gfx1201",
                "LD_LIBRARY_PATH": "/runtime",
            },
            clear=True,
        ):
            environment = llama_validation._cpu_environment()
        self.assertEqual(environment, {"LD_LIBRARY_PATH": "/runtime"})

    def test_llama_binary_oracle_reads_f32_without_numpy(self) -> None:
        with temporary_root() as root:
            path = root / "output.bin"
            path.write_bytes(b"\x00\x00\x80?\x00\x00\x00@")
            self.assertEqual(llama_validation._read_f32(path), (1.0, 2.0))
            path.write_bytes(b"short")
            with self.assertRaises(ValueError):
                llama_validation._read_f32(path)

    def test_llama_oracle_writes_fault_runner_result(self) -> None:
        with temporary_root() as root:
            path = root / "oracle.json"
            with mock.patch.dict(
                os.environ, {"CONSAN_ROW_RESULT_PATH": str(path)}, clear=True
            ):
                llama_validation._write_oracle_result(
                    "pass", {"llama-mul-mat-vec-q": {"oracle_passed": True}}
                )
            result = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(result["oracle"], "pass")
        self.assertEqual(result["source_diagnostics"]["outcome"], "not_applicable")

    def test_tensile_gfx1250_uses_numeric_runner_once(self) -> None:
        workload = validation.WORKLOAD_BY_ID["tensile-sk-mxf8gemm-explicit"]
        with mock.patch.dict(
            os.environ,
            {validation.TENSILE_PYTHON_ENV: "/workspace/venv/bin/python"},
        ):
            command = validation._workload_command(
                Path("/workspace"),
                "gfx1250",
                workload,
                "clean",
                Path("/artifacts/benchmark.json"),
            )
        self.assertEqual(command[0], "/workspace/venv/bin/python")
        self.assertTrue(command[1].endswith("consan_tensile_validation.py"))
        self.assertEqual(command[command.index("--repetitions") + 1], "1")
        self.assertEqual(
            command[command.index("--config") + 1],
            str(Path("/workspace") / workload.corpus / workload.relative_path),
        )
        self.assertNotIn("--streamk-fixed-grid", command)
        self.assertNotIn("--timeout-seconds", command)
        self.assertNotIn("--expect-numeric-rows", command)
        self.assertEqual(
            command[command.index("--output-dir") + 1],
            "/artifacts/tensile-work",
        )

    def test_gfx1250_manifest_registers_f16_sb_tensile_closure(self) -> None:
        manifest = validation._manifest("gfx1250")
        workloads = {workload["id"]: workload for workload in manifest["workloads"]}
        workload = workloads["tensile-spmm-f16-sb"]
        self.assertEqual(workload["priority"], "P2")
        self.assertEqual(workload["kind"], "tensile")
        self.assertEqual(
            workload["relative_path"],
            (
                "corpus/tensile/configs/Tensile/Tests/common/sparse/gfx1250/"
                "spmm_f16_sb.yaml"
            ),
        )
        self.assertEqual(workload["fault_families"], ("barrier-drop",))

    def test_f16_sb_tensile_closure_uses_exact_runner_once(self) -> None:
        workload = validation.WORKLOAD_BY_ID["tensile-spmm-f16-sb"]
        with mock.patch.dict(
            os.environ,
            {validation.TENSILE_PYTHON_ENV: "/workspace/venv/bin/python"},
        ):
            command = validation._workload_command(
                Path("/workspace"),
                "gfx1250",
                workload,
                "clean",
                Path("/artifacts/benchmark.json"),
            )
        self.assertEqual(command[0], "/workspace/venv/bin/python")
        self.assertTrue(command[1].endswith("consan_tensile_validation.py"))
        self.assertEqual(command[command.index("--repetitions") + 1], "1")
        self.assertEqual(
            command[command.index("--config") + 1],
            str(Path("/workspace") / workload.corpus / workload.relative_path),
        )
        self.assertEqual(
            command[command.index("--output-dir") + 1],
            "/artifacts/tensile-work",
        )

    def test_bounded_tensile_smoke_is_workspace_native_and_fixed_grid(self) -> None:
        workload = validation.WORKLOAD_BY_ID["tensile-sk-sgemm-runtime-smoke"]
        command = validation._workload_command(
            Path("/workspace"),
            "gfx1250",
            workload,
            "clean",
            Path("/artifacts/benchmark.json"),
        )
        self.assertEqual(workload.priority, "P1")
        self.assertEqual(workload.corpus, "rocm-systems")
        self.assertEqual(workload.run_timeout_seconds, 60)
        self.assertEqual(workload.tensile_inner_timeout_seconds, 55)
        self.assertEqual(workload.tensile_expected_numeric_rows, 1)
        self.assertEqual(
            command[command.index("--config") + 1],
            (
                "/workspace/rocm-systems/emulation/rocjitsu/tests/dbi/consan/"
                "fixtures/gfx1250_tensile_streamk_smoke.yaml"
            ),
        )
        self.assertEqual(
            command[command.index("--timeout-seconds") + 1],
            str(workload.tensile_inner_timeout_seconds),
        )
        self.assertEqual(command[command.index("--expect-numeric-rows") + 1], "1")
        self.assertEqual(command[command.index("--streamk-fixed-grid") + 1], "4")
        self.assertEqual(command[command.index("--require-streamk-mode") + 1], "3")

    def test_tensile_required_paths_follow_selected_corpus(self) -> None:
        smoke = validation.WORKLOAD_BY_ID["tensile-sk-sgemm-runtime-smoke"]
        external = validation.WORKLOAD_BY_ID["tensile-sk-sgemm-quick"]
        resolved = mock.Mock(
            tensilelite=Path("/toolchain/tensilelite"),
            rocm=Path("/toolchain/rocm"),
        )
        with (
            mock.patch.object(
                validation,
                "resolve_tensile_validation_paths",
                return_value=resolved,
            ),
            mock.patch.object(
                validation,
                "git_identity",
                side_effect=lambda path: {"root": str(path)},
            ),
        ):
            smoke_paths = validation._required_paths(Path("/workspace"), (smoke,))
            external_paths = validation._required_paths(Path("/workspace"), (external,))
            smoke_sources = validation._source_identities(Path("/workspace"), smoke)
        self.assertNotIn("rocjitsu-test-corpus", smoke_paths)
        self.assertNotIn(
            "/workspace/rocjitsu-test-corpus",
            {source["root"] for source in smoke_sources},
        )
        self.assertEqual(
            external_paths["rocjitsu-test-corpus"],
            Path("/workspace/rocjitsu-test-corpus"),
        )

    def test_tensile_input_inventory_covers_the_resolved_toolchain(self) -> None:
        workload = validation.WORKLOAD_BY_ID["tensile-sk-sgemm-runtime-smoke"]
        resolved = mock.Mock(
            tensilelite=Path("/toolchain/tensilelite"),
            rocm=Path("/toolchain/rocm"),
            client=Path("/toolchain/client"),
            wrapper=Path("/toolchain/wrapper"),
            rocjitsu=Path("/toolchain/rocjitsu"),
            rocjitsu_config=Path("/toolchain/gfx1250.json"),
            llvm_readelf=Path("/toolchain/llvm-readelf"),
        )
        with mock.patch.object(
            validation,
            "resolve_tensile_validation_paths",
            return_value=resolved,
        ):
            inputs = validation._input_files(Path("/workspace"), "gfx1250", workload)
        self.assertEqual(
            set(inputs),
            {
                "python",
                "workload-source",
                "support-source",
                "config",
                "client",
                "wrapper",
                "rocjitsu",
                "rocjitsu-config",
                "llvm-readelf",
                "amdclang++",
            },
        )
        self.assertEqual(inputs["client"], resolved.client)
        self.assertEqual(inputs["amdclang++"], resolved.rocm / "bin" / "amdclang++")

    def test_tensile_doctor_checks_executable_permissions(self) -> None:
        workload = validation.WORKLOAD_BY_ID["tensile-sk-sgemm-runtime-smoke"]
        with temporary_root() as root:
            inputs = {}
            for label in (
                "python",
                "workload-source",
                "support-source",
                "config",
                "client",
                "wrapper",
                "rocjitsu",
                "rocjitsu-config",
                "llvm-readelf",
                "amdclang++",
            ):
                path = root / label
                path.write_text("input\n", encoding="utf-8")
                path.chmod(0o755)
                inputs[label] = path
            inputs["wrapper"].chmod(0o644)
            with (
                mock.patch.object(validation, "_required_paths", return_value={}),
                mock.patch.object(validation, "_input_files", return_value=inputs),
                mock.patch.object(validation.shutil, "which", return_value="/tool"),
            ):
                rejected = validation._doctor(root, "gfx1250", (workload.id,))
                inputs["wrapper"].chmod(0o755)
                accepted = validation._doctor(root, "gfx1250", (workload.id,))
        self.assertFalse(rejected["ok"])
        self.assertFalse(
            rejected["paths"][f"workload:{workload.id}:wrapper"]["present"]
        )
        self.assertTrue(accepted["ok"], accepted)

    def test_pytorch_json_reports_independent_variant_medians(self) -> None:
        document = {
            "one-cta": {"median_ms": 4.0, "oracle_passed": True},
            "two-cta-cluster": {"median_ms": 7.0, "oracle_passed": True},
        }
        self.assertEqual(
            validation._json_medians(json.dumps(document), "Pytorch"),
            {"one-cta": 4.0, "two-cta-cluster": 7.0},
        )

    def test_single_qwen_iteration_is_its_median(self) -> None:
        with temporary_root() as root:
            result = root / "benchmark.json"
            result.write_text(
                json.dumps(
                    {
                        "benchmarks": [
                            {
                                "name": "BM_main/process_time/real_time",
                                "run_type": "iteration",
                                "repetitions": 1,
                                "real_time": 12.5,
                                "time_unit": "ms",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(validation._benchmark_median(result), 12.5)

    def test_explain_expands_commands_and_marks_only_real_tuning(self) -> None:
        audit = validation._explain_contract(
            Path("/workspace"),
            "gfx1201",
            ("qwen-prefill",),
            validation.PROFILE_IDS,
            None,
            allow_reference=False,
        )
        workload = audit["workloads"][0]
        expected = validation._workload_command(
            Path("/workspace"),
            "gfx1201",
            validation.WORKLOAD_BY_ID["qwen-prefill"],
            "clean",
            Path("$ARTIFACT_ROOT/qwen-prefill/clean/$PROFILE/benchmark-0.json"),
        )
        self.assertEqual(workload["commands"]["clean"]["payload_argv"], expected)
        settings = {
            item["name"]: item
            for profile in workload["profiles"]
            for item in profile["settings"]
        }
        self.assertNotIn("CTEST_PARALLEL_LEVEL", settings)
        for forbidden in validation.ORDINARY_FORBIDDEN_ENVIRONMENT:
            self.assertNotIn(forbidden, settings)
        sampled = next(
            profile for profile in workload["profiles"] if profile["id"] == "sampled"
        )
        record_replay = next(
            profile
            for profile in workload["profiles"]
            if profile["id"] == "record-replay"
        )
        self.assertEqual(
            {item["name"] for item in sampled["implicit_runtime_defaults"]},
            {
                "RJ_CONSAN_MOI_TRACK_BARRIERS",
                "RJ_CONSAN_MOI_TRACK_ATOMICS",
                "RJ_CONSAN_MOI_RUNTIME_SAMPLE_OFFSET",
                "RJ_CONSAN_MOI_RUNTIME_SAMPLE_STRIDE",
            },
        )
        self.assertEqual(sampled["usability_exceptions"], [])
        self.assertEqual(
            {item["name"] for item in record_replay["implicit_runtime_defaults"]},
            {
                "RJ_CONSAN_MOI_TRACK_BARRIERS",
                "RJ_CONSAN_MOI_TRACK_ATOMICS",
                "RJ_CONSAN_MOI_RUNTIME_SAMPLE_OFFSET",
                "RJ_CONSAN_MOI_RUNTIME_SAMPLE_STRIDE",
            },
        )
        self.assertEqual(
            audit["usability_audit"]["coverage_limiting_controls_present"], []
        )
        self.assertEqual(
            audit["usability_audit"]["explicit_event_family_overrides"], []
        )
        self.assertEqual(audit["usability_audit"]["workload_specific_tuning"], [])
        sampled_defaults = next(
            item
            for item in audit["usability_audit"]["automatic_profile_defaults"]
            if item["profile"] == "sampled"
        )
        self.assertEqual(
            sampled_defaults["settings"]["RJ_CONSAN_MOI_RUNTIME_SAMPLE_STRIDE"],
            "256",
        )
        record_replay_defaults = next(
            item
            for item in audit["usability_audit"]["automatic_profile_defaults"]
            if item["profile"] == "record-replay"
        )
        self.assertEqual(
            record_replay_defaults["settings"]["RJ_CONSAN_MOI_RUNTIME_SAMPLE_STRIDE"],
            "65536",
        )

    def test_explain_audits_reference_fault_outcomes_and_trial_knobs(self) -> None:
        path = Path(__file__).with_name(
            "consan_validation_faults_gfx1201_reference.json"
        )
        audit = validation._explain_contract(
            Path("/workspace"),
            "gfx1201",
            ("qwen-prefill",),
            validation.PROFILE_IDS,
            path,
            allow_reference=True,
        )
        fault = audit["workloads"][0]["faults"][0]
        sampled = next(
            item
            for item in fault["profile_expectations"]
            if item["profile"] == "sampled"
        )
        self.assertEqual(sampled["detector"], "statistical")
        self.assertEqual(sampled["oracle"], "fail")
        self.assertEqual(sampled["trial_count"], 32)
        self.assertEqual(
            sampled["trials"][0]["overrides"][0]["name"],
            "RJ_CONSAN_MOI_RUNTIME_SAMPLE_OFFSET",
        )
        self.assertIn("at least 1", sampled["required_diagnostic"])
        inline = next(
            item
            for item in fault["profile_expectations"]
            if item["profile"] == "inline-shadow"
        )
        effective = {
            setting["name"]: setting["value"]
            for setting in inline["trials"][0]["effective_settings"]
        }
        implicit = {
            setting["name"]: setting["value"]
            for setting in inline["trials"][0]["implicit_runtime_defaults"]
        }
        self.assertEqual(effective["RJ_CONSAN_MOI_REQUIRE_DIAGNOSTICS"], "1")
        self.assertNotIn("RJ_CONSAN_MOI_FORBID_DIAGNOSTICS", effective)
        self.assertEqual(implicit, validation.ORDINARY_MOI_RUNTIME_DEFAULTS)
        self.assertIn("$FAULT_SPEC", fault["validator_argv_template"])

    def test_tp1_decode_row_does_not_repeat_prefill(self) -> None:
        workload = validation.WORKLOAD_BY_ID["tp1-decode-combined"]
        command = validation._workload_command(
            Path("/workspace"), "gfx1201", workload, "clean", Path("/unused")
        )
        self.assertEqual(command[command.index("--mode") + 1], "decode-combined")

    def test_gfx1250_d128_fault_uses_fast_oracle_variant(self) -> None:
        workload = validation.WORKLOAD_BY_ID["d128-block"]
        command = validation._workload_command(
            Path("/workspace"), "gfx1250", workload, "fault", Path("/unused")
        )
        self.assertEqual(
            command[1],
            (
                "--gtest_filter=HipMoiGfx1250D128AttentionBlock."
                "SampledFastContextMatchesHostReference"
            ),
        )
        rdna_command = validation._workload_command(
            Path("/workspace"), "gfx1201", workload, "fault", Path("/unused")
        )
        self.assertEqual(
            rdna_command[1],
            "--gtest_filter=HipMoiRdna4D128AttentionBlock.*",
        )

    def test_overhead_uses_bracketing_baseline_mean_and_maximum_mode(self) -> None:
        results = [
            {"profile": "baseline", "timing_median_ms": {"a": 2.0, "b": 4.0}},
            {"profile": "sampled", "timing_median_ms": {"a": 6.0, "b": 10.0}},
            {"profile": "baseline", "timing_median_ms": {"a": 4.0, "b": 6.0}},
        ]
        summary = validation._overhead_summary(results)
        self.assertEqual(summary["paired_baseline_median_ms"], {"a": 3.0, "b": 5.0})
        self.assertEqual(summary["profiles"]["sampled"]["cell_slowdown"], 2.0)

    def test_inventory_parser_deduplicates_exact_identities(self) -> None:
        output = "\n".join(
            (
                "ConSan fault site reader=1 identity=site-a kind=barrier "
                "sync_sequence=sequence-b",
                "ConSan fault site reader=2 identity=site-a kind=barrier",
                "ConSan sync sequence reader=1 identity=sequence-a kind=barrier",
                "ConSan barrier destination reader=1 identity=destination-a container=k",
            )
        )
        self.assertEqual(
            validation._inventory_records(output),
            {
                "sites": ["site-a"],
                "sequences": ["sequence-a", "sequence-b"],
                "destinations": ["destination-a"],
            },
        )

    def test_inventory_completion_requires_relevant_site_then_matching_coverage(
        self,
    ) -> None:
        unrelated = "\n".join(
            (
                "ConSan fault site reader=7 identity=a kind=atomic container=k",
                "ConSan coverage reader=7 analysis_complete=true",
            )
        )
        self.assertFalse(
            validation._inventory_collection_complete(unrelated, "barrier-drop")
        )
        wrong_reader = "\n".join(
            (
                "ConSan fault site reader=7 identity=a kind=barrier container=k",
                "ConSan coverage reader=8 analysis_complete=true",
            )
        )
        self.assertFalse(
            validation._inventory_collection_complete(wrong_reader, "barrier-drop")
        )
        complete = "\n".join(
            (
                "ConSan fault site reader=7 identity=a kind=barrier container=k",
                "ConSan sync sequence reader=7 identity=s kind=barrier",
                "ConSan coverage reader=7 analysis_complete=false",
            )
        )
        self.assertTrue(
            validation._inventory_collection_complete(complete, "barrier-drop")
        )

    def test_family_inventory_records_exclude_unrelated_sites(self) -> None:
        output = "\n".join(
            (
                "ConSan fault site reader=7 identity=h|kind=ordinary-memory|pc=1 "
                "kind=ordinary-memory sync_sequence=-",
                "ConSan fault site reader=7 identity=h|kind=barrier|pc=2 "
                "kind=barrier sync_sequence=h|event=barrier|pc=2",
                "ConSan sync sequence reader=7 identity=h|event=atomic|pc=3 kind=atomic",
            )
        )
        self.assertEqual(
            validation._inventory_records(output, "barrier-drop"),
            {
                "sites": ["h|kind=barrier|pc=2"],
                "sequences": ["h|event=barrier|pc=2"],
                "destinations": [],
            },
        )

    def test_atomic_inventory_completion_rejects_barrier_only_reader(self) -> None:
        output = "\n".join(
            (
                "ConSan fault site reader=3 identity=a kind=barrier container=k",
                "ConSan coverage reader=3 analysis_complete=true",
            )
        )
        self.assertFalse(
            validation._inventory_collection_complete(output, "atomic-weaken-order")
        )

    def test_inventory_runner_stops_after_static_collection(self) -> None:
        program = "; ".join(
            (
                "import time",
                "print('ConSan fault site reader=9 identity=a kind=barrier container=k', flush=True)",
                "print('ConSan coverage reader=9 analysis_complete=false', flush=True)",
                "time.sleep(30)",
            )
        )
        with temporary_root() as root:
            returncode, elapsed, output, complete, outcome = (
                validation._run_inventory_process(
                    [sys.executable, "-c", program],
                    os.environ.copy(),
                    root / "inventory.log",
                    5,
                    "barrier-drop",
                )
            )
        self.assertLess(elapsed, 3)
        self.assertNotEqual(returncode, 0)
        self.assertIn("ConSan coverage reader=9", output)
        self.assertTrue(complete)
        self.assertEqual(outcome, "static-inventory-complete")

    def test_inventory_runner_rejects_timeout_before_matching_coverage(self) -> None:
        program = "; ".join(
            (
                "import time",
                "print('ConSan fault site reader=9 identity=a kind=barrier container=k', flush=True)",
                "time.sleep(30)",
            )
        )
        with temporary_root() as root:
            returncode, elapsed, output, complete, outcome = (
                validation._run_inventory_process(
                    [sys.executable, "-c", program],
                    os.environ.copy(),
                    root / "inventory.log",
                    1,
                    "barrier-drop",
                )
            )
        self.assertLess(elapsed, 3)
        self.assertEqual(returncode, 124)
        self.assertIn("validation timeout after 1s", output)
        self.assertFalse(complete)
        self.assertEqual(outcome, "timeout")

    def test_fault_parser_accepts_paired_health_command_overrides(self) -> None:
        args = validation._parse_args(
            [
                "fault",
                "--workload",
                "jakub-attention",
                "--spec",
                "/tmp/spec.json",
                "--fault",
                "barrier-drop",
                "--artifact-root",
                "/tmp/artifacts",
                "--health-timeout",
                "75",
                "--health-command-json",
                '["/bin/true"]',
                "--smoke-command-json",
                '["/tmp/smoke", "--short"]',
            ]
        )
        self.assertEqual(args.health_command_json, ["/bin/true"])
        self.assertEqual(args.smoke_command_json, ["/tmp/smoke", "--short"])
        self.assertEqual(args.health_timeout, 75.0)

    def test_fault_parser_rejects_unpaired_health_command_override(self) -> None:
        with self.assertRaises(SystemExit):
            validation._parse_args(
                [
                    "fault",
                    "--workload",
                    "jakub-attention",
                    "--spec",
                    "/tmp/spec.json",
                    "--fault",
                    "barrier-drop",
                    "--artifact-root",
                    "/tmp/artifacts",
                    "--health-command-json",
                    '["/bin/true"]',
                ]
            )

    def test_marker_smoke_stops_after_independent_success_marker(self) -> None:
        script = Path(__file__).with_name("consan_marker_smoke.py")
        program = "; ".join(
            (
                "import time",
                "print('checked result: PASS', flush=True)",
                "time.sleep(30)",
            )
        )
        result = subprocess.run(
            [
                sys.executable,
                str(script),
                "--success-marker",
                "checked result: PASS",
                "--timeout",
                "5",
                "--",
                sys.executable,
                "-c",
                program,
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=10,
            check=False,
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("checked result: PASS", result.stdout)

    def test_marker_smoke_rejects_natural_exit_without_marker(self) -> None:
        script = Path(__file__).with_name("consan_marker_smoke.py")
        result = subprocess.run(
            [
                sys.executable,
                str(script),
                "--success-marker",
                "PASS",
                "--",
                sys.executable,
                "-c",
                "print('FAIL')",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=10,
            check=False,
        )
        self.assertEqual(result.returncode, 1)

    def test_fault_inventory_enables_family_analysis_without_a_selector(self) -> None:
        barrier = validation._fault_inventory_environment("barrier-drop")
        self.assertEqual(barrier, {"RJ_CONSAN_FAULT_DROP_BARRIER": "1"})
        atomic = validation._fault_inventory_environment("atomic-weaken-order")
        self.assertEqual(
            atomic,
            {
                "RJ_CONSAN_FAULT_ATOMIC_WEAKEN_ORDER": "1",
                "RJ_CONSAN_FAULT_ATOMIC_ORDER_EDGE": "release",
            },
        )
        self.assertFalse(any(name.endswith("_IDENTITY") for name in atomic))

    def test_fault_template_matches_target_barrier_geometry(self) -> None:
        workload = validation.WORKLOAD_BY_ID["d128-block"]
        rdna_fault = validation._fault_template("gfx1201", workload)["faults"][0]
        self.assertIn(
            "RJ_CONSAN_FAULT_BARRIER_SEQUENCE_IDENTITY",
            rdna_fault["environment"],
        )
        for target in ("gfx942", "gfx950"):
            with self.subTest(target=target):
                cdna_fault = validation._fault_template(target, workload)["faults"][0]
                self.assertNotIn(
                    "RJ_CONSAN_FAULT_BARRIER_SEQUENCE_IDENTITY",
                    cdna_fault["environment"],
                )
                self.assertEqual(
                    cdna_fault["environment"]["RJ_CONSAN_FAULT_SITE_IDENTITY"],
                    "REPLACE_FROM_INVENTORY",
                )

    def test_fault_acceptance_rejects_an_unattributed_process_signal(self) -> None:
        accepted, reasons = validation._fault_acceptance(
            {
                "mutation": {
                    "accounting_schema_version": 2,
                    "requested": 1,
                    "planned": 1,
                    "applied": 1,
                    "installation_evidence_complete": True,
                },
                "sanitizer": {"outcome": "not_detected"},
                "oracle": {"outcome": "pass"},
                "execution": {
                    "outcome": "signal",
                    "timed_out": False,
                    "health_before": {"healthy": True},
                    "health_after": {"healthy": True},
                },
            },
            {"detector": "not_detected", "oracle": "pass"},
        )
        self.assertFalse(accepted)
        self.assertIn("invalid execution outcome=signal", reasons)

    def test_fault_acceptance_rejects_a_discarded_mutation(self) -> None:
        accepted, reasons = validation._fault_acceptance(
            {
                "mutation": {
                    "accounting_schema_version": 2,
                    "requested": 1,
                    "planned": 1,
                    "applied": 1,
                    "installation_evidence_complete": True,
                    "discarded_applied": 1,
                },
                "sanitizer": {"outcome": "detected"},
                "oracle": {"outcome": "fail"},
                "execution": {
                    "outcome": "passed",
                    "timed_out": False,
                    "health_before": {"healthy": True},
                    "health_after": {"healthy": True},
                },
            },
            {"detector": "detected", "oracle": "fail"},
        )
        self.assertFalse(accepted)
        self.assertIn("discarded_applied=1", reasons)

    def test_fault_acceptance_rejects_incomplete_installation_evidence(self) -> None:
        accepted, reasons = validation._fault_acceptance(
            {
                "mutation": {
                    "accounting_schema_version": 2,
                    "requested": 1,
                    "planned": 1,
                    "applied": 1,
                    "installation_evidence_complete": False,
                },
                "sanitizer": {"outcome": "not_detected"},
                "oracle": {"outcome": "pass"},
                "execution": {
                    "outcome": "passed",
                    "timed_out": False,
                    "health_before": {"healthy": True},
                    "health_after": {"healthy": True},
                },
            },
            {"detector": "not_detected", "oracle": "pass"},
        )
        self.assertFalse(accepted)
        self.assertIn("installation_evidence_complete=False", reasons)

        accepted, reasons = validation._fault_acceptance(
            {
                "mutation": {
                    "accounting_schema_version": 1,
                    "requested": 1,
                    "planned": 1,
                    "applied": 1,
                },
                "sanitizer": {"outcome": "not_detected"},
                "oracle": {"outcome": "pass"},
                "execution": {
                    "outcome": "passed",
                    "timed_out": False,
                    "health_before": {"healthy": True},
                    "health_after": {"healthy": True},
                },
            },
            {"detector": "not_detected", "oracle": "pass"},
        )
        self.assertFalse(accepted)
        self.assertIn(
            "accounting_schema_version=1, expected=2; rerun required",
            reasons,
        )
        self.assertFalse(
            any("installation_evidence_complete" in reason for reason in reasons)
        )

    def test_fault_does_not_execute_a_spec_not_applicable_profile(self) -> None:
        workload = validation.WORKLOAD_BY_ID["d128-block"]
        fault = {
            "id": "drop",
            "family": "barrier-drop",
            "environment": {
                "RJ_CONSAN_FAULT_DROP_BARRIER": "1",
                "RJ_CONSAN_FAULT_SITE_IDENTITY": "site-a",
            },
            "profiles": {
                "record-replay": {
                    "disposition": "not-applicable",
                    "reason": "profile has no qualified fault",
                    "tracking_issue": "bd-test",
                }
            },
        }
        with temporary_root() as root:
            spec = root / "fault.json"
            spec.write_text("{}", encoding="utf-8")
            args = validation._parse_args(
                [
                    "--target",
                    "gfx1201",
                    "fault",
                    "--workload",
                    workload.id,
                    "--profile",
                    "record-replay",
                    "--spec",
                    str(spec),
                    "--fault",
                    fault["id"],
                    "--artifact-root",
                    str(root / "artifacts"),
                    "--allow-destructive",
                ]
            )
            provenance = root / "provenance.json"
            with (
                mock.patch.object(
                    validation, "_workspace_from_environment", return_value=root
                ),
                mock.patch.object(validation, "_doctor", return_value={"ok": True}),
                mock.patch.object(validation, "_load_fault", return_value=fault),
                mock.patch.object(
                    validation, "_write_provenance", return_value=provenance
                ),
                mock.patch.object(
                    validation,
                    "_health_smoke_command",
                    return_value=["/bin/true"],
                ),
                mock.patch.object(validation.subprocess, "run") as run,
                redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(validation._fault(args), 0)
            run.assert_not_called()
            summary = json.loads(
                (
                    root
                    / "artifacts"
                    / workload.id
                    / "faults"
                    / fault["id"]
                    / "summary.json"
                ).read_text(encoding="utf-8")
            )
        self.assertTrue(summary["accepted"])
        self.assertEqual(
            summary["profiles"],
            [
                {
                    "accepted": True,
                    "disposition": "not-applicable",
                    "profile": "record-replay",
                    "reason": "profile has no qualified fault",
                    "tracking_issue": "bd-test",
                }
            ],
        )

    def test_fault_spec_requires_target_workload_and_exact_mutation(self) -> None:
        workload = validation.WORKLOAD_BY_ID["d128-block"]
        document = {
            "schema_version": validation.SCHEMA_VERSION,
            "target": "gfx1201",
            "workload": workload.id,
            "review_required": False,
            "faults": [
                {
                    "id": "drop",
                    "family": "barrier-drop",
                    "environment": {
                        "RJ_CONSAN_FAULT_DROP_BARRIER": "1",
                        "RJ_CONSAN_FAULT_SITE_IDENTITY": "site-a",
                        "RJ_CONSAN_FAULT_BARRIER_SEQUENCE_IDENTITY": "sequence-a",
                    },
                }
            ],
        }
        with temporary_root() as root:
            path = root / "faults.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            loaded = validation._load_fault(path, "gfx1201", workload, "drop")
        self.assertEqual(loaded["id"], "drop")

    def test_checked_in_gfx1201_fault_reference_is_a_valid_manifest_subset(
        self,
    ) -> None:
        path = Path(__file__).with_name(
            "consan_validation_faults_gfx1201_reference.json"
        )
        document = json.loads(path.read_text(encoding="utf-8"))
        with self.assertRaisesRegex(validation.ValidationError, "reference-only"):
            validation._load_fault(
                path,
                "gfx1201",
                validation.WORKLOAD_BY_ID["qwen-prefill"],
                "barrier-drop",
            )
        manifest_ids = {
            workload.id for workload in validation._workloads_for_target("gfx1201")
        }
        self.assertLessEqual(set(document["workloads"]), manifest_ids)
        for workload_id, workload_document in document["workloads"].items():
            workload = validation.WORKLOAD_BY_ID[workload_id]
            for fault_document in workload_document["faults"]:
                fault_id = fault_document["id"]
                fault = validation._load_fault(
                    path,
                    "gfx1201",
                    workload,
                    fault_id,
                    allow_reference=True,
                )
                for profile in validation.PROFILE_IDS:
                    policy, trials = validation._fault_trials(fault, profile)
                    self.assertTrue(trials)
                    self.assertIn(
                        policy.get("detector"),
                        {"detected", "not_detected", "statistical"},
                    )
        qwen = validation.WORKLOAD_BY_ID["qwen-prefill"]
        fault = validation._load_fault(
            path,
            "gfx1201",
            qwen,
            "barrier-drop",
            allow_reference=True,
        )
        policy, trials = validation._fault_trials(fault, "sampled")
        self.assertEqual(policy["minimum_detections"], 1)
        self.assertEqual(
            policy["environment"]["RJ_CONSAN_MOI_RUNTIME_SAMPLE_STRIDE"],
            "256",
        )
        self.assertEqual(len(trials), 32)
        self.assertEqual(trials[0], {"RJ_CONSAN_MOI_RUNTIME_SAMPLE_OFFSET": "0"})
        self.assertEqual(trials[-1], {"RJ_CONSAN_MOI_RUNTIME_SAMPLE_OFFSET": "31"})

    def test_checked_in_gfx1201_native_record_replay_fault_is_runnable(self) -> None:
        path = Path(__file__).with_name(
            "consan_validation_faults_gfx1201_native_record_replay.json"
        )
        workload = validation.WORKLOAD_BY_ID["llama-rdna4-mul-mat-vec-q"]
        fault = validation._load_fault(path, "gfx1201", workload, "barrier-drop")
        policy, trials = validation._fault_trials(fault, "record-replay")
        self.assertEqual(policy["detector"], "detected")
        self.assertEqual(policy["oracle"], "any")
        self.assertEqual(trials, [{}])
        self.assertIn(
            "pc=0x000000000000b8cc",
            fault["environment"]["RJ_CONSAN_FAULT_SITE_IDENTITY"],
        )

    def test_gfx1201_reference_retains_repeated_d128_record_replay_detection(
        self,
    ) -> None:
        path = Path(__file__).with_name(
            "consan_validation_faults_gfx1201_reference.json"
        )
        workload = validation.WORKLOAD_BY_ID["d128-pressure"]
        fault = validation._load_fault(
            path,
            "gfx1201",
            workload,
            "barrier-drop",
            allow_reference=True,
        )
        policy, trials = validation._fault_trials(fault, "record-replay")
        self.assertEqual(policy["detector"], "detected")
        self.assertEqual(policy["oracle"], "fail")
        self.assertEqual(trials, [{}, {}, {}, {}, {}])
        self.assertIn(
            "pc=0x000000000000a51c",
            fault["environment"]["RJ_CONSAN_FAULT_SITE_IDENTITY"],
        )

    def test_checked_in_gfx1201_native_matvec_miss_is_runnable(self) -> None:
        path = Path(__file__).with_name(
            "consan_validation_faults_gfx1201_native_matvec_miss.json"
        )
        workload = validation.WORKLOAD_BY_ID["llama-rdna4-mul-mat-vec-q"]
        fault = validation._load_fault(path, "gfx1201", workload, "barrier-drop")
        for profile in ("supercollider", "record-replay", "sampled"):
            policy, trials = validation._fault_trials(fault, profile)
            self.assertEqual(policy["detector"], "not_detected")
            self.assertEqual(policy["oracle"], "pass")
            self.assertEqual(trials, [{}])
        self.assertIn(
            "pc=0x000000000000b8cc",
            fault["environment"]["RJ_CONSAN_FAULT_SITE_IDENTITY"],
        )

    def test_checked_in_gfx1201_native_matvec_effective_fault_is_runnable(self) -> None:
        path = Path(__file__).with_name(
            "consan_validation_faults_gfx1201_native_matvec_rr_effective.json"
        )
        workload = validation.WORKLOAD_BY_ID["llama-rdna4-mul-mat-vec-q"]
        fault = validation._load_fault(path, "gfx1201", workload, "barrier-drop")
        for profile in ("record-replay", "sampled"):
            policy, trials = validation._fault_trials(fault, profile)
            self.assertEqual(policy["detector"], "not_detected")
            self.assertEqual(policy["oracle"], "fail")
            self.assertEqual(trials, [{}])

    def test_checked_in_gfx1201_native_rms_fault_is_runnable(self) -> None:
        path = Path(__file__).with_name(
            "consan_validation_faults_gfx1201_native_rms_norm.json"
        )
        workload = validation.WORKLOAD_BY_ID["llama-rdna4-rms-norm"]
        fault = validation._load_fault(path, "gfx1201", workload, "barrier-drop")
        for profile in validation.PROFILE_IDS:
            policy, trials = validation._fault_trials(fault, profile)
            self.assertEqual(policy["detector"], "detected")
            self.assertEqual(policy["oracle"], "any")
            self.assertEqual(trials, [{}])
        self.assertIn(
            "pc=0x0000000000001824",
            fault["environment"]["RJ_CONSAN_FAULT_SITE_IDENTITY"],
        )


if __name__ == "__main__":
    unittest.main()
