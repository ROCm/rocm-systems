#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import unittest


DBI_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DBI_DIR))
from consan_coverage_gate import (  # noqa: E402
    CoverageParseError,
    acceptance_decision,
    parse_coverage_evidence,
)
from consan_validation_test_support import (  # noqa: E402
    coverage,
    coverage_site,
    log,
    verdict,
)



class ConSanCoverageGateTest(unittest.TestCase):
    def test_accepts_complete_exact_evidence(self) -> None:
        decision = acceptance_decision(log(coverage(), verdict()))
        self.assertTrue(decision.accepted, decision.reasons)
        self.assertEqual(decision.evidence.coverage[0].reader, 7)
        self.assertEqual(decision.evidence.verdict.patched_supported["access"], (20, 20))

    def test_accepts_supercollider_aggregate_without_moi_site_rows(self) -> None:
        aggregate = coverage(
            flavor="supercollider",
            engine="supercollider",
            access_discovered="2",
            access_supported="2",
            access_selected="2",
            access_patched="2",
            barrier_discovered="0",
            barrier_supported="0",
            barrier_selected="0",
            barrier_patched="0",
            atomic_discovered="0",
            atomic_supported="0",
            atomic_selected="0",
            atomic_patched="0",
            fence_discovered="0",
            fence_supported="0",
            fence_selected="0",
            fence_patched="0",
        )
        decision = acceptance_decision(
            log(
                aggregate,
                verdict(access="2/2", barrier="0/0", atomic="0/0", fence="0/0"),
            )
        )
        self.assertTrue(decision.accepted, decision.reasons)
        self.assertEqual(decision.evidence.coverage[0].flavor, "supercollider")

    def test_rejects_moi_site_rows_attached_to_supercollider(self) -> None:
        aggregate = coverage(
            flavor="supercollider",
            engine="supercollider",
            access_discovered="1",
            access_supported="1",
            access_selected="1",
            access_patched="1",
            barrier_discovered="0",
            barrier_supported="0",
            barrier_selected="0",
            barrier_patched="0",
            atomic_discovered="0",
            atomic_supported="0",
            atomic_selected="0",
            atomic_patched="0",
            fence_discovered="0",
            fence_supported="0",
            fence_selected="0",
            fence_patched="0",
        )
        with self.assertRaisesRegex(CoverageParseError, "has MOI coverage_site rows"):
            parse_coverage_evidence(
                log(
                    aggregate,
                    coverage_site(
                        disposition="supported",
                        reason="none",
                        outcome="patched",
                        lowering_reason="none",
                    ),
                    verdict(access="1/1", barrier="0/0", atomic="0/0", fence="0/0"),
                    synthesize_sites=False,
                )
            )

    def test_accepts_multiple_consistent_code_objects(self) -> None:
        first = coverage(
            reader=7,
            access_discovered="12",
            access_supported="12",
            access_selected="12",
            access_patched="12",
        )
        second = coverage(
            reader=8,
            access_discovered="8",
            access_supported="8",
            access_selected="8",
            access_patched="8",
            barrier_supported="0",
            barrier_patched="0",
            atomic_supported="0",
            atomic_patched="0",
            fence_supported="0",
            fence_patched="0",
        )
        decision = acceptance_decision(
            log(first, second, verdict(applicable_code_objects="2"))
        )
        self.assertTrue(decision.accepted, decision.reasons)

    def test_aggregates_multiple_process_final_verdicts(self) -> None:
        nonapplicable = coverage(
            reader=8,
            analysis_complete="true",
            access_discovered="0",
            access_supported="0",
            access_selected="0",
            access_patched="0",
            barrier_supported="0",
            barrier_patched="0",
            atomic_supported="0",
            atomic_patched="0",
            fence_supported="0",
            fence_patched="0",
        )
        decision = acceptance_decision(
            log(
                coverage(),
                verdict(),
                nonapplicable,
                verdict(
                    applicable="false",
                    analysis_complete="false",
                    static_complete="false",
                    applicable_code_objects="0",
                    access="0/0",
                    barrier="0/0",
                    atomic="0/0",
                    fence="0/0",
                ),
            )
        )
        self.assertTrue(decision.accepted, decision.reasons)
        self.assertEqual(decision.evidence.verdict.applicable_code_objects, 1)

    def test_rejects_each_required_false_verdict(self) -> None:
        for name in (
            "applicable",
            "analysis_complete",
            "static_complete",
            "dynamic_complete",
        ):
            with self.subTest(name=name):
                updates = {name: "false"}
                if name in ("analysis_complete", "static_complete"):
                    updates["incomplete_code_objects"] = "1"
                    updates["access"] = "19/20"
                    coverage_line = coverage(
                        analysis_complete="false",
                        access_patched="19",
                        access_placement_or_lowering_failed="1",
                    )
                else:
                    coverage_line = coverage()
                if name == "applicable":
                    updates["applicable_code_objects"] = "0"
                    coverage_line = coverage(
                        access_discovered="0",
                        access_supported="0",
                        access_selected="0",
                        access_patched="0",
                        barrier_supported="0",
                        barrier_patched="0",
                        atomic_supported="0",
                        atomic_patched="0",
                        fence_supported="0", fence_patched="0",
                    )
                    updates.update(access="0/0", barrier="0/0", atomic="0/0", fence="0/0")
                decision = acceptance_decision(log(coverage_line, verdict(**updates)))
                self.assertFalse(decision.accepted)
                self.assertIn(f"verdict {name}=false", decision.reasons)

    def test_rejects_patched_supported_mismatch_even_if_booleans_claim_complete(self) -> None:
        with self.assertRaises(CoverageParseError):
            acceptance_decision(
                log(
                    coverage(access_patched="19", analysis_complete="true"),
                    verdict(access="19/20"),
                )
            )

    def test_rejects_cross_object_mismatches_hidden_by_matching_aggregate(self) -> None:
        first = coverage(
            reader=7,
            access_discovered="10",
            access_supported="10",
            access_selected="10",
            access_patched="9",
        )
        second = coverage(
            reader=8,
            access_discovered="10",
            access_supported="10",
            access_selected="10",
            access_patched="11",
            barrier_supported="0",
            barrier_patched="0",
            atomic_supported="0",
            atomic_patched="0",
            fence_supported="0",
            fence_patched="0",
        )
        with self.assertRaises(CoverageParseError):
            acceptance_decision(log(first, second, verdict(applicable_code_objects="2")))

    def test_rejects_unsupported_counter_even_if_booleans_claim_complete(self) -> None:
        with self.assertRaises(CoverageParseError):
            acceptance_decision(
                log(
                    coverage(
                        access_discovered="21",
                        access_unsupported="1",
                        analysis_complete="true",
                    ),
                    verdict(),
                )
            )

    def test_unsupported_only_object_is_applicable_and_incomplete(self) -> None:
        unsupported_only = coverage(
            access_discovered="1",
            access_supported="0",
            access_selected="0",
            access_patched="0",
            access_unsupported="1",
            barrier_supported="0",
            barrier_patched="0",
            atomic_supported="0",
            atomic_patched="0",
            fence_supported="0",
            fence_patched="0",
            analysis_complete="false",
        )
        decision = acceptance_decision(
            log(
                unsupported_only,
                verdict(
                    analysis_complete="false",
                    static_complete="false",
                    incomplete_code_objects="1",
                    access="0/0",
                    barrier="0/0",
                    atomic="0/0",
                    fence="0/0",
                ),
            )
        )
        self.assertFalse(decision.accepted)
        self.assertTrue(decision.evidence.coverage[0].applicable)
        self.assertIn("reader 7 access unsupported: 1", decision.reasons)

    def test_mixed_objects_cannot_hide_unsupported_only_object(self) -> None:
        unsupported_only = coverage(
            reader=8,
            access_discovered="1",
            access_supported="0",
            access_selected="0",
            access_patched="0",
            access_unsupported="1",
            barrier_supported="0",
            barrier_patched="0",
            atomic_supported="0",
            atomic_patched="0",
            fence_supported="0",
            fence_patched="0",
            analysis_complete="false",
        )
        decision = acceptance_decision(
            log(
                coverage(),
                unsupported_only,
                verdict(
                    analysis_complete="false",
                    static_complete="false",
                    applicable_code_objects="2",
                    incomplete_code_objects="1",
                ),
            )
        )
        self.assertFalse(decision.accepted)
        self.assertIn("reader 8 static analysis incomplete", decision.reasons)
        self.assertIn("reader 8 access unsupported: 1", decision.reasons)

    def test_each_failure_category_fails_closed_for_each_site_kind(self) -> None:
        supported_by_kind = {"access": 20, "barrier": 4, "atomic": 2, "fence": 1}
        for kind, supported in supported_by_kind.items():
            for category in (
                "unsupported",
                "resource_failed",
                "placement_or_lowering_failed",
                "expert_limit_omitted",
            ):
                with self.subTest(kind=kind, category=category):
                    coverage_updates = {"analysis_complete": "false"}
                    if category == "unsupported":
                        coverage_updates[f"{kind}_discovered"] = str(supported + 1)
                        coverage_updates[f"{kind}_unsupported"] = "1"
                        patched = supported
                    elif category == "expert_limit_omitted":
                        coverage_updates["expert_limit"] = "true"
                        coverage_updates[f"{kind}_selected"] = str(supported - 1)
                        coverage_updates[f"{kind}_patched"] = str(supported - 1)
                        coverage_updates[f"{kind}_{category}"] = "1"
                        patched = supported - 1
                    else:
                        coverage_updates[f"{kind}_patched"] = str(supported - 1)
                        coverage_updates[f"{kind}_{category}"] = "1"
                        patched = supported - 1
                    decision = acceptance_decision(
                        log(
                            coverage(**coverage_updates),
                            verdict(
                                analysis_complete="false",
                                static_complete="false",
                                incomplete_code_objects="1",
                                **{kind: f"{patched}/{supported}"},
                            ),
                        )
                    )
                    self.assertFalse(decision.accepted)
                    self.assertIn(f"reader 7 {kind} {category}: 1", decision.reasons)

    def test_retains_actual_hook_site_outcomes_and_detailed_reasons(self) -> None:
        text = log(
                coverage(
                    analysis_complete="false",
                    access_discovered="21",
                    access_unsupported="1",
                    barrier_patched="3",
                    barrier_resource_failed="1",
                    atomic_patched="1",
                    atomic_placement_or_lowering_failed="1",
                ),
                coverage_site(),
                coverage_site(
                    kind="barrier",
                    disposition="supported",
                    reason="none",
                    outcome="resource_failed",
                    lowering_reason="unsupported_resource_plan",
                    resource_reason="dynamic_stack",
                    text="0x20",
                    mnemonic="s_barrier_wait",
                ),
                coverage_site(
                    kind="atomic",
                    disposition="supported",
                    reason="none",
                    outcome="placement_or_lowering_failed",
                    lowering_reason="instrumentation_patch_missing",
                    text="0x30",
                    mnemonic="global_atomic_add",
                ),
                coverage_site(
                    kind="fence",
                    disposition="supported",
                    reason="none",
                    outcome="patched",
                    lowering_reason="none",
                    text="0x40",
                    mnemonic="fence",
                ),
                verdict(
                    analysis_complete="false",
                    static_complete="false",
                    incomplete_code_objects="1",
                    barrier="3/4",
                    atomic="1/2",
                ),
            )
        evidence = parse_coverage_evidence(text)
        self.assertEqual(len(evidence.sites), 28)
        self.assertEqual(evidence.sites[0].reason, "unsupported_mnemonic")
        self.assertEqual(evidence.sites[1].outcome, "resource_failed")
        self.assertEqual(evidence.sites[1].resource_reason, "dynamic_stack")
        self.assertEqual(
            evidence.sites[2].lowering_reason,
            "instrumentation_patch_missing",
        )
        self.assertEqual(evidence.sites[2].text_offset, 0x30)
        self.assertEqual(evidence.sites[3].outcome, "patched")
        completed = subprocess.run(
            [sys.executable, str(DBI_DIR / "consan_coverage_gate.py")],
            input=text,
            text=True,
            stdout=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(completed.returncode, 1)
        output = json.loads(completed.stdout)
        self.assertEqual(output["coverage_sites"][1]["outcome"], "resource_failed")
        self.assertEqual(output["coverage_sites"][1]["resource_reason"], "dynamic_stack")
        self.assertEqual(output["coverage_sites"][2]["text_offset"], 0x30)
        self.assertEqual(output["coverage_sites"][3]["outcome"], "patched")

    def test_rejects_unknown_or_inconsistent_site_diagnostics(self) -> None:
        cases = (
            coverage_site(outcome="invented"),
            coverage_site(reason="none"),
            coverage_site(
                disposition="supported",
                outcome="unsupported",
            ),
            coverage_site(
                disposition="supported",
                reason="unsupported_mnemonic",
                outcome="patched",
                lowering_reason="none",
            ),
            coverage_site(
                disposition="supported",
                reason="none",
                outcome="resource_failed",
                lowering_reason="unsupported_resource_plan",
                resource_reason="none",
            ),
            coverage_site(text="16"),
            coverage_site(reader="8"),
        )
        for site_line in cases:
            with self.subTest(site_line=site_line), self.assertRaises(CoverageParseError):
                parse_coverage_evidence(log(coverage(), site_line, verdict()))

    def test_rejects_missing_or_ambiguous_records(self) -> None:
        cases = {
            "missing coverage": log(verdict()),
            "missing verdict": log(coverage()),
            "duplicate verdict": log(coverage(), verdict(), verdict()),
            "duplicate reader": log(coverage(), coverage(), verdict()),
            "zero load occurrence": log(coverage(load="0"), verdict()),
        }
        for name, text in cases.items():
            with self.subTest(name=name), self.assertRaises(CoverageParseError):
                parse_coverage_evidence(text)

    def test_accepts_compact_moi_log_without_verbose_site_inventory(self) -> None:
        evidence = parse_coverage_evidence(
            log(coverage(), verdict(), synthesize_sites=False)
        )
        self.assertEqual(evidence.sites, ())
        self.assertEqual(evidence.coverage[0].counts["access_patched"], 20)

    def test_allows_reused_reader_only_for_aggregate_supercollider_records(self) -> None:
        updates = {
            "flavor": "supercollider",
            "engine": "supercollider",
            "analysis_complete": "false",
            "access_supported": "24",
            "access_patched": "24",
            "access_unsupported": "48",
        }
        for kind in ("barrier", "atomic", "fence"):
            updates.update({
                f"{kind}_supported": "0",
                f"{kind}_patched": "0",
                f"{kind}_unsupported": "0",
            })
        record = coverage(**updates)
        evidence = parse_coverage_evidence(log(
            record,
            record,
            verdict(
                analysis_complete="false",
                static_complete="false",
                applicable_code_objects="2",
                incomplete_code_objects="2",
                access="48/48",
                barrier="0/0",
                atomic="0/0",
                fence="0/0",
            ),
            synthesize_sites=False,
        ))
        self.assertEqual(len(evidence.coverage), 2)

    def test_moi_reused_reader_is_disambiguated_by_load_occurrence(self) -> None:
        first = coverage(reader=7, load="41")
        second = coverage(reader=7, load="42")
        evidence = parse_coverage_evidence(log(
            first,
            second,
            verdict(applicable_code_objects="2", access="40/40", barrier="8/8",
                    atomic="4/4", fence="2/2"),
        ))
        self.assertEqual(
            [(record.reader, record.load) for record in evidence.coverage],
            [(7, 41), (7, 42)],
        )
        self.assertEqual({site.load for site in evidence.sites}, {41, 42})

    def test_moi_rejects_duplicate_load_occurrence_even_when_reader_is_reused(self) -> None:
        with self.assertRaisesRegex(CoverageParseError, "duplicate coverage load identities"):
            parse_coverage_evidence(log(
                coverage(reader=7, load="41"),
                coverage(reader=7, load="41"),
                verdict(applicable_code_objects="2", access="40/40", barrier="8/8",
                        atomic="4/4", fence="2/2"),
            ))

    def test_rejects_duplicate_keys_and_malformed_fields(self) -> None:
        cases = {
            "duplicate key": coverage() + " access_supported=20",
            "bare token": coverage() + " garbage",
            "empty value": coverage() + " extra=",
        }
        for name, line in cases.items():
            with self.subTest(name=name), self.assertRaises(CoverageParseError):
                parse_coverage_evidence(log(line, verdict()))

    def test_rejects_malformed_counts_pairs_and_booleans(self) -> None:
        cases = (
            (coverage(access_supported="-1"), verdict()),
            (coverage(reader="0x7"), verdict()),
            (coverage(access_supported="020"), verdict()),
            (coverage(analysis_complete="yes"), verdict()),
            (coverage(), verdict(access="20")),
            (coverage(), verdict(access="20/-1")),
            (coverage(), verdict(dynamic_incomplete="nan")),
            (coverage(access_supported=str(1 << 64)), verdict()),
            (coverage(), verdict(access=f"{1 << 64}/{1 << 64}")),
        )
        for coverage_line, verdict_line in cases:
            with self.subTest(line=coverage_line), self.assertRaises(CoverageParseError):
                parse_coverage_evidence(log(coverage_line, verdict_line))

    def test_rejects_cross_record_count_disagreement(self) -> None:
        cases = (
            verdict(applicable_code_objects="2"),
            verdict(incomplete_code_objects="1"),
            verdict(access="19/20"),
            verdict(barrier="3/4"),
        )
        for verdict_line in cases:
            with self.subTest(line=verdict_line), self.assertRaises(CoverageParseError):
                parse_coverage_evidence(log(coverage(), verdict_line))

    def test_rejects_nonzero_dynamic_incompleteness_despite_true_claim(self) -> None:
        decision = acceptance_decision(
            log(coverage(), verdict(dynamic_incomplete="1"))
        )
        self.assertFalse(decision.accepted)
        self.assertIn("dynamic analysis incomplete: 1", decision.reasons)

    def test_cli_exit_status_is_the_gate(self) -> None:
        script = DBI_DIR / "consan_coverage_gate.py"
        accepted = subprocess.run(
            [sys.executable, str(script)],
            input=log(coverage(), verdict()),
            text=True,
            stdout=subprocess.PIPE,
            check=False,
        )
        rejected = subprocess.run(
            [sys.executable, str(script)],
            input=log(coverage(), verdict(dynamic_complete="false")),
            text=True,
            stdout=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(accepted.returncode, 0, accepted.stdout)
        self.assertEqual(rejected.returncode, 1, rejected.stdout)
        self.assertIn('"accepted": true', accepted.stdout)
        self.assertIn('"accepted": false', rejected.stdout)


if __name__ == "__main__":
    unittest.main()
