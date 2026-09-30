#!/usr/bin/env python3

from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

import consan_validation as validation
import consan_validation_faults as validation_faults


class ConSanValidationTargetAdmissionTest(unittest.TestCase):
    def test_gfx1100_admits_registered_native_gtests(self) -> None:
        expected = {
            "d128-block": (
                "hip-moi-build-gfx1100-tests/tests/"
                "hip_moi_instrumented_gfx1100_d128_attention_block_test",
                "HipMoiGfx1100D128AttentionBlock.*",
            ),
            "d128-pressure": (
                "hip-moi-build-gfx1100-tests/tests/"
                "hip_moi_instrumented_gfx1100_d128_attention_pressure_test",
                "HipMoiGfx1100D128AttentionPressure.*",
            ),
            "wmma-attention": (
                "hip-moi-build-gfx1100-tests/tests/"
                "hip_moi_instrumented_gfx1100_wmma_attention_block_test",
                "HipMoiGfx1100WmmaAttentionBlock.*",
            ),
            "streamk-arrival": (
                "hip-moi-build-gfx1100-tests/tests/"
                "hip_moi_instrumented_gfx1100_wmma_streamk_arrival_counter_test",
                "HipMoiGfx1100WmmaStreamKArrivalCounter."
                "ConSanOracleAcqRelFetchAddOrdersWmmaPartials",
            ),
            "tree-atomic-or": (
                "hip-moi-build-gfx1100-tests/tests/"
                "hip_moi_instrumented_gfx1100_wmma_streamk_tree_atomic_or_test",
                "HipMoiGfx1100WmmaStreamKTreeAtomicOr."
                "ConSanOracleAcqRelBitmaskOrdersWmmaPartials",
            ),
            "jakub-attention": (
                "hip-moi-build-gfx1100-tests/tests/"
                "hip_moi_reference_gfx1100_jakub_matmul",
                "SafeFp16Packed/JakubGfx1100MatmulReference." "MatchesHostReference/*",
            ),
        }
        manifest_ids = {
            workload["id"] for workload in validation._manifest("gfx1100")["workloads"]
        }
        self.assertTrue(set(expected).issubset(manifest_ids))

        self.assertEqual(set(expected), set(validation.NATIVE_GTEST_WORKLOAD_IDS))
        for workload_id, (relative_path, clean_filter) in expected.items():
            with self.subTest(workload=workload_id):
                workload = validation._workload_for_target("gfx1100", workload_id)
                resolved = validation._resolved_workload("gfx1100", workload)
                self.assertEqual(resolved.relative_path, relative_path)
                self.assertEqual(resolved.clean_filter, clean_filter)
        for workload_id in ("streamk-arrival", "tree-atomic-or"):
            with self.subTest(workload=workload_id):
                workload = validation._workload_for_target("gfx1100", workload_id)
                self.assertEqual(
                    validation._fault_families("gfx1100", workload),
                    ("atomic-weaken-order",),
                )

    def test_gfx1100_barrier_faults_target_reviewed_publications(self) -> None:
        catalog = Path(__file__).with_name("consan_validation_faults_gfx1100.json")
        expected = {
            "d128-block": (
                "barrier-drop-k-publication-group",
                "fnv1a64:43b0a96f4ab07982",
                "pc=0x0000000000054844",
                "pc=0x0000000000054850",
            ),
            "d128-pressure": (
                "barrier-drop-kv-publication-group",
                "fnv1a64:e969c5a2e1a058f7",
                "pc=0x000000000003ac0c",
                "pc=0x000000000003ac18",
            ),
            "wmma-attention": (
                "barrier-drop-kv-publication-group",
                "fnv1a64:f8a2c2abd1d454c2",
                "pc=0x0000000000015530",
                "pc=0x000000000001553c",
            ),
            "jakub-attention": (
                "barrier-drop-load-compute-publication",
                "fnv1a64:60f9f11b7f717a09",
                "pc=0x000000000000068c",
                None,
            ),
        }

        for workload_id, (
            fault_prefix,
            code_object,
            primary_pc,
            companion_pc,
        ) in expected.items():
            grouped = companion_pc is not None
            workload = validation.WORKLOAD_BY_ID[workload_id]
            with self.subTest(workload=workload_id, profile="default"):
                for preset in ("default", "high", "higher", "max"):
                    fault = validation_faults._load_fault(
                        catalog,
                        "gfx1100",
                        workload,
                        f"{fault_prefix}-preset-{preset}",
                    )
                    environment = fault["environment"]
                    self.assertEqual(environment["RJ_CONSAN_FAULT_DROP_BARRIER"], "1")
                    self.assertIn(
                        code_object, environment["RJ_CONSAN_FAULT_SITE_IDENTITY"]
                    )
                    self.assertIn(
                        primary_pc, environment["RJ_CONSAN_FAULT_SITE_IDENTITY"]
                    )
                    self.assertEqual(
                        "RJ_CONSAN_FAULT_BARRIER_SEQUENCE_IDENTITY" in environment,
                        grouped,
                    )
                    self.assertEqual(
                        "RJ_CONSAN_FAULT_BARRIER_COMPANION_SITE_IDENTITY"
                        in environment,
                        grouped,
                    )
                    self.assertEqual(
                        "RJ_CONSAN_FAULT_BARRIER_COMPANION_SEQUENCE_IDENTITY"
                        in environment,
                        grouped,
                    )
                    if companion_pc is not None:
                        self.assertIn(
                            companion_pc,
                            environment[
                                "RJ_CONSAN_FAULT_BARRIER_COMPANION_SITE_IDENTITY"
                            ],
                        )
                    policy, trials = validation_faults._fault_trials(fault, "default")
                    self.assertEqual(policy["detector"], "statistical")
                    self.assertEqual(policy["minimum_detections"], 6)
                    self.assertEqual(policy["oracle"], "any")
                    self.assertEqual(trials, [{"RJ_CONSAN_PRESET": preset}] * 8)

            with self.subTest(workload=workload_id, profile="supercollider"):
                fault = validation_faults._load_fault(
                    catalog,
                    "gfx1100",
                    workload,
                    f"{fault_prefix}-sc-sleep-15",
                )
                policy, trials = validation_faults._fault_trials(fault, "supercollider")
                self.assertEqual(policy["detector"], "statistical")
                self.assertEqual(policy["minimum_detections"], 6)
                self.assertEqual(policy["oracle"], "any")
                self.assertEqual(
                    trials,
                    [
                        {
                            "RJ_CONSAN_SC_DELAY_MODE": "sleep",
                            "RJ_CONSAN_SC_DELAY": "15",
                        }
                    ]
                    * 8,
                )

    def test_gfx1100_fails_closed_without_native_registry(self) -> None:
        with (
            mock.patch.dict(validation.NATIVE_GTEST_TARGETS, {}, clear=True),
            mock.patch.dict(validation.NATIVE_GTEST_WORKLOAD_OVERRIDES, {}, clear=True),
        ):
            manifest_ids = {
                workload["id"]
                for workload in validation._manifest("gfx1100")["workloads"]
            }
            self.assertTrue(
                manifest_ids.isdisjoint(validation.NATIVE_GTEST_WORKLOAD_IDS)
            )

            for workload_id in validation.NATIVE_GTEST_WORKLOAD_IDS:
                with self.subTest(workload=workload_id):
                    with self.assertRaisesRegex(
                        validation.ValidationError,
                        f"gfx1100 manifest excludes workload: {workload_id}",
                    ):
                        validation._workload_for_target("gfx1100", workload_id)

            with self.assertRaisesRegex(
                validation.ValidationError,
                "gfx1100 gtest workload has no target-specific registry",
            ):
                validation._resolved_workload(
                    "gfx1100", validation.WORKLOAD_BY_ID["d128-block"]
                )


if __name__ == "__main__":
    unittest.main()
