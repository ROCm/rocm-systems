#!/usr/bin/env python3

from __future__ import annotations

import unittest
from unittest import mock

import consan_validation as validation


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
