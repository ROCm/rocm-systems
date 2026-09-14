#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for the HIP test-plan generator."""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_test_plan


class GenTestPlanTests(unittest.TestCase):
    def _write_contract_fixture(self, root):
        catch_dir = root / "catch"
        domain = catch_dir / "contract" / "memory"
        domain.mkdir(parents=True)
        (domain / "test_hip_memory_contract.cc").write_text(
            """
// Helper comment that should stay in the contiguous block.
// @asserts: hipMalloc - allocates a non-null pointer
HIP_TEST_CASE(Contract_Memory_HipMalloc_Default_ReturnsPointer) {
}

// @asserts: hipFree — releases a prior allocation
HIP_TEST_CASE(
    Contract_Memory_HipFree_Default_ReleasesAllocation) {
}

HIP_TEST_CASE(Contract_Memory_HipMemset_Default_WritesBytes) {
}
""".lstrip(),
            encoding="utf-8",
        )
        return catch_dir

    def test_collect_cases_extracts_asserts_and_missing_tags(self):
        with tempfile.TemporaryDirectory() as tmp:
            catch_dir = self._write_contract_fixture(Path(tmp))

            cases = gen_test_plan.collect_cases(["contract"], str(catch_dir))

            self.assertEqual(
                [
                    "Contract_Memory_HipFree_Default_ReleasesAllocation",
                    "Contract_Memory_HipMalloc_Default_ReturnsPointer",
                    "Contract_Memory_HipMemset_Default_WritesBytes",
                ],
                [case["case"] for case in cases],
            )
            by_name = {case["case"]: case for case in cases}
            self.assertEqual("memory", by_name["Contract_Memory_HipMalloc_Default_ReturnsPointer"]["domain"])
            self.assertEqual("hipMalloc", by_name["Contract_Memory_HipMalloc_Default_ReturnsPointer"]["api"])
            self.assertEqual(
                "allocates a non-null pointer",
                by_name["Contract_Memory_HipMalloc_Default_ReturnsPointer"]["invariant"],
            )
            self.assertTrue(by_name["Contract_Memory_HipMalloc_Default_ReturnsPointer"]["tagged"])

            self.assertEqual("hipFree", by_name["Contract_Memory_HipFree_Default_ReleasesAllocation"]["api"])
            self.assertEqual(
                "releases a prior allocation",
                by_name["Contract_Memory_HipFree_Default_ReleasesAllocation"]["invariant"],
            )

            untagged = by_name["Contract_Memory_HipMemset_Default_WritesBytes"]
            self.assertFalse(untagged["tagged"])
            self.assertEqual("", untagged["api"])
            self.assertEqual("Memory Hip Memset Default Writes Bytes", untagged["invariant"])

    def test_render_markdown_includes_summary_domains_and_missing_tags(self):
        with tempfile.TemporaryDirectory() as tmp:
            catch_dir = self._write_contract_fixture(Path(tmp))
            cases = gen_test_plan.collect_cases(["contract"], str(catch_dir))

            rendered = gen_test_plan.render_markdown(cases, ["contract"])

            self.assertIn("| `contract` | 3 | 2 | 1 |", rendered)
            self.assertIn("### `memory` (3 cases)", rendered)
            self.assertIn("| `Contract_Memory_HipMalloc_Default_ReturnsPointer` | hipMalloc | allocates a non-null pointer |", rendered)
            self.assertIn("## Cases missing an `@asserts` tag", rendered)
            self.assertIn("- `Contract_Memory_HipMemset_Default_WritesBytes` (contract/memory)", rendered)

    def test_render_markdown_omits_missing_tag_section_when_all_cases_are_tagged(self):
        cases = [{
            "tier": "contract",
            "domain": "memory",
            "case": "Contract_Memory_HipMalloc_Default_ReturnsPointer",
            "api": "hipMalloc",
            "invariant": "allocates a non-null pointer",
            "tagged": True,
            "file": "contract/memory/test_hip_memory_contract.cc",
        }]

        rendered = gen_test_plan.render_markdown(cases, ["contract"])

        self.assertNotIn("## Cases missing an `@asserts` tag", rendered)

    def test_main_check_and_json_modes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catch_dir = self._write_contract_fixture(root)
            plan_path = root / "TEST_PLAN.md"

            with contextlib.redirect_stderr(io.StringIO()):
                missing_status = gen_test_plan.main([
                    "--check",
                    "--catch-dir", str(catch_dir),
                    "--out", str(plan_path),
                ])
            self.assertEqual(1, missing_status)

            cases = gen_test_plan.collect_cases(["contract"], str(catch_dir))
            plan_path.write_text(gen_test_plan.render_markdown(cases, ["contract"]), encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                fresh_status = gen_test_plan.main([
                    "--check",
                    "--catch-dir", str(catch_dir),
                    "--out", str(plan_path),
                ])
            self.assertEqual(0, fresh_status)

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                json_status = gen_test_plan.main([
                    "--json",
                    "--catch-dir", str(catch_dir),
                ])
            self.assertEqual(0, json_status)
            self.assertEqual(3, len(json.loads(stdout.getvalue())))


if __name__ == "__main__":
    unittest.main()
