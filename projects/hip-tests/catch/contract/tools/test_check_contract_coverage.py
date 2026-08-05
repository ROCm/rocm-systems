#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT
"""Unit tests for the contract coverage drift checker."""

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_contract_coverage as coverage


class ContractCoverageSnapshotTests(unittest.TestCase):
    def _write_fixture(self, root, snapshot=None):
        header = root / "hip_runtime_api.h"
        header.write_text(
            """
HIP_PUBLIC_API hipError_t hipAlpha(int* out);
HIP_PUBLIC_API hipError_t hipBeta(void);
HIP_PUBLIC_API hipError_t hipGamma(void);
""".lstrip(),
            encoding="utf-8",
        )

        contract_dir = root / "contract"
        domain = contract_dir / "memory"
        domain.mkdir(parents=True)
        (domain / "test_hip_memory_contract.cc").write_text(
            """
HIP_TEST_CASE(Contract_Memory_HipAlpha_Default_Succeeds) {
  hipAlpha(nullptr);
}
HIP_TEST_CASE(Contract_Memory_HipBeta_Default_Succeeds) {
  hipBeta();
}
""".lstrip(),
            encoding="utf-8",
        )

        allowlist = contract_dir / "uncovered_apis.txt"
        allowlist.write_text("hipGamma  # fixture gap\n", encoding="utf-8")

        if snapshot is None:
            snapshot = {
                "contract_tests": 2,
                "contract_domains": 1,
                "declared_apis": 3,
                "covered_apis": 2,
                "uncovered_allowlisted": 1,
                "coverage_pct": 66.7,
            }
        doc = root / "CONTRACT_COVERAGE.md"
        doc.write_text(
            """
# Fixture

<!-- contract-coverage-snapshot
contract_tests: {contract_tests}
contract_domains: {contract_domains}
declared_apis: {declared_apis}
covered_apis: {covered_apis}
uncovered_allowlisted: {uncovered_allowlisted}
coverage_pct: {coverage_pct}
-->
""".format(**snapshot).lstrip(),
            encoding="utf-8",
        )
        return header, contract_dir, allowlist, doc

    def test_snapshot_drift_is_empty_when_doc_matches_computed_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            header, contract_dir, allowlist, doc = self._write_fixture(Path(tmp))
            report = coverage.compute(header, contract_dir, allowlist)

            self.assertEqual([], coverage.snapshot_drift(report, contract_dir, doc))

    def test_snapshot_drift_reports_stale_contract_test_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            snapshot = {
                "contract_tests": 1,
                "contract_domains": 1,
                "declared_apis": 3,
                "covered_apis": 2,
                "uncovered_allowlisted": 1,
                "coverage_pct": 66.7,
            }
            header, contract_dir, allowlist, doc = self._write_fixture(Path(tmp), snapshot)
            report = coverage.compute(header, contract_dir, allowlist)

            self.assertEqual(
                ["contract_tests is stale: doc has 1, computed 2"],
                coverage.snapshot_drift(report, contract_dir, doc),
            )

    def test_missing_snapshot_block_is_reported_as_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            header, contract_dir, allowlist, doc = self._write_fixture(Path(tmp))
            doc.write_text("# Missing snapshot\n", encoding="utf-8")
            report = coverage.compute(header, contract_dir, allowlist)

            self.assertEqual(
                ["coverage doc is missing the contract-coverage-snapshot block"],
                coverage.snapshot_drift(report, contract_dir, doc),
            )

    def test_main_check_fails_when_snapshot_drifts(self):
        with tempfile.TemporaryDirectory() as tmp:
            snapshot = {
                "contract_tests": 2,
                "contract_domains": 1,
                "declared_apis": 3,
                "covered_apis": 1,
                "uncovered_allowlisted": 1,
                "coverage_pct": 66.7,
            }
            header, contract_dir, allowlist, doc = self._write_fixture(Path(tmp), snapshot)

            with contextlib.redirect_stdout(io.StringIO()):
                status = coverage.main([
                    "--check",
                    "--header", str(header),
                    "--contract-dir", str(contract_dir),
                    "--allowlist", str(allowlist),
                    "--coverage-doc", str(doc),
                ])
            self.assertEqual(1, status)


if __name__ == "__main__":
    unittest.main()
