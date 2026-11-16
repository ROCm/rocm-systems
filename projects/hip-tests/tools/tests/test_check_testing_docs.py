# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT
"""Focused unit tests for tools/check_testing_docs.py.

These tests exercise the documentation checker against small, self-contained
fixture document sets written into a temporary directory. They deliberately use
only the standard library plus the checker module itself (which needs only
PyYAML); no repository document is executed and no GPU, ROCm, or build is
required, so the suite is a fast static gate.

Run with:  python -m unittest projects/hip-tests/tools/tests/test_check_testing_docs.py
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

# The checker lives one directory up (tools/check_testing_docs.py).
_TOOLS_DIR = Path(__file__).resolve().parent.parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

import check_testing_docs as checker  # noqa: E402


# A minimal, internally consistent document set. Every negative test starts from
# this and mutates exactly one file so the failure it asserts is unambiguous.
VALID_INVENTORY = """\
schema_version: 1
component: hip-tests
source_of_truth:
  suite_placement: catch/*/CMakeLists.txt
  case_metadata: catch/config/configs/*.yaml
vocabularies:
  kind: [contract]
  disposition: [keep, merge_into]
  gating: [trusted]
suites:
  - id: suite-one
    owner: Team Alpha
    name: Suite One
    kind: contract
    domain: memory
    purpose: Pin a portable invariant.
    oracle: Return code.
    source:
      paths:
        - "external: outside-this-repo"
    metadata:
      framework: Catch2
    execution:
      cadence: pr
      gating: trusted
      ctest_labels: [contract]
      platforms: [linux]
      backends: [amd]
      prerequisites: [HIP runtime and device]
      ci_lane:
        external: documented elsewhere
    evidence:
      - path: "external: evidence-outside-repo"
        claim: Built under BUILD_CONTRACT_TESTS.
    history: Guarded per TI-001.
    disposition: keep
    disposition_rationale: Baseline contract coverage.
"""

VALID_COVERAGE = """\
schema_version: 1
component: hip-tests
coverage_model:
  axes:
    - id: api_surface
      classes: [memory]
    - id: architecture
      classes: [any_gpu]
    - id: capability
      classes: [baseline]
    - id: integration_boundary
      classes: [in_process]
  classes:
    - id: memory.basic
      owner: Team Alpha
      axes: [api_surface.memory, capability.baseline, integration_boundary.in_process]
      risk: high
      intent: Allocation invariants remain observable.
      expected_test_kinds: [contract]
      current_inventory: [suite-one]
      evidence:
        - path: "external: evidence-outside-repo"
          claim: Memory domain is built under BUILD_CONTRACT_TESTS.
      gap_status: covered_dual_tier
  pairwise_checks: []
"""

VALID_TESTING = """\
# Testing Strategy

See the [inventory](test_inventory.yaml) and the [known gaps](#known-gaps).

## Known gaps

Body text.
"""

VALID_INCIDENTS = """\
# Incidents

## TI-001: Example control

**Finding.** A control looked removable but was not.
**Control.** Guard the path.
**Protects.** The trusted signal.
**Origin.** Landed in `0d90b30fb6c`.
**Current status.** Active.
**Retirement condition.** Only when the hazard cannot recur.
"""

VALID_RESTRUCTURING = """\
# Roadmap

A proposal with no external references.
"""

DOCS = {
    "test_inventory.yaml": VALID_INVENTORY,
    "test_coverage_model.yaml": VALID_COVERAGE,
    "TESTING.md": VALID_TESTING,
    "TESTING_INCIDENTS.md": VALID_INCIDENTS,
    "TEST_RESTRUCTURING.md": VALID_RESTRUCTURING,
}


def _write_docs(root, overrides=None):
    """Write the valid document set into root, applying any per-file overrides."""
    overrides = overrides or {}
    for name, content in DOCS.items():
        text = overrides.get(name, content)
        (root / name).write_text(text, encoding="utf-8")


def _all_errors(results):
    return [err for errs in results.values() for err in errs]


class CheckTestingDocsTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def _chdir(self, path):
        """Run the rest of the test from path, restoring cwd afterward."""
        prev = os.getcwd()
        os.chdir(str(path))
        self.addCleanup(os.chdir, prev)

    # -- positive baseline ---------------------------------------------------

    def test_valid_set_passes_with_counts(self):
        _write_docs(self.root)
        results, counts = checker.validate(root=str(self.root))
        self.assertEqual(_all_errors(results), [], msg=results)
        self.assertEqual(counts["inventory_records"], 1)
        self.assertEqual(counts["coverage_classes"], 1)
        # References validated covers resolved incident refs, links, and anchors.
        self.assertGreater(counts["validated_references"], 0)

    # -- malformed YAML ------------------------------------------------------

    def test_malformed_yaml_is_reported(self):
        _write_docs(self.root, {"test_inventory.yaml": "suites: [unterminated\n"})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("does not parse as YAML" in err for err in results["load"]),
            msg=results["load"],
        )

    # -- top-level inventory schema: missing keys ----------------------------

    def test_missing_top_level_schema_version(self):
        broken = VALID_INVENTORY.replace("schema_version: 1\n", "", 1)
        _write_docs(self.root, {"test_inventory.yaml": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("missing top-level keys" in err and "schema_version" in err
                for err in results["inventory"]),
            msg=results["inventory"],
        )

    def test_missing_top_level_component(self):
        broken = VALID_INVENTORY.replace("component: hip-tests\n", "", 1)
        _write_docs(self.root, {"test_inventory.yaml": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("missing top-level keys" in err and "component" in err
                for err in results["inventory"]),
            msg=results["inventory"],
        )

    def test_missing_top_level_source_of_truth(self):
        broken = VALID_INVENTORY.replace(
            "source_of_truth:\n"
            "  suite_placement: catch/*/CMakeLists.txt\n"
            "  case_metadata: catch/config/configs/*.yaml\n",
            "", 1)
        _write_docs(self.root, {"test_inventory.yaml": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("missing top-level keys" in err and "source_of_truth" in err
                for err in results["inventory"]),
            msg=results["inventory"],
        )

    def test_missing_top_level_vocabularies(self):
        broken = VALID_INVENTORY.replace(
            "vocabularies:\n"
            "  kind: [contract]\n"
            "  disposition: [keep, merge_into]\n"
            "  gating: [trusted]\n",
            "", 1)
        _write_docs(self.root, {"test_inventory.yaml": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("missing top-level keys" in err and "vocabularies" in err
                for err in results["inventory"]),
            msg=results["inventory"],
        )

    # -- top-level inventory schema: malformed nested shapes -----------------

    def test_component_not_a_string(self):
        broken = VALID_INVENTORY.replace("component: hip-tests\n", "component: 5\n", 1)
        _write_docs(self.root, {"test_inventory.yaml": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("component must be a non-empty string" in err
                for err in results["inventory"]),
            msg=results["inventory"],
        )

    def test_schema_version_must_be_scalar(self):
        broken = VALID_INVENTORY.replace(
            "schema_version: 1\n", "schema_version: [1]\n", 1)
        _write_docs(self.root, {"test_inventory.yaml": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("schema_version must be a non-empty scalar" in err
                for err in results["inventory"]),
            msg=results["inventory"],
        )

    def test_source_of_truth_not_a_mapping(self):
        broken = VALID_INVENTORY.replace(
            "source_of_truth:\n"
            "  suite_placement: catch/*/CMakeLists.txt\n"
            "  case_metadata: catch/config/configs/*.yaml\n",
            "source_of_truth: just-a-string\n", 1)
        _write_docs(self.root, {"test_inventory.yaml": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("source_of_truth must be a non-empty mapping" in err
                for err in results["inventory"]),
            msg=results["inventory"],
        )

    def test_vocabularies_not_a_mapping(self):
        broken = VALID_INVENTORY.replace(
            "vocabularies:\n"
            "  kind: [contract]\n"
            "  disposition: [keep, merge_into]\n"
            "  gating: [trusted]\n",
            "vocabularies: []\n", 1)
        _write_docs(self.root, {"test_inventory.yaml": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("vocabularies must be a mapping" in err
                for err in results["inventory"]),
            msg=results["inventory"],
        )

    def test_vocabulary_category_must_be_a_list(self):
        broken = VALID_INVENTORY.replace("  kind: [contract]\n", "  kind: contract\n", 1)
        _write_docs(self.root, {"test_inventory.yaml": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("vocabularies 'kind' must be a list" in err
                for err in results["inventory"]),
            msg=results["inventory"],
        )

    # -- inventory suite source / evidence shapes ----------------------------

    def test_empty_source_paths_is_reported(self):
        broken = VALID_INVENTORY.replace(
            '      paths:\n        - "external: outside-this-repo"\n',
            "      paths: []\n", 1)
        _write_docs(self.root, {"test_inventory.yaml": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("source.paths must be a non-empty list" in err
                for err in results["inventory"]),
            msg=results["inventory"],
        )

    def test_evidence_entry_missing_claim(self):
        broken = VALID_INVENTORY.replace(
            "        claim: Built under BUILD_CONTRACT_TESTS.\n", "", 1)
        _write_docs(self.root, {"test_inventory.yaml": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("evidence entry missing a claim" in err
                for err in results["inventory"]),
            msg=results["inventory"],
        )

    # -- disposition targets -------------------------------------------------

    def test_merge_into_requires_disposition_target(self):
        broken = VALID_INVENTORY.replace(
            "    disposition: keep\n", "    disposition: merge_into\n", 1)
        _write_docs(self.root, {"test_inventory.yaml": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("must name disposition_target" in err
                for err in results["inventory"]),
            msg=results["inventory"],
        )

    def test_merge_into_target_must_resolve(self):
        broken = VALID_INVENTORY.replace(
            "    disposition: keep\n",
            "    disposition: merge_into\n    disposition_target: suite-missing\n", 1)
        _write_docs(self.root, {"test_inventory.yaml": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("is not an inventory id" in err for err in results["inventory"]),
            msg=results["inventory"],
        )

    def test_migrate_does_not_require_disposition_target(self):
        # migrate/demote/retire describe a lifecycle change whose destination may
        # be external, so omitting disposition_target is not an error.
        broken = VALID_INVENTORY.replace(
            "  disposition: [keep, merge_into]\n",
            "  disposition: [keep, merge_into, migrate]\n", 1).replace(
            "    disposition: keep\n", "    disposition: migrate\n", 1)
        _write_docs(self.root, {"test_inventory.yaml": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertEqual(results["inventory"], [], msg=results["inventory"])

    # -- malformed ids (non-empty-string guard) ------------------------------

    def test_malformed_suite_id_is_reported(self):
        broken = VALID_INVENTORY.replace("  - id: suite-one\n", "  - id: ''\n", 1)
        _write_docs(self.root, {"test_inventory.yaml": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("suite id must be a non-empty string" in err
                for err in results["inventory"]),
            msg=results["inventory"],
        )

    def test_malformed_coverage_axis_id_is_reported(self):
        broken = VALID_COVERAGE.replace("    - id: api_surface\n", "    - id: ''\n", 1)
        _write_docs(self.root, {"test_coverage_model.yaml": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("coverage axis id must be a non-empty string" in err
                for err in results["coverage_model"]),
            msg=results["coverage_model"],
        )

    def test_malformed_coverage_class_id_is_reported(self):
        broken = VALID_COVERAGE.replace("    - id: memory.basic\n", "    - id: ''\n", 1)
        _write_docs(self.root, {"test_coverage_model.yaml": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("coverage class id must be a non-empty string" in err
                for err in results["coverage_model"]),
            msg=results["coverage_model"],
        )

    # -- top-level coverage-model schema -------------------------------------

    def test_coverage_missing_top_level_schema_version(self):
        broken = VALID_COVERAGE.replace("schema_version: 1\n", "", 1)
        _write_docs(self.root, {"test_coverage_model.yaml": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("missing top-level keys" in err and "schema_version" in err
                for err in results["coverage_model"]),
            msg=results["coverage_model"],
        )

    def test_coverage_missing_top_level_component(self):
        broken = VALID_COVERAGE.replace("component: hip-tests\n", "", 1)
        _write_docs(self.root, {"test_coverage_model.yaml": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("missing top-level keys" in err and "component" in err
                for err in results["coverage_model"]),
            msg=results["coverage_model"],
        )

    def test_coverage_schema_version_must_be_scalar(self):
        broken = VALID_COVERAGE.replace(
            "schema_version: 1\n", "schema_version: [1]\n", 1)
        _write_docs(self.root, {"test_coverage_model.yaml": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("coverage model schema_version must be a non-empty scalar" in err
                for err in results["coverage_model"]),
            msg=results["coverage_model"],
        )

    def test_coverage_component_not_a_string(self):
        broken = VALID_COVERAGE.replace("component: hip-tests\n", "component: 5\n", 1)
        _write_docs(self.root, {"test_coverage_model.yaml": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("coverage model component must be a non-empty string" in err
                for err in results["coverage_model"]),
            msg=results["coverage_model"],
        )

    # -- coverage class evidence shapes --------------------------------------

    def test_coverage_class_evidence_empty_list(self):
        broken = VALID_COVERAGE.replace(
            '      evidence:\n'
            '        - path: "external: evidence-outside-repo"\n'
            '          claim: Memory domain is built under BUILD_CONTRACT_TESTS.\n',
            "      evidence: []\n", 1)
        _write_docs(self.root, {"test_coverage_model.yaml": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("evidence must be a non-empty list" in err
                for err in results["coverage_model"]),
            msg=results["coverage_model"],
        )

    def test_coverage_class_evidence_entry_not_a_mapping(self):
        broken = VALID_COVERAGE.replace(
            '        - path: "external: evidence-outside-repo"\n'
            '          claim: Memory domain is built under BUILD_CONTRACT_TESTS.\n',
            "        - not-a-mapping\n", 1)
        _write_docs(self.root, {"test_coverage_model.yaml": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("evidence entry is not a mapping" in err
                for err in results["coverage_model"]),
            msg=results["coverage_model"],
        )

    def test_coverage_class_evidence_entry_missing_claim(self):
        broken = VALID_COVERAGE.replace(
            "          claim: Memory domain is built under BUILD_CONTRACT_TESTS.\n",
            "", 1)
        _write_docs(self.root, {"test_coverage_model.yaml": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("evidence entry missing a claim" in err
                for err in results["coverage_model"]),
            msg=results["coverage_model"],
        )

    def test_coverage_class_evidence_entry_missing_path(self):
        broken = VALID_COVERAGE.replace(
            '        - path: "external: evidence-outside-repo"\n'
            '          claim: Memory domain is built under BUILD_CONTRACT_TESTS.\n',
            "        - claim: A claim with no path.\n", 1)
        _write_docs(self.root, {"test_coverage_model.yaml": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("evidence entry missing a path" in err
                for err in results["coverage_model"]),
            msg=results["coverage_model"],
        )

    # -- unknown inventory id in coverage ------------------------------------

    def test_unknown_inventory_id_in_coverage(self):
        broken = VALID_COVERAGE.replace("[suite-one]", "[suite-does-not-exist]")
        _write_docs(self.root, {"test_coverage_model.yaml": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("unknown inventory id" in err for err in results["coverage_model"]),
            msg=results["coverage_model"],
        )

    # -- missing required axis -----------------------------------------------

    def test_missing_required_axis(self):
        broken = VALID_COVERAGE.replace(
            "    - id: integration_boundary\n      classes: [in_process]\n", "")
        # Also drop the now-dangling class reference to isolate the axis error.
        broken = broken.replace(", integration_boundary.in_process", "")
        _write_docs(self.root, {"test_coverage_model.yaml": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("missing required axis 'integration_boundary'" in err
                for err in results["coverage_model"]),
            msg=results["coverage_model"],
        )

    # -- coverage model malformed record types (no AttributeError) -----------

    def test_axis_record_not_a_mapping(self):
        broken = VALID_COVERAGE.replace(
            "    - id: integration_boundary\n      classes: [in_process]\n",
            "    - id: integration_boundary\n      classes: [in_process]\n    - 42\n",
            1)
        _write_docs(self.root, {"test_coverage_model.yaml": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("axis is not a mapping" in err for err in results["coverage_model"]),
            msg=results["coverage_model"],
        )

    def test_class_record_not_a_mapping(self):
        broken = VALID_COVERAGE.replace(
            "      gap_status: covered_dual_tier\n  pairwise_checks: []\n",
            "      gap_status: covered_dual_tier\n    - not-a-mapping\n"
            "  pairwise_checks: []\n",
            1)
        _write_docs(self.root, {"test_coverage_model.yaml": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("class record is not a mapping" in err
                for err in results["coverage_model"]),
            msg=results["coverage_model"],
        )

    def test_pairwise_record_not_a_mapping(self):
        broken = VALID_COVERAGE.replace(
            "  pairwise_checks: []\n",
            "  pairwise_checks:\n    - not-a-mapping\n", 1)
        _write_docs(self.root, {"test_coverage_model.yaml": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("pairwise check record is not a mapping" in err
                for err in results["coverage_model"]),
            msg=results["coverage_model"],
        )

    # -- missing owner -------------------------------------------------------

    def test_missing_owner_on_suite(self):
        broken = VALID_INVENTORY.replace("    owner: Team Alpha\n", "", 1)
        _write_docs(self.root, {"test_inventory.yaml": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("missing keys" in err and "owner" in err
                for err in results["inventory"]),
            msg=results["inventory"],
        )

    # -- missing evidence path on disk ---------------------------------------

    def test_missing_evidence_path(self):
        broken = VALID_INVENTORY.replace(
            '      - path: "external: evidence-outside-repo"',
            "      - path: catch/does/not/exist.txt")
        _write_docs(self.root, {"test_inventory.yaml": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("evidence path missing" in err for err in results["inventory"]),
            msg=results["inventory"],
        )

    # -- unknown incident reference ------------------------------------------

    def test_unknown_incident_reference(self):
        broken = VALID_INVENTORY.replace("Guarded per TI-001.", "Guarded per TI-999.")
        _write_docs(self.root, {"test_inventory.yaml": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("TI-999" in err for err in results["incidents"]),
            msg=results["incidents"],
        )

    # -- incident missing a required labelled field --------------------------

    def test_incident_missing_required_field(self):
        broken = VALID_INCIDENTS.replace(
            "**Protects.** The trusted signal.\n", "")
        _write_docs(self.root, {"TESTING_INCIDENTS.md": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("missing the 'Protects' field" in err for err in results["incidents"]),
            msg=results["incidents"],
        )

    # -- broken local link ---------------------------------------------------

    def test_broken_local_link(self):
        broken = VALID_TESTING.replace("(test_inventory.yaml)", "(no_such_file.yaml)")
        _write_docs(self.root, {"TESTING.md": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("broken local link" in err for err in results["markdown_links"]),
            msg=results["markdown_links"],
        )

    # -- broken in-page anchor -----------------------------------------------

    def test_broken_anchor(self):
        broken = VALID_TESTING.replace("(#known-gaps)", "(#no-such-anchor)")
        _write_docs(self.root, {"TESTING.md": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("broken anchor" in err for err in results["markdown_links"]),
            msg=results["markdown_links"],
        )

    def test_external_url_is_preserved(self):
        extended = VALID_TESTING + "\nAn [external](https://example.com/page) link.\n"
        _write_docs(self.root, {"TESTING.md": extended})
        results, _ = checker.validate(root=str(self.root))
        self.assertEqual(results["markdown_links"], [], msg=results["markdown_links"])

    # -- negative --root paths -----------------------------------------------

    def test_empty_root_diagnoses_missing_docs(self):
        # An empty root must diagnose missing documents, not crash. Run from an
        # empty directory so the result does not depend on the caller's cwd.
        self._chdir(self.root)
        results, _ = checker.validate(root="")
        self.assertTrue(
            any("file not found" in err for err in results["load"]),
            msg=results["load"],
        )
        self.assertTrue(_all_errors(results))

    def test_main_empty_root_returns_failure_without_argparse_error(self):
        # main("--root", "") must return exit code 1, not raise SystemExit(2).
        # Run from an empty directory so the result is cwd-independent.
        self._chdir(self.root)
        rc = checker.main(["--root", "", "--json"])
        self.assertEqual(rc, 1)

    def test_main_root_override_passes(self):
        _write_docs(self.root)
        rc = checker.main(["--root", str(self.root)])
        self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()
