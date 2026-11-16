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
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

# git-backed tests (Origin-citation existence checks) need a real git binary;
# they are skipped rather than failed where git is unavailable.
_GIT = shutil.which("git")

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
  suite_placement: "external: catch/*/CMakeLists.txt"
  case_metadata: "external: catch/config/configs/*.yaml"
vocabularies:
  kind: [contract]
  disposition: [keep, merge_into]
  gating: [trusted]
  cadence: [pr]
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
      build_option: BUILD_CONTRACT_TESTS
      build_target: contract_tests
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
    - id: behavior
      classes: [data_integrity]
    - id: environment
      classes: [nvidia_backend]
    - id: execution_context
      classes: [single_gpu]
    - id: cadence
      classes: [pr]
    - id: architecture
      classes: [any_gpu]
    - id: capability
      classes: [baseline]
    - id: integration_boundary
      classes: [in_process]
  classes:
    - id: memory.basic
      owner: Team Alpha
      axes: [api_surface.memory, behavior.data_integrity, architecture.any_gpu, capability.baseline, integration_boundary.in_process]
      risk: high
      intent: Allocation invariants remain observable.
      expected_test_kinds: [contract]
      current_inventory: [suite-one]
      evidence:
        - path: "external: evidence-outside-repo"
          claim: Memory domain is built under BUILD_CONTRACT_TESTS.
      gap_status: covered_dual_tier
  pairwise_checks:
    - id: memory_x_data_integrity
      category: api_x_behavior
      pair: [api_surface.memory, behavior.data_integrity]
      status: review_priority
      priority: high
      rationale: Round-trips preserve bytes, not only return success.
      current_inventory: [suite-one]
      gap_status: covered_dual_tier
    - id: memory_x_nvidia_backend
      category: api_x_environment
      pair: [api_surface.memory, environment.nvidia_backend]
      status: review_priority
      priority: medium
      rationale: Contract memory invariants are portable to the NVIDIA backend.
      current_inventory: [suite-one]
      gap_status: covered_dual_tier
    - id: memory_x_single_gpu
      category: integration_x_execution_context
      pair: [api_surface.memory, execution_context.single_gpu]
      status: review_priority
      priority: low
      rationale: Behavior must be observable on a single device first.
      current_inventory: [suite-one]
      gap_status: covered_dual_tier
    - id: baseline_x_pr
      category: capability_x_cadence
      pair: [capability.baseline, cadence.pr]
      status: review_priority
      priority: low
      rationale: Baseline-capability paths run on PR cadence.
      current_inventory: [suite-one]
      gap_status: covered_dual_tier
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


# A fully-formed retiring suite: it names its replacement inventory id via
# disposition_target and documents equivalent-or-stronger coverage, CI evidence,
# and owner approval in a retirement mapping. _with_retire appends it to the
# valid inventory (whose disposition vocabulary it extends to allow ``retire``);
# each negative test mutates exactly one line of this block.
RETIRE_SUITE = """\
  - id: suite-two
    owner: Team Beta
    name: Suite Two
    kind: contract
    domain: memory
    purpose: Legacy coverage now superseded.
    oracle: Return code.
    source:
      paths:
        - "external: outside-this-repo"
      build_option: BUILD_CONTRACT_TESTS
      build_target: contract_tests
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
    history: Superseded coverage; repository evidence records the folding into suite-one.
    disposition: retire
    disposition_target: [suite-one]
    disposition_rationale: Folded into suite-one.
    retirement:
      coverage_equivalence: suite-one pins the same invariants and adds OOB checks.
      ci_evidence: Runs in the contract lane under BUILD_CONTRACT_TESTS.
      owner_approval: Approved by Team Beta on 2026-11-10.
"""


def _with_retire(suite_block):
    """Return the valid inventory extended with a ``retire`` suite block.

    The disposition vocabulary is widened to allow ``retire`` and suite_block is
    appended as a second suite so retirement enforcement can be exercised.
    """
    inventory = VALID_INVENTORY.replace(
        "  disposition: [keep, merge_into]\n",
        "  disposition: [keep, merge_into, retire]\n", 1)
    return inventory + suite_block


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

    def test_invalid_utf8_is_reported_deterministically(self):
        # A document that is not valid UTF-8 must produce a deterministic load
        # error rather than crashing the checker with an uncaught
        # UnicodeDecodeError. The error is the same across repeated runs.
        _write_docs(self.root)
        (self.root / "test_inventory.yaml").write_bytes(
            b"schema_version: 1\ncomponent: \xff\xfe not utf-8\n")
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("not valid UTF-8" in err for err in results["load"]),
            msg=results["load"],
        )
        # The same invalid input yields an identical load result every run.
        results_again, _ = checker.validate(root=str(self.root))
        self.assertEqual(results["load"], results_again["load"],
                         msg=results["load"])

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
            '  suite_placement: "external: catch/*/CMakeLists.txt"\n'
            '  case_metadata: "external: catch/config/configs/*.yaml"\n',
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

    def test_unsupported_schema_version_is_rejected(self):
        # A well-formed scalar that is not the supported version (1) is a schema
        # error in its own right, distinct from the non-scalar shape error.
        broken = VALID_INVENTORY.replace(
            "schema_version: 1\n", "schema_version: 2\n", 1)
        _write_docs(self.root, {"test_inventory.yaml": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("schema_version must be the supported version 1" in err
                for err in results["inventory"]),
            msg=results["inventory"],
        )

    def test_wrong_component_is_rejected(self):
        # A non-empty component string that is not hip-tests is rejected: the
        # document set was authored against a different component.
        broken = VALID_INVENTORY.replace(
            "component: hip-tests\n", "component: rocprofiler-sdk\n", 1)
        _write_docs(self.root, {"test_inventory.yaml": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("component must be 'hip-tests'" in err
                for err in results["inventory"]),
            msg=results["inventory"],
        )

    def test_source_of_truth_not_a_mapping(self):
        broken = VALID_INVENTORY.replace(
            "source_of_truth:\n"
            '  suite_placement: "external: catch/*/CMakeLists.txt"\n'
            '  case_metadata: "external: catch/config/configs/*.yaml"\n',
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
            "  gating: [trusted]\n"
            "  cadence: [pr]\n",
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
            any("vocabularies 'kind' must be a non-empty list" in err
                for err in results["inventory"]),
            msg=results["inventory"],
        )

    def test_vocabulary_category_empty_list_is_reported(self):
        # An empty declared vocabulary would vacuously pass every membership
        # check, so it is itself a schema error rather than silently disabling
        # the vocabulary gate for that category.
        broken = VALID_INVENTORY.replace("  kind: [contract]\n", "  kind: []\n", 1)
        _write_docs(self.root, {"test_inventory.yaml": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("vocabularies 'kind' must be a non-empty list" in err
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

    def test_source_missing_build_option_is_reported(self):
        # The source block must name the CMake switch that enables the suite; a
        # missing build_option is a schema error.
        broken = VALID_INVENTORY.replace(
            "      build_option: BUILD_CONTRACT_TESTS\n", "", 1)
        _write_docs(self.root, {"test_inventory.yaml": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("source is missing required 'build_option'" in err
                for err in results["inventory"]),
            msg=results["inventory"],
        )

    def test_source_missing_build_target_is_reported(self):
        broken = VALID_INVENTORY.replace(
            "      build_target: contract_tests\n", "", 1)
        _write_docs(self.root, {"test_inventory.yaml": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("source is missing required 'build_target'" in err
                for err in results["inventory"]),
            msg=results["inventory"],
        )

    def test_source_blank_build_option_is_reported(self):
        # A present-but-empty build_option names no switch and is a schema error.
        broken = VALID_INVENTORY.replace(
            "      build_option: BUILD_CONTRACT_TESTS\n",
            '      build_option: ""\n', 1)
        _write_docs(self.root, {"test_inventory.yaml": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("source.build_option must be a non-empty string" in err
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

    def test_demote_does_not_require_disposition_target(self):
        # demote, like migrate, describes a lifecycle change whose destination may
        # be external, so omitting disposition_target is not an error.
        broken = VALID_INVENTORY.replace(
            "  disposition: [keep, merge_into]\n",
            "  disposition: [keep, merge_into, demote]\n", 1).replace(
            "    disposition: keep\n", "    disposition: demote\n", 1)
        _write_docs(self.root, {"test_inventory.yaml": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertEqual(results["inventory"], [], msg=results["inventory"])

    # -- retire disposition: replacement + evidence enforcement --------------

    def test_valid_retire_suite_passes(self):
        # A retiring suite that names its replacement id and documents coverage
        # equivalence, CI evidence, and owner approval is consistent.
        _write_docs(self.root, {"test_inventory.yaml": _with_retire(RETIRE_SUITE)})
        results, _ = checker.validate(root=str(self.root))
        self.assertEqual(results["inventory"], [], msg=results["inventory"])

    def test_retire_requires_disposition_target(self):
        block = RETIRE_SUITE.replace(
            "    disposition_target: [suite-one]\n", "", 1)
        _write_docs(self.root, {"test_inventory.yaml": _with_retire(block)})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("must name disposition_target" in err and "retire" in err
                for err in results["inventory"]),
            msg=results["inventory"],
        )

    def test_retire_target_must_resolve(self):
        block = RETIRE_SUITE.replace(
            "    disposition_target: [suite-one]\n",
            "    disposition_target: [suite-missing]\n", 1)
        _write_docs(self.root, {"test_inventory.yaml": _with_retire(block)})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("is not an inventory id" in err for err in results["inventory"]),
            msg=results["inventory"],
        )

    def test_retire_requires_retirement_mapping(self):
        block = RETIRE_SUITE.replace(
            "    retirement:\n"
            "      coverage_equivalence: suite-one pins the same invariants and adds OOB checks.\n"
            "      ci_evidence: Runs in the contract lane under BUILD_CONTRACT_TESTS.\n"
            "      owner_approval: Approved by Team Beta on 2026-11-10.\n",
            "", 1)
        _write_docs(self.root, {"test_inventory.yaml": _with_retire(block)})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("must carry a retirement mapping" in err
                for err in results["inventory"]),
            msg=results["inventory"],
        )

    def test_retire_missing_coverage_equivalence_is_reported(self):
        block = RETIRE_SUITE.replace(
            "      coverage_equivalence: suite-one pins the same invariants and adds OOB checks.\n",
            "", 1)
        _write_docs(self.root, {"test_inventory.yaml": _with_retire(block)})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("missing a non-empty 'coverage_equivalence' field" in err
                for err in results["inventory"]),
            msg=results["inventory"],
        )

    def test_retire_missing_ci_evidence_is_reported(self):
        block = RETIRE_SUITE.replace(
            "      ci_evidence: Runs in the contract lane under BUILD_CONTRACT_TESTS.\n",
            "", 1)
        _write_docs(self.root, {"test_inventory.yaml": _with_retire(block)})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("missing a non-empty 'ci_evidence' field" in err
                for err in results["inventory"]),
            msg=results["inventory"],
        )

    def test_retire_missing_owner_approval_is_reported(self):
        block = RETIRE_SUITE.replace(
            "      owner_approval: Approved by Team Beta on 2026-11-10.\n",
            "", 1)
        _write_docs(self.root, {"test_inventory.yaml": _with_retire(block)})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("missing a non-empty 'owner_approval' field" in err
                for err in results["inventory"]),
            msg=results["inventory"],
        )

    # -- repository-path-bearing fields: stale values ------------------------

    def test_source_of_truth_stale_value_is_reported(self):
        # A non-external source_of_truth value that does not resolve on disk is a
        # stale pointer and must be reported.
        broken = VALID_INVENTORY.replace(
            '  suite_placement: "external: catch/*/CMakeLists.txt"\n',
            "  suite_placement: catch/does/not/exist.txt\n", 1)
        _write_docs(self.root, {"test_inventory.yaml": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("source_of_truth 'suite_placement' path missing" in err
                for err in results["inventory"]),
            msg=results["inventory"],
        )

    def test_source_of_truth_glob_resolves(self):
        # A glob value that matches at least one file on disk resolves cleanly.
        (self.root / "probe_lane.yml").write_text("x\n", encoding="utf-8")
        broken = VALID_INVENTORY.replace(
            '  suite_placement: "external: catch/*/CMakeLists.txt"\n',
            "  suite_placement: probe_*.yml\n", 1)
        _write_docs(self.root, {"test_inventory.yaml": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertEqual(results["inventory"], [], msg=results["inventory"])

    def test_glob_symlink_within_boundary_resolves(self):
        # A glob that matches a symlink which resolves (following the link) to a
        # file INSIDE the boundary is safe and resolves cleanly.
        (self.root / "real_target.yml").write_text("x\n", encoding="utf-8")
        os.symlink(self.root / "real_target.yml", self.root / "probe_link.yml")
        broken = VALID_INVENTORY.replace(
            '  suite_placement: "external: catch/*/CMakeLists.txt"\n',
            "  suite_placement: probe_*.yml\n", 1)
        _write_docs(self.root, {"test_inventory.yaml": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertEqual(results["inventory"], [], msg=results["inventory"])

    def test_glob_symlink_escaping_boundary_is_rejected(self):
        # A glob that matches a mix of an in-boundary file and a symlink whose
        # target resolves OUTSIDE the boundary must be rejected: a single safe
        # match may not mask the escaping one.
        outside = tempfile.TemporaryDirectory()
        self.addCleanup(outside.cleanup)
        outside_file = Path(outside.name) / "escape_target.yml"
        outside_file.write_text("x\n", encoding="utf-8")
        (self.root / "probe_safe.yml").write_text("x\n", encoding="utf-8")
        os.symlink(outside_file, self.root / "probe_escape.yml")
        broken = VALID_INVENTORY.replace(
            '  suite_placement: "external: catch/*/CMakeLists.txt"\n',
            "  suite_placement: probe_*.yml\n", 1)
        _write_docs(self.root, {"test_inventory.yaml": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("escapes repository root" in err for err in results["inventory"]),
            msg=results["inventory"],
        )

    def test_repository_path_absolute_value_is_rejected(self):
        broken = VALID_INVENTORY.replace(
            '  suite_placement: "external: catch/*/CMakeLists.txt"\n',
            "  suite_placement: /tmp\n", 1)
        _write_docs(self.root, {"test_inventory.yaml": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("must be repository-relative" in err for err in results["inventory"]),
            msg=results["inventory"],
        )

    def test_repository_path_escaping_traversal_is_rejected(self):
        broken = VALID_INVENTORY.replace(
            '  suite_placement: "external: catch/*/CMakeLists.txt"\n',
            "  suite_placement: ../../../../tmp\n", 1)
        _write_docs(self.root, {"test_inventory.yaml": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("escapes repository root" in err for err in results["inventory"]),
            msg=results["inventory"],
        )

    def test_metadata_config_stale_value_is_reported(self):
        broken = VALID_INVENTORY.replace(
            "    metadata:\n      framework: Catch2\n",
            "    metadata:\n      framework: Catch2\n"
            "      config: catch/does/not/exist.yaml\n", 1)
        _write_docs(self.root, {"test_inventory.yaml": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("metadata.config path missing" in err
                for err in results["inventory"]),
            msg=results["inventory"],
        )

    def test_metadata_generated_plan_stale_value_is_reported(self):
        broken = VALID_INVENTORY.replace(
            "    metadata:\n      framework: Catch2\n",
            "    metadata:\n      framework: Catch2\n"
            "      generated_plan: catch/NO_SUCH_PLAN.md\n", 1)
        _write_docs(self.root, {"test_inventory.yaml": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("metadata.generated_plan path missing" in err
                for err in results["inventory"]),
            msg=results["inventory"],
        )

    def test_ci_lane_in_repo_stale_value_is_reported(self):
        # An in-root lane path that resolves nowhere is a stale pointer. The path
        # stays within the supplied --root boundary so the failure is the missing
        # file, not a traversal escape (escape is covered separately below).
        broken = VALID_INVENTORY.replace(
            "      ci_lane:\n        external: documented elsewhere\n",
            "      ci_lane:\n"
            "        in_repo: workflows/does-not-exist.yml\n", 1)
        _write_docs(self.root, {"test_inventory.yaml": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("execution.ci_lane.in_repo path missing" in err
                for err in results["inventory"]),
            msg=results["inventory"],
        )

    def test_ci_lane_in_repo_escaping_supplied_root_is_rejected(self):
        # With an explicit --root the supplied root is the boundary: an in_repo
        # lane that traverses above it (the real-repo ``../../.github`` form, which
        # is legitimate only when the component sits two levels under the repo
        # root) escapes the temporary root and must be rejected, not silently
        # resolved against some ancestor of the temp directory.
        broken = VALID_INVENTORY.replace(
            "      ci_lane:\n        external: documented elsewhere\n",
            "      ci_lane:\n"
            "        in_repo: ../../.github/workflows/hip.yml\n", 1)
        _write_docs(self.root, {"test_inventory.yaml": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("execution.ci_lane.in_repo escapes repository root" in err
                for err in results["inventory"]),
            msg=results["inventory"],
        )

    def test_ci_lane_in_repo_qualifier_is_stripped(self):
        # The leading path token is the file; a trailing job/step qualifier is
        # descriptive and must not be treated as part of the path.
        (self.root / "made_up_lane.yml").write_text("name: x\n", encoding="utf-8")
        broken = VALID_INVENTORY.replace(
            "      ci_lane:\n        external: documented elsewhere\n",
            "      ci_lane:\n"
            '        in_repo: made_up_lane.yml (job build, step "run tests")\n', 1)
        _write_docs(self.root, {"test_inventory.yaml": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertEqual(results["inventory"], [], msg=results["inventory"])

    def test_ci_lane_empty_mapping_is_reported(self):
        # A ci_lane mapping that carries neither a non-empty in_repo nor a
        # non-empty external lane names no CI lane at all and is a schema error:
        # the suite would claim CI coverage it does not actually point at.
        broken = VALID_INVENTORY.replace(
            "      ci_lane:\n        external: documented elsewhere\n",
            "      ci_lane: {}\n", 1)
        _write_docs(self.root, {"test_inventory.yaml": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("execution.ci_lane must carry a non-empty in_repo or external"
                in err for err in results["inventory"]),
            msg=results["inventory"],
        )

    def test_ci_lane_empty_in_repo_and_external_is_reported(self):
        # Present-but-empty in_repo / external values are no lane either; the
        # mapping must carry at least one usable lane string.
        broken = VALID_INVENTORY.replace(
            "      ci_lane:\n        external: documented elsewhere\n",
            '      ci_lane:\n        in_repo: ""\n        external: ""\n', 1)
        _write_docs(self.root, {"test_inventory.yaml": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("execution.ci_lane must carry a non-empty in_repo or external"
                in err for err in results["inventory"]),
            msg=results["inventory"],
        )

    # -- robustness: scalar / unhashable nested collections ------------------

    def test_axis_classes_not_a_list_is_reported(self):
        # A non-list axis.classes must produce a deterministic error, not crash.
        broken = VALID_COVERAGE.replace(
            "    - id: api_surface\n      classes: [memory]\n",
            "    - id: api_surface\n      classes: 5\n", 1)
        _write_docs(self.root, {"test_coverage_model.yaml": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("classes must be a list" in err
                for err in results["coverage_model"]),
            msg=results["coverage_model"],
        )

    def test_pairwise_id_unhashable_is_reported(self):
        # An unhashable pairwise id must be reported, not raise on set membership.
        broken = VALID_COVERAGE.replace(
            "  pairwise_checks:\n",
            "  pairwise_checks:\n"
            "    - id: [not, hashable]\n"
            "      category: cross\n"
            "      pair: [api_surface.memory, capability.baseline]\n"
            "      status: proposed\n"
            "      priority: high\n"
            "      rationale: Exercise the id guard.\n"
            "      current_inventory: []\n"
            "      gap_status: open\n", 1)
        _write_docs(self.root, {"test_coverage_model.yaml": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("pairwise check id must be a non-empty string" in err
                for err in results["coverage_model"]),
            msg=results["coverage_model"],
        )

    def test_vocabulary_entry_not_scalar_is_reported(self):
        # A nested (unhashable) vocabulary entry must be reported, not raise on
        # set construction.
        broken = VALID_INVENTORY.replace(
            "  kind: [contract]\n", "  kind: [contract, [nested]]\n", 1)
        _write_docs(self.root, {"test_inventory.yaml": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("vocabularies 'kind' entry must be a non-empty scalar" in err
                for err in results["inventory"]),
            msg=results["inventory"],
        )

    # -- suite vocabulary tokens must be scalar strings (no TypeError) --------

    def test_suite_kind_non_scalar_is_reported(self):
        # A list-valued (unhashable) kind must be reported as a schema error and
        # must not crash the vocabulary set-membership test.
        broken = VALID_INVENTORY.replace(
            "    kind: contract\n", "    kind: [contract]\n", 1)
        _write_docs(self.root, {"test_inventory.yaml": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("kind must be a non-empty string" in err
                for err in results["inventory"]),
            msg=results["inventory"],
        )

    def test_suite_disposition_non_scalar_is_reported(self):
        # A list-valued disposition must be reported and must not crash the
        # move-disposition set-membership test downstream.
        broken = VALID_INVENTORY.replace(
            "    disposition: keep\n", "    disposition: [keep]\n", 1)
        _write_docs(self.root, {"test_inventory.yaml": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("disposition must be a non-empty string" in err
                for err in results["inventory"]),
            msg=results["inventory"],
        )

    def test_suite_gating_non_scalar_is_reported(self):
        # A list-valued execution.gating must be reported and must not crash the
        # gating set-membership test.
        broken = VALID_INVENTORY.replace(
            "      gating: trusted\n", "      gating: [trusted]\n", 1)
        _write_docs(self.root, {"test_inventory.yaml": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("gating must be a non-empty string" in err
                for err in results["inventory"]),
            msg=results["inventory"],
        )

    # -- disposition_target shape --------------------------------------------

    def test_disposition_target_mapping_shape_is_reported(self):
        # A mapping disposition_target is neither a string nor a list of ids; it
        # must be reported as a shape error, not crash the id-membership test.
        broken = VALID_INVENTORY.replace(
            "    disposition: keep\n",
            "    disposition: merge_into\n"
            "    disposition_target:\n      dest: suite-one\n", 1)
        _write_docs(self.root, {"test_inventory.yaml": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("disposition_target must be a non-empty string or a non-empty list"
                in err for err in results["inventory"]),
            msg=results["inventory"],
        )

    def test_disposition_target_non_string_entry_is_reported(self):
        # A list disposition_target whose entry is not a non-empty string is a
        # schema error and must not crash the unhashable set-membership test.
        broken = VALID_INVENTORY.replace(
            "    disposition: keep\n",
            "    disposition: merge_into\n    disposition_target: [123]\n", 1)
        _write_docs(self.root, {"test_inventory.yaml": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("disposition_target entry must be a non-empty inventory id" in err
                for err in results["inventory"]),
            msg=results["inventory"],
        )

    # -- suite evidence must be a non-empty list -----------------------------

    def test_suite_evidence_empty_list_is_reported(self):
        # An empty suite evidence list is a schema error, mirroring the
        # coverage-class evidence requirement.
        broken = VALID_INVENTORY.replace(
            '    evidence:\n'
            '      - path: "external: evidence-outside-repo"\n'
            "        claim: Built under BUILD_CONTRACT_TESTS.\n",
            "    evidence: []\n", 1)
        _write_docs(self.root, {"test_inventory.yaml": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("suite 'suite-one' evidence must be a non-empty list" in err
                for err in results["inventory"]),
            msg=results["inventory"],
        )

    def test_suite_evidence_non_list_is_reported(self):
        # A present but non-list (scalar) suite evidence block must be reported,
        # not silently skip the per-entry validation.
        broken = VALID_INVENTORY.replace(
            '    evidence:\n'
            '      - path: "external: evidence-outside-repo"\n'
            "        claim: Built under BUILD_CONTRACT_TESTS.\n",
            "    evidence: 5\n", 1)
        _write_docs(self.root, {"test_inventory.yaml": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("suite 'suite-one' evidence must be a non-empty list" in err
                for err in results["inventory"]),
            msg=results["inventory"],
        )

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

    def test_duplicate_coverage_axis_id_is_reported(self):
        # Two axis records sharing an id make the axis ambiguous: a later
        # declaration silently shadows the earlier one. Duplicating an existing
        # axis (same id and classes) isolates the duplicate-id error.
        broken = VALID_COVERAGE.replace(
            "    - id: api_surface\n      classes: [memory]\n",
            "    - id: api_surface\n      classes: [memory]\n"
            "    - id: api_surface\n      classes: [memory]\n", 1)
        _write_docs(self.root, {"test_coverage_model.yaml": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("duplicate coverage axis id: 'api_surface'" in err
                for err in results["coverage_model"]),
            msg=results["coverage_model"],
        )

    def test_duplicate_class_within_axis_is_reported(self):
        # A class listed twice in one axis is a dead duplicate: it silently
        # collapses to a single declared axis.class, so the repeat must be
        # reported rather than deduplicated away.
        broken = VALID_COVERAGE.replace(
            "    - id: api_surface\n      classes: [memory]\n",
            "    - id: api_surface\n      classes: [memory, memory]\n", 1)
        _write_docs(self.root, {"test_coverage_model.yaml": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("coverage axis 'api_surface' has duplicate class 'memory'" in err
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

    def test_coverage_unsupported_schema_version_is_rejected(self):
        broken = VALID_COVERAGE.replace(
            "schema_version: 1\n", "schema_version: 2\n", 1)
        _write_docs(self.root, {"test_coverage_model.yaml": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("coverage model schema_version must be the supported version 1" in err
                for err in results["coverage_model"]),
            msg=results["coverage_model"],
        )

    def test_coverage_wrong_component_is_rejected(self):
        broken = VALID_COVERAGE.replace(
            "component: hip-tests\n", "component: rocprofiler-sdk\n", 1)
        _write_docs(self.root, {"test_coverage_model.yaml": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("coverage model component must be 'hip-tests'" in err
                for err in results["coverage_model"]),
            msg=results["coverage_model"],
        )

    # -- cross-document identity consistency ---------------------------------

    def test_identity_mismatch_schema_version_between_documents(self):
        # The inventory stays at version 1; the coverage model is bumped to 2.
        # Beyond the coverage model's own unsupported-version error, the cross-
        # document check reports that the two documents disagree.
        broken = VALID_COVERAGE.replace(
            "schema_version: 1\n", "schema_version: 2\n", 1)
        _write_docs(self.root, {"test_coverage_model.yaml": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("schema_version 2 does not match inventory schema_version 1" in err
                for err in results["coverage_model"]),
            msg=results["coverage_model"],
        )

    def test_identity_mismatch_component_between_documents(self):
        broken = VALID_COVERAGE.replace(
            "component: hip-tests\n", "component: rocprofiler-sdk\n", 1)
        _write_docs(self.root, {"test_coverage_model.yaml": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("component 'rocprofiler-sdk' does not match inventory "
                "component 'hip-tests'" in err
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
            "      gap_status: covered_dual_tier\n  pairwise_checks:\n",
            "      gap_status: covered_dual_tier\n    - not-a-mapping\n"
            "  pairwise_checks:\n",
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
            "  pairwise_checks:\n",
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

    # -- incident required field present but empty ---------------------------

    def test_incident_empty_finding_field_is_rejected(self):
        # Regression: a required field whose label is present but whose value is
        # empty (``**Finding.**`` with nothing after it) was accepted because the
        # check only tested for the bold label. An empty field tells a reviewer
        # nothing, so a present-but-empty label is now a schema error distinct
        # from an absent one.
        broken = VALID_INCIDENTS.replace(
            "**Finding.** A control looked removable but was not.\n",
            "**Finding.**\n")
        _write_docs(self.root, {"TESTING_INCIDENTS.md": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("has an empty 'Finding' field" in err
                for err in results["incidents"]),
            msg=results["incidents"],
        )
        # An empty field is reported as empty, not as missing: the label is there.
        self.assertFalse(
            any("missing the 'Finding' field" in err
                for err in results["incidents"]),
            msg=results["incidents"],
        )

    def test_incident_empty_origin_field_is_rejected(self):
        # The same empty-value rule holds for every required label; Origin is the
        # one a reader most needs when tracing a control back to its source.
        broken = VALID_INCIDENTS.replace(
            "**Origin.** Landed in `0d90b30fb6c`.\n",
            "**Origin.** \n")
        _write_docs(self.root, {"TESTING_INCIDENTS.md": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("has an empty 'Origin' field" in err
                for err in results["incidents"]),
            msg=results["incidents"],
        )

    def test_incident_empty_trailing_field_is_rejected(self):
        # The last field in a section has no following label or blank line, so its
        # value runs to the end of the section; an empty trailing field must still
        # be caught rather than swallowed by the end-of-section boundary.
        broken = VALID_INCIDENTS.replace(
            "**Retirement condition.** Only when the hazard cannot recur.\n",
            "**Retirement condition.**\n")
        _write_docs(self.root, {"TESTING_INCIDENTS.md": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("has an empty 'Retirement condition' field" in err
                for err in results["incidents"]),
            msg=results["incidents"],
        )

    def test_incident_prefix_lookalike_label_does_not_satisfy_requirement(self):
        # Regression: a required label must match the exact ``**Label.**`` syntax,
        # not merely be a prefix of a longer bold run. Renaming ``**Origin.**`` to
        # a lookalike ``**Origin story.**`` means the required ``Origin`` field is
        # absent; the lookalike must not paper over it.
        broken = VALID_INCIDENTS.replace(
            "**Origin.** Landed in `0d90b30fb6c`.\n",
            "**Origin story.** Landed in `0d90b30fb6c`.\n")
        _write_docs(self.root, {"TESTING_INCIDENTS.md": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("missing the 'Origin' field" in err
                for err in results["incidents"]),
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

    # -- inline link with an optional CommonMark title -----------------------

    def test_markdown_link_with_double_quoted_title_resolves(self):
        # Regression: an inline link may carry a title after the destination,
        # ``[text](dest "title")``. The pre-fix capture treated the whole
        # parenthesized span (destination plus title) as the path, so a valid
        # titled link to an existing file failed existence resolution. The title
        # is descriptive and is now stripped before the destination is resolved.
        titled = VALID_TESTING.replace(
            "(test_inventory.yaml)", '(test_inventory.yaml "The suite inventory")')
        _write_docs(self.root, {"TESTING.md": titled})
        results, _ = checker.validate(root=str(self.root))
        self.assertEqual(results["markdown_links"], [], msg=results["markdown_links"])

    def test_markdown_link_with_single_quoted_title_resolves(self):
        titled = VALID_TESTING.replace(
            "(test_inventory.yaml)", "(test_inventory.yaml 'The suite inventory')")
        _write_docs(self.root, {"TESTING.md": titled})
        results, _ = checker.validate(root=str(self.root))
        self.assertEqual(results["markdown_links"], [], msg=results["markdown_links"])

    def test_markdown_link_with_title_still_resolves_fragment_anchor(self):
        # A titled link to a fragment anchor must still validate the anchor after
        # the title is stripped: the destination (and its ``#fragment``) is what
        # resolves, not the title text.
        titled = VALID_TESTING.replace(
            "(#known-gaps)", '(#known-gaps "Jump to gaps")')
        _write_docs(self.root, {"TESTING.md": titled})
        results, _ = checker.validate(root=str(self.root))
        self.assertEqual(results["markdown_links"], [], msg=results["markdown_links"])

    def test_markdown_link_with_title_still_reports_broken_destination(self):
        # Stripping the title must not mask a genuinely broken destination: a
        # titled link to a missing file is still reported.
        titled = VALID_TESTING.replace(
            "(test_inventory.yaml)", '(no_such_file.yaml "Missing target")')
        _write_docs(self.root, {"TESTING.md": titled})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("broken local link" in err for err in results["markdown_links"]),
            msg=results["markdown_links"],
        )

    # -- inline link with CommonMark backslash-escaped destination ------------

    def test_markdown_link_escaped_parenthesis_destination_resolves(self):
        # CommonMark lets a destination escape ASCII punctuation such as
        # parentheses (``file\(1\).yaml``); the escapes are resolved before the
        # path is looked up so the link points at the real on-disk file instead
        # of failing with the backslashes embedded.
        (self.root / "file(1).yaml").write_text("# placeholder\n", encoding="utf-8")
        linked = VALID_TESTING.replace(
            "(test_inventory.yaml)", "(file\\(1\\).yaml)")
        _write_docs(self.root, {"TESTING.md": linked})
        results, _ = checker.validate(root=str(self.root))
        self.assertEqual(results["markdown_links"], [], msg=results["markdown_links"])

    # -- inline link with balanced parentheses in destination / title --------

    def test_markdown_link_with_parenthesized_destination_resolves(self):
        # Regression: a link destination may itself contain a balanced pair of
        # parentheses, e.g. ``[inventory](test_(v2)_inventory.yaml)``. The
        # pre-fix capture stopped at the first ``)`` and truncated the
        # destination, so a valid link to an existing file failed existence
        # resolution. The balanced-parenthesis scan now resolves the whole
        # destination.
        (self.root / "test_(v2)_inventory.yaml").write_text(
            "# placeholder\n", encoding="utf-8")
        linked = VALID_TESTING.replace(
            "(test_inventory.yaml)", "(test_(v2)_inventory.yaml)")
        _write_docs(self.root, {"TESTING.md": linked})
        results, _ = checker.validate(root=str(self.root))
        self.assertEqual(results["markdown_links"], [], msg=results["markdown_links"])

    def test_markdown_link_with_parenthesized_title_resolves(self):
        # Regression: an inline link title may be given in parentheses rather
        # than quotes, ``[inventory](test_inventory.yaml (The suite inventory))``.
        # The pre-fix capture stopped at the first ``)`` inside the title,
        # truncating the span so the title stripper never saw a balanced
        # ``(title)`` and the destination failed to resolve. The scan now yields
        # the whole body and the parenthesized title is stripped before the
        # destination is resolved.
        titled = VALID_TESTING.replace(
            "(test_inventory.yaml)",
            "(test_inventory.yaml (The suite inventory))")
        _write_docs(self.root, {"TESTING.md": titled})
        results, _ = checker.validate(root=str(self.root))
        self.assertEqual(results["markdown_links"], [], msg=results["markdown_links"])

    def test_markdown_link_with_parenthesized_destination_broken_is_reported(self):
        # The balanced-parenthesis scan must not mask a genuinely broken
        # destination: a link whose parenthesized destination names no file on
        # disk is still reported as broken.
        linked = VALID_TESTING.replace(
            "(test_inventory.yaml)", "(no_(v2)_file.yaml)")
        _write_docs(self.root, {"TESTING.md": linked})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("broken local link" in err for err in results["markdown_links"]),
            msg=results["markdown_links"],
        )

    # -- Markdown link boundary: absolute / traversal targets ----------------

    def test_markdown_link_absolute_target_is_rejected(self):
        # Regression: an absolute Markdown link target is rejected even when it
        # names a real file. The pre-fix link check resolved the target and only
        # tested existence, so an absolute path to an existing file resolved and
        # was silently accepted; the boundary now holds Markdown links to the
        # same repository-relative rule as every other path pointer.
        target = self.root / "abs_target.md"
        target.write_text("# Heading\n", encoding="utf-8")
        broken = VALID_TESTING.replace(
            "(test_inventory.yaml)", "({})".format(target))
        _write_docs(self.root, {"TESTING.md": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("link target must be repository-relative" in err
                for err in results["markdown_links"]),
            msg=results["markdown_links"],
        )

    def test_markdown_link_parent_traversal_target_is_rejected(self):
        # Regression: with an explicit --root the supplied root is the escape
        # boundary. A link that traverses above it to a real file genuinely
        # exists, so the pre-fix check resolved and accepted it; a correct
        # boundary rejects the escape even though the target is present.
        component = self.root / "component"
        component.mkdir()
        (self.root / "escape_target.md").write_text("# Heading\n", encoding="utf-8")
        broken = VALID_TESTING.replace(
            "(test_inventory.yaml)", "(../escape_target.md)")
        _write_docs(component, {"TESTING.md": broken})
        results, _ = checker.validate(root=str(component))
        self.assertTrue(
            any("link target escapes repository root" in err
                for err in results["markdown_links"]),
            msg=results["markdown_links"],
        )

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

    # -- coverage model collection-type guards (no TypeError) ----------------

    def _coverage_model(self, **coverage_model):
        return {"schema_version": 1, "component": "hip-tests",
                "coverage_model": coverage_model}

    def test_coverage_axes_not_a_list_is_reported(self):
        model = self._coverage_model(axes=5, classes=[], pairwise_checks=[])
        errors, _, _ = checker.check_coverage_model(model, set(), str(self.root))
        self.assertTrue(
            any("axes must be a list" in err for err in errors), msg=errors)

    def test_coverage_classes_not_a_list_is_reported(self):
        model = self._coverage_model(axes=[], classes=7, pairwise_checks=[])
        errors, _, _ = checker.check_coverage_model(model, set(), str(self.root))
        self.assertTrue(
            any("classes must be a list" in err for err in errors), msg=errors)

    def test_coverage_pairwise_checks_not_a_list_is_reported(self):
        model = self._coverage_model(axes=[], classes=[], pairwise_checks=9)
        errors, _, _ = checker.check_coverage_model(model, set(), str(self.root))
        self.assertTrue(
            any("pairwise_checks must be a list" in err for err in errors),
            msg=errors)

    def test_coverage_class_evidence_scalar_is_reported(self):
        # A present but non-list (scalar) class evidence must be reported, not
        # crash the per-entry iteration.
        cls = {"id": "c1", "owner": "o", "axes": [], "risk": "low",
               "intent": "i", "expected_test_kinds": [], "current_inventory": [],
               "evidence": 5, "gap_status": "x"}
        model = self._coverage_model(axes=[], classes=[cls], pairwise_checks=[])
        errors, _, _ = checker.check_coverage_model(model, set(), str(self.root))
        self.assertTrue(
            any("evidence must be a non-empty list" in err for err in errors),
            msg=errors)

    # -- duplicate heading anchors (GitHub -N suffix) ------------------------

    def test_duplicate_heading_anchor_suffix_resolves(self):
        # GitHub disambiguates repeated heading slugs with a ``-1`` suffix on the
        # second occurrence; a link to that suffixed anchor must resolve.
        testing = (
            "# Testing Strategy\n\n"
            "See the [first](#section) and the [second](#section-1).\n\n"
            "## Section\n\nFirst occurrence.\n\n"
            "## Section\n\nSecond occurrence.\n"
        )
        _write_docs(self.root, {"TESTING.md": testing})
        results, _ = checker.validate(root=str(self.root))
        self.assertEqual(
            results["markdown_links"], [], msg=results["markdown_links"])

    # -- strict YAML loader rejects duplicate keys ---------------------------

    def test_duplicate_yaml_key_is_rejected(self):
        # A mapping that repeats a key would silently drop the earlier value under
        # yaml.safe_load; the strict loader reports it as a parse failure instead.
        broken = VALID_INVENTORY.replace(
            "    disposition_rationale: Baseline contract coverage.\n",
            "    disposition_rationale: Baseline contract coverage.\n"
            "    disposition_rationale: A silently-dropped duplicate.\n", 1)
        _write_docs(self.root, {"test_inventory.yaml": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("does not parse as YAML" in err and "duplicate key" in err
                for err in results["load"]),
            msg=results["load"],
        )

    # -- strict YAML loader rejects cyclic aliases ---------------------------

    def test_cyclic_yaml_alias_is_rejected(self):
        # A self-referential anchor/alias builds a recursive object graph that a
        # later walk of the document cannot terminate on; the strict loader must
        # reject it as a parse failure rather than returning a cyclic structure.
        cyclic = "root: &anchor\n  child: *anchor\n" + VALID_INVENTORY
        _write_docs(self.root, {"test_inventory.yaml": cyclic})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("does not parse as YAML" in err and "cyclic anchor" in err
                for err in results["load"]),
            msg=results["load"],
        )

    def test_noncyclic_yaml_alias_is_allowed(self):
        # A non-cyclic alias refers to an already fully-composed anchor and is a
        # legitimate YAML convenience; the cyclic rejection must not over-reach to
        # an ordinary shared anchor, so the whole valid set still passes.
        shared = VALID_INVENTORY.replace(
            "  kind: [contract]\n", "  kind: [&c contract]\n", 1).replace(
            "    kind: contract\n", "    kind: *c\n", 1)
        _write_docs(self.root, {"test_inventory.yaml": shared})
        results, _ = checker.validate(root=str(self.root))
        self.assertEqual(_all_errors(results), [], msg=results)

    # -- required inventory field type / nonempty constraints ----------------

    def test_suite_empty_descriptive_field_is_rejected(self):
        broken = VALID_INVENTORY.replace(
            "    purpose: Pin a portable invariant.\n", '    purpose: ""\n', 1)
        _write_docs(self.root, {"test_inventory.yaml": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("purpose must be a non-empty string" in err
                for err in results["inventory"]),
            msg=results["inventory"],
        )

    def test_suite_empty_cadence_is_rejected(self):
        broken = VALID_INVENTORY.replace(
            "      cadence: pr\n", '      cadence: ""\n', 1)
        _write_docs(self.root, {"test_inventory.yaml": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("execution.cadence must be a non-empty string" in err
                for err in results["inventory"]),
            msg=results["inventory"],
        )

    def test_suite_cadence_must_be_declared_in_vocabulary(self):
        broken = VALID_INVENTORY.replace("      cadence: pr\n", "      cadence: build\n", 1)
        _write_docs(self.root, {"test_inventory.yaml": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("execution.cadence 'build' not in declared vocabulary" in err
                for err in results["inventory"]),
            msg=results["inventory"],
        )

    def test_suite_empty_platforms_list_is_rejected(self):
        broken = VALID_INVENTORY.replace(
            "      platforms: [linux]\n", "      platforms: []\n", 1)
        _write_docs(self.root, {"test_inventory.yaml": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("execution.platforms must be a non-empty list" in err
                for err in results["inventory"]),
            msg=results["inventory"],
        )

    def test_suite_ctest_labels_may_be_empty_but_must_be_a_list(self):
        # An empty ctest_labels list is allowed (a build-free static check registers
        # no CTest labels); a non-list value is a schema error.
        ok = VALID_INVENTORY.replace(
            "      ctest_labels: [contract]\n", "      ctest_labels: []\n", 1)
        _write_docs(self.root, {"test_inventory.yaml": ok})
        results, _ = checker.validate(root=str(self.root))
        self.assertEqual(results["inventory"], [], msg=results["inventory"])

        broken = VALID_INVENTORY.replace(
            "      ctest_labels: [contract]\n", "      ctest_labels: contract\n", 1)
        _write_docs(self.root, {"test_inventory.yaml": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("execution.ctest_labels must be a list" in err
                for err in results["inventory"]),
            msg=results["inventory"],
        )

    # -- coverage class / pairwise semantics ---------------------------------

    def test_class_empty_axes_is_rejected(self):
        cls = {"id": "c1", "owner": "o", "axes": [], "risk": "low",
               "intent": "i", "expected_test_kinds": [], "current_inventory": ["s"],
               "evidence": [{"path": "external: x", "claim": "c"}],
               "gap_status": "x"}
        model = self._coverage_model(axes=[], classes=[cls], pairwise_checks=[])
        errors, _, _ = checker.check_coverage_model(model, {"s"}, str(self.root))
        self.assertTrue(
            any("axes must be a non-empty list" in err for err in errors),
            msg=errors)

    def test_class_empty_current_inventory_is_rejected(self):
        cls = {"id": "c1", "owner": "o", "axes": ["a.b"], "risk": "low",
               "intent": "i", "expected_test_kinds": [], "current_inventory": [],
               "evidence": [{"path": "external: x", "claim": "c"}],
               "gap_status": "x"}
        model = self._coverage_model(axes=[], classes=[cls], pairwise_checks=[])
        errors, _, _ = checker.check_coverage_model(model, set(), str(self.root))
        self.assertTrue(
            any("current_inventory must be a non-empty list" in err
                for err in errors),
            msg=errors)

    def test_pairwise_pair_must_have_exactly_two_refs(self):
        pc = {"id": "p1", "category": "x", "pair": ["a.b"], "status": "s",
              "priority": "low", "rationale": "r", "current_inventory": [],
              "gap_status": "x"}
        model = self._coverage_model(axes=[], classes=[], pairwise_checks=[pc])
        errors, _, _ = checker.check_coverage_model(model, set(), str(self.root))
        self.assertTrue(
            any("pair must name exactly two" in err for err in errors),
            msg=errors)

    def test_pairwise_pair_requires_distinct_axes(self):
        broken = VALID_COVERAGE.replace(
            "pair: [api_surface.memory, behavior.data_integrity]",
            "pair: [api_surface.memory, api_surface.other]", 1).replace(
            "    - id: api_surface\n      classes: [memory]\n",
            "    - id: api_surface\n      classes: [memory, other]\n", 1)
        _write_docs(self.root, {"test_coverage_model.yaml": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("must reference distinct axes" in err
                for err in results["coverage_model"]),
            msg=results["coverage_model"],
        )

    def test_pairwise_pair_requires_distinct_members(self):
        broken = VALID_COVERAGE.replace(
            "pair: [api_surface.memory, behavior.data_integrity]",
            "pair: [api_surface.memory, api_surface.memory]", 1)
        _write_docs(self.root, {"test_coverage_model.yaml": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("must reference distinct members" in err
                for err in results["coverage_model"]),
            msg=results["coverage_model"],
        )

    # -- reference-style Markdown links --------------------------------------

    def test_reference_style_link_resolves(self):
        testing = (
            "# Testing Strategy\n\n"
            "See the [inventory][inv].\n\n"
            "## Known gaps\n\nBody.\n\n"
            "[inv]: test_inventory.yaml\n"
        )
        _write_docs(self.root, {"TESTING.md": testing})
        results, _ = checker.validate(root=str(self.root))
        self.assertEqual(
            results["markdown_links"], [], msg=results["markdown_links"])

    def test_reference_style_collapsed_link_resolves(self):
        testing = (
            "# Testing Strategy\n\n"
            "See [test_inventory.yaml][].\n\n"
            "## Known gaps\n\nBody.\n\n"
            "[test_inventory.yaml]: test_inventory.yaml\n"
        )
        _write_docs(self.root, {"TESTING.md": testing})
        results, _ = checker.validate(root=str(self.root))
        self.assertEqual(
            results["markdown_links"], [], msg=results["markdown_links"])

    def test_reference_style_label_internal_whitespace_normalized(self):
        # CommonMark collapses internal whitespace in a reference label, so a
        # usage and a definition that differ only in internal spacing name the
        # same label and resolve - neither "no matching definition" nor the
        # definition reported as "never used".
        testing = (
            "# Testing Strategy\n\n"
            "See the [the inventory][big   label].\n\n"
            "## Known gaps\n\nBody.\n\n"
            "[big label]: test_inventory.yaml\n"
        )
        _write_docs(self.root, {"TESTING.md": testing})
        results, _ = checker.validate(root=str(self.root))
        self.assertEqual(
            results["markdown_links"], [], msg=results["markdown_links"])

    def test_reference_style_link_without_definition_is_reported(self):
        testing = (
            "# Testing Strategy\n\n"
            "See the [inventory][inv].\n\n"
            "## Known gaps\n\nBody.\n"
        )
        _write_docs(self.root, {"TESTING.md": testing})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("has no matching definition" in err
                for err in results["markdown_links"]),
            msg=results["markdown_links"],
        )

    def test_reference_style_definition_unused_is_reported(self):
        testing = (
            "# Testing Strategy\n\n"
            "Body with no reference usage.\n\n"
            "## Known gaps\n\nBody.\n\n"
            "[inv]: test_inventory.yaml\n"
        )
        _write_docs(self.root, {"TESTING.md": testing})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("is never used" in err for err in results["markdown_links"]),
            msg=results["markdown_links"],
        )

    def test_reference_style_link_broken_target_is_reported(self):
        testing = (
            "# Testing Strategy\n\n"
            "See the [inventory][inv].\n\n"
            "## Known gaps\n\nBody.\n\n"
            "[inv]: no_such_file.yaml\n"
        )
        _write_docs(self.root, {"TESTING.md": testing})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("broken local link" in err for err in results["markdown_links"]),
            msg=results["markdown_links"],
        )

    # -- code spans and fenced code blocks are not links ---------------------

    def test_link_inside_inline_code_span_is_ignored(self):
        # A ``[text](path)`` inside an inline code span is shown, not linked: the
        # destination names no file on disk but must not be reported as broken.
        testing = (
            "# Testing Strategy\n\n"
            "Write a link like `[name](no_such_file.md)` to point at a doc.\n\n"
            "## Known gaps\n\nBody.\n"
        )
        _write_docs(self.root, {"TESTING.md": testing})
        results, _ = checker.validate(root=str(self.root))
        self.assertEqual(
            results["markdown_links"], [], msg=results["markdown_links"])

    def test_link_inside_fenced_code_block_is_ignored(self):
        # A ``[text](path)`` inside a fenced code block is example text, not a
        # link, and must not be existence-checked.
        testing = (
            "# Testing Strategy\n\n"
            "Example:\n\n"
            "```\n"
            "[broken](no_such_file.md)\n"
            "[label]: also_missing.md\n"
            "```\n\n"
            "## Known gaps\n\nBody.\n"
        )
        _write_docs(self.root, {"TESTING.md": testing})
        results, _ = checker.validate(root=str(self.root))
        self.assertEqual(
            results["markdown_links"], [], msg=results["markdown_links"])

    def test_reference_definition_inside_fenced_code_block_is_ignored(self):
        # A ``[label]: target`` line inside a fenced code block is not a real
        # reference definition, so it is neither resolved nor reported as unused.
        testing = (
            "# Testing Strategy\n\n"
            "```\n"
            "[inv]: no_such_file.yaml\n"
            "```\n\n"
            "## Known gaps\n\nBody.\n"
        )
        _write_docs(self.root, {"TESTING.md": testing})
        results, _ = checker.validate(root=str(self.root))
        self.assertEqual(
            results["markdown_links"], [], msg=results["markdown_links"])

    def test_tilde_fenced_code_block_is_ignored(self):
        # Fences may be drawn with tildes as well as backticks; a link inside a
        # ``~~~`` block is still example text.
        testing = (
            "# Testing Strategy\n\n"
            "~~~\n"
            "[broken](no_such_file.md)\n"
            "~~~\n\n"
            "## Known gaps\n\nBody.\n"
        )
        _write_docs(self.root, {"TESTING.md": testing})
        results, _ = checker.validate(root=str(self.root))
        self.assertEqual(
            results["markdown_links"], [], msg=results["markdown_links"])

    def test_real_link_outside_code_still_validated(self):
        # Masking code must not blind the checker to real links: a genuine broken
        # link outside any code span/fence is still reported even when a code
        # example with its own (ignored) link is present in the same document.
        testing = (
            "# Testing Strategy\n\n"
            "A real [inventory](no_such_file.yaml) link.\n\n"
            "Example code: `[x](also_ignored.md)` and:\n\n"
            "```\n"
            "[y](still_ignored.md)\n"
            "```\n\n"
            "## Known gaps\n\nBody.\n"
        )
        _write_docs(self.root, {"TESTING.md": testing})
        results, _ = checker.validate(root=str(self.root))
        broken = [err for err in results["markdown_links"]
                  if "broken local link" in err]
        self.assertEqual(len(broken), 1, msg=results["markdown_links"])
        self.assertIn("no_such_file.yaml", broken[0])

    def test_real_link_with_inline_code_in_text_still_resolves(self):
        # A real link whose visible text is an inline code span but whose
        # destination lies outside the backticks (``[`inv`](test_inventory.yaml)``)
        # must still resolve: only the code span in the text is masked, not the
        # link itself.
        testing = (
            "# Testing Strategy\n\n"
            "See [`inv`](test_inventory.yaml) and the [gaps](#known-gaps).\n\n"
            "## Known gaps\n\nBody.\n"
        )
        _write_docs(self.root, {"TESTING.md": testing})
        results, _ = checker.validate(root=str(self.root))
        self.assertEqual(
            results["markdown_links"], [], msg=results["markdown_links"])

    # -- Markdown local link target must be a file, not a directory ----------

    def test_markdown_link_to_directory_is_reported(self):
        # A local link that resolves to a directory exposes no document to read
        # and is held to the same file-not-directory rule as any evidence path.
        (self.root / "a_directory").mkdir()
        broken = VALID_TESTING.replace(
            "(test_inventory.yaml)", "(a_directory)")
        _write_docs(self.root, {"TESTING.md": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("link target is a directory, not a file" in err
                for err in results["markdown_links"]),
            msg=results["markdown_links"],
        )

    # -- historical context: history or incidents required -------------------

    def test_suite_missing_history_and_incidents_is_reported(self):
        # A suite that documents neither a history narrative nor an incidents
        # list is history-blind and must be reported.
        broken = VALID_INVENTORY.replace(
            "    history: Guarded per TI-001.\n", "", 1)
        _write_docs(self.root, {"test_inventory.yaml": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("must carry governing incidents or a history that establishes "
                "real context" in err
                for err in results["inventory"]),
            msg=results["inventory"],
        )

    def test_suite_incidents_list_satisfies_history(self):
        # A non-empty incidents list naming a defined incident satisfies the
        # historical-context requirement in place of a history narrative, and the
        # listed id resolves like any other TI reference.
        broken = VALID_INVENTORY.replace(
            "    history: Guarded per TI-001.\n", "    incidents: [TI-001]\n", 1)
        _write_docs(self.root, {"test_inventory.yaml": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertEqual(results["inventory"], [], msg=results["inventory"])
        self.assertEqual(results["incidents"], [], msg=results["incidents"])

    def test_suite_empty_history_is_reported(self):
        broken = VALID_INVENTORY.replace(
            "    history: Guarded per TI-001.\n", '    history: ""\n', 1)
        _write_docs(self.root, {"test_inventory.yaml": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("history must be a non-empty string" in err
                for err in results["inventory"]),
            msg=results["inventory"],
        )

    def test_suite_incidents_empty_list_is_reported(self):
        broken = VALID_INVENTORY.replace(
            "    history: Guarded per TI-001.\n", "    incidents: []\n", 1)
        _write_docs(self.root, {"test_inventory.yaml": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("incidents must be a non-empty list" in err
                for err in results["inventory"]),
            msg=results["inventory"],
        )

    def test_suite_incidents_non_ti_entry_is_reported(self):
        # An incidents entry that is not a bare TI-NNN id is a schema error.
        broken = VALID_INVENTORY.replace(
            "    history: Guarded per TI-001.\n", "    incidents: [not-an-incident]\n", 1)
        _write_docs(self.root, {"test_inventory.yaml": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("incidents entry must be a TI-NNN incident id" in err
                for err in results["inventory"]),
            msg=results["inventory"],
        )

    def test_suite_incidents_unknown_ti_is_reported(self):
        # A well-formed TI id that no heading defines must fail the incident
        # reference check, exactly as an unknown TI in a history string would.
        broken = VALID_INVENTORY.replace(
            "    history: Guarded per TI-001.\n", "    incidents: [TI-404]\n", 1)
        _write_docs(self.root, {"test_inventory.yaml": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("TI-404" in err for err in results["incidents"]),
            msg=results["incidents"],
        )

    # -- strict loader: unhashable mapping key -------------------------------

    def test_unhashable_yaml_key_is_rejected(self):
        # A complex (list) mapping key is unhashable; the strict loader must
        # report it as a parse failure rather than crashing on set membership.
        broken = "? [complex, key]\n: value\n" + VALID_INVENTORY
        _write_docs(self.root, {"test_inventory.yaml": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("does not parse as YAML" in err and "unhashable key" in err
                for err in results["load"]),
            msg=results["load"],
        )

    # -- evidence paths must be files, not directories -----------------------

    def test_suite_evidence_path_directory_is_reported(self):
        (self.root / "evidence_dir").mkdir()
        broken = VALID_INVENTORY.replace(
            '      - path: "external: evidence-outside-repo"\n', "      - path: evidence_dir\n")
        _write_docs(self.root, {"test_inventory.yaml": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("evidence path is a directory, not a file" in err
                for err in results["inventory"]),
            msg=results["inventory"],
        )

    def test_coverage_class_evidence_path_directory_is_reported(self):
        (self.root / "evidence_dir").mkdir()
        broken = VALID_COVERAGE.replace(
            '        - path: "external: evidence-outside-repo"\n',
            "        - path: evidence_dir\n")
        _write_docs(self.root, {"test_coverage_model.yaml": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("evidence path is a directory, not a file" in err
                for err in results["coverage_model"]),
            msg=results["coverage_model"],
        )

    def test_coverage_class_evidence_path_outside_repository_is_rejected(self):
        broken = VALID_COVERAGE.replace(
            '        - path: "external: evidence-outside-repo"\n',
            "        - path: ../../../../tmp\n")
        _write_docs(self.root, {"test_coverage_model.yaml": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("evidence path escapes repository root" in err
                for err in results["coverage_model"]),
            msg=results["coverage_model"],
        )

    # -- coverage class / pairwise required field validation -----------------

    def test_class_risk_not_in_vocabulary_is_reported(self):
        broken = VALID_COVERAGE.replace("      risk: high\n", "      risk: critical\n", 1)
        _write_docs(self.root, {"test_coverage_model.yaml": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("risk 'critical' not in" in err for err in results["coverage_model"]),
            msg=results["coverage_model"],
        )

    def test_class_empty_intent_is_reported(self):
        broken = VALID_COVERAGE.replace(
            "      intent: Allocation invariants remain observable.\n",
            '      intent: ""\n', 1)
        _write_docs(self.root, {"test_coverage_model.yaml": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("intent must be a non-empty string" in err
                for err in results["coverage_model"]),
            msg=results["coverage_model"],
        )

    def test_class_empty_expected_test_kinds_is_reported(self):
        broken = VALID_COVERAGE.replace(
            "      expected_test_kinds: [contract]\n",
            "      expected_test_kinds: []\n", 1)
        _write_docs(self.root, {"test_coverage_model.yaml": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("expected_test_kinds must be a non-empty list" in err
                for err in results["coverage_model"]),
            msg=results["coverage_model"],
        )

    def test_class_expected_test_kind_must_be_nonempty_string(self):
        broken = VALID_COVERAGE.replace(
            "      expected_test_kinds: [contract]\n",
            "      expected_test_kinds: ['']\n", 1)
        _write_docs(self.root, {"test_coverage_model.yaml": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("expected_test_kinds entry must be a non-empty string" in err
                for err in results["coverage_model"]),
            msg=results["coverage_model"],
        )

    def test_class_expected_test_kind_must_be_inventory_kind(self):
        broken = VALID_COVERAGE.replace(
            "      expected_test_kinds: [contract]\n",
            "      expected_test_kinds: [functional]\n", 1)
        _write_docs(self.root, {"test_coverage_model.yaml": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("expected_test_kinds entry 'functional' not in inventory kind vocabulary" in err
                for err in results["coverage_model"]),
            msg=results["coverage_model"],
        )

    def test_pairwise_priority_not_in_vocabulary_is_reported(self):
        pc = {"id": "p1", "category": "x", "pair": ["a.b", "c.d"], "status": "s",
              "priority": "urgent", "rationale": "r", "current_inventory": [],
              "gap_status": "x"}
        model = self._coverage_model(axes=[], classes=[], pairwise_checks=[pc])
        errors, _, _ = checker.check_coverage_model(model, set(), str(self.root))
        self.assertTrue(
            any("priority 'urgent' not in" in err for err in errors), msg=errors)

    def test_pairwise_empty_category_is_reported(self):
        pc = {"id": "p1", "category": "", "pair": ["a.b", "c.d"], "status": "s",
              "priority": "low", "rationale": "r", "current_inventory": [],
              "gap_status": "x"}
        model = self._coverage_model(axes=[], classes=[], pairwise_checks=[pc])
        errors, _, _ = checker.check_coverage_model(model, set(), str(self.root))
        self.assertTrue(
            any("category must be a non-empty string" in err for err in errors),
            msg=errors)

    # -- duplicate incident definition ---------------------------------------

    def test_duplicate_incident_definition_is_reported(self):
        # A second ## TI-001: heading makes the id ambiguous and must be reported.
        broken = VALID_INCIDENTS + (
            "\n## TI-001: A duplicate heading\n\n"
            "**Finding.** x\n**Control.** x\n**Protects.** x\n"
            "**Origin.** x\n**Current status.** x\n**Retirement condition.** x\n")
        _write_docs(self.root, {"TESTING_INCIDENTS.md": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("duplicate incident definition: TI-001" in err
                for err in results["incidents"]),
            msg=results["incidents"],
        )

    # -- pairwise_checks: non-empty and all required categories --------------

    def test_empty_pairwise_checks_is_reported(self):
        # The pairwise block is the model's cross-axis audit; an empty block
        # claims no audit at all and must be rejected.
        model = self._coverage_model(axes=[], classes=[], pairwise_checks=[])
        errors, _, _ = checker.check_coverage_model(model, set(), str(self.root))
        self.assertTrue(
            any("pairwise_checks must be a non-empty list" in err for err in errors),
            msg=errors)

    def test_missing_required_pairwise_category_is_reported(self):
        # Drop the capability_x_cadence category (re-label its entry) so a whole
        # required audit dimension is absent; the pair refs are left intact so the
        # only new error is the missing category.
        broken = VALID_COVERAGE.replace(
            "      category: capability_x_cadence\n",
            "      category: api_x_behavior\n", 1)
        _write_docs(self.root, {"test_coverage_model.yaml": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("missing required categories" in err
                and "capability_x_cadence" in err
                for err in results["coverage_model"]),
            msg=results["coverage_model"],
        )

    # -- declared axis classes must be audited (or marked informational) -----

    def test_unreferenced_axis_class_is_reported(self):
        # A declared class that no coverage class or pairwise check references,
        # and that is not marked informational, is a dead vocabulary entry.
        broken = VALID_COVERAGE.replace(
            "    - id: execution_context\n      classes: [single_gpu]\n",
            "    - id: execution_context\n      classes: [single_gpu, multi_gpu]\n", 1)
        _write_docs(self.root, {"test_coverage_model.yaml": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("execution_context.multi_gpu" in err and "never referenced" in err
                for err in results["coverage_model"]),
            msg=results["coverage_model"],
        )

    def test_informational_axis_class_is_exempt(self):
        # The same unreferenced class passes once the axis explicitly marks it
        # informational.
        broken = VALID_COVERAGE.replace(
            "    - id: execution_context\n      classes: [single_gpu]\n",
            "    - id: execution_context\n      classes: [single_gpu, multi_gpu]\n"
            "      informational_classes: [multi_gpu]\n", 1)
        _write_docs(self.root, {"test_coverage_model.yaml": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertEqual(_all_errors(results), [], msg=results)

    def test_informational_class_not_declared_is_reported(self):
        # An informational marker may only name a class the axis actually declares;
        # marking an undeclared class is a schema error, not an exemption.
        broken = VALID_COVERAGE.replace(
            "    - id: execution_context\n      classes: [single_gpu]\n",
            "    - id: execution_context\n      classes: [single_gpu]\n"
            "      informational_classes: [multi_gpu]\n", 1)
        _write_docs(self.root, {"test_coverage_model.yaml": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("informational class 'multi_gpu' is not a declared class" in err
                for err in results["coverage_model"]),
            msg=results["coverage_model"],
        )

    # -- history: a bare/generic placeholder is not historical context -------

    def test_bare_history_placeholder_is_reported(self):
        # The stock "origin not located; repository evidence records current
        # purpose." line must no longer pass on its own: it is a bare placeholder.
        broken = VALID_INVENTORY.replace(
            "    history: Guarded per TI-001.\n",
            "    history: Historical origin not located; repository evidence "
            "records current purpose.\n", 1)
        _write_docs(self.root, {"test_inventory.yaml": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("bare/generic placeholder" in err for err in results["inventory"]),
            msg=results["inventory"],
        )

    def test_history_without_evidence_marker_is_reported(self):
        # A narrative with real-looking prose but no evidence/origin marker does
        # not establish provenance and is rejected.
        broken = VALID_INVENTORY.replace(
            "    history: Guarded per TI-001.\n",
            "    history: This suite checks allocation behavior.\n", 1)
        _write_docs(self.root, {"test_inventory.yaml": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("evidence/origin marker" in err for err in results["inventory"]),
            msg=results["inventory"],
        )

    def test_history_with_evidence_marker_and_context_passes(self):
        # An honest unknown-origin narrative that points at the repository
        # evidence that does exist (a concrete source/build path, no invented
        # ticket) is accepted: the record-specific path is what anchors it.
        broken = VALID_INVENTORY.replace(
            "    history: Guarded per TI-001.\n",
            "    history: Historical origin not located in available change "
            "history; the behavior it pins is evidenced by its catalogued "
            "sources under catch/contract/transfer built via "
            "catch/contract/CMakeLists.txt.\n", 1)
        _write_docs(self.root, {"test_inventory.yaml": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertEqual(_all_errors(results), [], msg=results)

    def test_lightly_expanded_generic_history_is_reported(self):
        # Regression: a generic "origin not located" placeholder that is lightly
        # reworded with a few extra words (but still names no evidence specific to
        # THIS record) must not pass on the strength of the "repository evidence"
        # marker alone. A generic unknown-origin narrative now has to cite
        # record-specific context (a concrete repository path or a governing
        # incident), so the reworded template is reported.
        broken = VALID_INVENTORY.replace(
            "    history: Guarded per TI-001.\n",
            "    history: Historical origin not located in available change "
            "history; repository evidence (source paths and CMake wiring) "
            "documents the behavior it currently pins.\n", 1)
        _write_docs(self.root, {"test_inventory.yaml": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("bare/generic placeholder" in err
                and "record-specific context" in err
                for err in results["inventory"]),
            msg=results["inventory"],
        )

    def test_generic_unknown_origin_with_incident_passes(self):
        # A generic unknown-origin opener is still accepted when the suite points
        # at a governing incident: the incident is the record-specific context.
        broken = VALID_INVENTORY.replace(
            "    history: Guarded per TI-001.\n",
            "    history: Historical origin not located in available change "
            "history; the control is governed by TI-001.\n", 1)
        _write_docs(self.root, {"test_inventory.yaml": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertEqual(_all_errors(results), [], msg=results)


    # -- supplied --root is the path boundary (temporary-root escape) --------

    def test_supplied_root_is_boundary_rejects_escape_to_existing_file(self):
        # Regression: with an explicit --root the supplied root is the escape
        # boundary. Place a real file in the PARENT of the supplied root and
        # reference it with a traversal from a source path. The file genuinely
        # exists, so under the old ``parents[1]`` boundary it would resolve and be
        # silently accepted; a correct boundary rejects the escape even though the
        # target is present.
        component = self.root / "component"
        component.mkdir()
        (self.root / "escape_target.txt").write_text("x\n", encoding="utf-8")
        broken = VALID_INVENTORY.replace(
            '      paths:\n        - "external: outside-this-repo"\n',
            "      paths:\n        - ../escape_target.txt\n", 1)
        _write_docs(component, {"test_inventory.yaml": broken})
        results, _ = checker.validate(root=str(component))
        self.assertTrue(
            any("source path escapes repository root" in err
                for err in results["inventory"]),
            msg=results["inventory"],
        )

    def test_default_boundary_allows_sibling_tree_but_root_does_not(self):
        # Without an explicit root the boundary is two levels above the component
        # directory, so a component at <root>/a/b may legitimately reference
        # <root>/sibling via ``../../sibling`` (the real ``../../.github``
        # pattern). Supplying that same component as --root makes it the boundary,
        # which rejects the identical traversal.
        comp = self.root / "a" / "b"
        comp.mkdir(parents=True)
        (self.root / "sibling").mkdir()
        self.assertEqual(
            checker._path_location_status(str(comp), "../../sibling"), "ok")
        self.assertEqual(
            checker._path_location_status(
                str(comp), "../../sibling", repo_root=str(comp)),
            "outside")

    # -- related_restructuring: non-empty string + target path ---------------

    def _with_related(self, value):
        """Return the valid inventory with a related_restructuring line appended."""
        return VALID_INVENTORY.replace(
            "    disposition_rationale: Baseline contract coverage.\n",
            "    disposition_rationale: Baseline contract coverage.\n"
            "    related_restructuring: {}\n".format(value), 1)

    def test_related_restructuring_valid_target_resolves(self):
        # A present pointer that names a real in-root document resolves cleanly.
        _write_docs(self.root,
                    {"test_inventory.yaml": self._with_related("TEST_RESTRUCTURING.md")})
        results, _ = checker.validate(root=str(self.root))
        self.assertEqual(results["inventory"], [], msg=results["inventory"])

    def test_related_restructuring_empty_string_is_reported(self):
        _write_docs(self.root, {"test_inventory.yaml": self._with_related('""')})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("related_restructuring must be a non-empty string" in err
                for err in results["inventory"]),
            msg=results["inventory"],
        )

    def test_related_restructuring_non_string_scalar_is_reported(self):
        _write_docs(self.root, {"test_inventory.yaml": self._with_related("5")})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("related_restructuring must be a non-empty string" in err
                for err in results["inventory"]),
            msg=results["inventory"],
        )

    def test_related_restructuring_list_is_reported(self):
        _write_docs(self.root,
                    {"test_inventory.yaml": self._with_related("[a, b]")})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("related_restructuring must be a non-empty string" in err
                for err in results["inventory"]),
            msg=results["inventory"],
        )

    def test_related_restructuring_mapping_is_reported(self):
        # A mapping value is appended as a nested block under the key.
        broken = VALID_INVENTORY.replace(
            "    disposition_rationale: Baseline contract coverage.\n",
            "    disposition_rationale: Baseline contract coverage.\n"
            "    related_restructuring:\n      doc: TEST_RESTRUCTURING.md\n", 1)
        _write_docs(self.root, {"test_inventory.yaml": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("related_restructuring must be a non-empty string" in err
                for err in results["inventory"]),
            msg=results["inventory"],
        )

    def test_related_restructuring_missing_target_is_reported(self):
        _write_docs(self.root,
                    {"test_inventory.yaml": self._with_related("NO_SUCH_DOC.md")})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("related_restructuring missing" in err
                for err in results["inventory"]),
            msg=results["inventory"],
        )

    def test_related_restructuring_absolute_path_is_reported(self):
        _write_docs(self.root, {"test_inventory.yaml": self._with_related("/tmp")})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("related_restructuring must be repository-relative" in err
                for err in results["inventory"]),
            msg=results["inventory"],
        )

    def test_related_restructuring_escaping_traversal_is_reported(self):
        _write_docs(self.root,
                    {"test_inventory.yaml": self._with_related("../../../../tmp")})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("related_restructuring escapes repository root" in err
                for err in results["inventory"]),
            msg=results["inventory"],
        )

    # -- execution list-element validation -----------------------------------

    def test_platforms_entry_must_be_nonempty_string(self):
        broken = VALID_INVENTORY.replace(
            "      platforms: [linux]\n", "      platforms: ['']\n", 1)
        _write_docs(self.root, {"test_inventory.yaml": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("execution.platforms entry must be a non-empty string" in err
                for err in results["inventory"]),
            msg=results["inventory"],
        )

    def test_backends_entry_null_is_reported(self):
        broken = VALID_INVENTORY.replace(
            "      backends: [amd]\n", "      backends: [~]\n", 1)
        _write_docs(self.root, {"test_inventory.yaml": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("execution.backends entry must be a non-empty string" in err
                for err in results["inventory"]),
            msg=results["inventory"],
        )

    def test_prerequisites_entry_non_string_is_reported(self):
        broken = VALID_INVENTORY.replace(
            "      prerequisites: [HIP runtime and device]\n",
            "      prerequisites: [5]\n", 1)
        _write_docs(self.root, {"test_inventory.yaml": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("execution.prerequisites entry must be a non-empty string" in err
                for err in results["inventory"]),
            msg=results["inventory"],
        )

    def test_ctest_labels_entry_must_be_nonempty_string(self):
        # ctest_labels may be empty, but a label it does carry must be a non-empty
        # scalar string.
        broken = VALID_INVENTORY.replace(
            "      ctest_labels: [contract]\n", "      ctest_labels: ['']\n", 1)
        _write_docs(self.root, {"test_inventory.yaml": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("execution.ctest_labels entry must be a non-empty string" in err
                for err in results["inventory"]),
            msg=results["inventory"],
        )

    # -- pairwise current_inventory: nonempty or explicit gap metadata -------

    def test_pairwise_empty_current_inventory_without_gap_is_reported(self):
        pc = {"id": "p1", "category": "api_x_behavior", "pair": ["a.b", "c.d"],
              "status": "s", "priority": "low", "rationale": "r",
              "current_inventory": [], "gap_status": "x"}
        model = self._coverage_model(axes=[], classes=[], pairwise_checks=[pc])
        errors, _, _ = checker.check_coverage_model(model, set(), str(self.root))
        self.assertTrue(
            any("must name a non-empty current_inventory or carry explicit gap"
                in err for err in errors),
            msg=errors)

    def test_pairwise_empty_current_inventory_with_gap_metadata_passes(self):
        # An explicit gap acknowledgement lets a check document a known uncovered
        # interaction without silently claiming empty coverage.
        pc = {"id": "p1", "category": "api_x_behavior", "pair": ["a.b", "c.d"],
              "status": "s", "priority": "low", "rationale": "r",
              "current_inventory": [], "gap_status": "x", "gap": True}
        model = self._coverage_model(axes=[], classes=[], pairwise_checks=[pc])
        errors, _, _ = checker.check_coverage_model(model, set(), str(self.root))
        self.assertFalse(
            any("must name a non-empty current_inventory" in err for err in errors),
            msg=errors)

    def test_pairwise_empty_current_inventory_in_full_model_is_reported(self):
        # The same rule fires through the full validate() path: emptying one
        # pairwise check's current_inventory (with no gap marker) is reported.
        broken = VALID_COVERAGE.replace(
            "      pair: [api_surface.memory, behavior.data_integrity]\n"
            "      status: review_priority\n"
            "      priority: high\n"
            "      rationale: Round-trips preserve bytes, not only return success.\n"
            "      current_inventory: [suite-one]\n",
            "      pair: [api_surface.memory, behavior.data_integrity]\n"
            "      status: review_priority\n"
            "      priority: high\n"
            "      rationale: Round-trips preserve bytes, not only return success.\n"
            "      current_inventory: []\n", 1)
        _write_docs(self.root, {"test_coverage_model.yaml": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("must name a non-empty current_inventory or carry explicit gap"
                in err for err in results["coverage_model"]),
            msg=results["coverage_model"],
        )


    # -- review fix 1: duplicate reference-style link definitions ------------

    def test_reference_style_duplicate_definition_is_reported(self):
        # Two definitions for the same label are ambiguous: CommonMark keeps the
        # first and silently drops the rest, so a copy/pasted or conflicting
        # redefinition must be reported rather than swallowed.
        testing = (
            "# Testing Strategy\n\n"
            "See the [inventory][inv].\n\n"
            "## Known gaps\n\nBody.\n\n"
            "[inv]: test_inventory.yaml\n"
            "[inv]: test_coverage_model.yaml\n"
        )
        _write_docs(self.root, {"TESTING.md": testing})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("duplicate reference-style link definition" in err
                and "[inv]" in err
                for err in results["markdown_links"]),
            msg=results["markdown_links"],
        )

    def test_reference_style_duplicate_definition_is_case_insensitive(self):
        # Markdown reference labels are case-insensitive, so ``[inv]`` and
        # ``[INV]`` collide; the second (any-case) redefinition is reported and
        # the first is preserved as authoritative (no spurious "never used").
        testing = (
            "# Testing Strategy\n\n"
            "See the [inventory][inv].\n\n"
            "## Known gaps\n\nBody.\n\n"
            "[inv]: test_inventory.yaml\n"
            "[INV]: test_coverage_model.yaml\n"
        )
        _write_docs(self.root, {"TESTING.md": testing})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("duplicate reference-style link definition" in err
                for err in results["markdown_links"]),
            msg=results["markdown_links"],
        )
        # The surviving first definition is used, so no "never used" noise fires.
        self.assertFalse(
            any("is never used" in err for err in results["markdown_links"]),
            msg=results["markdown_links"],
        )

    def test_reference_style_single_definition_still_passes(self):
        # Regression guard: a lone definition is not mistaken for a duplicate.
        testing = (
            "# Testing Strategy\n\n"
            "See the [inventory][inv].\n\n"
            "## Known gaps\n\nBody.\n\n"
            "[inv]: test_inventory.yaml\n"
        )
        _write_docs(self.root, {"TESTING.md": testing})
        results, _ = checker.validate(root=str(self.root))
        self.assertEqual(
            results["markdown_links"], [], msg=results["markdown_links"])

    # -- review fix 2: empty/whitespace source.paths and bare external -------

    def test_empty_string_source_path_is_reported(self):
        # An empty source path resolves to the component directory itself (which
        # exists) and would otherwise count as a spuriously satisfied reference.
        broken = VALID_INVENTORY.replace(
            '        - "external: outside-this-repo"\n',
            '        - ""\n', 1)
        _write_docs(self.root, {"test_inventory.yaml": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("source path must be a non-empty string" in err
                for err in results["inventory"]),
            msg=results["inventory"],
        )

    def test_whitespace_only_source_path_is_reported(self):
        # A blank, whitespace-only path carries no target and must be rejected
        # rather than stripped to the component directory.
        broken = VALID_INVENTORY.replace(
            '        - "external: outside-this-repo"\n',
            '        - "   "\n', 1)
        _write_docs(self.root, {"test_inventory.yaml": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("source path must be a non-empty string" in err
                for err in results["inventory"]),
            msg=results["inventory"],
        )

    def test_bare_external_source_path_marker_is_reported(self):
        # The ``external:`` marker must name the external reference it stands in
        # for; a bare marker with no target names nothing.
        broken = VALID_INVENTORY.replace(
            '        - "external: outside-this-repo"\n',
            '        - "external:"\n', 1)
        _write_docs(self.root, {"test_inventory.yaml": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("external marker must name a target" in err
                for err in results["inventory"]),
            msg=results["inventory"],
        )

    def test_bare_external_source_path_marker_with_trailing_space_is_reported(self):
        # Trailing whitespace after the marker is still an empty target.
        broken = VALID_INVENTORY.replace(
            '        - "external: outside-this-repo"\n',
            '        - "external:   "\n', 1)
        _write_docs(self.root, {"test_inventory.yaml": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("external marker must name a target" in err
                for err in results["inventory"]),
            msg=results["inventory"],
        )

    def test_named_external_source_path_still_passes(self):
        # Regression guard: a named external target remains accepted.
        _write_docs(self.root)
        results, _ = checker.validate(root=str(self.root))
        self.assertEqual(results["inventory"], [], msg=results["inventory"])

    # -- review fix 3: active CMake suite groups vs inventory ----------------

    def _write_cmake_area(self, area, body):
        """Write a per-area catch/<area>/CMakeLists.txt under the temp root."""
        area_dir = self.root / "catch" / area
        area_dir.mkdir(parents=True, exist_ok=True)
        (area_dir / "CMakeLists.txt").write_text(body, encoding="utf-8")

    def _recon_inventory(self, paths=None, exclusions=None):
        """Build a one-suite inventory dict for reconciliation tests."""
        inv = {"suites": [{"id": "s1", "source": {"paths": list(paths or [])}}]}
        if exclusions is not None:
            inv["cmake_suite_group_exclusions"] = exclusions
        return inv

    def test_cmake_active_group_unrepresented_is_reported(self):
        self._write_cmake_area("unit", "add_subdirectory(memory)\n")
        errors = checker.check_cmake_reconciliation(
            self._recon_inventory(paths=[]), str(self.root))
        self.assertTrue(
            any("catch/unit/memory" in err and "not represented" in err
                for err in errors),
            msg=errors)

    def test_cmake_active_group_represented_passes(self):
        self._write_cmake_area("unit", "add_subdirectory(memory)\n")
        errors = checker.check_cmake_reconciliation(
            self._recon_inventory(paths=["catch/unit/memory"]), str(self.root))
        self.assertEqual(errors, [], msg=errors)

    def test_cmake_group_represented_by_path_beneath_passes(self):
        # A source path naming a file beneath the group directory represents it.
        self._write_cmake_area("unit", "add_subdirectory(memory)\n")
        errors = checker.check_cmake_reconciliation(
            self._recon_inventory(paths=["catch/unit/memory/alloc.cc"]),
            str(self.root))
        self.assertEqual(errors, [], msg=errors)

    def test_cmake_commented_group_is_not_required(self):
        # A commented-out add_subdirectory is a deliberately-disabled group and
        # must not be treated as an active group that needs a suite.
        self._write_cmake_area(
            "unit",
            "#add_subdirectory(gcc)   # TODO link error\n"
            "add_subdirectory(memory)\n")
        errors = checker.check_cmake_reconciliation(
            self._recon_inventory(paths=["catch/unit/memory"]), str(self.root))
        self.assertEqual(errors, [], msg=errors)

    def test_cmake_missing_area_is_feasibility_skipped(self):
        # No catch tree on disk -> the reconciliation is a no-op for a fixture
        # that ships no CMake tree.
        errors = checker.check_cmake_reconciliation(
            self._recon_inventory(paths=[]), str(self.root))
        self.assertEqual(errors, [], msg=errors)

    def test_cmake_exclusion_allows_unrepresented_group(self):
        # A documented, categorized exclusion suppresses the unrepresented report.
        self._write_cmake_area("unit", "add_subdirectory(gcc)\n")
        exclusions = [{
            "group": "catch/unit/gcc", "reason": "not_built",
            "note": "link error under TheRock build; kept for reference",
        }]
        errors = checker.check_cmake_reconciliation(
            self._recon_inventory(paths=[], exclusions=exclusions), str(self.root))
        self.assertEqual(errors, [], msg=errors)

    def test_cmake_exclusion_invalid_reason_is_reported(self):
        self._write_cmake_area("unit", "add_subdirectory(gcc)\n")
        exclusions = [{
            "group": "catch/unit/gcc", "reason": "because", "note": "x",
        }]
        errors = checker.check_cmake_reconciliation(
            self._recon_inventory(paths=[], exclusions=exclusions), str(self.root))
        self.assertTrue(
            any("reason must be one of" in err for err in errors), msg=errors)

    def test_cmake_exclusion_missing_note_is_reported(self):
        self._write_cmake_area("unit", "add_subdirectory(gcc)\n")
        exclusions = [{"group": "catch/unit/gcc", "reason": "not_built"}]
        errors = checker.check_cmake_reconciliation(
            self._recon_inventory(paths=[], exclusions=exclusions), str(self.root))
        self.assertTrue(
            any("must carry a non-empty note" in err for err in errors), msg=errors)

    def test_cmake_exclusion_for_represented_group_is_contradiction(self):
        # A group both covered by a suite and documented as excluded is
        # contradictory metadata and is reported.
        self._write_cmake_area("unit", "add_subdirectory(memory)\n")
        exclusions = [{
            "group": "catch/unit/memory", "reason": "not_built", "note": "x",
        }]
        errors = checker.check_cmake_reconciliation(
            self._recon_inventory(
                paths=["catch/unit/memory"], exclusions=exclusions),
            str(self.root))
        self.assertTrue(
            any("already represented" in err for err in errors), msg=errors)

    def test_cmake_exclusions_not_a_list_is_reported(self):
        self._write_cmake_area("unit", "add_subdirectory(gcc)\n")
        inv = self._recon_inventory(paths=[])
        inv["cmake_suite_group_exclusions"] = {"group": "catch/unit/gcc"}
        errors = checker.check_cmake_reconciliation(inv, str(self.root))
        self.assertTrue(
            any("cmake_suite_group_exclusions must be a list" in err
                for err in errors),
            msg=errors)

    def test_cmake_reconciliation_runs_through_validate(self):
        # End-to-end: the reconciliation is wired into validate() under its own
        # summary key and fires against a catch tree the inventory does not cover.
        _write_docs(self.root)
        self._write_cmake_area("unit", "add_subdirectory(memory)\n")
        results, _ = checker.validate(root=str(self.root))
        self.assertIn("cmake_reconciliation", results)
        self.assertTrue(
            any("catch/unit/memory" in err and "not represented" in err
                for err in results["cmake_reconciliation"]),
            msg=results.get("cmake_reconciliation"),
        )

    def test_cmake_quoted_add_subdirectory_is_recognized(self):
        # A quoted directory argument (add_subdirectory("memory")) names the same
        # active group as the unquoted form and must be reconciled identically.
        self._write_cmake_area("unit", 'add_subdirectory("memory")\n')
        errors = checker.check_cmake_reconciliation(
            self._recon_inventory(paths=[]), str(self.root))
        self.assertTrue(
            any("catch/unit/memory" in err and "not represented" in err
                for err in errors),
            msg=errors)

    def test_cmake_bracket_comment_group_is_not_required(self):
        # An add_subdirectory inside a CMake bracket comment (#[[ ... ]]), even
        # one spanning several lines, is commented out and must not register as an
        # active group that needs a suite.
        self._write_cmake_area(
            "unit",
            "#[[\n"
            "add_subdirectory(gcc)\n"
            "disabled block\n"
            "]]\n"
            "add_subdirectory(memory)\n")
        errors = checker.check_cmake_reconciliation(
            self._recon_inventory(paths=["catch/unit/memory"]), str(self.root))
        self.assertEqual(errors, [], msg=errors)

    # -- final fix: host-test (hip_add_host_test) reconciliation --------------

    def _write_host_cmake(self, body):
        """Write catch/hipTestMain/tests/CMakeLists.txt under the temp root."""
        host_dir = self.root / "catch" / "hipTestMain" / "tests"
        host_dir.mkdir(parents=True, exist_ok=True)
        (host_dir / "CMakeLists.txt").write_text(body, encoding="utf-8")

    def test_hosttest_function_declaration_is_not_registered(self):
        # The function declaration takes a formal TARGET parameter; it is not a
        # host-test registration. Calls retain direct, multiline, quoted, and
        # case-insensitive matching.
        self._write_host_cmake(
            "function(hip_add_host_test TARGET)\n"
            "endfunction()\n"
            "hip_add_host_test(DirectHostTest)\n"
            "HIP_ADD_HOST_TEST(\n"
            "  \"QuotedMultilineHostTest\"\n"
            ")\n")
        cmake_path = self.root / "catch" / "hipTestMain" / "tests" / "CMakeLists.txt"
        self.assertEqual(
            checker._active_host_test_targets(cmake_path),
            ["DirectHostTest", "QuotedMultilineHostTest"])

    def test_hosttest_spaced_function_declaration_is_not_registered(self):
        # A declaration with whitespace between the ``function`` keyword and its
        # ``(`` (``function (hip_add_host_test TARGET)``) is still a declaration,
        # not a registration of a target named ``TARGET``. The fixed-width
        # lookbehind this masking replaces could only exclude the zero-space
        # ``function(`` form; robust declaration detection excludes the spaced
        # form too while keeping the real registration that follows.
        self._write_host_cmake(
            "function (hip_add_host_test TARGET)\n"
            "  add_dependencies(build_tests ${TARGET})\n"
            "endfunction()\n"
            "hip_add_host_test(SpacedDeclHostTest)\n")
        cmake_path = self.root / "catch" / "hipTestMain" / "tests" / "CMakeLists.txt"
        self.assertEqual(
            checker._active_host_test_targets(cmake_path),
            ["SpacedDeclHostTest"])

    def test_hosttest_multiline_mixedcase_declaration_is_not_registered(self):
        # A declaration split across physical lines (the declared name on the line
        # after the opening paren) and written in mixed case is still a
        # declaration. Only the genuine registration call is enumerated.
        self._write_host_cmake(
            "FUNCTION(\n"
            "  hip_add_host_test TARGET\n"
            ")\n"
            "endfunction()\n"
            "hip_add_host_test(MultilineDeclHostTest)\n")
        cmake_path = self.root / "catch" / "hipTestMain" / "tests" / "CMakeLists.txt"
        self.assertEqual(
            checker._active_host_test_targets(cmake_path),
            ["MultilineDeclHostTest"])

    def test_hosttest_real_registrations_are_all_enumerated(self):
        # Real registrations in direct, quoted, multiline, and mixed-case forms
        # are all enumerated in source order; replacing the lookbehind with
        # declaration masking preserves every genuine registration.
        self._write_host_cmake(
            "function(hip_add_host_test TARGET)\n"
            "endfunction()\n"
            "hip_add_host_test(FirstHostTest)\n"
            'HIP_ADD_HOST_TEST("SecondHostTest")\n'
            "hip_add_host_test(\n"
            "  ThirdHostTest\n"
            "  ENVIRONMENT HIP_TEST_LEVEL=x\n"
            ")\n")
        cmake_path = self.root / "catch" / "hipTestMain" / "tests" / "CMakeLists.txt"
        self.assertEqual(
            checker._active_host_test_targets(cmake_path),
            ["FirstHostTest", "SecondHostTest", "ThirdHostTest"])

    def test_hosttest_unrepresented_is_reported(self):
        # A host-test executable registered via hip_add_host_test with no covering
        # inventory suite is a silent coverage hole and is reported, naming it.
        self._write_host_cmake(
            "hip_add_host_test(HipTestLevelUnitTest)\n"
            "hip_add_host_test(HipTestLevelAbortTest ENVIRONMENT HIP_TEST_LEVEL=x)\n")
        errors = checker.check_cmake_reconciliation(
            self._recon_inventory(paths=[]), str(self.root))
        self.assertTrue(
            any("HipTestLevelUnitTest" in err and "HipTestLevelAbortTest" in err
                and "hip_add_host_test" in err and "not represented" in err
                for err in errors),
            msg=errors)

    def test_hosttest_represented_passes(self):
        # A host/harness suite whose source.paths covers catch/hipTestMain/tests
        # represents the registered host tests; no report.
        self._write_host_cmake("hip_add_host_test(HipTestLevelUnitTest)\n")
        errors = checker.check_cmake_reconciliation(
            self._recon_inventory(paths=["catch/hipTestMain/tests"]),
            str(self.root))
        self.assertEqual(errors, [], msg=errors)

    def test_hosttest_represented_by_path_beneath_passes(self):
        # A source path naming a file beneath the host-test directory represents it.
        self._write_host_cmake("hip_add_host_test(HipTestLevelUnitTest)\n")
        errors = checker.check_cmake_reconciliation(
            self._recon_inventory(
                paths=["catch/hipTestMain/tests/hip_test_level_test.cc"]),
            str(self.root))
        self.assertEqual(errors, [], msg=errors)

    def test_hosttest_commented_registration_is_not_required(self):
        # A commented-out hip_add_host_test registers no active host test.
        self._write_host_cmake(
            "# hip_add_host_test(HipTestLevelUnitTest)\n")
        errors = checker.check_cmake_reconciliation(
            self._recon_inventory(paths=[]), str(self.root))
        self.assertEqual(errors, [], msg=errors)

    def test_hosttest_missing_file_is_feasibility_skipped(self):
        # No host-test CMakeLists on disk -> the host reconciliation is a no-op.
        errors = checker.check_cmake_reconciliation(
            self._recon_inventory(paths=[]), str(self.root))
        self.assertEqual(errors, [], msg=errors)

    def test_hosttest_exclusion_allows_unrepresented(self):
        # A documented exclusion for the host-test directory suppresses the report.
        self._write_host_cmake("hip_add_host_test(HipTestLevelUnitTest)\n")
        exclusions = [{
            "group": "catch/hipTestMain/tests", "reason": "no_cases",
            "note": "host harness tests tracked elsewhere",
        }]
        errors = checker.check_cmake_reconciliation(
            self._recon_inventory(paths=[], exclusions=exclusions), str(self.root))
        self.assertEqual(errors, [], msg=errors)

    def test_hosttest_missing_inventory_runs_through_validate(self):
        # End-to-end: with a host-test CMakeLists present but no covering suite,
        # validate() surfaces the host-test coverage hole under reconciliation.
        _write_docs(self.root)
        self._write_host_cmake("hip_add_host_test(HipTestLevelUnitTest)\n")
        results, _ = checker.validate(root=str(self.root))
        self.assertIn("cmake_reconciliation", results)
        self.assertTrue(
            any("HipTestLevelUnitTest" in err and "hip_add_host_test" in err
                for err in results["cmake_reconciliation"]),
            msg=results.get("cmake_reconciliation"),
        )

    # -- Step 5: command-aware CMake discovery (message false positives,
    #    quoted args, normalized inventory group paths / exclusions) ----------

    def test_cmake_message_string_add_subdirectory_is_not_detected(self):
        # Command-aware parsing only treats a real top-level ``add_subdirectory``
        # command as an active group. An ``add_subdirectory(...)`` token that
        # appears only inside another command's quoted string argument (here a
        # ``message(...)`` reminder) is an argument of that command, not a group
        # registration, so it must not demand an inventory suite.
        self._write_cmake_area(
            "unit",
            'message(STATUS "remember to add_subdirectory(gcc) once it links")\n'
            "add_subdirectory(memory)\n")
        errors = checker.check_cmake_reconciliation(
            self._recon_inventory(paths=["catch/unit/memory"]), str(self.root))
        self.assertEqual(errors, [], msg=errors)

    def test_cmake_message_string_with_paren_in_quotes_is_not_detected(self):
        # A quoted command argument may itself contain parentheses. Command-aware
        # parsing skips the quoted string when finding a command's argument span,
        # so neither the stray ``)`` nor the ``add_subdirectory(gcc)`` token inside
        # the quote ends the ``message`` command or registers a group; only the
        # genuine call that follows is active.
        self._write_cmake_area(
            "unit",
            'message(STATUS "note: do not add_subdirectory(gcc) here )")\n'
            "add_subdirectory(memory)\n")
        errors = checker.check_cmake_reconciliation(
            self._recon_inventory(paths=["catch/unit/memory"]), str(self.root))
        self.assertEqual(errors, [], msg=errors)

    def test_hosttest_message_string_registration_is_not_detected(self):
        # The same command-awareness protects host-test discovery: a
        # ``hip_add_host_test(...)`` token inside a ``message(...)`` string is an
        # argument, not a registration, so only the real call is enumerated.
        self._write_host_cmake(
            'message(STATUS "call hip_add_host_test(GhostTest) when ready")\n'
            "hip_add_host_test(RealHostTest)\n")
        cmake_path = self.root / "catch" / "hipTestMain" / "tests" / "CMakeLists.txt"
        self.assertEqual(
            checker._active_host_test_targets(cmake_path),
            ["RealHostTest"])

    def test_cmake_group_represented_by_normalized_source_path_passes(self):
        # Inventory group paths are normalized for reconciliation only: a source
        # path that names the same directory with a cosmetic ``./`` prefix still
        # represents the discovered ``catch/<area>/<group>`` group.
        self._write_cmake_area("unit", "add_subdirectory(memory)\n")
        errors = checker.check_cmake_reconciliation(
            self._recon_inventory(paths=["./catch/unit/memory"]), str(self.root))
        self.assertEqual(errors, [], msg=errors)

    def test_cmake_normalized_exclusion_allows_unrepresented_group(self):
        # A documented exclusion is matched after the same normalization, so an
        # exclusion that names the group with a ``./`` prefix and a trailing slash
        # still suppresses the unrepresented report for ``catch/unit/gcc``.
        self._write_cmake_area("unit", "add_subdirectory(gcc)\n")
        exclusions = [{
            "group": "./catch/unit/gcc/", "reason": "not_built",
            "note": "link error under TheRock build; kept for reference",
        }]
        errors = checker.check_cmake_reconciliation(
            self._recon_inventory(paths=[], exclusions=exclusions), str(self.root))
        self.assertEqual(errors, [], msg=errors)

    # -- review fix 4: incident Origin commit citations ----------------------

    def test_is_commit_hash_classifies_hashes_and_words(self):
        # A commit-like token is 7-40 hex chars with at least one digit; an
        # all-letter hex word ("defaced") and a too-short token are not hashes.
        self.assertTrue(checker._is_commit_hash("0d90b30fb6c"))
        self.assertTrue(checker._is_commit_hash("a1b2c3d"))
        self.assertFalse(checker._is_commit_hash("defaced"))
        self.assertFalse(checker._is_commit_hash("abc12"))
        self.assertFalse(checker._is_commit_hash("TI-001"))
        self.assertFalse(checker._is_commit_hash("not-a-hash"))

    def _init_git_repo(self):
        """Initialize a throwaway git repo in the temp root; return a runner."""
        def run(*args):
            return subprocess.run(
                ["git", "-C", str(self.root), *args],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                check=True)
        run("init", "-q")
        run("config", "user.email", "tester@example.com")
        run("config", "user.name", "Tester")
        run("config", "commit.gpgsign", "false")
        return run

    @unittest.skipUnless(_GIT, "git not available")
    def test_origin_citation_bogus_hash_is_reported(self):
        # A tracked incidents doc that cites a hash absent from the object
        # database makes provenance unfollowable and is reported.
        run = self._init_git_repo()
        path = self.root / "TESTING_INCIDENTS.md"
        path.write_text(VALID_INCIDENTS, encoding="utf-8")
        run("add", "TESTING_INCIDENTS.md")
        run("commit", "-q", "-m", "add incidents")
        errors = checker.check_incident_origin_citations(str(path))
        self.assertTrue(
            any("does not exist" in err and "0d90b30fb6c" in err
                for err in errors),
            msg=errors)

    @unittest.skipUnless(_GIT, "git not available")
    def test_origin_citation_real_commit_passes(self):
        # A citation that resolves to a real commit in the repo is accepted.
        run = self._init_git_repo()
        path = self.root / "TESTING_INCIDENTS.md"
        path.write_text("placeholder\n", encoding="utf-8")
        run("add", "TESTING_INCIDENTS.md")
        run("commit", "-q", "-m", "seed")
        head = run("rev-parse", "HEAD").stdout.strip()[:12]
        path.write_text(
            VALID_INCIDENTS.replace("`0d90b30fb6c`", "`{}`".format(head)),
            encoding="utf-8")
        errors = checker.check_incident_origin_citations(str(path))
        self.assertEqual(errors, [], msg=errors)

    @unittest.skipUnless(_GIT, "git not available")
    def test_origin_citation_external_marker_is_skipped(self):
        # An explicitly external citation is preserved as-is, not existence-checked.
        run = self._init_git_repo()
        path = self.root / "TESTING_INCIDENTS.md"
        path.write_text(
            VALID_INCIDENTS.replace(
                "Landed in `0d90b30fb6c`.", "Landed in `external: 0d90b30fb6c`."),
            encoding="utf-8")
        run("add", "TESTING_INCIDENTS.md")
        run("commit", "-q", "-m", "add incidents")
        errors = checker.check_incident_origin_citations(str(path))
        self.assertEqual(errors, [], msg=errors)

    @unittest.skipUnless(_GIT, "git not available")
    def test_origin_citation_unverified_marker_is_skipped(self):
        run = self._init_git_repo()
        path = self.root / "TESTING_INCIDENTS.md"
        path.write_text(
            VALID_INCIDENTS.replace(
                "Landed in `0d90b30fb6c`.", "Landed in `unverified: 0d90b30fb6c`."),
            encoding="utf-8")
        run("add", "TESTING_INCIDENTS.md")
        run("commit", "-q", "-m", "add incidents")
        errors = checker.check_incident_origin_citations(str(path))
        self.assertEqual(errors, [], msg=errors)

    @unittest.skipUnless(_GIT, "git not available")
    def test_origin_citation_untracked_file_is_skipped(self):
        # An untracked fixture under some ambient checkout cites commits of a
        # different repository, so its citations are not validated here.
        self._init_git_repo()
        path = self.root / "TESTING_INCIDENTS.md"
        path.write_text(VALID_INCIDENTS, encoding="utf-8")  # written, not committed
        errors = checker.check_incident_origin_citations(str(path))
        self.assertEqual(errors, [], msg=errors)

    def test_origin_citation_non_git_dir_is_skipped(self):
        # Outside a git work tree, hash existence cannot be verified; the check
        # is feasibility-gated to a no-op rather than a spurious failure.
        path = self.root / "TESTING_INCIDENTS.md"
        path.write_text(VALID_INCIDENTS, encoding="utf-8")
        errors = checker.check_incident_origin_citations(str(path))
        self.assertEqual(errors, [], msg=errors)

    # -- final fix 1: explicit --root usable with a real checkout -------------

    @unittest.skipUnless(_GIT, "git not available")
    def test_explicit_root_real_checkout_allows_sibling_github_reference(self):
        # Regression: with an explicit --root that points at a real checkout /
        # component root, the boundary is the git repository root - not the
        # supplied root - so a document that legitimately references a sibling
        # tree (the real ``../../.github`` pattern from projects/hip-tests) still
        # resolves instead of being rejected as an escape. The inventory is
        # tracked so the boundary derives the git root rather than the component.
        run = self._init_git_repo()
        component = self.root / "projects" / "hip-tests"
        component.mkdir(parents=True)
        workflows = self.root / ".github" / "workflows"
        workflows.mkdir(parents=True)
        (workflows / "hip.yml").write_text("name: x\n", encoding="utf-8")
        inv = VALID_INVENTORY.replace(
            "      ci_lane:\n        external: documented elsewhere\n",
            "      ci_lane:\n"
            "        in_repo: ../../.github/workflows/hip.yml\n", 1)
        _write_docs(component, {"test_inventory.yaml": inv})
        run("add", "projects/hip-tests/test_inventory.yaml")
        run("commit", "-q", "-m", "add inventory")
        results, _ = checker.validate(root=str(component))
        self.assertFalse(
            any("escapes repository root" in err
                for err in results["inventory"]),
            msg=results["inventory"],
        )

    def test_untracked_fixture_root_stays_bounded(self):
        # The companion guarantee: a self-contained temporary root (untracked,
        # even if it happened to sit under some ambient checkout) remains its own
        # boundary, so a ``../../`` traversal above it is still an escape.
        component = self.root / "projects" / "hip-tests"
        component.mkdir(parents=True)
        (self.root / "escape_target.txt").write_text("x\n", encoding="utf-8")
        broken = VALID_INVENTORY.replace(
            '      paths:\n        - "external: outside-this-repo"\n',
            "      paths:\n        - ../../escape_target.txt\n", 1)
        _write_docs(component, {"test_inventory.yaml": broken})
        results, _ = checker.validate(root=str(component))
        self.assertTrue(
            any("source path escapes repository root" in err
                for err in results["inventory"]),
            msg=results["inventory"],
        )

    # -- final fix 2: Markdown comments / escaped openers / fenced anchors ----

    def test_link_inside_html_comment_is_ignored(self):
        # A ``[text](path)`` hidden inside an HTML comment is not a live link and
        # must not be existence-checked.
        testing = (
            "# Testing Strategy\n\n"
            "See the [inventory](test_inventory.yaml).\n\n"
            "<!-- [hidden](no_such_file.md) -->\n\n"
            "## Known gaps\n\nBody.\n"
        )
        _write_docs(self.root, {"TESTING.md": testing})
        results, _ = checker.validate(root=str(self.root))
        self.assertEqual(
            results["markdown_links"], [], msg=results["markdown_links"])

    def test_link_inside_multiline_html_comment_is_ignored(self):
        # A comment may span multiple lines; a link anywhere inside it is still
        # hidden, and a real link after the comment is still validated.
        testing = (
            "# Testing Strategy\n\n"
            "<!--\n"
            "[hidden](no_such_file.md)\n"
            "still inside the comment\n"
            "-->\n\n"
            "A real [inventory](test_inventory.yaml) link.\n\n"
            "## Known gaps\n\nBody.\n"
        )
        _write_docs(self.root, {"TESTING.md": testing})
        results, _ = checker.validate(root=str(self.root))
        self.assertEqual(
            results["markdown_links"], [], msg=results["markdown_links"])

    def test_escaped_link_opener_is_ignored(self):
        # A backslash-escaped opener ``\[text](path)`` is literal text, not a
        # link, so its destination is neither validated nor reported.
        testing = (
            "# Testing Strategy\n\n"
            "A literal \\[name](no_such_file.md) shown as text.\n\n"
            "## Known gaps\n\nBody.\n"
        )
        _write_docs(self.root, {"TESTING.md": testing})
        results, _ = checker.validate(root=str(self.root))
        self.assertEqual(
            results["markdown_links"], [], msg=results["markdown_links"])

    def test_escaped_and_real_link_together_reports_only_real(self):
        # Masking the escaped opener must not blind the checker to a genuine
        # broken link on the same line.
        testing = (
            "# Testing Strategy\n\n"
            "A literal \\[a](ignored.md) and a real [b](no_such_file.md).\n\n"
            "## Known gaps\n\nBody.\n"
        )
        _write_docs(self.root, {"TESTING.md": testing})
        results, _ = checker.validate(root=str(self.root))
        broken = [err for err in results["markdown_links"]
                  if "broken local link" in err]
        self.assertEqual(len(broken), 1, msg=results["markdown_links"])
        self.assertIn("no_such_file.md", broken[0])

    # -- escape parity: even/odd backslash runs before a link opener ----------

    def test_even_backslash_run_before_inline_opener_is_live(self):
        # Two backslashes before ``[`` are an escaped-backslash literal, NOT an
        # escaped opener: the inline link stays live and its broken destination
        # must be reported. A simple immediate-backslash rejection would wrongly
        # drop this real (if broken) link.
        testing = (
            "# Testing Strategy\n\n"
            "A literal backslash then a real \\\\[name](no_such_file.md) link.\n\n"
            "## Known gaps\n\nBody.\n"
        )
        _write_docs(self.root, {"TESTING.md": testing})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("broken local link" in err and "no_such_file.md" in err
                for err in results["markdown_links"]),
            msg=results["markdown_links"],
        )

    def test_odd_backslash_run_before_inline_opener_is_escaped(self):
        # Three backslashes before ``[`` are an escaped backslash followed by an
        # escaped opener (odd run): the bracket is literal text, so the broken
        # destination must be neither validated nor reported.
        testing = (
            "# Testing Strategy\n\n"
            "An escaped \\\\\\[name](no_such_file.md) shown as text.\n\n"
            "## Known gaps\n\nBody.\n"
        )
        _write_docs(self.root, {"TESTING.md": testing})
        results, _ = checker.validate(root=str(self.root))
        self.assertEqual(
            results["markdown_links"], [], msg=results["markdown_links"])

    def test_even_backslash_run_before_reference_opener_is_live(self):
        # Two backslashes before a reference-style usage ``[text][label]`` leave
        # the opener live, so the usage is resolved through its definition: its
        # broken target is reported and the definition counts as used (no
        # "never used"). An even run must not be mistaken for an escape.
        testing = (
            "# Testing Strategy\n\n"
            "A literal backslash then a \\\\[inventory][inv] link.\n\n"
            "## Known gaps\n\nBody.\n\n"
            "[inv]: no_such_file.yaml\n"
        )
        _write_docs(self.root, {"TESTING.md": testing})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("broken local link" in err for err in results["markdown_links"]),
            msg=results["markdown_links"],
        )
        self.assertFalse(
            any("is never used" in err for err in results["markdown_links"]),
            msg=results["markdown_links"],
        )

    def test_odd_backslash_run_before_reference_opener_is_escaped(self):
        # One backslash before a reference-style usage escapes the opener (odd
        # run): the usage is literal text and is not resolved, so its definition
        # is left unused (reported as such) and no "no matching definition" is
        # raised for the escaped label.
        testing = (
            "# Testing Strategy\n\n"
            "An escaped \\[inventory][inv] shown as text.\n\n"
            "## Known gaps\n\nBody.\n\n"
            "[inv]: test_inventory.yaml\n"
        )
        _write_docs(self.root, {"TESTING.md": testing})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("is never used" in err for err in results["markdown_links"]),
            msg=results["markdown_links"],
        )
        self.assertFalse(
            any("has no matching definition" in err
                for err in results["markdown_links"]),
            msg=results["markdown_links"],
        )

    def test_fenced_heading_does_not_satisfy_anchor(self):
        # A ``# Heading`` inside a fenced code block is example text, not a real
        # heading, so an anchor pointing at its slug is broken.
        testing = (
            "# Testing Strategy\n\n"
            "See the [fake](#fake-heading).\n\n"
            "```\n"
            "# Fake Heading\n"
            "```\n\n"
            "## Known gaps\n\nBody.\n"
        )
        _write_docs(self.root, {"TESTING.md": testing})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("broken anchor" in err and "#fake-heading" in err
                for err in results["markdown_links"]),
            msg=results["markdown_links"],
        )

    def test_commented_heading_does_not_satisfy_anchor(self):
        # A ``# Heading`` inside an HTML comment is likewise not a real heading.
        testing = (
            "# Testing Strategy\n\n"
            "See the [hidden](#hidden-heading).\n\n"
            "<!-- # Hidden Heading -->\n\n"
            "## Known gaps\n\nBody.\n"
        )
        _write_docs(self.root, {"TESTING.md": testing})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("broken anchor" in err and "#hidden-heading" in err
                for err in results["markdown_links"]),
            msg=results["markdown_links"],
        )

    def test_heading_with_inline_code_still_anchors(self):
        # Regression guard for the fence/comment-only masking: a heading that
        # merely contains an inline ``code`` span keeps that text in its GitHub
        # slug, so an anchor to it must still resolve (inline code is NOT masked
        # when deriving heading anchors).
        testing = (
            "# Testing Strategy\n\n"
            "See the [api](#the-hipmalloc-api).\n\n"
            "## The `hipMalloc` api\n\nBody.\n"
        )
        _write_docs(self.root, {"TESTING.md": testing})
        results, _ = checker.validate(root=str(self.root))
        self.assertEqual(
            results["markdown_links"], [], msg=results["markdown_links"])

    def test_heading_with_markdown_link_slugs_from_label(self):
        # GitHub derives a heading's anchor from its rendered text, so a heading
        # that carries an inline link contributes only the visible link label -
        # the destination (and any title) are not part of the slug. An anchor to
        # the label-based slug must resolve, while the destination text that the
        # naive slug used to fold in (``...-httpsexamplecomguide``) is absent.
        testing = (
            "# Testing Strategy\n\n"
            "See the [guide](#the-testing-guide).\n\n"
            "## The [testing guide](https://example.com/guide)\n\nBody.\n"
        )
        _write_docs(self.root, {"TESTING.md": testing})
        results, _ = checker.validate(root=str(self.root))
        self.assertEqual(
            results["markdown_links"], [], msg=results["markdown_links"])

    def test_heading_link_keeps_surrounding_text_drops_destination(self):
        # Plain text on either side of an inline link is preserved in the slug
        # while the link destination is dropped, so an anchor built from the
        # surrounding words plus the link label resolves. The destination only
        # resolves to nothing in the slug: if it leaked in, the surrounding-text
        # anchor below would not match.
        testing = (
            "# Testing Strategy\n\n"
            "Jump to [it](#run-the-smoke-suite-before-merging).\n\n"
            "## Run the [smoke suite](https://example.com/smoke) before merging\n\n"
            "Body.\n"
        )
        _write_docs(self.root, {"TESTING.md": testing})
        results, _ = checker.validate(root=str(self.root))
        self.assertEqual(
            results["markdown_links"], [], msg=results["markdown_links"])

    def test_duplicate_linked_heading_anchor_suffix_resolves(self):
        # Two headings whose visible text reduces to the same label collide even
        # when their link destinations differ, so GitHub disambiguates the second
        # with a ``-1`` suffix. A link to both the base and the suffixed anchor
        # must resolve, proving the duplicate suffixing runs on the label-derived
        # slug rather than on destination-polluted text.
        testing = (
            "# Testing Strategy\n\n"
            "See [first](#the-plan) and [second](#the-plan-1).\n\n"
            "## The [plan](https://example.com/a)\n\nFirst.\n\n"
            "## The [plan](https://example.com/b)\n\nSecond.\n"
        )
        _write_docs(self.root, {"TESTING.md": testing})
        results, _ = checker.validate(root=str(self.root))
        self.assertEqual(
            results["markdown_links"], [], msg=results["markdown_links"])

    def test_heading_with_markdown_image_slugs_from_alt_text(self):
        # An inline image in a heading contributes only its alt text to the slug;
        # the image source is excluded exactly like a link destination. The
        # surrounding word after the image is kept, so the alt-plus-text anchor
        # resolves.
        testing = (
            "# Testing Strategy\n\n"
            "See the [badge](#status-badge-overview).\n\n"
            "## ![status badge](https://example.com/badge.png) overview\n\n"
            "Body.\n"
        )
        _write_docs(self.root, {"TESTING.md": testing})
        results, _ = checker.validate(root=str(self.root))
        self.assertEqual(
            results["markdown_links"], [], msg=results["markdown_links"])

    # -- final fix 3: CMake add_subdirectory case / multiline parsing ---------

    def test_cmake_uppercase_add_subdirectory_is_detected(self):
        # CMake commands are case-insensitive; an uppercase invocation names an
        # active group exactly like the lowercase form.
        self._write_cmake_area("unit", "ADD_SUBDIRECTORY(memory)\n")
        errors = checker.check_cmake_reconciliation(
            self._recon_inventory(paths=[]), str(self.root))
        self.assertTrue(
            any("catch/unit/memory" in err and "not represented" in err
                for err in errors),
            msg=errors)

    def test_cmake_multiline_add_subdirectory_is_detected(self):
        # A call whose group name sits on the line after the opening paren is
        # still an active group that must be represented.
        self._write_cmake_area(
            "unit", "add_subdirectory(\n    memory\n)\n")
        errors = checker.check_cmake_reconciliation(
            self._recon_inventory(paths=[]), str(self.root))
        self.assertTrue(
            any("catch/unit/memory" in err and "not represented" in err
                for err in errors),
            msg=errors)

    def test_cmake_multiline_with_inline_comment_is_detected(self):
        # Comment stripping is per-line before the command is scanned whole, so a
        # trailing ``# ...`` on the opening line does not hide the group that
        # follows on the next line; a represented group passes.
        self._write_cmake_area(
            "unit", "add_subdirectory( # add the memory group\n    memory\n)\n")
        errors = checker.check_cmake_reconciliation(
            self._recon_inventory(paths=["catch/unit/memory"]), str(self.root))
        self.assertEqual(errors, [], msg=errors)

    def test_cmake_commented_multiline_call_is_not_detected(self):
        # A fully commented-out multiline-style call contributes nothing, so only
        # the live group is required.
        self._write_cmake_area(
            "unit",
            "# add_subdirectory(\n#   gcc\n# )\n"
            "add_subdirectory(memory)\n")
        errors = checker.check_cmake_reconciliation(
            self._recon_inventory(paths=["catch/unit/memory"]), str(self.root))
        self.assertEqual(errors, [], msg=errors)

    def test_cmake_mixed_case_multiline_is_detected(self):
        # Case-insensitivity and multiline parsing combine: a MixedCase call split
        # across lines is detected.
        self._write_cmake_area(
            "unit", "Add_SubDirectory(\n  memory\n)\n")
        errors = checker.check_cmake_reconciliation(
            self._recon_inventory(paths=[]), str(self.root))
        self.assertTrue(
            any("catch/unit/memory" in err and "not represented" in err
                for err in errors),
            msg=errors)

    # -- final fix 4: unquoted commit-like hashes in Origin citations ---------

    @unittest.skipUnless(_GIT, "git not available")
    def test_origin_citation_unquoted_bogus_hash_is_reported(self):
        # A commit-like hash cited WITHOUT backticks is still validated: dropping
        # the backticks must not let a fabricated hash evade the check.
        run = self._init_git_repo()
        path = self.root / "TESTING_INCIDENTS.md"
        incidents = VALID_INCIDENTS.replace(
            "**Origin.** Landed in `0d90b30fb6c`.\n",
            "**Origin.** Landed in commit 0d90b30fb6c (no backticks).\n")
        path.write_text(incidents, encoding="utf-8")
        run("add", "TESTING_INCIDENTS.md")
        run("commit", "-q", "-m", "add incidents")
        errors = checker.check_incident_origin_citations(str(path))
        self.assertTrue(
            any("does not exist" in err and "0d90b30fb6c" in err
                for err in errors),
            msg=errors)

    @unittest.skipUnless(_GIT, "git not available")
    def test_origin_citation_unquoted_marker_is_skipped(self):
        # A bare ``external:``/``unverified:`` marker immediately before an
        # unquoted hash preserves it, just as for a backtick citation.
        run = self._init_git_repo()
        path = self.root / "TESTING_INCIDENTS.md"
        incidents = VALID_INCIDENTS.replace(
            "**Origin.** Landed in `0d90b30fb6c`.\n",
            "**Origin.** Landed in unverified: 0d90b30fb6c elsewhere.\n")
        path.write_text(incidents, encoding="utf-8")
        run("add", "TESTING_INCIDENTS.md")
        run("commit", "-q", "-m", "add incidents")
        errors = checker.check_incident_origin_citations(str(path))
        self.assertEqual(errors, [], msg=errors)

    @unittest.skipUnless(_GIT, "git not available")
    def test_origin_unquoted_hex_word_is_not_a_hash(self):
        # A bare all-letter hex word ("defaced") in Origin prose has no digit and
        # is not mistaken for a commit hash.
        run = self._init_git_repo()
        path = self.root / "TESTING_INCIDENTS.md"
        incidents = VALID_INCIDENTS.replace(
            "**Origin.** Landed in `0d90b30fb6c`.\n",
            "**Origin.** The guard was defaced then restored by maintainers.\n")
        path.write_text(incidents, encoding="utf-8")
        run("add", "TESTING_INCIDENTS.md")
        run("commit", "-q", "-m", "add incidents")
        errors = checker.check_incident_origin_citations(str(path))
        self.assertEqual(errors, [], msg=errors)

    @unittest.skipUnless(_GIT, "git not available")
    def test_origin_citation_descriptive_backtick_bogus_hash_is_reported(self):
        # A fabricated hash hidden inside a *descriptive* backticked span such as
        # ``commit deadbee1`` must not evade validation: the span is scanned for
        # hash occurrences rather than treated as an opaque non-hash token.
        run = self._init_git_repo()
        path = self.root / "TESTING_INCIDENTS.md"
        incidents = VALID_INCIDENTS.replace(
            "**Origin.** Landed in `0d90b30fb6c`.\n",
            "**Origin.** Landed in `commit deadbee1` upstream.\n")
        path.write_text(incidents, encoding="utf-8")
        run("add", "TESTING_INCIDENTS.md")
        run("commit", "-q", "-m", "add incidents")
        errors = checker.check_incident_origin_citations(str(path))
        self.assertTrue(
            any("does not exist" in err and "deadbee1" in err
                for err in errors),
            msg=errors)

    @unittest.skipUnless(_GIT, "git not available")
    def test_origin_citation_descriptive_backtick_real_hash_passes(self):
        # A real hash inside a descriptive backticked span resolves and is
        # accepted - scanning the span must not false-alarm on a landed commit.
        run = self._init_git_repo()
        path = self.root / "TESTING_INCIDENTS.md"
        path.write_text("placeholder\n", encoding="utf-8")
        run("add", "TESTING_INCIDENTS.md")
        run("commit", "-q", "-m", "seed")
        head = run("rev-parse", "HEAD").stdout.strip()[:12]
        incidents = VALID_INCIDENTS.replace(
            "**Origin.** Landed in `0d90b30fb6c`.\n",
            "**Origin.** Landed in `commit {}` upstream.\n".format(head))
        path.write_text(incidents, encoding="utf-8")
        errors = checker.check_incident_origin_citations(str(path))
        self.assertEqual(errors, [], msg=errors)

    @unittest.skipUnless(_GIT, "git not available")
    def test_origin_citation_descriptive_backtick_marker_is_skipped(self):
        # An ``external:``/``unverified:`` marker that opens a descriptive span
        # preserves every hash the span carries, so an honest out-of-repository
        # reference wrapped in prose is not existence-checked.
        run = self._init_git_repo()
        path = self.root / "TESTING_INCIDENTS.md"
        incidents = VALID_INCIDENTS.replace(
            "**Origin.** Landed in `0d90b30fb6c`.\n",
            "**Origin.** Landed in `external: commit deadbee1` elsewhere.\n")
        path.write_text(incidents, encoding="utf-8")
        run("add", "TESTING_INCIDENTS.md")
        run("commit", "-q", "-m", "add incidents")
        errors = checker.check_incident_origin_citations(str(path))
        self.assertEqual(errors, [], msg=errors)

    # -- fence/comment masking of incident headings, fields, Origin hashes ----

    def test_fenced_incident_heading_is_not_defined(self):
        # A ``## TI-NNN:`` heading inside a fenced code block is template/example
        # text, not a real incident definition, so it must not register an id.
        incidents = VALID_INCIDENTS + (
            "\n## Appendix\n\n"
            "An example of the incident template (not a real incident):\n\n"
            "```markdown\n"
            "## TI-999: Example only\n"
            "```\n"
        )
        path = self.root / "TESTING_INCIDENTS.md"
        path.write_text(incidents, encoding="utf-8")
        errors = []
        defined = checker.defined_incident_ids(str(path), errors)
        self.assertIn("TI-001", defined, msg=defined)
        self.assertNotIn("TI-999", defined, msg=defined)
        self.assertEqual(errors, [], msg=errors)

    def test_tilde_fenced_incident_heading_is_not_defined(self):
        # The tilde fence form (``~~~``) is masked identically to the backtick form.
        incidents = VALID_INCIDENTS + (
            "\n## Appendix\n\n"
            "An example of the incident template (not a real incident):\n\n"
            "~~~markdown\n"
            "## TI-999: Example only\n"
            "~~~\n"
        )
        path = self.root / "TESTING_INCIDENTS.md"
        path.write_text(incidents, encoding="utf-8")
        errors = []
        defined = checker.defined_incident_ids(str(path), errors)
        self.assertIn("TI-001", defined, msg=defined)
        self.assertNotIn("TI-999", defined, msg=defined)
        self.assertEqual(errors, [], msg=errors)

    def test_commented_incident_heading_is_not_defined(self):
        # A ``## TI-NNN:`` heading hidden inside an HTML comment is likewise not a
        # real incident definition.
        incidents = VALID_INCIDENTS + (
            "\n## Appendix\n\n"
            "<!--\n"
            "## TI-999: Example only\n"
            "-->\n"
        )
        path = self.root / "TESTING_INCIDENTS.md"
        path.write_text(incidents, encoding="utf-8")
        errors = []
        defined = checker.defined_incident_ids(str(path), errors)
        self.assertIn("TI-001", defined, msg=defined)
        self.assertNotIn("TI-999", defined, msg=defined)
        self.assertEqual(errors, [], msg=errors)

    def test_fenced_required_fields_do_not_satisfy_incident(self):
        # A real ``## TI-001:`` heading whose required labelled fields appear only
        # inside a fenced code block carries no real fields: the fenced labels are
        # example text and must not satisfy the requirement.
        incidents = (
            "# Incidents\n\n"
            "## TI-001: Example control\n\n"
            "```\n"
            "**Finding.** A control looked removable but was not.\n"
            "**Control.** Guard the path.\n"
            "**Protects.** The trusted signal.\n"
            "**Origin.** Landed in `0d90b30fb6c`.\n"
            "**Current status.** Active.\n"
            "**Retirement condition.** Only when the hazard cannot recur.\n"
            "```\n"
        )
        path = self.root / "TESTING_INCIDENTS.md"
        path.write_text(incidents, encoding="utf-8")
        errors = checker.check_incident_fields(str(path))
        self.assertTrue(
            any("missing the 'Finding' field" in err for err in errors),
            msg=errors)

    def test_tilde_fenced_required_fields_do_not_satisfy_incident(self):
        incidents = (
            "# Incidents\n\n"
            "## TI-001: Example control\n\n"
            "~~~\n"
            "**Finding.** A control looked removable but was not.\n"
            "**Control.** Guard the path.\n"
            "**Protects.** The trusted signal.\n"
            "**Origin.** Landed in `0d90b30fb6c`.\n"
            "**Current status.** Active.\n"
            "**Retirement condition.** Only when the hazard cannot recur.\n"
            "~~~\n"
        )
        path = self.root / "TESTING_INCIDENTS.md"
        path.write_text(incidents, encoding="utf-8")
        errors = checker.check_incident_fields(str(path))
        self.assertTrue(
            any("missing the 'Finding' field" in err for err in errors),
            msg=errors)

    def test_commented_required_fields_do_not_satisfy_incident(self):
        incidents = (
            "# Incidents\n\n"
            "## TI-001: Example control\n\n"
            "<!--\n"
            "**Finding.** A control looked removable but was not.\n"
            "**Control.** Guard the path.\n"
            "**Protects.** The trusted signal.\n"
            "**Origin.** Landed in `0d90b30fb6c`.\n"
            "**Current status.** Active.\n"
            "**Retirement condition.** Only when the hazard cannot recur.\n"
            "-->\n"
        )
        path = self.root / "TESTING_INCIDENTS.md"
        path.write_text(incidents, encoding="utf-8")
        errors = checker.check_incident_fields(str(path))
        self.assertTrue(
            any("missing the 'Finding' field" in err for err in errors),
            msg=errors)

    @unittest.skipUnless(_GIT, "git not available")
    def test_origin_citation_fenced_fabricated_hash_is_ignored(self):
        # A fabricated hash that lives inside a fenced code block within the Origin
        # field span is example text, not a citation, so it must not be validated.
        run = self._init_git_repo()
        path = self.root / "TESTING_INCIDENTS.md"
        incidents = VALID_INCIDENTS.replace(
            "**Origin.** Landed in `0d90b30fb6c`.\n",
            "**Origin.** See the fenced example of a stale citation:\n"
            "```\n"
            "reverted in commit deadbee1\n"
            "```\n")
        path.write_text(incidents, encoding="utf-8")
        run("add", "TESTING_INCIDENTS.md")
        run("commit", "-q", "-m", "add incidents")
        errors = checker.check_incident_origin_citations(str(path))
        self.assertEqual(errors, [], msg=errors)

    @unittest.skipUnless(_GIT, "git not available")
    def test_origin_citation_tilde_fenced_fabricated_hash_is_ignored(self):
        run = self._init_git_repo()
        path = self.root / "TESTING_INCIDENTS.md"
        incidents = VALID_INCIDENTS.replace(
            "**Origin.** Landed in `0d90b30fb6c`.\n",
            "**Origin.** See the fenced example of a stale citation:\n"
            "~~~\n"
            "reverted in commit deadbee1\n"
            "~~~\n")
        path.write_text(incidents, encoding="utf-8")
        run("add", "TESTING_INCIDENTS.md")
        run("commit", "-q", "-m", "add incidents")
        errors = checker.check_incident_origin_citations(str(path))
        self.assertEqual(errors, [], msg=errors)

    @unittest.skipUnless(_GIT, "git not available")
    def test_origin_citation_commented_fabricated_hash_is_ignored(self):
        # A fabricated hash hidden inside an HTML comment within the Origin field
        # is likewise not a live citation.
        run = self._init_git_repo()
        path = self.root / "TESTING_INCIDENTS.md"
        incidents = VALID_INCIDENTS.replace(
            "**Origin.** Landed in `0d90b30fb6c`.\n",
            "**Origin.** Historic note <!-- reverted in commit deadbee1 --> "
            "see upstream.\n")
        path.write_text(incidents, encoding="utf-8")
        run("add", "TESTING_INCIDENTS.md")
        run("commit", "-q", "-m", "add incidents")
        errors = checker.check_incident_origin_citations(str(path))
        self.assertEqual(errors, [], msg=errors)

    @unittest.skipUnless(_GIT, "git not available")
    def test_origin_citation_inline_hash_preserved_despite_fence(self):
        # Masking is fence/comment-only: a real backticked Origin hash OUTSIDE any
        # fence must survive and still be validated even when the section also
        # carries a fenced example hash. The inline `0d90b30fb6c` is bogus and must
        # still be reported, while the fenced deadbee1 is ignored.
        run = self._init_git_repo()
        path = self.root / "TESTING_INCIDENTS.md"
        incidents = VALID_INCIDENTS.replace(
            "**Retirement condition.** Only when the hazard cannot recur.\n",
            "**Retirement condition.** Only when the hazard cannot recur.\n\n"
            "```\n"
            "ignored in commit deadbee1\n"
            "```\n")
        path.write_text(incidents, encoding="utf-8")
        run("add", "TESTING_INCIDENTS.md")
        run("commit", "-q", "-m", "add incidents")
        errors = checker.check_incident_origin_citations(str(path))
        self.assertTrue(
            any("0d90b30fb6c" in err for err in errors), msg=errors)
        self.assertFalse(
            any("deadbee1" in err for err in errors), msg=errors)

    # -- final fix 5: bare external: markers in path-bearing fields -----------

    def test_bare_external_source_of_truth_is_reported(self):
        broken = VALID_INVENTORY.replace(
            '  suite_placement: "external: catch/*/CMakeLists.txt"\n',
            '  suite_placement: "external:"\n', 1)
        _write_docs(self.root, {"test_inventory.yaml": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("source_of_truth 'suite_placement'" in err
                and "external marker must name a target" in err
                for err in results["inventory"]),
            msg=results["inventory"],
        )

    def test_bare_external_suite_evidence_path_is_reported(self):
        broken = VALID_INVENTORY.replace(
            '      - path: "external: evidence-outside-repo"\n',
            '      - path: "external:"\n', 1)
        _write_docs(self.root, {"test_inventory.yaml": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("evidence path external marker must name a target" in err
                for err in results["inventory"]),
            msg=results["inventory"],
        )

    def test_bare_external_class_evidence_path_is_reported(self):
        broken = VALID_COVERAGE.replace(
            '        - path: "external: evidence-outside-repo"\n',
            '        - path: "external:"\n', 1)
        _write_docs(self.root, {"test_coverage_model.yaml": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("evidence path external marker must name a target" in err
                for err in results["coverage_model"]),
            msg=results["coverage_model"],
        )

    def test_bare_external_ci_lane_in_repo_is_reported(self):
        broken = VALID_INVENTORY.replace(
            "      ci_lane:\n        external: documented elsewhere\n",
            '      ci_lane:\n        in_repo: "external:"\n', 1)
        _write_docs(self.root, {"test_inventory.yaml": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("ci_lane.in_repo" in err
                and "external marker must name a target" in err
                for err in results["inventory"]),
            msg=results["inventory"],
        )

    def test_bare_external_metadata_config_is_reported(self):
        broken = VALID_INVENTORY.replace(
            "    metadata:\n      framework: Catch2\n",
            '    metadata:\n      framework: Catch2\n      config: "external:"\n',
            1)
        _write_docs(self.root, {"test_inventory.yaml": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("metadata.config" in err
                and "external marker must name a target" in err
                for err in results["inventory"]),
            msg=results["inventory"],
        )

    def test_named_external_in_path_fields_still_passes(self):
        # Regression guard: a NAMED external target in these fields is still
        # accepted (only the bare marker is rejected). The valid baseline already
        # uses named external: markers for source_of_truth and evidence.
        _write_docs(self.root)
        results, _ = checker.validate(root=str(self.root))
        self.assertEqual(results["inventory"], [], msg=results["inventory"])


# Cadence words and a strategy-section template for the cadence-prose check. The
# ``## Testing strategy and layers`` section is the per-tier region the check
# bounds to; the trailing ``## Nightly validation`` section is intentionally
# present so a test can prove that legitimate external-validation prose outside
# the strategy section is never scanned.
_STRATEGY_DOC = (
    "# Testing strategy\n"
    "\n"
    "## Testing strategy and layers\n"
    "\n"
    "{body}\n"
    "\n"
    "## Nightly validation\n"
    "\n"
    "Nightly and multi-arch runs extend the per-PR flow; release validation "
    "adds broader coverage.\n"
)


def _inventory_with_cadences(*cadences):
    """A minimal parsed inventory whose suites claim the given cadences."""
    return {"suites": [{"execution": {"cadence": c}} for c in cadences]}


def _coverage_with_cadence_axis(informational, classes=None):
    """A minimal parsed coverage model carrying one cadence axis.

    ``informational`` is the cadence axis's informational_classes list (or None
    to omit it); ``classes`` defaults to the full declared cadence vocabulary.
    """
    axis = {"id": "cadence",
            "classes": classes or ["pr", "nightly", "release", "on_demand"]}
    if informational is not None:
        axis["informational_classes"] = informational
    return {"coverage_model": {"axes": [axis]}}


class CheckCadenceProseTest(unittest.TestCase):
    """Unit tests for the per-tier cadence-prose consistency check."""

    def test_flags_unaudited_cadence_word_in_strategy_section(self):
        # An informational cadence word (nightly) asserted as a per-tier cadence
        # in the strategy section, with no suite claiming it, is reported.
        text = _STRATEGY_DOC.format(
            body="Integration runs as a PR/nightly subset depending on platform.")
        errs = checker.check_cadence_prose(
            text,
            _inventory_with_cadences("pr", "on_demand"),
            _coverage_with_cadence_axis(["nightly", "release"]))
        self.assertTrue(
            any("nightly" in err and "unaudited per-tier cadence" in err
                for err in errs),
            msg=errs)

    def test_clean_strategy_section_passes(self):
        # Describing the tier with cadences the inventory backs is accepted.
        text = _STRATEGY_DOC.format(
            body="Integration runs as a PR or on-demand subset, platform-gated.")
        errs = checker.check_cadence_prose(
            text,
            _inventory_with_cadences("pr", "on_demand"),
            _coverage_with_cadence_axis(["nightly", "release"]))
        self.assertEqual(errs, [], msg=errs)

    def test_nightly_validation_section_is_out_of_scope(self):
        # The guarded words appear only in the separate ``## Nightly validation``
        # section (part of the template), never in the strategy body; the check
        # bounds its scan to the strategy section, so this is clean.
        text = _STRATEGY_DOC.format(
            body="Each tier runs on PR or on-demand cadence.")
        errs = checker.check_cadence_prose(
            text,
            _inventory_with_cadences("pr", "on_demand"),
            _coverage_with_cadence_axis(["nightly", "release"]))
        self.assertEqual(errs, [], msg=errs)

    def test_noop_without_informational_cadence_classes(self):
        # With no informational cadence classes declared the guarded set is empty
        # and the check is a no-op even if the word appears in the strategy body.
        text = _STRATEGY_DOC.format(
            body="Integration runs as a PR/nightly subset depending on platform.")
        errs = checker.check_cadence_prose(
            text,
            _inventory_with_cadences("pr", "on_demand"),
            _coverage_with_cadence_axis(None))
        self.assertEqual(errs, [], msg=errs)

    def test_cadence_actually_claimed_by_a_suite_is_not_guarded(self):
        # A cadence that is listed as informational but that a suite actually
        # claims is data-driven out of the guarded set, so it is not flagged.
        text = _STRATEGY_DOC.format(
            body="Integration runs on nightly cadence.")
        errs = checker.check_cadence_prose(
            text,
            _inventory_with_cadences("pr", "nightly"),
            _coverage_with_cadence_axis(["nightly", "release"]))
        self.assertEqual(errs, [], msg=errs)

    def test_cadence_word_inside_code_span_is_not_flagged(self):
        # A cadence word shown in an inline code span is masked before scanning.
        text = _STRATEGY_DOC.format(
            body="The `nightly` ctest label is unrelated to build cadence.")
        errs = checker.check_cadence_prose(
            text,
            _inventory_with_cadences("pr", "on_demand"),
            _coverage_with_cadence_axis(["nightly", "release"]))
        self.assertEqual(errs, [], msg=errs)

    def test_noop_without_strategy_section(self):
        # No ``## Testing strategy and layers`` section means nothing to scan.
        text = "# Doc\n\n## Other\n\nIntegration runs on nightly cadence.\n"
        errs = checker.check_cadence_prose(
            text,
            _inventory_with_cadences("pr", "on_demand"),
            _coverage_with_cadence_axis(["nightly", "release"]))
        self.assertEqual(errs, [], msg=errs)

    def test_live_testing_md_cadence_prose_matches_inventory(self):
        # Live regression guard: the real TESTING.md per-tier strategy prose must
        # not assert a cadence the real inventory does not back. This reads the
        # committed component documents directly through the checker's loaders.
        import yaml  # available transitively via the checker module

        component = checker.HIP_TESTS_DIR
        inventory = yaml.safe_load(
            (component / "test_inventory.yaml").read_text(encoding="utf-8"))
        coverage = yaml.safe_load(
            (component / "test_coverage_model.yaml").read_text(encoding="utf-8"))
        text = (component / "TESTING.md").read_text(encoding="utf-8")
        errs = checker.check_cadence_prose(text, inventory, coverage)
        self.assertEqual(errs, [], msg=errs)


class LiveConfigGovernanceInventoryTest(unittest.TestCase):
    """Live guards: the real inventory records the catch/config governance checks.

    The build-time governance checks wired by catch/config/CMakeLists.txt -
    check_config.py (CMake target check_hip_tests_config) and check_sources.py
    (CMake target check_hip_tests_sources) - must each carry a stable inventory
    record so the two active governance checks are inventoried, not invisible.
    These tests read the committed component documents and the CMakeLists
    directly, so the record, its CMake target wiring, and its cadence/gating
    semantics are checked against reality rather than a fixture.
    """

    @classmethod
    def setUpClass(cls):
        import yaml  # available transitively via the checker module

        cls.component = checker.HIP_TESTS_DIR
        cls.inventory = yaml.safe_load(
            (cls.component / "test_inventory.yaml").read_text(encoding="utf-8"))
        cls.config_cmake = (
            cls.component / "catch" / "config" / "CMakeLists.txt"
        ).read_text(encoding="utf-8")
        cls.catch_cmake = (
            cls.component / "catch" / "CMakeLists.txt"
        ).read_text(encoding="utf-8")

    def _record_by_build_target(self, target):
        for suite in self.inventory.get("suites") or []:
            if not isinstance(suite, dict):
                continue
            source = suite.get("source")
            if isinstance(source, dict) and source.get("build_target") == target:
                return suite
        return None

    def test_config_check_record_present_with_target_and_script(self):
        rec = self._record_by_build_target("check_hip_tests_config")
        self.assertIsNotNone(
            rec, "no inventory record wires the check_hip_tests_config target")
        source = rec["source"]
        self.assertEqual(source.get("build_option"), "ENABLE_YAML_TAGS")
        self.assertIn("catch/config/check_config.py", source.get("paths") or [])

    def test_sources_check_record_present_with_target_and_script(self):
        rec = self._record_by_build_target("check_hip_tests_sources")
        self.assertIsNotNone(
            rec, "no inventory record wires the check_hip_tests_sources target")
        source = rec["source"]
        self.assertEqual(source.get("build_option"), "ENABLE_YAML_TAGS")
        self.assertIn("catch/config/check_sources.py", source.get("paths") or [])

    def test_config_check_gating_is_strict(self):
        # check_config.py exits non-zero on any violation, so its record must
        # declare the hard-gating (trusted) semantics.
        rec = self._record_by_build_target("check_hip_tests_config")
        self.assertIsNotNone(rec)
        self.assertEqual(rec["execution"].get("gating"), "trusted")

    def test_config_governance_checks_run_at_build_cadence(self):
        # Both custom targets run as part of every enabled hip-tests build, not
        # as PR workflow checks. Their records must therefore use the declared
        # build cadence.
        for target in ("check_hip_tests_config", "check_hip_tests_sources"):
            rec = self._record_by_build_target(target)
            self.assertIsNotNone(rec)
            self.assertEqual(rec["execution"].get("cadence"), "build")

    def test_sources_check_gating_is_warning_by_default(self):
        # check_sources.py only warns by default (escalating to a hard failure
        # under THEROCK_REQUIRE_HIP_YAML_ENTRIES), so its record must declare the
        # informational (warning-by-default) semantics, not a hard gate.
        rec = self._record_by_build_target("check_hip_tests_sources")
        self.assertIsNotNone(rec)
        self.assertEqual(rec["execution"].get("gating"), "informational")

    def test_sources_check_documents_strict_env(self):
        # The strict-env escalation is the governing semantic difference between
        # the two checks; the record must name THEROCK_REQUIRE_HIP_YAML_ENTRIES
        # somewhere in its prose so a reader learns how to make it blocking.
        rec = self._record_by_build_target("check_hip_tests_sources")
        self.assertIsNotNone(rec)
        import yaml
        blob = yaml.safe_dump(rec)
        self.assertIn("THEROCK_REQUIRE_HIP_YAML_ENTRIES", blob)

    def test_cmake_registers_both_targets_under_enable_yaml_tags(self):
        # The records claim ENABLE_YAML_TAGS-gated ALL custom targets; confirm the
        # CMakeLists actually registers both targets and the option defaults ON.
        for target in ("check_hip_tests_config", "check_hip_tests_sources"):
            self.assertRegex(
                self.config_cmake,
                r"add_custom_target\(\s*" + target + r"\s+ALL",
                "catch/config/CMakeLists.txt does not register {} as an ALL "
                "custom target".format(target))
        self.assertIn("if(ENABLE_YAML_TAGS)", self.config_cmake)
        self.assertRegex(
            self.catch_cmake,
            r'option\(ENABLE_YAML_TAGS\s+"[^"]*"\s+ON\)',
            "catch/CMakeLists.txt does not default ENABLE_YAML_TAGS ON")

    def test_both_records_pass_the_inventory_checker(self):
        # The new records must be schema-valid: the full inventory still validates
        # cleanly through the checker (no schema/path/vocabulary regressions).
        errors, _ids, _refs, _counts = checker.check_inventory(
            self.inventory, self.component)
        self.assertEqual(errors, [], msg=errors)


if __name__ == "__main__":
    unittest.main()
