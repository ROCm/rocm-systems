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
  suite_placement: "external: catch/*/CMakeLists.txt"
  case_metadata: "external: catch/config/configs/*.yaml"
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
        broken = VALID_INVENTORY.replace(
            "      ci_lane:\n        external: documented elsewhere\n",
            "      ci_lane:\n"
            "        in_repo: ../../.github/workflows/does-not-exist.yml\n", 1)
        _write_docs(self.root, {"test_inventory.yaml": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertTrue(
            any("execution.ci_lane.in_repo path missing" in err
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
        # evidence that does exist (no invented ticket) is accepted.
        broken = VALID_INVENTORY.replace(
            "    history: Guarded per TI-001.\n",
            "    history: Historical origin not located in available change "
            "history; repository evidence (source paths and CMake wiring) "
            "documents the behavior it currently pins.\n", 1)
        _write_docs(self.root, {"test_inventory.yaml": broken})
        results, _ = checker.validate(root=str(self.root))
        self.assertEqual(_all_errors(results), [], msg=results)


if __name__ == "__main__":
    unittest.main()
