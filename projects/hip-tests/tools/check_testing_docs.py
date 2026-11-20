#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT
"""Deterministic consistency checker for the HIP testing documentation set.

Validates the hand-maintained testing documents under ``projects/hip-tests``
as a connected set:

  * ``test_inventory.yaml``        - the suite inventory
  * ``test_coverage_model.yaml``   - the intended-coverage model
  * ``TESTING.md``                 - the current-state testing strategy
  * ``TESTING_INCIDENTS.md``       - the test-control rationale / incident log
  * ``TEST_RESTRUCTURING.md``      - the restructuring roadmap (proposal)

The checks implement the acceptance criteria that TEST_RESTRUCTURING.md states
for itself (workstreams WS-1 and WS-2):

  schema keys / vocabularies
    - the inventory carries its top-level identity/provenance/vocabulary blocks
      (``schema_version``, ``component``, ``source_of_truth``, ``vocabularies``,
      ``suites``) with the required mapping/list shapes; its ``schema_version``
      is the supported version (the integer 1) and its ``component`` is
      ``hip-tests`` - a well-formed but unsupported version or a non-matching
      component is a schema error in its own right;
    - every inventory suite, coverage class, and pairwise check carries its
      required keys; suites and coverage classes additionally carry an explicit
      ``owner`` (pairwise checks inherit ownership from the classes they
      reference and are not separately owner-stamped);
    - every inventory ``id`` is unique;
    - each required vocabulary (``kind`` / ``disposition`` / ``gating`` /
      ``cadence``) is a non-empty list of non-empty scalar entries (an empty list
      is rejected because it would vacuously pass every membership check);
    - every suite ``kind`` / ``disposition`` and every ``execution.gating`` /
      ``execution.cadence`` is a non-empty scalar string that is within the
      vocabulary the inventory declares for itself (the ``vocabularies:`` block);
      a non-scalar token is a schema error rather than an unhashable
      set-membership crash;
    - every suite carries a non-empty ``evidence`` list whose entries are mappings
      with a non-empty path and claim (paths resolve to a file on disk - not a
      directory - unless marked ``external:``), matching the coverage-class
      evidence requirement;
    - every suite documents its historical context: it carries either a non-empty
      ``incidents`` list or a ``history`` narrative that actually establishes
      context, so a reader can trace a non-obvious control back to its origin or
      its governing incident. A bare/generic placeholder (for example a stock
      "origin not located; repository evidence records current purpose." line)
      does not count on its own: a sufficient narrative must carry an explicit
      evidence/origin marker (such as ``repository evidence``) AND not be a
      contentless sentinel. A generic "unknown origin" narrative is held to a
      higher bar: a lightly expanded placeholder (the stock "origin not located"
      line with a few extra words) still names no evidence specific to the
      record, so it must additionally cite record-specific context - a concrete
      repository path or a governing incident - or the suite must name governing
      incidents instead. This keeps an honest unknown-origin record anchored to
      the repository evidence that does exist (no invented tickets) while
      rejecting a reused template that passes on the strength of a marker word.
      An ``incidents`` list, when present, is a non-empty list of ``TI-NNN`` ids,
      each of which must resolve to a defined incident (as any TI reference does).
    - every required suite field is present with a usable value: the descriptive
      scalars (``name`` / ``domain`` / ``purpose`` / ``oracle`` /
      ``disposition_rationale``) are non-empty strings, ``execution.cadence`` is a
      non-empty string, and ``execution.platforms`` / ``backends`` /
      ``prerequisites`` are non-empty lists of non-empty scalar strings
      (``ctest_labels`` may be an empty list for a build-free static check, so it
      is only required to be a list, but any label it does carry must likewise be
      a non-empty scalar string);
      ``execution.ci_lane`` is a mapping that must name a lane - it carries a
      non-empty ``in_repo`` path or a non-empty ``external`` string (a ci_lane
      with neither points at no CI coverage and is a schema error).
    - the YAML is parsed with a strict loader that rejects any mapping carrying a
      duplicate key, so a repeated key (for example a copy/pasted suite ``id`` or
      ``evidence`` key) is a parse error rather than a silently dropped value.

  move / retire targets
    - a suite whose ``disposition`` is ``merge_into`` or ``retire`` must name a
      ``disposition_target`` (the destination record, or the replacement
      inventory id(s) that preserve retired coverage); ``migrate`` / ``demote``
      may omit it. A ``disposition_target``, when present, must be a non-empty
      string or a non-empty list of non-empty inventory ids; any target a move
      disposition names must resolve to existing inventory id(s); no move may name
      a missing or self target. A ``retire`` suite must additionally carry a
      ``retirement`` mapping that documents equivalent-or-stronger replacement
      coverage, the CI evidence that the replacement runs, and owner approval.

  coverage refs
    - the coverage model carries its top-level identity block (``schema_version``,
      ``component``) with the same shape and supported-identity requirement the
      inventory has (version 1, component ``hip-tests``), and the two documents
      must agree with each other on both fields (a coverage model may not be
      paired with an inventory of a different schema version or component); every
      inventory / axis / class id is a non-empty string before it participates
      in duplicate or set-membership checks;
    - every ``current_inventory`` id in the coverage model resolves to an
      inventory suite id;
    - every coverage class declares a non-empty ``axes`` list and a non-empty
      ``current_inventory`` list, and every pairwise check's ``pair`` names exactly
      two ``axis.class`` references; every pairwise check must likewise name a
      non-empty ``current_inventory`` or explicitly acknowledge an uncovered
      interaction via ``gap`` metadata (so an empty coverage claim is a documented
      gap rather than a silent hole);
    - every coverage class carries its descriptive fields with a usable value:
      ``intent`` / ``gap_status`` are non-empty strings, ``expected_test_kinds`` is
      a non-empty list, and ``risk`` is within the documented high/medium/low
      vocabulary; every pairwise check likewise carries non-empty ``category`` /
      ``status`` / ``rationale`` / ``gap_status`` strings and a ``priority`` within
      the same high/medium/low vocabulary;
    - every class / pairwise axis reference is a declared ``axis.class``;
    - every declared axis ``id`` is unique, and every class within a single axis
      is declared once (a duplicate axis id shadows the earlier declaration, and
      a class repeated within an axis collapses to one ``axis.class``); every
      coverage ``class`` id is likewise unique;
    - the model declares the required coverage dimensions (architecture,
      capability, integration_boundary) as explicit axes;
    - the ``pairwise_checks`` block is a non-empty cross-axis audit that exercises
      all four required categories (``api_x_behavior``, ``api_x_environment``,
      ``integration_x_execution_context``, ``capability_x_cadence``);
    - every declared ``axis.class`` is meaningfully audited - referenced by at
      least one coverage class or pairwise check - unless the axis explicitly
      marks it under ``informational_classes`` (a declared-but-unreferenced,
      non-informational class is a dead vocabulary entry and is rejected);
    - every coverage class carries a non-empty ``evidence`` list whose entries
      are mappings with a non-empty path and claim (paths resolve to a file on
      disk - not a directory - unless marked ``external:``).

  evidence paths
    - every ``source.paths`` entry resolves to a path on disk (a source directory
      or a file) relative to the component directory, and every ``evidence.path``
      resolves specifically to a file (not a directory), unless explicitly marked
      ``external:``; an empty or whitespace-only ``source.paths`` entry is
      rejected rather than resolved to the component directory, and a bare
      ``external:`` with no target is rejected in every path-bearing field
      (``source.paths``, ``evidence.path``, ``source_of_truth`` values,
      ``metadata.config`` / ``metadata.generated_plan``, and
      ``execution.ci_lane.in_repo``): the marker must name the external reference
      it stands in for.

  repository path pointers
    - every repository-path-bearing field resolves relative to the component
      directory unless marked ``external:`` (or preserved as a URL): the
      top-level ``source_of_truth`` values (literal paths or globs), each
      suite's ``metadata.config`` and ``metadata.generated_plan``, and
      ``execution.ci_lane.in_repo`` (whose trailing job/step qualifier is
      descriptive prose and is not path-checked).

  CMake suite-group reconciliation
    - for each Catch2 area (unit / contract / performance / stress) whose
      per-area ``CMakeLists.txt`` is present, every actively built group
      (``add_subdirectory(<group>)`` - matched case-insensitively and across a
      multiline command, with line comments stripped first) must be represented
      by an inventory suite whose ``source.paths`` names ``catch/<area>/<group>``
      (or a path beneath it); a built group that no suite covers is reported,
      unless it is explicitly documented under the inventory's optional
      ``cmake_suite_group_exclusions`` block with a categorized reason
      (not_built / commented / no_cases / bare / external) and a note. The
      reconciliation is feasibility-gated: an area with no CMakeLists.txt on disk
      is skipped, so the check is a no-op for a fixture that ships no CMake tree.

  incident IDs and fields
    - every ``TI-NNN`` referenced in the YAML resolves to an incident defined by
      a ``## TI-NNN:`` heading in TESTING_INCIDENTS.md;
    - every incident id is defined exactly once: a ``TI-NNN`` carried by two
      ``## TI-NNN:`` headings is a schema error (the id is meant to be stable and
      unique);
    - every defined incident carries the required labelled fields (Finding,
      Control, Protects, Origin, Current status, Retirement condition);
    - every commit-like citation in an incident Origin field (an abbreviated or
      full git SHA, whether in backticks or bare in prose) resolves to a commit
      in the repository; a hash that git reports does not exist is reported
      unless the author explicitly marks the citation ``external:`` or
      ``unverified:``. This
      check is feasibility-gated: when the document is not inside a git work tree
      (or git is unavailable) hash existence cannot be verified and the check is
      skipped.

  local Markdown refs and anchors
    - every relative Markdown link ``[text](path)`` in TESTING.md,
      TESTING_INCIDENTS.md, and TEST_RESTRUCTURING.md resolves to a file on disk
      (not a directory) within the boundary root (http(s)/mailto URLs are
      preserved as external and not existence-checked); a local link target that
      is absolute, that resolves outside the supplied ``--root`` / default
      repository boundary, or that resolves to a directory rather than a file is
      rejected exactly like any other repository path pointer, while valid
      relative links and fragment anchors continue to resolve; link-like text
      inside inline code spans, fenced code blocks, and HTML comments is shown
      rather than linked and is masked out before scanning, so a ``[text](path)``
      in a code example or an ``<!-- commented-out link -->`` is neither validated
      nor reported, and a backslash-escaped opener ``\\[text](path)`` is treated
      as literal text rather than a link;
    - reference-style links ``[text][label]`` / ``[label][]`` are resolved through
      their ``[label]: target`` definitions and validated like inline links; a
      usage with no matching definition, or a definition that is never used, is
      reported; a label defined more than once (case-insensitively) is reported
      and the first definition is preserved as authoritative rather than silently
      overwritten;
    - every in-page ``#anchor`` and ``file.md#anchor`` fragment resolves to a
      heading in the target Markdown document; heading anchors are derived from
      the same fence-masked and comment-masked Markdown, so a ``# Heading`` that
      lives inside a fenced code block or an HTML comment does not satisfy a real
      anchor;
    - every ``related_restructuring`` pointer in the inventory, when present, is a
      non-empty string that resolves on disk within the boundary root (an empty or
      non-string value, an absolute path, or a traversal outside the boundary root
      is a schema error).

  cadence prose
    - TESTING.md's ``## Testing strategy and layers`` section may describe each
      tier only with a cadence the inventory actually backs. A cadence word the
      coverage model keeps as declared vocabulary but that no inventory suite runs
      on - the cadence axis's ``informational_classes`` (for example
      nightly/release) minus any cadence an inventory suite actually claims - may
      not appear in that section as a concrete per-tier cadence, because doing so
      asserts an audited cadence the inventory does not support. The guarded word
      set is derived from the documents, so the check follows the data rather than
      a hardcoded list; the separate ``## Nightly validation`` / ``## Release
      validation`` sections (which describe broader external validation, not a
      per-tier build cadence) are out of scope and never scanned. The check is
      feasibility-gated: it is a no-op when the documents name no guarded cadence
      word or the strategy section is absent.

This is a pure static consistency check: it needs only Python 3 and PyYAML, no
ROCm, GPU, or build, so it is safe as a fast gate. Paths are taken as arguments
(``--root`` or the per-document overrides) or resolved repo-relative from this
script's location; nothing in the documents is executed.

Exit status: non-zero if any check fails, zero otherwise.
"""

import argparse
import glob
import json
import re
import subprocess
import sys
from pathlib import Path

import yaml

# Resolve component-relative paths from this script's location so the checker
# works from any working directory. Layout:
#   <repo>/projects/hip-tests/tools/check_testing_docs.py
_THIS_DIR = Path(__file__).resolve().parent               # tools
HIP_TESTS_DIR = _THIS_DIR.parent                          # projects/hip-tests

# File names of the documents under the component/repository root.
INVENTORY_NAME = "test_inventory.yaml"
COVERAGE_NAME = "test_coverage_model.yaml"
TESTING_NAME = "TESTING.md"
INCIDENTS_NAME = "TESTING_INCIDENTS.md"
RESTRUCTURING_NAME = "TEST_RESTRUCTURING.md"

# Required top-level inventory keys. The inventory is a connected schema whose
# identity (schema_version, component), provenance (source_of_truth), and
# self-declared vocabularies must be present with the right shape before any
# per-suite check is meaningful; a missing or malformed block is a schema error
# in its own right (TEST_RESTRUCTURING.md WS-1).
REQUIRED_INVENTORY_TOP_KEYS = frozenset({
    "schema_version", "component", "source_of_truth", "vocabularies", "suites",
})

# Required top-level coverage-model keys. The coverage model carries the same
# identity block as the inventory (schema_version, component) plus its own
# coverage_model body; a missing or malformed identity block is a schema error
# in its own right before any per-class check is meaningful (TEST_RESTRUCTURING.md
# WS-2).
REQUIRED_COVERAGE_TOP_KEYS = frozenset({
    "schema_version", "component", "coverage_model",
})

# Supported identity of the document set. The inventory and the coverage model
# are two halves of one connected schema; each must declare the single schema
# version this checker understands (the integer 1) and the component it belongs
# to (hip-tests). A well-formed but unsupported schema_version or a non-matching
# component is a schema error in its own right (the document set was authored
# against, or copied from, a different contract), and the two documents must
# additionally agree with each other on both fields (TEST_RESTRUCTURING.md
# WS-1/WS-2).
SUPPORTED_SCHEMA_VERSION = 1
SUPPORTED_COMPONENT = "hip-tests"

# Vocabulary categories the inventory must declare for itself, each as a list of
# allowed values.
REQUIRED_VOCABULARIES = ("kind", "disposition", "gating", "cadence")

# Required keys per record type. Derived from the documents as authored; a
# missing key is a schema error.
REQUIRED_SUITE_KEYS = frozenset({
    "id", "name", "kind", "domain", "purpose", "oracle", "owner", "source",
    "metadata", "execution", "evidence", "disposition", "disposition_rationale",
})
REQUIRED_EXECUTION_KEYS = frozenset({
    "cadence", "gating", "ctest_labels", "platforms", "backends",
    "prerequisites", "ci_lane",
})
REQUIRED_CLASS_KEYS = frozenset({
    "id", "owner", "axes", "risk", "intent", "expected_test_kinds",
    "current_inventory", "evidence", "gap_status",
})
REQUIRED_PAIRWISE_KEYS = frozenset({
    "id", "category", "pair", "status", "priority", "rationale",
    "current_inventory", "gap_status",
})

# Coverage dimensions the model must declare explicitly as axes so that
# architecture/capability and integration-boundary coverage can be audited.
REQUIRED_AXES = frozenset({"architecture", "capability", "integration_boundary"})

# Pairwise-check categories the coverage model must populate. The pairwise block
# is the model's cross-axis audit; it must be non-empty and must exercise all
# four review dimensions so the audit is not silently missing a whole class of
# interactions:
#   * api_x_behavior                   - an API surface against a behavior mode
#   * api_x_environment                - an API surface against an environment
#   * integration_x_execution_context  - graph/module/RTC (and multi-GPU) feature
#                                        integration against an execution context
#   * capability_x_cadence             - a capability-gated path against a cadence
# A missing category (or an empty pairwise block) is a schema error: the model
# would claim an audit it does not actually perform (TEST_RESTRUCTURING.md WS-2).
REQUIRED_PAIRWISE_CATEGORIES = frozenset({
    "api_x_behavior", "api_x_environment",
    "integration_x_execution_context", "capability_x_cadence",
})

# The severity/priority vocabulary the coverage model documents for itself (a
# coverage class ``risk`` and a pairwise check ``priority`` are review-priority
# levels). A value outside this set is a schema error so the audit priorities
# stay comparable across records.
COVERAGE_RISK_VALUES = frozenset({"high", "medium", "low"})

# The labelled fields every incident account must carry so that a reviewer can
# see what a control protects and when it may be retired.
REQUIRED_INCIDENT_LABELS = (
    "Finding", "Control", "Protects", "Origin",
    "Current status", "Retirement condition",
)

# Dispositions that may name a relocation target; when a target is present it
# must resolve to an existing inventory id and never to the record itself.
MOVE_DISPOSITIONS = frozenset({"merge_into", "migrate", "demote", "retire"})

# Dispositions that MUST name a disposition_target. ``merge_into`` asserts a
# concrete destination record that has to exist today; ``retire`` drops coverage
# and must name the replacement inventory id(s) that preserve it. ``migrate`` /
# ``demote`` describe a lifecycle change whose destination may be external or not
# yet an inventory id, so a target is optional for them (but still validated when
# one is given).
REQUIRE_TARGET_DISPOSITIONS = frozenset({"merge_into", "retire"})

# The disposition that drops a suite from the live set. Retiring coverage is only
# safe when the dropped coverage is demonstrably preserved elsewhere, so a
# retiring suite must carry a ``retirement`` mapping documenting three things, on
# top of the replacement inventory id(s) it names via disposition_target: that the
# replacement coverage is equivalent or stronger, the CI evidence that the
# replacement actually runs, and the owner approval for the retire.
RETIRE_DISPOSITION = "retire"
REQUIRED_RETIREMENT_EVIDENCE = (
    "coverage_equivalence", "ci_evidence", "owner_approval",
)

# Historical-context enforcement. A suite documents its provenance either by
# naming the governing incident(s) (an ``incidents`` list) or by carrying a
# ``history`` narrative. A narrative is only accepted when it does more than
# restate a stock placeholder: it must carry an explicit evidence/origin marker
# AND not be a bare, contentless sentinel. This keeps the provenance chain real -
# a generic "origin not located" line alone no longer satisfies the requirement -
# while still permitting an honest "origin unknown" record that points at the
# repository evidence that does exist (no invented tickets).
#
# Markers that signal a narrative actually cites evidence or an origin: an
# explicit repository-evidence phrase, a governing-incident reference, or the
# control vocabulary the incident log uses to describe why a guard exists.
HISTORY_EVIDENCE_MARKERS = (
    "repository evidence", "origin", "incident", "testing_incidents", "ti-",
    "enforced", "guarded", "gated", "hazard", "drift",
)
# A generic "unknown origin" narrative. When a history opens by declaring that
# the origin could not be located, the marker words it carries ("origin",
# "repository evidence") are generic and a lightly reworded placeholder would
# otherwise pass on their strength alone. Such a narrative is therefore held to a
# higher bar (see ``_history_is_sufficient``): it must additionally cite
# record-specific context. The pattern deliberately requires ``origin``
# immediately followed by the not-located phrasing so that an incidental use of
# the word ``origin`` elsewhere in a real narrative is not treated as generic.
_UNKNOWN_ORIGIN_RE = re.compile(
    r"origin\s+(?:not\s+located|not\s+found|unknown|unidentified)",
    re.IGNORECASE)
# A concrete repository path token (a slash-joined path that either descends at
# least two levels or ends in a file extension). Its presence in an otherwise
# generic unknown-origin narrative is the record-specific evidence that keeps the
# provenance pointed at something real rather than a reused template. A single
# incidental slash with no further segment or extension (``and/or``) is
# intentionally not enough.
_RECORD_SPECIFIC_PATH_RE = re.compile(
    r"[\w.\-]+/[\w.\-]+(?:/[\w./\-]+|\.[A-Za-z0-9]+)")
# Bare / generic history sentinels that, standing alone, do NOT establish
# historical context. These are normalized (lower-cased, whitespace-collapsed)
# before comparison. A suite whose entire history is one of these must instead
# name governing incidents or carry real context; the stock "origin not located;
# repository evidence records current purpose." line is explicitly here so it can
# no longer pass on the strength of the marker words it happens to contain.
GENERIC_HISTORY_SENTINELS = frozenset({
    "historical origin not located; repository evidence records current purpose.",
    "historical origin not located.",
    "repository evidence records current purpose.",
    "origin not located.",
    "unknown.",
    "none.",
    "n/a.",
    "no history.",
    "tbd.",
})

# A path value starting with this marker is a CI lane defined outside the repo
# and is not existence-checked (TEST_RESTRUCTURING.md WS-1).
EXTERNAL_MARKER = "external:"

# Catch2 test areas whose per-area CMakeLists.txt enumerates the active test
# groups (one ``add_subdirectory(<group>)`` per built group). The checker
# reconciles these active groups against the inventory so a group that is built
# but carries no inventory suite is surfaced rather than silently uncovered.
CMAKE_RECONCILIATION_AREAS = ("unit", "contract", "performance", "stress")

# The CMakeLists that registers the host-side harness tests (built through
# ``hip_add_host_test`` and labelled ``hosttest``). These executables are not
# under a per-area ``add_subdirectory`` group, so they are reconciled separately:
# when this file actively registers host tests, the inventory must carry a
# host/harness suite whose ``source.paths`` covers the registering directory.
HOSTTEST_CMAKE_REL = ("catch", "hipTestMain", "tests", "CMakeLists.txt")
HOSTTEST_GROUP_DIR = "catch/hipTestMain/tests"

# The CMake commands this checker discovers, keyed by lower-cased name. CMake
# commands are case-insensitive, so a scanned command name is compared in lower
# case (``ADD_SUBDIRECTORY`` is the same command as ``add_subdirectory``).
_CMAKE_ADD_SUBDIRECTORY = "add_subdirectory"
_CMAKE_HIP_ADD_HOST_TEST = "hip_add_host_test"

# The start of a CMake command invocation: an identifier (the command name)
# followed by its opening parenthesis, with optional whitespace - including
# newlines - between them. Capturing the whole identifier is what makes the scan
# command-aware: ``foo_add_subdirectory(`` yields the command name
# ``foo_add_subdirectory`` (not ``add_subdirectory``), and a bare token such as
# ``add_subdirectory`` that appears only inside another command's quoted argument
# is never a command start because the scanner skips over each command's argument
# span (see ``_iter_cmake_commands``) rather than re-scanning it. A declaration
# header ``function(hip_add_host_test TARGET)`` is likewise the ``function``
# command whose first *argument* is ``hip_add_host_test``, so it is not mistaken
# for a host-test registration.
_CMAKE_COMMAND_START_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(")

# The first argument token of a CMake command's argument list: optional leading
# whitespace (which may span newlines so a call split across physical lines is
# handled), an optional opening double quote (CMake treats a quoted argument the
# same as an unquoted one, and the closing quote is simply not part of the
# token), then the token itself. Used to pull the group directory out of an
# ``add_subdirectory`` call and the target out of a ``hip_add_host_test`` call.
_CMAKE_FIRST_ARG_RE = re.compile(r'\s*"?([A-Za-z0-9_.+\-]+)')

# A CMake bracket comment ``#[[ ... ]]`` (optionally with equals signs in the
# delimiter, ``#[=[ ... ]=]``) is a block comment that may span several physical
# lines. Line-comment stripping alone would miss it - an ``add_subdirectory``
# sitting inside a bracket comment is NOT an active group - so bracket comments
# are masked out (content replaced by spaces, newlines preserved) before line
# comments are stripped. The ``(?s)`` flag lets ``.`` cross newlines; the ``\1``
# back-reference pairs the closing ``]=*]`` to the opening ``[=*[``.
_CMAKE_BRACKET_COMMENT_RE = re.compile(r"#\[(=*)\[.*?\]\1\]", re.DOTALL)

# The reasons a built CMake suite group may legitimately carry no inventory
# suite. A documented exclusion must name one of these so the allowance is an
# explicit, categorized decision rather than a silent hole:
#   * not_built - the group is guarded off by default / builds nothing here
#   * commented - the add_subdirectory is commented out (kept for reference)
#   * no_cases  - the group directory holds only config/bare scaffolding, no
#                 runnable contract/unit/perf/stress cases to inventory
#   * bare      - a placeholder group with no test sources yet
#   * external  - the group's coverage is owned by an external record
CMAKE_EXCLUSION_REASONS = frozenset({
    "not_built", "commented", "no_cases", "bare", "external",
})

# A commit-like citation in an incident Origin field: a 7-40 character hex string
# (an abbreviated or full git SHA). The ``_is_commit_hash`` helper additionally
# requires at least one digit so an all-letter English word that happens to be
# hex ("defaced") is not mistaken for a hash.
_COMMIT_HASH_RE = re.compile(r"^[0-9a-f]{7,40}$")

# Inline-code markers that exempt a cited commit from existence validation: a
# citation the author explicitly flags as living outside this repository or as
# unverified is preserved as-is (no invented ticket, no false alarm) rather than
# checked against the local object database.
ORIGIN_CITATION_MARKERS = ("external:", "unverified:")

# A repository-path-bearing field value may be a glob pattern (for example the
# source_of_truth suite_placement/case_metadata/contract_assertions patterns)
# rather than a single literal path; these metacharacters mark such a value so it
# is resolved by glob match instead of literal existence.
_GLOB_CHARS = ("*", "?", "[")

_INCIDENT_HEADING_RE = re.compile(r"^##\s+(TI-\d+):", re.MULTILINE)
_INCIDENT_REF_RE = re.compile(r"\bTI-\d+\b")
# A single incident id, used to validate the explicit ``incidents:`` list a
# suite may carry in place of a free-text ``history`` narrative. The whole
# token must be a ``TI-NNN`` id, not merely contain one.
_INCIDENT_ID_FULL_RE = re.compile(r"^TI-\d+$")
# Inline Markdown link opener ``[text](``. The destination/title body that
# follows is not captured here: a naive ``([^)]+)`` capture stops at the first
# ``)`` and so truncates a valid link whose destination carries a balanced pair
# of parentheses (``[t](a/(v2)/b.md)``) or whose title is parenthesized
# (``[t](dest (title))``). The body is extracted by _iter_inline_link_targets,
# which scans for the matching close paren instead. A backslash-escaped opener
# ``\[`` is a literal bracket, not a link (CommonMark shows ``\[text](path)`` as
# text). Escape detection is deliberately NOT a lookbehind here: a fixed-width
# ``(?<!\\)`` only rejects an opener preceded by a single backslash and so
# misclassifies an even-length backslash run (``\\[`` - two backslashes that are
# themselves an escaped-backslash literal) as escaped when the opener is in fact
# live. _iter_inline_link_targets instead measures the preceding backslash run
# and treats the opener as escaped only when that run is odd (see
# ``_opener_is_escaped``).
_MD_LINK_OPEN_RE = re.compile(r"\[[^\]]*\]\(")
# Reference-style Markdown links: a usage ``[text][label]`` (or the collapsed
# form ``[label][]``) paired with a ``[label]: target`` definition. The checker
# resolves these through their definitions and validates them exactly like inline
# links, so a reference-style link is a supported, checked form rather than one
# that is silently ignored. Escape parity is applied by the scanner the same way
# it is for inline openers: a usage whose opening ``[`` carries an odd-length
# preceding backslash run is an escaped (literal) bracket and is ignored, while
# an even run (including none) leaves the opener live (see ``_opener_is_escaped``).
_MD_REF_USE_RE = re.compile(r"\[([^\]]*)\]\[([^\]]*)\]")
_MD_REF_DEF_RE = re.compile(r"^[ ]{0,3}\[([^\]]+)\]:[ \t]*(\S+)", re.MULTILINE)
_HEADING_RE = re.compile(r"^#{1,6}\s+(.*?)\s*$", re.MULTILINE)
# A Markdown link/image opener inside a heading: an optional image bang ``!``,
# the ``[visible]`` label (or image alt text), and the ``(`` that begins the
# destination. GitHub derives a heading's anchor slug from the heading's
# *rendered* text, so only the visible label/alt text contributes - the
# destination and any optional title do not. ``_heading_visible_text`` uses this
# to keep the label and drop the following balanced ``(...)`` destination before
# the slug is formed; the label capture ``[^\]]*`` stops at the first ``]`` and
# the destination is consumed by a balanced-parenthesis scan (not a naive
# ``[^)]*``) so a destination carrying a balanced pair of parentheses or a
# parenthesized title is fully excluded.
_MD_INLINE_LABEL_RE = re.compile(r"!?\[([^\]]*)\]\(")
# A fenced code-block delimiter line: three or more backticks or tildes, possibly
# indented and carrying an info string. Link scanning masks everything between a
# matching opening and closing fence so a ``[text](path)`` inside a code block is
# shown, not validated.
_CODE_FENCE_RE = re.compile(r"^(\s*)(`{3,}|~{3,})(.*)$")
# An HTML comment ``<!-- ... -->`` (which may span multiple lines). Link scanning
# and heading extraction mask these out so a ``[text](path)`` or a ``# Heading``
# hidden inside a comment is neither linked nor registered as a real anchor.
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
# URLs preserved as external references; existence is not checked for these.
_EXTERNAL_LINK_PREFIXES = ("http://", "https://", "mailto:")

# Cadence-prose consistency (TESTING.md per-tier strategy section vs inventory).
# The ``## Testing strategy and layers`` section describes each tier's build
# status and *cadence*. A cadence word the coverage model keeps as declared
# vocabulary but that no inventory suite actually runs on - the cadence axis's
# ``informational_classes`` (for example nightly/release) minus any cadence an
# inventory suite actually claims - must not be presented there as a concrete
# per-tier cadence, because that asserts an audited cadence the inventory does not
# back. The guarded word set is derived from the documents (not hardcoded) so it
# follows the data: the day a suite genuinely declares a nightly cadence, that
# word leaves the guarded set automatically. The separate ``## Nightly
# validation`` / ``## Release validation`` sections are deliberately out of scope
# (they describe broader external validation flows, not a per-tier build cadence)
# and are never scanned.
CADENCE_STRATEGY_SECTION_TITLE = "Testing strategy and layers"
CADENCE_AXIS_ID = "cadence"


class _UniqueKeyLoader(yaml.SafeLoader):
    """A SafeLoader that rejects a mapping with a duplicate key.

    ``yaml.safe_load`` silently keeps the last value when a mapping repeats a
    key, so a copy/paste that duplicates (for example) a suite's ``id`` or an
    ``evidence`` key would load without complaint and quietly drop the earlier
    value. For a consistency checker that is itself a schema error: the document
    no longer says what it appears to say. This loader walks each mapping node's
    raw key list and raises a ``ConstructorError`` (a ``yaml.YAMLError`` subclass,
    so ``_load_yaml`` reports it as a parse failure) on the first repeat.
    """

    def compose_node(self, parent, index):
        # Reject a cyclic anchor/alias while still allowing an ordinary
        # (non-cyclic) shared alias. ``yaml.safe_load`` composes a recursive node
        # graph for a self-referential anchor (``a: &A {b: *A}``) and constructs a
        # Python object that references itself; a later recursive walk of the
        # document (for example ``_collect_incident_refs``) then cannot terminate.
        # For a consistency checker that is a schema error: a testing document is
        # a finite tree, never a cycle. An anchor is tracked from the moment its
        # node begins composing until that node is fully composed, so an alias
        # that resolves to an anchor still in progress (an ancestor of the current
        # node) is a cyclic reference and is reported as a parse failure, whereas
        # a non-cyclic alias - one that names an anchor already fully composed -
        # is left untouched and resolves normally.
        if not hasattr(self, "_composing_anchors"):
            self._composing_anchors = set()
        if self.check_event(yaml.events.AliasEvent):
            anchor = self.peek_event().anchor
            if anchor in self._composing_anchors:
                raise yaml.composer.ComposerError(
                    None, None,
                    "found cyclic anchor {!r}".format(anchor),
                    self.peek_event().start_mark)
            return super().compose_node(parent, index)
        anchor = self.peek_event().anchor
        if anchor is not None:
            self._composing_anchors.add(anchor)
        node = super().compose_node(parent, index)
        if anchor is not None:
            self._composing_anchors.discard(anchor)
        return node

    def construct_mapping(self, node, deep=False):
        seen = set()
        for key_node, _value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            # A non-scalar (list/mapping) key is unhashable, so the duplicate
            # test below would raise a bare TypeError rather than a reportable
            # parse error. YAML does allow complex keys, but the testing-doc
            # schema never uses one, and an accidental one (for example a
            # mis-indented list that collapses into a key) is a schema error we
            # want surfaced as a parse failure, not a crash.
            try:
                duplicate = key in seen
            except TypeError:
                raise yaml.constructor.ConstructorError(
                    "while constructing a mapping", node.start_mark,
                    "found unhashable key {!r}".format(key), key_node.start_mark)
            if duplicate:
                raise yaml.constructor.ConstructorError(
                    "while constructing a mapping", node.start_mark,
                    "found duplicate key {!r}".format(key), key_node.start_mark)
            seen.add(key)
        return super().construct_mapping(node, deep)


def _load_yaml(path, errors):
    # Read and decode first so a file that is not valid UTF-8 produces a
    # deterministic load error rather than an uncaught UnicodeDecodeError: the
    # testing documents are authored as UTF-8, and a stray non-UTF-8 byte (a
    # bad copy/paste or a mis-encoded save) is a load failure, not a crash. The
    # UnicodeDecodeError message is itself deterministic (it names the offending
    # byte and position), so the reported error is stable across runs.
    try:
        text = Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        errors.append("file not found: {}".format(path))
        return None
    except UnicodeDecodeError as exc:
        errors.append("{} is not valid UTF-8: {}".format(path, exc))
        return None
    try:
        return yaml.load(text, Loader=_UniqueKeyLoader)
    except yaml.YAMLError as exc:
        errors.append("{} does not parse as YAML: {}".format(path, exc))
    return None


def _repository_root(component_dir):
    """Return the repository root for the projects/hip-tests component."""
    return Path(component_dir).resolve().parents[1]


def _boundary_root(component_dir, repo_root):
    """Return the directory a repository path may not resolve outside of.

    When the caller supplies an explicit ``repo_root`` (the ``--root`` it named),
    that root is itself the boundary: an arbitrary temporary root is self-
    contained, so a path may not traverse above it. When no root is supplied the
    boundary defaults to the repository root two levels above the component
    directory, which is what lets the real documents reference sibling trees such
    as ``../../.github`` from ``projects/hip-tests``.
    """
    if repo_root is not None:
        return Path(repo_root).resolve()
    return _repository_root(component_dir)


def _derive_boundary_root(root, component_dir, inventory_path):
    """Return the boundary a repository path may not resolve outside of.

    This separates the *document root* a caller names from the *repository
    boundary* a path may not escape, so an explicit ``--root`` that points at a
    real checkout is usable rather than over-constrained:

      * No explicit root (the default real-repo run): the boundary is the
        repository root two levels above the component directory, which permits
        the documents' legitimate ``../../.github`` references.
      * An explicit root that is a real checkout / component root: it lives
        inside a git work tree that tracks its documents, so the git repository
        root is the boundary - a real ``--root projects/hip-tests`` then
        validates documents that reference sibling trees like ``../../.github``
        instead of rejecting them as escapes.
      * An explicit root that is a self-contained temporary fixture: it is
        untracked (even if it happens to sit under some ambient checkout), so it
        remains its own boundary and a path may not traverse above it.

    Anchoring the real-checkout case on tracked-ness (the same signal the
    Origin-citation check uses) is what keeps a throwaway fixture bounded while
    letting a genuine checkout reach its sibling trees.
    """
    if root is None:
        return _repository_root(component_dir)
    root_resolved = Path(root).resolve()
    git_root = _git_repo_root(root_resolved)
    if git_root is not None and _git_tracks(
            str(git_root), Path(inventory_path).resolve()):
        return git_root
    return root_resolved


def _path_location_status(component_dir, rel_path, repo_root=None):
    """Classify a repository path as usable, absolute, or outside its boundary.

    The boundary is the supplied ``repo_root`` when given and otherwise the
    repository root two levels above the component directory (see
    ``_boundary_root``). An absolute path is rejected outright; a relative path
    that resolves outside the boundary is ``outside``.
    """
    path = Path(rel_path)
    if path.is_absolute():
        return "absolute"
    resolved = (Path(component_dir) / path).resolve()
    try:
        resolved.relative_to(_boundary_root(component_dir, repo_root))
    except ValueError:
        return "outside"
    return "ok"


def _path_exists(component_dir, rel_path, repo_root=None):
    """True when rel_path resolves to an existing path within the boundary root."""
    return (_path_location_status(component_dir, rel_path, repo_root) == "ok"
            and (Path(component_dir) / rel_path).resolve().exists())


def _path_is_file(component_dir, rel_path, repo_root=None):
    """True when rel_path resolves to an existing file within the boundary root."""
    return (_path_location_status(component_dir, rel_path, repo_root) == "ok"
            and (Path(component_dir) / rel_path).resolve().is_file())


def _looks_external(value):
    """True when value is an explicit external: marker or a preserved URL."""
    return value.startswith(EXTERNAL_MARKER) or value.startswith(_EXTERNAL_LINK_PREFIXES)


def _path_token(value):
    """Return the leading whitespace-delimited path token of value.

    Repository-path-bearing fields may carry a human-readable qualifier after the
    path itself: a CI lane names the workflow file and then, in parentheses, the
    job and step, and a source_of_truth value may append prose such as
    ``and external CI references in TESTING.md``. Only the leading token is the
    path; the remainder is descriptive and is not existence-checked.
    """
    stripped = value.strip()
    return stripped.split()[0] if stripped else ""


def _glob_location_status(component_dir, pattern, repo_root=None):
    """Classify a glob pattern by whether its matches stay within the boundary.

    Returns ``ok`` when the pattern (component-relative, may use ** and ..)
    matches at least one path and every match resolves - following symlinks -
    within the boundary root; ``outside`` when any match escapes the boundary (for
    example a matched symlink whose target points out of the repository), so a
    pattern that mixes in-boundary and escaping matches is rejected rather than
    passing on the strength of a single safe match; and ``missing`` when the
    pattern matches nothing. The boundary is the supplied ``repo_root`` when given
    and otherwise the repository root two levels above the component directory
    (see ``_boundary_root``). Resolving each match (not just the literal pattern
    token) is what makes the check symlink-safe: the literal-location guard in
    ``_path_location_status`` only rejects a ``..`` escape written into the
    pattern, never a symlink that escapes the boundary at match time.
    """
    boundary = _boundary_root(component_dir, repo_root)
    try:
        matches = list(
            glob.iglob(pattern, root_dir=str(component_dir), recursive=True))
    except (ValueError, OSError):
        return "missing"
    if not matches:
        return "missing"
    for rel in matches:
        resolved = (Path(component_dir) / rel).resolve()
        try:
            resolved.relative_to(boundary)
        except ValueError:
            return "outside"
    return "ok"


def _resolve_repo_path(component_dir, value, repo_root=None):
    """Classify a repository-path-bearing field value.

    Returns one of ``external`` (a named external: marker or URL, preserved and
    not checked), ``bare_external`` (a bare ``external:`` marker that names no
    target), ``empty`` (no path token), ``ok`` (the leading path token resolves
    on disk, by literal existence or glob match), or ``missing``. A token carrying
    a glob metacharacter is resolved by recursive glob; otherwise it is
    existence-checked relative to the component directory. The boundary the token
    may not escape is the supplied ``repo_root`` when given (see
    ``_path_location_status``).
    """
    stripped = value.strip()
    if _looks_external(stripped):
        # A bare ``external:`` marker names nothing; like every other
        # path-bearing field it must carry the external reference it stands in
        # for. A URL or a named ``external: <target>`` is preserved as external.
        if (stripped.startswith(EXTERNAL_MARKER)
                and not stripped[len(EXTERNAL_MARKER):].strip()):
            return "bare_external"
        return "external"
    token = _path_token(stripped)
    if not token:
        return "empty"
    location = _path_location_status(component_dir, token, repo_root)
    if location != "ok":
        return location
    if any(ch in token for ch in _GLOB_CHARS):
        # A glob is resolved by match, and each match must itself stay within the
        # boundary (a matched symlink may escape it even when the literal pattern
        # does not); an escaping match makes the whole pattern ``outside``.
        return _glob_location_status(component_dir, token, repo_root)
    return "ok" if _path_exists(component_dir, token, repo_root) else "missing"


def _check_repo_path_field(component_dir, value, label, errors, repo_root=None):
    """Validate one repository-path-bearing field value.

    Appends a deterministic error when the value is not a string, is empty, or
    names a path (or glob) that does not resolve; external: markers and URLs are
    preserved. Returns 1 when the value resolved to a real path (for the
    validated-reference count), 0 otherwise.
    """
    if not isinstance(value, str):
        errors.append("{} must be a string path: {!r}".format(label, value))
        return 0
    status = _resolve_repo_path(component_dir, value, repo_root)
    if status == "empty":
        errors.append("{} is empty".format(label))
        return 0
    if status == "bare_external":
        errors.append(
            "{} external marker must name a target: {}".format(
                label, value.strip()))
        return 0
    if status == "absolute":
        errors.append("{} must be repository-relative: {}".format(label, value.strip()))
        return 0
    if status == "outside":
        errors.append("{} escapes repository root: {}".format(label, value.strip()))
        return 0
    if status == "missing":
        errors.append("{} path missing: {}".format(label, value.strip()))
        return 0
    # 'external' is preserved and not counted; 'ok' counts as a resolved ref.
    return 1 if status == "ok" else 0


def _nonempty_str(value):
    return isinstance(value, str) and bool(value.strip())


def _normalize_history(value):
    """Lower-case and whitespace-collapse a history string for sentinel matching."""
    return re.sub(r"\s+", " ", value.strip().lower())


def _history_cites_record_specific_context(value):
    """True when a history narrative points at evidence specific to the record.

    Record-specific context is either a governing-incident reference (a
    ``TI-NNN`` id) or a concrete repository path token (see
    ``_RECORD_SPECIFIC_PATH_RE``). This is what distinguishes an honest
    unknown-origin record - which names the real source/wiring that documents the
    behavior it pins - from a reused generic template that names nothing.
    """
    if _INCIDENT_REF_RE.search(value):
        return True
    return bool(_RECORD_SPECIFIC_PATH_RE.search(value))


def _history_is_sufficient(value):
    """True when a history narrative establishes real historical context.

    A sufficient narrative is a non-empty string that carries an explicit
    evidence/origin marker (``repository evidence``, a governing-incident
    reference, or the control vocabulary the incident log uses) AND is not a
    bare/generic sentinel (for example a stock "origin not located" line). The
    marker requirement rejects a context-free note that names no evidence; the
    sentinel check rejects a stock placeholder that happens to contain a marker
    word but no actual context.

    A generic "unknown origin" narrative is held to a higher bar: its marker
    words ("origin", "repository evidence") are generic, so a lightly expanded
    placeholder would otherwise pass on their strength alone. Such a narrative is
    accepted only when it additionally cites record-specific context - a concrete
    repository path or a governing incident - so the provenance points at
    something real rather than a reused template.
    """
    if not _nonempty_str(value):
        return False
    if _normalize_history(value) in GENERIC_HISTORY_SENTINELS:
        return False
    low = value.lower()
    if not any(marker in low for marker in HISTORY_EVIDENCE_MARKERS):
        return False
    if (_UNKNOWN_ORIGIN_RE.search(value)
            and not _history_cites_record_specific_context(value)):
        return False
    return True


def _is_scalar(value):
    """True for a present, non-container scalar value.

    A number is accepted as-is; a string must be non-empty once stripped. Null
    and mapping/list containers are rejected so that schema_version cannot be
    silently omitted or given a structured value.
    """
    if value is None or isinstance(value, (dict, list)):
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return True


def _is_supported_schema_version(value):
    """True only for the exact integer schema version this checker supports.

    ``schema_version`` must equal the integer 1. ``bool`` is a subclass of
    ``int`` in Python (so ``True == 1``) and a float ``1.0`` would compare equal
    to 1 as well; both are rejected here so that the stored version is literally
    the integer 1 and nothing that merely coerces to it.
    """
    return (isinstance(value, int) and not isinstance(value, bool)
            and value == SUPPORTED_SCHEMA_VERSION)


def _vocab_values(vocab, name):
    """Return the declared vocabulary for name as a set of its scalar entries.

    Only hashable scalar entries participate; a malformed (non-scalar) entry is
    reported separately by the vocabulary-shape check and is skipped here so that
    set construction cannot raise on an unhashable value.
    """
    value = vocab.get(name)
    if not isinstance(value, list):
        return set()
    return {item for item in value if isinstance(item, (str, int, float, bool))}


def check_inventory(inventory, component_dir, repo_root=None):
    """Validate the suite inventory.

    Returns (errors, suite_ids, incident_refs, counts) where counts is a dict
    with 'records' and 'refs' (references successfully resolved). ``repo_root``,
    when given, is the boundary a repository path may not resolve outside of (see
    ``_path_location_status``); it defaults to the repository root two levels
    above the component directory.
    """
    errors = []
    suite_ids = set()
    incident_refs = set()
    refs_ok = 0

    if not isinstance(inventory, dict):
        return (["inventory did not parse into a mapping"], suite_ids,
                incident_refs, {"records": 0, "refs": 0})

    # Top-level schema: identity, provenance, and vocabulary blocks must be
    # present with the right shape before any per-suite check is meaningful.
    missing_top = REQUIRED_INVENTORY_TOP_KEYS - set(inventory.keys())
    if missing_top:
        errors.append(
            "inventory missing top-level keys: {}".format(sorted(missing_top)))

    if "schema_version" in inventory:
        schema_version = inventory.get("schema_version")
        if not _is_scalar(schema_version):
            errors.append("inventory schema_version must be a non-empty scalar")
        elif not _is_supported_schema_version(schema_version):
            errors.append(
                "inventory schema_version must be the supported version {} "
                "(got {!r})".format(SUPPORTED_SCHEMA_VERSION, schema_version))

    if "component" in inventory:
        component = inventory.get("component")
        if not _nonempty_str(component):
            errors.append("inventory component must be a non-empty string")
        elif component != SUPPORTED_COMPONENT:
            errors.append(
                "inventory component must be {!r} (got {!r})".format(
                    SUPPORTED_COMPONENT, component))

    if "source_of_truth" in inventory:
        sot = inventory.get("source_of_truth")
        if not isinstance(sot, dict) or not sot:
            errors.append("inventory source_of_truth must be a non-empty mapping")
        else:
            # Every source_of_truth value is a repository-path-bearing pointer (a
            # literal path or a glob) and must resolve unless marked external:.
            for key in sorted(sot):
                refs_ok += _check_repo_path_field(
                    component_dir, sot.get(key),
                    "source_of_truth {!r}".format(key), errors, repo_root)

    vocab = inventory.get("vocabularies")
    if "vocabularies" in inventory and not isinstance(vocab, dict):
        errors.append("inventory vocabularies must be a mapping")
    if not isinstance(vocab, dict):
        vocab = {}
    for name in REQUIRED_VOCABULARIES:
        if name not in vocab:
            errors.append("vocabularies block is missing {!r}".format(name))
        elif not isinstance(vocab.get(name), list) or not vocab.get(name):
            # A declared vocabulary must be a non-empty list: an empty list would
            # silently disable the membership check for that category (every value
            # would vacuously pass), so it is itself a schema error.
            errors.append(
                "vocabularies {!r} must be a non-empty list".format(name))
        else:
            # Each declared value must be a plain scalar; a nested list/mapping is
            # a schema error and would otherwise be an unhashable vocabulary entry.
            for entry in vocab.get(name):
                if not _is_scalar(entry):
                    errors.append(
                        "vocabularies {!r} entry must be a non-empty scalar: {!r}".format(
                            name, entry))
    kind_vocab = _vocab_values(vocab, "kind")
    disp_vocab = _vocab_values(vocab, "disposition")
    gating_vocab = _vocab_values(vocab, "gating")
    cadence_vocab = _vocab_values(vocab, "cadence")

    suites = inventory.get("suites")
    if not isinstance(suites, list) or not suites:
        errors.append("inventory has no suites")
        return errors, suite_ids, incident_refs, {"records": 0, "refs": 0}

    # First pass: collect ids so move-target checks can resolve forward refs. An
    # id must be a non-empty string before it can meaningfully participate in
    # duplicate detection or forward-reference resolution; a malformed id is a
    # schema error in its own right, so it is reported and not added to the id
    # set.
    for suite in suites:
        sid = suite.get("id") if isinstance(suite, dict) else None
        if sid is None:
            continue
        if not _nonempty_str(sid):
            errors.append("suite id must be a non-empty string: {!r}".format(sid))
            continue
        if sid in suite_ids:
            errors.append("duplicate suite id: {!r}".format(sid))
        suite_ids.add(sid)

    for suite in suites:
        if not isinstance(suite, dict):
            errors.append("suite record is not a mapping: {!r}".format(suite))
            continue
        sid = suite.get("id", "<no-id>")

        missing = REQUIRED_SUITE_KEYS - set(suite.keys())
        if missing:
            errors.append("suite {!r} missing keys: {}".format(sid, sorted(missing)))

        if "owner" in suite and not _nonempty_str(suite.get("owner")):
            errors.append("suite {!r} owner must be a non-empty string".format(sid))

        # The remaining required descriptive fields are free-text scalars that
        # carry the record's meaning; a present-but-empty or non-string value is a
        # schema error (the missing-key check above handles outright absence).
        for field in ("name", "domain", "purpose", "oracle", "disposition_rationale"):
            if field in suite and not _nonempty_str(suite.get(field)):
                errors.append(
                    "suite {!r} {} must be a non-empty string".format(sid, field))

        # ``kind`` and ``disposition`` are vocabulary tokens: each must be a
        # non-empty scalar string before it can be compared against the declared
        # vocabulary set. A list/mapping value is both a schema error and an
        # unhashable set-membership crash, so it is guarded here and the vocabulary
        # check is skipped for it.
        kind = suite.get("kind")
        if kind is not None and not _nonempty_str(kind):
            errors.append("suite {!r} kind must be a non-empty string: {!r}".format(sid, kind))
        elif _nonempty_str(kind) and kind_vocab and kind not in kind_vocab:
            errors.append("suite {!r} kind {!r} not in declared vocabulary".format(sid, kind))

        disp = suite.get("disposition")
        if disp is not None and not _nonempty_str(disp):
            errors.append("suite {!r} disposition must be a non-empty string: {!r}".format(sid, disp))
        elif _nonempty_str(disp) and disp_vocab and disp not in disp_vocab:
            errors.append("suite {!r} disposition {!r} not in declared vocabulary".format(sid, disp))

        execution = suite.get("execution")
        if isinstance(execution, dict):
            exec_missing = REQUIRED_EXECUTION_KEYS - set(execution.keys())
            if exec_missing:
                errors.append("suite {!r} execution missing keys: {}".format(sid, sorted(exec_missing)))
            # ``cadence`` is a vocabulary token and must be a non-empty scalar
            # string before its membership can be checked.
            cadence = execution.get("cadence")
            if cadence is not None and not _nonempty_str(cadence):
                errors.append(
                    "suite {!r} execution.cadence must be a non-empty string: {!r}".format(
                        sid, cadence))
            elif _nonempty_str(cadence) and cadence_vocab and cadence not in cadence_vocab:
                errors.append(
                    "suite {!r} execution.cadence {!r} not in declared vocabulary".format(
                        sid, cadence))
            # platforms / backends / prerequisites enumerate where and how the
            # suite runs; each must be a non-empty list whose entries are each a
            # non-empty scalar string (a null, blank, or structured entry carries
            # no usable value and is a schema error). ctest_labels may legitimately
            # be empty (a build-free static check registers no CTest labels), so it
            # is only required to be a list, but any label it does carry must
            # likewise be a non-empty scalar string.
            for field in ("platforms", "backends", "prerequisites"):
                if field in execution:
                    value = execution.get(field)
                    if not isinstance(value, list) or not value:
                        errors.append(
                            "suite {!r} execution.{} must be a non-empty list".format(
                                sid, field))
                    else:
                        for entry in value:
                            if not _nonempty_str(entry):
                                errors.append(
                                    "suite {!r} execution.{} entry must be a non-empty "
                                    "string: {!r}".format(sid, field, entry))
            if "ctest_labels" in execution:
                labels = execution.get("ctest_labels")
                if not isinstance(labels, list):
                    errors.append(
                        "suite {!r} execution.ctest_labels must be a list".format(sid))
                else:
                    for entry in labels:
                        if not _nonempty_str(entry):
                            errors.append(
                                "suite {!r} execution.ctest_labels entry must be a "
                                "non-empty string: {!r}".format(sid, entry))
            # ``gating`` is a vocabulary token with the same scalar-string
            # requirement as kind/disposition; guard it before the membership check.
            gating = execution.get("gating")
            if gating is not None and not _nonempty_str(gating):
                errors.append("suite {!r} gating must be a non-empty string: {!r}".format(sid, gating))
            elif _nonempty_str(gating) and gating_vocab and gating not in gating_vocab:
                errors.append("suite {!r} gating {!r} not in declared vocabulary".format(sid, gating))
            # The in-repo CI lane names a workflow file (optionally followed by a
            # job/step qualifier); the path must resolve. An external: lane is a
            # separate key and is not existence-checked. Either way the mapping
            # must name a lane: a ci_lane carrying neither a non-empty ``in_repo``
            # nor a non-empty ``external`` string points at no CI coverage at all
            # and is a schema error in its own right (a suite may not claim a CI
            # lane it does not actually reference).
            ci_lane = execution.get("ci_lane")
            if isinstance(ci_lane, dict):
                if not _nonempty_str(ci_lane.get("in_repo")) and not _nonempty_str(
                        ci_lane.get("external")):
                    errors.append(
                        "suite {!r} execution.ci_lane must carry a non-empty "
                        "in_repo or external lane".format(sid))
                if "in_repo" in ci_lane:
                    refs_ok += _check_repo_path_field(
                        component_dir, ci_lane.get("in_repo"),
                        "suite {!r} execution.ci_lane.in_repo".format(sid), errors,
                        repo_root)
            elif "ci_lane" in execution:
                errors.append("suite {!r} execution.ci_lane is not a mapping".format(sid))
        elif "execution" in suite:
            errors.append("suite {!r} execution is not a mapping".format(sid))

        # Metadata path pointers: config (case-metadata or authoring file) and
        # generated_plan (the derived plan) are repository paths that must resolve
        # unless marked external:.
        metadata = suite.get("metadata")
        if isinstance(metadata, dict):
            for key in ("config", "generated_plan"):
                if key in metadata:
                    refs_ok += _check_repo_path_field(
                        component_dir, metadata.get(key),
                        "suite {!r} metadata.{}".format(sid, key), errors, repo_root)
        elif "metadata" in suite:
            errors.append("suite {!r} metadata is not a mapping".format(sid))

        # Merge / move / retire targets. ``merge_into`` and ``retire`` must name a
        # disposition_target (the destination record, or the replacement inventory
        # id(s) that preserve retired coverage); ``migrate`` / ``demote`` may omit
        # it. Any target a move disposition does name is resolved against the
        # inventory and may not point at the record itself.
        if _nonempty_str(disp) and disp in MOVE_DISPOSITIONS:
            target = suite.get("disposition_target")
            if target is None:
                if disp in REQUIRE_TARGET_DISPOSITIONS:
                    errors.append(
                        "suite {!r} disposition {!r} must name disposition_target".format(sid, disp))
            else:
                # A disposition_target is a single inventory id or a non-empty list
                # of ids. Validate the container shape before resolving so a
                # mapping, an empty value, or a non-string entry is reported as a
                # schema error rather than crashing the unhashable set-membership
                # test below.
                if _nonempty_str(target):
                    targets = [target]
                elif isinstance(target, list) and target:
                    targets = target
                else:
                    errors.append(
                        "suite {!r} disposition_target must be a non-empty string or a "
                        "non-empty list of inventory ids: {!r}".format(sid, target))
                    targets = []
                for tid in targets:
                    if not _nonempty_str(tid):
                        errors.append(
                            "suite {!r} disposition_target entry must be a non-empty "
                            "inventory id: {!r}".format(sid, tid))
                    elif tid == sid:
                        errors.append("suite {!r} {} target is itself".format(sid, disp))
                    elif tid not in suite_ids:
                        errors.append(
                            "suite {!r} {} target {!r} is not an inventory id".format(sid, disp, tid))

        # Retirement evidence. A retiring suite drops coverage, so beyond naming
        # the replacement inventory id(s) via disposition_target (required and
        # resolved above) it must carry a ``retirement`` mapping documenting that
        # the replacement coverage is equivalent or stronger, the CI evidence that
        # it runs, and the owner approval for the retire.
        if disp == RETIRE_DISPOSITION:
            retirement = suite.get("retirement")
            if not isinstance(retirement, dict):
                errors.append(
                    "suite {!r} disposition 'retire' must carry a retirement mapping "
                    "with {}".format(
                        sid, ", ".join(REQUIRED_RETIREMENT_EVIDENCE)))
            else:
                for field in REQUIRED_RETIREMENT_EVIDENCE:
                    if not _nonempty_str(retirement.get(field)):
                        errors.append(
                            "suite {!r} retirement is missing a non-empty {!r} field".format(
                                sid, field))

        # Source paths: a suite must name at least one source path, and every
        # path must be a string that resolves on disk (or an external: marker).
        source = suite.get("source")
        if isinstance(source, dict):
            paths = source.get("paths")
            if not isinstance(paths, list) or not paths:
                errors.append(
                    "suite {!r} source.paths must be a non-empty list".format(sid))
            else:
                for rel in paths:
                    if not isinstance(rel, str):
                        errors.append(
                            "suite {!r} source path is not a string: {!r}".format(sid, rel))
                        continue
                    # An empty or whitespace-only source path carries no target.
                    # It must be rejected rather than resolved: Path("") (or a
                    # blank string) resolves to the component directory itself,
                    # which exists, so it would otherwise count as a spuriously
                    # satisfied reference.
                    if not rel.strip():
                        errors.append(
                            "suite {!r} source path must be a non-empty string".format(sid))
                        continue
                    if rel.startswith(EXTERNAL_MARKER):
                        # A bare ``external:`` with no target names nothing; the
                        # marker must carry the external reference it stands in for.
                        if not rel[len(EXTERNAL_MARKER):].strip():
                            errors.append(
                                "suite {!r} source path external marker must name a "
                                "target: {!r}".format(sid, rel))
                        continue
                    location = _path_location_status(component_dir, rel, repo_root)
                    if location == "absolute":
                        errors.append(
                            "suite {!r} source path must be repository-relative: {}".format(
                                sid, rel))
                    elif location == "outside":
                        errors.append(
                            "suite {!r} source path escapes repository root: {}".format(
                                sid, rel))
                    elif _path_exists(component_dir, rel, repo_root):
                        refs_ok += 1
                    else:
                        errors.append("suite {!r} source path missing: {}".format(sid, rel))
            # The source block must also name how the suite is wired into the
            # build: ``build_option`` is the CMake switch that enables it and
            # ``build_target`` is the aggregate target it builds under. These are
            # the facts CMake reconciliation and a reader tracing a suite back to
            # its CMakeLists both rely on, so a present-but-empty or non-string
            # value (or outright absence) is a schema error.
            for field in ("build_option", "build_target"):
                if field not in source:
                    errors.append(
                        "suite {!r} source is missing required {!r}".format(sid, field))
                elif not _nonempty_str(source.get(field)):
                    errors.append(
                        "suite {!r} source.{} must be a non-empty string".format(
                            sid, field))
        elif "source" in suite:
            errors.append("suite {!r} source is not a mapping".format(sid))

        # Evidence: the block must be a non-empty list, and each entry must be a
        # mapping carrying both a path and a claim; the path must resolve on disk
        # unless marked external. A present but empty or non-list evidence block is
        # a schema error, not an iteration target. Validated equivalently to the
        # coverage-model class evidence.
        suite_evidence = suite.get("evidence")
        if "evidence" in suite and (not isinstance(suite_evidence, list) or not suite_evidence):
            errors.append("suite {!r} evidence must be a non-empty list".format(sid))
        for ev in suite_evidence if isinstance(suite_evidence, list) else []:
            if not isinstance(ev, dict):
                errors.append(
                    "suite {!r} evidence entry is not a mapping: {!r}".format(sid, ev))
                continue
            rel = ev.get("path")
            if not _nonempty_str(rel):
                errors.append("suite {!r} evidence entry missing a path".format(sid))
            if not _nonempty_str(ev.get("claim")):
                errors.append("suite {!r} evidence entry missing a claim".format(sid))
            if not isinstance(rel, str):
                continue
            if rel.startswith(EXTERNAL_MARKER):
                # A bare ``external:`` with no target names nothing; the marker
                # must carry the external reference it stands in for.
                if not rel[len(EXTERNAL_MARKER):].strip():
                    errors.append(
                        "suite {!r} evidence path external marker must name a "
                        "target: {}".format(sid, rel))
                continue
            location = _path_location_status(component_dir, rel, repo_root)
            if location == "absolute":
                errors.append(
                    "suite {!r} evidence path must be repository-relative: {}".format(sid, rel))
            elif location == "outside":
                errors.append(
                    "suite {!r} evidence path escapes repository root: {}".format(sid, rel))
            elif _path_is_file(component_dir, rel, repo_root):
                refs_ok += 1
            elif _path_exists(component_dir, rel, repo_root):
                errors.append(
                    "suite {!r} evidence path is a directory, not a file: {}".format(sid, rel))
            else:
                errors.append("suite {!r} evidence path missing: {}".format(sid, rel))

        # Historical context. Every suite must say where it came from or which
        # control governs it, so a reader encountering a non-obvious guard can
        # trace the rationale: either a non-empty ``incidents`` list naming the
        # controlling TI-NNN incident(s), or a ``history`` narrative that actually
        # establishes context. A bare/generic placeholder (for example a stock
        # "origin not located; repository evidence records current purpose." line)
        # does NOT count on its own: a sufficient narrative must carry an explicit
        # evidence/origin marker AND not be a contentless sentinel, and a generic
        # unknown-origin narrative must additionally cite record-specific context
        # (a concrete repository path or a governing incident) so a lightly
        # reworded template cannot pass on a marker word alone. An honest unknown
        # origin is still accepted as long as it points at the repository evidence
        # that does exist; no ticket is invented to satisfy the rule.
        history = suite.get("history")
        incidents = suite.get("incidents")
        if "history" in suite and not _nonempty_str(history):
            errors.append("suite {!r} history must be a non-empty string".format(sid))
        has_incidents = isinstance(incidents, list) and bool(incidents)
        if "incidents" in suite:
            if not has_incidents:
                errors.append(
                    "suite {!r} incidents must be a non-empty list".format(sid))
            else:
                for inc in incidents:
                    # Each entry must be a bare TI-NNN id; its resolution to a
                    # defined incident is handled by the incident-reference check
                    # (the id is collected below like any other TI reference).
                    if not _nonempty_str(inc) or not _INCIDENT_ID_FULL_RE.match(inc.strip()):
                        errors.append(
                            "suite {!r} incidents entry must be a TI-NNN incident id: "
                            "{!r}".format(sid, inc))
        # A suite is history-blind unless it names governing incidents or carries
        # a sufficient narrative. Report the most specific diagnosis: a present
        # but insufficient history (a bare/generic placeholder, a lightly expanded
        # unknown-origin template, or a note with no evidence/origin marker) gets a
        # targeted message; an entirely absent one gets the general requirement.
        if not has_incidents and not _history_is_sufficient(history):
            if _nonempty_str(history):
                errors.append(
                    "suite {!r} history is a bare/generic placeholder: it must carry "
                    "an explicit evidence/origin marker (e.g. 'repository evidence') "
                    "with record-specific context (a concrete repository path or a "
                    "governing incident), not a reworded generic sentinel; or the "
                    "suite must name governing incidents".format(sid))
            else:
                errors.append(
                    "suite {!r} must carry governing incidents or a history that "
                    "establishes real context".format(sid))

        # related_restructuring is a structured local Markdown pointer. When a
        # suite carries it, it must be a non-empty string that names a path which
        # resolves on disk within the boundary root: a present-but-malformed value
        # (an empty/blank string, a non-string scalar, or a list/mapping) is a
        # schema error in its own right, and an absolute path or a traversal
        # outside the boundary root is rejected exactly like any other repository
        # path pointer rather than silently skipped.
        if "related_restructuring" in suite:
            rr = suite.get("related_restructuring")
            if not _nonempty_str(rr):
                errors.append(
                    "suite {!r} related_restructuring must be a non-empty string: "
                    "{!r}".format(sid, rr))
            else:
                location = _path_location_status(component_dir, rr, repo_root)
                if location == "absolute":
                    errors.append(
                        "suite {!r} related_restructuring must be repository-relative: "
                        "{}".format(sid, rr))
                elif location == "outside":
                    errors.append(
                        "suite {!r} related_restructuring escapes repository root: "
                        "{}".format(sid, rr))
                elif _path_exists(component_dir, rr, repo_root):
                    refs_ok += 1
                else:
                    errors.append(
                        "suite {!r} related_restructuring missing: {}".format(sid, rr))

        # Incident references anywhere in the suite record.
        incident_refs.update(_collect_incident_refs(suite))

    return errors, suite_ids, incident_refs, {"records": len(suites), "refs": refs_ok}


def _strip_cmake_comments(cmake_path):
    """Return a CMakeLists' text with all comments removed, offsets by line kept.

    CMake commands are case-insensitive and may span several physical lines, so a
    caller scans the comment-free text as a whole. A bracket comment
    (``#[[ ... ]]`` / ``#[=[ ... ]=]``) is masked out first - it may itself span
    lines - then a line comment (everything from the first ``#`` to end of line)
    is dropped from each line. A fully commented-out call, an inline trailing
    ``# ...`` comment, and a bracketed block comment therefore all contribute
    nothing to the scanned text.
    """
    masked = _CMAKE_BRACKET_COMMENT_RE.sub(
        lambda m: _blank_like(m.group(0)), cmake_path.read_text(encoding="utf-8"))
    cleaned = []
    for line in masked.splitlines():
        hash_index = line.find("#")
        if hash_index != -1:
            line = line[:hash_index]
        cleaned.append(line)
    return "\n".join(cleaned)


def _iter_cmake_commands(text):
    """Yield ``(name, args)`` for each top-level CMake command in comment-free text.

    A CMake command is an identifier followed by a parenthesized argument list;
    this yields the command name and its argument text (the span inside the outer
    parentheses, without them) for every invocation in source order. The scan is
    command-aware: after a command is found its whole argument span is consumed
    before scanning resumes, so a token that only appears *inside* another
    command's arguments (for example an ``add_subdirectory(...)`` reminder inside
    a ``message(...)`` string, or ``hip_add_host_test`` as the declared name of a
    ``function(hip_add_host_test TARGET)`` header) is never mistaken for a command
    of its own.

    While locating the matching close parenthesis the scan skips over double-quoted
    strings (honoring backslash escapes) and counts nested parentheses, so a ``(``
    or ``)`` inside a quoted argument - or a balanced nested group such as
    ``if(NOT (A AND B))`` - does not end the command prematurely. ``text`` must
    already have comments stripped (see ``_strip_cmake_comments``).
    """
    pos = 0
    length = len(text)
    while True:
        match = _CMAKE_COMMAND_START_RE.search(text, pos)
        if match is None:
            return
        name = match.group(1)
        arg_start = match.end()
        i = arg_start
        depth = 1
        while i < length and depth > 0:
            ch = text[i]
            if ch == '"':
                # Skip the quoted string wholesale: parentheses inside it are part
                # of the argument value, not command structure.
                i += 1
                while i < length:
                    if text[i] == "\\":
                        i += 2
                        continue
                    if text[i] == '"':
                        i += 1
                        break
                    i += 1
                continue
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        yield name, text[arg_start:i]
        # Resume after this command's closing parenthesis so its argument span is
        # not re-scanned for nested command-like tokens.
        pos = i + 1


def _first_cmake_argument(args):
    """Return the first argument token of a CMake command's argument text, or None.

    Leading whitespace (which may span newlines) and an optional opening double
    quote are skipped before the token is captured; the token stops at the next
    quote, whitespace, or closing parenthesis.
    """
    match = _CMAKE_FIRST_ARG_RE.match(args)
    return match.group(1) if match else None


def _active_cmake_groups(cmake_path):
    """Return the list of group names a CMakeLists.txt actively builds.

    Each real top-level ``add_subdirectory(<group>)`` command names an active
    group. The comment-free text (see ``_strip_cmake_comments``) is parsed
    command-by-command (see ``_iter_cmake_commands``) and matched
    case-insensitively, so a call split across physical lines or written in any
    case is recognized, a deliberately disabled (commented) group is treated as
    not-built, and an ``add_subdirectory(...)`` token that appears only inside
    another command's quoted string (for example a ``message(...)`` reminder) is
    not mistaken for an active group that must be represented by an inventory
    suite.
    """
    groups = []
    for name, args in _iter_cmake_commands(_strip_cmake_comments(cmake_path)):
        if name.lower() == _CMAKE_ADD_SUBDIRECTORY:
            group = _first_cmake_argument(args)
            if group:
                groups.append(group)
    return groups


def _active_host_test_targets(cmake_path):
    """Return the host-test executables a CMakeLists registers via hip_add_host_test.

    Each real top-level ``hip_add_host_test(<TARGET> ...)`` command registers a
    host-side harness test (labelled ``hosttest``) built under the ``build_tests``
    aggregate target. These executables live outside the per-area
    add_subdirectory groups, so they are enumerated separately and reconciled
    against a host/harness inventory record. The comment-free text is parsed
    command-by-command (see ``_iter_cmake_commands``), so a commented-out
    registration contributes nothing and a ``hip_add_host_test(...)`` token inside
    another command's quoted string is not registered.

    Command-aware parsing also excludes a ``function``/``macro`` declaration that
    *defines* ``hip_add_host_test`` itself: ``function(hip_add_host_test TARGET)``
    is the ``function`` command whose first argument is ``hip_add_host_test``, not
    a ``hip_add_host_test`` command, so no fixed-width lookbehind or masking is
    needed to tell a declaration (in any spacing/case/multiline form) apart from a
    genuine registration.
    """
    targets = []
    for name, args in _iter_cmake_commands(_strip_cmake_comments(cmake_path)):
        if name.lower() == _CMAKE_HIP_ADD_HOST_TEST:
            target = _first_cmake_argument(args)
            if target:
                targets.append(target)
    return targets


def _normalize_group_path(path):
    """Normalize a repository-relative directory token for CMake reconciliation.

    Reconciliation compares inventory ``source.paths`` entries (and documented
    exclusion group names) against the discovered ``catch/<area>/<group>`` token.
    Cosmetic variations in how the same directory is written must not defeat the
    match, so a leading ``./`` is dropped, back-slashes are folded to forward
    slashes, duplicate slashes are collapsed, and a trailing slash is removed.
    This normalization is applied ONLY for reconciliation matching; the stored
    inventory value is never rewritten.
    """
    norm = path.strip().replace("\\", "/")
    while norm.startswith("./"):
        norm = norm[2:]
    norm = re.sub(r"/{2,}", "/", norm)
    if len(norm) > 1:
        norm = norm.rstrip("/")
    return norm


def _represented_group_dirs(inventory):
    """Return the set of normalized directory tokens the inventory names.

    Every non-external ``source.paths`` entry of every suite is normalized for
    reconciliation (see ``_normalize_group_path``); a CMake group directory
    ``catch/<area>/<group>`` is considered represented when some suite names that
    directory or a path beneath it.
    """
    paths = set()
    if not isinstance(inventory, dict):
        return paths
    for suite in inventory.get("suites") or []:
        if not isinstance(suite, dict):
            continue
        source = suite.get("source")
        if not isinstance(source, dict):
            continue
        for rel in source.get("paths") or []:
            if isinstance(rel, str) and rel.strip() and not rel.startswith(
                    EXTERNAL_MARKER):
                paths.add(_normalize_group_path(rel))
    return paths


def _group_is_represented(group_dir, represented):
    """True when some inventory source path names group_dir or a path beneath it.

    ``group_dir`` is normalized the same way the ``represented`` set is (see
    ``_normalize_group_path``) so the comparison is insensitive to cosmetic path
    variations on either side.
    """
    group_dir = _normalize_group_path(group_dir)
    prefix = group_dir + "/"
    return any(path == group_dir or path.startswith(prefix)
              for path in represented)


def check_cmake_reconciliation(inventory, component_dir):
    """Reconcile active CMake suite groups against the inventory.

    For each Catch2 area (unit / contract / performance / stress) whose
    per-area ``CMakeLists.txt`` exists, every actively built group
    (``add_subdirectory(<group>)``) must be represented by an inventory suite
    whose ``source.paths`` names ``catch/<area>/<group>`` (or a path beneath it).
    A built group that no suite covers is a silent coverage hole and is reported,
    unless it is explicitly documented under the inventory's optional
    ``cmake_suite_group_exclusions`` block with a categorized reason and a note.

    The reconciliation is feasibility-gated: an area whose CMakeLists.txt is not
    present on disk is skipped, so the check is a no-op for a self-contained
    fixture that ships no CMake tree.

    Returns a list of deterministic error strings.
    """
    errors = []
    if not isinstance(inventory, dict):
        return errors

    component_dir = Path(component_dir)
    represented = _represented_group_dirs(inventory)

    # Documented exclusions: an explicit, categorized allowance for a built group
    # that intentionally carries no inventory suite. Each entry must name the
    # group, a reason within the documented vocabulary, and a note (the clear
    # metadata that records why the allowance is safe).
    excluded = set()
    raw_exclusions = inventory.get("cmake_suite_group_exclusions")
    if raw_exclusions is not None:
        if not isinstance(raw_exclusions, list):
            errors.append("cmake_suite_group_exclusions must be a list")
            raw_exclusions = []
        for entry in raw_exclusions:
            if not isinstance(entry, dict):
                errors.append(
                    "cmake_suite_group_exclusions entry is not a mapping: "
                    "{!r}".format(entry))
                continue
            group = entry.get("group")
            reason = entry.get("reason")
            note = entry.get("note")
            if not _nonempty_str(group):
                errors.append(
                    "cmake_suite_group_exclusions entry must name a non-empty "
                    "group: {!r}".format(group))
                continue
            group = group.strip()
            if not _nonempty_str(reason) or reason not in CMAKE_EXCLUSION_REASONS:
                errors.append(
                    "cmake_suite_group_exclusions group {!r} reason must be one "
                    "of {}: {!r}".format(
                        group, sorted(CMAKE_EXCLUSION_REASONS), reason))
                continue
            if not _nonempty_str(note):
                errors.append(
                    "cmake_suite_group_exclusions group {!r} must carry a "
                    "non-empty note".format(group))
                continue
            # A group that is both represented by a suite and documented as
            # excluded is contradictory metadata: the exclusion claims no suite
            # covers it while a suite does. Report it rather than let the two
            # records disagree silently.
            if _group_is_represented(group, represented):
                errors.append(
                    "cmake_suite_group_exclusions group {!r} is already "
                    "represented by an inventory suite; remove the "
                    "exclusion".format(group))
                continue
            # Store the normalized group so reconciliation matches it against the
            # discovered ``catch/<area>/<group>`` token regardless of cosmetic
            # path variations (a ``./`` prefix, a trailing slash); the displayed
            # ``group`` in any diagnostic above stays as authored.
            excluded.add(_normalize_group_path(group))

    for area in CMAKE_RECONCILIATION_AREAS:
        cmake_path = component_dir / "catch" / area / "CMakeLists.txt"
        if not cmake_path.is_file():
            continue
        for group in _active_cmake_groups(cmake_path):
            group_dir = "catch/{}/{}".format(area, group)
            if _group_is_represented(group_dir, represented):
                continue
            if _normalize_group_path(group_dir) in excluded:
                continue
            errors.append(
                "active CMake suite group {!r} is not represented by any "
                "inventory suite (add a suite whose source.paths covers it, or "
                "document it under cmake_suite_group_exclusions)".format(group_dir))

    # Host-side harness tests are registered through ``hip_add_host_test`` in a
    # dedicated CMakeLists rather than a per-area add_subdirectory group, so they
    # are reconciled on their own: when that file actively registers host-test
    # executables, the inventory must carry a host/harness suite whose
    # source.paths covers the registering directory. An unrepresented host test
    # is the same silent coverage hole the group reconciliation guards against,
    # and is reported (naming the executables) unless the registering directory
    # is documented under cmake_suite_group_exclusions.
    host_cmake = component_dir.joinpath(*HOSTTEST_CMAKE_REL)
    if host_cmake.is_file():
        host_targets = _active_host_test_targets(host_cmake)
        if (host_targets
                and not _group_is_represented(HOSTTEST_GROUP_DIR, represented)
                and _normalize_group_path(HOSTTEST_GROUP_DIR) not in excluded):
            errors.append(
                "active host-test executable(s) {} registered via hip_add_host_test "
                "under {} are not represented by any inventory suite (add a "
                "host/harness suite whose source.paths covers {!r}, or document it "
                "under cmake_suite_group_exclusions)".format(
                    ", ".join(sorted(host_targets)),
                    "/".join(HOSTTEST_CMAKE_REL), HOSTTEST_GROUP_DIR))
    return errors


def _has_gap_metadata(pairwise):
    """True when a pairwise check explicitly acknowledges an uncovered interaction.

    A pairwise check normally claims coverage by naming the inventory suite(s)
    that exercise the interaction (a non-empty ``current_inventory``). A check
    that covers nothing yet must say so on purpose rather than silently leave the
    list empty: it carries an explicit ``gap`` acknowledgement - a boolean ``true``
    or a non-empty string/mapping/list documenting the known, deliberate gap. The
    ``gap`` field is optional metadata the real model does not need (every real
    pairwise names covering inventory), so this only matters for a check that
    declares an empty ``current_inventory``.
    """
    gap = pairwise.get("gap")
    if gap is True:
        return True
    if isinstance(gap, str):
        return bool(gap.strip())
    if isinstance(gap, (dict, list)):
        return bool(gap)
    return False


def check_coverage_model(model, suite_ids, component_dir, inventory_kind_vocab=None,
                         repo_root=None):
    """Validate the coverage model against the inventory.

    Returns (errors, incident_refs, counts) where counts has 'classes' and
    'refs'. ``repo_root``, when given, is the boundary a repository path may not
    resolve outside of (see ``_path_location_status``); it defaults to the
    repository root two levels above the component directory.
    """
    errors = []
    incident_refs = set()
    refs_ok = 0
    inventory_kind_vocab = inventory_kind_vocab or set()

    if not isinstance(model, dict):
        return (["coverage model did not parse into a mapping"], incident_refs,
                {"classes": 0, "refs": 0})

    # Top-level schema: the coverage model carries the same identity block as the
    # inventory and is validated equivalently before any per-class check runs.
    missing_top = REQUIRED_COVERAGE_TOP_KEYS - set(model.keys())
    if missing_top:
        errors.append(
            "coverage model missing top-level keys: {}".format(sorted(missing_top)))

    if "schema_version" in model:
        schema_version = model.get("schema_version")
        if not _is_scalar(schema_version):
            errors.append("coverage model schema_version must be a non-empty scalar")
        elif not _is_supported_schema_version(schema_version):
            errors.append(
                "coverage model schema_version must be the supported version {} "
                "(got {!r})".format(SUPPORTED_SCHEMA_VERSION, schema_version))

    if "component" in model:
        component = model.get("component")
        if not _nonempty_str(component):
            errors.append("coverage model component must be a non-empty string")
        elif component != SUPPORTED_COMPONENT:
            errors.append(
                "coverage model component must be {!r} (got {!r})".format(
                    SUPPORTED_COMPONENT, component))

    cm = model.get("coverage_model")
    if not isinstance(cm, dict):
        errors.append("coverage model is missing the coverage_model block")
        return errors, incident_refs, {"classes": 0, "refs": 0}

    declared_axes = set()
    declared_axis_ids = set()
    # Axis.class references that are declared but intentionally not yet claimed by
    # any coverage class or pairwise check. Such a class is kept in the vocabulary
    # for completeness (it names a dimension value the model acknowledges) but is
    # exempted from the "every declared class is audited" rule only when the axis
    # explicitly lists it under ``informational_classes``.
    informational_axes = set()
    # Axis.class references actually used by a coverage class or pairwise check.
    referenced_axes = set()
    axes = cm.get("axes")
    if axes is None:
        axes = []
    elif not isinstance(axes, list):
        errors.append("coverage model axes must be a list: {!r}".format(axes))
        axes = []
    for axis in axes:
        if not isinstance(axis, dict):
            errors.append("coverage model axis is not a mapping: {!r}".format(axis))
            continue
        axis_id = axis.get("id")
        if not _nonempty_str(axis_id):
            errors.append(
                "coverage axis id must be a non-empty string: {!r}".format(axis_id))
            continue
        # An axis id must be unique: a second axis record sharing an id makes the
        # dimension ambiguous (a later declaration silently shadows the earlier
        # one), so the repeat is a schema error.
        if axis_id in declared_axis_ids:
            errors.append("duplicate coverage axis id: {!r}".format(axis_id))
        declared_axis_ids.add(axis_id)
        classes = axis.get("classes")
        if classes is not None and not isinstance(classes, list):
            errors.append(
                "coverage axis {!r} classes must be a list".format(axis_id))
            classes = []
        axis_classes = set()
        for cls in classes or []:
            if not _nonempty_str(cls):
                errors.append(
                    "coverage axis {!r} class must be a non-empty string: {!r}".format(
                        axis_id, cls))
                continue
            # A class listed twice within one axis is a dead duplicate: it
            # collapses to a single declared axis.class, so the repeat says
            # nothing new and is reported rather than silently deduplicated.
            if cls in axis_classes:
                errors.append(
                    "coverage axis {!r} has duplicate class {!r}".format(
                        axis_id, cls))
                continue
            axis_classes.add(cls)
            declared_axes.add("{}.{}".format(axis_id, cls))
        # Optional informational-class marker: each entry must be a non-empty
        # string that names a class the axis actually declares (an informational
        # marker for an undeclared class is a schema error, not an exemption).
        info = axis.get("informational_classes")
        if info is not None and not isinstance(info, list):
            errors.append(
                "coverage axis {!r} informational_classes must be a list".format(axis_id))
            info = []
        for cls in info or []:
            if not _nonempty_str(cls):
                errors.append(
                    "coverage axis {!r} informational class must be a non-empty "
                    "string: {!r}".format(axis_id, cls))
                continue
            if cls not in axis_classes:
                errors.append(
                    "coverage axis {!r} informational class {!r} is not a declared "
                    "class of that axis".format(axis_id, cls))
                continue
            informational_axes.add("{}.{}".format(axis_id, cls))

    for req in sorted(REQUIRED_AXES):
        if req not in declared_axis_ids:
            errors.append("coverage model is missing required axis {!r}".format(req))

    def check_axis_refs(owner, refs):
        if refs is None:
            return
        if not isinstance(refs, list):
            errors.append("{} axis refs must be a list: {!r}".format(owner, refs))
            return
        for ref in refs:
            if not isinstance(ref, str):
                errors.append("{} axis ref must be a string: {!r}".format(owner, ref))
                continue
            referenced_axes.add(ref)
            if ref not in declared_axes:
                errors.append("{} axis ref {!r} is not a declared axis.class".format(owner, ref))

    def check_inventory_refs(owner, refs):
        nonlocal refs_ok
        if refs is None:
            return
        if not isinstance(refs, list):
            errors.append("{} inventory refs must be a list: {!r}".format(owner, refs))
            return
        for ref in refs:
            if not isinstance(ref, str):
                errors.append("{} inventory ref must be a string: {!r}".format(owner, ref))
                continue
            if ref not in suite_ids:
                errors.append("{} references unknown inventory id {!r}".format(owner, ref))
            else:
                refs_ok += 1

    seen_class_ids = set()
    classes = cm.get("classes")
    if classes is None:
        classes = []
    elif not isinstance(classes, list):
        errors.append("coverage model classes must be a list: {!r}".format(classes))
        classes = []
    for cls in classes:
        if not isinstance(cls, dict):
            errors.append("coverage class record is not a mapping: {!r}".format(cls))
            continue
        cid = cls.get("id", "<no-id>")
        # An id must be a non-empty string before duplicate detection or any
        # cross-reference is meaningful; a malformed id is a schema error and is
        # not added to the seen-id set.
        if not _nonempty_str(cls.get("id")):
            errors.append(
                "coverage class id must be a non-empty string: {!r}".format(cls.get("id")))
        elif cid in seen_class_ids:
            errors.append("duplicate coverage class id: {!r}".format(cid))
        else:
            seen_class_ids.add(cid)
        missing = REQUIRED_CLASS_KEYS - set(cls.keys())
        if missing:
            errors.append("class {!r} missing keys: {}".format(cid, sorted(missing)))
        if "owner" in cls and not _nonempty_str(cls.get("owner")):
            errors.append("class {!r} owner must be a non-empty string".format(cid))
        # The descriptive scalars carry the class's meaning: ``intent`` and
        # ``gap_status`` are free-text and must be non-empty strings, and
        # ``risk`` is a review-priority token that must be within the documented
        # high/medium/low vocabulary. (The missing-key check above handles
        # outright absence.)
        for field in ("intent", "gap_status"):
            if field in cls and not _nonempty_str(cls.get(field)):
                errors.append(
                    "class {!r} {} must be a non-empty string".format(cid, field))
        risk = cls.get("risk")
        if "risk" in cls and not _nonempty_str(risk):
            errors.append("class {!r} risk must be a non-empty string".format(cid))
        elif _nonempty_str(risk) and risk not in COVERAGE_RISK_VALUES:
            errors.append(
                "class {!r} risk {!r} not in {}".format(
                    cid, risk, sorted(COVERAGE_RISK_VALUES)))
        # expected_test_kinds enumerates the tiers the class is meant to hold and
        # must be a non-empty list.
        expected_kinds = cls.get("expected_test_kinds")
        if "expected_test_kinds" in cls and (
                not isinstance(expected_kinds, list) or not expected_kinds):
            errors.append(
                "class {!r} expected_test_kinds must be a non-empty list".format(cid))
        elif isinstance(expected_kinds, list):
            for kind in expected_kinds:
                if not _nonempty_str(kind):
                    errors.append(
                        "class {!r} expected_test_kinds entry must be a non-empty string: {!r}".format(
                            cid, kind))
                elif inventory_kind_vocab and kind not in inventory_kind_vocab:
                    errors.append(
                        "class {!r} expected_test_kinds entry {!r} not in inventory kind vocabulary".format(
                            cid, kind))
        # axes and current_inventory are the class's required cross-references and
        # must each be a non-empty list: an empty axes list leaves the class
        # unanchored to any coverage dimension, and an empty current_inventory
        # claims a class that nothing covers. Per-entry membership is checked by the
        # helpers below; here the container shape must be a populated list.
        if "axes" in cls and (
                not isinstance(cls.get("axes"), list) or not cls.get("axes")):
            errors.append("class {!r} axes must be a non-empty list".format(cid))
        if "current_inventory" in cls and (
                not isinstance(cls.get("current_inventory"), list)
                or not cls.get("current_inventory")):
            errors.append(
                "class {!r} current_inventory must be a non-empty list".format(cid))
        check_axis_refs("class {!r}".format(cid), cls.get("axes"))
        check_inventory_refs("class {!r}".format(cid), cls.get("current_inventory"))
        # Evidence: the list must be non-empty, and each entry must be a mapping
        # carrying both a non-empty path and a non-empty claim; the path must
        # resolve on disk unless marked external. Validated equivalently to the
        # inventory suite evidence.
        evidence = cls.get("evidence")
        if "evidence" in cls and (not isinstance(evidence, list) or not evidence):
            errors.append("class {!r} evidence must be a non-empty list".format(cid))
        for ev in evidence if isinstance(evidence, list) else []:
            if not isinstance(ev, dict):
                errors.append(
                    "class {!r} evidence entry is not a mapping: {!r}".format(cid, ev))
                continue
            rel = ev.get("path")
            if not _nonempty_str(rel):
                errors.append("class {!r} evidence entry missing a path".format(cid))
            if not _nonempty_str(ev.get("claim")):
                errors.append("class {!r} evidence entry missing a claim".format(cid))
            if not isinstance(rel, str):
                continue
            if rel.startswith(EXTERNAL_MARKER):
                # A bare ``external:`` with no target names nothing; the marker
                # must carry the external reference it stands in for.
                if not rel[len(EXTERNAL_MARKER):].strip():
                    errors.append(
                        "class {!r} evidence path external marker must name a "
                        "target: {}".format(cid, rel))
                continue
            location = _path_location_status(component_dir, rel, repo_root)
            if location == "absolute":
                errors.append(
                    "class {!r} evidence path must be repository-relative: {}".format(cid, rel))
            elif location == "outside":
                errors.append(
                    "class {!r} evidence path escapes repository root: {}".format(cid, rel))
            elif _path_is_file(component_dir, rel, repo_root):
                refs_ok += 1
            elif _path_exists(component_dir, rel, repo_root):
                errors.append(
                    "class {!r} evidence path is a directory, not a file: {}".format(cid, rel))
            else:
                errors.append("class {!r} evidence path missing: {}".format(cid, rel))
        incident_refs.update(_collect_incident_refs(cls))

    seen_pair_ids = set()
    present_categories = set()
    pairwise_checks = cm.get("pairwise_checks")
    if pairwise_checks is None:
        pairwise_checks = []
    elif not isinstance(pairwise_checks, list):
        errors.append(
            "coverage model pairwise_checks must be a list: {!r}".format(pairwise_checks))
        pairwise_checks = []
    for pc in pairwise_checks:
        if not isinstance(pc, dict):
            errors.append("pairwise check record is not a mapping: {!r}".format(pc))
            continue
        pid = pc.get("id", "<no-id>")
        # An id must be a non-empty string before duplicate detection; a malformed
        # (for example unhashable) id is a schema error and is not added to the set.
        if not _nonempty_str(pc.get("id")):
            errors.append(
                "pairwise check id must be a non-empty string: {!r}".format(pc.get("id")))
        elif pid in seen_pair_ids:
            errors.append("duplicate pairwise check id: {!r}".format(pid))
        else:
            seen_pair_ids.add(pid)
        missing = REQUIRED_PAIRWISE_KEYS - set(pc.keys())
        if missing:
            errors.append("pairwise {!r} missing keys: {}".format(pid, sorted(missing)))
        # The descriptive scalars carry the check's meaning: ``category``,
        # ``status``, ``rationale``, and ``gap_status`` are free-text and must be
        # non-empty strings, and ``priority`` is a review-priority token that must
        # be within the documented high/medium/low vocabulary.
        for field in ("category", "status", "rationale", "gap_status"):
            if field in pc and not _nonempty_str(pc.get(field)):
                errors.append(
                    "pairwise {!r} {} must be a non-empty string".format(pid, field))
        if _nonempty_str(pc.get("category")):
            present_categories.add(pc.get("category"))
        priority = pc.get("priority")
        if "priority" in pc and not _nonempty_str(priority):
            errors.append("pairwise {!r} priority must be a non-empty string".format(pid))
        elif _nonempty_str(priority) and priority not in COVERAGE_RISK_VALUES:
            errors.append(
                "pairwise {!r} priority {!r} not in {}".format(
                    pid, priority, sorted(COVERAGE_RISK_VALUES)))
        # A pairwise check pairs exactly two axis.class dimensions; a pair with any
        # other number of refs is not a pair and is a schema error. (Membership of
        # each ref is validated by check_axis_refs; the non-list case is reported
        # there as well.)
        pair = pc.get("pair")
        if isinstance(pair, list) and len(pair) != 2:
            errors.append(
                "pairwise {!r} pair must name exactly two axis.class refs (got {})".format(
                    pid, len(pair)))
        elif isinstance(pair, list):
            first, second = pair
            if isinstance(first, str) and isinstance(second, str):
                first_axis, _, _ = first.partition(".")
                second_axis, _, _ = second.partition(".")
                if first == second:
                    errors.append(
                        "pairwise {!r} pair must reference distinct members".format(pid))
                if first_axis == second_axis:
                    errors.append(
                        "pairwise {!r} pair must reference distinct axes".format(pid))
        check_axis_refs("pairwise {!r}".format(pid), pc.get("pair"))
        check_inventory_refs("pairwise {!r}".format(pid), pc.get("current_inventory"))
        # A pairwise check must either name the inventory that covers the
        # interaction (a non-empty current_inventory) or explicitly acknowledge an
        # uncovered one via gap metadata. An empty current_inventory with no gap
        # acknowledgement is an undocumented coverage hole: the check claims a
        # cross-axis audit it neither covers nor records as a known gap.
        current_inventory = pc.get("current_inventory")
        has_current = isinstance(current_inventory, list) and bool(current_inventory)
        if not has_current and not _has_gap_metadata(pc):
            errors.append(
                "pairwise {!r} must name a non-empty current_inventory or carry "
                "explicit gap metadata (a 'gap' acknowledgement)".format(pid))
        incident_refs.update(_collect_incident_refs(pc))

    # The pairwise block is the model's cross-axis audit; it must actually be
    # performed. An empty block claims no audit at all, and a missing category
    # means a whole dimension of interactions (API x behavior, API x environment,
    # feature-integration x execution context, or capability x cadence) is never
    # reviewed.
    if not pairwise_checks:
        errors.append(
            "coverage model pairwise_checks must be a non-empty list of cross-axis checks")
    else:
        missing_categories = REQUIRED_PAIRWISE_CATEGORIES - present_categories
        if missing_categories:
            errors.append(
                "coverage model pairwise_checks is missing required "
                "categories: {}".format(sorted(missing_categories)))

    # Every declared axis.class must be meaningfully audited: referenced by at
    # least one coverage class ``axes`` list or pairwise ``pair``, unless the axis
    # explicitly marks it ``informational_classes``. A declared-but-unreferenced,
    # non-informational class is a dead vocabulary entry that claims a dimension
    # value the model never actually audits.
    unaudited = sorted(declared_axes - referenced_axes - informational_axes)
    for axis_class in unaudited:
        errors.append(
            "coverage axis class {!r} is declared but never referenced by a "
            "coverage class or pairwise check (reference it, or mark it under the "
            "axis's informational_classes)".format(axis_class))

    return errors, incident_refs, {"classes": len(classes), "refs": refs_ok}


def check_identity_consistency(inventory, coverage):
    """Cross-check that the inventory and coverage model declare the same identity.

    The two documents are a connected set companion to each other, so beyond each
    declaring the supported schema_version and component on its own, they must
    agree with each other on both fields: a coverage model must never be paired
    with an inventory of a different schema version or component. Only runs when
    both documents parse into mappings and both carry the field; the per-document
    presence/shape checks report an absent or malformed field on their own.
    """
    errors = []
    if not isinstance(inventory, dict) or not isinstance(coverage, dict):
        return errors
    for field in ("schema_version", "component"):
        if field in inventory and field in coverage:
            inv_value = inventory.get(field)
            cov_value = coverage.get(field)
            if inv_value != cov_value:
                errors.append(
                    "coverage model {} {!r} does not match inventory {} {!r}".format(
                        field, cov_value, field, inv_value))
    return errors


def _collect_incident_refs(obj):
    """Return the set of TI-NNN tokens appearing in any string within obj."""
    refs = set()
    if isinstance(obj, str):
        refs.update(_INCIDENT_REF_RE.findall(obj))
    elif isinstance(obj, dict):
        for value in obj.values():
            refs.update(_collect_incident_refs(value))
    elif isinstance(obj, list):
        for item in obj:
            refs.update(_collect_incident_refs(item))
    return refs


def _incident_sections(text):
    """Yield (incident_id, section_text) for each ## TI-NNN: heading.

    Fenced code blocks and HTML comments are masked (offsets preserved) before
    headings are located and sections are sliced, so a ``## TI-NNN:`` heading or
    a ``**Label.**`` field that lives inside a code fence or a comment is example
    text - it defines no incident and satisfies no required field. Inline code
    spans are deliberately NOT masked: a backticked real Origin hash outside a
    fence or comment must survive so the citation check can still validate it.
    """
    masked = _mask_comments(_mask_fences(text))
    matches = list(_INCIDENT_HEADING_RE.finditer(masked))
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(masked)
        yield match.group(1), masked[start:end]


def defined_incident_ids(incidents_path, errors):
    """Return the set of TI-NNN ids defined by headings in TESTING_INCIDENTS.md.

    A TI-NNN defined by more than one ``## TI-NNN:`` heading is a schema error:
    the id is meant to be stable and unique, and a second definition makes the
    account a reader (or a suite reference) resolves to ambiguous. Each repeat is
    reported, and the id still resolves (so downstream reference checks are not
    additionally confused by it).
    """
    path = Path(incidents_path)
    if not path.exists():
        errors.append("incidents doc not found: {}".format(path))
        return set()
    # Mask fenced code blocks and HTML comments (offsets preserved) before
    # locating headings, so an example ``## TI-NNN:`` heading inside a fence or a
    # comment is not registered as a real incident definition.
    text = _mask_comments(_mask_fences(path.read_text(encoding="utf-8")))
    defined = set()
    for tid in _INCIDENT_HEADING_RE.findall(text):
        if tid in defined:
            errors.append("duplicate incident definition: {}".format(tid))
        defined.add(tid)
    return defined


def check_incident_refs(refs, defined):
    """Every referenced TI-NNN must be defined in the incidents doc."""
    errors = []
    for ref in sorted(refs):
        if ref not in defined:
            errors.append("incident reference {!r} has no definition in TESTING_INCIDENTS.md".format(ref))
    return errors


def check_incident_fields(incidents_path):
    """Every defined incident must carry the required labelled fields with a value.

    A required field is authored as a bold label followed by its content, for
    example ``**Finding.** <text>``. The label must match the exact ``**Label.**``
    syntax, not merely be a prefix of a longer bold run: a lookalike such as
    ``**Origin story.**`` or ``**Finding of fact.**`` is a different label and does
    NOT satisfy the required ``Origin`` / ``Finding`` field, so a reviewer cannot
    paper over a missing required field with a similarly named one. A missing label
    is a schema error; so is a label that is present but carries no value (a bare
    ``**Origin.**`` with nothing after it), because a reviewer then cannot see what
    the control protects or when it may be retired. The field value is the text
    between the label's closing ``**`` and the next field label, the next paragraph
    break, a heading, or the end of the incident section; when that text is empty
    (or whitespace only) the field is reported as empty rather than silently
    accepted on the strength of the label alone.
    """
    errors = []
    path = Path(incidents_path)
    if not path.exists():
        # Missing-file diagnosis is already reported by defined_incident_ids.
        return errors
    text = path.read_text(encoding="utf-8")
    for incident_id, section in _incident_sections(text):
        for label in REQUIRED_INCIDENT_LABELS:
            # Labels are authored in bold with the exact ``**Label.**`` syntax,
            # e.g. ``**Finding.**`` / ``**Control.**``. The label, an optional
            # surrounding space, and the terminating period must be the entire
            # bold run: a prefix match (``**Origin story.**`` for ``Origin``) is a
            # different label and must not satisfy the required field. Capture the
            # value after the label's closing ``**`` up to the next field label (a
            # line beginning with ``**``), a paragraph break, a heading, or the end
            # of the section, so a present-but-empty field is distinguishable from
            # a populated one.
            match = re.search(
                r"\*\*\s*" + re.escape(label) + r"\s*\.\s*\*\*(.*?)"
                r"(?=\n[ \t]*\n|\n[ \t]*\*\*|\n#{1,6}\s|\Z)",
                section, re.DOTALL)
            if match is None:
                errors.append(
                    "incident {} is missing the {!r} field".format(incident_id, label))
            elif not match.group(1).strip():
                errors.append(
                    "incident {} has an empty {!r} field".format(incident_id, label))
    return errors


def _is_commit_hash(token):
    """True when token looks like an abbreviated or full git commit SHA.

    A commit-like token is a 7-40 character hex string that also carries at least
    one digit, so an all-letter English word that happens to be hex ("defaced")
    is not mistaken for a hash.
    """
    low = token.lower()
    return bool(_COMMIT_HASH_RE.match(low)) and any(ch.isdigit() for ch in low)


def _git_work_tree(path):
    """Return str(path) when path is inside a git work tree, else None.

    Hash validation is feasibility-gated: when git is unavailable or the document
    does not live in a checkout (for example a self-contained test fixture under
    a temporary directory), the Origin-citation check is skipped rather than
    raising or reporting a spurious failure.
    """
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--is-inside-work-tree"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except (OSError, ValueError):
        return None
    if result.returncode != 0 or result.stdout.strip() != "true":
        return None
    return str(path)


def _git_tracks(repo_dir, path):
    """True when path is a tracked file in the git repo at repo_dir.

    Origin citations are validated only for a document the repository actually
    tracks: a throwaway fixture written into (or beneath) some ambient checkout is
    untracked, and its citations refer to a different repository, so validating
    them against the ambient repo would be meaningless. Anchoring on tracked-ness
    keeps the check meaningful for the real, version-controlled document while
    skipping self-contained fixtures.
    """
    try:
        result = subprocess.run(
            ["git", "-C", repo_dir, "ls-files", "--error-unmatch", "--",
             str(path)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except (OSError, ValueError):
        return False
    return result.returncode == 0


def _git_repo_root(path):
    """Return the git work-tree root that contains path, or None.

    Used to separate the *document root* a caller names via ``--root`` from the
    *repository boundary* a path may not escape. A real checkout (or a component
    root such as ``projects/hip-tests``) lives inside a git work tree whose
    top-level is the natural boundary, so documents that legitimately reference
    sibling trees (``../../.github``) resolve. When git is unavailable, or path
    is not inside a work tree (a self-contained temporary fixture), None is
    returned and the caller falls back to treating the supplied root as its own
    boundary.
    """
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--show-toplevel"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except (OSError, ValueError):
        return None
    if result.returncode != 0:
        return None
    top = result.stdout.strip()
    return Path(top).resolve() if top else None


def _git_commit_exists(repo_dir, commit):
    """Tri-state existence of a commit object in repo_dir.

    Returns True when ``commit`` resolves to a commit object, False when git ran
    and reported it does not, and None when existence could not be determined
    (git unavailable). The ``commit`` argument is validated as pure hex by the
    caller, so it is safe to pass as a revision.
    """
    try:
        result = subprocess.run(
            ["git", "-C", repo_dir, "rev-parse", "--verify", "--quiet",
             commit + "^{commit}"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except (OSError, ValueError):
        return None
    return result.returncode == 0


def _strip_citation_marker(token):
    """Return (bare_token, marked): drop a leading external:/unverified: marker.

    A citation the author explicitly flags with an inline ``external:`` or
    ``unverified:`` marker is preserved as-is and not existence-checked, so an
    honest out-of-repository or unverifiable reference is kept without inventing a
    ticket. ``marked`` is True when such a marker was stripped.
    """
    low = token.lower()
    for marker in ORIGIN_CITATION_MARKERS:
        if low.startswith(marker):
            return token[len(marker):].strip(), True
    return token, False


def _scan_commit_hashes(text):
    """Yield (hash, marked) for every commit-like hash occurrence in text.

    A mandatory hex run (group 2) means a lone ``external:`` with no following
    hash matches nothing, so marker prose is not mistaken for a citation. An
    ``external:``/``unverified:`` marker that immediately precedes the hash
    (group 1) flags that occurrence as preserved.
    """
    for match in re.finditer(
            r"(external:|unverified:)?\s*\b([0-9a-f]{7,40})\b",
            text, re.IGNORECASE):
        yield match.group(2), bool(match.group(1))


def _iter_origin_commit_citations(origin):
    """Yield (token, marked) for every commit-like citation in an Origin value.

    Both backtick-quoted citations (``Landed in `0d90b30fb6c```) and bare,
    unquoted hashes (``Landed in 0d90b30fb6c``) are yielded, so dropping the
    backticks cannot let a fabricated hash evade validation. A backticked span is
    not treated as an opaque token: a *descriptive* span such as
    ``commit deadbee1`` is scanned for hash occurrences too, so a fabricated hash
    wrapped in prose-bearing backticks is still validated. An
    ``external:``/``unverified:`` marker is honored two ways - when it opens the
    span (``external: 0d90b30fb6c`` / ``external: commit deadbee1``) it preserves
    every hash the span carries, and when it immediately precedes a hash (in a
    span or in the bare text) it preserves that single occurrence. The quoted
    spans are blanked (offsets preserved) before the bare scan so a hash is never
    yielded twice.
    """
    quoted = []

    def _blank(match):
        quoted.append(match.group(1))
        return " " * len(match.group(0))

    remainder = re.sub(r"`([^`]+)`", _blank, origin)
    for span in quoted:
        body, span_marked = _strip_citation_marker(span.strip())
        for token, marked in _scan_commit_hashes(body):
            yield token, marked or span_marked
    # Bare citations: a marker may immediately precede the hash.
    for token, marked in _scan_commit_hashes(remainder):
        yield token, marked


def check_incident_origin_citations(incidents_path):
    """Validate commit-like citations in each incident's Origin field.

    An incident Origin names where a control was introduced, often citing an
    abbreviated git SHA in backticks (``Landed in `0d90b30fb6c```) or bare in
    prose (``Landed in 0d90b30fb6c``). A fabricated or stale hash makes the
    provenance unfollowable, so each commit-like citation - quoted or unquoted -
    is checked against the repository's object database: a hash that git reports
    does not exist is reported. A citation the author explicitly flags with an
    inline ``external:`` or ``unverified:`` marker is preserved as-is and not
    existence-checked, so an honest out-of-repository or unverifiable reference is
    kept without inventing a ticket.

    The check is feasibility-gated: when the document is not inside a git work
    tree (or git is unavailable) hash existence cannot be verified, so the check
    is skipped rather than reporting a spurious failure.
    """
    errors = []
    path = Path(incidents_path)
    if not path.exists():
        # Missing-file diagnosis is already reported by defined_incident_ids.
        return errors
    repo_dir = _git_work_tree(path.resolve().parent)
    if repo_dir is None:
        return errors
    # Only validate citations for a document the repository actually tracks: an
    # untracked fixture under some ambient checkout cites commits of a different
    # repository, so resolving them here would be meaningless.
    if not _git_tracks(repo_dir, path.resolve()):
        return errors
    text = path.read_text(encoding="utf-8")
    for incident_id, section in _incident_sections(text):
        # The Origin value runs from its bold label to the next field label, a
        # paragraph break, a heading, or the end of the section - the same span
        # the required-field check uses - so a multi-line Origin that cites more
        # than one commit is scanned whole.
        match = re.search(
            r"\*\*\s*Origin\s*\.\s*\*\*(.*?)"
            r"(?=\n[ \t]*\n|\n[ \t]*\*\*|\n#{1,6}\s|\Z)",
            section, re.DOTALL)
        if match is None:
            continue
        origin = match.group(1)
        for token, marked in _iter_origin_commit_citations(origin):
            if not _is_commit_hash(token):
                continue
            if marked:
                # Explicitly external / unverified: preserved, not checked.
                continue
            if _git_commit_exists(repo_dir, token.lower()) is False:
                errors.append(
                    "incident {} Origin cites commit {!r} which does not exist in "
                    "the repository (cite a real landed commit, or mark it "
                    "'external:'/'unverified:')".format(incident_id, token))
    return errors


def _heading_anchors(md_path, cache):
    """Return the set of GitHub-style heading slugs for a Markdown file."""
    resolved = Path(md_path).resolve()
    if resolved in cache:
        return cache[resolved]
    anchors = set()
    if resolved.exists():
        # Derive anchors from the same fence-masked and comment-masked Markdown
        # the link scanner sees, so a ``# Heading`` that lives inside a fenced
        # code block or an HTML comment is shown rather than registered as a real
        # anchor. Inline code spans are deliberately NOT masked here: a heading
        # that merely contains an inline ``code`` span keeps that span's text in
        # its GitHub anchor slug, so masking it would corrupt a legitimate slug.
        text = _mask_comments(_mask_fences(resolved.read_text(encoding="utf-8")))
        # GitHub disambiguates repeated heading slugs by appending ``-1``, ``-2``,
        # ... to the second and later occurrences, so track how many times each
        # base slug has been seen and register the suffixed anchor as well.
        seen = {}
        for heading in _HEADING_RE.findall(text):
            slug = _slugify(heading)
            count = seen.get(slug, 0)
            anchors.add(slug if count == 0 else "{}-{}".format(slug, count))
            seen[slug] = count + 1
    cache[resolved] = anchors
    return anchors


def _heading_visible_text(heading_text):
    """Reduce a heading's Markdown link/image markup to its visible text.

    GitHub derives a heading's anchor slug from the heading's *rendered* text,
    so an inline link ``[label](dest "title")`` contributes only ``label`` and an
    inline image ``![alt](src)`` contributes only ``alt`` - the destination and
    any optional title never appear in the slug. Plain text surrounding a link,
    and any inline ``code`` span a heading carries, are preserved verbatim so a
    heading that merely contains inline code still slugs the same way. A
    destination may carry a balanced pair of parentheses or a parenthesized
    title, so the destination is consumed by a balanced-parenthesis scan rather
    than stopping at the first ``)``; an escaped parenthesis inside the
    destination does not end it early. A malformed opener whose destination never
    closes consumes to end of text, which simply drops the dangling markup from
    the visible text rather than crashing.
    """
    out = []
    pos = 0
    n = len(heading_text)
    for match in _MD_INLINE_LABEL_RE.finditer(heading_text):
        # A later opener may fall inside a destination already consumed by an
        # earlier iteration; skip it so its text is not re-emitted.
        if match.start() < pos:
            continue
        out.append(heading_text[pos:match.start()])
        out.append(match.group(1))  # the visible link label / image alt text
        # Consume the destination: the balanced ``(...)`` that begins right after
        # the matched opening parenthesis, honoring backslash escapes.
        i = match.end()
        depth = 1
        while i < n:
            ch = heading_text[i]
            if ch == "\\":
                i += 2
                continue
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    i += 1
                    break
            i += 1
        pos = i
    out.append(heading_text[pos:])
    return "".join(out)


def _slugify(heading_text):
    """Approximate GitHub's heading-to-anchor slug.

    Reduce any Markdown link/image markup to its visible label/alt text first
    (see ``_heading_visible_text``) so a destination or title never pollutes the
    slug, then lowercase, drop any character that is not a word character,
    whitespace, or hyphen (this strips Markdown punctuation such as backticks and
    colons), then collapse whitespace runs to single hyphens. Inline-code and
    plain heading text are preserved by the visible-text reduction, so a heading
    carrying neither a link nor an image slugs exactly as before.
    """
    text = _heading_visible_text(heading_text)
    text = text.strip().lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"\s+", "-", text)
    return text


def _strip_link_title(target):
    """Return a link destination with any optional CommonMark title removed.

    An inline Markdown link destination may be followed by a title:
    ``[text](dest "title")`` / ``[text](dest 'title')`` / ``[text](dest (title))``.
    The title is descriptive and is not part of the path, so a naive capture of
    everything between the parentheses would make a perfectly valid link fail
    existence resolution. The destination is the leading token; a whitespace-
    separated remainder that is quoted or parenthesized is a title and is dropped.
    An angle-bracketed destination (``<dest>``) is unwrapped. A target with no
    title (including a fragment-only ``#anchor`` or a reference-definition token
    that never carries spaces) is returned stripped but otherwise unchanged.
    """
    stripped = target.strip()
    if not stripped:
        return stripped
    if stripped.startswith("<"):
        end = stripped.find(">")
        if end != -1:
            return stripped[1:end].strip()
    parts = stripped.split(None, 1)
    if len(parts) == 2:
        dest, remainder = parts[0], parts[1].strip()
        if len(remainder) >= 2 and (
                (remainder[0] == '"' and remainder[-1] == '"')
                or (remainder[0] == "'" and remainder[-1] == "'")
                or (remainder[0] == "(" and remainder[-1] == ")")):
            return dest
    return stripped


def _normalize_ref_label(label):
    """Return a CommonMark-normalized reference label for matching.

    CommonMark matches reference labels after stripping leading/trailing
    whitespace, collapsing every internal whitespace run (including newlines) to a
    single space, and case-folding. So ``[Known Gaps]`` and ``[known   gaps]``
    (and a label whose whitespace is split across a line break) name the same
    definition; without the internal collapse a cosmetic whitespace difference
    would wrongly read as two distinct labels.
    """
    return re.sub(r"\s+", " ", label.strip()).lower()


def _unescape_md(text):
    """Resolve CommonMark backslash escapes in a link destination or label.

    CommonMark lets any ASCII-punctuation character be backslash-escaped; the
    backslash is then not literal and the punctuation stands for itself. A
    destination that escapes a parenthesis (``file\\(1\\).md``) must therefore
    resolve to the real on-disk path, not to a path with the backslashes
    embedded. A backslash before any non-punctuation character (a space, a
    letter) is itself literal and is kept, matching CommonMark exactly.
    """
    return re.sub(r"\\([!-/:-@\[-`{-~])", r"\1", text)


def _opener_is_escaped(text, index):
    """True when the Markdown opener ``[`` at ``index`` is backslash-escaped.

    Counts the run of consecutive backslashes immediately preceding ``index``. A
    backslash escapes the character that follows it, and a backslash can itself
    be escaped, so only an ODD-length run leaves a trailing unpaired backslash
    that escapes the opener; an EVEN-length run (including a run of zero) is a
    sequence of escaped-backslash literals that leaves the opener live. This is
    CommonMark's rule: ``\\[t](p)`` is an escaped bracket shown as text, while
    ``\\\\[t](p)`` is an escaped backslash followed by a LIVE link opener. A
    simple "immediately preceded by a backslash" rejection would wrongly treat
    the even-run ``\\\\[`` case as escaped and so silently drop a real link.
    """
    run = 0
    j = index - 1
    while j >= 0 and text[j] == "\\":
        run += 1
        j -= 1
    return run % 2 == 1


def _iter_inline_link_targets(text):
    """Yield the raw ``(...)`` body of every inline Markdown link in ``text``.

    The inline link opener ``[text](`` is matched with a regular expression, but
    the destination/title body is extracted by a balanced-parenthesis scan rather
    than a naive ``[^)]+`` capture. CommonMark permits unescaped parentheses in a
    link destination as long as they are balanced (``[t](a/(v2)/b.md)``) and in a
    parenthesized title (``[t](dest (title))``), so stopping at the first ``)``
    would truncate a perfectly valid link and make it fail existence resolution.

    An opener whose ``[`` is backslash-escaped is literal text, not a link, and is
    skipped. Escaping is decided by the parity of the preceding backslash run (see
    ``_opener_is_escaped``): an odd run escapes the opener (``\\[t](p)``), while an
    even run leaves it live (``\\\\[t](p)`` is an escaped backslash before a real
    link), so an even-length run is not mistaken for an escape.

    For each live opener the scan tracks parenthesis depth from the opening ``(``
    and yields the substring up to the matching close. A backslash escapes the
    next character, so an escaped parenthesis does not change the depth. If the
    parentheses never balance before end of line or end of text the link is
    malformed; the body gathered so far is still yielded so a malformed/broken
    link keeps being reported rather than silently dropped.
    """
    n = len(text)
    for opener in _MD_LINK_OPEN_RE.finditer(text):
        if _opener_is_escaped(text, opener.start()):
            continue
        start = opener.end()      # first character inside the destination parens
        i = start
        depth = 1
        while i < n:
            ch = text[i]
            if ch == "\\":
                i += 2            # escaped character: consume it literally
                continue
            if ch == "\n":
                break             # unbalanced: a link body stays on one line
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        yield text[start:i]


def _check_link_target(doc_name, base, doc_resolved, target, anchors_cache,
                       boundary_root=None):
    """Validate one link target ``local#fragment`` and count resolved references.

    Returns (errors, refs_ok). An empty or external (http(s)/mailto) target is
    preserved and not existence-checked. A local path must resolve on disk; a
    ``#fragment`` must resolve to a heading anchor in the target Markdown file
    (``doc_resolved`` itself for a fragment-only target). An optional CommonMark
    link title (``dest "title"``) is stripped first so a titled link resolves on
    its destination alone.

    A local target is held to the same boundary as every other repository path
    pointer: an absolute target is rejected outright, and a relative target that
    resolves outside ``boundary_root`` (the supplied ``--root`` or, by default,
    the repository root two levels above the component directory) is rejected as
    an escape. A local target must resolve to a file, not a directory (a bare
    directory link exposes no document to read and is held to the same
    file-not-directory rule as every other evidence path). Valid relative links
    and fragment anchors are unaffected.
    """
    errors = []
    refs_ok = 0
    target = _strip_link_title(target)
    if not target or target.startswith(_EXTERNAL_LINK_PREFIXES):
        # Preserve external URLs: not existence-checked.
        return errors, refs_ok
    local, _, fragment = target.partition("#")
    # Resolve CommonMark backslash escapes so a destination that escapes ASCII
    # punctuation (``file\(1\).md``) resolves to the real on-disk path rather
    # than failing with the backslashes embedded.
    local = _unescape_md(local)
    fragment = _unescape_md(fragment)
    if local:
        if Path(local).is_absolute():
            errors.append(
                "{}: link target must be repository-relative: {}".format(
                    doc_name, target))
            return errors, refs_ok
        resolved = (base / local).resolve()
        if boundary_root is not None:
            try:
                resolved.relative_to(boundary_root)
            except ValueError:
                errors.append(
                    "{}: link target escapes repository root: {}".format(
                        doc_name, target))
                return errors, refs_ok
        if not resolved.exists():
            errors.append("{}: broken local link: {}".format(doc_name, target))
            return errors, refs_ok
        if resolved.is_dir():
            errors.append(
                "{}: link target is a directory, not a file: {}".format(
                    doc_name, target))
            return errors, refs_ok
        refs_ok += 1
        anchor_source = resolved
    else:
        # Fragment-only link refers to a heading in this same document.
        anchor_source = doc_resolved
    if fragment and str(anchor_source).endswith(".md"):
        # Only Markdown documents expose heading anchors we can validate.
        if fragment in _heading_anchors(anchor_source, anchors_cache):
            refs_ok += 1
        else:
            errors.append("{}: broken anchor: {}".format(doc_name, target))
    return errors, refs_ok


def _blank_like(text):
    """Return text with every character replaced by a space, newlines preserved.

    Used to blank out code content while keeping character offsets and line
    numbers unchanged, so the surrounding document still scans with the same
    positions and the ``^``-anchored reference-definition regex still sees real
    line starts.
    """
    return "".join("\n" if ch == "\n" else " " for ch in text)


def _mask_inline_code(text):
    """Return text with inline code spans blanked out (offsets preserved).

    A CommonMark inline code span opens with a run of one or more backticks and
    closes with a run of the same length. The span (delimiters and content) is
    replaced with spaces so a ``[text](path)`` or ``[label]: target`` that appears
    inside code is not mistaken for a real link or reference definition. A backtick
    run with no matching closing run is not a code span and is left untouched.
    """
    out = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch != "`":
            out.append(ch)
            i += 1
            continue
        # Measure the opening backtick run.
        j = i
        while j < n and text[j] == "`":
            j += 1
        run = j - i
        # Search for a closing run of exactly the same length.
        k = j
        closed = False
        while k < n:
            if text[k] == "`":
                m = k
                while m < n and text[m] == "`":
                    m += 1
                if m - k == run:
                    out.append(_blank_like(text[i:m]))
                    i = m
                    closed = True
                    break
                k = m
            else:
                k += 1
        if not closed:
            # No closing run: the backticks are literal, not a code span.
            out.append(text[i:j])
            i = j
    return "".join(out)


def _mask_comments(text):
    """Return text with HTML comment regions (``<!-- ... -->``) blanked out.

    Offsets and line numbers are preserved (each comment character becomes a
    space, newlines kept) so the surrounding document still scans with the same
    positions and the ``^``-anchored reference-definition regex still sees real
    line starts. A ``[text](path)`` or a ``# Heading`` hidden inside a comment is
    thus neither linked nor registered as a heading anchor.
    """
    return _HTML_COMMENT_RE.sub(lambda match: _blank_like(match.group(0)), text)


def _mask_fences(text):
    """Return text with fenced code blocks blanked out (offsets preserved).

    Every line inside a fenced block (``` or ~~~) is replaced by spaces of equal
    length (newlines preserved) so a link opener, reference definition, or
    heading inside a code block is shown rather than interpreted. Inline code
    spans are left intact; mask them separately with ``_mask_inline_code`` where
    a link opener (but not a heading slug) must be hidden.
    """
    lines = text.split("\n")
    out = []
    fence = None  # (marker_char, length) while inside a fenced block
    for line in lines:
        match = _CODE_FENCE_RE.match(line)
        if fence is None:
            if match:
                fence = (match.group(2)[0], len(match.group(2)))
                out.append(_blank_like(line))
            else:
                out.append(line)
        else:
            # Inside a fence: blank every line. A fence of the same marker that is
            # at least as long and carries no trailing info string closes it.
            out.append(_blank_like(line))
            if match:
                marker_char, length = match.group(2)[0], len(match.group(2))
                if (marker_char == fence[0] and length >= fence[1]
                        and not match.group(3).strip()):
                    fence = None
    return "\n".join(out)


def _mask_code(text):
    """Return text with fenced code blocks, HTML comments, and inline code spans
    blanked out.

    Code and comment content is replaced by spaces of equal length (newlines
    preserved) so character offsets and line numbers are unchanged, but no
    ``[``, ``(``, or backtick inside code - nor any link-like text inside an HTML
    comment - can be mistaken for a Markdown link, link opener, or reference
    definition. Real links and definitions outside code/comments are untouched.
    Fenced blocks are masked first (so backticks that open/close a fence are not
    re-interpreted as inline code), then HTML comments (so a comment wrapping a
    backtick run is not re-read as a code span), then inline spans on the
    remainder.
    """
    return _mask_inline_code(_mask_comments(_mask_fences(text)))


def check_markdown_links(md_path, anchors_cache, boundary_root=None):
    """Validate inline and reference-style links and in-page anchors in md_path.

    Returns (errors, refs_ok). Inline ``[text](target)`` links and reference-style
    ``[text][label]`` / collapsed ``[label][]`` links (resolved through their
    ``[label]: target`` definitions) are both validated the same way; external
    http(s)/mailto URLs are preserved and not existence-checked, and fragment
    anchors resolve against the headings of the target Markdown document (the file
    itself for fragment-only links). A reference-style usage with no matching
    definition, and a definition that nothing references, are both reported so the
    two link forms stay consistent.

    Every local link target is held to ``boundary_root`` (the supplied ``--root``
    or, by default, the repository root two levels above the component directory):
    an absolute target or a relative target that resolves outside the boundary is
    rejected exactly like any other repository path pointer, while valid relative
    links and fragment anchors continue to resolve.

    Link-like text inside inline code spans and fenced code blocks is shown, not
    linked: such regions are masked out before scanning so a ``[text](path)`` in a
    code example is neither validated nor reported, while real links and reference
    definitions outside code are unaffected.
    """
    errors = []
    refs_ok = 0
    path = Path(md_path)
    if not path.exists():
        return ["markdown doc not found: {}".format(path)], 0
    base = path.resolve().parent
    doc_resolved = path.resolve()
    # Mask inline code spans and fenced code blocks so link-like text inside code
    # is not scanned; offsets and line starts are preserved for the ^-anchored
    # reference-definition regex.
    text = _mask_code(path.read_text(encoding="utf-8"))

    # Reference-style link definitions; Markdown reference labels are
    # case-insensitive, so normalize to lower case for matching. A label defined
    # more than once (case-insensitively) is ambiguous: CommonMark keeps the
    # first definition and silently ignores the rest, so a copy/pasted or
    # conflicting redefinition would otherwise be dropped without a word. The
    # first definition is preserved as authoritative and each later redefinition
    # is reported rather than silently overwriting it.
    definitions = {}
    for match in _MD_REF_DEF_RE.finditer(text):
        label = _normalize_ref_label(match.group(1))
        if label in definitions:
            errors.append(
                "{}: duplicate reference-style link definition [{}] "
                "(Markdown reference labels are case-insensitive; the first "
                "definition is authoritative)".format(path.name, label))
            continue
        definitions[label] = match.group(2).strip()

    # Inline links. The destination/title body is extracted with a balanced-
    # parenthesis scan so a destination or title that legitimately contains
    # parentheses is resolved whole instead of truncated at the first ``)``.
    for raw in _iter_inline_link_targets(text):
        errs, got = _check_link_target(
            path.name, base, doc_resolved, raw.strip(), anchors_cache,
            boundary_root)
        errors.extend(errs)
        refs_ok += got

    # Reference-style usages, resolved through their definitions. A collapsed
    # ``[label][]`` names its label in the text part; a full ``[text][label]``
    # names it in the label part. A usage whose opening ``[`` is backslash-escaped
    # is literal text, not a link, and is skipped; escaping is decided by the same
    # preceding-backslash-run parity as inline openers (an odd run escapes it, an
    # even run - including none - leaves it live).
    used_labels = set()
    for match in _MD_REF_USE_RE.finditer(text):
        if _opener_is_escaped(text, match.start()):
            continue
        text_part, label_part = match.group(1), match.group(2)
        label = _normalize_ref_label(label_part if label_part.strip() else text_part)
        used_labels.add(label)
        if label not in definitions:
            errors.append(
                "{}: reference-style link [{}] has no matching definition".format(
                    path.name, label))
            continue
        errs, got = _check_link_target(
            path.name, base, doc_resolved, definitions[label], anchors_cache,
            boundary_root)
        errors.extend(errs)
        refs_ok += got

    # A reference label that is defined but never used is dead metadata and is
    # reported so the two link forms stay consistent.
    for label in sorted(definitions):
        if label not in used_labels:
            errors.append(
                "{}: reference-style link definition [{}] is never used".format(
                    path.name, label))

    return errors, refs_ok


def _inventory_claimed_cadences(inventory):
    """Return the set of ``execution.cadence`` values any inventory suite claims.

    These are the cadences the inventory actually backs with a suite (for the
    live documents, ``pr`` and ``on_demand``). A malformed or missing cadence is
    simply not added; cadence shape itself is validated by ``check_inventory``.
    """
    cadences = set()
    if not isinstance(inventory, dict):
        return cadences
    for suite in inventory.get("suites") or []:
        if not isinstance(suite, dict):
            continue
        execution = suite.get("execution")
        if isinstance(execution, dict):
            cadence = execution.get("cadence")
            if isinstance(cadence, str) and cadence.strip():
                cadences.add(cadence.strip())
    return cadences


def _coverage_informational_cadences(coverage):
    """Return the cadence axis's ``informational_classes`` from the coverage model.

    These are the cadence values the model declares as vocabulary but does not
    audit as a concrete claim (nightly/release). An absent cadence axis or
    ``informational_classes`` list yields the empty set, which makes the
    cadence-prose check a no-op (feasibility-gated).
    """
    informational = set()
    if not isinstance(coverage, dict):
        return informational
    cm = coverage.get("coverage_model")
    if not isinstance(cm, dict):
        return informational
    for axis in cm.get("axes") or []:
        if not isinstance(axis, dict) or axis.get("id") != CADENCE_AXIS_ID:
            continue
        for cls in axis.get("informational_classes") or []:
            if isinstance(cls, str) and cls.strip():
                informational.add(cls.strip())
    return informational


def _h2_section_body(text, title):
    """Return the body of the ``## <title>`` section, or None when it is absent.

    The body runs from just after the heading line to the next level-2 (``## ``)
    heading or end of document. Bounding the scan to one section keeps the
    separate ``## Nightly validation`` / ``## Release validation`` sections -
    which legitimately describe broader external validation - out of the
    per-tier cadence-prose check.
    """
    if not isinstance(text, str):
        return None
    heading = re.compile(
        r"^##[ \t]+" + re.escape(title) + r"[ \t]*$", re.MULTILINE)
    match = heading.search(text)
    if match is None:
        return None
    start = match.end()
    nxt = re.compile(r"^##[ \t]+", re.MULTILINE).search(text, start)
    end = nxt.start() if nxt else len(text)
    return text[start:end]


def check_cadence_prose(testing_text, inventory, coverage):
    """Flag per-tier cadence prose that overclaims an unaudited cadence.

    Within TESTING.md's ``## Testing strategy and layers`` section, each tier's
    build-status/cadence prose must describe only a cadence the inventory
    actually backs. A cadence word the coverage model keeps as declared
    vocabulary but that no inventory suite runs on - the cadence axis's
    ``informational_classes`` (e.g. nightly/release) minus any cadence an
    inventory suite actually claims - must not appear there as a concrete
    per-tier cadence, because it asserts an audited cadence the inventory does not
    support. The separate ``## Nightly validation`` / ``## Release validation``
    sections are out of scope (they describe broader external validation, not a
    per-tier build cadence) and are never scanned.

    The check is feasibility-gated: when the documents name no guarded cadence
    word (the coverage model declares no informational cadence class, or every
    such class is actually claimed by a suite) or the strategy section is absent,
    it is a no-op. Code spans, fenced blocks, and HTML comments are masked before
    scanning, so a cadence word shown in an example is not flagged.

    Returns a list of deterministic error strings.
    """
    errors = []
    if not isinstance(testing_text, str):
        return errors
    informational = _coverage_informational_cadences(coverage)
    claimed = _inventory_claimed_cadences(inventory)
    guarded = {word for word in informational if word not in claimed}
    if not guarded:
        return errors
    body = _h2_section_body(testing_text, CADENCE_STRATEGY_SECTION_TITLE)
    if body is None:
        return errors
    masked = _mask_code(body)
    backed = ", ".join(sorted(claimed)) or "none"
    for word in sorted(guarded):
        if re.search(r"\b" + re.escape(word) + r"\b", masked, re.IGNORECASE):
            errors.append(
                "TESTING.md {!r} section asserts an unaudited per-tier cadence "
                "{!r}: no inventory suite declares that cadence (it is a "
                "coverage-model informational cadence class). Describe the tier "
                "with a cadence the inventory backs ({}), or make a suite declare "
                "{!r} in test_inventory.yaml.".format(
                    CADENCE_STRATEGY_SECTION_TITLE, word, backed, word))
    return errors


def validate(root=None, inventory_path=None, coverage_path=None,
             testing_path=None, incidents_path=None, restructuring_path=None):
    """Run every check.

    Document locations default to ``root`` (or the component directory alongside
    this script when root is None); any per-document override takes precedence.

    Returns (results, counts) where results maps category -> sorted error list
    and counts carries the summary totals.
    """
    base_root = Path(root) if root is not None else HIP_TESTS_DIR
    inventory_path = inventory_path or (base_root / INVENTORY_NAME)
    coverage_path = coverage_path or (base_root / COVERAGE_NAME)
    testing_path = testing_path or (base_root / TESTING_NAME)
    incidents_path = incidents_path or (base_root / INCIDENTS_NAME)
    restructuring_path = restructuring_path or (base_root / RESTRUCTURING_NAME)

    component_dir = Path(inventory_path).resolve().parent
    # Boundary a repository path may not resolve outside of. This separates the
    # document root the caller names from the repository boundary a path may not
    # escape: the default real-repo run (root is None) keeps the repository root
    # two levels above the component directory; an explicit ``--root`` that is a
    # real checkout / component root derives the git repository root (so the
    # documents' legitimate ``../../.github`` references resolve); and an explicit
    # ``--root`` that is a self-contained temporary fixture stays its own boundary
    # so a path may not traverse above it (see ``_derive_boundary_root``).
    repo_root = _derive_boundary_root(root, component_dir, inventory_path)
    results = {}
    counts = {"inventory_records": 0, "coverage_classes": 0,
              "validated_references": 0}

    load_errors = []
    inventory = _load_yaml(inventory_path, load_errors)
    coverage = _load_yaml(coverage_path, load_errors)
    results["load"] = sorted(load_errors)

    suite_ids = set()
    incident_refs = set()
    refs_total = 0

    if inventory is not None:
        inv_errors, suite_ids, inv_refs, inv_counts = check_inventory(
            inventory, component_dir, repo_root)
        results["inventory"] = sorted(inv_errors)
        incident_refs |= inv_refs
        counts["inventory_records"] = inv_counts["records"]
        refs_total += inv_counts["refs"]
        # Reconcile active CMake suite groups against the inventory. This is a
        # feasibility-gated cross-check against the component's catch/ tree, so it
        # runs here (where component_dir is known) rather than inside
        # check_inventory, and is a no-op when no CMake tree is present.
        results["cmake_reconciliation"] = sorted(
            check_cmake_reconciliation(inventory, component_dir))
    else:
        results["inventory"] = ["inventory not loaded"]
        results["cmake_reconciliation"] = []

    if coverage is not None:
        inventory_kind_vocab = set()
        if isinstance(inventory, dict) and isinstance(
                inventory.get("vocabularies"), dict):
            inventory_kind_vocab = _vocab_values(
                inventory["vocabularies"], "kind")
        cov_errors, cov_refs, cov_counts = check_coverage_model(
            coverage, suite_ids, component_dir, inventory_kind_vocab, repo_root)
        # The coverage model and the inventory are a connected set: cross-check
        # that they declare the same identity (schema_version, component).
        cov_errors.extend(check_identity_consistency(inventory, coverage))
        results["coverage_model"] = sorted(cov_errors)
        incident_refs |= cov_refs
        counts["coverage_classes"] = cov_counts["classes"]
        refs_total += cov_counts["refs"]
    else:
        results["coverage_model"] = ["coverage model not loaded"]

    incident_errors = []
    defined = defined_incident_ids(incidents_path, incident_errors)
    ref_errors = check_incident_refs(incident_refs, defined)
    incident_errors.extend(ref_errors)
    incident_errors.extend(check_incident_fields(incidents_path))
    incident_errors.extend(check_incident_origin_citations(incidents_path))
    results["incidents"] = sorted(incident_errors)
    # Count incident references that resolved to a defined incident.
    refs_total += len([ref for ref in incident_refs if ref in defined])

    anchors_cache = {}
    md_errors = []
    for md_path in (testing_path, incidents_path, restructuring_path):
        errs, refs_ok = check_markdown_links(md_path, anchors_cache, repo_root)
        md_errors.extend(errs)
        refs_total += refs_ok
    results["markdown_links"] = sorted(md_errors)

    # Cadence-prose consistency: the per-tier strategy section of TESTING.md may
    # not assert a cadence (nightly/release) that no inventory suite backs. This
    # is a live cross-check between TESTING.md, the coverage model (which declares
    # the informational cadence classes), and the inventory (which declares the
    # cadences actually claimed), and is a no-op when the documents name no
    # guarded cadence word or the strategy section is absent.
    cadence_text = None
    try:
        cadence_text = Path(testing_path).read_text(encoding="utf-8")
    except (FileNotFoundError, UnicodeDecodeError):
        cadence_text = None
    if cadence_text is None:
        results["cadence_prose"] = []
    else:
        results["cadence_prose"] = sorted(
            check_cadence_prose(cadence_text, inventory, coverage))

    counts["validated_references"] = refs_total
    return results, counts


_SUMMARY_KEYS = ("load", "inventory", "cmake_reconciliation", "coverage_model",
                 "incidents", "markdown_links", "cadence_prose")


def _print_summary(results, counts):
    labels = {
        "load": "Document parse",
        "inventory": "Inventory schema / vocab / targets / paths",
        "cmake_reconciliation": "Active CMake suite groups vs inventory",
        "coverage_model": "Coverage model refs / axes / paths",
        "incidents": "Incident ID references and fields",
        "markdown_links": "Local Markdown links and anchors",
        "cadence_prose": "TESTING.md per-tier cadence vs inventory",
    }
    print("HIP testing-docs consistency check")
    total = 0
    for key in _SUMMARY_KEYS:
        errs = results.get(key, [])
        total += len(errs)
        status = "OK" if not errs else "FAIL ({})".format(len(errs))
        print("  {:<42} {}".format(labels[key] + ":", status))
    print("")
    if total:
        print("Problems:")
        for key in _SUMMARY_KEYS:
            for err in results.get(key, []):
                print("  - [{}] {}".format(key, err))
        print("")
        print("RESULT: FAIL - {} reference/consistency problem(s).".format(total))
    else:
        print("RESULT: OK - testing documents are internally consistent.")
        print("  Inventory records:    {}".format(counts["inventory_records"]))
        print("  Coverage classes:     {}".format(counts["coverage_classes"]))
        print("  References validated:  {}".format(counts["validated_references"]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--root", default=None,
                        help="component/repository root holding the testing docs "
                             "(default: the component directory alongside this "
                             "script). A root inside a real checkout uses the git "
                             "repository root as the path boundary, so sibling "
                             "references such as ../../.github still resolve; a "
                             "self-contained temporary root is its own boundary")
    parser.add_argument("--inventory", default=None,
                        help="override path to test_inventory.yaml")
    parser.add_argument("--coverage-model", default=None,
                        help="override path to test_coverage_model.yaml")
    parser.add_argument("--testing", default=None,
                        help="override path to TESTING.md")
    parser.add_argument("--incidents", default=None,
                        help="override path to TESTING_INCIDENTS.md")
    parser.add_argument("--restructuring", default=None,
                        help="override path to TEST_RESTRUCTURING.md")
    parser.add_argument("--json", action="store_true",
                        help="emit the full result as JSON")
    args = parser.parse_args(argv)

    results, counts = validate(
        root=args.root,
        inventory_path=args.inventory,
        coverage_path=args.coverage_model,
        testing_path=args.testing,
        incidents_path=args.incidents,
        restructuring_path=args.restructuring,
    )

    if args.json:
        print(json.dumps({"results": results, "counts": counts},
                         indent=2, sort_keys=True))
    else:
        _print_summary(results, counts)

    failed = any(results[key] for key in results)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
