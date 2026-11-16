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
    - each required vocabulary (``kind`` / ``disposition`` / ``gating``) is a
      non-empty list of non-empty scalar entries (an empty list is rejected
      because it would vacuously pass every membership check);
    - every suite ``kind`` / ``disposition`` and every ``execution.gating`` is a
      non-empty scalar string that is within the vocabulary the inventory declares
      for itself (the ``vocabularies:`` block); a non-scalar token is a schema
      error rather than an unhashable set-membership crash;
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
      contentless sentinel. An honest unknown origin is still accepted when it
      points at the repository evidence that does exist (no invented tickets). An
      ``incidents`` list, when present, is a non-empty list of ``TI-NNN`` ids, each
      of which must resolve to a defined incident (as any TI reference does).
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
      ``external:``.

  repository path pointers
    - every repository-path-bearing field resolves relative to the component
      directory unless marked ``external:`` (or preserved as a URL): the
      top-level ``source_of_truth`` values (literal paths or globs), each
      suite's ``metadata.config`` and ``metadata.generated_plan``, and
      ``execution.ci_lane.in_repo`` (whose trailing job/step qualifier is
      descriptive prose and is not path-checked).

  incident IDs and fields
    - every ``TI-NNN`` referenced in the YAML resolves to an incident defined by
      a ``## TI-NNN:`` heading in TESTING_INCIDENTS.md;
    - every incident id is defined exactly once: a ``TI-NNN`` carried by two
      ``## TI-NNN:`` headings is a schema error (the id is meant to be stable and
      unique);
    - every defined incident carries the required labelled fields (Finding,
      Control, Protects, Origin, Current status, Retirement condition).

  local Markdown refs and anchors
    - every relative Markdown link ``[text](path)`` in TESTING.md,
      TESTING_INCIDENTS.md, and TEST_RESTRUCTURING.md resolves to a file on disk
      within the boundary root (http(s)/mailto URLs are preserved as external and
      not existence-checked); a local link target that is absolute, or that
      resolves outside the supplied ``--root`` / default repository boundary, is
      rejected exactly like any other repository path pointer, while valid
      relative links and fragment anchors continue to resolve;
    - reference-style links ``[text][label]`` / ``[label][]`` are resolved through
      their ``[label]: target`` definitions and validated like inline links; a
      usage with no matching definition, or a definition that is never used, is
      reported;
    - every in-page ``#anchor`` and ``file.md#anchor`` fragment resolves to a
      heading in the target Markdown document;
    - every ``related_restructuring`` pointer in the inventory, when present, is a
      non-empty string that resolves on disk within the boundary root (an empty or
      non-string value, an absolute path, or a traversal outside the boundary root
      is a schema error).

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
REQUIRED_VOCABULARIES = ("kind", "disposition", "gating")

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
_MD_LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
# Reference-style Markdown links: a usage ``[text][label]`` (or the collapsed
# form ``[label][]``) paired with a ``[label]: target`` definition. The checker
# resolves these through their definitions and validates them exactly like inline
# links, so a reference-style link is a supported, checked form rather than one
# that is silently ignored.
_MD_REF_USE_RE = re.compile(r"\[([^\]]*)\]\[([^\]]*)\]")
_MD_REF_DEF_RE = re.compile(r"^[ ]{0,3}\[([^\]]+)\]:[ \t]*(\S+)", re.MULTILINE)
_HEADING_RE = re.compile(r"^#{1,6}\s+(.*?)\s*$", re.MULTILINE)
# URLs preserved as external references; existence is not checked for these.
_EXTERNAL_LINK_PREFIXES = ("http://", "https://", "mailto:")


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
    try:
        return yaml.load(Path(path).read_text(encoding="utf-8"),
                         Loader=_UniqueKeyLoader)
    except FileNotFoundError:
        errors.append("file not found: {}".format(path))
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


def _glob_matches(component_dir, pattern):
    """True when pattern (component-relative, may use ** and ..) matches a path."""
    try:
        return any(glob.iglob(pattern, root_dir=str(component_dir), recursive=True))
    except (ValueError, OSError):
        return False


def _resolve_repo_path(component_dir, value, repo_root=None):
    """Classify a repository-path-bearing field value.

    Returns one of ``external`` (an external: marker or URL, preserved and not
    checked), ``empty`` (no path token), ``ok`` (the leading path token resolves
    on disk, by literal existence or glob match), or ``missing``. A token carrying
    a glob metacharacter is resolved by recursive glob; otherwise it is
    existence-checked relative to the component directory. The boundary the token
    may not escape is the supplied ``repo_root`` when given (see
    ``_path_location_status``).
    """
    stripped = value.strip()
    if _looks_external(stripped):
        return "external"
    token = _path_token(stripped)
    if not token:
        return "empty"
    location = _path_location_status(component_dir, token, repo_root)
    if location != "ok":
        return location
    if any(ch in token for ch in _GLOB_CHARS):
        return "ok" if _glob_matches(component_dir, token) else "missing"
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


def _history_is_sufficient(value):
    """True when a history narrative establishes real historical context.

    A sufficient narrative is a non-empty string that carries an explicit
    evidence/origin marker (``repository evidence``, a governing-incident
    reference, or the control vocabulary the incident log uses) AND is not a
    bare/generic sentinel (for example a stock "origin not located" line). The
    marker requirement rejects a context-free note that names no evidence; the
    sentinel check rejects a stock placeholder that happens to contain a marker
    word but no actual context.
    """
    if not _nonempty_str(value):
        return False
    if _normalize_history(value) in GENERIC_HISTORY_SENTINELS:
        return False
    low = value.lower()
    return any(marker in low for marker in HISTORY_EVIDENCE_MARKERS)


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
            # ``cadence`` is a free-text scalar and must be a non-empty string.
            if "cadence" in execution and not _nonempty_str(execution.get("cadence")):
                errors.append(
                    "suite {!r} execution.cadence must be a non-empty string".format(sid))
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
                    if rel.startswith(EXTERNAL_MARKER):
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
            if not isinstance(rel, str) or rel.startswith(EXTERNAL_MARKER):
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
        # evidence/origin marker AND not be a contentless sentinel. An honest
        # unknown origin is still accepted as long as it points at the repository
        # evidence that does exist; no ticket is invented to satisfy the rule.
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
        # but insufficient history (a bare/generic placeholder, or a note with no
        # evidence/origin marker) gets a targeted message; an entirely absent one
        # gets the general requirement.
        if not has_incidents and not _history_is_sufficient(history):
            if _nonempty_str(history):
                errors.append(
                    "suite {!r} history is a bare/generic placeholder: it must carry "
                    "an explicit evidence/origin marker (e.g. 'repository evidence') "
                    "with real context, or the suite must name governing "
                    "incidents".format(sid))
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
            if not isinstance(rel, str) or rel.startswith(EXTERNAL_MARKER):
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
    """Yield (incident_id, section_text) for each ## TI-NNN: heading."""
    matches = list(_INCIDENT_HEADING_RE.finditer(text))
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        yield match.group(1), text[start:end]


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
    text = path.read_text(encoding="utf-8")
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
    example ``**Finding.** <text>``. A missing label is a schema error; so is a
    label that is present but carries no value (a bare ``**Origin.**`` with nothing
    after it), because a reviewer then cannot see what the control protects or when
    it may be retired. The field value is the text between the label's closing
    ``**`` and the next field label, the next paragraph break, a heading, or the
    end of the incident section; when that text is empty (or whitespace only) the
    field is reported as empty rather than silently accepted on the strength of the
    label alone.
    """
    errors = []
    path = Path(incidents_path)
    if not path.exists():
        # Missing-file diagnosis is already reported by defined_incident_ids.
        return errors
    text = path.read_text(encoding="utf-8")
    for incident_id, section in _incident_sections(text):
        for label in REQUIRED_INCIDENT_LABELS:
            # Labels are authored in bold, e.g. ``**Finding.**`` / ``**Control.**``.
            # Capture the value after the label's closing ``**`` up to the next
            # field label (a line beginning with ``**``), a paragraph break, a
            # heading, or the end of the section, so a present-but-empty field is
            # distinguishable from a populated one.
            match = re.search(
                r"\*\*\s*" + re.escape(label) + r"\b[^*]*\*\*(.*?)"
                r"(?=\n[ \t]*\n|\n[ \t]*\*\*|\n#{1,6}\s|\Z)",
                section, re.DOTALL)
            if match is None:
                errors.append(
                    "incident {} is missing the {!r} field".format(incident_id, label))
            elif not match.group(1).strip():
                errors.append(
                    "incident {} has an empty {!r} field".format(incident_id, label))
    return errors


def _heading_anchors(md_path, cache):
    """Return the set of GitHub-style heading slugs for a Markdown file."""
    resolved = Path(md_path).resolve()
    if resolved in cache:
        return cache[resolved]
    anchors = set()
    if resolved.exists():
        text = resolved.read_text(encoding="utf-8")
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


def _slugify(heading_text):
    """Approximate GitHub's heading-to-anchor slug.

    Lowercase, drop any character that is not a word character, whitespace, or
    hyphen (this strips Markdown punctuation such as backticks and colons), then
    collapse whitespace runs to single hyphens.
    """
    text = heading_text.strip().lower()
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
    an escape. Valid relative links and fragment anchors are unaffected.
    """
    errors = []
    refs_ok = 0
    target = _strip_link_title(target)
    if not target or target.startswith(_EXTERNAL_LINK_PREFIXES):
        # Preserve external URLs: not existence-checked.
        return errors, refs_ok
    local, _, fragment = target.partition("#")
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
    """
    errors = []
    refs_ok = 0
    path = Path(md_path)
    if not path.exists():
        return ["markdown doc not found: {}".format(path)], 0
    base = path.resolve().parent
    doc_resolved = path.resolve()
    text = path.read_text(encoding="utf-8")

    # Reference-style link definitions; Markdown reference labels are
    # case-insensitive, so normalize to lower case for matching.
    definitions = {}
    for match in _MD_REF_DEF_RE.finditer(text):
        definitions[match.group(1).strip().lower()] = match.group(2).strip()

    # Inline links.
    for raw in _MD_LINK_RE.findall(text):
        errs, got = _check_link_target(
            path.name, base, doc_resolved, raw.strip(), anchors_cache,
            boundary_root)
        errors.extend(errs)
        refs_ok += got

    # Reference-style usages, resolved through their definitions. A collapsed
    # ``[label][]`` names its label in the text part; a full ``[text][label]``
    # names it in the label part.
    used_labels = set()
    for text_part, label_part in _MD_REF_USE_RE.findall(text):
        label = (label_part.strip() or text_part.strip()).lower()
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
    # Boundary a repository path may not resolve outside of. An explicitly
    # supplied ``root`` is its own boundary (an arbitrary temporary root is self-
    # contained, so a path may not traverse above it); the default real-repo run
    # (root is None) keeps the repository root two levels above the component
    # directory as the boundary, which is what permits the documents' legitimate
    # ``../../.github`` references.
    repo_root = Path(root).resolve() if root is not None else _repository_root(
        component_dir)
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
    else:
        results["inventory"] = ["inventory not loaded"]

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

    counts["validated_references"] = refs_total
    return results, counts


_SUMMARY_KEYS = ("load", "inventory", "coverage_model", "incidents", "markdown_links")


def _print_summary(results, counts):
    labels = {
        "load": "Document parse",
        "inventory": "Inventory schema / vocab / targets / paths",
        "coverage_model": "Coverage model refs / axes / paths",
        "incidents": "Incident ID references and fields",
        "markdown_links": "Local Markdown links and anchors",
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
                             "(default: the component directory alongside this script)")
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
