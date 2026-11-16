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
      ``suites``) with the required mapping/list shapes;
    - every inventory suite, coverage class, and pairwise check carries its
      required keys; suites and coverage classes additionally carry an explicit
      ``owner`` (pairwise checks inherit ownership from the classes they
      reference and are not separately owner-stamped);
    - every inventory ``id`` is unique;
    - every ``kind`` / ``disposition`` / ``gating`` value is within the
      vocabulary the inventory declares for itself (the ``vocabularies:`` block).

  merge targets
    - a suite whose ``disposition`` is ``merge_into`` must name a
      ``disposition_target``; ``migrate`` / ``demote`` / ``retire`` may omit it.
      Any target named by a move disposition must resolve to existing inventory
      id(s); no move may name a missing or self target.

  coverage refs
    - the coverage model carries its top-level identity block (``schema_version``,
      ``component``) with the same shape the inventory requires, and every
      inventory / axis / class id is a non-empty string before it participates
      in duplicate or set-membership checks;
    - every ``current_inventory`` id in the coverage model resolves to an
      inventory suite id;
    - every class / pairwise axis reference is a declared ``axis.class``;
    - the model declares the required coverage dimensions (architecture,
      capability, integration_boundary) as explicit axes;
    - every coverage class carries a non-empty ``evidence`` list whose entries
      are mappings with a non-empty path and claim (paths resolve on disk unless
      marked ``external:``).

  evidence paths
    - every ``source.paths`` entry and every ``evidence.path`` resolves to a
      file on disk (relative to the component directory) unless explicitly
      marked ``external:``.

  incident IDs and fields
    - every ``TI-NNN`` referenced in the YAML resolves to an incident defined by
      a ``## TI-NNN:`` heading in TESTING_INCIDENTS.md;
    - every defined incident carries the required labelled fields (Finding,
      Control, Protects, Origin, Current status, Retirement condition).

  local Markdown refs and anchors
    - every relative Markdown link ``[text](path)`` in TESTING.md,
      TESTING_INCIDENTS.md, and TEST_RESTRUCTURING.md resolves to a file on disk
      (http(s)/mailto URLs are preserved as external and not existence-checked);
    - every in-page ``#anchor`` and ``file.md#anchor`` fragment resolves to a
      heading in the target Markdown document;
    - every ``related_restructuring`` pointer in the inventory resolves.

This is a pure static consistency check: it needs only Python 3 and PyYAML, no
ROCm, GPU, or build, so it is safe as a fast gate. Paths are taken as arguments
(``--root`` or the per-document overrides) or resolved repo-relative from this
script's location; nothing in the documents is executed.

Exit status: non-zero if any check fails, zero otherwise.
"""

import argparse
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

# The labelled fields every incident account must carry so that a reviewer can
# see what a control protects and when it may be retired.
REQUIRED_INCIDENT_LABELS = (
    "Finding", "Control", "Protects", "Origin",
    "Current status", "Retirement condition",
)

# Dispositions that may name a relocation target; when a target is present it
# must resolve to an existing inventory id and never to the record itself.
MOVE_DISPOSITIONS = frozenset({"merge_into", "migrate", "demote", "retire"})

# Dispositions that MUST name a disposition_target. Only ``merge_into`` asserts a
# concrete destination record that has to exist today; ``migrate`` / ``demote`` /
# ``retire`` describe a lifecycle change whose destination may be external or not
# yet an inventory id, so a target is optional for them (but still validated when
# one is given).
REQUIRE_TARGET_DISPOSITIONS = frozenset({"merge_into"})

# A path value starting with this marker is a CI lane defined outside the repo
# and is not existence-checked (TEST_RESTRUCTURING.md WS-1).
EXTERNAL_MARKER = "external:"

_INCIDENT_HEADING_RE = re.compile(r"^##\s+(TI-\d+):", re.MULTILINE)
_INCIDENT_REF_RE = re.compile(r"\bTI-\d+\b")
_MD_LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
_HEADING_RE = re.compile(r"^#{1,6}\s+(.*?)\s*$", re.MULTILINE)
# URLs preserved as external references; existence is not checked for these.
_EXTERNAL_LINK_PREFIXES = ("http://", "https://", "mailto:")


def _load_yaml(path, errors):
    try:
        return yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        errors.append("file not found: {}".format(path))
    except yaml.YAMLError as exc:
        errors.append("{} does not parse as YAML: {}".format(path, exc))
    return None


def _path_exists(component_dir, rel_path):
    """True when rel_path (component-relative) resolves to an existing file."""
    return (Path(component_dir) / rel_path).resolve().exists()


def _nonempty_str(value):
    return isinstance(value, str) and bool(value.strip())


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


def _vocab_values(vocab, name):
    """Return the declared vocabulary list for name as a set (empty if absent)."""
    value = vocab.get(name)
    return set(value) if isinstance(value, list) else set()


def check_inventory(inventory, component_dir):
    """Validate the suite inventory.

    Returns (errors, suite_ids, incident_refs, counts) where counts is a dict
    with 'records' and 'refs' (references successfully resolved).
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

    if "schema_version" in inventory and not _is_scalar(inventory.get("schema_version")):
        errors.append("inventory schema_version must be a non-empty scalar")

    if "component" in inventory and not _nonempty_str(inventory.get("component")):
        errors.append("inventory component must be a non-empty string")

    if "source_of_truth" in inventory:
        sot = inventory.get("source_of_truth")
        if not isinstance(sot, dict) or not sot:
            errors.append("inventory source_of_truth must be a non-empty mapping")

    vocab = inventory.get("vocabularies")
    if "vocabularies" in inventory and not isinstance(vocab, dict):
        errors.append("inventory vocabularies must be a mapping")
    if not isinstance(vocab, dict):
        vocab = {}
    for name in REQUIRED_VOCABULARIES:
        if name not in vocab:
            errors.append("vocabularies block is missing {!r}".format(name))
        elif not isinstance(vocab.get(name), list):
            errors.append("vocabularies {!r} must be a list".format(name))
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

        kind = suite.get("kind")
        if kind is not None and kind_vocab and kind not in kind_vocab:
            errors.append("suite {!r} kind {!r} not in declared vocabulary".format(sid, kind))

        disp = suite.get("disposition")
        if disp is not None and disp_vocab and disp not in disp_vocab:
            errors.append("suite {!r} disposition {!r} not in declared vocabulary".format(sid, disp))

        execution = suite.get("execution")
        if isinstance(execution, dict):
            exec_missing = REQUIRED_EXECUTION_KEYS - set(execution.keys())
            if exec_missing:
                errors.append("suite {!r} execution missing keys: {}".format(sid, sorted(exec_missing)))
            gating = execution.get("gating")
            if gating is not None and gating_vocab and gating not in gating_vocab:
                errors.append("suite {!r} gating {!r} not in declared vocabulary".format(sid, gating))
        elif "execution" in suite:
            errors.append("suite {!r} execution is not a mapping".format(sid))

        # Merge / move targets. Only merge_into must name a disposition_target;
        # the other move dispositions may omit it, but any target they do name is
        # still resolved against the inventory.
        if disp in MOVE_DISPOSITIONS:
            target = suite.get("disposition_target")
            if target is None:
                if disp in REQUIRE_TARGET_DISPOSITIONS:
                    errors.append(
                        "suite {!r} disposition {!r} must name disposition_target".format(sid, disp))
            else:
                targets = target if isinstance(target, list) else [target]
                for tid in targets:
                    if tid == sid:
                        errors.append("suite {!r} {} target is itself".format(sid, disp))
                    elif tid not in suite_ids:
                        errors.append(
                            "suite {!r} {} target {!r} is not an inventory id".format(sid, disp, tid))

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
                    if _path_exists(component_dir, rel):
                        refs_ok += 1
                    else:
                        errors.append("suite {!r} source path missing: {}".format(sid, rel))
        elif "source" in suite:
            errors.append("suite {!r} source is not a mapping".format(sid))

        # Evidence: each entry must be a mapping carrying both a path and a
        # claim; the path must resolve on disk unless marked external.
        for ev in suite.get("evidence", []) or []:
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
            if _path_exists(component_dir, rel):
                refs_ok += 1
            else:
                errors.append("suite {!r} evidence path missing: {}".format(sid, rel))

        # related_restructuring is a structured local Markdown pointer.
        rr = suite.get("related_restructuring")
        if isinstance(rr, str):
            if _path_exists(component_dir, rr):
                refs_ok += 1
            else:
                errors.append("suite {!r} related_restructuring missing: {}".format(sid, rr))

        # Incident references anywhere in the suite record.
        incident_refs.update(_collect_incident_refs(suite))

    return errors, suite_ids, incident_refs, {"records": len(suites), "refs": refs_ok}


def check_coverage_model(model, suite_ids, component_dir):
    """Validate the coverage model against the inventory.

    Returns (errors, incident_refs, counts) where counts has 'classes' and
    'refs'.
    """
    errors = []
    incident_refs = set()
    refs_ok = 0

    if not isinstance(model, dict):
        return (["coverage model did not parse into a mapping"], incident_refs,
                {"classes": 0, "refs": 0})

    # Top-level schema: the coverage model carries the same identity block as the
    # inventory and is validated equivalently before any per-class check runs.
    missing_top = REQUIRED_COVERAGE_TOP_KEYS - set(model.keys())
    if missing_top:
        errors.append(
            "coverage model missing top-level keys: {}".format(sorted(missing_top)))

    if "schema_version" in model and not _is_scalar(model.get("schema_version")):
        errors.append("coverage model schema_version must be a non-empty scalar")

    if "component" in model and not _nonempty_str(model.get("component")):
        errors.append("coverage model component must be a non-empty string")

    cm = model.get("coverage_model")
    if not isinstance(cm, dict):
        errors.append("coverage model is missing the coverage_model block")
        return errors, incident_refs, {"classes": 0, "refs": 0}

    declared_axes = set()
    declared_axis_ids = set()
    for axis in cm.get("axes", []) or []:
        if not isinstance(axis, dict):
            errors.append("coverage model axis is not a mapping: {!r}".format(axis))
            continue
        axis_id = axis.get("id")
        if not _nonempty_str(axis_id):
            errors.append(
                "coverage axis id must be a non-empty string: {!r}".format(axis_id))
            continue
        declared_axis_ids.add(axis_id)
        for cls in axis.get("classes", []) or []:
            declared_axes.add("{}.{}".format(axis_id, cls))

    for req in sorted(REQUIRED_AXES):
        if req not in declared_axis_ids:
            errors.append("coverage model is missing required axis {!r}".format(req))

    def check_axis_refs(owner, refs):
        for ref in refs or []:
            if ref not in declared_axes:
                errors.append("{} axis ref {!r} is not a declared axis.class".format(owner, ref))

    def check_inventory_refs(owner, refs):
        nonlocal refs_ok
        for ref in refs or []:
            if ref not in suite_ids:
                errors.append("{} references unknown inventory id {!r}".format(owner, ref))
            else:
                refs_ok += 1

    seen_class_ids = set()
    classes = cm.get("classes", []) or []
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
        check_axis_refs("class {!r}".format(cid), cls.get("axes"))
        check_inventory_refs("class {!r}".format(cid), cls.get("current_inventory"))
        # Evidence: the list must be non-empty, and each entry must be a mapping
        # carrying both a non-empty path and a non-empty claim; the path must
        # resolve on disk unless marked external. Validated equivalently to the
        # inventory suite evidence.
        evidence = cls.get("evidence")
        if "evidence" in cls and (not isinstance(evidence, list) or not evidence):
            errors.append("class {!r} evidence must be a non-empty list".format(cid))
        for ev in evidence or []:
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
            if _path_exists(component_dir, rel):
                refs_ok += 1
            else:
                errors.append("class {!r} evidence path missing: {}".format(cid, rel))
        incident_refs.update(_collect_incident_refs(cls))

    seen_pair_ids = set()
    for pc in cm.get("pairwise_checks", []) or []:
        if not isinstance(pc, dict):
            errors.append("pairwise check record is not a mapping: {!r}".format(pc))
            continue
        pid = pc.get("id", "<no-id>")
        if pid in seen_pair_ids:
            errors.append("duplicate pairwise check id: {!r}".format(pid))
        seen_pair_ids.add(pid)
        missing = REQUIRED_PAIRWISE_KEYS - set(pc.keys())
        if missing:
            errors.append("pairwise {!r} missing keys: {}".format(pid, sorted(missing)))
        check_axis_refs("pairwise {!r}".format(pid), pc.get("pair"))
        check_inventory_refs("pairwise {!r}".format(pid), pc.get("current_inventory"))
        incident_refs.update(_collect_incident_refs(pc))

    return errors, incident_refs, {"classes": len(classes), "refs": refs_ok}


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
    """Return the set of TI-NNN ids defined by headings in TESTING_INCIDENTS.md."""
    path = Path(incidents_path)
    if not path.exists():
        errors.append("incidents doc not found: {}".format(path))
        return set()
    text = path.read_text(encoding="utf-8")
    return set(_INCIDENT_HEADING_RE.findall(text))


def check_incident_refs(refs, defined):
    """Every referenced TI-NNN must be defined in the incidents doc."""
    errors = []
    for ref in sorted(refs):
        if ref not in defined:
            errors.append("incident reference {!r} has no definition in TESTING_INCIDENTS.md".format(ref))
    return errors


def check_incident_fields(incidents_path):
    """Every defined incident must carry the required labelled fields."""
    errors = []
    path = Path(incidents_path)
    if not path.exists():
        # Missing-file diagnosis is already reported by defined_incident_ids.
        return errors
    text = path.read_text(encoding="utf-8")
    for incident_id, section in _incident_sections(text):
        for label in REQUIRED_INCIDENT_LABELS:
            # Labels are authored in bold, e.g. ``**Finding.**`` / ``**Control.**``.
            if not re.search(r"\*\*\s*" + re.escape(label) + r"\b", section):
                errors.append(
                    "incident {} is missing the {!r} field".format(incident_id, label))
    return errors


def _heading_anchors(md_path, cache):
    """Return the set of GitHub-style heading slugs for a Markdown file."""
    resolved = Path(md_path).resolve()
    if resolved in cache:
        return cache[resolved]
    anchors = set()
    if resolved.exists():
        text = resolved.read_text(encoding="utf-8")
        for heading in _HEADING_RE.findall(text):
            anchors.add(_slugify(heading))
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


def check_markdown_links(md_path, anchors_cache):
    """Validate relative links and in-page anchors in md_path.

    Returns (errors, refs_ok). External http(s)/mailto URLs are preserved and
    not existence-checked. Fragment anchors are resolved against the headings of
    the target Markdown document (the file itself for fragment-only links).
    """
    errors = []
    refs_ok = 0
    path = Path(md_path)
    if not path.exists():
        return ["markdown doc not found: {}".format(path)], 0
    base = path.resolve().parent
    text = path.read_text(encoding="utf-8")
    for raw in _MD_LINK_RE.findall(text):
        target = raw.strip()
        if not target or target.startswith(_EXTERNAL_LINK_PREFIXES):
            # Preserve external URLs: not existence-checked.
            continue
        local, _, fragment = target.partition("#")

        if local:
            resolved = (base / local).resolve()
            if not resolved.exists():
                errors.append("{}: broken local link: {}".format(path.name, target))
                continue
            refs_ok += 1
            anchor_source = resolved
        else:
            # Fragment-only link refers to a heading in this same document.
            anchor_source = path.resolve()

        if fragment:
            # Only Markdown documents expose heading anchors we can validate.
            if str(anchor_source).endswith(".md"):
                if fragment in _heading_anchors(anchor_source, anchors_cache):
                    refs_ok += 1
                else:
                    errors.append(
                        "{}: broken anchor: {}".format(path.name, target))
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
            inventory, component_dir)
        results["inventory"] = sorted(inv_errors)
        incident_refs |= inv_refs
        counts["inventory_records"] = inv_counts["records"]
        refs_total += inv_counts["refs"]
    else:
        results["inventory"] = ["inventory not loaded"]

    if coverage is not None:
        cov_errors, cov_refs, cov_counts = check_coverage_model(
            coverage, suite_ids, component_dir)
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
        errs, refs_ok = check_markdown_links(md_path, anchors_cache)
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
