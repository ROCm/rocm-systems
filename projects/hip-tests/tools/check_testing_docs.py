#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT
"""Deterministic consistency checker for the HIP testing documentation set.

Validates the four hand-maintained testing documents under
``projects/hip-tests`` as a connected set:

  * ``test_inventory.yaml``        - the suite inventory
  * ``test_coverage_model.yaml``   - the intended-coverage model
  * ``TESTING_INCIDENTS.md``       - the test-control rationale / incident log
  * ``TEST_RESTRUCTURING.md``      - the restructuring roadmap (proposal)

The checks implement the acceptance criteria that TEST_RESTRUCTURING.md states
for itself (workstreams WS-1 and WS-2):

  schema keys / vocabularies
    - every inventory suite, coverage class, and pairwise check carries its
      required keys;
    - every inventory ``id`` is unique;
    - every ``kind`` / ``disposition`` / ``gating`` value is within the
      vocabulary the inventory declares for itself (the ``vocabularies:`` block).

  merge targets
    - a suite whose ``disposition`` proposes a move (merge_into / migrate /
      demote / retire) must name a ``disposition_target`` of existing inventory
      id(s); no move may name a missing or self target.

  coverage refs
    - every ``current_inventory`` id in the coverage model resolves to an
      inventory suite id;
    - every class / pairwise axis reference is a declared ``axis.class``.

  evidence paths
    - every ``source.paths`` entry and every ``evidence.path`` resolves to a
      file on disk (relative to the component directory) unless explicitly
      marked ``external:``.

  incident IDs
    - every ``TI-NNN`` referenced in the YAML resolves to an incident defined by
      a ``## TI-NNN:`` heading in TESTING_INCIDENTS.md.

  local Markdown refs
    - every relative Markdown link ``[text](path)`` in TESTING_INCIDENTS.md and
      TEST_RESTRUCTURING.md resolves to a file on disk (http(s) and in-page
      ``#anchor`` links are ignored);
    - every ``related_restructuring`` pointer in the inventory resolves.

This is a pure static consistency check: it needs only Python 3 and PyYAML, no
ROCm, GPU, or build, so it is safe as a fast PR gate. Paths are taken as
arguments (or resolved repo-relative from this script's location); nothing in
the documents is executed.

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

INVENTORY_PATH = HIP_TESTS_DIR / "test_inventory.yaml"
COVERAGE_PATH = HIP_TESTS_DIR / "test_coverage_model.yaml"
INCIDENTS_PATH = HIP_TESTS_DIR / "TESTING_INCIDENTS.md"
RESTRUCTURING_PATH = HIP_TESTS_DIR / "TEST_RESTRUCTURING.md"

# Required keys per record type. Derived from the documents as authored; a
# missing key is a schema error.
REQUIRED_SUITE_KEYS = frozenset({
    "id", "name", "kind", "domain", "purpose", "oracle", "source", "metadata",
    "execution", "evidence", "disposition", "disposition_rationale",
})
REQUIRED_EXECUTION_KEYS = frozenset({
    "cadence", "gating", "ctest_labels", "platforms", "backends",
    "prerequisites", "ci_lane",
})
REQUIRED_CLASS_KEYS = frozenset({
    "id", "axes", "risk", "intent", "expected_test_kinds",
    "current_inventory", "evidence", "gap_status",
})
REQUIRED_PAIRWISE_KEYS = frozenset({
    "id", "category", "pair", "status", "priority", "rationale",
    "current_inventory", "gap_status",
})

# Dispositions that propose relocating a record; each must name a target.
MOVE_DISPOSITIONS = frozenset({"merge_into", "migrate", "demote", "retire"})

# A path value starting with this marker is a CI lane defined outside the repo
# and is not existence-checked (TEST_RESTRUCTURING.md WS-1).
EXTERNAL_MARKER = "external:"

_INCIDENT_HEADING_RE = re.compile(r"^##\s+(TI-\d+):", re.MULTILINE)
_INCIDENT_REF_RE = re.compile(r"\bTI-\d+\b")
_MD_LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
_SKIP_LINK_PREFIXES = ("http://", "https://", "mailto:", "#")


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


def check_inventory(inventory, component_dir):
    """Validate the suite inventory. Returns (errors, suite_ids, incident_refs)."""
    errors = []
    suite_ids = set()
    incident_refs = set()

    if not isinstance(inventory, dict):
        return ["inventory did not parse into a mapping"], suite_ids, incident_refs

    vocab = inventory.get("vocabularies") or {}
    for name in ("kind", "disposition", "gating"):
        if name not in vocab:
            errors.append("vocabularies block is missing {!r}".format(name))
    kind_vocab = set(vocab.get("kind", []))
    disp_vocab = set(vocab.get("disposition", []))
    gating_vocab = set(vocab.get("gating", []))

    suites = inventory.get("suites")
    if not isinstance(suites, list) or not suites:
        errors.append("inventory has no suites")
        return errors, suite_ids, incident_refs

    # First pass: collect ids so move-target checks can resolve forward refs.
    for suite in suites:
        sid = suite.get("id") if isinstance(suite, dict) else None
        if sid is not None:
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

        # Merge / move targets.
        if disp in MOVE_DISPOSITIONS:
            target = suite.get("disposition_target")
            if target is None:
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

        # Source paths.
        source = suite.get("source")
        if isinstance(source, dict):
            for rel in source.get("paths", []) or []:
                if isinstance(rel, str) and rel.startswith(EXTERNAL_MARKER):
                    continue
                if isinstance(rel, str) and not _path_exists(component_dir, rel):
                    errors.append("suite {!r} source path missing: {}".format(sid, rel))

        # Evidence paths.
        for ev in suite.get("evidence", []) or []:
            if not isinstance(ev, dict):
                continue
            rel = ev.get("path")
            if not isinstance(rel, str) or rel.startswith(EXTERNAL_MARKER):
                continue
            if not _path_exists(component_dir, rel):
                errors.append("suite {!r} evidence path missing: {}".format(sid, rel))

        # related_restructuring is a structured local Markdown pointer.
        rr = suite.get("related_restructuring")
        if isinstance(rr, str) and not _path_exists(component_dir, rr):
            errors.append("suite {!r} related_restructuring missing: {}".format(sid, rr))

        # Incident references anywhere in the suite record.
        incident_refs.update(_collect_incident_refs(suite))

    return errors, suite_ids, incident_refs


def check_coverage_model(model, suite_ids, component_dir):
    """Validate the coverage model against the inventory. Returns (errors, refs)."""
    errors = []
    incident_refs = set()

    if not isinstance(model, dict):
        return ["coverage model did not parse into a mapping"], incident_refs

    cm = model.get("coverage_model")
    if not isinstance(cm, dict):
        errors.append("coverage model is missing the coverage_model block")
        return errors, incident_refs

    declared_axes = set()
    for axis in cm.get("axes", []) or []:
        axis_id = axis.get("id")
        for cls in axis.get("classes", []) or []:
            declared_axes.add("{}.{}".format(axis_id, cls))

    def check_axis_refs(owner, refs):
        for ref in refs or []:
            if ref not in declared_axes:
                errors.append("{} axis ref {!r} is not a declared axis.class".format(owner, ref))

    def check_inventory_refs(owner, refs):
        for ref in refs or []:
            if ref not in suite_ids:
                errors.append("{} references unknown inventory id {!r}".format(owner, ref))

    seen_class_ids = set()
    for cls in cm.get("classes", []) or []:
        cid = cls.get("id", "<no-id>")
        if cid in seen_class_ids:
            errors.append("duplicate coverage class id: {!r}".format(cid))
        seen_class_ids.add(cid)
        missing = REQUIRED_CLASS_KEYS - set(cls.keys())
        if missing:
            errors.append("class {!r} missing keys: {}".format(cid, sorted(missing)))
        check_axis_refs("class {!r}".format(cid), cls.get("axes"))
        check_inventory_refs("class {!r}".format(cid), cls.get("current_inventory"))
        for ev in cls.get("evidence", []) or []:
            if not isinstance(ev, dict):
                continue
            rel = ev.get("path")
            if not isinstance(rel, str) or rel.startswith(EXTERNAL_MARKER):
                continue
            if not _path_exists(component_dir, rel):
                errors.append("class {!r} evidence path missing: {}".format(cid, rel))
        incident_refs.update(_collect_incident_refs(cls))

    seen_pair_ids = set()
    for pc in cm.get("pairwise_checks", []) or []:
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

    return errors, incident_refs


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


def check_markdown_links(md_path):
    """Every relative Markdown link in md_path must resolve on disk."""
    errors = []
    path = Path(md_path)
    if not path.exists():
        return ["markdown doc not found: {}".format(path)]
    base = path.resolve().parent
    text = path.read_text(encoding="utf-8")
    for target in _MD_LINK_RE.findall(text):
        target = target.strip()
        if not target or target.startswith(_SKIP_LINK_PREFIXES):
            continue
        local = target.split("#", 1)[0]
        if not local:
            continue
        if not (base / local).resolve().exists():
            errors.append("{}: broken local link: {}".format(path.name, target))
    return errors


def validate(inventory_path, coverage_path, incidents_path, restructuring_path):
    """Run every check. Returns a dict of category -> sorted error list."""
    component_dir = Path(inventory_path).resolve().parent
    results = {}

    load_errors = []
    inventory = _load_yaml(inventory_path, load_errors)
    coverage = _load_yaml(coverage_path, load_errors)
    results["load"] = sorted(load_errors)

    suite_ids = set()
    incident_refs = set()

    if inventory is not None:
        inv_errors, suite_ids, inv_refs = check_inventory(inventory, component_dir)
        results["inventory"] = sorted(inv_errors)
        incident_refs |= inv_refs
    else:
        results["inventory"] = ["inventory not loaded"]

    if coverage is not None:
        cov_errors, cov_refs = check_coverage_model(coverage, suite_ids, component_dir)
        results["coverage_model"] = sorted(cov_errors)
        incident_refs |= cov_refs
    else:
        results["coverage_model"] = ["coverage model not loaded"]

    incident_errors = []
    defined = defined_incident_ids(incidents_path, incident_errors)
    incident_errors.extend(check_incident_refs(incident_refs, defined))
    results["incidents"] = sorted(incident_errors)

    md_errors = []
    md_errors.extend(check_markdown_links(incidents_path))
    md_errors.extend(check_markdown_links(restructuring_path))
    results["markdown_links"] = sorted(md_errors)

    return results


def _print_summary(results):
    labels = {
        "load": "Document parse",
        "inventory": "Inventory schema / vocab / targets / paths",
        "coverage_model": "Coverage model refs / paths",
        "incidents": "Incident ID references",
        "markdown_links": "Local Markdown links",
    }
    print("HIP testing-docs consistency check")
    total = 0
    for key in ("load", "inventory", "coverage_model", "incidents", "markdown_links"):
        errs = results.get(key, [])
        total += len(errs)
        status = "OK" if not errs else "FAIL ({})".format(len(errs))
        print("  {:<42} {}".format(labels[key] + ":", status))
    print("")
    if total:
        print("Problems:")
        for key in ("load", "inventory", "coverage_model", "incidents", "markdown_links"):
            for err in results.get(key, []):
                print("  - [{}] {}".format(key, err))
        print("")
        print("RESULT: FAIL - {} reference/consistency problem(s).".format(total))
    else:
        print("RESULT: OK - testing documents are internally consistent.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--inventory", default=INVENTORY_PATH,
                        help="path to test_inventory.yaml (default: component-relative)")
    parser.add_argument("--coverage-model", default=COVERAGE_PATH,
                        help="path to test_coverage_model.yaml (default: component-relative)")
    parser.add_argument("--incidents", default=INCIDENTS_PATH,
                        help="path to TESTING_INCIDENTS.md (default: component-relative)")
    parser.add_argument("--restructuring", default=RESTRUCTURING_PATH,
                        help="path to TEST_RESTRUCTURING.md (default: component-relative)")
    parser.add_argument("--json", action="store_true",
                        help="emit the full result as JSON")
    args = parser.parse_args(argv)

    results = validate(args.inventory, args.coverage_model,
                       args.incidents, args.restructuring)

    if args.json:
        print(json.dumps(results, indent=2, sort_keys=True))
    else:
        _print_summary(results)

    failed = any(results[key] for key in results)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
