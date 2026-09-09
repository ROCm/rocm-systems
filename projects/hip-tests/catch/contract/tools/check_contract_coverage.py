#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT
"""Contract-test coverage drift checker.

Compares the public HIP runtime APIs declared in
``projects/hip/include/hip/hip_runtime_api.h`` against the APIs actually
exercised by the contract-test sources in ``catch/contract`` and an explicit
allowlist of intentionally-uncovered APIs (``uncovered_apis.txt``).

This is a *name-level* check: it verifies that some contract source calls each
declared API, not that the behavior is correct. That matches the documented
coverage methodology in ``projects/hip-tests/CONTRACT_COVERAGE.md``.

Exit status (with --check): non-zero if there is any declared API that is neither
covered by a contract test nor listed in the allowlist, if the allowlist has stale
entries, if a contract case name violates the structured naming convention, or if
``CONTRACT_COVERAGE.md``'s snapshot block has drifted. Without --check the script
only reports and always exits 0.

Runs anywhere with Python 3.6+ and the standard library only; no ROCm, GPU, or
build is required (pure static analysis), so it is safe as a fast PR gate.
"""

import argparse
import json
from pathlib import Path
import re
import sys

# Resolve repo-relative paths from this script's location so the checker works
# from any working directory. Layout:
#   <repo>/projects/hip-tests/catch/contract/tools/check_contract_coverage.py
#   <repo>/projects/hip/include/hip/hip_runtime_api.h
_THIS_DIR = Path(__file__).resolve().parent
CONTRACT_DIR = _THIS_DIR.parent                                 # catch/contract
_CATCH_DIR = CONTRACT_DIR.parent                                # catch
_HIP_TESTS_DIR = _CATCH_DIR.parent                              # projects/hip-tests
_PROJECTS_DIR = _HIP_TESTS_DIR.parent                           # projects
REPO_ROOT = _PROJECTS_DIR.parent                                # <repo>

HEADER_PATH = REPO_ROOT / "projects" / "hip" / "include" / "hip" / "hip_runtime_api.h"
ALLOWLIST_PATH = CONTRACT_DIR / "uncovered_apis.txt"
COVERAGE_DOC_PATH = _HIP_TESTS_DIR / "CONTRACT_COVERAGE.md"

# Names that are parsed as prototypes but are not public contract targets. These
# are excluded from the denominator entirely (they never count as covered or as
# violations).
NON_API_NAMES = frozenset({
    "hip_init",  # internal initialization entry point, not a public contract target
})

# A declaration looks like:  <ret-type> [*] hipXxx(  ... )
# Anchored at a statement boundary (start-of-line or after ; { }) and allowing the
# usual decorators. This intentionally over-matches slightly then is filtered by
# NON_API_NAMES and the *_t typedef-name exclusion below. Validated to reproduce
# the documented 494-name denominator.
_DECL_RE = re.compile(
    r"(?:^|[;{}])\s*"
    r"(?:HIP_PUBLIC_API\s+|extern\s+|static\s+|inline\s+)*"
    r"(?:const\s+)?[A-Za-z_][A-Za-z0-9_]*\s*\*?\s*"
    r"(hip[A-Za-z0-9_]+)\s*\(",
    re.MULTILINE,
)

# Any hipXxx( token in a contract source counts as exercising that API.
_CALL_RE = re.compile(r"\b(hip[A-Za-z0-9_]+)\s*\(")
_CASE_RE = re.compile(r"HIP_TEST_CASE\s*\(\s*([A-Za-z0-9_]+)\s*\)")
_NAME_SEGMENT_RE = re.compile(r"^(?:[A-Z][A-Za-z0-9]*|[0-9]+D[A-Za-z0-9]*)$")
_SNAPSHOT_RE = re.compile(
    r"<!--\s*contract-coverage-snapshot\s*(.*?)\s*-->",
    re.DOTALL,
)
_SNAPSHOT_FIELDS = (
    "contract_tests",
    "contract_domains",
    "declared_apis",
    "covered_apis",
    "uncovered_allowlisted",
    "coverage_pct",
)


def _strip_comments(text):
    """Remove /* */ and // comments so prototypes in doc examples are not matched."""
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    text = re.sub(r"//[^\n]*", "", text)
    return text


def declared_apis(header_path=HEADER_PATH):
    """Return the set of declared public HIP APIs (the coverage denominator)."""
    source = _strip_comments(Path(header_path).read_text(encoding="utf-8", errors="replace"))
    names = set()
    for match in _DECL_RE.finditer(source):
        name = match.group(1)
        if name.endswith("_t"):
            continue          # struct/enum/typedef names, not functions
        if name in NON_API_NAMES:
            continue
        names.add(name)
    return names


def covered_apis(contract_dir=CONTRACT_DIR):
    """Return the set of HIP APIs called by any contract-test source."""
    called = set()
    for path in Path(contract_dir).rglob("test_hip_*_contract.cc"):
        for match in _CALL_RE.finditer(path.read_text(encoding="utf-8", errors="replace")):
            called.add(match.group(1))
    return called


def load_allowlist(allowlist_path=ALLOWLIST_PATH):
    """Return {api_name: reason} parsed from the allowlist file."""
    entries = {}
    path = Path(allowlist_path)
    if not path.exists():
        return entries
    with path.open(encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            # `APIName  # reason`  or bare `APIName`
            if "#" in line:
                name, reason = line.split("#", 1)
                name, reason = name.strip(), reason.strip()
            else:
                name, reason = line, ""
            if name:
                entries[name] = reason
    return entries


def contract_test_counts(contract_dir=CONTRACT_DIR):
    """Return (test_count, domain_count) for contract source files."""
    domains = set()
    tests = 0
    for path in Path(contract_dir).rglob("test_hip_*_contract.cc"):
        domains.add(path.parent.name)
        tests += path.read_text(encoding="utf-8", errors="replace").count("HIP_TEST_CASE(")
    return tests, len(domains)


def _iter_contract_cases(contract_dir=CONTRACT_DIR):
    """Yield (path, case_name) for every contract HIP_TEST_CASE."""
    root = Path(contract_dir)
    for path in root.rglob("test_hip_*_contract.cc"):
        text = path.read_text(encoding="utf-8", errors="replace")
        for match in _CASE_RE.finditer(text):
            yield path, match.group(1)


def _case_name_violation(name):
    """Return a naming violation reason, or None when the case name is valid."""
    parts = name.split("_")
    if len(parts) != 5:
        return "expected 5 underscore-separated segments"
    if parts[0] != "Contract":
        return "expected first segment to be Contract"
    for part in parts[1:]:
        if not _NAME_SEGMENT_RE.match(part):
            return "segment {!r} is not PascalCase".format(part)
    if not parts[2].startswith("Hip"):
        return "subject segment {!r} must start with Hip".format(parts[2])
    return None


def naming_violations(contract_dir=CONTRACT_DIR):
    """Return structured contract-case naming violations."""
    root = Path(contract_dir)
    violations = []
    for path, name in _iter_contract_cases(root):
        reason = _case_name_violation(name)
        if reason:
            violations.append("{}: {} ({})".format(path.relative_to(root), name, reason))
    return sorted(violations)


def load_coverage_snapshot(doc_path=COVERAGE_DOC_PATH):
    """Return the machine-readable CONTRACT_COVERAGE.md snapshot, or an error."""
    path = Path(doc_path)
    if not path.exists():
        return {}, ["coverage doc not found: {}".format(path)]
    text = path.read_text(encoding="utf-8", errors="replace")
    match = _SNAPSHOT_RE.search(text)
    if not match:
        return {}, ["coverage doc is missing the contract-coverage-snapshot block"]

    snapshot = {}
    errors = []
    for raw in match.group(1).splitlines():
        line = raw.strip()
        if not line:
            continue
        if ":" not in line:
            errors.append("malformed snapshot line: {}".format(line))
            continue
        name, value = line.split(":", 1)
        name, value = name.strip(), value.strip()
        if name not in _SNAPSHOT_FIELDS:
            errors.append("unknown snapshot field: {}".format(name))
            continue
        snapshot[name] = value

    missing = [name for name in _SNAPSHOT_FIELDS if name not in snapshot]
    if missing:
        errors.append("snapshot is missing field(s): {}".format(", ".join(missing)))
    return snapshot, errors


def _snapshot_value(name, value):
    if name == "coverage_pct":
        return float(value)
    return int(value)


def snapshot_drift(report, contract_dir=CONTRACT_DIR, doc_path=COVERAGE_DOC_PATH):
    """Return human-readable CONTRACT_COVERAGE.md snapshot drift messages."""
    snapshot, errors = load_coverage_snapshot(doc_path)
    if errors:
        return errors

    contract_tests, contract_domains = contract_test_counts(contract_dir)
    expected = {
        "contract_tests": contract_tests,
        "contract_domains": contract_domains,
        "declared_apis": report["declared_count"],
        "covered_apis": report["covered_count"],
        "uncovered_allowlisted": report["uncovered_count"] - len(report["violations"]),
        "coverage_pct": report["coverage_pct"],
    }

    drift = []
    for name in _SNAPSHOT_FIELDS:
        try:
            actual = _snapshot_value(name, snapshot[name])
        except ValueError:
            drift.append("{} has non-numeric snapshot value: {}".format(name, snapshot[name]))
            continue
        if actual != expected[name]:
            drift.append("{} is stale: doc has {}, computed {}".format(
                name, actual, expected[name]))
    return drift


def compute(header_path=HEADER_PATH, contract_dir=CONTRACT_DIR,
            allowlist_path=ALLOWLIST_PATH):
    """Compute the coverage report as a dict of sorted lists / counts."""
    declared = declared_apis(header_path)
    covered = covered_apis(contract_dir) & declared
    allow = load_allowlist(allowlist_path)
    allow_names = set(allow)

    uncovered = declared - covered
    violations = sorted(uncovered - allow_names)      # uncovered & not justified
    stale = sorted(allow_names - declared)            # allowlisted but not declared
    redundant = sorted(allow_names & covered)         # allowlisted but now covered

    return {
        "declared_count": len(declared),
        "covered_count": len(covered),
        "uncovered_count": len(uncovered),
        "coverage_pct": round(100.0 * len(covered) / len(declared), 1) if declared else 0.0,
        "uncovered": sorted(uncovered),
        "violations": violations,
        "allowlisted_stale": stale,
        "allowlisted_redundant": redundant,
        "allowlist": allow,
    }


def _print_summary(report, doc_drift, name_violations):
    print("HIP contract-test coverage")
    print("  declared public APIs : {}".format(report["declared_count"]))
    print("  covered by a test    : {}".format(report["covered_count"]))
    print("  uncovered            : {}".format(report["uncovered_count"]))
    print("  coverage             : {}%".format(report["coverage_pct"]))
    print("")
    allow = report["allowlist"]
    if report["uncovered"]:
        print("Uncovered APIs (allowlisted unless marked VIOLATION):")
        viol = set(report["violations"])
        for name in report["uncovered"]:
            if name in viol:
                print("  VIOLATION  {}  (no test, not allowlisted)".format(name))
            else:
                reason = allow.get(name, "")
                print("  allowed    {}  # {}".format(name, reason))
        print("")
    if report["allowlisted_stale"]:
        print("Stale allowlist entries (API no longer declared - remove them):")
        for name in report["allowlisted_stale"]:
            print("  {}".format(name))
        print("")
    if report["allowlisted_redundant"]:
        print("Redundant allowlist entries (a test now exists - remove them):")
        for name in report["allowlisted_redundant"]:
            print("  {}".format(name))
        print("")
    if doc_drift:
        print("CONTRACT_COVERAGE.md snapshot drift:")
        for item in doc_drift:
            print("  {}".format(item))
        print("")
    if name_violations:
        print("Contract case naming violations:")
        for item in name_violations:
            print("  {}".format(item))
        print("Expected: Contract_<DomainPascal>_<Subject>_<Scenario>_<ExpectedOutcome>")
        print("")
    if report["violations"]:
        print("RESULT: FAIL - {} API(s) need a contract test or an "
              "allowlist entry.".format(len(report["violations"])))
        print("See projects/hip-tests/catch/contract/AUTHORING.md for how to add "
              "a contract test,")
        print("or add the API to projects/hip-tests/catch/contract/uncovered_apis.txt "
              "with a reason if it genuinely cannot be covered.")
    elif report["allowlisted_stale"] or report["allowlisted_redundant"]:
        print("RESULT: FAIL - allowlist is out of date (see above).")
    elif doc_drift:
        print("RESULT: FAIL - CONTRACT_COVERAGE.md snapshot is out of date (see above).")
    elif name_violations:
        print("RESULT: FAIL - contract case names do not match the required convention.")
    else:
        print("RESULT: OK - every declared API is covered or justifiably allowlisted.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--check", action="store_true",
                        help="exit non-zero on any violation or stale/redundant "
                             "allowlist entry (for CI)")
    parser.add_argument("--list-uncovered", action="store_true",
                        help="print only the uncovered API names, one per line")
    parser.add_argument("--json", action="store_true",
                        help="emit the full report as JSON")
    parser.add_argument("--header", default=HEADER_PATH,
                        help="path to hip_runtime_api.h (default: repo-relative)")
    parser.add_argument("--contract-dir", default=CONTRACT_DIR,
                        help="path to catch/contract (default: repo-relative)")
    parser.add_argument("--allowlist", default=ALLOWLIST_PATH,
                        help="path to uncovered_apis.txt (default: repo-relative)")
    parser.add_argument("--coverage-doc", default=COVERAGE_DOC_PATH,
                        help="path to CONTRACT_COVERAGE.md (default: repo-relative)")
    args = parser.parse_args(argv)

    if not Path(args.header).exists():
        parser.error("header not found: {}".format(args.header))

    report = compute(args.header, args.contract_dir, args.allowlist)
    doc_drift = snapshot_drift(report, args.contract_dir, args.coverage_doc)
    name_violations = naming_violations(args.contract_dir)

    if args.list_uncovered:
        for name in report["uncovered"]:
            print(name)
    elif args.json:
        printable = {k: v for k, v in report.items() if k != "allowlist"}
        printable["allowlist"] = report["allowlist"]
        printable["coverage_doc_drift"] = doc_drift
        printable["naming_violations"] = name_violations
        print(json.dumps(printable, indent=2, sort_keys=True))
    else:
        _print_summary(report, doc_drift, name_violations)

    failed = bool(report["violations"] or report["allowlisted_stale"]
                  or report["allowlisted_redundant"] or doc_drift
                  or name_violations)
    if args.check and failed:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
