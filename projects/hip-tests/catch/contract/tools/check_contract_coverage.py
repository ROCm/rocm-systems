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
covered by a contract test nor listed in the allowlist, or if the allowlist has
stale entries. Without --check the script only reports and always exits 0.

Runs anywhere with Python 3.6+ and the standard library only; no ROCm, GPU, or
build is required (pure static analysis), so it is safe as a fast PR gate.
"""

import argparse
import json
import os
import re
import sys

# Resolve repo-relative paths from this script's location so the checker works
# from any working directory. Layout:
#   <repo>/projects/hip-tests/catch/contract/tools/check_contract_coverage.py
#   <repo>/projects/hip/include/hip/hip_runtime_api.h
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CONTRACT_DIR = os.path.dirname(_THIS_DIR)                       # catch/contract
_CATCH_DIR = os.path.dirname(CONTRACT_DIR)                      # catch
_HIP_TESTS_DIR = os.path.dirname(_CATCH_DIR)                    # projects/hip-tests
_PROJECTS_DIR = os.path.dirname(_HIP_TESTS_DIR)                 # projects
REPO_ROOT = os.path.dirname(_PROJECTS_DIR)                      # <repo>

HEADER_PATH = os.path.join(
    REPO_ROOT, "projects", "hip", "include", "hip", "hip_runtime_api.h")
ALLOWLIST_PATH = os.path.join(CONTRACT_DIR, "uncovered_apis.txt")

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


def _strip_comments(text):
    """Remove /* */ and // comments so prototypes in doc examples are not matched."""
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    text = re.sub(r"//[^\n]*", "", text)
    return text


def declared_apis(header_path=HEADER_PATH):
    """Return the set of declared public HIP APIs (the coverage denominator)."""
    with open(header_path, encoding="utf-8", errors="replace") as handle:
        source = _strip_comments(handle.read())
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
    for root, _dirs, files in os.walk(contract_dir):
        for name in files:
            if name.startswith("test_hip_") and name.endswith("_contract.cc"):
                path = os.path.join(root, name)
                with open(path, encoding="utf-8", errors="replace") as handle:
                    for match in _CALL_RE.finditer(handle.read()):
                        called.add(match.group(1))
    return called


def load_allowlist(allowlist_path=ALLOWLIST_PATH):
    """Return {api_name: reason} parsed from the allowlist file."""
    entries = {}
    if not os.path.exists(allowlist_path):
        return entries
    with open(allowlist_path, encoding="utf-8", errors="replace") as handle:
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


def _print_summary(report):
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
    if report["violations"]:
        print("RESULT: FAIL - {} API(s) need a contract test or an "
              "allowlist entry.".format(len(report["violations"])))
        print("See projects/hip-tests/catch/contract/AUTHORING.md for how to add "
              "a contract test,")
        print("or add the API to projects/hip-tests/catch/contract/uncovered_apis.txt "
              "with a reason if it genuinely cannot be covered.")
    elif report["allowlisted_stale"] or report["allowlisted_redundant"]:
        print("RESULT: FAIL - allowlist is out of date (see above).")
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
    args = parser.parse_args(argv)

    if not os.path.exists(args.header):
        sys.stderr.write("error: header not found: {}\n".format(args.header))
        return 2

    report = compute(args.header, args.contract_dir, args.allowlist)

    if args.list_uncovered:
        for name in report["uncovered"]:
            print(name)
    elif args.json:
        printable = {k: v for k, v in report.items() if k != "allowlist"}
        printable["allowlist"] = report["allowlist"]
        print(json.dumps(printable, indent=2, sort_keys=True))
    else:
        _print_summary(report)

    failed = bool(report["violations"] or report["allowlisted_stale"]
                  or report["allowlisted_redundant"])
    if args.check and failed:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
