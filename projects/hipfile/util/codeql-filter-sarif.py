#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

# The file name uses hyphens, as STYLEGUIDE.md asks, so it isn't a valid
# Python module name. Nothing imports this script.
# pylint: disable=invalid-name

"""Remove the CodeQL results that come from how a test framework works.

hipFile's CodeQL workflow writes its results to SARIF files instead of
uploading them, runs this script on them, and then uploads what's left. Each
entry in EXCLUSIONS drops one query's results under one directory, so the
query still checks the rest of hipFile. Only add an entry for a finding that
can't be fixed in the code, and say why.

A result is kept unless its query ID and path both match an entry, so if
CodeQL changes how it writes either one, the results come back rather than
silently disappearing.
"""

import argparse
import json
import sys

# Each entry is (query ID, path prefix from the top of the monorepo).
EXCLUSIONS = (
    # GoogleTest's INSTANTIATE_TEST_SUITE_P defines two static functions that
    # only the initializer of a [[maybe_unused]] static variable uses. The
    # initializer registers the test suite when the program starts, but these
    # queries don't count [[maybe_unused]] as a use, so they report the
    # functions and the variable as dead code.
    ("cpp/unused-static-function", "projects/hipfile/test/"),
    ("cpp/unused-static-variable", "projects/hipfile/test/"),
)


def result_rule(result):
    """Return the query ID of a SARIF result, or None if it has none."""
    rule_id = result.get("ruleId")
    if rule_id is None:
        rule_id = result.get("rule", {}).get("id")
    return rule_id


def result_path(result, artifacts):
    """Return the path of a SARIF result's primary location, or None."""
    locations = result.get("locations") or []
    if not locations:
        return None
    artifact = locations[0].get("physicalLocation", {}).get("artifactLocation", {})
    uri = artifact.get("uri")
    if uri is None:
        # The location can name its file by its index in the run's artifacts.
        index = artifact.get("index")
        if isinstance(index, int) and 0 <= index < len(artifacts):
            uri = artifacts[index].get("location", {}).get("uri")
    return uri


def is_excluded(rule, path):
    """Return whether EXCLUSIONS drops a result with this query ID and path."""
    if rule is None or path is None:
        return False
    return any(rule == r and path.startswith(p) for r, p in EXCLUSIONS)


def filter_sarif(filename):
    """Drop the excluded results from a SARIF file, in place.

    Returns the number of results dropped.
    """
    with open(filename, encoding="utf-8") as f:
        sarif = json.load(f)

    dropped = 0
    for run in sarif.get("runs", []):
        if "results" not in run:
            continue
        artifacts = run.get("artifacts", [])
        kept = []
        for result in run["results"]:
            rule = result_rule(result)
            path = result_path(result, artifacts)
            if is_excluded(rule, path):
                print(f"{filename}: dropped {rule} in {path}")
                dropped += 1
            else:
                kept.append(result)
        run["results"] = kept

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(sarif, f)
    return dropped


def main(argv=None):
    """Filter each SARIF file named on the command line."""
    parser = argparse.ArgumentParser(
        description="Remove the CodeQL results that come from how a test "
        "framework works, editing each SARIF file in place."
    )
    parser.add_argument("sarif", nargs="+", help="SARIF file to filter")
    args = parser.parse_args(argv)

    for filename in args.sarif:
        try:
            dropped = filter_sarif(filename)
        except (OSError, json.JSONDecodeError) as err:
            print(f"{filename}: {err}", file=sys.stderr)
            return 1
        print(f"{filename}: dropped {dropped} result(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
