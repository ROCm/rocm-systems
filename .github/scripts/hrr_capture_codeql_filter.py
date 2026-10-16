#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Keep only the CodeQL results that belong to HRR capture.

The capture sources include rocclr and HIP headers, and CodeQL reports what it
finds in those too. They are not this job's: the job scans capture, not the
HIP runtime. This drops every result whose primary location is outside
projects/clr/hipamd/src/hrr/, writes the rest as a new SARIF file, lists them
on stderr and prints how many there are on stdout.

A result with no primary location is kept: it cannot be shown to be outside
capture, and dropping it would hide it.

Usage: hrr_capture_codeql_filter.py IN.sarif OUT.sarif
"""

import json
import sys

CAPTURE_PREFIX = "projects/clr/hipamd/src/hrr/"


def primary_uri(result):
    try:
        return result["locations"][0]["physicalLocation"]["artifactLocation"]["uri"]
    except (KeyError, IndexError, TypeError):
        return None


def in_capture(result):
    uri = primary_uri(result)
    if uri is None:
        return True
    # CodeQL writes paths relative to the source root, but an absolute
    # file:// URI is valid SARIF too.
    uri = uri.removeprefix("file://")
    return uri.startswith(CAPTURE_PREFIX) or ("/" + CAPTURE_PREFIX) in uri


def filter_sarif(sarif):
    """Filters SARIF in place and returns the results kept."""
    kept = []
    for run in sarif.get("runs", []):
        run["results"] = [r for r in run.get("results", []) if in_capture(r)]
        kept.extend(run["results"])
    return kept


def describe(result):
    uri = primary_uri(result) or "<no location>"
    try:
        line = result["locations"][0]["physicalLocation"]["region"]["startLine"]
        uri = f"{uri}:{line}"
    except (KeyError, IndexError, TypeError):
        pass
    message = result.get("message", {}).get("text", "")
    return f"{uri}: {result.get('ruleId', '?')}: {message}"


def main(argv):
    if len(argv) != 3:
        print(__doc__.strip().splitlines()[-1], file=sys.stderr)
        return 2
    with open(argv[1], encoding="utf-8") as f:
        sarif = json.load(f)
    kept = filter_sarif(sarif)
    with open(argv[2], "w", encoding="utf-8") as f:
        json.dump(sarif, f)
    for result in kept:
        print(describe(result), file=sys.stderr)
    print(len(kept))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
