"""Summarize RDC pytest JUnit XML results into Markdown for GITHUB_STEP_SUMMARY.

Usage: python3 tests/ci/junit_summary.py [results-glob ...]
Writes a Markdown table to stdout; the workflow appends it to the job summary.
"""

import glob
import sys
import xml.etree.ElementTree as ET


def main() -> int:
    patterns = sys.argv[1:] or ["build/test-results-*.xml"]
    files = sorted({f for pattern in patterns for f in glob.glob(pattern)})
    totals = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}
    for path in files:
        try:
            root = ET.parse(path).getroot()
        except (ET.ParseError, OSError):
            continue
        for suite in root.iter("testsuite"):
            for key in totals:
                totals[key] += int(suite.get(key) or 0)

    print("### RDC test results\n")
    if not files:
        print("_No test result files found._")
        return 0
    passed = totals["tests"] - totals["failures"] - totals["errors"] - totals["skipped"]
    print("| Total | Passed | Failed | Errors | Skipped |")
    print("|------:|-------:|-------:|-------:|--------:|")
    print(
        f"| {totals['tests']} | {passed} | {totals['failures']} "
        f"| {totals['errors']} | {totals['skipped']} |"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
