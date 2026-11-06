# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import argparse
import html
import re
import subprocess
import os
import sys

OLD_HEADER = "amdsmi_old.h"
NEW_HEADER = "amdsmi_new.h"
REPORT_FILE = "abi_report.html"
# The x86_64 library is built with ENABLE_ESMI_LIB, which declares the CPU API.
HEADER_PREFIX = b"#define ENABLE_ESMI_LIB 1\n"

# The patterns below match abi-compliance-checker 2.3 (ACC_COMMIT in
# .github/workflows/abi-compliance-check.yml). Refresh the fixtures in abi_check_test.py
# when bumping it. REPORT_MARKER is in every 2.3 report, so a format change fails the
# check instead of reading as a clean report.
REPORT_MARKER = "<th>Removed Symbols</th>"
# abi-compliance-checker 2.3 prints each count as "&nbsp;N&nbsp;" in the section heading.
REMOVED_SYMBOLS = re.compile(r"<h2>Removed Symbols <span class='\w+'>&nbsp;(\d+)&nbsp;")
PROBLEMS = re.compile(
    r"<h2>Problems with ([\w ]+?), (High|Medium|Low) Severity <span class='\w+'>&nbsp;(\d+)&nbsp;"
)
# The checker rates struct growth Low (it assumes the library allocates the struct), but
# amd-smi callers allocate their output structs, so any size change breaks old binaries.
TYPE_OR_SIZE_CHANGE = re.compile(
    r"<span class='ttype'>\w+</span> ([\w:]+)"
    r"|Size of this type has been changed from <b>(\d+)</b> bytes to <b>(\d+)</b> bytes"
    r"|(<h2>)"
)
SOURCE_REPORT = "<div id='SourceTab'"
# Callers size arrays and check limits with these constants when they are built, so a new value
# breaks old binaries although the checker rates it Low. Its binary report already leaves out
# version macros such as AMDSMI_LIB_VERSION_MINOR. String constants are format strings.
# abi-compliance-checker exits 0 when compatible and 1 when it found incompatible changes;
# any other code means it could not compare the headers.
EXIT_BREAK = 1
EXIT_ERROR = 2

CHANGED_CONSTANT = re.compile(
    r"The value of constant <b>(\w+)</b> has been changed from <b>(?!&quot;)(.*?)</b> to <b>(.*?)</b>"
)


def merge_base(base_ref):
    """Where HEAD forked from base_ref, so later changes on base_ref are not blamed on HEAD.

    Comparing against the tip of base_ref instead would report everything base_ref gained
    since the fork as removed, so a missing merge base is an error."""
    try:
        # universal_newlines, not text=: text= is Python 3.7+.
        return subprocess.check_output(
            ["git", "merge-base", base_ref, "HEAD"], universal_newlines=True
        ).strip()
    except subprocess.CalledProcessError:
        print(f"Error: no merge base between {base_ref} and HEAD.")
        return None


def prepare_headers(base_ref, project_path):
    header_rel_path = "include/amd_smi/amdsmi.h"
    header_full_path = os.path.join(project_path, header_rel_path)

    if os.path.exists(header_full_path):
        with open(header_full_path, "rb") as src, open(NEW_HEADER, "wb") as dst:
            dst.write(HEADER_PREFIX + src.read())
    else:
        print(f"Error: New header not found at {header_full_path}")
        return False

    git_path = f"{project_path}/{header_rel_path}"
    print(f"Fetching base version from {base_ref} : {git_path}...")

    try:
        with open(OLD_HEADER, "wb") as f:
            f.write(HEADER_PREFIX)
            f.flush()
            subprocess.check_call(["git", "show", f"{base_ref}:{git_path}"], stdout=f)
    except subprocess.CalledProcessError:
        print(f"Error: could not fetch {git_path} at {base_ref}.")
        return False

    return True


def run_abi_checker(strict=False):
    cmd = [
        "abi-compliance-checker",
        "-lib",
        "amdsmi",
        "-old",
        OLD_HEADER,
        "-new",
        NEW_HEADER,
        "-report-path",
        REPORT_FILE,
    ]

    if strict:
        cmd.append("-strict")

    try:
        return subprocess.call(cmd)
    except FileNotFoundError:
        print("Error: abi-compliance-checker tool not found. Please install it.")
        sys.exit(EXIT_ERROR)


def check_report_content(report_path):
    """True if the report is clean, False if it lists a break, None if it cannot be read."""
    if not os.path.exists(report_path):
        print(f"Error: Report file {report_path} not generated.")
        return None

    with open(report_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    if REPORT_MARKER not in content:
        print(f"Error: {report_path} is not an abi-compliance-checker 2.3 report.")
        return None

    # Binary and source compatibility are reported separately, so most findings appear twice.
    failures = set()
    for count in REMOVED_SYMBOLS.findall(content):
        if int(count) > 0:
            failures.add(f"Removed Symbols: {count}")
    for kind, severity, count in PROBLEMS.findall(content):
        if severity in ("High", "Medium") and int(count) > 0:
            failures.add(f"Problems with {kind}, {severity} Severity: {count}")
    type_name = "unknown type"
    for match in TYPE_OR_SIZE_CHANGE.finditer(content):
        if match.group(4):
            type_name = "unknown type"
        elif match.group(1):
            type_name = match.group(1)
        else:
            failures.add(
                f"Size of {type_name} changed from {match.group(2)} to {match.group(3)} bytes"
            )

    binary_report = content.split(SOURCE_REPORT, 1)[0]
    for name, old, new in CHANGED_CONSTANT.findall(binary_report):
        failures.add(f"Value of {name} changed from {html.unescape(old)} to {html.unescape(new)}")

    for failure in sorted(failures):
        print(f"  [FAIL] {failure}")
    return not failures


def main():
    parser = argparse.ArgumentParser(description="Run ABI Compliance Checker for AMDSMI")
    parser.add_argument(
        "--base-ref",
        required=True,
        help="Branch or commit to compare against; its merge base with HEAD is used",
    )
    parser.add_argument("--head-ref", default="current local workspace", help="Feature branch name")
    parser.add_argument(
        "--project-path", default="projects/amdsmi", help="Path to sub-project from repo root"
    )
    parser.add_argument("--mode", choices=["major", "minor"], default="major", help="Check mode")

    args = parser.parse_args()

    print("=" * 60)
    print(f"ABI Compliance Check: {args.mode.upper()} Mode")
    print(f"Comparing: {args.base_ref}  ->  {args.head_ref}")
    print("=" * 60)

    base = merge_base(args.base_ref)
    if base is None or not prepare_headers(base, args.project_path):
        print("ABI Check ERROR: could not prepare the headers to compare.")
        sys.exit(EXIT_ERROR)

    is_strict = args.mode == "minor"
    returncode = run_abi_checker(strict=is_strict)
    if returncode not in (0, 1):
        print(f"ABI Check ERROR: abi-compliance-checker exited {returncode} without comparing.")
        sys.exit(EXIT_ERROR)
    report_clean = check_report_content(REPORT_FILE)
    if report_clean is None:
        print("ABI Check ERROR: no usable report.")
        sys.exit(EXIT_ERROR)

    if returncode == 0 and report_clean:
        print("\nABI Check PASSED.")
        return
    if returncode == 1:
        print("  [FAIL] abi-compliance-checker reported incompatible changes")
    print(f"\nABI Check FAILED. See {REPORT_FILE} for details.")
    sys.exit(EXIT_BREAK)


if __name__ == "__main__":
    main()
