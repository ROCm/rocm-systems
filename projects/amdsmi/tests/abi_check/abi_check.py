# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import argparse
import re
import subprocess
import os
import sys

OLD_HEADER = "amdsmi_old.h"
NEW_HEADER = "amdsmi_new.h"
REPORT_FILE = "abi_report.html"

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
)


def run_command(cmd, shell=False, stdout=None, stderr=None):
    try:
        print(f"Running: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
        subprocess.check_call(cmd, shell=shell, stdout=stdout, stderr=stderr)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        return False
    return True


def merge_base(base_ref):
    """Where HEAD forked from base_ref, so later changes on base_ref are not blamed on HEAD."""
    try:
        return subprocess.check_output(["git", "merge-base", base_ref, "HEAD"], text=True).strip()
    except subprocess.CalledProcessError:
        print(f"Warning: no merge base between {base_ref} and HEAD; comparing against {base_ref}.")
        return base_ref


def prepare_headers(base_ref, project_path):
    header_rel_path = "include/amd_smi/amdsmi.h"
    header_full_path = os.path.join(project_path, header_rel_path)

    if os.path.exists(header_full_path):
        if not run_command(["cp", header_full_path, NEW_HEADER]):
            return False
    else:
        print(f"Error: New header not found at {header_full_path}")
        return False

    git_path = f"{project_path}/{header_rel_path}"
    print(f"Fetching base version from {base_ref} : {git_path}...")

    try:
        with open(OLD_HEADER, "w") as f:
            subprocess.check_call(["git", "show", f"{base_ref}:{git_path}"], stdout=f)
    except subprocess.CalledProcessError:
        print(f"Warning: Could not fetch old header from {base_ref}. It might be a new file.")
        open(OLD_HEADER, "w").close()

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
        subprocess.check_call(cmd)
        return True
    except subprocess.CalledProcessError:
        return False
    except FileNotFoundError:
        print("Error: abi-compliance-checker tool not found. Please install it.")
        sys.exit(1)


def check_report_content(report_path):
    if not os.path.exists(report_path):
        print(f"Error: Report file {report_path} not generated.")
        return False

    with open(report_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

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
        if match.group(1):
            type_name = match.group(1)
        else:
            failures.add(
                f"Size of {type_name} changed from {match.group(2)} to {match.group(3)} bytes"
            )

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
    if not prepare_headers(base, args.project_path):
        print("Failed to prepare headers.")
        sys.exit(1)

    is_strict = args.mode == "minor"
    tool_success = run_abi_checker(strict=is_strict)
    report_clean = check_report_content(REPORT_FILE)

    if tool_success and report_clean:
        print("\nABI Check PASSED.")
        return
    if not tool_success:
        print("  [FAIL] abi-compliance-checker reported incompatible changes or could not run")
    print(f"\nABI Check FAILED. See {REPORT_FILE} for details.")
    sys.exit(1)


if __name__ == "__main__":
    main()
