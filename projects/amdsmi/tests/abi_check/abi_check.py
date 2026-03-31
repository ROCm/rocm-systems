import argparse
import re
import subprocess
import os
import sys

OLD_HEADER = "amdsmi_old.h"
NEW_HEADER = "amdsmi_new.h"
REPORT_FILE = "abi_report.html"


def run_command(cmd, shell=False, stdout=None, stderr=None):
    try:
        print(f"Running: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
        subprocess.check_call(cmd, shell=shell, stdout=stdout, stderr=stderr)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        return False
    return True


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

    found_failure = False

    removed_symbols_pattern = re.search(r"Removed Symbols.*?\((\d+)\)", content)
    if removed_symbols_pattern:
        count = int(removed_symbols_pattern.group(1))
        if count > 0:
            print(f"  [FAIL] Removed Symbols detected: {count}")
            found_failure = True

    data_type_problems_pattern = re.search(r"Problems with Data Types.*?\((\d+)\)", content)
    if data_type_problems_pattern:
        count = int(data_type_problems_pattern.group(1))
        if count > 0:
            print(f"  [FAIL] Data Type Problems detected: {count}")
            found_failure = True

    if "Verdict" in content and "Incompatible" in content:
        print("  [FAIL] Overall Verdict: Incompatible")
        found_failure = True

    return not found_failure


def main():
    parser = argparse.ArgumentParser(description="Run ABI Compliance Checker for AMDSMI")
    parser.add_argument(
        "--base-ref", required=True, help="Git reference to compare against (e.g. origin/develop)"
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

    if not prepare_headers(args.base_ref, args.project_path):
        print("Failed to prepare headers.")
        sys.exit(1)

    is_strict = args.mode == "minor"
    tool_success = run_abi_checker(strict=is_strict)
    report_clean = check_report_content(REPORT_FILE)

    if not tool_success and not report_clean:
        print(f"\nABI Check FAILED. See {REPORT_FILE} for details.")
        sys.exit(1)
    elif not tool_success and report_clean:
        print(f"\nABI Check finished with warnings. See {REPORT_FILE}.")
        if is_strict:
            sys.exit(1)
    else:
        print("\nABI Check PASSED.")


if __name__ == "__main__":
    main()
