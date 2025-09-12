#!/usr/bin/env python3
##############################################################################
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
##############################################################################

"""
CI script to upload coverage XML files to CDash
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def detect_repo_structure():
    """Detect if in monorepo or standalone structure"""
    cwd = Path.cwd()

    if "projects/rocprofiler-compute" in str(cwd):
        parts = cwd.parts
        idx = parts.index("rocprofiler-compute")
        if idx > 0 and parts[idx - 1] == "projects":
            monorepo_root = Path(*parts[: idx - 1])
            project_root = monorepo_root / "projects" / "rocprofiler-compute"
            return True, monorepo_root, project_root

    if (cwd / "CMakeLists.txt").exists():
        with open(cwd / "CMakeLists.txt") as f:
            content = f.read()
            if (
                "project(rocprofiler-compute" in content
                or "project(\n    rocprofiler-compute" in content
            ):
                return False, None, cwd

    return False, None, cwd


def create_ctest_script(args):
    """Generate a CTest script for uploading coverage data"""

    script_content = f"""
cmake_minimum_required(VERSION 3.19)

set(CTEST_PROJECT_NAME "{args.project_name}")
set(CTEST_NIGHTLY_START_TIME "01:00:00 UTC")
set(CTEST_DROP_SITE_CDASH TRUE)

if(CMAKE_VERSION VERSION_GREATER 3.14)
    set(CTEST_SUBMIT_URL "{args.submit_url}")
else()
    set(CTEST_DROP_METHOD "https")
    set(CTEST_DROP_SITE "{args.cdash_host}")
    set(CTEST_DROP_LOCATION "{args.submit_path}")
endif()

set(CTEST_SITE "{args.site}")
set(CTEST_BUILD_NAME "{args.build_name}")
set(CTEST_SOURCE_DIRECTORY "{args.source_dir}")
set(CTEST_BINARY_DIRECTORY "{args.binary_dir}")

file(MAKE_DIRECTORY "${{CTEST_BINARY_DIRECTORY}}")

if(NOT EXISTS "${{CTEST_BINARY_DIRECTORY}}/CMakeCache.txt")
    file(WRITE "${{CTEST_BINARY_DIRECTORY}}/CMakeCache.txt"
         "CMAKE_PROJECT_NAME:STATIC={args.project_name}\\n")
endif()

file(WRITE "${{CTEST_BINARY_DIRECTORY}}/CTestConfig.cmake" "
set(CTEST_PROJECT_NAME \\"{args.project_name}\\")
set(CTEST_NIGHTLY_START_TIME \\"01:00:00 UTC\\")
if(CMAKE_VERSION VERSION_GREATER 3.14)
    set(CTEST_SUBMIT_URL \\"{args.submit_url}\\")
else()
    set(CTEST_DROP_METHOD \\"https\\")
    set(CTEST_DROP_SITE \\"{args.cdash_host}\\")
    set(CTEST_DROP_LOCATION \\"{args.submit_path}\\")
endif()
set(CTEST_DROP_SITE_CDASH TRUE)
")

ctest_start({args.mode})

set(COVERAGE_SRC_FILE "{args.coverage_file}")
set(COVERAGE_DEST_FILE "${{CTEST_BINARY_DIRECTORY}}/coverage.xml")

if(NOT EXISTS "${{COVERAGE_SRC_FILE}}")
    message(FATAL_ERROR "Coverage file not found: ${{COVERAGE_SRC_FILE}}")
endif()

file(COPY "${{COVERAGE_SRC_FILE}}" DESTINATION "${{CTEST_BINARY_DIRECTORY}}")
get_filename_component(SRC_FILENAME "${{COVERAGE_SRC_FILE}}" NAME)
if(NOT SRC_FILENAME STREQUAL "coverage.xml")
    file(RENAME "${{CTEST_BINARY_DIRECTORY}}/${{SRC_FILENAME}}"
                "${{COVERAGE_DEST_FILE}}")
endif()

message(STATUS "Processing coverage file: ${{COVERAGE_DEST_FILE}}")

file(MAKE_DIRECTORY "${{CTEST_BINARY_DIRECTORY}}/Testing")
file(MAKE_DIRECTORY "${{CTEST_BINARY_DIRECTORY}}/Testing/CoverageInfo")

file(COPY "${{COVERAGE_DEST_FILE}}"
     DESTINATION "${{CTEST_BINARY_DIRECTORY}}/Testing/CoverageInfo")

ctest_coverage()

ctest_submit(PARTS Coverage
             RETRY_COUNT 3
             RETRY_DELAY 5
             CAPTURE_CMAKE_ERROR submit_error)

if(submit_error)
    message(FATAL_ERROR "Failed to submit coverage to CDash. "
                        "Error code: ${{submit_error}}")
endif()

message(STATUS "Successfully submitted coverage to CDash")
"""
    return script_content


def main():
    if not os.getenv("CI"):
        print("WARNING: CDash upload should normally only be done from CI/CD")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != "y":
            print("Aborted.")
            return 1

    parser = argparse.ArgumentParser(
        description="Upload coverage XML to CDash for rocprofiler-compute"
    )

    # required
    parser.add_argument(
        "--coverage-file", required=True, help="Path to coverage XML file"
    )
    parser.add_argument("--build-name", required=True, help="Build name for CDash")

    # optional
    parser.add_argument(
        "--project-name", default="rocprofiler-compute", help="CDash project name"
    )
    parser.add_argument(
        "--submit-url",
        default="https://my.cdash.org/submit.php?project=rocprofiler-compute",
        help="CDash submission URL",
    )
    parser.add_argument(
        "--cdash-host", default="my.cdash.org", help="CDash host (for older CMake)"
    )
    parser.add_argument(
        "--submit-path",
        default="/submit.php?project=rocprofiler-compute",
        help="CDash submission path (for older CMake)",
    )
    parser.add_argument(
        "--site",
        default=None,
        help="Site name for CDash (auto-detected if not provided)",
    )
    parser.add_argument(
        "--source-dir",
        default=None,
        help="Source directory path (auto-detected if not provided)",
    )
    parser.add_argument(
        "--binary-dir",
        default=None,
        help="Binary directory path (defaults to source-dir/build)",
    )
    parser.add_argument(
        "--mode",
        default="Experimental",
        choices=["Experimental", "Nightly", "Continuous"],
        help="CTest dashboard mode",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Generate script but don't execute"
    )

    args = parser.parse_args()

    is_monorepo, monorepo_root, project_root = detect_repo_structure()

    if not args.source_dir:
        args.source_dir = str(project_root)

    if not args.binary_dir:
        args.binary_dir = os.path.join(args.source_dir, "build")

    if not args.site:
        if is_monorepo:
            args.site = f"Monorepo-{os.uname().nodename}"
        else:
            args.site = os.uname().nodename

    coverage_path = Path(args.coverage_file)
    if not coverage_path.is_absolute():
        if not coverage_path.exists():
            coverage_path = project_root / args.coverage_file

    if not coverage_path.exists():
        print(f"ERROR: Coverage file not found: {coverage_path}", file=sys.stderr)
        print(f"  Searched in: {Path(args.coverage_file).absolute()}")
        print(f"  And in: {project_root / args.coverage_file}")
        return 1

    try:
        import xml.etree.ElementTree as ET

        tree = ET.parse(coverage_path)
        root = tree.getroot()
        if root.tag != "coverage":
            print(
                f"ERROR: File does not appear to be a coverage XML file "
                f"(root tag: {root.tag})",
                file=sys.stderr,
            )
            return 1
        line_rate = float(root.get("line-rate", 0)) * 100
        print(f"Line Coverage: {line_rate:.1f}%")
    except Exception as e:
        print(f"ERROR: Could not parse coverage XML file: {e}", file=sys.stderr)
        return 1

    args.coverage_file = str(coverage_path.absolute())
    args.source_dir = str(Path(args.source_dir).absolute())
    args.binary_dir = str(Path(args.binary_dir).absolute())

    print(f"Repository type: {'Monorepo' if is_monorepo else 'Standalone'}")
    print(f"Project root: {project_root}")
    print(f"Uploading coverage file: {args.coverage_file}")
    print(f"Build name: {args.build_name}")
    print(f"Project: {args.project_name}")
    print(f"Site: {args.site}")
    print(f"Submit URL: {args.submit_url}")

    # Ensure build directory exists
    os.makedirs(args.binary_dir, exist_ok=True)

    # Copy CTestConfig.cmake if it exists in source
    source_ctest_config = Path(args.source_dir) / "CTestConfig.cmake"
    if source_ctest_config.exists():
        shutil.copy2(source_ctest_config, Path(args.binary_dir) / "CTestConfig.cmake")
        print("Copied CTestConfig.cmake to build directory")

    script_content = create_ctest_script(args)

    if args.dry_run:
        print("\nGenerated CTest script:")
        print("=" * 50)
        print(script_content)
        return 0

    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".cmake", delete=False) as f:
            f.write(script_content)
            script_path = f.name

        print(f"\nScript written to: {script_path}")

        original_dir = os.getcwd()
        os.chdir(args.source_dir)

        cmd = ["ctest", "-S", script_path, "-V"]
        print(f"Executing: {' '.join(cmd)}")
        print(f"Working directory: {os.getcwd()}")

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.stdout:
            print("\nSTDOUT:")
            print(result.stdout)

        if result.stderr:
            print("\nSTDERR:")
            print(result.stderr)

        os.chdir(original_dir)

        if result.returncode != 0:
            print(f"\nCTest failed with return code: {result.returncode}")
            return result.returncode

        print("\n✅ Coverage successfully uploaded to CDash!")
        cdash_url = args.submit_url.replace(
            "/submit.php?project=", "/index.php?project="
        )
        print(f"View results at: {cdash_url}")
        return 0

    except Exception as e:
        print(f"Error executing CTest script: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1
    finally:
        if "script_path" in locals():
            try:
                os.unlink(script_path)
            except Exception:
                pass


if __name__ == "__main__":
    sys.exit(main())
