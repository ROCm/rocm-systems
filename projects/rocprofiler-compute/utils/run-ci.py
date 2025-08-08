##############################################################################
# MIT License
#
# Copyright (c) 2021 - 2025 Advanced Micro Devices, Inc. All Rights Reserved.
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


#!/usr/bin/env python3

"""
CI script to upload coverage XML files to CDash
"""

import os
import sys
import argparse
import subprocess
import tempfile
from pathlib import Path

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

set(CTEST_COVERAGE_XML_FILE "{args.coverage_file}")

ctest_start({args.mode})

if(EXISTS "${{CTEST_COVERAGE_XML_FILE}}")
    message(STATUS "Submitting coverage file: ${{CTEST_COVERAGE_XML_FILE}}")
    ctest_submit(FILES "${{CTEST_COVERAGE_XML_FILE}}" 
                 PARTS Coverage
                 RETRY_COUNT 3
                 RETRY_DELAY 5
                 CAPTURE_CMAKE_ERROR submit_error)
    
    if(NOT submit_error EQUAL 0)
        message(FATAL_ERROR "Failed to submit coverage to CDash: ${{submit_error}}")
    else()
        message(STATUS "Successfully submitted coverage to CDash")
    endif()
else()
    message(WARNING "Coverage file not found: ${{CTEST_COVERAGE_XML_FILE}}")
    message(FATAL_ERROR "Cannot submit coverage - file missing")
endif()
"""
    return script_content

def main():
    parser = argparse.ArgumentParser(description="Upload coverage XML to CDash")
    
    # required args
    parser.add_argument("--coverage-file", required=True,
                       help="Path to coverage XML file")
    parser.add_argument("--build-name", required=True,
                       help="Build name for CDash")
    
    # optional args
    parser.add_argument("--project-name", default="rocprofiler-compute",
                       help="CDash project name")
    parser.add_argument("--submit-url", 
                       default="https://my.cdash.org/submit.php?project=rocprofiler-compute",
                       help="CDash submission URL")
    parser.add_argument("--cdash-host", default="my.cdash.org",
                       help="CDash host (for older CMake)")
    parser.add_argument("--submit-path", 
                       default="/submit.php?project=rocprofiler-compute",
                       help="CDash submission path (for older CMake)")
    parser.add_argument("--site", default=os.uname().nodename,
                       help="Site name for CDash")
    parser.add_argument("--source-dir", default=os.getcwd(),
                       help="Source directory path")
    parser.add_argument("--binary-dir", 
                       help="Binary directory path (defaults to source-dir/build)")
    parser.add_argument("--mode", default="Experimental",
                       choices=["Experimental", "Nightly", "Continuous"],
                       help="CTest dashboard mode")
    parser.add_argument("--dry-run", action="store_true",
                       help="Generate script but don't execute")
    
    args = parser.parse_args()
    
    if not args.binary_dir:
        args.binary_dir = os.path.join(args.source_dir, "build")
    
    coverage_path = Path(args.coverage_file)
    if not coverage_path.exists():
        print(f"ERROR: Coverage file not found: {coverage_path}", file=sys.stderr)
        return 1
    
    args.coverage_file = str(coverage_path.absolute())
    args.source_dir = str(Path(args.source_dir).absolute())
    args.binary_dir = str(Path(args.binary_dir).absolute())
    
    print(f"Uploading coverage file: {args.coverage_file}")
    print(f"Build name: {args.build_name}")
    print(f"Project: {args.project_name}")
    print(f"Submit URL: {args.submit_url}")
    
    script_content = create_ctest_script(args)
    
    if args.dry_run:
        print("\nGenerated CTest script:")
        print("=" * 50)
        print(script_content)
        return 0
    
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cmake', delete=False) as f:
            f.write(script_content)
            script_path = f.name
        
        cmd = ['ctest', '-S', script_path, '-V']
        print(f"Executing: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        if result.returncode != 0:
            print(f"CTest failed with return code: {result.returncode}")
            return result.returncode
        
        print("Coverage successfully uploaded to CDash!")
        return 0
        
    except Exception as e:
        print(f"Error executing CTest script: {e}", file=sys.stderr)
        return 1
    finally:
        if 'script_path' in locals():
            try:
                os.unlink(script_path)
            except:
                pass

if __name__ == "__main__":
    sys.exit(main())