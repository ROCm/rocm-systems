#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
#
# Copyright (C) Advanced Micro Devices. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import argparse
import datetime
import json
import os
import sys
import pathlib
import textwrap

import subprocess
import traceback

version_number = "1.0.0"
build_date = f"{datetime.datetime.now():%b %d %Y}"
verbose_choices = ["DEBUG", "INFO", "WARNING", "ERROR", "EXCEPTION", "ALWAYS"]


# For print json with pathlib objects
class PathEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, pathlib.Path):
            return str(obj)
        return super().default(obj)


def print_with_cond(cond, msg, end=None):
    if isinstance(cond, str):
        # Check verbose level
        if cond in verbose_choices:
            cond_num = verbose_choices.index(cond)
            if cond_num < args.verbose_num:
                return
    elif not cond:
        return

    if end == None:
        print(msg)
    else:
        print(msg, end)
    return


def run_cmd(cmd):
    if isinstance(cmd, str):
        cmd = cmd.split()

    try:
        print_with_cond("DEBUG", f"Running {cmd}")
        result = subprocess.run(cmd, capture_output=True, text=True)
    except Exception as e:
        print_with_cond("EXCEPTION", str(e))
        if False:
            traceback.print_exc()
        else:
            traceback_str = traceback.format_exc()
            print_with_cond("EXCEPTION", traceback_str)
        return ("", "")
    return (result.stdout, result.stderr)


def find_root_path():
    last_path = pathlib.Path("/")
    root_path = pathlib.Path.cwd()
    while True:
        git_path = root_path / ".git"
        if git_path.exists():
            return root_path
        if root_path == last_path:
            break
        root_path = root_path.parent
    return None


def check_inputs(test_map):
    root_path = find_root_path()

    # Check for tests
    for _test_name in test_map:
        found = False
        for test_path in test_map[_test_name]["exe_paths"]:
            if test_path.exists() and test_path.is_file():
                test_map[_test_name]["exe_path"] = test_path
                found = True
                break
        if not found:
            test_path = root_path / test_map[_test_name]["exe_paths"][0]
            if test_path.exists() and test_path.is_file():
                test_map[_test_name]["exe_path"] = test_path
    return


def main():
    if not args.output_dir.exists():
        args.output_dir.mkdir(parents=True)

    for test_name in args.test_map:
        log_path = args.test_map[test_name]["log_path"]
        if log_path and log_path.exists():
            log_path.unlink()

    test_paths = []
    missing_test_paths = []
    for test_name in args.test_map:
        test_path = args.test_map[test_name]["exe_path"]
        if test_path:
            test_paths.append((test_name, test_path))
        else:
            missing_test_paths.append((test_name, test_path))
    if len(test_paths):
        if args.print_settings:
            print_with_cond("ALWAYS", f"Determined settings:")
            for test_name, test_path in test_paths:
                print_with_cond("ALWAYS", f"\ttest_name = {test_name:16s} test_path={test_path}")
            if args.output_dir:
                print_with_cond("ALWAYS", f"\toutput_dir= {args.output_dir}")
            return 0
        else:
            print_with_cond("INFO", f"Found tests:")
            for test_name, test_path in test_paths:
                print_with_cond("INFO", f"\ttest_name={test_name:16s} test_path={test_path}")
    if len(missing_test_paths):
        print_with_cond("INFO", f"Missing tests:")
        for test_name, test_path in missing_test_paths:
            print_with_cond("INFO", f"\ttest_name={test_name:16s} test_path={test_path}")

    if len(test_paths):
        for test_name, test_path in test_paths:
            print_with_cond("INFO", f"Running test {test_name}")
            cmd = f"{str(test_path)} -v"
            if test_name == "c_integration":
                cmd += " 2"
            if test_name == "api_summary":
                cmd = "./" + str(test_path)
            print_with_cond("INFO", f"Running {cmd}")
            std_out, std_err = run_cmd(cmd)

            file_std_out = args.output_dir / f"_{test_name}.log"
            print_with_cond("INFO", f"stdout file {file_std_out}  len={len(std_out)}")
            if std_out:
                file_std_out.write_text(std_out)
            file_std_err = args.output_dir / f"_{test_name}_err.log"
            print_with_cond("INFO", f"stderr file {file_std_err}  len={len(std_err)}")
            if std_err:
                file_std_err.write_text(std_err)

    return 0


def Parse_Command_Line(cmds=None):
    msg_description = "Create API coverage report for unit_test.py and integration_test.py tests"
    msg_epilog = "Example:\n\t%(prog)s --c_unit_test <c_unit_test.log> --py_integration_test <py_integration_test.log>"
    parser = argparse.ArgumentParser(
        description=msg_description,
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=textwrap.dedent(msg_epilog),
    )

    parser_info = parser.add_argument_group("Information")
    parser_info.add_argument(
        "--version", action="version", version=version_number, help="Show version and exit"
    )
    parser_info.add_argument(
        "--build", action="version", version=build_date, help="Show build and exit"
    )
    parser_info.add_argument(
        "--verbose",
        choices=verbose_choices,
        type=str,
        default="WARNING",
        help="Level of information to output, default=%(default)s",
    )

    parser_summary = parser.add_argument_group("Summary")
    parser_summary.add_argument(
        "--api_summary", default="api_summary.py", help="API summary report, default=%(default)s"
    )
    parser_summary.add_argument(
        "--api_summary_prefix",
        default="_api_summary",
        help="Prefix for API summary report files, default=%(default)s",
    )

    parser_test = parser.add_argument_group("Tests")
    parser_test.add_argument(
        "--c_unit_test", default="NONE_amdsmitst", help="C unit_test test, default=%(default)s"
    )
    parser_test.add_argument(
        "--c_integration", default="amdsmitst", help="C integration test, default=%(default)s"
    )
    parser_test.add_argument(
        "--py_unit_test", default="unit_tests.py", help="python unit_test test, default=%(default)s"
    )
    parser_test.add_argument(
        "--py_integration",
        default="integration_test.py",
        help="python integration test, default=%(default)s",
    )

    parser_logs = parser.add_argument_group("Log Files")
    parser_logs.add_argument(
        "--c_unit_test_log",
        default="_c_unit_test.log",
        help="Filename for C unit_test output, default=%(default)s",
    )
    parser_logs.add_argument(
        "--c_integration_log",
        default="_c_integration.log",
        help="Filename for C integration_test output, default=%(default)s",
    )
    parser_logs.add_argument(
        "--py_unit_test_log",
        default="_py_unit_test.log",
        help="Filename for python unit_test output, default=%(default)s",
    )
    parser_logs.add_argument(
        "--py_integration_log",
        default="_py_integration.log",
        help="Filename for python integration_test output, default=%(default)s",
    )

    parser_output = parser.add_argument_group("Output")
    parser_output.add_argument(
        "--output_dir", default=None, help="Path to output folder, default=%(default)s"
    )
    parser_output.add_argument(
        "--print_settings",
        action="store_true",
        help="Just print settings used, default=%(default)s",
    )

    if cmds:
        args = parser.parse_args(cmds.split())
    else:
        args = parser.parse_args()

    args.verbose_num = verbose_choices.index(args.verbose)

    if not args.output_dir:
        root_path = find_root_path() / "projects/amdsmi"
        args.output_dir = root_path / "build"
    else:
        args.output_dir = pathlib.Path(args.output_dir)

    args.test_map = {}
    args.test_map["c_unit_test"] = {}
    args.test_map["c_unit_test"]["exe_paths"] = [
        pathlib.Path(args.c_unit_test),
        pathlib.Path("."),
        pathlib.Path("."),
    ]
    args.test_map["c_unit_test"]["exe_path"] = None
    args.test_map["c_unit_test"]["log_path"] = args.output_dir / pathlib.Path(args.c_unit_test_log)

    args.test_map["c_integration"] = {}
    args.test_map["c_integration"]["exe_paths"] = [
        pathlib.Path(args.c_integration),
        pathlib.Path("build/tests/amd_smi_test/amdsmitst"),
        pathlib.Path("/opt/rocm/share/amd_smi/tests/amdsmitst"),
    ]
    args.test_map["c_integration"]["exe_path"] = None
    args.test_map["c_integration"]["log_path"] = args.output_dir / pathlib.Path(
        args.c_integration_log
    )

    args.test_map["py_unit_test"] = {}
    args.test_map["py_unit_test"]["exe_paths"] = [
        pathlib.Path(args.py_unit_test),
        pathlib.Path("tests/python_unittest/unit_tests.py"),
        pathlib.Path("/opt/rocm/share/amd_smi/tests/python_unittest/unit_tests.py"),
    ]
    args.test_map["py_unit_test"]["exe_path"] = None
    args.test_map["py_unit_test"]["log_path"] = args.output_dir / pathlib.Path(
        args.py_unit_test_log
    )

    args.test_map["py_integration"] = {}
    args.test_map["py_integration"]["exe_paths"] = [
        pathlib.Path(args.py_integration),
        pathlib.Path("tests/python_unittest/integration_test.py"),
        pathlib.Path("/opt/rocm/share/amd_smi/tests/python_unittest/integration_test.py"),
    ]
    args.test_map["py_integration"]["exe_path"] = None
    args.test_map["py_integration"]["log_path"] = args.output_dir / pathlib.Path(
        args.py_integration_log
    )

    args.test_map["api_summary"] = {}
    args.test_map["api_summary"]["exe_paths"] = [
        pathlib.Path(args.api_summary),
        pathlib.Path("tests/api_summary.py"),
        pathlib.Path("/opt/rocm/share/amd_smi/tests/api_summary.py"),
    ]
    args.test_map["api_summary"]["exe_path"] = None
    args.test_map["api_summary"]["log_path"] = None

    check_inputs(args.test_map)

    cond_num = verbose_choices.index("DEBUG")
    if cond_num >= args.verbose_num:
        print(json.dumps(args.test_map, cls=PathEncoder, sort_keys=False, indent=4), flush=True)

    return args


if __name__ == "__main__":
    args = Parse_Command_Line()
    rc = main()
    sys.exit(rc)
