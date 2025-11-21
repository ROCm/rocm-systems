#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

import os
import re
import sys
import glob
import socket
import shutil
import argparse
import multiprocessing


def which(cmd, require):
    v = shutil.which(cmd)
    if require and v is None:
        raise RuntimeError(f"{cmd} not found")
    return v if v is not None else ""


def generate_custom(args, cmake_args, ctest_args):
    if not os.path.exists(args.binary_dir):
        os.makedirs(args.binary_dir)

    NAME = args.name
    SITE = args.site
    BUILD_JOBS = args.build_jobs
    SUBMIT_URL = args.submit_url
    SOURCE_DIR = os.path.realpath(args.source_dir)
    BINARY_DIR = os.path.realpath(args.binary_dir)
    CMAKE_ARGS = " ".join(cmake_args)
    CTEST_ARGS = " ".join(ctest_args)

    GIT_CMD = which("git", require=True)
    GCOV_CMD = which("gcov", require=False)
    CMAKE_CMD = which("cmake", require=True)
    CTEST_CMD = which("ctest", require=True)

    NAME = re.sub(r"(.*)-([0-9]+)/merge", "PR_\\2_\\1", NAME)

    return f"""
        set(CTEST_PROJECT_NAME "rocprofiler-systems")
        set(CTEST_NIGHTLY_START_TIME "05:00:00 UTC")

        set(CTEST_DROP_METHOD "http")
        set(CTEST_DROP_SITE_CDASH TRUE)
        set(CTEST_SUBMIT_URL "https://{SUBMIT_URL}")

        set(CTEST_UPDATE_TYPE git)
        set(CTEST_UPDATE_VERSION_ONLY TRUE)
        set(CTEST_GIT_INIT_SUBMODULES TRUE)

        set(CTEST_OUTPUT_ON_FAILURE TRUE)
        set(CTEST_USE_LAUNCHERS TRUE)
        set(CMAKE_CTEST_ARGUMENTS --output-on-failure {CTEST_ARGS})

        set(CTEST_CUSTOM_MAXIMUM_NUMBER_OF_ERRORS "100")
        set(CTEST_CUSTOM_MAXIMUM_NUMBER_OF_WARNINGS "100")
        set(CTEST_CUSTOM_MAXIMUM_PASSED_TEST_OUTPUT_SIZE "51200")
        set(CTEST_CUSTOM_COVERAGE_EXCLUDE "/usr/.*;.*external/.*;.*examples/.*")

        set(CTEST_SITE "{SITE}")
        set(CTEST_BUILD_NAME "{NAME}")

        set(CTEST_SOURCE_DIRECTORY {SOURCE_DIR})
        set(CTEST_BINARY_DIRECTORY {BINARY_DIR})

        set(CTEST_UPDATE_COMMAND {GIT_CMD})
        set(CTEST_CONFIGURE_COMMAND "{CMAKE_CMD} -B {BINARY_DIR} {SOURCE_DIR} -DROCPROFSYS_BUILD_CI=ON {CMAKE_ARGS}")
        set(CTEST_BUILD_COMMAND "{CMAKE_CMD} --build {BINARY_DIR} --target all --parallel {BUILD_JOBS}")
        set(CTEST_COVERAGE_COMMAND {GCOV_CMD})
        """


def generate_dashboard_script(args):
    CODECOV = 1 if args.coverage else 0
    DASHBOARD_MODE = args.mode
    SOURCE_DIR = os.path.realpath(args.source_dir)
    BINARY_DIR = os.path.realpath(args.binary_dir)

    _script = """

        include("${CMAKE_CURRENT_LIST_DIR}/CTestCustom.cmake")

        macro(handle_error _message _ret)
            if(NOT ${${_ret}} EQUAL 0)
                ctest_submit(PARTS Done RETURN_VALUE _submit_ret)
                message(FATAL_ERROR "${_message} failed: ${${_ret}}")
            endif()
        endmacro()
        """

    _script += f"""
        ctest_start({DASHBOARD_MODE})
        ctest_update(SOURCE "{SOURCE_DIR}")
        ctest_configure(BUILD "{BINARY_DIR}" RETURN_VALUE _configure_ret)
        ctest_submit(PARTS Start Update Configure RETURN_VALUE _submit_ret)

        handle_error("Configure" _configure_ret)

        ctest_build(BUILD "{BINARY_DIR}" RETURN_VALUE _build_ret)
        ctest_submit(PARTS Build RETURN_VALUE _submit_ret)

        handle_error("Build" _build_ret)

        ctest_test(BUILD "{BINARY_DIR}" RETURN_VALUE _test_ret)
        ctest_submit(PARTS Test RETURN_VALUE _submit_ret)

        if("{CODECOV}" GREATER 0)
            ctest_coverage(
                BUILD "{BINARY_DIR}"
                RETURN_VALUE _coverage_ret
                CAPTURE_CMAKE_ERROR _coverage_err)
            ctest_submit(PARTS Coverage RETURN_VALUE _submit_ret)
        endif()

        handle_error("Testing" _test_ret)

        ctest_submit(PARTS Done RETURN_VALUE _submit_ret)
        """
    return _script


def parse_cdash_args(args):
    BUILD_JOBS = multiprocessing.cpu_count()
    DASHBOARD_MODE = "Continuous"
    DASHBOARD_STAGES = [
        "Start",
        "Update",
        "Configure",
        "Build",
        "Test",
        "MemCheck",
        "Coverage",
        "Submit",
    ]
    SOURCE_DIR = os.getcwd()
    BINARY_DIR = os.path.join(SOURCE_DIR, "build")
    SITE = socket.gethostname()
    NAME = None
    SUBMIT_URL = "my.cdash.org/submit.php?project=rocprofiler-systems"
    CODECOV = False

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-n", "--name", help="Job name", default=None, type=str, required=True
    )
    parser.add_argument("-s", "--site", help="Site name", default=SITE, type=str)
    parser.add_argument(
        "-c", "--coverage", help="Enable code coverage", action="store_true"
    )
    parser.add_argument(
        "-j", "--build-jobs", help="Number of build tasks", default=BUILD_JOBS, type=int
    )
    parser.add_argument(
        "-B", "--binary-dir", help="Build directory", default=BINARY_DIR, type=str
    )
    parser.add_argument(
        "-S", "--source-dir", help="Source directory", default=SOURCE_DIR, type=str
    )
    parser.add_argument(
        "-M",
        "--mode",
        help="Dashboard mode",
        default=DASHBOARD_MODE,
        choices=("Continuous", "Nightly", "Experimental"),
        type=str,
    )
    parser.add_argument(
        "-T",
        "--stages",
        help="Dashboard stages",
        nargs="+",
        default=DASHBOARD_STAGES,
        choices=DASHBOARD_STAGES,
        type=str,
    )
    parser.add_argument(
        "--submit-url", help="CDash submission site", default=SUBMIT_URL, type=str
    )
    parser.add_argument(
        "--repeat-until-pass", help="<N> for --repeat until-pass:<N>", default=3, type=int
    )
    parser.add_argument(
        "--repeat-until-fail",
        help="<N> for --repeat until-fail:<N>",
        default=None,
        type=int,
    )
    parser.add_argument(
        "--repeat-after-timeout",
        help="<N> for --repeat after-timeout:<N>",
        default=2,
        type=int,
    )

    return parser.parse_args(args)


def parse_args(args=None):
    if args is None:
        args = sys.argv[1:]

    index = 0
    input_args = []
    ctest_args = []
    cmake_args = []
    data = [input_args, cmake_args, ctest_args]

    for itr in args:
        if itr == "--":
            index += 1
            if index > 2:
                raise RuntimeError("Usage: <options> -- <ctest-args> -- <cdash-args>")
        else:
            data[index].append(itr)

    cdash_args = parse_cdash_args(input_args)

    if cdash_args.coverage:
        cmake_args += [
            "-DROCPROFSYS_BUILD_CODECOV=ON",
            "-DROCPROFSYS_STRIP_LIBRARIES=OFF",
        ]

    def get_repeat_val(_param):
        _value = getattr(cdash_args, f"repeat_{_param}".replace("-", "_"))
        return [f"{_param}:{_value}"] if _value is not None and _value > 1 else []

    repeat_args = (
        get_repeat_val("until-pass")
        + get_repeat_val("until-fail")
        + get_repeat_val("after-timeout")
    )
    ctest_args += ["--repeat"] + repeat_args if len(repeat_args) > 0 else []

    return [cdash_args, cmake_args, ctest_args]


def run(*args, **kwargs):
    import subprocess

    return subprocess.run(*args, **kwargs)


if __name__ == "__main__":
    args, cmake_args, ctest_args = parse_args()

    if not os.path.exists(args.binary_dir):
        os.makedirs(args.binary_dir)

    from textwrap import dedent

    _config = dedent(generate_custom(args, cmake_args, ctest_args))
    _script = dedent(generate_dashboard_script(args))

    with open(os.path.join(args.binary_dir, "CTestCustom.cmake"), "w") as f:
        f.write(f"{_config}\n")

    with open(os.path.join(args.binary_dir, "dashboard.cmake"), "w") as f:
        f.write(f"{_script}\n")

    CTEST_CMD = which("ctest", require=True)

    dashboard_args = ["-D"]
    for itr in args.stages:
        dashboard_args.append(f"{args.mode}{itr}")

    try:
        run(
            [CTEST_CMD]
            + dashboard_args
            + [
                "-S",
                os.path.join(args.binary_dir, "dashboard.cmake"),
                "--output-on-failure",
                "-V",
            ]
            + ctest_args,
            check=True,
        )
    finally:
        if "-VV" not in ctest_args:
            for file in glob.glob(
                os.path.join(args.binary_dir, "Testing/**"), recursive=True
            ):
                if not os.path.isfile(file):
                    continue
                print(f"\n\n\n###### Reading {file}... ######\n\n\n")
                with open(file, "r") as inpf:
                    fdata = inpf.read()
                    if "LastTest" not in file and "Coverage" not in file:
                        print(fdata)
                    oname = os.path.basename(file)
                    if oname.endswith(".log"):
                        oname += ".log"
                    with open(os.path.join(args.binary_dir, oname), "w") as outf:
                        print(f"\n\n###### Writing {oname}... ######\n\n")
                        outf.write(fdata)
