#!/usr/bin/env python3


import os
import re
import sys
import glob
import socket
import shutil
import argparse
import multiprocessing

# this constant is used to define CTEST_PROJECT_NAME
# and default value for CTEST_SUBMIT_URL
# _PROJECT_NAME = "rocprofiler-v2-internal"
# _BASE_URL = "10.194.116.31/cdash"
_PROJECT_NAME = "rocprofiler-sdk"
_BASE_URL = "cdash.rocprofiler.amd.com"
_GCOVR_GENERATE_CMD = None

# these are various default values
_VISIBLE_PROJECT_NAME = _PROJECT_NAME.replace("-internal", "")
_DEFAULT_ROCM_PATH = os.path.realpath(os.environ.get("ROCM_PATH", "/opt/rocm"))
_DEFAULT_INSTALL_PREFIX = (
    os.path.realpath(_DEFAULT_ROCM_PATH)
    if os.path.exists(_DEFAULT_ROCM_PATH)
    else f"/opt/{_VISIBLE_PROJECT_NAME}"
)
_DEFAULT_GPU_TARGETS = os.environ.get(
    "GPU_TARGETS",
    "gfx900 gfx906 gfx908 gfx90a gfx942 gfx950 gfx1030 gfx1100 gfx1101 gfx1102",
).split()


def which(cmd, require):
    v = shutil.which(cmd)
    if require and v is None:
        raise RuntimeError(f"{cmd} not found")
    return v if v is not None else ""


def generate_custom(args, cmake_args, ctest_args):
    if not os.path.exists(args.binary_dir):
        os.makedirs(args.binary_dir)

    if args.memcheck is not None:
        if args.coverage:
            raise ValueError(
                f"Enabling --memcheck={args.memcheck} and --coverage not supported"
            )
        cmake_args += [f"-DROCPROFILER_MEMCHECK={args.memcheck}"]

    NAME = args.name
    SITE = args.site
    BUILD_JOBS = args.build_jobs
    SUBMIT_URL = args.submit_url
    SOURCE_DIR = os.path.realpath(args.source_dir)
    BINARY_DIR = os.path.realpath(args.binary_dir)
    CMAKE_ARGS = " ".join(cmake_args)
    CTEST_ARGS = " ".join(['"{}"'.format(x.replace('"', '\\"')) for x in ctest_args])

    GIT_CMD = which("git", require=True)
    GCOV_CMD = which("gcov", require=False)
    GCOVR_CMD = which("gcovr", require=False)
    CMAKE_CMD = which("cmake", require=True)
    # CTEST_CMD = which("ctest", require=True)

    NAME = re.sub(r"(.*)-([0-9]+)/merge", "PR_\\2_\\1", NAME)

    def option_in_args(_key, _args):
        _union = [x for x in _args if f"{_key}=" in x]
        return len(_union) != 0

    DEFAULT_CMAKE_ARGS = []
    for key, value in dict(
        [
            ["CMAKE_BUILD_TYPE", "RelWithDebInfo"],
            ["CMAKE_INSTALL_PREFIX", f"{_DEFAULT_INSTALL_PREFIX}"],
            ["CPACK_PACKAGING_INSTALL_PREFIX", f"{_DEFAULT_INSTALL_PREFIX}"],
            ["CPACK_GENERATOR", "DEB;RPM;TGZ"],
            ["Python3_EXECUTABLE", sys.executable],
        ]
        + [[f"ROCPROFILER_BUILD_{x}", "ON"] for x in ["CI", "TESTS", "SAMPLES"]]
    ).items():
        if not option_in_args(key, cmake_args):
            DEFAULT_CMAKE_ARGS += [f"-D{key}={value}"]

    DEFAULT_CMAKE_ARGS = " ".join(DEFAULT_CMAKE_ARGS)

    GPU_TARGETS = ";".join(args.gpu_targets)
    MEMCHECK_TYPE = "" if args.memcheck is None else args.memcheck

    MEMCHECK_SANITIZER_OPTIONS = ""
    MEMCHECK_SUPPRESSION_FILE = ""

    if MEMCHECK_TYPE == "AddressSanitizer":
        MEMCHECK_SANITIZER_OPTIONS = "detect_leaks=0 use_sigaltstack=0"
        MEMCHECK_SUPPRESSION_FILE = (
            f"{SOURCE_DIR}/source/scripts/address-sanitizer-suppr.txt"
        )
    elif MEMCHECK_TYPE == "LeakSanitizer":
        MEMCHECK_SUPPRESSION_FILE = (
            f"{SOURCE_DIR}/source/scripts/leak-sanitizer-suppr.txt"
        )
    elif MEMCHECK_TYPE == "ThreadSanitizer":
        external_symbolizer_path = ""
        for version in range(8, 20):
            _symbolizer = shutil.which(f"llvm-symbolizer-{version}")
            if _symbolizer:
                external_symbolizer_path = f"external_symbolizer_path={_symbolizer}"
        os.environ["TSAN_OPTIONS"] = " ".join(
            [
                "history_size=5",
                "detect_deadlocks=0",
                f"suppressions={SOURCE_DIR}/source/scripts/thread-sanitizer-suppr.txt",
                external_symbolizer_path,
                os.environ.get("TSAN_OPTIONS", ""),
            ]
        )
    elif MEMCHECK_TYPE == "UndefinedBehaviorSanitizer":
        MEMCHECK_SUPPRESSION_FILE = (
            f"{SOURCE_DIR}/source/scripts/undef-behavior-sanitizer-suppr.txt"
        )
        os.environ["UBSAN_OPTIONS"] = " ".join(
            [
                "print_stacktrace=1",
                f"suppressions={SOURCE_DIR}/source/scripts/undef-behavior-sanitizer-suppr.txt",
                os.environ.get("UBSAN_OPTIONS", ""),
            ]
        )

    codecov_exclude = [
        "/usr/.*",
        "/opt/.*",
        "external/.*",
        "samples/.*",
        "tests/.*",
        ".*/external/.*",
        ".*/samples/.*",
        ".*/tests/.*",
        ".*/details/.*",
        ".*/counters/parser/.*",
    ]
    if args.coverage == "samples":
        codecov_exclude += [
            ".*/lib/common/.*",
            ".*/lib/output/.*",
            ".*/lib/att-tool/.*",
            ".*/lib/rocprofiler-sdk-tool/.*",
        ]

    COVERAGE_EXCLUDE = ";".join(codecov_exclude)

    if args.coverage and GCOVR_CMD:
        global _GCOVR_GENERATE_CMD

        codecov_dir = os.path.join(args.source_dir, ".codecov")
        codecov_xml = os.path.join(codecov_dir, f"{args.coverage}.xml")
        codecov_html = os.path.join(codecov_dir, f"{args.coverage}.html")

        if not os.path.exists(codecov_dir):
            os.makedirs(codecov_dir)

        with open(os.path.join(codecov_dir, ".gitignore"), "w") as f:
            f.write("/*\n")

        gcovr_codecov_exclude = []
        for itr in codecov_exclude:
            gcovr_codecov_exclude += ["--exclude", f"{itr}"]

        _GCOVR_GENERATE_CMD = (
            [GCOVR_CMD]
            + [
                "--root",
                f"{args.source_dir}",
                "--exclude-unreachable-branches",
                "--exclude-throw-branches",
                "--gcov-ignore-parse-errors",
                "--gcov-executable",
                GCOV_CMD,
                "-s",
                "-p",
                "--xml",
                codecov_xml,
                "--html-details",
                codecov_html,
            ]
            + gcovr_codecov_exclude
            + [args.source_dir]
        )

    return f"""
        set(CTEST_PROJECT_NAME "{_PROJECT_NAME}")
        set(CTEST_NIGHTLY_START_TIME "05:00:00 UTC")

        set(CTEST_DROP_METHOD "http")
        set(CTEST_DROP_SITE_CDASH TRUE)
        set(CTEST_SUBMIT_URL "http://{SUBMIT_URL}")

        set(CTEST_UPDATE_TYPE git)
        set(CTEST_UPDATE_VERSION_ONLY TRUE)
        set(CTEST_GIT_COMMAND {GIT_CMD})
        set(CTEST_GIT_INIT_SUBMODULES FALSE)

        set(CTEST_OUTPUT_ON_FAILURE TRUE)
        set(CTEST_USE_LAUNCHERS TRUE)
        set(CMAKE_CTEST_ARGUMENTS "--output-on-failure" {CTEST_ARGS})

        set(CTEST_CUSTOM_MAXIMUM_NUMBER_OF_ERRORS "100")
        set(CTEST_CUSTOM_MAXIMUM_NUMBER_OF_WARNINGS "100")
        set(CTEST_CUSTOM_MAXIMUM_PASSED_TEST_OUTPUT_SIZE "51200")
        set(CTEST_CUSTOM_COVERAGE_EXCLUDE "{COVERAGE_EXCLUDE}")

        set(CTEST_MEMORYCHECK_TYPE "{MEMCHECK_TYPE}")
        set(CTEST_MEMORYCHECK_SUPPRESSIONS_FILE "{MEMCHECK_SUPPRESSION_FILE}")
        set(CTEST_MEMORYCHECK_SANITIZER_OPTIONS "{MEMCHECK_SANITIZER_OPTIONS}")

        set(CTEST_SITE "{SITE}")
        set(CTEST_BUILD_NAME "{NAME}")

        set(CTEST_SOURCE_DIRECTORY {SOURCE_DIR})
        set(CTEST_BINARY_DIRECTORY {BINARY_DIR})

        set(CTEST_CONFIGURE_COMMAND "{CMAKE_CMD} -B {BINARY_DIR} {SOURCE_DIR} {DEFAULT_CMAKE_ARGS} -DGPU_TARGETS={GPU_TARGETS} {CMAKE_ARGS}")
        set(CTEST_BUILD_COMMAND "{CMAKE_CMD} --build {BINARY_DIR} --target all --parallel {BUILD_JOBS}")
        set(CTEST_COVERAGE_COMMAND {GCOV_CMD})
        """


def generate_dashboard_script(args):
    CODECOV = 1 if args.coverage else 0
    DASHBOARD_MODE = args.mode
    SOURCE_DIR = os.path.realpath(args.source_dir)
    BINARY_DIR = os.path.realpath(args.binary_dir)
    MEMCHECK = 1 if args.memcheck is not None else 0
    SUBMIT = 0 if args.disable_cdash else 1
    STRICT_SUBMIT = 1 if args.require_cdash_submission else 0
    ARGN = "${ARGN}"
    SUBMIT_ERR = "${_cdash_submit_err}"

    if args.memcheck == "ThreadSanitizer":
        MEMCHECK = 0

    _script = f"""
        cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

        macro(dashboard_submit)
            if("{SUBMIT}" GREATER 0)
                ctest_submit({ARGN}
                             RETRY_COUNT 0
                             RETRY_DELAY 10
                             CAPTURE_CMAKE_ERROR _cdash_submit_err)

                if(NOT _cdash_submit_err EQUAL 0)
                    if("{STRICT_SUBMIT}" GREATER 0)
                        message(FATAL_ERROR "CDash submission failed: {SUBMIT_ERR}")
                    else()
                        message(AUTHOR_WARNING "CDash submission failure ignored due to absence of --require-cdash-submission")
                    endif()
                endif()
            endif()
        endmacro()
    """

    _script += """

        include("${CMAKE_CURRENT_LIST_DIR}/CTestCustom.cmake")

        macro(handle_error _message _ret)
            if(NOT ${${_ret}} EQUAL 0)
                dashboard_submit(PARTS Done RETURN_VALUE _submit_ret)
                message(FATAL_ERROR "${_message} failed: ${${_ret}}")
            endif()
        endmacro()
        """

    STAGES = ";".join([itr.upper() for itr in args.stages])

    _script += f"""
        set(STAGES "{STAGES}")
        ctest_start({DASHBOARD_MODE})
        ctest_update(SOURCE "{SOURCE_DIR}" RETURN_VALUE _update_ret
                     CAPTURE_CMAKE_ERROR _update_err)
        ctest_configure(BUILD "{BINARY_DIR}" RETURN_VALUE _configure_ret)
        dashboard_submit(PARTS Start Update Configure RETURN_VALUE _submit_ret)

        if(NOT _update_err EQUAL 0)
            message(WARNING "ctest_update failed")
        endif()

        handle_error("Configure" _configure_ret)

        if("BUILD" IN_LIST STAGES)
            ctest_build(BUILD "{BINARY_DIR}" RETURN_VALUE _build_ret)
            dashboard_submit(PARTS Build RETURN_VALUE _submit_ret)

            handle_error("Build" _build_ret)
        endif()

        if("TEST" IN_LIST STAGES)
            if("{MEMCHECK}" GREATER 0)
                ctest_memcheck(BUILD "{BINARY_DIR}" RETURN_VALUE _test_ret)
                dashboard_submit(PARTS Test RETURN_VALUE _submit_ret)
            else()
                ctest_test(BUILD "{BINARY_DIR}" RETURN_VALUE _test_ret)
                dashboard_submit(PARTS Test RETURN_VALUE _submit_ret)
            endif()
        endif()

        if("{CODECOV}" GREATER 0 AND "COVERAGE" IN_LIST STAGES)
            ctest_coverage(
                BUILD "{BINARY_DIR}"
                RETURN_VALUE _coverage_ret
                CAPTURE_CMAKE_ERROR _coverage_err)
            dashboard_submit(PARTS Coverage RETURN_VALUE _submit_ret)
        endif()

        handle_error("Testing" _test_ret)

        dashboard_submit(PARTS Done RETURN_VALUE _submit_ret)
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
    SUBMIT_URL = f"{_BASE_URL}/submit.php?project={_PROJECT_NAME}"

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-n", "--name", help="Job name", default=None, type=str, required=True
    )
    parser.add_argument("-s", "--site", help="Site name", default=SITE, type=str)
    parser.add_argument(
        "-q", "--quiet", help="Disable printing logs", action="store_true"
    )
    parser.add_argument(
        "-c",
        "--coverage",
        help="Enable code coverage",
        choices=("all", "tests", "samples"),
        type=str,
        default=None,
    )
    parser.add_argument(
        "-j",
        "--build-jobs",
        help="Number of build tasks",
        default=BUILD_JOBS,
        type=int,
    )
    parser.add_argument(
        "-B",
        "--binary-dir",
        help="Build directory",
        default=BINARY_DIR,
        type=str,
    )
    parser.add_argument(
        "-S",
        "--source-dir",
        help="Source directory",
        default=SOURCE_DIR,
        type=str,
    )
    parser.add_argument(
        "-F",
        "--clean",
        help="Remove existing build directory",
        action="store_true",
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
        "--submit-url",
        help="CDash submission site",
        default=SUBMIT_URL,
        type=str,
    )
    parser.add_argument(
        "--repeat-until-pass",
        help="<N> for --repeat until-pass:<N>",
        default=None,
        type=int,
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
        default=None,
        type=int,
    )
    parser.add_argument(
        "--disable-cdash",
        help="Disable submitting results to CDash dashboard",
        action="store_true",
    )
    parser.add_argument(
        "--require-cdash-submission",
        help="Failure to submit results to CDash dashboard causes CTest failure",
        action="store_true",
    )
    parser.add_argument(
        "--gpu-targets",
        help="GPU build architectures",
        default=_DEFAULT_GPU_TARGETS,
        type=str,
        nargs="+",
    )
    parser.add_argument(
        "--memcheck",
        help="Run dynamic analysis tool",
        default=None,
        type=str,
        choices=(
            "ThreadSanitizer",
            "AddressSanitizer",
            "LeakSanitizer",
            "MemorySanitizer",
            "UndefinedBehaviorSanitizer",
        ),
    )
    parser.add_argument(
        "--linter",
        help="Enable linting tool",
        default=None,
        type=str,
        choices=("clang-tidy",),
    )
    parser.add_argument(
        "--run-attempt",
        help="If > 1, will enable verbose logging of tests",
        default=1,
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
    cmd = os.path.basename(sys.argv[0])

    for itr in args:
        if itr == "--":
            index += 1
            if index > 2:
                raise RuntimeError(
                    f"Usage: {cmd} <options> -- <cmake-args> -- <ctest-args>"
                )
        else:
            data[index].append(itr)

    cdash_args = parse_cdash_args(input_args)

    if cdash_args.run_attempt > 1:
        os.environ["ROCPROFILER_LOG_LEVEL"] = "info"
        os.environ["ROCPROF_LOG_LEVEL"] = "info"

    if cdash_args.coverage:
        cmake_args += ["-DROCPROFILER_BUILD_CODECOV=ON"]
        if cdash_args.coverage == "samples":
            ctest_args += ["-L", "samples"]
        elif cdash_args.coverage == "tests":
            ctest_args += ["-L", "tests"]

    if cdash_args.linter == "clang-tidy":
        cmake_args += ["-DROCPROFILER_ENABLE_CLANG_TIDY=ON"]

    if (
        cdash_args.mode == "Nightly"
        and not cdash_args.require_cdash_submission
        and not cdash_args.disable_cdash
    ):
        sys.stderr.write(
            "Enabling --require-cdash-submission for Nightly mode. Use --disable-cdash to suppress\n"
        )
        sys.stderr.flush()
        cdash_args.require_cdash_submission = True

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

    if args.clean and os.path.exists(args.binary_dir):
        if args.source_dir == args.binary_dir:
            raise RuntimeError(
                f"cannot clean binary directory == source directory ({args.source_dir})"
            )

        shutil.rmtree(args.binary_dir)

    if not os.path.exists(args.binary_dir):
        os.makedirs(args.binary_dir)

    from textwrap import dedent

    _config = dedent(generate_custom(args, cmake_args, ctest_args))
    _script = dedent(generate_dashboard_script(args))

    if not args.quiet:
        sys.stderr.write(f"##### CTestCustom.cmake #####\n\n{_config}\n\n")
        sys.stderr.write(f"##### dashboard.cmake #####\n\n{_script}\n\n")

    with open(os.path.join(args.binary_dir, "CTestCustom.cmake"), "w") as f:
        f.write(f"{_config}\n")

    with open(os.path.join(args.binary_dir, "dashboard.cmake"), "w") as f:
        f.write(f"{_script}\n")

    CTEST_CMD = which("ctest", require=True)

    dashboard_args = ["-D"]
    for itr in args.stages:
        dashboard_args.append(f"{args.mode}{itr}")

    try:
        verbose_options = (
            "--progress",
            "-V",
            "-VV",
            "--debug",
            "--output-on-failure",
            "-Q",
            "--quiet",
        )
        if not args.quiet and len(ctest_args) == 0:
            ctest_args = ["--output-on-failure", "-V"]
        elif not args.quiet:
            opts_union = [x for x in ctest_args if x in verbose_options]
            if len(opts_union) == 0:
                ctest_args += ["--progress", "--output-on-failure", "-V"]

        # always fail if no tests exist
        ctest_args += ["--no-tests=error"]

        run_args = (
            [CTEST_CMD]
            + dashboard_args
            + [
                "-S",
                os.path.join(args.binary_dir, "dashboard.cmake"),
            ]
            + ctest_args
        )

        print("CTest command: {}".format(" ".join(run_args)))

        run(
            run_args,
            check=True,
        )
    finally:
        if "-VV" not in ctest_args and not args.quiet:
            tag = None
            tagfpath = os.path.join(args.binary_dir, "Testing/TAG")
            with open(tagfpath, "r") as f:
                tag = f.readline().strip()

            for file in glob.glob(
                os.path.join(args.binary_dir, "Testing", tag, "**"),
                recursive=True,
            ):
                if not os.path.isfile(file):
                    continue
                elif "CoverageLog-" in os.path.basename(file):
                    continue
                elif "Test.xml" in os.path.basename(file):
                    continue
                print(f"\n\n###### Reading {file}... ######\n\n")
                with open(file, "r") as inpf:
                    fdata = inpf.read()
                    print(fdata)
            # print out memory checker files
            for file in glob.glob(
                os.path.join(args.binary_dir, "Testing/Temporary/MemoryChecker.*"),
                recursive=True,
            ):
                if not os.path.isfile(file):
                    continue
                print(f"\n\n\n###### Reading {file}... ######\n\n\n")
                with open(file, "r") as inpf:
                    fdata = inpf.read()
                    print(fdata)

    if _GCOVR_GENERATE_CMD:
        print("\n\n\n###### Generating Cobertura XML... ######")
        print(
            "###### GCOVR command: '{}'... ######\n".format(" ".join(_GCOVR_GENERATE_CMD))
        )
        with open("/dev/null", "w") as devnull:
            run(_GCOVR_GENERATE_CMD, stderr=devnull)

            codecov_dir = os.path.join(args.source_dir, ".codecov")
            codecov_xml = os.path.join(codecov_dir, f"{args.coverage}.xml")
            codecov_md = os.path.join(codecov_dir, f"{args.coverage}.md")

            PYCOBERTURA_CMD = which("pycobertura", require=False)
            if PYCOBERTURA_CMD:
                run(
                    [
                        PYCOBERTURA_CMD,
                        "show",
                        "--format",
                        "markdown",
                        "--output",
                        codecov_md,
                        codecov_xml,
                    ]
                )
