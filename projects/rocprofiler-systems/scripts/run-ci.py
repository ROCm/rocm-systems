#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
import re
import sys
import glob
import zlib
import base64
import socket
import shutil
import argparse
import multiprocessing
import xml.etree.ElementTree as ET

IS_GITHUB_ACTIONS = os.environ.get("GITHUB_ACTIONS") == "true"

# ---------------------------------------------------------------------------
# ANSI color helpers — GHA renders these natively in step logs
# ---------------------------------------------------------------------------

_ANSI = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "red": "\033[31m",
    "yellow": "\033[33m",
    "green": "\033[32m",
    "cyan": "\033[36m",
    "magenta": "\033[35m",
    "white": "\033[97m",
}


def _color(text, *codes):
    """Wrap *text* in ANSI codes when the output is not a dumb terminal."""
    if os.environ.get("NO_COLOR") or os.environ.get("TERM") == "dumb":
        return text
    prefix = "".join(_ANSI[c] for c in codes if c in _ANSI)
    return f"{prefix}{text}{_ANSI['reset']}"


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def log(msg, level="info"):
    if IS_GITHUB_ACTIONS:
        if level == "warning":
            print(f"::warning::{msg}", flush=True)
        elif level == "error":
            print(f"::error::{msg}", flush=True)
        else:
            print(msg, flush=True)
    else:
        if level == "warning":
            print(_color(f"[WARNING] {msg}", "yellow", "bold"), flush=True)
        elif level == "error":
            print(_color(f"[ERROR] {msg}", "red", "bold"), flush=True)
        else:
            print(f"[INFO] {msg}", flush=True)


def log_group_start(title):
    if IS_GITHUB_ACTIONS:
        print(f"::group::{title}", flush=True)
    else:
        bar = "=" * 60
        print(_color(f"\n{bar}\n{title}\n{bar}", "cyan", "bold"), flush=True)


def log_group_end():
    if IS_GITHUB_ACTIONS:
        print("::endgroup::", flush=True)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def which(cmd, require):
    v = shutil.which(cmd)
    if require and v is None:
        raise RuntimeError(f"{cmd} not found")
    return v if v is not None else ""


# ---------------------------------------------------------------------------
# CMake file generators
# ---------------------------------------------------------------------------


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

        set(CTEST_USE_LAUNCHERS TRUE)
        set(CMAKE_CTEST_ARGUMENTS --output-junit "{BINARY_DIR}/test-results.xml" {CTEST_ARGS})

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


def _script_preamble():
    """Common macro definitions included in every stage script."""
    return """
        include("${CMAKE_CURRENT_LIST_DIR}/dashboard_config.cmake")

        macro(safe_submit)
            ctest_submit(${ARGN} RETURN_VALUE _submit_ret CAPTURE_CMAKE_ERROR _submit_err)
            if(NOT _submit_ret EQUAL 0 OR NOT _submit_err EQUAL 0)
                message(WARNING "CDash submit failed (ret=${_submit_ret}, err=${_submit_err}), continuing...")
                if("$ENV{GITHUB_ACTIONS}" STREQUAL "true")
                    message("::warning::CDash submit failed (ret=${_submit_ret}, err=${_submit_err})")
                endif()
            endif()
        endmacro()

        macro(handle_error _message _ret)
            if(NOT ${${_ret}} EQUAL 0)
                safe_submit(PARTS Done)
                message(FATAL_ERROR "${_message} failed: ${${_ret}}")
            endif()
        endmacro()
        """


def _pytest_setup_block(binary_dir):
    """CMake snippet that discovers pytest and generates the config header."""
    return f"""
        set(_py_ver_flag "")
        set(_py_dir_flag "")
        set(_pytest_hints "")
        file(STRINGS "{binary_dir}/CMakeCache.txt" _cache_lines)
        foreach(_line IN LISTS _cache_lines)
            if(_line MATCHES "^ROCPROFSYS_PYTHON_VERSIONS:[A-Z]+=(.+)")
                string(REPLACE ";" "\\\\;" _pv "${{CMAKE_MATCH_1}}")
                set(_py_ver_flag "--python-versions=${{_pv}}")
            elseif(_line MATCHES "^ROCPROFSYS_PYTHON_ROOT_DIRS:[A-Z]+=(.+)")
                string(REPLACE ";" "\\\\;" _pd "${{CMAKE_MATCH_1}}")
                set(_py_dir_flag "--python-root-dirs=${{_pd}}")
                foreach(_root IN LISTS CMAKE_MATCH_1)
                    list(APPEND _pytest_hints "${{_root}}/bin")
                endforeach()
            endif()
        endforeach()

        find_program(_pytest_exe NAMES pytest HINTS ${{_pytest_hints}} REQUIRED)
        execute_process(
            COMMAND ${{_pytest_exe}}
                "{binary_dir}/share/rocprofiler-systems/tests/pytest"
                --show-config-only
                -p no:cacheprovider
                ${{_py_ver_flag}} ${{_py_dir_flag}}
            WORKING_DIRECTORY "{binary_dir}"
            COMMAND_ERROR_IS_FATAL ANY
        )

        set(ROCPROFSYS_PYTHON_HINTS "")
        if(NOT "${{_py_dir_flag}}" STREQUAL "")
            string(REGEX REPLACE "^--python-root-dirs=" "" _raw_roots "${{_py_dir_flag}}")
            string(REPLACE "\\\\;" ";" _root_list "${{_raw_roots}}")
            foreach(_root IN LISTS _root_list)
                list(APPEND ROCPROFSYS_PYTHON_HINTS "${{_root}}/bin" "${{_root}}")
            endforeach()
        endif()
        """


# ---------------------------------------------------------------------------
# Per-stage dashboard scripts
# ---------------------------------------------------------------------------


def generate_configure_script(args):
    """Stage 1: Start CDash session, update sources, configure.

    This is the only stage that calls ctest_start() without APPEND — it
    creates the TAG file in <binary_dir>/Testing/TAG that subsequent stages
    use to attach to the same CDash build entry.
    """
    SOURCE_DIR = os.path.realpath(args.source_dir)
    BINARY_DIR = os.path.realpath(args.binary_dir)
    DASHBOARD_MODE = args.mode

    return _script_preamble() + f"""
        ctest_start({DASHBOARD_MODE})
        ctest_update(SOURCE "{SOURCE_DIR}")
        ctest_configure(BUILD "{BINARY_DIR}" RETURN_VALUE _configure_ret)
        safe_submit(PARTS Start Update Configure)
        handle_error("Configure" _configure_ret)
        """


def generate_build_script(args):
    """Stage 2: Attach to existing CDash session (APPEND) and build.

    ctest_start(APPEND) reads the TAG file written by the configure stage
    so CDash knows to merge this into the same build entry.
    """
    BINARY_DIR = os.path.realpath(args.binary_dir)
    DASHBOARD_MODE = args.mode

    return _script_preamble() + f"""
        ctest_start({DASHBOARD_MODE} APPEND)
        ctest_build(BUILD "{BINARY_DIR}" RETURN_VALUE _build_ret)
        safe_submit(PARTS Build)
        handle_error("Build" _build_ret)
        """


def _ctest_test_params(ctest_args, binary_dir):
    """Parse ctest command-line args into cmake ctest_test() named parameters.

    Returns the inline keyword block (a string) for ctest_test().

    cmake 3.24+ ctest_test() has named parameters for filtering:
      -E  <regex>   -> EXCLUDE
      -R  <regex>   -> INCLUDE
      -LE <regex>   -> EXCLUDE_LABEL
      -L  <regex>   -> INCLUDE_LABEL

    OUTPUT_JUNIT is always set to produce the JUnit test-results file.
    """

    exclude = None
    include = None
    exclude_label = None
    include_label = None

    args_list = list(ctest_args or [])
    i = 0
    while i < len(args_list):
        a = args_list[i]
        if a == "-E" and i + 1 < len(args_list):
            exclude = args_list[i + 1]
            i += 2
        elif a == "-R" and i + 1 < len(args_list):
            include = args_list[i + 1]
            i += 2
        elif a == "-LE" and i + 1 < len(args_list):
            exclude_label = args_list[i + 1]
            i += 2
        elif a == "-L" and i + 1 < len(args_list):
            include_label = args_list[i + 1]
            i += 2
        elif a == "--repeat":
            # Repeat is handled by the outer ctest -S invocation in split
            # stages, matching the pre-refactor dashboard behavior.
            i += 1
            while i < len(args_list) and not args_list[i].startswith("-"):
                i += 1
        elif re.match(r"^(until-pass|until-fail|after-timeout):\d+$", a):
            i += 1
        else:
            i += 1

    # inline ctest_test() keyword arguments
    kw_lines = [f'OUTPUT_JUNIT "{binary_dir}/test-results.xml"']
    if exclude:
        kw_lines.append(f'EXCLUDE "{exclude}"')
    if include:
        kw_lines.append(f'INCLUDE "{include}"')
    if exclude_label:
        kw_lines.append(f'EXCLUDE_LABEL "{exclude_label}"')
    if include_label:
        kw_lines.append(f'INCLUDE_LABEL "{include_label}"')

    return "\n            ".join(kw_lines)


def generate_test_script(args, ctest_args=None):
    """Stage 3: Attach to existing CDash session (APPEND), run tests, submit Done.

    CMAKE_CTEST_ARGUMENTS is NOT applied by ctest_test() in dashboard scripts
    (it is only used when cmake --build --target test is invoked).  Test
    filtering and JUnit output are passed via ctest_test()'s named parameters
    (cmake 3.24+) which directly map to ctest command-line flags.
    """
    CODECOV = 1 if args.coverage else 0
    BINARY_DIR = os.path.realpath(args.binary_dir)
    DASHBOARD_MODE = args.mode

    test_params = _ctest_test_params(ctest_args, BINARY_DIR)

    _cov_block = ""
    if CODECOV:
        _cov_block = f"""
        ctest_coverage(
            BUILD "{BINARY_DIR}"
            RETURN_VALUE _coverage_ret
            CAPTURE_CMAKE_ERROR _coverage_err)
        safe_submit(PARTS Coverage)
        """

    return _script_preamble() + _pytest_setup_block(BINARY_DIR) + f"""
        ctest_start({DASHBOARD_MODE} APPEND)
        ctest_test(BUILD "{BINARY_DIR}"
            {test_params}
            RETURN_VALUE _test_ret)
        safe_submit(PARTS Test)
        {_cov_block}
        handle_error("Testing" _test_ret)
        safe_submit(PARTS Done)
        """


def generate_all_script(args, ctest_args=None):
    """Legacy single-script mode (all stages in one ctest -S invocation)."""
    CODECOV = 1 if args.coverage else 0
    DASHBOARD_MODE = args.mode
    SOURCE_DIR = os.path.realpath(args.source_dir)
    BINARY_DIR = os.path.realpath(args.binary_dir)

    test_params = _ctest_test_params(ctest_args, BINARY_DIR)

    _cov_block = ""
    if CODECOV:
        _cov_block = f"""
        ctest_coverage(
            BUILD "{BINARY_DIR}"
            RETURN_VALUE _coverage_ret
            CAPTURE_CMAKE_ERROR _coverage_err)
        safe_submit(PARTS Coverage)
        """

    return (
        _script_preamble()
        + f"""
        ctest_start({DASHBOARD_MODE})
        ctest_update(SOURCE "{SOURCE_DIR}")
        ctest_configure(BUILD "{BINARY_DIR}" RETURN_VALUE _configure_ret)
        safe_submit(PARTS Start Update Configure)
        handle_error("Configure" _configure_ret)

        ctest_build(BUILD "{BINARY_DIR}" RETURN_VALUE _build_ret)
        safe_submit(PARTS Build)
        handle_error("Build" _build_ret)
        """
        + _pytest_setup_block(BINARY_DIR)
        + f"""
        ctest_test(BUILD "{BINARY_DIR}"
            {test_params}
            RETURN_VALUE _test_ret)
        safe_submit(PARTS Test)
        {_cov_block}
        handle_error("Testing" _test_ret)
        safe_submit(PARTS Done)
        """
    )


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

_VALID_STAGES = ["generate", "configure", "build", "test", "all"]

_DASHBOARD_STAGES = [
    "Start",
    "Update",
    "Configure",
    "Build",
    "Test",
    "MemCheck",
    "Coverage",
    "Submit",
]


def parse_cdash_args(args):
    BUILD_JOBS = multiprocessing.cpu_count()
    SOURCE_DIR = os.getcwd()
    BINARY_DIR = os.path.join(SOURCE_DIR, "build")
    SITE = socket.gethostname()
    SUBMIT_URL = "my.cdash.org/submit.php?project=rocprofiler-systems"

    parser = argparse.ArgumentParser(
        description=(
            "CDash / CTest CI driver.\n\n"
            "Stages:\n"
            "  generate   — write dashboard_config.cmake + per-stage cmake scripts, no ctest run\n"
            "  configure  — ctest_start() + update + configure + CDash submit\n"
            "  build      — ctest_start(APPEND) + build + CDash submit\n"
            "  test       — ctest_start(APPEND) + test + coverage + CDash submit Done\n"
            "  all        — original single-step behaviour (default)\n\n"
            "Splitting into generate → configure → build → test allows separate\n"
            "GitHub Actions steps while keeping a single CDash build entry.\n"
            "The binary-dir must be shared between the split steps (it persists\n"
            "within a single GHA job automatically)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--stage",
        choices=_VALID_STAGES,
        default="all",
        help=(
            "Which stage to execute. Use 'generate' first to write cmake files,"
            " then run 'configure', 'build', 'test' as separate CI steps."
            " Defaults to 'all' for backward compatibility."
        ),
    )
    parser.add_argument(
        "-n", "--name", help="CDash build name", default=None, type=str, required=True
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
        default="Continuous",
        choices=("Continuous", "Nightly", "Experimental"),
        type=str,
    )
    parser.add_argument(
        "-T",
        "--stages",
        help="Dashboard stages (only used in 'all' mode)",
        nargs="+",
        default=_DASHBOARD_STAGES,
        choices=_DASHBOARD_STAGES,
        type=str,
    )
    parser.add_argument(
        "--submit-url", help="CDash submission site", default=SUBMIT_URL, type=str
    )
    parser.add_argument("--repeat-until-pass", default=3, type=int)
    parser.add_argument("--repeat-until-fail", default=None, type=int)
    parser.add_argument("--repeat-after-timeout", default=2, type=int)

    return parser.parse_args(args)


def parse_args(args=None):
    if args is None:
        args = sys.argv[1:]

    index = 0
    input_args, cmake_args, ctest_args = [], [], []
    data = [input_args, cmake_args, ctest_args]

    for itr in args:
        if itr == "--":
            index += 1
            if index > 2:
                raise RuntimeError("Usage: <options> -- <cmake-args> -- <ctest-args>")
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
    ctest_args += ["--repeat"] + repeat_args if repeat_args else []

    return [cdash_args, cmake_args, ctest_args]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def run(*args, **kwargs):
    import subprocess

    return subprocess.run(*args, **kwargs)


def set_python_hints_from_cmake_args(cmake_args):
    prefix = None
    envs = None
    for arg in cmake_args:
        if arg.startswith("-DROCPROFSYS_PYTHON_PREFIX="):
            prefix = arg.split("=", 1)[1]
        elif arg.startswith("-DROCPROFSYS_PYTHON_ENVS="):
            envs = arg.split("=", 1)[1].split(";")
    if prefix and envs:
        hints = ";".join(f"{prefix}/{env}/bin" for env in envs)
        os.environ["ROCPROFSYS_PYTHON_HINTS"] = hints
        log(f"ROCPROFSYS_PYTHON_HINTS={hints}")
        # In split-stage mode each stage is a separate process, so the env
        # var set above would not survive into the test stage.  Writing to
        # $GITHUB_ENV makes GitHub Actions propagate it to all subsequent
        # steps in the job — including the ctest invocation that needs it
        # to locate pytest inside the conda environments.
        github_env = os.environ.get("GITHUB_ENV")
        if github_env:
            with open(github_env, "a") as f:
                f.write(f"ROCPROFSYS_PYTHON_HINTS={hints}\n")


# ---------------------------------------------------------------------------
# Failure log surfacing
#
# ctest_configure()/ctest_test() always write Testing/Temporary/Last*.log
# with the real ESC bytes intact, regardless of outcome. ctest_build() does
# NOT: under CTEST_USE_LAUNCHERS (set in dashboard_config.cmake), a failed
# build writes no LastBuild*.log at all — the failing rule's stdout/stderr
# lives only inline in Testing/<TAG>/Build.xml, where CTest's XML writer has
# already replaced every non-XML-safe control byte (including the ESC byte
# that starts each compiler color code) with the literal text
# "[NON-XML-CHAR-0xNN]". Build failures therefore need XML recovery instead
# of a raw-log read; configure/test do not.
# ---------------------------------------------------------------------------

_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_NON_XML_CHAR_RE = re.compile(r"\[NON-XML-CHAR-0x([0-9A-Fa-f]{2})\]")

_RAW_LOG_GLOBS = {
    "configure": "LastConfigure*.log",
    "build": "LastBuild*.log",
    "test": "LastTest*.log",
}


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences when color output is disabled."""
    return _ANSI_ESCAPE_RE.sub("", text)


def _restore_non_xml_chars(text: str) -> str:
    """Reverse CTest's XML-safe escaping of non-XML control bytes."""
    return _NON_XML_CHAR_RE.sub(lambda m: chr(int(m.group(1), 16)), text)


def _newest_match(pattern: str):
    matches = glob.glob(pattern)
    return max(matches, key=os.path.getmtime) if matches else None


def _current_run_tag_dir(args):
    """Resolve Testing/<TAG> for the run currently in progress.

    CTest writes the tag name as the first line of Testing/TAG on
    ctest_start(), and every later ctest_start(APPEND) in the same job
    reuses it — this is the exact directory for the current run, unlike
    picking a Testing/*/ subdirectory by mtime.
    """
    tag_path = os.path.join(args.binary_dir, "Testing", "TAG")
    try:
        with open(tag_path, encoding="utf-8") as f:
            tag = f.readline().strip()
    except OSError:
        return None
    return os.path.join(args.binary_dir, "Testing", tag) if tag else None


def _read_raw_stage_log(args, stage_label: str):
    """Read Testing/Temporary/Last<Stage>*.log, or None if it doesn't exist.

    Written unconditionally (pass or fail) for configure/test. For build,
    only written on success — see _read_build_xml_failures for the failure
    case.
    """
    pattern = _RAW_LOG_GLOBS.get(stage_label)
    if pattern is None:
        return None
    path = _newest_match(os.path.join(args.binary_dir, "Testing", "Temporary", pattern))
    if path is None:
        return None
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


def _read_build_xml_failures(args):
    """Recover per-rule stdout/stderr for a failed build from CTest's
    launcher XML report (Testing/<TAG>/Build.xml), un-escaping the
    "[NON-XML-CHAR-0xNN]" tokens back to their original bytes. Returns a
    list of text blocks, one per <Failure>/<Warning> entry; empty if
    Build.xml doesn't exist (e.g. the build was never reached).
    """
    tag_dir = _current_run_tag_dir(args)
    if tag_dir is None:
        return []
    path = os.path.join(tag_dir, "Build.xml")
    if not os.path.isfile(path):
        return []
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError:
        return []

    blocks = []
    for entry in root.findall(".//Failure") + root.findall(".//Warning"):
        for tag in ("StdOut", "StdErr"):
            element = entry.find(f"./Result/{tag}")
            if element is not None and element.text:
                blocks.append(_restore_non_xml_chars(element.text))
    return blocks


def print_failure_log(args, stage_label: str) -> None:
    """Print the recovered CTest output for a failed stage so its ANSI
    color renders in the GitHub Actions step log."""
    if stage_label == "build":
        blocks = _read_build_xml_failures(args)
        if not blocks:
            return
        content = "\n".join(blocks)
        source = "Build.xml (recovered)"
    else:
        content = _read_raw_stage_log(args, stage_label)
        if content is None:
            return
        source = f"Last{stage_label.capitalize()}.log"

    if os.environ.get("NO_COLOR") or os.environ.get("TERM") == "dumb":
        content = _strip_ansi(content)

    log_group_start(f"{stage_label} log: {source}")
    print(content, flush=True)
    log_group_end()


def _count_diagnostics(text: str):
    """Count gcc/clang-style ' error:'/' warning:' markers in build text.

    Callers must pass ANSI-stripped text: with CMAKE_COLOR_DIAGNOSTICS on,
    gcc emits a literal space, then the color-start escape sequence, then
    the bare word "error"/"warning" — so " error:"/" warning:" never occurs
    as a contiguous substring in colored text, even though a human (or a
    color-aware terminal) reads it as one marker.
    """
    return text.count(" error:"), text.count(" warning:")


def annotate_build_diagnostics(args, failed: bool) -> None:
    """Emit a "::warning::Build: N error(s), M warning(s)" GitHub Actions
    annotation, replacing the identical shell step every calling workflow
    used to duplicate. That step grepped Testing/Temporary/LastBuild*.log
    directly, which is silently wrong on failure (see module docstring
    above) — this covers both outcomes from a single place.
    """
    if failed:
        text = "\n".join(_read_build_xml_failures(args))
    else:
        text = _read_raw_stage_log(args, "build") or ""
    if not text:
        return

    errors, warnings = _count_diagnostics(_strip_ansi(text))
    if errors + warnings > 0:
        log(
            f"Build: {errors} error(s), {warnings} warning(s) — expand step for details",
            level="warning",
        )


# ---------------------------------------------------------------------------
# Skipped-test reporting
#
# CTest only reports "<test> (Skipped)" with no reason. The real reason is
# recorded by the pytest harness as a CDash "Completion Status" measurement
# (or, failing that, embedded in pytest's own captured output) inside
# Testing/<TAG>/Test.xml, base64/gzip-encoded per CDash's XML schema.
# ---------------------------------------------------------------------------

_SKIP_REASON_PATTERNS = (
    # pytest's untruncated summary line: "SKIPPED [1] file:line: reason", or
    # "SKIPPED [1] file: reason" (no line number for collection-time skips).
    re.compile(r"SKIPPED \[\d+\][^:\n]+:(?:\d+:)?\s*(.+)"),
    # Fallback: the verbose progress line, which pytest truncates with "..."
    # to the terminal width.
    re.compile(r"SKIPPED \(([^)]*)\)"),
)


def _gzip_decompress_any(raw: bytes):
    """Try each zlib/gzip wbits convention CTest's compressed payloads use."""
    for wbits in (zlib.MAX_WBITS | 16, zlib.MAX_WBITS, -zlib.MAX_WBITS):
        try:
            return zlib.decompress(raw, wbits)
        except zlib.error:
            continue
    return None


def _decode_ctest_value(value_el) -> str:
    """Decode a CTest <Value>: base64+gzip for Measurements, plain text for
    NamedMeasurements (no "encoding" attribute)."""
    text = value_el.text or ""
    if value_el.get("encoding") != "base64":
        return text
    try:
        raw = base64.b64decode(text)
    except (ValueError, TypeError):
        return ""
    if value_el.get("compression"):
        raw = _gzip_decompress_any(raw)
        if raw is None:
            return ""
    return raw.decode("utf-8", errors="replace")


def _completion_status(test_el) -> str:
    """Return the "Completion Status" NamedMeasurement's value, if present."""
    measurement = next(
        (
            nm
            for nm in test_el.iter("NamedMeasurement")
            if nm.get("name") == "Completion Status"
        ),
        None,
    )
    return (
        (measurement.findtext("Value") or "").strip() if measurement is not None else ""
    )


def _reason_from_output(output: str) -> str:
    """Extract a skip reason from pytest's captured output, preferring the
    untruncated summary line over the truncated progress line."""
    for pattern in _SKIP_REASON_PATTERNS:
        match = pattern.search(output)
        if match:
            return match.group(1).strip()
    return ""


def _skip_reason(test_el) -> str:
    """Resolve a skipped test's reason: prefer CDash's "Completion Status"
    measurement; fall back to parsing pytest's captured output if it's
    missing or just the generic SKIP_RETURN_CODE placeholder."""
    reason = _completion_status(test_el)
    if reason and not reason.startswith("SKIP_RETURN_CODE"):
        return reason
    output = "\n".join(_decode_ctest_value(v) for v in test_el.iter("Value"))
    return _reason_from_output(output) or reason or "(no reason captured)"


def _collect_skip_reasons(xml_path) -> dict:
    """Map skipped test name -> reason, from a single Test.xml."""
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        return {}
    return {
        test.findtext("Name") or "<unknown>": _skip_reason(test)
        for test in root.iter("Test")
        if test.get("Status") == "notrun"
    }


def summarize_skipped_tests(args) -> None:
    """Print a concise "test -> skip reason" section from CTest's Test.xml."""
    tag_dir = _current_run_tag_dir(args)
    if tag_dir is None:
        return
    xml_path = os.path.join(tag_dir, "Test.xml")
    if not os.path.isfile(xml_path):
        return

    reasons = _collect_skip_reasons(xml_path)
    if not reasons:
        return

    log_group_start(f"Skipped tests ({len(reasons)})")
    width = max(len(name) for name in reasons)
    for name in sorted(reasons):
        log(f"{name.ljust(width)}  {reasons[name]}")
    log_group_end()


def collect_test_artifacts(args, ctest_args, print_xml: bool = True):
    log_group_start("Collecting dashboard artifacts")
    if "-VV" not in ctest_args:
        # Session-management files that are never useful as CI artifacts.
        _skip_artifact = {"TAG", "LastStart.log"}
        for file in glob.glob(
            os.path.join(args.binary_dir, "Testing/**"), recursive=True
        ):
            if not os.path.isfile(file):
                continue
            oname = os.path.basename(file)
            if oname in _skip_artifact:
                continue
            log(f"Reading {file}")
            with open(file) as inpf:
                fdata = inpf.read()
            # Only print CDash XML files to stdout — they contain the
            # configure/build/test summaries that are meaningful in CI logs.
            # Temporary/*.log files (LastTest, LastBuild, etc.) are verbose
            # raw output; upload them as artifacts but don't flood the log.
            # On failure print_xml is False: print_failure_log() already
            # showed the colored raw log, so the escaped-ANSI XML dump would
            # be redundant noise.
            if file.endswith(".xml") and print_xml:
                print(fdata)
            if not oname.endswith(".log"):
                oname += ".log"
            out = os.path.join(args.binary_dir, oname)
            with open(out, "w") as outf:
                log(f"Writing {oname}")
                outf.write(fdata)
    log_group_end()


# ---------------------------------------------------------------------------
# Stage execution
# ---------------------------------------------------------------------------


def do_generate(args, cmake_args, ctest_args):
    """Write dashboard_config.cmake and all stage scripts to the binary dir.

    dashboard_config.cmake holds the CDash connection settings and build
    commands.  It is written here and included by every stage script.
    It is intentionally NOT named CTestCustom.cmake: cmake's configure_file()
    in CMakeLists.txt copies cmake/CTestCustom.cmake (error limits only) to
    build/CTestCustom.cmake, which would overwrite these CDash settings and
    cause ctest_start(APPEND) to fail in the build and test stages.
    """
    from textwrap import dedent

    if not os.path.exists(args.binary_dir):
        os.makedirs(args.binary_dir)

    config_path = os.path.join(args.binary_dir, "dashboard_config.cmake")
    with open(config_path, "w") as f:
        f.write(dedent(generate_custom(args, cmake_args, ctest_args)) + "\n")
    log(f"Wrote {config_path}")

    scripts = {
        "dashboard_configure.cmake": generate_configure_script(args),
        "dashboard_build.cmake": generate_build_script(args),
        "dashboard_test.cmake": generate_test_script(args, ctest_args),
        "dashboard.cmake": generate_all_script(args, ctest_args),  # kept for compat
    }
    for name, content in scripts.items():
        path = os.path.join(args.binary_dir, name)
        with open(path, "w") as f:
            f.write(dedent(content) + "\n")
        log(f"Wrote {path}")


def do_stage(args, script_name, stage_label, ctest_args=None):
    CTEST_CMD = which("ctest", require=True)
    script_path = os.path.join(args.binary_dir, script_name)

    if not os.path.exists(script_path):
        raise RuntimeError(f"{script_path} not found — run '--stage generate' first")

    cmd = [CTEST_CMD, "-S", script_path, "--output-on-failure", "-V"]
    if stage_label == "test":
        cmd += list(ctest_args or [])

    log_group_start(f"Running CTest stage: {stage_label}")
    log(_color(f"Command: {' '.join(cmd)}", "white"))
    log_group_end()

    failed = False
    try:
        run(cmd, check=True)
    except Exception as e:
        failed = True
        log(f"CTest {stage_label} failed: {e}", level="error")
        raise
    finally:
        if failed:
            print_failure_log(args, stage_label)
            collect_test_artifacts(
                args,
                ctest_args or [] if stage_label == "test" else [],
                print_xml=False,
            )
        elif stage_label == "test":
            collect_test_artifacts(args, ctest_args or [], print_xml=True)

        if stage_label == "build":
            annotate_build_diagnostics(args, failed)
        elif stage_label == "test":
            summarize_skipped_tests(args)


def do_all(args, ctest_args):
    """Original single-step mode, retained for backward compatibility."""
    CTEST_CMD = which("ctest", require=True)
    script_path = os.path.join(args.binary_dir, "dashboard.cmake")

    dashboard_args = ["-D"]
    for itr in args.stages:
        dashboard_args.append(f"{args.mode}{itr}")

    cmd = (
        [CTEST_CMD]
        + dashboard_args
        + ["-S", script_path, "--output-on-failure", "-V"]
        + ctest_args
    )

    log_group_start("Running CTest dashboard (all stages)")
    log(_color(f"Command: {' '.join(cmd)}", "white"))
    log_group_end()

    try:
        run(cmd, check=True)
    except Exception as e:
        log(f"CTest dashboard failed: {e}", level="error")
        raise
    finally:
        collect_test_artifacts(args, ctest_args)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args, cmake_args, ctest_args = parse_args()

    if not os.path.exists(args.binary_dir):
        os.makedirs(args.binary_dir)

    set_python_hints_from_cmake_args(cmake_args)

    log_group_start("CTest CI configuration")
    log(_color(f"Stage:      {args.stage}", "green", "bold"))
    log(f"Name:       {args.name}")
    log(f"Site:       {args.site}")
    log(f"Mode:       {args.mode}")
    log(f"Source dir: {os.path.realpath(args.source_dir)}")
    log(f"Binary dir: {os.path.realpath(args.binary_dir)}")
    log(f"Build jobs: {args.build_jobs}")
    log(f"Coverage:   {args.coverage}")
    log(f"Submit URL: {args.submit_url}")
    if cmake_args:
        log(f"CMake args: {' '.join(cmake_args)}")
    if ctest_args:
        log(f"CTest args: {' '.join(ctest_args)}")
    log_group_end()

    # cmake files are written only when generating (or running 'all').
    # Subsequent stages (configure/build/test) use the already-written
    # scripts so they don't need cmake_args or ctest_args passed again.
    if args.stage in ("generate", "all"):
        do_generate(args, cmake_args, ctest_args)
    else:
        # Validate that generate was already run
        _stage_scripts = {
            "configure": "dashboard_configure.cmake",
            "build": "dashboard_build.cmake",
            "test": "dashboard_test.cmake",
        }
        expected = os.path.join(args.binary_dir, _stage_scripts[args.stage])
        if not os.path.exists(expected):
            raise RuntimeError(
                f"{expected} not found — run '--stage generate' before '{args.stage}'"
            )

    if args.stage == "generate":
        log(
            _color(
                "Files written. Run configure / build / test as separate steps.", "green"
            )
        )
    elif args.stage == "configure":
        do_stage(args, "dashboard_configure.cmake", "configure")
    elif args.stage == "build":
        do_stage(args, "dashboard_build.cmake", "build")
    elif args.stage == "test":
        do_stage(args, "dashboard_test.cmake", "test", ctest_args)
    else:  # "all"
        do_all(args, ctest_args)
