# CDashSubmit.cmake — strip ANSI from CDash XML files then submit to CDash.
#
# Must be invoked via ctest -S (not cmake -P): ctest_start/ctest_submit are
# CTest dashboard functions unavailable in plain cmake script mode.
#
# Usage (from the rocprofiler-systems-test composite action):
#   export BUILD_DIR=<rel-or-abs-build-dir>
#   export BUILD_NAME=<cdash-build-name>
#   export SITE=<runner-hostname>
#   ctest -S cmake/CDashSubmit.cmake

cmake_minimum_required(VERSION 3.15)

# Resolve paths relative to this script's location (the cmake/ sub-directory).
cmake_path(SET _src_dir NORMALIZE "${CMAKE_CURRENT_LIST_DIR}/..")

# Read configuration from environment variables (ctest -S does not support -D).
if(NOT DEFINED BUILD_DIR OR BUILD_DIR STREQUAL "")
    set(BUILD_DIR "$ENV{BUILD_DIR}")
endif()
if(NOT DEFINED BUILD_NAME OR BUILD_NAME STREQUAL "")
    set(BUILD_NAME "$ENV{BUILD_NAME}")
endif()
if(NOT DEFINED SITE OR SITE STREQUAL "")
    set(SITE "$ENV{SITE}")
endif()

if(BUILD_DIR STREQUAL "")
    set(BUILD_DIR "build")
endif()

cmake_path(IS_ABSOLUTE BUILD_DIR _is_abs)
if(_is_abs)
    set(_bin_dir "${BUILD_DIR}")
else()
    cmake_path(SET _bin_dir NORMALIZE "${_src_dir}/${BUILD_DIR}")
endif()

set(CTEST_SOURCE_DIRECTORY "${_src_dir}")
set(CTEST_BINARY_DIRECTORY "${_bin_dir}")
set(CTEST_SITE "${SITE}")

# Transform PR build names to the CDash-friendly format used historically.
# GitHub Actions sets ref_name to "<pr_number>/merge" for pull requests, which
# embeds a slash that CDash cannot search. Apply the same transformation that
# run-ci.py used: <owner>-<PR>/merge-<suffix> -> PR_<PR>_<owner>-<suffix>
set(CTEST_BUILD_NAME "${BUILD_NAME}")
string(
    REGEX REPLACE
    "^(.*)-([0-9]+)/merge(.*)$"
    "PR_\\2_\\1\\3"
    CTEST_BUILD_NAME
    "${CTEST_BUILD_NAME}"
)

# Load CDash connection settings from the project's CTestConfig.cmake.
include("${_src_dir}/CTestConfig.cmake")

# Strip ANSI escape codes from CDash XML output files before submission.
# ESC (0x1B) is not valid in XML 1.0; without stripping it CDash renders
# colour codes as [NON-XML-CHAR-0x1B]. Terminal colour output in GitHub
# Actions is unaffected — only the XML that CDash receives is cleaned.
string(ASCII 27 _esc)
file(
    GLOB _xml_files
    "${_bin_dir}/Testing/*/Build.xml"
    "${_bin_dir}/Testing/*/Test.xml"
    "${_bin_dir}/Testing/*/Configure.xml"
    "${_bin_dir}/Testing/*/Update.xml"
)
foreach(_f IN LISTS _xml_files)
    file(READ "${_f}" _content)
    string(REGEX REPLACE "${_esc}\\[[0-9;]*[mGKHJFABCDEFnsr]" "" _content "${_content}")
    file(WRITE "${_f}" "${_content}")
endforeach()

# CDash model: Nightly for scheduled runs, Continuous for everything else
# (including pull_request) — matches the original run-ci.py behaviour where
# DASHBOARD_MODE defaulted to Continuous for all non-scheduled builds.
if("$ENV{GITHUB_EVENT_NAME}" STREQUAL "schedule")
    set(_cdash_model "Nightly")
else()
    set(_cdash_model "Continuous")
endif()

# APPEND: reuse the TAG opened by configure's ctest -T <Model>Start rather
# than creating a new one, so Update/Configure/Build/Test share one CDash entry.
ctest_start(${_cdash_model} APPEND)

macro(safe_submit)
    ctest_submit(${ARGN} RETURN_VALUE _ret CAPTURE_CMAKE_ERROR _err)
    if(NOT _ret EQUAL 0 OR NOT _err EQUAL 0)
        message(WARNING "CDash submit failed (ret=${_ret}, err=${_err}) — continuing")
        if("$ENV{GITHUB_ACTIONS}" STREQUAL "true")
            message("::warning::CDash submit failed (ret=${_ret}, err=${_err})")
        endif()
    endif()
endmacro()

safe_submit(PARTS Configure Build Test Done)
