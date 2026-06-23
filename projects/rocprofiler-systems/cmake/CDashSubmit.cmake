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
set(CTEST_BUILD_NAME "${BUILD_NAME}")

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
)
foreach(_f IN LISTS _xml_files)
    file(READ "${_f}" _content)
    string(REGEX REPLACE "${_esc}\\[[0-9;]*[mGKHJFABCDEFnsr]" "" _content "${_content}")
    file(WRITE "${_f}" "${_content}")
endforeach()

# APPEND: reuse the TAG opened by configure's ctest -T Start rather than
# creating a new one, so Build and Test results share one CDash entry.
ctest_start(Continuous APPEND)

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
