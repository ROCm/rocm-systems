# cdash-submit.cmake — submit existing ctest results to CDash.
#
# Called after an explicit `ctest --test-dir <dir>` run so that configure/
# build/test are separate GitHub Actions steps while CDash still receives
# the test report.
#
# Usage:
#   cmake -DBUILD_DIR=<rel-or-abs-build-dir> \
#         -DBUILD_NAME=<cdash-build-name>     \
#         -DSITE=<runner-hostname>             \
#         -P cmake/CDashSubmit.cmake
#
# The script uses APPEND mode so it attaches to the TAG written by the
# preceding ctest run rather than creating a new one.

cmake_minimum_required(VERSION 3.15)

# Resolve paths relative to this script's location (the cmake/ sub-directory).
cmake_path(SET _src_dir NORMALIZE "${CMAKE_CURRENT_LIST_DIR}/..")

if(NOT DEFINED BUILD_DIR OR BUILD_DIR STREQUAL "")
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

# APPEND: reuse the TAG created by the preceding ctest run, don't start fresh.
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

safe_submit(PARTS Test Done)
