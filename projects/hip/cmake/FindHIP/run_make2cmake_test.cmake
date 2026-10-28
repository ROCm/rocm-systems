# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

###############################################################################
# Regression test for run_make2cmake.cmake.
#
# A dependency line that points at a file that no longer exists on disk must
# be dropped from HIP_HIPCC_DEPEND, not replaced by the current working
# directory. See the sibling bug fixed upstream in CMake's own
# FindCUDA/make2cmake.cmake, guarded there by "if(file AND NOT IS_DIRECTORY
# ...)": IS_DIRECTORY on an empty string returns FALSE, so a missing
# dependency that was reset to an empty string still passed that check here
# and get_filename_component(... ABSOLUTE) on the empty string resolved to
# CMAKE_CURRENT_SOURCE_DIR.
#
# Run with: cmake -P projects/hip/cmake/FindHIP/run_make2cmake_test.cmake
###############################################################################

set(_test_dir "${CMAKE_CURRENT_LIST_DIR}/run_make2cmake_test_tmp")
file(REMOVE_RECURSE "${_test_dir}")
file(MAKE_DIRECTORY "${_test_dir}")

set(_existing_file "${_test_dir}/exists.h")
file(WRITE "${_existing_file}" "")

set(_missing_file "${_test_dir}/missing.h")
# Deliberately not created: this is the dependency that should be dropped.

set(_input_file "${_test_dir}/dep.d")
file(WRITE "${_input_file}" "foo.o: ${_existing_file} \\\n ${_missing_file}\n")

set(_output_file "${_test_dir}/out.cmake")

execute_process(
    COMMAND "${CMAKE_COMMAND}"
        -D "input_file=${_input_file}"
        -D "output_file=${_output_file}"
        -P "${CMAKE_CURRENT_LIST_DIR}/run_make2cmake.cmake"
    RESULT_VARIABLE _result
    ERROR_VARIABLE _stderr
)
if(NOT _result EQUAL 0)
    message(FATAL_ERROR "run_make2cmake.cmake exited with ${_result}: ${_stderr}")
endif()

set(HIP_HIPCC_DEPEND)
include("${_output_file}")

list(LENGTH HIP_HIPCC_DEPEND _n)
if(NOT _n EQUAL 1)
    message(FATAL_ERROR "expected exactly one dependency after dropping the missing file, got ${_n}: ${HIP_HIPCC_DEPEND}")
endif()

list(GET HIP_HIPCC_DEPEND 0 _only)
if(NOT _only STREQUAL "${_existing_file}")
    message(FATAL_ERROR "expected the only dependency to be ${_existing_file}, got ${_only}")
endif()

file(REMOVE_RECURSE "${_test_dir}")
message(STATUS "run_make2cmake_test.cmake: PASS")
