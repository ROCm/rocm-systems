# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Apply a patch idempotently.
#
# CMake's ExternalProject regenerates `*-patch-info.txt` on every reconfigure,
# which makes the patch step appear out-of-date and re-runs it. Vanilla
# `patch -p1` errors when its target is already patched, so subsequent
# reconfigures fail. Wrap the patch invocation with a reverse-dry-run probe
# so an already-applied patch is treated as success.
#
# Usage (from PATCH_COMMAND):
#   ${CMAKE_COMMAND}
#     -DSRC=<SOURCE_DIR>
#     -DPATCH=<absolute path to patch file>
#     -DPATCH_EXE=${PATCH_EXECUTABLE}
#     -P ${CMAKE_CURRENT_LIST_DIR}/apply_patch_idempotent.cmake

foreach(_arg SRC PATCH PATCH_EXE)
    if(NOT DEFINED ${_arg} OR "${${_arg}}" STREQUAL "")
        message(FATAL_ERROR "apply_patch_idempotent: -D${_arg}=<value> is required")
    endif()
endforeach()

execute_process(
    COMMAND ${PATCH_EXE} --dry-run --reverse --silent --force -p1 -d ${SRC} -i ${PATCH}
    RESULT_VARIABLE _reverse_check
    OUTPUT_QUIET
    ERROR_QUIET
)

if(_reverse_check EQUAL 0)
    message(STATUS "Patch already applied, skipping: ${PATCH}")
    return()
endif()

execute_process(
    COMMAND ${PATCH_EXE} -p1 -d ${SRC} -i ${PATCH}
    RESULT_VARIABLE _apply_result
)

if(NOT _apply_result EQUAL 0)
    message(FATAL_ERROR "Failed to apply patch ${PATCH} (exit ${_apply_result})")
endif()
