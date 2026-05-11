# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

if(NOT DEFINED GIT_EXECUTABLE OR NOT GIT_EXECUTABLE)
  message(FATAL_ERROR "GIT_EXECUTABLE is required")
endif()
if(NOT DEFINED REPO_ROOT OR NOT REPO_ROOT)
  message(FATAL_ERROR "REPO_ROOT is required")
endif()
if(NOT DEFINED GENERATED_DIR OR NOT GENERATED_DIR)
  message(FATAL_ERROR "GENERATED_DIR is required")
endif()

execute_process(
  COMMAND "${GIT_EXECUTABLE}" diff --quiet --exit-code -- "${GENERATED_DIR}"
  WORKING_DIRECTORY "${REPO_ROOT}"
  RESULT_VARIABLE tracked_result)
if(NOT tracked_result EQUAL 0 AND NOT tracked_result EQUAL 1)
  message(FATAL_ERROR "git diff failed while checking ${GENERATED_DIR}")
endif()

execute_process(
  COMMAND "${GIT_EXECUTABLE}" ls-files --others --exclude-standard -- "${GENERATED_DIR}"
  WORKING_DIRECTORY "${REPO_ROOT}"
  RESULT_VARIABLE untracked_result
  OUTPUT_VARIABLE untracked_output
  ERROR_VARIABLE untracked_error
  OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT untracked_result EQUAL 0)
  message(FATAL_ERROR "git ls-files failed while checking ${GENERATED_DIR}: ${untracked_error}")
endif()

if(tracked_result EQUAL 1 OR untracked_output)
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" status --short -- "${GENERATED_DIR}"
    WORKING_DIRECTORY "${REPO_ROOT}"
    RESULT_VARIABLE status_result
    OUTPUT_VARIABLE status_output
    ERROR_VARIABLE status_error
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(NOT status_result EQUAL 0)
    set(status_output "git status failed while checking ${GENERATED_DIR}: ${status_error}")
  endif()
  message(FATAL_ERROR
    "AMDGPU ISA codegen changed checked-in generated sources.\n"
    "Run the codegen target locally, review the generated diff, and commit the updates.\n"
    "${status_output}")
endif()