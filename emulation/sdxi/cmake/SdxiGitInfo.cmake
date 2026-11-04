# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

include_guard(GLOBAL)

find_program(GIT_EXECUTABLE git)

set(SDXI_GIT_SHA_FULL
  "0000000000000000000000000000000000000000"
  CACHE STRING "Full git SHA of the current commit")
set(SDXI_GIT_SHA_SHORT
  "unknown"
  CACHE STRING "Short git SHA of the current commit")

if(GIT_EXECUTABLE)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} rev-parse HEAD
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        OUTPUT_VARIABLE sdxi_git_sha_full
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
        RESULT_VARIABLE sdxi_git_result
    )
    if(sdxi_git_result EQUAL 0)
        set(SDXI_GIT_SHA_FULL "${sdxi_git_sha_full}"
          CACHE STRING "Full git SHA of the current commit"
          FORCE)
        execute_process(
            COMMAND ${GIT_EXECUTABLE} rev-parse --short=8 HEAD
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
            OUTPUT_VARIABLE sdxi_git_sha_short
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
            RESULT_VARIABLE sdxi_git_short_result
        )
        if(sdxi_git_short_result EQUAL 0)
            set(SDXI_GIT_SHA_SHORT "${sdxi_git_sha_short}"
              CACHE STRING "Short git SHA of the current commit"
              FORCE)
        endif()
    endif()
endif()

message(STATUS "Git SHA: ${SDXI_GIT_SHA_SHORT} (${SDXI_GIT_SHA_FULL})")
