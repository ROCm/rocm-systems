# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# ------------------------------------------------------------------------------
# Standalone script that performs a monorepo sparse checkout of profiler-hub.
# Invoked via `cmake -P` from ExternalProject_Add's DOWNLOAD_COMMAND so the
# git operations run at build time instead of blocking configure.
#
# Required -D arguments:
#   GIT_EXECUTABLE   path to the git binary
#   REPO_URL         git repository URL to clone
#   GIT_TAG          git tag/branch to check out
#   GIT_SUBDIR       subdirectory inside the repository containing profiler-hub
#   CHECKOUT_DIR     destination directory for the sparse checkout
#   STAMP_FILE       file used to detect whether the existing checkout is
#                    already up to date, avoiding redundant re-clones
# ------------------------------------------------------------------------------

foreach(
    _required_var
    GIT_EXECUTABLE
    REPO_URL
    GIT_TAG
    GIT_SUBDIR
    CHECKOUT_DIR
    STAMP_FILE
)
    if(NOT DEFINED ${_required_var})
        message(FATAL_ERROR "[profiler-hub] Missing required argument: ${_required_var}")
    endif()
endforeach()

set(_PROFILER_HUB_SOURCE_DIR "${CHECKOUT_DIR}/${GIT_SUBDIR}")

set(_PROFILER_HUB_NEEDS_CHECKOUT TRUE)
if(EXISTS "${STAMP_FILE}")
    file(READ "${STAMP_FILE}" _stamp_contents)
    if(_stamp_contents STREQUAL "${REPO_URL}@${GIT_TAG}:${GIT_SUBDIR}")
        set(_PROFILER_HUB_NEEDS_CHECKOUT FALSE)
    endif()
endif()

if(_PROFILER_HUB_NEEDS_CHECKOUT)
    message(
        STATUS
        "[profiler-hub] Sparse-checking out ${GIT_SUBDIR} into ${CHECKOUT_DIR}"
    )

    if(EXISTS "${CHECKOUT_DIR}")
        file(REMOVE_RECURSE "${CHECKOUT_DIR}")
    endif()
    file(MAKE_DIRECTORY "${CHECKOUT_DIR}")

    # ExternalProject_Add runs this script's DOWNLOAD_COMMAND with CHECKOUT_DIR
    # itself as the process's working directory. CHECKOUT_DIR was just removed
    # and recreated above, which invalidates that cwd (getcwd() fails), so the
    # clone step needs an explicit, unrelated working directory rather than
    # inheriting the (now-stale) default.
    execute_process(
        COMMAND
            ${GIT_EXECUTABLE} clone --filter=blob:none --no-checkout --depth 1 --branch
            ${GIT_TAG} ${REPO_URL} ${CHECKOUT_DIR}
        WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
        RESULT_VARIABLE _git_result
        OUTPUT_VARIABLE _git_output
        ERROR_VARIABLE _git_error
    )
    if(NOT _git_result EQUAL 0)
        message(
            FATAL_ERROR
            "[profiler-hub] git clone failed (${_git_result}): ${_git_error}\n${_git_output}"
        )
    endif()

    execute_process(
        COMMAND ${GIT_EXECUTABLE} sparse-checkout init --cone
        WORKING_DIRECTORY ${CHECKOUT_DIR}
        RESULT_VARIABLE _git_result
        ERROR_VARIABLE _git_error
    )
    if(NOT _git_result EQUAL 0)
        message(FATAL_ERROR "[profiler-hub] sparse-checkout init failed: ${_git_error}")
    endif()

    execute_process(
        COMMAND ${GIT_EXECUTABLE} sparse-checkout set ${GIT_SUBDIR}
        WORKING_DIRECTORY ${CHECKOUT_DIR}
        RESULT_VARIABLE _git_result
        ERROR_VARIABLE _git_error
    )
    if(NOT _git_result EQUAL 0)
        message(FATAL_ERROR "[profiler-hub] sparse-checkout set failed: ${_git_error}")
    endif()

    execute_process(
        COMMAND ${GIT_EXECUTABLE} checkout ${GIT_TAG}
        WORKING_DIRECTORY ${CHECKOUT_DIR}
        RESULT_VARIABLE _git_result
        ERROR_VARIABLE _git_error
    )
    if(NOT _git_result EQUAL 0)
        message(FATAL_ERROR "[profiler-hub] git checkout failed: ${_git_error}")
    endif()

    if(NOT EXISTS "${_PROFILER_HUB_SOURCE_DIR}/CMakeLists.txt")
        message(
            FATAL_ERROR
            "[profiler-hub] Sparse checkout completed but CMakeLists.txt missing at "
            "${_PROFILER_HUB_SOURCE_DIR}"
        )
    endif()

    file(WRITE "${STAMP_FILE}" "${REPO_URL}@${GIT_TAG}:${GIT_SUBDIR}")
else()
    message(
        STATUS
        "[profiler-hub] Reusing existing sparse checkout at ${_PROFILER_HUB_SOURCE_DIR}"
    )
endif()
