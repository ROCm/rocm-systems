# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

include_guard(GLOBAL)

# ----------------------------------------------------------------------------------------#
# function rocprofiler_compute_checkout_git_submodule()
#
# Ensures a git submodule (or external repo) is checked out. If TEST_FILE exists in the
# submodule directory, returns immediately. Otherwise:
#
#   - If .gitmodules exists: runs "git submodule update --init" for the submodule.
#   - If REPO_URL is provided: clones the repo with "git clone -b REPO_BRANCH", then
#     optionally runs "git submodule update --init --recursive" when RECURSIVE is set.
#
# ARGS:
#   RECURSIVE (option)       -- add "--recursive" to git submodule update
#   RELATIVE_PATH (one)      -- path to submodule, relative to WORKING_DIRECTORY
#   WORKING_DIRECTORY (one)  -- base directory (default: PROJECT_SOURCE_DIR)
#   TEST_FILE (one)          -- file whose existence indicates checkout (default: CMakeLists.txt)
#   REPO_URL (one)           -- fallback: clone this URL when submodule dir missing
#   REPO_BRANCH (one)        -- branch for REPO_URL clone (default: master)
#   ADDITIONAL_CMDS (many)   -- extra args passed to git submodule update or clone
#
function(ROCPROFILER_COMPUTE_CHECKOUT_GIT_SUBMODULE)
    # parse args
    cmake_parse_arguments(
        CHECKOUT
        "RECURSIVE"
        "RELATIVE_PATH;WORKING_DIRECTORY;TEST_FILE;REPO_URL;REPO_BRANCH"
        "ADDITIONAL_CMDS"
        ${ARGN}
    )

    if(NOT CHECKOUT_WORKING_DIRECTORY)
        set(CHECKOUT_WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
    endif()

    if(NOT CHECKOUT_TEST_FILE)
        set(CHECKOUT_TEST_FILE "CMakeLists.txt")
    endif()

    # default assumption
    if(NOT CHECKOUT_REPO_BRANCH)
        set(CHECKOUT_REPO_BRANCH "master")
    endif()

    find_package(Git)
    set(_DIR "${CHECKOUT_WORKING_DIRECTORY}/${CHECKOUT_RELATIVE_PATH}")
    # ensure the (possibly empty) directory exists
    if(NOT EXISTS "${_DIR}")
        if(NOT CHECKOUT_REPO_URL)
            message(FATAL_ERROR "submodule directory does not exist")
        endif()
    endif()

    # if this file exists --> project has been checked out if not exists --> not been
    # checked out
    set(_TEST_FILE "${_DIR}/${CHECKOUT_TEST_FILE}")

    set(_TEST_FILE_EXISTS OFF)
    if(EXISTS "${_TEST_FILE}" AND NOT IS_DIRECTORY "${_TEST_FILE}")
        set(_TEST_FILE_EXISTS ON)
    endif()

    if(_TEST_FILE_EXISTS)
        return()
    endif()

    find_package(Git REQUIRED)

    # .gitmodules lives at the monorepo root, not under PROJECT_SOURCE_DIR, because
    # rocprofiler-compute may be part of a monorepo. Use git rev-parse to locate the root
    # (result cached in CMakeCache.txt after first configure run).
    if(NOT DEFINED CACHE{ROCPROFILER_COMPUTE_GIT_TOPLEVEL})
        execute_process(
            COMMAND ${GIT_EXECUTABLE} rev-parse --show-toplevel
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
            OUTPUT_VARIABLE _GIT_TOPLEVEL
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE _GIT_TOPLEVEL_RET
        )
        if(NOT _GIT_TOPLEVEL_RET EQUAL 0)
            message(
                FATAL_ERROR
                "Failed to determine git top-level directory. "
                "Ensure this project is inside a git repository."
            )
        endif()
        set(ROCPROFILER_COMPUTE_GIT_TOPLEVEL
            "${_GIT_TOPLEVEL}"
            CACHE INTERNAL
            "Git top-level directory"
        )
    endif()
    set(_SUBMODULE "${ROCPROFILER_COMPUTE_GIT_TOPLEVEL}/.gitmodules")

    set(_SUBMODULE_EXISTS OFF)
    if(EXISTS "${_SUBMODULE}" AND NOT IS_DIRECTORY "${_SUBMODULE}")
        set(_SUBMODULE_EXISTS ON)
    endif()

    set(_HAS_REPO_URL OFF)
    if(NOT "${CHECKOUT_REPO_URL}" STREQUAL "")
        set(_HAS_REPO_URL ON)
    endif()

    set(_RECURSE "")
    if(CHECKOUT_RECURSIVE)
        set(_RECURSE "--recursive")
    endif()

    # if the module has not been checked out
    if(NOT _TEST_FILE_EXISTS AND _SUBMODULE_EXISTS)
        # perform the checkout
        execute_process(
            COMMAND
                ${GIT_EXECUTABLE} submodule update --init ${_RECURSE}
                ${CHECKOUT_ADDITIONAL_CMDS} ${CHECKOUT_RELATIVE_PATH}
            WORKING_DIRECTORY ${CHECKOUT_WORKING_DIRECTORY}
            RESULT_VARIABLE RET
        )

        # check the return code
        if(RET GREATER 0)
            set(_CMD
                "${GIT_EXECUTABLE} submodule update --init ${_RECURSE}
                ${CHECKOUT_ADDITIONAL_CMDS} ${CHECKOUT_RELATIVE_PATH}"
            )
            message(STATUS "function(rocprofiler_compute_checkout_git_submodule) failed.")
            message(FATAL_ERROR "Command: \"${_CMD}\"")
        else()
            set(_TEST_FILE_EXISTS ON)
        endif()
    endif()

    if(NOT _TEST_FILE_EXISTS AND _HAS_REPO_URL)
        message(
            STATUS
            "Checking out '${CHECKOUT_REPO_URL}' @ '${CHECKOUT_REPO_BRANCH}'..."
        )

        # remove the existing directory
        if(EXISTS "${_DIR}")
            execute_process(COMMAND ${CMAKE_COMMAND} -E remove_directory ${_DIR})
        endif()

        # perform the checkout
        execute_process(
            COMMAND
                ${GIT_EXECUTABLE} clone -b ${CHECKOUT_REPO_BRANCH}
                ${CHECKOUT_ADDITIONAL_CMDS} ${CHECKOUT_REPO_URL} ${CHECKOUT_RELATIVE_PATH}
            WORKING_DIRECTORY ${CHECKOUT_WORKING_DIRECTORY}
            RESULT_VARIABLE RET
        )

        # perform the submodule update
        if(CHECKOUT_RECURSIVE AND EXISTS "${_DIR}" AND IS_DIRECTORY "${_DIR}")
            execute_process(
                COMMAND ${GIT_EXECUTABLE} submodule update --init ${_RECURSE}
                WORKING_DIRECTORY ${_DIR}
                RESULT_VARIABLE RET
            )
        endif()

        # check the return code
        if(RET GREATER 0)
            set(_CMD
                "${GIT_EXECUTABLE} clone -b ${CHECKOUT_REPO_BRANCH}
                ${CHECKOUT_ADDITIONAL_CMDS} ${CHECKOUT_REPO_URL} ${CHECKOUT_RELATIVE_PATH}"
            )
            message(STATUS "function(rocprofiler_compute_checkout_git_submodule) failed.")
            message(FATAL_ERROR "Command: \"${_CMD}\"")
        else()
            set(_TEST_FILE_EXISTS ON)
        endif()
    endif()

    if(NOT EXISTS "${_TEST_FILE}" OR NOT _TEST_FILE_EXISTS)
        message(
            FATAL_ERROR
            "Error checking out submodule: '${CHECKOUT_RELATIVE_PATH}' to '${_DIR}'"
        )
    endif()
endfunction()
