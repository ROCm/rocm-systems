# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Generate a version header from the project version and, when this is a Git
# checkout, the exact repository revision it came from. Keep Git optional so
# source archives and package builds still configure successfully.

set(ROCJITSU_GIT_REVISION "unknown")
set(ROCJITSU_GIT_COMMIT_DATE "unknown")
set(ROCJITSU_GIT_COMMIT_TITLE "unknown")

find_package(Git QUIET)
if(Git_FOUND)
    # Make an ordinary build reconfigure after a commit or checkout. Ask Git for
    # the paths so linked worktrees and shared packed refs are handled without
    # assuming a .git directory layout.
    set(_rj_git_configure_inputs)
    foreach(_rj_git_path HEAD logs/HEAD packed-refs)
        execute_process(
            COMMAND
                "${GIT_EXECUTABLE}" -C "${PROJECT_SOURCE_DIR}" rev-parse
                --path-format=absolute --git-path "${_rj_git_path}"
            OUTPUT_VARIABLE _rj_git_resolved_path
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE _rj_git_path_result
            ERROR_QUIET
        )
        if(_rj_git_path_result EQUAL 0 AND EXISTS "${_rj_git_resolved_path}")
            list(APPEND _rj_git_configure_inputs "${_rj_git_resolved_path}")
        endif()
    endforeach()

    execute_process(
        COMMAND
            "${GIT_EXECUTABLE}" -C "${PROJECT_SOURCE_DIR}" symbolic-ref -q HEAD
        OUTPUT_VARIABLE _rj_git_head_ref
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _rj_git_head_ref_result
        ERROR_QUIET
    )
    if(_rj_git_head_ref_result EQUAL 0)
        execute_process(
            COMMAND
                "${GIT_EXECUTABLE}" -C "${PROJECT_SOURCE_DIR}" rev-parse
                --path-format=absolute --git-path "${_rj_git_head_ref}"
            OUTPUT_VARIABLE _rj_git_head_ref_path
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE _rj_git_head_ref_path_result
            ERROR_QUIET
        )
        if(
            _rj_git_head_ref_path_result EQUAL 0
            AND EXISTS "${_rj_git_head_ref_path}"
        )
            list(APPEND _rj_git_configure_inputs "${_rj_git_head_ref_path}")
        endif()
    endif()

    if(_rj_git_configure_inputs)
        list(REMOVE_DUPLICATES _rj_git_configure_inputs)
        set_property(
            DIRECTORY
            APPEND
            PROPERTY CMAKE_CONFIGURE_DEPENDS ${_rj_git_configure_inputs}
        )
    endif()

    execute_process(
        COMMAND
            "${GIT_EXECUTABLE}" -C "${PROJECT_SOURCE_DIR}" show -s --format=%H
            HEAD
        OUTPUT_VARIABLE ROCJITSU_GIT_REVISION
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _rj_git_revision_result
        ERROR_QUIET
    )
    execute_process(
        COMMAND
            "${GIT_EXECUTABLE}" -C "${PROJECT_SOURCE_DIR}" show -s --format=%cI
            HEAD
        OUTPUT_VARIABLE ROCJITSU_GIT_COMMIT_DATE
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _rj_git_date_result
        ERROR_QUIET
    )
    execute_process(
        COMMAND
            "${GIT_EXECUTABLE}" -C "${PROJECT_SOURCE_DIR}" show -s --format=%s
            HEAD
        OUTPUT_VARIABLE ROCJITSU_GIT_COMMIT_TITLE
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _rj_git_title_result
        ERROR_QUIET
    )

    if(NOT _rj_git_revision_result EQUAL 0)
        set(ROCJITSU_GIT_REVISION "unknown")
    endif()
    if(NOT _rj_git_date_result EQUAL 0)
        set(ROCJITSU_GIT_COMMIT_DATE "unknown")
    endif()
    if(NOT _rj_git_title_result EQUAL 0)
        set(ROCJITSU_GIT_COMMIT_TITLE "unknown")
    endif()
endif()

# These values are substituted into C string literals below. Commit subjects
# can contain backslashes or quotes, so escape both before configuring.
foreach(
    _rj_git_value
    ROCJITSU_GIT_REVISION
    ROCJITSU_GIT_COMMIT_DATE
    ROCJITSU_GIT_COMMIT_TITLE
)
    string(REPLACE "\\" "\\\\" ${_rj_git_value} "${${_rj_git_value}}")
    string(REPLACE "\"" "\\\"" ${_rj_git_value} "${${_rj_git_value}}")
endforeach()

configure_file(
    ${CMAKE_CURRENT_LIST_DIR}/version.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/include/rocjitsu/version.h
    @ONLY
)

# Expose the generated header path for targets that need it.
set(RJ_VERSION_INCLUDE_DIR
    ${CMAKE_CURRENT_BINARY_DIR}/include
    CACHE INTERNAL
    ""
)
