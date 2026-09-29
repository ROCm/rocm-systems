# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Run by `cmake -P` from the add_custom_command in sqlite3.cmake. A COMMAND
# list cannot branch on a result, so the clone's full-clone retry lives here.

if(CMAKE_HOST_WIN32)
    # Official pre-built amalgamation zip. No git, nmake, or Tcl.
    string(REGEX REPLACE "^version-" "" _sqlite3_version "${SQLITE3_GIT_TAG}")
    string(REPLACE "." ";" _sqlite3_version_parts "${_sqlite3_version}")
    list(LENGTH _sqlite3_version_parts _sqlite3_version_len)
    if(_sqlite3_version_len LESS 3)
        message(
            FATAL_ERROR
            "[profiler-hub] SQLITE3_GIT_TAG must look like version-M.m.p (got ${SQLITE3_GIT_TAG})"
        )
    endif()
    list(GET _sqlite3_version_parts 0 _sqlite3_major)
    list(GET _sqlite3_version_parts 1 _sqlite3_minor)
    list(GET _sqlite3_version_parts 2 _sqlite3_patch)
    math(
        EXPR
        _sqlite3_amalg_version
        "${_sqlite3_major} * 1000000 + ${_sqlite3_minor} * 10000 + ${_sqlite3_patch} * 100"
    )

    get_filename_component(
        _sqlite3_external_dir
        "${SQLITE3_SOURCE_DIR}"
        DIRECTORY
    )
    file(MAKE_DIRECTORY "${_sqlite3_external_dir}")
    set(_sqlite3_zip
        "${_sqlite3_external_dir}/sqlite-amalgamation-${_sqlite3_amalg_version}.zip"
    )
    set(_sqlite3_download_url
        "https://www.sqlite.org/${SQLITE3_AMALGAMATION_YEAR}/sqlite-amalgamation-${_sqlite3_amalg_version}.zip"
    )

    message(
        STATUS
        "[profiler-hub] Downloading SQLite3 amalgamation ${_sqlite3_version} from ${_sqlite3_download_url}"
    )
    file(
        DOWNLOAD "${_sqlite3_download_url}"
        "${_sqlite3_zip}"
        STATUS _sqlite3_download_status
    )
    list(GET _sqlite3_download_status 0 _sqlite3_download_rc)
    if(NOT _sqlite3_download_rc EQUAL 0)
        list(GET _sqlite3_download_status 1 _sqlite3_download_msg)
        message(
            FATAL_ERROR
            "[profiler-hub] SQLite3 amalgamation download failed (rc=${_sqlite3_download_rc}): ${_sqlite3_download_msg}"
        )
    endif()
    if(EXISTS "${SQLITE3_SOURCE_DIR}")
        file(REMOVE_RECURSE "${SQLITE3_SOURCE_DIR}")
    endif()
    file(
        ARCHIVE_EXTRACT
        INPUT "${_sqlite3_zip}"
        DESTINATION "${_sqlite3_external_dir}"
    )
    file(
        RENAME
            "${_sqlite3_external_dir}/sqlite-amalgamation-${_sqlite3_amalg_version}"
        "${SQLITE3_SOURCE_DIR}"
    )
    return()
endif()

if(EXISTS "${SQLITE3_SOURCE_DIR}")
    file(REMOVE_RECURSE "${SQLITE3_SOURCE_DIR}")
endif()

message(
    STATUS
    "[profiler-hub] Cloning SQLite3 from ${SQLITE3_GIT_URL} @ ${SQLITE3_GIT_TAG}"
)

execute_process(
    COMMAND
        ${GIT_EXECUTABLE} clone --depth 1 --filter=blob:none --branch
        ${SQLITE3_GIT_TAG} ${SQLITE3_GIT_URL} ${SQLITE3_SOURCE_DIR}
    RESULT_VARIABLE _sqlite3_clone_rc
)

if(NOT _sqlite3_clone_rc EQUAL 0)
    message(STATUS "[profiler-hub] Optimized clone failed; retrying full clone")
    if(EXISTS "${SQLITE3_SOURCE_DIR}")
        file(REMOVE_RECURSE "${SQLITE3_SOURCE_DIR}")
    endif()
    execute_process(
        COMMAND
            ${GIT_EXECUTABLE} clone --branch ${SQLITE3_GIT_TAG}
            ${SQLITE3_GIT_URL} ${SQLITE3_SOURCE_DIR}
        RESULT_VARIABLE _sqlite3_clone_rc
    )
endif()

if(NOT _sqlite3_clone_rc EQUAL 0)
    message(
        FATAL_ERROR
        "[profiler-hub] git clone of SQLite3 failed (rc=${_sqlite3_clone_rc})"
    )
endif()

message(STATUS "[profiler-hub] Generating SQLite3 amalgamation")

execute_process(
    COMMAND ./configure --disable-tcl
    WORKING_DIRECTORY ${SQLITE3_SOURCE_DIR}
    RESULT_VARIABLE _sqlite3_configure_rc
)

if(NOT _sqlite3_configure_rc EQUAL 0)
    message(
        FATAL_ERROR
        "[profiler-hub] SQLite3 ./configure failed (rc=${_sqlite3_configure_rc})"
    )
endif()

execute_process(
    COMMAND ${MAKE_COMMAND} sqlite3.c
    WORKING_DIRECTORY ${SQLITE3_SOURCE_DIR}
    RESULT_VARIABLE _sqlite3_make_rc
)

if(NOT _sqlite3_make_rc EQUAL 0)
    message(
        FATAL_ERROR
        "[profiler-hub] SQLite3 amalgamation generation failed (rc=${_sqlite3_make_rc})"
    )
endif()

if(
    NOT EXISTS "${SQLITE3_SOURCE_DIR}/sqlite3.c"
    OR NOT EXISTS "${SQLITE3_SOURCE_DIR}/sqlite3.h"
)
    message(
        FATAL_ERROR
        "[profiler-hub] SQLite3 amalgamation files not found after build"
    )
endif()
