# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Run by `cmake -P` from the add_custom_command in sqlite3.cmake. A COMMAND
# list cannot branch on a result, so the clone's full-clone retry lives here.

if(CMAKE_HOST_WIN32)
    # Official pre-built amalgamation zip. No git, nmake, or Tcl.
    # SQLITE3_GIT_TAG still names the extracted directory inside the zip.
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

    # a previously staged zip is reused only when its contents still match
    if(EXISTS "${_sqlite3_zip}")
        file(SHA256 "${_sqlite3_zip}" _sha256)
        if(NOT "${_sha256}" STREQUAL "${SQLITE3_AMALGAMATION_SHA256}")
            file(REMOVE "${_sqlite3_zip}")
        endif()
    endif()

    if(NOT EXISTS "${_sqlite3_zip}")
        message(STATUS "[profiler-hub] Downloading ${SQLITE3_AMALGAMATION_URL}")
        # TLS_VERIFY must be set here: this script is invoked via cmake -P, so
        # CMAKE_TLS_VERIFY from the parent configure is not inherited. CMake
        # only defaults it ON from 3.31; Windows CI pins an older CMake.
        file(
            DOWNLOAD "${SQLITE3_AMALGAMATION_URL}"
            "${_sqlite3_zip}.tmp"
            STATUS _download_status
            TLS_VERIFY ON
            INACTIVITY_TIMEOUT 10
            TIMEOUT 40
        )
        list(GET _download_status 0 _download_code)
        list(GET _download_status 1 _download_message)

        set(_download_error "")
        if(NOT _download_code EQUAL 0)
            set(_download_error "${_download_message}")
        else()
            file(SHA256 "${_sqlite3_zip}.tmp" _sha256)
            if("${_sha256}" STREQUAL "${SQLITE3_AMALGAMATION_SHA256}")
                file(RENAME "${_sqlite3_zip}.tmp" "${_sqlite3_zip}")
            else()
                set(_download_error
                    "expected SHA256 ${SQLITE3_AMALGAMATION_SHA256}, got ${_sha256}"
                )
            endif()
        endif()

        if(_download_error)
            file(REMOVE "${_sqlite3_zip}.tmp")
            # Hard-fail: unlike rocprofiler-systems' optional test binary, this
            # zip is compiled into profiler-hub-sqlite3.
            message(
                FATAL_ERROR
                "[profiler-hub] Could not stage SQLite3 amalgamation from ${SQLITE3_AMALGAMATION_URL} (${_download_error})"
            )
        endif()
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
