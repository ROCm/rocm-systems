# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# ----------------------------------------------------------------------------------------#
#
# SQLite3 - built locally from the official upstream amalgamation
#
# The amalgamation (sqlite3.c + sqlite3.h) is fetched as the pre-generated source
# archive published on sqlite.org. That is the same C source `make sqlite3.c`
# emits from a git checkout, but taking it directly keeps configure free of
# autoconf, make and - critically - Tcl. SQLite's `sqlite3.c` makefile target
# runs tool/mksqlite3c.tcl and aborts with
#
#   ERROR: This makefile target requires tclsh 8.4 or later
#
# on hosts without tclsh, which is not a documented build dependency of this
# project (see ROCm/rocm-systems#10059). `./configure --disable-tcl` only drops
# the Tcl *bindings*; it does not lift that requirement.
#
# No binary blobs are vendored in the source tree either way. If sqlite.org is
# unreachable, fall back to the previous git clone + autotools route, which does
# need tclsh and now says so up front instead of failing with an opaque rc=2.
#
# ----------------------------------------------------------------------------------------#

set(SQLITE3_GIT_URL
    "https://github.com/sqlite/sqlite.git"
    CACHE STRING
    "Upstream SQLite3 git repository URL (fallback source)"
)
set(SQLITE3_GIT_TAG
    "version-3.45.3"
    CACHE STRING
    "Upstream SQLite3 version to build, as a git tag"
)
set(SQLITE3_AMALGAMATION_URL_BASE
    "https://www.sqlite.org"
    CACHE STRING
    "Base URL serving the official SQLite3 amalgamation archives"
)
set(SQLITE3_AMALGAMATION_YEAR
    "2024"
    CACHE STRING
    "Release year folder on sqlite.org for the amalgamation download"
)
set(SQLITE3_AMALGAMATION_SHA256
    "ea170e73e447703e8359308ca2e4366a3ae0c4304a8665896f068c736781c651"
    CACHE STRING
    "SHA256 of the SQLite3 amalgamation archive; empty disables verification"
)

set(SQLITE3_SOURCE_DIR "${PROJECT_BINARY_DIR}/external/sqlite3")
set(SQLITE3_AMALG_C "${SQLITE3_SOURCE_DIR}/sqlite3.c")
set(SQLITE3_AMALG_H "${SQLITE3_SOURCE_DIR}/sqlite3.h")

if(NOT EXISTS "${SQLITE3_AMALG_C}" OR NOT EXISTS "${SQLITE3_AMALG_H}")
    # sqlite.org names its archives with the numeric version SQLite reports as
    # SQLITE_VERSION_NUMBER: 3.45.3 -> 3045003 -> "3450300" in the file name.
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

    set(_sqlite3_external_dir "${PROJECT_BINARY_DIR}/external")
    set(_sqlite3_amalg_name "sqlite-amalgamation-${_sqlite3_amalg_version}")
    set(_sqlite3_amalg_zip "${_sqlite3_external_dir}/${_sqlite3_amalg_name}.zip")
    set(_sqlite3_amalg_url
        "${SQLITE3_AMALGAMATION_URL_BASE}/${SQLITE3_AMALGAMATION_YEAR}/${_sqlite3_amalg_name}.zip"
    )

    file(MAKE_DIRECTORY "${_sqlite3_external_dir}")

    message(
        STATUS
        "[profiler-hub] Downloading SQLite3 ${_sqlite3_version} amalgamation from ${_sqlite3_amalg_url}"
    )
    file(
        DOWNLOAD ${_sqlite3_amalg_url}
        ${_sqlite3_amalg_zip}
        STATUS _sqlite3_download_status
    )
    list(GET _sqlite3_download_status 0 _sqlite3_download_rc)
    list(GET _sqlite3_download_status 1 _sqlite3_download_msg)

    set(_sqlite3_have_amalgamation FALSE)

    if(_sqlite3_download_rc EQUAL 0)
        if(SQLITE3_AMALGAMATION_SHA256)
            file(SHA256 "${_sqlite3_amalg_zip}" _sqlite3_actual_sha256)
            if(NOT _sqlite3_actual_sha256 STREQUAL SQLITE3_AMALGAMATION_SHA256)
                message(
                    FATAL_ERROR
                    "[profiler-hub] SQLite3 amalgamation checksum mismatch for "
                    "${_sqlite3_amalg_url}\n"
                    "  expected: ${SQLITE3_AMALGAMATION_SHA256}\n"
                    "  actual:   ${_sqlite3_actual_sha256}\n"
                    "If SQLITE3_GIT_TAG was changed, update SQLITE3_AMALGAMATION_SHA256 "
                    "and SQLITE3_AMALGAMATION_YEAR to match."
                )
            endif()
        endif()

        file(
            ARCHIVE_EXTRACT
            INPUT ${_sqlite3_amalg_zip}
            DESTINATION ${_sqlite3_external_dir}
        )
        file(MAKE_DIRECTORY "${SQLITE3_SOURCE_DIR}")
        foreach(_sqlite3_file IN ITEMS sqlite3.c sqlite3.h sqlite3ext.h)
            if(EXISTS "${_sqlite3_external_dir}/${_sqlite3_amalg_name}/${_sqlite3_file}")
                file(
                    COPY "${_sqlite3_external_dir}/${_sqlite3_amalg_name}/${_sqlite3_file}"
                    DESTINATION ${SQLITE3_SOURCE_DIR}
                )
            endif()
        endforeach()

        if(EXISTS "${SQLITE3_AMALG_C}" AND EXISTS "${SQLITE3_AMALG_H}")
            set(_sqlite3_have_amalgamation TRUE)
        else()
            message(
                STATUS
                "[profiler-hub] SQLite3 amalgamation archive did not contain the expected "
                "sources; falling back to a git checkout"
            )
        endif()
    else()
        message(
            STATUS
            "[profiler-hub] SQLite3 amalgamation download failed "
            "(rc=${_sqlite3_download_rc}: ${_sqlite3_download_msg}); "
            "falling back to a git checkout"
        )
    endif()

    if(NOT _sqlite3_have_amalgamation)
        # Generating the amalgamation from a git checkout runs tool/mksqlite3c.tcl,
        # so Tcl is mandatory here. Say so before spending time on a clone.
        find_program(
            TCLSH_EXECUTABLE
            NAMES tclsh tclsh8.6 tclsh8.5 tclsh8.4
        )
        if(NOT TCLSH_EXECUTABLE)
            message(
                FATAL_ERROR
                "[profiler-hub] Cannot obtain SQLite3: downloading the prebuilt "
                "amalgamation from ${_sqlite3_amalg_url} failed, and generating it from "
                "${SQLITE3_GIT_URL} requires tclsh 8.4 or later, which was not found.\n"
                "Either make sqlite.org reachable, or install Tcl "
                "(Debian/Ubuntu: 'tcl', RHEL/Fedora: 'tcl')."
            )
        endif()

        message(
            STATUS
            "[profiler-hub] Cloning SQLite3 from ${SQLITE3_GIT_URL} @ ${SQLITE3_GIT_TAG}"
        )

        find_package(Git REQUIRED)
        find_program(MAKE_COMMAND NAMES make gmake REQUIRED)

        # checkout: shallow + partial first, retry full on failure
        if(NOT EXISTS "${SQLITE3_SOURCE_DIR}/configure")
            if(EXISTS "${SQLITE3_SOURCE_DIR}")
                file(REMOVE_RECURSE "${SQLITE3_SOURCE_DIR}")
            endif()
            execute_process(
                COMMAND
                    ${GIT_EXECUTABLE} clone --depth 1 --filter=blob:none --branch
                    ${SQLITE3_GIT_TAG} ${SQLITE3_GIT_URL} ${SQLITE3_SOURCE_DIR}
                RESULT_VARIABLE _sqlite3_clone_rc
            )
            if(NOT _sqlite3_clone_rc EQUAL 0)
                message(
                    STATUS
                    "[profiler-hub] Optimized clone failed; retrying full clone"
                )
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
        if(NOT EXISTS "${SQLITE3_AMALG_C}" OR NOT EXISTS "${SQLITE3_AMALG_H}")
            message(
                FATAL_ERROR
                "[profiler-hub] SQLite3 amalgamation files not found after build"
            )
        endif()
    endif()
endif()

add_library(profiler-hub-sqlite3-static STATIC ${SQLITE3_AMALG_C})

target_include_directories(
    profiler-hub-sqlite3-static
    PUBLIC $<BUILD_INTERFACE:${SQLITE3_SOURCE_DIR}>
)

target_compile_definitions(
    profiler-hub-sqlite3-static
    PRIVATE
        SQLITE_DEFAULT_MEMSTATUS=0
        SQLITE_THREADSAFE=1
        SQLITE_DEFAULT_WAL_SYNCHRONOUS=1
        SQLITE_LIKE_DOESNT_MATCH_BLOBS=1
        SQLITE_OMIT_DEPRECATED=1
        SQLITE_OMIT_PROGRESS_CALLBACK=1
        SQLITE_OMIT_SHARED_CACHE=1
)

# Seal the bundled SQLite symbols so they are not exported from
# libprofiler-hub.{so,a} and cannot collide with (or be interposed by)
# other sqlite3 versions bundled by sibling components on TheRock.
target_compile_options(
    profiler-hub-sqlite3-static
    PRIVATE -O2 -fPIC -fvisibility=hidden
)

set_target_properties(
    profiler-hub-sqlite3-static
    PROPERTIES POSITION_INDEPENDENT_CODE ON C_STANDARD 11
)

add_library(profiler-hub-sqlite3 INTERFACE)
target_link_libraries(
    profiler-hub-sqlite3
    INTERFACE profiler-hub-sqlite3-static ${CMAKE_DL_LIBS}
)

message(
    STATUS
    "[profiler-hub] SQLite3 amalgamation source: ${SQLITE3_SOURCE_DIR}"
)
