# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# ----------------------------------------------------------------------------------------#
#
# SQLite3 - cloned from upstream and built from the amalgamation
#
# Mirrors the pattern used by sibling rocprofiler-systems
# (projects/rocprofiler-systems/cmake/SQLite3.cmake): fetch the upstream
# git repository at a pinned tag, then build locally. Avoids vendoring
# any binary blobs in the source tree.
#
# ----------------------------------------------------------------------------------------#

set(SQLITE3_GIT_URL
    "https://github.com/sqlite/sqlite.git"
    CACHE STRING
    "Upstream SQLite3 git repository URL"
)
set(SQLITE3_GIT_TAG
    "version-3.45.3"
    CACHE STRING
    "Upstream SQLite3 git tag to check out"
)
set(SQLITE3_AMALGAMATION_YEAR
    "2024"
    CACHE STRING
    "Release year folder on sqlite.org for the amalgamation download (Windows)"
)

set(SQLITE3_SOURCE_DIR "${PROJECT_BINARY_DIR}/external/sqlite3")
set(SQLITE3_AMALG_C "${SQLITE3_SOURCE_DIR}/sqlite3.c")
set(SQLITE3_AMALG_H "${SQLITE3_SOURCE_DIR}/sqlite3.h")

if(WIN32)
    # Official amalgamation zip. No git, nmake, or Tcl.
    if(NOT EXISTS "${SQLITE3_AMALG_C}" OR NOT EXISTS "${SQLITE3_AMALG_H}")
        string(
            REGEX REPLACE
            "^version-"
            ""
            _sqlite3_version
            "${SQLITE3_GIT_TAG}"
        )
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
    endif()
else()
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

    # generate amalgamation (sqlite3.c + sqlite3.h) via upstream autotools
    if(NOT EXISTS "${SQLITE3_AMALG_C}" OR NOT EXISTS "${SQLITE3_AMALG_H}")
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

if(NOT MSVC)
    # Seal the bundled SQLite symbols so they are not exported from
    # libprofiler-hub.{so,a} and cannot collide with (or be interposed by)
    # other sqlite3 versions bundled by sibling components on TheRock.
    target_compile_options(
        profiler-hub-sqlite3-static
        PRIVATE -O2 -fPIC -fvisibility=hidden
    )
endif()

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
