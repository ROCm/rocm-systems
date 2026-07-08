# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# ----------------------------------------------------------------------------------------#
#
# SQLite3 - built from the official amalgamation (sqlite3.c + sqlite3.h)
#
# Mirrors the pattern used by sibling rocprofiler-systems
# (projects/rocprofiler-systems/cmake/SQLite3.cmake): fetch upstream at a
# pinned version and build locally. Avoids vendoring any binary blobs in
# the source tree.
#
# Linux: git clone @ SQLITE3_GIT_TAG, then ./configure + make sqlite3.c
# Windows: download official amalgamation zip from sqlite.org (same version)
#
# ----------------------------------------------------------------------------------------#

option(
    PROFILER_HUB_USE_SYSTEM_SQLITE3
    "Use system-installed SQLite3 if available"
    OFF
)

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

set(PROFILER_HUB_SQLITE3_USE_SYSTEM FALSE)

if(PROFILER_HUB_USE_SYSTEM_SQLITE3)
    find_package(SQLite3 QUIET)
    if(SQLite3_FOUND)
        set(PROFILER_HUB_SQLITE3_USE_SYSTEM TRUE)
    endif()
endif()

if(PROFILER_HUB_SQLITE3_USE_SYSTEM)
    message(
        STATUS
        "[profiler-hub] Using system SQLite3 library (version ${SQLite3_VERSION})"
    )

    add_library(profiler-hub-sqlite3 INTERFACE)
    target_link_libraries(profiler-hub-sqlite3 INTERFACE SQLite::SQLite3)
else()
    set(SQLITE3_SOURCE_DIR "${PROJECT_BINARY_DIR}/external/sqlite3")
    set(SQLITE3_AMALG_C "${SQLITE3_SOURCE_DIR}/sqlite3.c")
    set(SQLITE3_AMALG_H "${SQLITE3_SOURCE_DIR}/sqlite3.h")

    if(NOT EXISTS "${SQLITE3_AMALG_C}" OR NOT EXISTS "${SQLITE3_AMALG_H}")
        if(WIN32)
            # Windows: official pre-built amalgamation zip (no git, nmake, or Tcl).
            string(
                REGEX REPLACE
                "^version-"
                ""
                _sqlite3_version
                ${SQLITE3_GIT_TAG}
            )
            string(REPLACE "." ";" _sqlite3_version_parts ${_sqlite3_version})
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

            set(_sqlite3_zip
                "${PROJECT_BINARY_DIR}/external/sqlite-amalgamation-${_sqlite3_amalg_version}.zip"
            )
            set(_sqlite3_download_url
                "https://www.sqlite.org/${SQLITE3_AMALGAMATION_YEAR}/sqlite-amalgamation-${_sqlite3_amalg_version}.zip"
            )

            message(
                STATUS
                "[profiler-hub] Downloading SQLite3 amalgamation ${_sqlite3_version} from ${_sqlite3_download_url}"
            )
            file(
                DOWNLOAD ${_sqlite3_download_url}
                ${_sqlite3_zip}
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
                INPUT ${_sqlite3_zip}
                DESTINATION ${PROJECT_BINARY_DIR}/external
            )
            file(
                RENAME
                    "${PROJECT_BINARY_DIR}/external/sqlite-amalgamation-${_sqlite3_amalg_version}"
                ${SQLITE3_SOURCE_DIR}
            )
        else()
            # Linux: upstream git checkout + autotools amalgamation (unchanged).
            message(
                STATUS
                "[profiler-hub] Cloning SQLite3 from ${SQLITE3_GIT_URL} @ ${SQLITE3_GIT_TAG}"
            )

            find_package(Git REQUIRED)

            if(NOT EXISTS "${SQLITE3_SOURCE_DIR}/configure")
                if(EXISTS "${SQLITE3_SOURCE_DIR}")
                    file(REMOVE_RECURSE "${SQLITE3_SOURCE_DIR}")
                endif()
                execute_process(
                    COMMAND
                        ${GIT_EXECUTABLE} clone --depth 1 --filter=blob:none
                        --branch ${SQLITE3_GIT_TAG} ${SQLITE3_GIT_URL}
                        ${SQLITE3_SOURCE_DIR}
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
            find_program(MAKE_COMMAND NAMES make gmake REQUIRED)
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
        endif()

        if(NOT EXISTS "${SQLITE3_AMALG_C}" OR NOT EXISTS "${SQLITE3_AMALG_H}")
            message(
                FATAL_ERROR
                "[profiler-hub] SQLite3 amalgamation files not found after fetch"
            )
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
        target_compile_options(profiler-hub-sqlite3-static PRIVATE -O2 -fPIC)
    endif()

    if(WIN32)
        # WINDOWS_EXPORT_ALL_SYMBOLS won't re-export symbols from a static lib,
        # so mark the SQLite API dllexport to share one copy across the process.
        target_compile_definitions(
            profiler-hub-sqlite3-static
            PRIVATE "SQLITE_API=__declspec(dllexport)"
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
endif()
