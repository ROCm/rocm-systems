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
    message(
        STATUS
        "[profiler-hub] Cloning SQLite3 from ${SQLITE3_GIT_URL} @ ${SQLITE3_GIT_TAG}"
    )

    find_package(Git REQUIRED)
    find_program(MAKE_COMMAND NAMES make gmake REQUIRED)

    set(SQLITE3_SOURCE_DIR "${PROJECT_BINARY_DIR}/external/sqlite3")
    set(SQLITE3_AMALG_C "${SQLITE3_SOURCE_DIR}/sqlite3.c")
    set(SQLITE3_AMALG_H "${SQLITE3_SOURCE_DIR}/sqlite3.h")

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

    target_compile_options(profiler-hub-sqlite3-static PRIVATE -O2 -fPIC)

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
