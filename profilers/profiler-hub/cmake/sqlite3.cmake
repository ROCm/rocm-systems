# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

include_guard(DIRECTORY)

# The rocpd schema profiler-hub executes uses JSON_EXTRACT, which only became an
# unconditional part of the SQLite core in 3.38.0 (before that it was the opt-in
# SQLITE_ENABLE_JSON1 extension). Everything else profiler-hub calls is far older.
set(SQLITE3_VERSION "3.38.0" CACHE STRING "Minimum SQLite3 version")

# Fetching is Off by default: a missing or old package errors out.
if(NOT PROFILER_HUB_FETCH_DEPENDENCIES)
    find_package(SQLite3 ${SQLITE3_VERSION})

    if(NOT SQLite3_FOUND)
        message(
            FATAL_ERROR
            "profiler-hub requires SQLite3 ${SQLITE3_VERSION} or newer on CMAKE_PREFIX_PATH. Configure with -DPROFILER_HUB_FETCH_DEPENDENCIES=ON to download it instead."
        )
    endif()

    message(STATUS "Using system SQLite3 (version ${SQLite3_VERSION})")

    # SQLite::SQLite3 is the target name both CMake's own FindSQLite3 module and
    # TheRock's bundled sqlite3-config.cmake define, so one line covers both.
    add_library(profiler-hub-sqlite3 INTERFACE)
    target_link_libraries(
        profiler-hub-sqlite3
        INTERFACE SQLite::SQLite3 ${CMAKE_DL_LIBS}
    )
else()
    # SQLite3 is not vendored into this tree. rocprofiler-sdk takes the same
    # position for its own SQLite3, sourcing it from upstream as a git submodule
    # (projects/rocprofiler-sdk/external/).
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

    find_package(Git REQUIRED)
    find_program(MAKE_COMMAND NAMES make gmake REQUIRED)

    set(SQLITE3_SOURCE_DIR "${PROJECT_BINARY_DIR}/external/sqlite3")
    set(SQLITE3_AMALG_C "${SQLITE3_SOURCE_DIR}/sqlite3.c")
    set(SQLITE3_AMALG_H "${SQLITE3_SOURCE_DIR}/sqlite3.h")

    # The retry lives in the -P script because a COMMAND list cannot branch on
    # a result. No USES_TERMINAL: it would put this in ninja's console pool and
    # serialise the very edge this moved out of configure.
    add_custom_command(
        OUTPUT ${SQLITE3_AMALG_C} ${SQLITE3_AMALG_H}
        COMMAND
            ${CMAKE_COMMAND} -DGIT_EXECUTABLE=${GIT_EXECUTABLE}
            -DMAKE_COMMAND=${MAKE_COMMAND} -DSQLITE3_GIT_URL=${SQLITE3_GIT_URL}
            -DSQLITE3_GIT_TAG=${SQLITE3_GIT_TAG}
            -DSQLITE3_SOURCE_DIR=${SQLITE3_SOURCE_DIR} -P
            ${CMAKE_CURRENT_LIST_DIR}/fetch_sqlite3.cmake
        DEPENDS ${CMAKE_CURRENT_LIST_DIR}/fetch_sqlite3.cmake
        COMMENT "[profiler-hub] Fetching SQLite3 ${SQLITE3_GIT_TAG}"
        VERBATIM
    )

    add_library(
        profiler-hub-sqlite3-static
        STATIC
        ${SQLITE3_AMALG_C}
        ${SQLITE3_AMALG_H}
    )

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

    # Seal the bundled SQLite symbols so no shared library linking this archive
    # exports them, and they cannot collide with (or be interposed by) other
    # sqlite3 versions bundled by sibling components on TheRock.
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
endif()
