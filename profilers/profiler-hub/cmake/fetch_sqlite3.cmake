# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Run by `cmake -P` from the add_custom_command in sqlite3.cmake. A COMMAND
# list cannot branch on a result, so the clone's full-clone retry lives here.

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
