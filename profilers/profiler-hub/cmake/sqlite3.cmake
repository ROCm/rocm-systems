# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# ----------------------------------------------------------------------------------------#
#
# SQLite3 - fetched as the upstream amalgamation source archive
#
# Downloads the official SQLite3 amalgamation (sqlite3.c + sqlite3.h) from an
# AMD-hosted mirror and builds it locally. Avoids vendoring any binary blobs
# in the source tree.
#
# ----------------------------------------------------------------------------------------#

include(FetchContent)

set(SQLITE3_URL
    "https://rocm-third-party-deps.s3.us-east-2.amazonaws.com/sqlite-amalgamation-3510300.zip"
    CACHE STRING
    "SQLite3 amalgamation archive URL"
)
set(SQLITE3_URL_HASH
    "SHA256=acb1e6f5d832484bf6d32b681e858c38add8b2acdfd42ac5df24b8afb46552b4"
    CACHE STRING
    "SQLite3 amalgamation archive hash"
)

message(
    STATUS
    "[profiler-hub] Fetching SQLite3 amalgamation from ${SQLITE3_URL}"
)

FetchContent_Declare(
    sqlite3
    URL "${SQLITE3_URL}"
    URL_HASH "${SQLITE3_URL_HASH}"
    DOWNLOAD_EXTRACT_TIMESTAMP true
)
FetchContent_MakeAvailable(sqlite3)

set(SQLITE3_SOURCE_DIR "${sqlite3_SOURCE_DIR}")
set(SQLITE3_AMALG_C "${SQLITE3_SOURCE_DIR}/sqlite3.c")
set(SQLITE3_AMALG_H "${SQLITE3_SOURCE_DIR}/sqlite3.h")

if(NOT EXISTS "${SQLITE3_AMALG_C}" OR NOT EXISTS "${SQLITE3_AMALG_H}")
    message(
        FATAL_ERROR
        "[profiler-hub] SQLite3 amalgamation files not found in fetched archive"
    )
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
