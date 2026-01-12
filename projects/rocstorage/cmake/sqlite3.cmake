# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# ----------------------------------------------------------------------------------------#
#
# SQLite3 - Using amalgamation source for fast builds
#
# ----------------------------------------------------------------------------------------#

option(
    ROCSTORAGE_USE_SYSTEM_SQLITE3
    "Use system-installed SQLite3 if available"
    OFF
)

set(SQLITE3_VERSION
    "3450300"
    CACHE STRING
    "SQLite3 version (amalgamation format)"
)
set(SQLITE3_YEAR "2024" CACHE STRING "SQLite3 release year")

set(ROCSTORAGE_SQLITE3_USE_SYSTEM FALSE)

if(ROCSTORAGE_USE_SYSTEM_SQLITE3)
    find_package(SQLite3 QUIET)
    if(SQLite3_FOUND)
        set(ROCSTORAGE_SQLITE3_USE_SYSTEM TRUE)
    endif()
endif()

if(ROCSTORAGE_SQLITE3_USE_SYSTEM)
    message(
        STATUS
        "[rocstorage] Using system SQLite3 library (version ${SQLite3_VERSION})"
    )

    add_library(rocstorage-sqlite3 INTERFACE)
    target_link_libraries(rocstorage-sqlite3 INTERFACE SQLite::SQLite3)
else()
    message(
        STATUS
        "[rocstorage] Building SQLite3 from amalgamation source (version ${SQLITE3_VERSION})"
    )

    include(FetchContent)

    set(SQLITE3_SOURCE_DIR "${PROJECT_BINARY_DIR}/external/sqlite3")

    # gersemi: off
    FetchContent_Declare(
        sqlite3_amalgamation
        URL "${CMAKE_CURRENT_LIST_DIR}/sqlite3-${SQLITE3_VERSION}.zip"
        SOURCE_DIR ${SQLITE3_SOURCE_DIR}
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    # gersemi: on

    FetchContent_MakeAvailable(sqlite3_amalgamation)

    add_library(
        rocstorage-sqlite3-static
        STATIC
        ${SQLITE3_SOURCE_DIR}/sqlite3.c
    )

    target_include_directories(
        rocstorage-sqlite3-static
        PUBLIC $<BUILD_INTERFACE:${SQLITE3_SOURCE_DIR}>
    )

    target_compile_definitions(
        rocstorage-sqlite3-static
        PRIVATE
            SQLITE_DEFAULT_MEMSTATUS=0
            SQLITE_THREADSAFE=1
            SQLITE_DEFAULT_WAL_SYNCHRONOUS=1
            SQLITE_LIKE_DOESNT_MATCH_BLOBS
            SQLITE_OMIT_DEPRECATED
            SQLITE_OMIT_PROGRESS_CALLBACK
            SQLITE_OMIT_SHARED_CACHE
    )

    target_compile_options(rocstorage-sqlite3-static PRIVATE -O3 -fPIC)

    set_target_properties(
        rocstorage-sqlite3-static
        PROPERTIES POSITION_INDEPENDENT_CODE ON C_STANDARD 11
    )

    add_library(rocstorage-sqlite3 INTERFACE)
    target_link_libraries(
        rocstorage-sqlite3
        INTERFACE rocstorage-sqlite3-static ${CMAKE_DL_LIBS}
    )

    message(
        STATUS
        "[rocstorage] SQLite3 amalgamation source: ${SQLITE3_SOURCE_DIR}"
    )
endif()
