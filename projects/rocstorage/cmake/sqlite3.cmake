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
# SQLite3
#
# ----------------------------------------------------------------------------------------#

# message(STATUS "[rocstorage] Looking for SQLite3 library...")
# find_package(SQLite3 QUIET)

# if(SQLite3_FOUND)
#     message(STATUS "[rocstorage] Using system SQLite3 library")

#     add_library(rocstorage-sqlite3 INTERFACE)
#     target_link_libraries(rocstorage-sqlite3 INTERFACE SQLite::SQLite3)
# else()
message(STATUS "[rocstorage] Building SQLite3 from source!")

include(FetchContent)
include(ExternalProject)

set(SQLITE3_VERSION "3.45.3")
set(SQLITE3_INSTALL_DIR "${PROJECT_BINARY_DIR}/lib/sqlite3")
set(SQLITE3_LIB_FILE "${SQLITE3_INSTALL_DIR}/lib/libsqlite3.a")
set(SQLITE3_INCLUDE_DIR "${SQLITE3_INSTALL_DIR}/include")

FetchContent_Declare(
    sqlite3_source
    GIT_REPOSITORY https://github.com/sqlite/sqlite.git
    GIT_TAG "version-${SQLITE3_VERSION}"
    GIT_SHALLOW TRUE
    SOURCE_DIR
    "${PROJECT_BINARY_DIR}/external/sqlite/source"
)

FetchContent_GetProperties(sqlite3_source)
if(NOT sqlite3_source_POPULATED)
    message(
        STATUS
        "[rocstorage] Fetching SQLite3 version ${SQLITE3_VERSION}..."
    )
    FetchContent_Populate(sqlite3_source)
endif()

find_program(MAKE_COMMAND NAMES make gmake PATH_SUFFIXES bin REQUIRED)

ExternalProject_Add(
    rocstorage-sqlite3-build
    SOURCE_DIR ${sqlite3_source_SOURCE_DIR}
    PREFIX ${PROJECT_BINARY_DIR}/external/sqlite/build
    BUILD_IN_SOURCE 0
    CONFIGURE_COMMAND
        ${sqlite3_source_SOURCE_DIR}/configure --prefix=${SQLITE3_INSTALL_DIR}
        --libdir=${SQLITE3_INSTALL_DIR}/lib --disable-shared --enable-static
        --with-tempstore=yes --enable-all --disable-tcl
        CFLAGS=-O3\ -g1\ -fPIC\ -DSQLITE_DEFAULT_MEMSTATUS=0
    BUILD_COMMAND ${MAKE_COMMAND} -j${CMAKE_BUILD_PARALLEL_LEVEL}
    INSTALL_COMMAND ${MAKE_COMMAND} install
    BUILD_BYPRODUCTS ${SQLITE3_LIB_FILE}
    LOG_CONFIGURE ON
    LOG_BUILD ON
    LOG_INSTALL ON
)

add_library(rocstorage-sqlite3-static STATIC IMPORTED GLOBAL)
set_target_properties(
    rocstorage-sqlite3-static
    PROPERTIES
        IMPORTED_LOCATION ${SQLITE3_LIB_FILE}
        INTERFACE_INCLUDE_DIRECTORIES ${SQLITE3_INCLUDE_DIR}
)

add_dependencies(rocstorage-sqlite3-static rocstorage-sqlite3-build)

add_library(rocstorage-sqlite3 INTERFACE)
target_link_libraries(
    rocstorage-sqlite3
    INTERFACE rocstorage-sqlite3-static ${CMAKE_DL_LIBS}
)

target_include_directories(
    rocstorage-sqlite3
    SYSTEM
    INTERFACE $<BUILD_INTERFACE:${SQLITE3_INCLUDE_DIR}>
)

message(
    STATUS
    "[rocstorage] SQLite3 will be installed to: ${SQLITE3_INSTALL_DIR}"
)
# endif()
