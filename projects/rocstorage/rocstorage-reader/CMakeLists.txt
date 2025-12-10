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

cmake_minimum_required(VERSION 3.21 FATAL_ERROR)
project(rocstorage-reader VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ------------------------------------------------------------ #
# Build Options
# ------------------------------------------------------------ #

option(ROCSTORAGE_READER_BUILD_TESTS "Build unit tests" ON)

if(ROCSTORAGE_READER_BUILD_TESTS)
    enable_testing()
endif()

# ------------------------------------------------------------ #
# Library Source Files
# ------------------------------------------------------------ #

set(ROCSTORAGE_READER_DATAMODEL_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/source/datamodel/rocprofvis_c_interface.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/datamodel/rocprofvis_dm_base.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/datamodel/rocprofvis_dm_trace.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/datamodel/rocprofvis_dm_track.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/datamodel/rocprofvis_dm_track_slice.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/datamodel/rocprofvis_dm_event_track_slice.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/datamodel/rocprofvis_dm_pmc_track_slice.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/datamodel/rocprofvis_dm_flow_trace.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/datamodel/rocprofvis_dm_stack_trace.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/datamodel/rocprofvis_dm_ext_data.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/datamodel/rocprofvis_dm_table.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/datamodel/rocprofvis_dm_table_row.cpp
)

set(ROCSTORAGE_READER_DATABASE_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/source/database/rocprofvis_db.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/database/rocprofvis_db_sqlite.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/database/rocprofvis_db_rocpd.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/database/rocprofvis_db_rocprof.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/database/rocprofvis_db_profile.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/database/rocprofvis_db_future.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/database/rocprofvis_db_query_factory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/database/rocprofvis_db_query_builder.cpp
)

# ------------------------------------------------------------ #
# Library Target
# ------------------------------------------------------------ #

add_library(rocstorage-reader SHARED
    ${ROCSTORAGE_READER_DATAMODEL_SOURCES}
    ${ROCSTORAGE_READER_DATABASE_SOURCES}
)

target_compile_definitions(rocstorage-reader PRIVATE CPPBUILD)

if(UNIX)
    target_compile_options(rocstorage-reader PRIVATE -fPIC)
endif()

# ------------------------------------------------------------ #
# Include Directories
# ------------------------------------------------------------ #

target_include_directories(rocstorage-reader
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/rocstorage-reader>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/source/datamodel
        ${CMAKE_CURRENT_SOURCE_DIR}/source/database
        ${CMAKE_CURRENT_SOURCE_DIR}/../shared
)

# ------------------------------------------------------------ #
# Dependencies - SQLite3 (from parent rocstorage project)
# ------------------------------------------------------------ #

# If built as standalone, we need to set up dependencies
if(NOT TARGET rocstorage-sqlite3)
    # Check if parent rocstorage has sqlite3 cmake module
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/sqlite3.cmake)
        set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../cmake ${CMAKE_MODULE_PATH})
        include(sqlite3)
    else()
        find_package(SQLite3 REQUIRED)
        add_library(rocstorage-sqlite3 ALIAS SQLite::SQLite3)
    endif()
endif()

target_link_libraries(rocstorage-reader PRIVATE $<BUILD_INTERFACE:rocstorage-sqlite3>)

# ------------------------------------------------------------ #
# Dependencies - spdlog (from parent rocstorage project)
# ------------------------------------------------------------ #

if(NOT TARGET spdlog::spdlog)
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/spdlog.cmake)
        set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../cmake ${CMAKE_MODULE_PATH})
        include(spdlog)
    else()
        find_package(spdlog REQUIRED)
    endif()
endif()

target_link_libraries(rocstorage-reader PRIVATE $<BUILD_INTERFACE:spdlog::spdlog>)

# ------------------------------------------------------------ #
# Copy Public Headers to Build Tree
# ------------------------------------------------------------ #

set(rocstorage_reader_public_headers
    ${CMAKE_CURRENT_SOURCE_DIR}/include/rocprofvis_interface.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/rocprofvis_interface_types.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/rocprofvis_c_interface.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/rocprofvis_c_interface_types.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/rocprofvis_reader_enums.h
)

foreach(HEADER_FILE ${rocstorage_reader_public_headers})
    file(
        RELATIVE_PATH
        HEADER_REL
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${HEADER_FILE}
    )
    configure_file(
        ${HEADER_FILE}
        ${CMAKE_BINARY_DIR}/include/rocstorage-reader/${HEADER_REL}
        COPYONLY
    )
endforeach()

# ------------------------------------------------------------ #
# Install Configuration
# ------------------------------------------------------------ #

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME rocstorage-reader)
set(PACKAGE_NAME "rocstorage-reader")

install(
    TARGETS ${PACKAGE_NAME}
    EXPORT ${PACKAGE_NAME}-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(
    FILES ${rocstorage_reader_public_headers}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rocstorage-reader
)

install(
    EXPORT ${PACKAGE_NAME}-targets
    FILE ${PACKAGE_NAME}-targets.cmake
    NAMESPACE ${PACKAGE_NAME}::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PACKAGE_NAME}
)

# ------------------------------------------------------------ #
# Build Tree Support
# ------------------------------------------------------------ #

export(
    TARGETS ${PACKAGE_NAME}
    NAMESPACE ${PACKAGE_NAME}::
    FILE ${CMAKE_CURRENT_BINARY_DIR}/${PACKAGE_NAME}-targets.cmake
)

# ------------------------------------------------------------ #
# Tests
# ------------------------------------------------------------ #

if(ROCSTORAGE_READER_BUILD_TESTS)
    add_subdirectory(tests)
endif()
