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
project(rocstorage VERSION 0.1.0)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ------------------------------------------------------------ #
# Build Options
# ------------------------------------------------------------ #

option(ROCSTORAGE_BUILD_TESTS "Build unit tests" ON)
option(ROCSTORAGE_BUILD_BENCHMARKS "Build performance benchmarks" ON)
option(ROCSTORAGE_ENABLE_LOGGING "Enable logging" OFF)

if(ROCSTORAGE_BUILD_TESTS)
    enable_testing()
endif()

# ------------------------------------------------------------ #
# Version
# ------------------------------------------------------------ #

set(rocstorage_VERSION_MAJOR 0)
set(rocstorage_VERSION_MINOR 1)
set(rocstorage_VERSION_PATCH 0)
set(rocstorage_VERSION
    ${rocstorage_VERSION_MAJOR}.${rocstorage_VERSION_MINOR}.${rocstorage_VERSION_PATCH}
)

# ------------------------------------------------------------ #
# Library
# ------------------------------------------------------------ #

add_library(rocstorage SHARED)

add_subdirectory(source)

# ------------------------------------------------------------ #
# Public API
# ------------------------------------------------------------ #

set(rocstorage_public_headers
    ${CMAKE_CURRENT_LIST_DIR}/include/writer.hpp
    ${CMAKE_CURRENT_LIST_DIR}/include/reader.hpp
    ${CMAKE_CURRENT_LIST_DIR}/include/storage.hpp
)

foreach(HEADER_FILE ${rocstorage_public_headers})
    file(
        RELATIVE_PATH
        HEADER_REL
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${HEADER_FILE}
    )
    configure_file(
        ${HEADER_FILE}
        ${CMAKE_BINARY_DIR}/include/rocstorage/${HEADER_REL}
        COPYONLY
    )
endforeach()

target_include_directories(
    rocstorage
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# ------------------------------------------------------------ #
# CMake modules
# ------------------------------------------------------------ #

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake ${CMAKE_MODULE_PATH})

# ------------------------------------------------------------ #
# SQLite3
# ------------------------------------------------------------ #

include(sqlite3)
target_link_libraries(rocstorage PRIVATE $<BUILD_INTERFACE:rocstorage-sqlite3>)

# ------------------------------------------------------------ #
# spdlog
# ------------------------------------------------------------ #

include(spdlog)
target_link_libraries(rocstorage PRIVATE $<BUILD_INTERFACE:spdlog::spdlog>)

# ------------------------------------------------------------ #
# Schema from rocprofiler-sdk-rocpd
# ------------------------------------------------------------ #

set(SQL_SCHEMA_DIR ${CMAKE_CURRENT_LIST_DIR}/source/data_storage/schema)
set(SQL_SCHEMA_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/source/data_storage/)

include(rocm)
include(rocprofiler-sdk-rocpd)

if(USE_SCHEMA_FROM_ROCPROFILER_SDK_ROCPD)
    target_link_libraries(
        rocstorage
        PRIVATE rocprofiler-sdk-rocpd::rocprofiler-sdk-rocpd
    )

    target_compile_definitions(
        rocstorage
        PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:USE_SCHEMA_FROM_ROCPROFILER_SDK_ROCPD=1>
    )
else()
    target_include_directories(rocstorage PRIVATE ${SQL_SCHEMA_BINARY_DIR})

    target_compile_definitions(
        rocstorage
        PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:USE_SCHEMA_FROM_ROCPROFILER_SDK_ROCPD=0>
    )
endif()

# ------------------------------------------------------------ #
# Logging
# ------------------------------------------------------------ #

if(ROCSTORAGE_ENABLE_LOGGING)
    target_compile_definitions(
        rocstorage
        PRIVATE $<$<COMPILE_LANGUAGE:CXX>:ROCSTORAGE_ENABLE_LOGGING=1>
    )
else()
    target_compile_definitions(
        rocstorage
        PRIVATE $<$<COMPILE_LANGUAGE:CXX>:ROCSTORAGE_ENABLE_LOGGING=0>
    )
endif()

# ------------------------------------------------------------ #
# Install package support
# ------------------------------------------------------------ #

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME rocstorage)
set(PACKAGE_NAME "rocstorage")

install(
    TARGETS ${PACKAGE_NAME}
    EXPORT ${PACKAGE_NAME}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} COMPONENT rocstorage
)

install(
    EXPORT ${PACKAGE_NAME}
    FILE ${PACKAGE_NAME}-targets.cmake
    NAMESPACE ${PACKAGE_NAME}::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PACKAGE_NAME}
    COMPONENT rocstorage
)

set(INCLUDE_INSTALL_DIR ${CMAKE_INSTALL_INCLUDEDIR})
set(LIB_INSTALL_DIR ${CMAKE_INSTALL_LIBDIR})

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/rocstorage-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/${PACKAGE_NAME}/${PACKAGE_NAME}-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PACKAGE_NAME}
    PATH_VARS INCLUDE_INSTALL_DIR LIB_INSTALL_DIR
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/${PACKAGE_NAME}/${PACKAGE_NAME}-config-version.cmake
    VERSION ${rocstorage_VERSION}
    COMPATIBILITY SameMinorVersion
)

install(
    FILES
        ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/${PACKAGE_NAME}/${PACKAGE_NAME}-config.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/${PACKAGE_NAME}/${PACKAGE_NAME}-config-version.cmake
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Find${PACKAGE_NAME}.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PACKAGE_NAME}
    COMPONENT rocstorage
)

# ------------------------------------------------------------ #
# Build tree support
# ------------------------------------------------------------ #

export(
    TARGETS ${PACKAGE_NAME}
    NAMESPACE ${PACKAGE_NAME}::
    FILE ${CMAKE_CURRENT_BINARY_DIR}/${PACKAGE_NAME}-targets.cmake
)

file(
    WRITE ${CMAKE_CURRENT_BINARY_DIR}/${PACKAGE_NAME}-config.cmake
    "include_guard(DIRECTORY)
set(${PACKAGE_NAME}_INCLUDE_DIR \"${CMAKE_BINARY_DIR}/include\")
set(${PACKAGE_NAME}_LIB_DIR \"${CMAKE_CURRENT_BINARY_DIR}\")
include(\"\${CMAKE_CURRENT_LIST_DIR}/${PACKAGE_NAME}-targets.cmake\")
"
)

file(
    COPY
        ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/${PACKAGE_NAME}/${PACKAGE_NAME}-config-version.cmake
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Find${PACKAGE_NAME}.cmake
    DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
)

set(CMAKE_EXPORT_PACKAGE_REGISTRY ON)
export(PACKAGE ${PACKAGE_NAME})

# ------------------------------------------------------------ #
# Tests and Benchmarks
# ------------------------------------------------------------ #

if(ROCSTORAGE_BUILD_TESTS OR ROCSTORAGE_BUILD_BENCHMARKS)
    add_subdirectory(tests)
endif()
