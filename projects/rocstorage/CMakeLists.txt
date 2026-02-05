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

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# ------------------------------------------------------------ #
# Build Options
# ------------------------------------------------------------ #

option(ROCSTORAGE_BUILD_TESTS "Build unit tests" ON)
option(ROCSTORAGE_BUILD_BENCHMARKS "Build performance benchmarks" ON)
option(ROCSTORAGE_ENABLE_LOGGING "Enable logging" OFF)
option(ROCSTORAGE_ENABLE_COVERAGE "Enable code coverage" OFF)

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

target_compile_definitions(
    rocstorage
    PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:ROCSTORAGE_VERSION_MAJOR=${rocstorage_VERSION_MAJOR}>
    PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:ROCSTORAGE_VERSION_MINOR=${rocstorage_VERSION_MINOR}>
    PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:ROCSTORAGE_VERSION_PATCH=${rocstorage_VERSION_PATCH}>
)

target_compile_options(
    rocstorage
    PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:
        -Wall
        -Wextra
        -Wshadow
        -Wvla
        -Wpedantic
        -Wconversion # Warn on implicit type conversions
        -Wsign-conversion # Warn on sign conversions
        -Wnon-virtual-dtor # Warn if class with virtual functions has non-virtual dtor
        -Woverloaded-virtual # Warn if you override a virtual function incorrectly
        -Wnull-dereference # Warn if null pointer is dereferenced
        >
)

# ------------------------------------------------------------ #
# Source
# ------------------------------------------------------------ #

add_subdirectory(source)

# ------------------------------------------------------------ #
# Public API
# ------------------------------------------------------------ #

set(rocstorage_public_headers
    ${CMAKE_CURRENT_LIST_DIR}/include/writer.hpp
    ${CMAKE_CURRENT_LIST_DIR}/include/reader.hpp
    ${CMAKE_CURRENT_LIST_DIR}/include/storage.hpp
    ${CMAKE_CURRENT_LIST_DIR}/include/storage_types.hpp
    ${CMAKE_CURRENT_LIST_DIR}/include/writer_types.hpp
    ${CMAKE_CURRENT_LIST_DIR}/include/reader_types.hpp
    ${CMAKE_CURRENT_LIST_DIR}/include/shared_types.hpp
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
# Code Coverage
# ------------------------------------------------------------ #

include(coverage)

if(ROCSTORAGE_ENABLE_COVERAGE)
    rocstorage_add_coverage_flags(rocstorage)
endif()

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
# nlohmann_json
# ------------------------------------------------------------ #

include(nlohmann_json)
target_link_libraries(
    rocstorage
    PRIVATE $<BUILD_INTERFACE:nlohmann_json::nlohmann_json>
)

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
# Clang-Tidy Custom Target
# ------------------------------------------------------------ #

find_program(
    CLANG_TIDY_EXE
    NAMES clang-tidy clang-tidy-18 clang-tidy-17 clang-tidy-16
)

if(CLANG_TIDY_EXE)
    message(STATUS "clang-tidy found: ${CLANG_TIDY_EXE}")

    # Collect source files for clang-tidy
    file(
        GLOB_RECURSE ROCSTORAGE_CLANG_TIDY_SOURCES
        ${CMAKE_SOURCE_DIR}/source/*.cpp
        ${CMAKE_SOURCE_DIR}/source/*.hpp
        ${CMAKE_SOURCE_DIR}/include/*.hpp
    )

    add_custom_target(
        clang-tidy
        COMMAND
            ${CLANG_TIDY_EXE} -p ${CMAKE_BINARY_DIR}
            --config-file=${CMAKE_SOURCE_DIR}/.clang-tidy
            ${ROCSTORAGE_CLANG_TIDY_SOURCES}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Running clang-tidy static analysis..."
        VERBATIM
    )

    # Target to run clang-tidy with automatic fixes
    add_custom_target(
        clang-tidy-fix
        COMMAND
            ${CLANG_TIDY_EXE} -p ${CMAKE_BINARY_DIR}
            --config-file=${CMAKE_SOURCE_DIR}/.clang-tidy --fix --fix-errors
            ${ROCSTORAGE_CLANG_TIDY_SOURCES}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Running clang-tidy with auto-fix..."
        VERBATIM
    )
else()
    message(
        STATUS
        "clang-tidy not found - clang-tidy target will not be available"
    )
endif()

# ------------------------------------------------------------ #
# Tests and Benchmarks
# ------------------------------------------------------------ #

if(ROCSTORAGE_BUILD_TESTS OR ROCSTORAGE_BUILD_BENCHMARKS)
    add_subdirectory(tests)
endif()
