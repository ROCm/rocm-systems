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
project(rocpdsna VERSION 0.1.0)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# ------------------------------------------------------------ #
# Build Options
# ------------------------------------------------------------ #

option(ROCPDSNA_BUILD_TESTS "Build unit tests" ON)
option(ROCPDSNA_BUILD_BENCHMARKS "Build performance benchmarks" ON)
option(ROCPDSNA_ENABLE_LOGGING "Enable logging" OFF)
option(ROCPDSNA_ENABLE_COVERAGE "Enable code coverage" OFF)

if(ROCPDSNA_BUILD_TESTS)
    enable_testing()
endif()

# ------------------------------------------------------------ #
# Version
# ------------------------------------------------------------ #

set(rocpdsna_VERSION_MAJOR 0)
set(rocpdsna_VERSION_MINOR 1)
set(rocpdsna_VERSION_PATCH 0)
set(rocpdsna_VERSION
    ${rocpdsna_VERSION_MAJOR}.${rocpdsna_VERSION_MINOR}.${rocpdsna_VERSION_PATCH}
)

# ------------------------------------------------------------ #
# Object Library (compiled once, used by both shared and static)
# ------------------------------------------------------------ #

add_library(rocpdsna-objects OBJECT)
set_target_properties(rocpdsna-objects PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_compile_definitions(
    rocpdsna-objects
    PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:ROCPDSNA_VERSION_MAJOR=${rocpdsna_VERSION_MAJOR}>
    PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:ROCPDSNA_VERSION_MINOR=${rocpdsna_VERSION_MINOR}>
    PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:ROCPDSNA_VERSION_PATCH=${rocpdsna_VERSION_PATCH}>
)

target_compile_options(
    rocpdsna-objects
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

set(rocpdsna_public_headers
    ${CMAKE_CURRENT_LIST_DIR}/include/writer.hpp
    ${CMAKE_CURRENT_LIST_DIR}/include/reader.hpp
    ${CMAKE_CURRENT_LIST_DIR}/include/storage.hpp
    ${CMAKE_CURRENT_LIST_DIR}/include/storage_types.hpp
    ${CMAKE_CURRENT_LIST_DIR}/include/writer_types.hpp
    ${CMAKE_CURRENT_LIST_DIR}/include/reader_types.hpp
    ${CMAKE_CURRENT_LIST_DIR}/include/shared_types.hpp
)

foreach(HEADER_FILE ${rocpdsna_public_headers})
    file(
        RELATIVE_PATH
        HEADER_REL
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${HEADER_FILE}
    )
    configure_file(
        ${HEADER_FILE}
        ${CMAKE_BINARY_DIR}/include/rocpdsna/${HEADER_REL}
        COPYONLY
    )
endforeach()

target_include_directories(
    rocpdsna-objects
    PUBLIC $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
)

# ------------------------------------------------------------ #
# CMake modules
# ------------------------------------------------------------ #

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake ${CMAKE_MODULE_PATH})

# ------------------------------------------------------------ #
# Code Coverage
# ------------------------------------------------------------ #

include(coverage)

if(ROCPDSNA_ENABLE_COVERAGE)
    rocpdsna_add_coverage_flags(rocpdsna-objects)
endif()

# ------------------------------------------------------------ #
# SQLite3
# ------------------------------------------------------------ #

include(sqlite3)
target_link_libraries(rocpdsna-objects PRIVATE rocpdsna-sqlite3)

# ------------------------------------------------------------ #
# spdlog
# ------------------------------------------------------------ #

include(spdlog)
target_link_libraries(rocpdsna-objects PRIVATE spdlog::spdlog)

# ------------------------------------------------------------ #
# nlohmann_json
# ------------------------------------------------------------ #

include(nlohmann_json)
target_link_libraries(rocpdsna-objects PRIVATE nlohmann_json::nlohmann_json)

# ------------------------------------------------------------ #
# Schema from rocprofiler-sdk-rocpd
# ------------------------------------------------------------ #

set(SQL_SCHEMA_DIR ${CMAKE_CURRENT_LIST_DIR}/source/data_storage/schema)
set(SQL_SCHEMA_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/source/data_storage/)

include(rocm)
include(rocprofiler-sdk-rocpd)

if(USE_SCHEMA_FROM_ROCPROFILER_SDK_ROCPD)
    target_link_libraries(
        rocpdsna-objects
        PRIVATE rocprofiler-sdk-rocpd::rocprofiler-sdk-rocpd
    )

    target_compile_definitions(
        rocpdsna-objects
        PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:USE_SCHEMA_FROM_ROCPROFILER_SDK_ROCPD=1>
    )
else()
    target_include_directories(
        rocpdsna-objects
        PRIVATE ${SQL_SCHEMA_BINARY_DIR}
    )

    target_compile_definitions(
        rocpdsna-objects
        PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:USE_SCHEMA_FROM_ROCPROFILER_SDK_ROCPD=0>
    )
endif()

# ------------------------------------------------------------ #
# Logging
# ------------------------------------------------------------ #

if(ROCPDSNA_ENABLE_LOGGING)
    target_compile_definitions(
        rocpdsna-objects
        PRIVATE $<$<COMPILE_LANGUAGE:CXX>:ROCPDSNA_ENABLE_LOGGING=1>
    )
else()
    target_compile_definitions(
        rocpdsna-objects
        PRIVATE $<$<COMPILE_LANGUAGE:CXX>:ROCPDSNA_ENABLE_LOGGING=0>
    )
endif()

# ------------------------------------------------------------ #
# Shared and Static Libraries
# ------------------------------------------------------------ #

add_library(rocpdsna SHARED $<TARGET_OBJECTS:rocpdsna-objects>)
add_library(rocpdsna-static STATIC $<TARGET_OBJECTS:rocpdsna-objects>)
set_target_properties(rocpdsna-static PROPERTIES OUTPUT_NAME rocpdsna)

foreach(_target rocpdsna rocpdsna-static)
    target_include_directories(
        ${_target}
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    )

    target_link_libraries(
        ${_target}
        PRIVATE
            $<BUILD_INTERFACE:rocpdsna_queries>
            $<BUILD_INTERFACE:rocpdsna-sqlite3>
            $<BUILD_INTERFACE:spdlog::spdlog>
            $<BUILD_INTERFACE:nlohmann_json::nlohmann_json>
    )

    if(USE_SCHEMA_FROM_ROCPROFILER_SDK_ROCPD)
        target_link_libraries(
            ${_target}
            PRIVATE
                $<BUILD_INTERFACE:rocprofiler-sdk-rocpd::rocprofiler-sdk-rocpd>
        )
    endif()

    if(ROCPDSNA_ENABLE_COVERAGE)
        target_link_options(${_target} PRIVATE ${ROCPDSNA_COVERAGE_LINK_FLAGS})
    endif()
endforeach()

# ------------------------------------------------------------ #
# Install package support
# ------------------------------------------------------------ #

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME rocpdsna)
set(PACKAGE_NAME "rocpdsna")

install(
    TARGETS ${PACKAGE_NAME}
    EXPORT ${PACKAGE_NAME}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} COMPONENT rocpdsna
)

install(
    TARGETS ${PACKAGE_NAME}-static
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT rocpdsna
)

install(
    EXPORT ${PACKAGE_NAME}
    FILE ${PACKAGE_NAME}-targets.cmake
    NAMESPACE ${PACKAGE_NAME}::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PACKAGE_NAME}
    COMPONENT rocpdsna
)

set(INCLUDE_INSTALL_DIR ${CMAKE_INSTALL_INCLUDEDIR})
set(LIB_INSTALL_DIR ${CMAKE_INSTALL_LIBDIR})

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/rocpdsna-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/${PACKAGE_NAME}/${PACKAGE_NAME}-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PACKAGE_NAME}
    PATH_VARS INCLUDE_INSTALL_DIR LIB_INSTALL_DIR
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/${PACKAGE_NAME}/${PACKAGE_NAME}-config-version.cmake
    VERSION ${rocpdsna_VERSION}
    COMPATIBILITY SameMinorVersion
)

install(
    FILES
        ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/${PACKAGE_NAME}/${PACKAGE_NAME}-config.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/${PACKAGE_NAME}/${PACKAGE_NAME}-config-version.cmake
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Find${PACKAGE_NAME}.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PACKAGE_NAME}
    COMPONENT rocpdsna
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
        GLOB_RECURSE ROCPDSNA_CLANG_TIDY_SOURCES
        ${CMAKE_SOURCE_DIR}/source/*.cpp
        ${CMAKE_SOURCE_DIR}/source/*.hpp
        ${CMAKE_SOURCE_DIR}/include/*.hpp
    )

    add_custom_target(
        clang-tidy
        COMMAND
            ${CLANG_TIDY_EXE} -p ${CMAKE_BINARY_DIR}
            --config-file=${CMAKE_SOURCE_DIR}/.clang-tidy
            ${ROCPDSNA_CLANG_TIDY_SOURCES}
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
            ${ROCPDSNA_CLANG_TIDY_SOURCES}
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

if(ROCPDSNA_BUILD_TESTS OR ROCPDSNA_BUILD_BENCHMARKS)
    add_subdirectory(tests)
endif()
