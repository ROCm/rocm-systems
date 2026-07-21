# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.21 FATAL_ERROR)
project(profiler-hub VERSION 0.1.0)
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

option(PROFILER_HUB_BUILD_TESTS "Build unit tests" ON)
option(PROFILER_HUB_BUILD_BENCHMARKS "Build performance benchmarks" ON)
option(PROFILER_HUB_ENABLE_LOGGING "Enable logging" OFF)
option(PROFILER_HUB_ENABLE_COVERAGE "Enable code coverage" OFF)

if(PROFILER_HUB_BUILD_TESTS)
    enable_testing()
endif()

include(GNUInstallDirs)

# ------------------------------------------------------------ #
# Version
# ------------------------------------------------------------ #

# Version is the single source of truth in `project(profiler-hub VERSION ...)`
# above. CMake automatically populates profiler-hub_VERSION_{MAJOR,MINOR,PATCH}
# and profiler-hub_VERSION from it.

# ------------------------------------------------------------ #
# Object Library (compiled once, used by both shared and static)
# ------------------------------------------------------------ #

add_library(profiler-hub-objects OBJECT)
set_target_properties(
    profiler-hub-objects
    PROPERTIES POSITION_INDEPENDENT_CODE ON
)

target_compile_definitions(
    profiler-hub-objects
    PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:PROFILER_HUB_VERSION_MAJOR=${profiler-hub_VERSION_MAJOR}>
    PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:PROFILER_HUB_VERSION_MINOR=${profiler-hub_VERSION_MINOR}>
    PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:PROFILER_HUB_VERSION_PATCH=${profiler-hub_VERSION_PATCH}>
)

target_compile_options(
    profiler-hub-objects
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

set(profiler-hub_public_headers
    ${CMAKE_CURRENT_LIST_DIR}/include/writer.hpp
    ${CMAKE_CURRENT_LIST_DIR}/include/reader.hpp
    ${CMAKE_CURRENT_LIST_DIR}/include/storage.hpp
    ${CMAKE_CURRENT_LIST_DIR}/include/version.hpp
    ${CMAKE_CURRENT_LIST_DIR}/include/writer_types.hpp
    ${CMAKE_CURRENT_LIST_DIR}/include/reader_types.hpp
    ${CMAKE_CURRENT_LIST_DIR}/include/shared_types.hpp
)

foreach(HEADER_FILE ${profiler-hub_public_headers})
    file(
        RELATIVE_PATH
        HEADER_REL
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${HEADER_FILE}
    )
    configure_file(
        ${HEADER_FILE}
        ${CMAKE_BINARY_DIR}/include/profiler-hub/${HEADER_REL}
        COPYONLY
    )
endforeach()

target_include_directories(
    profiler-hub-objects
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

if(PROFILER_HUB_ENABLE_COVERAGE)
    profiler_hub_add_coverage_flags(profiler-hub-objects)
endif()

# ------------------------------------------------------------ #
# SQLite3
# ------------------------------------------------------------ #

include(sqlite3)
target_link_libraries(profiler-hub-objects PRIVATE profiler-hub-sqlite3)

# ------------------------------------------------------------ #
# fmt
# ------------------------------------------------------------ #

include(fmt)
target_link_libraries(profiler-hub-objects PRIVATE fmt::fmt)

# ------------------------------------------------------------ #
# spdlog
# ------------------------------------------------------------ #

include(spdlog)
target_link_libraries(profiler-hub-objects PRIVATE spdlog::spdlog)

# ------------------------------------------------------------ #
# nlohmann_json
# ------------------------------------------------------------ #

include(nlohmann_json)
target_link_libraries(profiler-hub-objects PRIVATE nlohmann_json::nlohmann_json)

# ------------------------------------------------------------ #
# Schema from rocprofiler-sdk-rocpd
# ------------------------------------------------------------ #

set(SQL_SCHEMA_DIR ${CMAKE_CURRENT_LIST_DIR}/source/data_storage/schema)
set(SQL_SCHEMA_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/source/data_storage/)

include(rocm)
include(rocprofiler-sdk-rocpd)

if(USE_SCHEMA_FROM_ROCPROFILER_SDK_ROCPD)
    target_link_libraries(
        profiler-hub-objects
        PRIVATE rocprofiler-sdk-rocpd::rocprofiler-sdk-rocpd
    )

    target_compile_definitions(
        profiler-hub-objects
        PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:USE_SCHEMA_FROM_ROCPROFILER_SDK_ROCPD=1>
    )
else()
    target_include_directories(
        profiler-hub-objects
        PRIVATE ${SQL_SCHEMA_BINARY_DIR}
    )

    target_compile_definitions(
        profiler-hub-objects
        PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:USE_SCHEMA_FROM_ROCPROFILER_SDK_ROCPD=0>
    )
endif()

# ------------------------------------------------------------ #
# Logging
# ------------------------------------------------------------ #

if(PROFILER_HUB_ENABLE_LOGGING)
    target_compile_definitions(
        profiler-hub-objects
        PRIVATE $<$<COMPILE_LANGUAGE:CXX>:PROFILER_HUB_ENABLE_LOGGING=1>
    )
else()
    target_compile_definitions(
        profiler-hub-objects
        PRIVATE $<$<COMPILE_LANGUAGE:CXX>:PROFILER_HUB_ENABLE_LOGGING=0>
    )
endif()

# ------------------------------------------------------------ #
# Shared and Static Libraries
# ------------------------------------------------------------ #

add_library(profiler-hub SHARED $<TARGET_OBJECTS:profiler-hub-objects>)
add_library(profiler-hub-static STATIC $<TARGET_OBJECTS:profiler-hub-objects>)
set_target_properties(profiler-hub-static PROPERTIES OUTPUT_NAME profiler-hub)

target_link_options(
    profiler-hub
    PRIVATE -Wl,--exclude-libs,$<TARGET_FILE_NAME:profiler-hub-sqlite3-static>
)

# Embed semantic version + ABI version on the shared library so that
# libprofiler-hub.so.<MAJOR> is the SONAME and libprofiler-hub.so.<MAJOR>.<MINOR>.<PATCH>
# is the actual file, with both symlinks installed alongside libprofiler-hub.so.
set_target_properties(
    profiler-hub
    PROPERTIES VERSION ${PROJECT_VERSION} SOVERSION ${PROJECT_VERSION_MAJOR}
)

foreach(_target profiler-hub profiler-hub-static)
    target_include_directories(
        ${_target}
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    )

    # Link against the object library itself (not just its
    # $<TARGET_OBJECTS:>) so its PRIVATE dependencies (fmt, spdlog,
    # nlohmann_json, sqlite3, queries, rocprofiler-sdk-rocpd) are inherited
    # as usage requirements of profiler-hub-objects instead of being
    # re-declared here. $<BUILD_INTERFACE:> keeps that reference out of
    # ${_target}'s own link requirements in the installed export: for a
    # STATIC library, an un-guarded PRIVATE dependency is copied into
    # INTERFACE_LINK_LIBRARIES (the well-known reason install(EXPORT) needs
    # $<BUILD_INTERFACE:> for in-tree-only deps), and the same guard is
    # required for the SHARED target too, because install(EXPORT)'s
    # export-set validation for a SHARED library inspects its raw PRIVATE
    # link dependencies directly rather than the (normally unpopulated)
    # INTERFACE_LINK_LIBRARIES property.
    target_link_libraries(
        ${_target}
        PRIVATE $<BUILD_INTERFACE:profiler-hub-objects>
    )

    if(PROFILER_HUB_ENABLE_COVERAGE)
        target_link_options(
            ${_target}
            PRIVATE ${PROFILER_HUB_COVERAGE_LINK_FLAGS}
        )
    endif()
endforeach()

# profiler-hub-static's own link requirements no longer come from
# target_link_libraries() above (that only supplies the $<BUILD_INTERFACE:>
# object library), so consumers of the *installed* static package would
# otherwise get no transitive link requirements at all. Explicitly restore
# the ones that are genuinely resolvable by real consumers; queries,
# sqlite3, and profiler-hub-objects itself are always in-tree-only and
# never belong here. nlohmann_json is deliberately excluded too: it is a
# PRIVATE, header-only dependency of profiler-hub-objects that never appears
# in the installed public headers, so it carries no link symbols a static
# consumer needs. profiler-hub (SHARED) needs no such list: its dependencies
# are already fully baked into the .so.
#
# Each dependency added here must also get a matching find_dependency() call
# in profiler-hub-config.cmake.in (accumulated below into
# profiler_hub_static_find_dependencies), or a consumer's find_package(
# profiler-hub) + link against profiler-hub-static fails at generate time
# because the INSTALL_INTERFACE target below is unresolvable.
#
# Deliberately not keyed off ${_dep}_FOUND: that variable just reflects the
# most recent find_package(${_dep}) call in this directory scope, and can
# flip to TRUE later on (e.g. a nested FetchContent'd dependency probing for
# its own external ${_dep}) even though the actual ${_dep}::${_dep} target
# still refers to the FIRST definition - which, when the initial
# find_package() failed and we fell back to FetchContent, is our own
# in-tree, non-IMPORTED build target (aliased, not exported). Superbuild
# environments that redirect find_package() to a super-project package
# (e.g. TheRock) can legitimately fail a version-pinned find_package() while
# a later, unversioned one for the same package succeeds elsewhere in this
# scope, so check the target's actual IMPORTED-ness instead of the latching
# _FOUND variable.
set(profiler_hub_static_iface_libs "")
set(profiler_hub_static_find_dependencies "")
foreach(_dep fmt spdlog)
    set(_profiler_hub_dep_target "${_dep}::${_dep}")
    if(TARGET ${_profiler_hub_dep_target})
        get_target_property(
            _profiler_hub_dep_aliased
            ${_profiler_hub_dep_target}
            ALIASED_TARGET
        )
        if(_profiler_hub_dep_aliased)
            set(_profiler_hub_dep_target "${_profiler_hub_dep_aliased}")
        endif()
        get_target_property(
            _profiler_hub_dep_is_imported
            ${_profiler_hub_dep_target}
            IMPORTED
        )
        if(_profiler_hub_dep_is_imported)
            list(
                APPEND profiler_hub_static_iface_libs
                "$<INSTALL_INTERFACE:${_dep}::${_dep}>"
            )
            string(
                APPEND profiler_hub_static_find_dependencies
                "find_dependency(${_dep})\n"
            )
        endif()
    endif()
endforeach()
unset(_profiler_hub_dep_target)
unset(_profiler_hub_dep_aliased)
unset(_profiler_hub_dep_is_imported)
if(USE_SCHEMA_FROM_ROCPROFILER_SDK_ROCPD)
    # rocprofiler-sdk-rocpd is only ever discovered via find_package() (see
    # cmake/rocprofiler-sdk-rocpd.cmake); unlike fmt/spdlog it has no in-tree
    # FetchContent fallback, so the target is always IMPORTED here and needs
    # no IMPORTED-ness check.
    list(
        APPEND profiler_hub_static_iface_libs
        "$<INSTALL_INTERFACE:rocprofiler-sdk-rocpd::rocprofiler-sdk-rocpd>"
    )
    string(
        APPEND profiler_hub_static_find_dependencies
        "find_dependency(rocprofiler-sdk-rocpd)\n"
    )
endif()
set_target_properties(
    profiler-hub-static
    PROPERTIES INTERFACE_LINK_LIBRARIES "${profiler_hub_static_iface_libs}"
)

# ------------------------------------------------------------ #
# Install package support
# ------------------------------------------------------------ #

include(CMakePackageConfigHelpers)

set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME profiler-hub)
set(PACKAGE_NAME "profiler-hub")

install(
    TARGETS ${PACKAGE_NAME}
    EXPORT ${PACKAGE_NAME}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(
    DIRECTORY ${CMAKE_BINARY_DIR}/include/${PACKAGE_NAME}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    COMPONENT profiler-hub
    FILES_MATCHING
    PATTERN "*.hpp"
)

install(
    TARGETS ${PACKAGE_NAME}-static
    EXPORT ${PACKAGE_NAME}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT profiler-hub
)

install(
    EXPORT ${PACKAGE_NAME}
    FILE ${PACKAGE_NAME}-targets.cmake
    NAMESPACE ${PACKAGE_NAME}::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PACKAGE_NAME}
    COMPONENT profiler-hub
)

set(INCLUDE_INSTALL_DIR ${CMAKE_INSTALL_INCLUDEDIR})
set(LIB_INSTALL_DIR ${CMAKE_INSTALL_LIBDIR})

# Built from profiler_hub_static_iface_libs's accompanying
# profiler_hub_static_find_dependencies (see the comment above that loop):
# one find_dependency() per real, IMPORTED target that profiler-hub-static's
# INTERFACE_LINK_LIBRARIES references, so a consumer's find_package(
# profiler-hub) can resolve every one of them before profiler-hub-targets.cmake
# is included.
if(NOT profiler_hub_static_find_dependencies STREQUAL "")
    set(PROFILER_HUB_FIND_DEPENDENCIES
        "include(CMakeFindDependencyMacro)\n${profiler_hub_static_find_dependencies}"
    )
else()
    set(PROFILER_HUB_FIND_DEPENDENCIES "")
endif()

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/profiler-hub-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/${PACKAGE_NAME}/${PACKAGE_NAME}-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PACKAGE_NAME}
    PATH_VARS INCLUDE_INSTALL_DIR LIB_INSTALL_DIR
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/${PACKAGE_NAME}/${PACKAGE_NAME}-config-version.cmake
    VERSION ${profiler-hub_VERSION}
    COMPATIBILITY SameMinorVersion
)

install(
    FILES
        ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/${PACKAGE_NAME}/${PACKAGE_NAME}-config.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/${PACKAGE_NAME}/${PACKAGE_NAME}-config-version.cmake
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Find${PACKAGE_NAME}.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PACKAGE_NAME}
    COMPONENT profiler-hub
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
        GLOB_RECURSE PROFILER_HUB_CLANG_TIDY_SOURCES
        ${CMAKE_SOURCE_DIR}/source/*.cpp
        ${CMAKE_SOURCE_DIR}/source/*.hpp
        ${CMAKE_SOURCE_DIR}/include/*.hpp
    )

    add_custom_target(
        clang-tidy
        COMMAND
            ${CLANG_TIDY_EXE} -p ${CMAKE_BINARY_DIR}
            --config-file=${CMAKE_SOURCE_DIR}/.clang-tidy
            ${PROFILER_HUB_CLANG_TIDY_SOURCES}
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
            ${PROFILER_HUB_CLANG_TIDY_SOURCES}
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

if(PROFILER_HUB_BUILD_TESTS OR PROFILER_HUB_BUILD_BENCHMARKS)
    add_subdirectory(tests)
endif()
