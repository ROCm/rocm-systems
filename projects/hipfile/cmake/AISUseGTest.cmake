# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

# This module sets up GoogleTest and applies settings to its targets, so it must
# run exactly once.
include_guard(GLOBAL)

include(AISSanitizers)
include(FetchContent)

# GoogleTest's ThreadLocal<T> implementation uses POSIX thread-key APIs
# (pthread_key_create, pthread_setspecific, etc.). These must be linked
# explicitly: lld (used by the sanitizer toolchain) errors on the undefined
# symbols in libgtest.a / libgmock.a, whereas GNU ld resolves them
# transitively and masks the missing dependency.
find_package(Threads REQUIRED)

# Decide whether to check for a system GoogleTest install first. When building
# with the sanitizers, we HAVE to build GoogleTest from source.
if(AIS_USE_SANITIZERS OR AIS_USE_THREAD_SANITIZER)
    set(AIS_GTEST_TRY_SYSTEM FALSE)
else()
    set(AIS_GTEST_TRY_SYSTEM TRUE)
endif()

set(INSTALL_GTEST OFF CACHE BOOL "Don't install GoogleTest")
set(GTEST_HAS_ABSL OFF CACHE BOOL "Don't use Abseil for GoogleTest")

# We use find_package manually instead of FIND_PACKAGE_ARGS with
# FetchContent_Declare, since FIND_PACKAGE_ARGS was introduced in CMake
# version 3.24, which is unavailable on some operating systems:
# https://cmake.org/cmake/help/latest/module/FetchContent.html
if(AIS_GTEST_TRY_SYSTEM)
    find_package(GTest QUIET)
endif()

if(NOT GTest_FOUND)
# lint_cmake: -readability/wonkycase
    FetchContent_Declare(
      googletest
      URL https://github.com/google/googletest/releases/download/v1.17.0/googletest-1.17.0.tar.gz
      DOWNLOAD_EXTRACT_TIMESTAMP true
      SYSTEM
    )
    FetchContent_MakeAvailable(googletest)
# lint_cmake: +readability/wonkycase
endif()

if(googletest_SOURCE_DIR)
    message(STATUS "Using fetched GoogleTest")
else()
    message(STATUS "Using system GoogleTest")
endif()

if(AIS_USE_SANITIZERS OR AIS_USE_THREAD_SANITIZER)
    ais_add_sanitizers(GTest::gtest)
    ais_add_sanitizers(GTest::gmock)
endif()

# Propagate pthread linkage to anything that links a GTest target, so the
# individual test CMakeLists files don't each have to add Threads::Threads.
# Fetched GoogleTest exposes GTest::gtest/GTest::gmock as ALIAS targets, which
# cannot be modified directly, so resolve to the underlying target first. We
# use set_property(... APPEND ...) rather than target_link_libraries() so this
# works uniformly for the IMPORTED targets that a system find_package(GTest)
# produces as well as the normal targets from a fetched build. A system GTest
# install may provide gtest without gmock, so only touch targets that exist.
#
# A from-source GoogleTest already links Threads::Threads on its targets, so
# check before appending to avoid a duplicate entry in the link interface.
foreach(gtest_target GTest::gtest GTest::gmock)
    if(NOT TARGET ${gtest_target})
        continue()
    endif()
    get_target_property(aliased ${gtest_target} ALIASED_TARGET)
    if(aliased)
        set(gtest_target ${aliased})
    endif()
    get_target_property(existing_libs ${gtest_target} INTERFACE_LINK_LIBRARIES)
    if(NOT existing_libs MATCHES "Threads::Threads")
        set_property(TARGET ${gtest_target} APPEND PROPERTY
            INTERFACE_LINK_LIBRARIES Threads::Threads)
    endif()

    # Mark GoogleTest's headers as system includes so consumers pull them in
    # via -isystem. clang-tidy suppresses diagnostics from system headers by
    # default, which keeps GoogleTest's macros and templates from polluting our
    # test builds with warnings we can't fix. The SYSTEM keyword passed to
    # FetchContent_Declare only takes effect on CMake >= 3.25, and a system
    # find_package(GTest) is not guaranteed to mark them either, so set the
    # property here directly to cover every path.
    set_target_properties(${gtest_target} PROPERTIES
        INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
            "$<TARGET_PROPERTY:${gtest_target},INTERFACE_INCLUDE_DIRECTORIES>")
endforeach()

include(GoogleTest)

function(ais_gtest_discover_tests target)
    cmake_language(CALL gtest_discover_tests ${ARGV})

    if(AIS_USE_CODE_COVERAGE)
        set(options)
        set(oneValueArgs TEST_LIST)
        set(multiValueArgs)
        cmake_parse_arguments(PARSE_ARGV 0 arg "${options}" "${oneValueArgs}" "${multiValueArgs}")
        if(NOT arg_TEST_LIST)
            # Set to target name if not specified. This will result in collisions
            # if we run the same test binary in different configurations without
            # specifying a test list.
            set(arg_TEST_LIST ${target})
        endif()

        set(coverage_include_file "${CMAKE_CURRENT_BINARY_DIR}/${arg_TEST_LIST}_coverage_include.cmake")
        set_property(
            DIRECTORY APPEND PROPERTY
            TEST_INCLUDE_FILES
            "${HIPFILE_ROOT_PATH}/cmake/AISSetCoverageFile.cmake"
            "${coverage_include_file}"
        )

        file(WRITE "${coverage_include_file}"
            "ais_set_coverage_file(\"${arg_TEST_LIST}\" \"${CMAKE_CURRENT_BINARY_DIR}\")"
        )
    endif()
endfunction()
