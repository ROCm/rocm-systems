# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

include(AISSanitizers)
include(FetchContent)

# Decide whether to check for a system GoogleTest install first. When building
# with the sanitizers, we HAVE to build GoogleTest from source.
if(AIS_USE_SANITIZERS OR AIS_USE_THREAD_SANITIZER)
    set(AIS_GTEST_TRY_SYSTEM FALSE)
else()
    set(AIS_GTEST_TRY_SYSTEM TRUE)
endif()

set(INSTALL_GTEST OFF CACHE BOOL "Don't install GoogleTest")
set(GTEST_HAS_ABSL OFF CACHE BOOL "Don't use Abseil for GoogleTest")

# FIND_PACKAGE_ARGS was implemented in CMake 3.24:
# https://cmake.org/cmake/help/latest/module/FetchContent.html
# CMake 3.24 is not available on all systems by default.
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24")
    if(AIS_GTEST_TRY_SYSTEM)
        set(AIS_LOCAL_GTEST_CHECK FIND_PACKAGE_ARGS NAMES GTest)
    else()
        set(AIS_LOCAL_GTEST_CHECK "")
    endif()

# lint_cmake: -readability/wonkycase
    FetchContent_Declare(
      googletest
      URL https://github.com/google/googletest/releases/download/v1.17.0/googletest-1.17.0.tar.gz
      DOWNLOAD_EXTRACT_TIMESTAMP true
      ${AIS_LOCAL_GTEST_CHECK}
      SYSTEM
    )
    FetchContent_MakeAvailable(googletest)
# lint_cmake: +readability/wonkycase
else()
    # CMake < 3.24: FIND_PACKAGE_ARGS unavailable, do the find/fetch manually.
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
endif()

if(googletest_SOURCE_DIR)
    message(STATUS "Using fetched GoogleTest")
else()
    message(STATUS "Using system GoogleTest")
endif()

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
