# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

include(AISSanitizers)
include(FetchContent)

# This line is used (or not) in FetchContent_Declare to determine
# if we check for a system GoogleTest install first. When building
# with the sanitizers, we HAVE to build GoogleTest from source.
if(AIS_USE_SANITIZERS OR AIS_USE_THREAD_SANITIZER)
    set(AIS_LOCAL_GTEST_CHECK "")
else()
    set(AIS_LOCAL_GTEST_CHECK FIND_PACKAGE_ARGS NAMES GTest)
endif()

# lint_cmake: -readability/wonkycase
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/releases/download/v1.17.0/googletest-1.17.0.tar.gz
  DOWNLOAD_EXTRACT_TIMESTAMP true
  ${AIS_LOCAL_GTEST_CHECK}
  SYSTEM
)
# lint_cmake: +readability/wonkycase

set(INSTALL_GTEST OFF CACHE BOOL "Don't install GoogleTest")
set(GTEST_HAS_ABSL OFF CACHE BOOL "Don't use Abseil for GoogleTest")

# lint_cmake: -readability/wonkycase
FetchContent_MakeAvailable(googletest)
# lint_cmake: +readability/wonkycase

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
