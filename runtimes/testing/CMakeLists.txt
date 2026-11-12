# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.25)
project(ROCmRuntimeTests LANGUAGES NONE)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
    message(
        FATAL_ERROR
        "Runtime tests require an out-of-source build directory"
    )
endif()
if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" OR CMAKE_CROSSCOMPILING)
    message(
        FATAL_ERROR
        "Runtime test artifacts currently require a native Linux build"
    )
endif()
if(CMAKE_CONFIGURATION_TYPES)
    message(
        FATAL_ERROR
        "Runtime test prefixes require a single-configuration generator"
    )
endif()
if(DEFINED BUILD_TESTING AND NOT BUILD_TESTING)
    message(FATAL_ERROR "The standalone testing root requires BUILD_TESTING=ON")
endif()
set(ROCM_RUNTIMES_TEST_FAMILIES
    runtime-api
    CACHE STRING
    "Selected test families"
)
if(NOT ROCM_RUNTIMES_TEST_FAMILIES)
    message(FATAL_ERROR "ROCM_RUNTIMES_TEST_FAMILIES must be nonempty")
endif()
set(_selected_families "${ROCM_RUNTIMES_TEST_FAMILIES}")
list(REMOVE_DUPLICATES _selected_families)
foreach(_family IN LISTS _selected_families)
    if(NOT _family MATCHES "^(runtime-api|ddi|kfd)$")
        message(FATAL_ERROR "Unsupported test family: ${_family}")
    endif()
endforeach()
set(_selection_identity "${_selected_families}")
list(SORT _selection_identity)
if(
    DEFINED _runtime_test_configured_families
    AND NOT _runtime_test_configured_families STREQUAL _selection_identity
)
    message(
        FATAL_ERROR
        "Changing test families requires a fresh build directory; the staged prefix retains its original ownership"
    )
endif()
set(_runtime_test_configured_families
    "${_selection_identity}"
    CACHE INTERNAL
    "Artifact family ownership"
)

set(runtime_test_prefix "${CMAKE_CURRENT_BINARY_DIR}/tests")
include(cmake/RegisterSuite.cmake)
foreach(_family IN LISTS _selected_families)
    if(_family STREQUAL "runtime-api")
        add_subdirectory(runtime-api/amdf)
    elseif(_family STREQUAL "ddi")
        add_subdirectory(ddi/rocddi)
    elseif(_family STREQUAL "kfd")
        add_subdirectory(drivers/kfd)
    endif()
endforeach()
configure_file(
    cmake/RunTests.cmake
    "${runtime_test_prefix}/cmake/RunTests.cmake"
    COPYONLY
)
configure_file(
    cmake/CTestTestfile.cmake.in
    "${runtime_test_prefix}/CTestTestfile.cmake"
    @ONLY
)
install(
    DIRECTORY "${runtime_test_prefix}/"
    DESTINATION .
    USE_SOURCE_PERMISSIONS
)
message(
    STATUS
    "Test families: ${_selected_families}; completed artifact prefix: ${runtime_test_prefix}"
)
