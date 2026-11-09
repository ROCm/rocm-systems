# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT


option(HRR_BUILD_INTEGRATION_TESTS "Build HRR GPU capture/replay integration tests" OFF)

find_package(Catch2 3.8.1 QUIET)
if(NOT Catch2_FOUND)
  include(FetchContent)
  FetchContent_Declare(
    Catch2
    URL "https://rocm-third-party-deps.s3.us-east-2.amazonaws.com/Catch2-3.8.1.tar.gz"
  )
  FetchContent_MakeAvailable(Catch2)
endif()

find_package(hip CONFIG REQUIRED)

add_subdirectory(unit)

find_package(Python3 COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND)
  add_test(NAME hrr-generator-rules
    COMMAND ${Python3_EXECUTABLE} -m unittest discover -s ${CMAKE_CURRENT_SOURCE_DIR}/../tools/tests
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..)
endif()

if(HRR_BUILD_INTEGRATION_TESTS)
  if(NOT HRR_BUILD_PLAYBACK)
    message(FATAL_ERROR "HRR_BUILD_INTEGRATION_TESTS requires HRR_BUILD_PLAYBACK because integration tests spawn hrr-playback")
  endif()
  add_subdirectory(integration)
endif()
