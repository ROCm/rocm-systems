# Copyright (c) 2026 Advanced Micro Devices, Inc.

# CPU-only host unit tests for rccl-tests helpers (no GPU/MPI).
option(BUILD_RCCL_TESTS_HOST_UNIT_TESTS "Build rccl-tests host unit tests" ON)

if(NOT BUILD_RCCL_TESTS_HOST_UNIT_TESTS)
  return()
endif()

find_package(GTest QUIET)
if(GTest_FOUND)
  set(_RCCL_TESTS_GTEST_LIB GTest::gtest)
  set(_RCCL_TESTS_GTEST_MAIN GTest::gtest_main)
else()
  include(FetchContent)
  message(STATUS "rccl-tests: downloading GTest via FetchContent")
  FetchContent_Declare(googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0
    GIT_SHALLOW    TRUE
  )
  FetchContent_MakeAvailable(googletest)
  set(_RCCL_TESTS_GTEST_LIB gtest)
  set(_RCCL_TESTS_GTEST_MAIN gtest_main)
endif()

add_executable(rccl-tests-GinSdmaDevtimeHostUnit gin_sdma_devtime_host_test.cpp)
target_include_directories(rccl-tests-GinSdmaDevtimeHostUnit PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_link_libraries(rccl-tests-GinSdmaDevtimeHostUnit PRIVATE
  ${_RCCL_TESTS_GTEST_LIB} ${_RCCL_TESTS_GTEST_MAIN})
set_target_properties(rccl-tests-GinSdmaDevtimeHostUnit PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test")
add_test(NAME rccl-tests-GinSdmaDevtimeHostUnit
         COMMAND rccl-tests-GinSdmaDevtimeHostUnit)
set_tests_properties(rccl-tests-GinSdmaDevtimeHostUnit PROPERTIES LABELS "unit")
