# ########################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Host unit tests for the GIN-SDMA put-segmentation math in
# ../src/gin_sdma_collective_policy.h. No GPU required: the header is compiled
# as plain host C++ (GIN_SDMA_HOST_ONLY drops the __host__ __device__
# attributes), and the same functions are called by the device kernels, so
# these exercise the real segmentation code. Runs under ctest.
# ########################################################################

find_package(GTest QUIET)
if (NOT GTest_FOUND AND NOT TARGET gtest_main)
  message(STATUS "gin-sdma tests: system GTest not found, fetching googletest")
  include(FetchContent)
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz
  )
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
endif()

add_executable(gin_sdma_policy_test gin_sdma_policy_test.cpp)
# Pure host C++; keep it off the HIP device-compilation path even when the
# project compiler is hipcc/amdclang++ (GIN_SDMA_HOST_ONLY drops the
# __host__ __device__ attributes so no offload arch / device codegen is needed).
set_source_files_properties(gin_sdma_policy_test.cpp PROPERTIES LANGUAGE CXX)
target_include_directories(gin_sdma_policy_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_compile_definitions(gin_sdma_policy_test PRIVATE GIN_SDMA_HOST_ONLY)

if (TARGET GTest::gtest_main)
  target_link_libraries(gin_sdma_policy_test PRIVATE GTest::gtest_main)
else()
  target_link_libraries(gin_sdma_policy_test PRIVATE gtest_main)
endif()

set_target_properties(gin_sdma_policy_test PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

add_test(NAME gin_sdma_policy_test COMMAND gin_sdma_policy_test)
set_tests_properties(gin_sdma_policy_test PROPERTIES LABELS "unit")
