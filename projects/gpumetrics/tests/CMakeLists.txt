# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

find_package(GTest REQUIRED)
include(GoogleTest)

# Two mock plugin .so's built from one source with distinct names + local-index
# bases, so the two-plugin correlation path runs in a single test process.
add_library(mockA MODULE mock_plugin.cpp)
target_compile_definitions(mockA PRIVATE MOCK_NAME="mockA" MOCK_LOCAL_BASE=0)
target_include_directories(mockA PRIVATE ${CMAKE_SOURCE_DIR}/include)
set_target_properties(mockA PROPERTIES PREFIX "lib" OUTPUT_NAME gpumetrics_mockA)

add_library(mockB MODULE mock_plugin.cpp)
target_compile_definitions(mockB PRIVATE MOCK_NAME="mockB" MOCK_LOCAL_BASE=100)
target_include_directories(mockB PRIVATE ${CMAKE_SOURCE_DIR}/include)
set_target_properties(mockB PROPERTIES PREFIX "lib" OUTPUT_NAME gpumetrics_mockB)

# device_registry test links the core sources it needs directly (the registry
# is an internal header under src/).
add_executable(test_device_registry
  test_device_registry.cpp
  ${CMAKE_SOURCE_DIR}/src/device_registry.cpp
  ${CMAKE_SOURCE_DIR}/src/status.cpp
)
target_include_directories(test_device_registry PRIVATE
  ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_device_registry PRIVATE GTest::gtest GTest::gtest_main)
gtest_discover_tests(test_device_registry)

# Collector end-to-end test links the built core library and dlopens the mocks.
add_executable(test_collector test_collector.cpp)
target_include_directories(test_collector PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_collector PRIVATE gpumetrics GTest::gtest GTest::gtest_main)
add_dependencies(test_collector mockA mockB)
gtest_discover_tests(test_collector
  PROPERTIES ENVIRONMENT "GPUM_TEST_PLUGIN_DIR=${CMAKE_CURRENT_BINARY_DIR}")

# Flat C API test (mirrors test_collector via the FFI surface the Rust bindings
# use). Runs against the mock plugins, no hardware needed.
add_executable(test_capi test_capi.cpp)
target_include_directories(test_capi PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_capi PRIVATE gpumetrics GTest::gtest GTest::gtest_main)
add_dependencies(test_capi mockA mockB)
gtest_discover_tests(test_capi
  PROPERTIES ENVIRONMENT "GPUM_TEST_PLUGIN_DIR=${CMAKE_CURRENT_BINARY_DIR}")

# ---------------------------------------------------------------------------
# Hardware integration tests + amd-smi comparison harness.
#
# Both exercise the REAL backend plugins against live GPUs. They discover the
# built plugin .so's via the GPUM_HW_PLUGIN_DIRS env var (colon-separated),
# pointing at the amdsmi + rocprofiler build output dirs. Both are written to
# SKIP / no-op gracefully when no GPU or plugin is present, so they are safe to
# run in CI on machines without hardware.
# ---------------------------------------------------------------------------

# Real plugin build directories (guarded: a plugin target only exists if its
# build option is ON). Fall back to sensible defaults when a plugin is disabled.
set(_gpum_amdsmi_dir "${CMAKE_BINARY_DIR}/plugins/amdsmi")
set(_gpum_rocprof_dir "${CMAKE_BINARY_DIR}/plugins/rocprofiler")
set(_gpum_hw_plugin_dirs "${_gpum_amdsmi_dir}:${_gpum_rocprof_dir}")

# Hardware integration test (GTest).
add_executable(test_hw_integration test_hw_integration.cpp)
target_include_directories(test_hw_integration PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_hw_integration PRIVATE gpumetrics GTest::gtest GTest::gtest_main)
if(TARGET gpumetrics_amdsmi)
  add_dependencies(test_hw_integration gpumetrics_amdsmi)
endif()
if(TARGET gpumetrics_rocprofiler)
  add_dependencies(test_hw_integration gpumetrics_rocprofiler)
endif()
gtest_discover_tests(test_hw_integration
  TEST_PREFIX "hw."
  PROPERTIES ENVIRONMENT "GPUM_HW_PLUGIN_DIRS=${_gpum_hw_plugin_dirs}")

# amd-smi comparison harness (standalone main(), registered as a plain test).
add_executable(compare_amdsmi compare_amdsmi.cpp)
target_include_directories(compare_amdsmi PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(compare_amdsmi PRIVATE gpumetrics)
if(TARGET gpumetrics_amdsmi)
  add_dependencies(compare_amdsmi gpumetrics_amdsmi)
endif()
if(TARGET gpumetrics_rocprofiler)
  add_dependencies(compare_amdsmi gpumetrics_rocprofiler)
endif()
add_test(NAME compare_amdsmi COMMAND compare_amdsmi)
set_tests_properties(compare_amdsmi PROPERTIES
  ENVIRONMENT "GPUM_HW_PLUGIN_DIRS=${_gpum_hw_plugin_dirs}")
