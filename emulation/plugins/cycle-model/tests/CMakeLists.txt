# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

add_executable(cycle_model_smoke smoke_test.cpp)
target_link_libraries(cycle_model_smoke PRIVATE cycle_model GTest::gtest_main)

# The sidecar arch config lives in the rocjitsu tree, not the lib dir.
target_compile_definitions(cycle_model_smoke PRIVATE
  CYCLE_MODEL_CDNA4_JSON="${CMAKE_CURRENT_SOURCE_DIR}/../../../rocjitsu/configs/cycle_model/cdna4.json")

add_executable(cycle_model_scheduler_test scheduler_test.cpp)
target_link_libraries(cycle_model_scheduler_test PRIVATE cycle_model GTest::gtest_main)

add_executable(cycle_model_memory_test memory_system_test.cpp)
target_link_libraries(cycle_model_memory_test PRIVATE cycle_model GTest::gtest_main)
target_compile_definitions(cycle_model_memory_test PRIVATE
  CYCLE_MODEL_CDNA4_JSON="${CMAKE_CURRENT_SOURCE_DIR}/../../../rocjitsu/configs/cycle_model/cdna4.json")

include(GoogleTest)
gtest_discover_tests(cycle_model_smoke)
gtest_discover_tests(cycle_model_scheduler_test)
gtest_discover_tests(cycle_model_memory_test)
