# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# Host-side unit tests for ROCclr internals. Opt-in via -DBUILD_TESTS=ON (the
# same switch that enables the OpenCL tests); requires an installed GoogleTest.
find_package(GTest REQUIRED)

add_executable(rocclr_unit_tests memobjmap_erase_test.cpp)
target_compile_features(rocclr_unit_tests PRIVATE cxx_std_17)
target_include_directories(rocclr_unit_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../..)
target_link_libraries(rocclr_unit_tests PRIVATE GTest::gtest_main)

add_test(NAME rocclr_unit_tests COMMAND rocclr_unit_tests)
