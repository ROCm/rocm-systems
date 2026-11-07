# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

add_executable(bench_hooks_test test_bench_hooks.cpp)
target_link_libraries(bench_hooks_test PRIVATE bench_hooks GTest::gtest_main)
target_compile_options(bench_hooks_test PRIVATE -O2 -Wall -Wextra)
# The gtest in /usr/local/lib is a non-PIC static archive; a PIE link rejects it.
target_compile_options(bench_hooks_test PRIVATE -fno-pie)
target_link_options(bench_hooks_test PRIVATE -no-pie)

include(GoogleTest)
gtest_discover_tests(bench_hooks_test)
