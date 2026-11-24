# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

include(GoogleTest)

foreach(name bench_hooks parse_args)
    add_executable(${name}_test test_${name}.cpp)
    target_link_libraries(${name}_test PRIVATE bench_hooks GTest::gtest_main)
    target_include_directories(${name}_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
    target_compile_options(${name}_test PRIVATE -O2 -Wall -Wextra)
    # The gtest in /usr/local/lib is a non-PIC static archive; a PIE link rejects it.
    target_compile_options(${name}_test PRIVATE -fno-pie)
    target_link_options(${name}_test PRIVATE -no-pie)
    gtest_discover_tests(${name}_test)
endforeach()
