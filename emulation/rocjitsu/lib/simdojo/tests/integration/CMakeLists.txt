# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.22)
project(simdojo_integration_test LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(simdojo REQUIRED)

add_executable(find_package_test find_package_test.cpp)
target_link_libraries(find_package_test PRIVATE simdojo::simdojo)

enable_testing()
add_test(NAME find_package_test COMMAND find_package_test)
