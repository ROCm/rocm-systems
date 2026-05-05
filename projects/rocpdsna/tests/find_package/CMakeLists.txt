# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.21)
project(rocpdsna_find_package_test LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(rocpdsna REQUIRED)

add_executable(rocpdsna_consumer consumer.cpp)
target_link_libraries(rocpdsna_consumer PRIVATE rocpdsna::rocpdsna)

enable_testing()
add_test(NAME rocpdsna_consumer_smoke COMMAND rocpdsna_consumer)
