# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.21)
project(profiler-hub_find_package_test LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(profiler-hub REQUIRED)

add_executable(profiler-hub_consumer consumer.cpp)
target_link_libraries(profiler-hub_consumer PRIVATE profiler-hub::profiler-hub)

enable_testing()
add_test(NAME profiler-hub_consumer_smoke COMMAND profiler-hub_consumer)
