# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# A consumer that is C and nothing else, as the API allows.
cmake_minimum_required(VERSION 3.20)
project(amdcuid_c_consumer LANGUAGES C)
find_package(amdcuid CONFIG REQUIRED)
add_executable(c_consumer main.c)
target_link_libraries(c_consumer PRIVATE amdcuid::amdcuid)
