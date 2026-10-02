# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.27)
project(ROCclrFlagsTests LANGUAGES CXX)

enable_testing()
add_executable(rocclr_flags_test flags_test.cpp)
target_compile_features(rocclr_flags_test PRIVATE cxx_std_17)
target_compile_definitions(
    rocclr_flags_test
    PRIVATE CL_TARGET_OPENCL_VERSION=220
)
target_include_directories(
    rocclr_flags_test
    PRIVATE ../include .. ../../opencl/khronos/headers/opencl2.2
)
add_test(NAME rocclr_flags_test COMMAND rocclr_flags_test)
