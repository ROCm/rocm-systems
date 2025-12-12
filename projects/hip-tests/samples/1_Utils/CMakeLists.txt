# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.21.3)
add_custom_target(build_utils)
add_subdirectory(hipDispatchLatency)
add_subdirectory(hipInfo)