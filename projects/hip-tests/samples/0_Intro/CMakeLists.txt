
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.21.3)
add_custom_target(build_intro)
add_subdirectory(bit_extract)
add_subdirectory(module_api)
add_subdirectory(module_api_global)
add_subdirectory(square)
add_subdirectory(generic_target)
