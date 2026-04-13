# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

rj_add_object_library(rocjitsu_code
    amdgpu_code_object.cpp
    basic_block.cpp
    executable.cpp
    kernel_metadata.cpp
    rj_code.cpp
)
target_link_libraries(rocjitsu_code PUBLIC msgpack-cxx)
