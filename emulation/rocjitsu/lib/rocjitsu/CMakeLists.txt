# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

set(ROCJITSU_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(ROCJITSU_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

include(rj_add_object_library)

add_subdirectory(src/rocjitsu/analysis)
add_subdirectory(src/rocjitsu/code)
add_subdirectory(src/rocjitsu/isa)
add_subdirectory(src/rocjitsu/vm)
add_subdirectory(src/rocjitsu/kmd)
add_subdirectory(src/rocjitsu/config)
add_subdirectory(src/rocjitsu/hooks)
