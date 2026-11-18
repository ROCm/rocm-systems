# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

add_library(util STATIC src/statistics.cpp)
set_target_properties(util PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(util PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(util PRIVATE Threads::Threads)
include(rj_configure_target)
rj_configure_target(util WARNINGS HIDDEN)
