# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

add_executable(ph-cpp-client)

target_sources(ph-cpp-client PRIVATE ${CMAKE_CURRENT_LIST_DIR}/main.cpp)

target_link_libraries(ph-cpp-client PRIVATE profiler-hub fmt)
