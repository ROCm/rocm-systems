# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

add_executable(ph-c-client)

target_sources(ph-c-client PRIVATE ${CMAKE_CURRENT_LIST_DIR}/main.c)

target_link_libraries(ph-c-client PRIVATE profiler-hub)
