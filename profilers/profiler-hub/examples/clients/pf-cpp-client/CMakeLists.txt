# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

add_executable(prof-hub-client)

target_sources(prof-hub-client PRIVATE ${CMAKE_CURRENT_LIST_DIR}/main.cpp)

target_link_libraries(prof-hub-client PRIVATE profiler-hub fmt)
