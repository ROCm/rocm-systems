# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

set(data_storage_sources ${CMAKE_CURRENT_LIST_DIR}/backends/sqlite_backend.cpp)

target_sources(profiler-hub-objects PRIVATE ${data_storage_sources})
