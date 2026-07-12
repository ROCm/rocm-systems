# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

add_executable(gpumetrics_cli main.cpp)
set_target_properties(gpumetrics_cli PROPERTIES OUTPUT_NAME gpumetrics)
target_include_directories(gpumetrics_cli PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(gpumetrics_cli PRIVATE gpumetrics)
install(TARGETS gpumetrics_cli RUNTIME DESTINATION bin)
