# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

find_package(amd_smi REQUIRED)

add_library(gpumetrics_amdsmi MODULE plugin.cpp)
target_include_directories(gpumetrics_amdsmi PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(gpumetrics_amdsmi PRIVATE amd_smi)
set_target_properties(gpumetrics_amdsmi PROPERTIES
  PREFIX "lib"
  OUTPUT_NAME gpumetrics_amdsmi)
