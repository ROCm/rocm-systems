# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

find_package(rocprofiler-sdk REQUIRED)
find_package(hsa-runtime64 REQUIRED)

add_library(gpumetrics_rocprofiler MODULE plugin.cpp)

# Device-counting APIs are gated behind the experimental define.
target_compile_definitions(gpumetrics_rocprofiler PRIVATE
  ROCPROFILER_SDK_EXPERIMENTAL
)

target_include_directories(gpumetrics_rocprofiler PRIVATE
  ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(gpumetrics_rocprofiler PRIVATE
  rocprofiler-sdk::rocprofiler-sdk
  hsa-runtime64::hsa-runtime64
  dl
  pthread
)

# The core dlopens the plugin RTLD_GLOBAL; rocprofiler_configure and the
# gpum entry point must be visible in the global namespace.
set_target_properties(gpumetrics_rocprofiler PROPERTIES
  PREFIX "lib"
  OUTPUT_NAME gpumetrics_rocprofiler
  CXX_VISIBILITY_PRESET hidden
)

install(TARGETS gpumetrics_rocprofiler LIBRARY DESTINATION lib/gpumetrics)
