# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

add_library(gpumetrics SHARED
  status.cpp
  plugin_host.cpp
  device_registry.cpp
  collector.cpp
  capi.cpp
)
add_library(gpumetrics::gpumetrics ALIAS gpumetrics)

target_include_directories(gpumetrics PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(gpumetrics PUBLIC ${CMAKE_DL_LIBS})
set_target_properties(gpumetrics PROPERTIES
  VERSION ${PROJECT_VERSION}
  SOVERSION ${PROJECT_VERSION_MAJOR}
)

install(TARGETS gpumetrics EXPORT gpumetricsTargets LIBRARY DESTINATION lib)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/gpumetrics DESTINATION include)
