#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

set(SRC_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/plugin.c
)

add_library(rccl-env-json SHARED ${SRC_FILES})

target_include_directories(rccl-env-json PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../example
)

set_target_properties(rccl-env-json PROPERTIES
  OUTPUT_NAME "rccl-env-json"
  PREFIX "lib"
  POSITION_INDEPENDENT_CODE ON
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/test/unit/plugins
)
