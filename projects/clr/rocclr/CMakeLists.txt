# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

project(ROCclr)

if (${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_CURRENT_SOURCE_DIR})
  message(AUTHOR_WARNING "ROCclr is being built as a standalone project. This isn't supported anymore.")
endif()

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(ROCclr)

# Install ISA table schema for documentation
install(FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/device/isa_table_schema.json
  DESTINATION ${CMAKE_INSTALL_PREFIX}/share/rocclr
  COMPONENT development
  OPTIONAL
)

# Install internal ISA table if it exists (for internal builds with additional ISAs)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/device/isa_table_internal.json)
  install(FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/device/isa_table_internal.json
    DESTINATION ${CMAKE_INSTALL_PREFIX}/share/rocclr
    COMPONENT runtime
  )
  message(STATUS "Will install internal ISA table")
endif()
