# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

add_executable(clinfo clinfo.cpp)

target_compile_definitions(clinfo PRIVATE CL_TARGET_OPENCL_VERSION=220 HAVE_CL2_HPP)

#todo: to be updated to use header files from other repos
target_include_directories(clinfo PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../../khronos/headers/opencl2.2")

target_link_libraries(clinfo PRIVATE ${OPENCL_ICD_LIB})

INSTALL(TARGETS clinfo
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
