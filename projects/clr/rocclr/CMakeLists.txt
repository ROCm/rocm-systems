# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

project(ROCclr)

if (${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_CURRENT_SOURCE_DIR})
  message(AUTHOR_WARNING "ROCclr is being built as a standalone project. This isn't supported anymore.")
endif()

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(ROCclr)

# Blit kernels built at build time for the ROCm backend; needs amdclang and the device
# libraries under ROCM_PATH. Without them the kernels are compiled at runtime.
option(CLR_OFFLINE_BLIT_KERNELS "Build the ROCm blit kernels offline" ON)
if(CLR_OFFLINE_BLIT_KERNELS AND ROCCLR_ENABLE_HSA)
  add_subdirectory(device/blit_kernels)
endif()

# Offline tests of the code object formats (no GPU); fetches Catch2
option(CODEOBJECT_BUILD_TESTS "Build the code object unit tests (codeobject_test)" OFF)
if(CODEOBJECT_BUILD_TESTS)
  add_subdirectory(device/codeobject/tests)
endif()
