## Copyright (c) Advanced Micro Devices, Inc.
## SPDX-License-Identifier:  MIT
set(target gsl_assert)

add_library(${target} INTERFACE gsl_assert.h)
target_include_directories(${target} INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
