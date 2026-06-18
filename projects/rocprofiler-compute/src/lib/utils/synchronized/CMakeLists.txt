## Copyright (c) Advanced Micro Devices, Inc.
## SPDX-License-Identifier:  MIT
set(target synchronized)

add_library(${target} INTERFACE synchronized.hpp)

target_include_directories(${target} INTERFACE .)
