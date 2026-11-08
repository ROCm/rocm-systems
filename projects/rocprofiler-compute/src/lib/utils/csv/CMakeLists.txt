## Copyright (c) Advanced Micro Devices, Inc.
## SPDX-License-Identifier:  MIT
set(target csv)

add_library(${target} STATIC csv.cpp csv.h)

# Consumers include "csv/<header>.h", so the parent directory is what goes on
# the include path.
target_include_directories(${target} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)

if(ENABLE_TESTS)
    add_subdirectory(tests)
endif()
