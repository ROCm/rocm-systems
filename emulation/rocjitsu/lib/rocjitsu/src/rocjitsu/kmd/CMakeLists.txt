# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_subdirectory(linux)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    add_subdirectory(windows)
else()
    message(WARNING "KMD emulation not supported on ${CMAKE_SYSTEM_NAME}")
endif()
