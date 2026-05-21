# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# ----------------------------------------------------------------------------------------#
#
# ROCm
#
# ----------------------------------------------------------------------------------------#

set(DEFAULT_ROCM_PATH /opt/rocm CACHE PATH "Default search path for ROCm")
if(EXISTS ${DEFAULT_ROCM_PATH})
    get_filename_component(_DEFAULT_ROCM_PATH "${DEFAULT_ROCM_PATH}" REALPATH)

    if(NOT "${_DEFAULT_ROCM_PATH}" STREQUAL "${DEFAULT_ROCM_PATH}")
        set(ROCPROFSYS_DEFAULT_ROCM_PATH
            "${_DEFAULT_ROCM_PATH}"
            CACHE PATH
            "Default search path for ROCm"
            FORCE
        )
    endif()
endif()

set(CMAKE_PREFIX_PATH ${DEFAULT_ROCM_PATH} ${CMAKE_PREFIX_PATH})
string(
    REPLACE
    ":"
    ";"
    CMAKE_PREFIX_PATH
    "$ENV{CMAKE_PREFIX_PATH};${CMAKE_PREFIX_PATH}"
)
