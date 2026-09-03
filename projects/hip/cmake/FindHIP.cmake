# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

###############################################################################
# FindHIP.cmake — Compatibility shim for legacy find_package(HIP MODULE) callers.
#
# New projects should use CMake's native HIP language support directly:
#   project(myproject LANGUAGES CXX HIP)
#   find_package(hip CONFIG REQUIRED)
#
# This shim exists so existing callers that do find_package(HIP) or
# find_package(HIP QUIET) continue to work without modification.
# All callers in TheRock only use HIP_FOUND and the hip::host / hip::device
# imported targets, both of which are provided by find_package(hip CONFIG).
###############################################################################

# Delegate to CMake's HIP CONFIG package which provides hip::host, hip::device,
# and hip_VERSION without requiring hipcc or hipconfig.
find_package(hip CONFIG QUIET
    PATHS
        "${HIP_DIR}"
        "${ROCM_PATH}"
        "$ENV{ROCM_PATH}"
        "$ENV{HIP_PATH}"
        /opt/rocm
)

# Populate the legacy HIP_FOUND variable that callers check.
if(hip_FOUND)
    set(HIP_FOUND TRUE)
    set(HIP_VERSION "${hip_VERSION}")
    if(HIP_VERSION)
        string(REPLACE "." ";" _hip_version_list "${HIP_VERSION}")
        list(GET _hip_version_list 0 HIP_VERSION_MAJOR)
        list(GET _hip_version_list 1 HIP_VERSION_MINOR)
        list(GET _hip_version_list 2 HIP_VERSION_PATCH)
    endif()
else()
    set(HIP_FOUND FALSE)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(HIP
    REQUIRED_VARS HIP_FOUND
    VERSION_VAR   HIP_VERSION
)
