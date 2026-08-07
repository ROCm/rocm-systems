# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

include(rj_find_amdcxx)
rj_find_amdcxx(AMDCXX)

if(NOT AMDCXX)
    message(
        STATUS
        "amdclang++ not found - VGPR saturation benchmark check disabled"
    )
    return()
endif()

if(DEFINED SKIP_DEVICE_KERNELS AND SKIP_DEVICE_KERNELS)
    message(
        STATUS
        "SKIP_DEVICE_KERNELS set - VGPR saturation benchmark check disabled"
    )
    return()
endif()

find_package(Python3 COMPONENTS Interpreter REQUIRED)

set(VGPR_SATURATION_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/vgpr_saturation.hip)
set(VGPR_SATURATION_COUNTS
    ${CMAKE_CURRENT_SOURCE_DIR}/vgpr_saturation_counts.def
)
set(VGPR_SATURATION_VERIFIER ${CMAKE_CURRENT_SOURCE_DIR}/verify_resources.py)
set(VGPR_SATURATION_BINARY
    ${CMAKE_CURRENT_BINARY_DIR}/vgpr_saturation${CMAKE_EXECUTABLE_SUFFIX}
)
set(VGPR_SATURATION_RESOURCE_STAMP
    ${CMAKE_CURRENT_BINARY_DIR}/vgpr_saturation_resources.stamp
)

add_custom_command(
    OUTPUT ${VGPR_SATURATION_BINARY}
    COMMAND
        ${AMDCXX} -x hip -std=c++20 -O3 -Wall -Wextra -Werror
        --offload-arch=gfx942 --rocm-path=${ROCM_PATH} ${VGPR_SATURATION_SOURCE}
        -o ${VGPR_SATURATION_BINARY}
    DEPENDS ${VGPR_SATURATION_SOURCE} ${VGPR_SATURATION_COUNTS}
    COMMENT "Building VGPR saturation benchmark (gfx942)"
    VERBATIM
)

add_custom_command(
    OUTPUT ${VGPR_SATURATION_RESOURCE_STAMP}
    COMMAND
        ${Python3_EXECUTABLE} ${VGPR_SATURATION_VERIFIER} --compiler ${AMDCXX}
        --rocm-path ${ROCM_PATH} --arch gfx942
    COMMAND ${CMAKE_COMMAND} -E touch ${VGPR_SATURATION_RESOURCE_STAMP}
    DEPENDS
        ${VGPR_SATURATION_SOURCE}
        ${VGPR_SATURATION_COUNTS}
        ${VGPR_SATURATION_VERIFIER}
    COMMENT "Verifying VGPR saturation benchmark resources (gfx942)"
    VERBATIM
)

add_custom_target(
    vgpr_saturation_check
    DEPENDS ${VGPR_SATURATION_BINARY} ${VGPR_SATURATION_RESOURCE_STAMP}
)

add_test(
    NAME VgprSaturationResourceCheck
    COMMAND
        ${Python3_EXECUTABLE} ${VGPR_SATURATION_VERIFIER} --compiler ${AMDCXX}
        --rocm-path ${ROCM_PATH} --arch gfx942
)
set_tests_properties(
    VgprSaturationResourceCheck
    PROPERTIES LABELS benchmark TIMEOUT 120
)
