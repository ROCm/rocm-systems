# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

if(NOT DEFINED ROCJITSU_SOURCE_DIR)
    message(FATAL_ERROR "ROCJITSU_SOURCE_DIR is required")
endif()

set(GPU_MEMORY_HEADER
    "${ROCJITSU_SOURCE_DIR}/lib/rocjitsu/src/rocjitsu/vm/amdgpu/gpu_memory.h"
)
file(READ "${GPU_MEMORY_HEADER}" GPU_MEMORY_CONTENT)

foreach(
    FORBIDDEN_TOKEN
    IN
    ITEMS
        "LegacyAddressSpace"
        "simdojo::Port"
        "cpl_port"
        "vmid"
        "register_process"
        "set_passthrough"
        "translate("
)
    string(FIND "${GPU_MEMORY_CONTENT}" "${FORBIDDEN_TOKEN}" TOKEN_POSITION)
    if(NOT TOKEN_POSITION EQUAL -1)
        message(
            FATAL_ERROR
            "gpu_memory.h contains VM or transport responsibility ${FORBIDDEN_TOKEN}"
        )
    endif()
endforeach()
