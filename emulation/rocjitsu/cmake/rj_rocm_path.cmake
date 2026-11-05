# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# An explicitly empty path disables the environment and /opt/rocm fallbacks.
if(NOT DEFINED ROCM_PATH)
    if(DEFINED ENV{ROCM_PATH} AND NOT "$ENV{ROCM_PATH}" STREQUAL "")
        set(ROCM_PATH
            "$ENV{ROCM_PATH}"
            CACHE PATH
            "Path to the ROCm installation"
        )
    elseif(IS_DIRECTORY "/opt/rocm")
        set(ROCM_PATH "/opt/rocm" CACHE PATH "Path to the ROCm installation")
    endif()
endif()

if(NOT "${ROCM_PATH}" STREQUAL "" AND NOT IS_DIRECTORY "${ROCM_PATH}")
    message(FATAL_ERROR "ROCM_PATH must be set to a valid directory")
endif()

# Re-evaluate tools in the selected SDK instead of retaining another SDK's cache.
function(rj_find_rocm_program out_var program_name)
    find_program(
        _rj_selected_program
        NAMES ${program_name}
        PATHS "${ROCM_PATH}"
        PATH_SUFFIXES lib/llvm/bin bin
        NO_DEFAULT_PATH
        NO_CACHE
    )
    set(${out_var}
        "${_rj_selected_program}"
        CACHE FILEPATH
        "${program_name} from the selected ROCM_PATH"
        FORCE
    )
    set(${out_var} "${_rj_selected_program}" PARENT_SCOPE)
endfunction()
