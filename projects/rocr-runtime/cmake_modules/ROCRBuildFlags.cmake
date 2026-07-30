# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

if(DEFINED ROCR_BUILD_FLAGS_MODULE_INCLUDED)
    return()
endif()
set(ROCR_BUILD_FLAGS_MODULE_INCLUDED TRUE)

if(NOT ROCM_SYSTEMS_ROOT)
    get_filename_component(
        ROCM_SYSTEMS_ROOT
        "${CMAKE_CURRENT_LIST_DIR}/../../.."
        ABSOLUTE
    )
endif()
include("${ROCM_SYSTEMS_ROOT}/cmake/ROCMBuildFlags.cmake")

set(_ROCR_BUILD_FLAGS_HEADER_TEMPLATE
    "${CMAKE_CURRENT_LIST_DIR}/../runtime/hsa-runtime/core/inc/rocr_build_flags.h.in"
)

function(rocr_configure_build_flags output_file)
    rocm_resolve_build_flag(
      NAME HSA_HOTSWAP_ENABLE
      TYPE INTEGER
      CACHE_VARIABLE ROCR_HSA_HOTSWAP_ENABLE_DEFAULT
      DEFAULT_VALUE 1
      VALID_VALUES 0 1 2
      DESCRIPTION "Default HSA hotswap backend (0=disabled, 1=COMGR, 2=ROCJIT)"
      OUTPUT_VARIABLE _hsa_hotswap_enable
    )

    get_filename_component(_output_dir "${output_file}" DIRECTORY)
    file(MAKE_DIRECTORY "${_output_dir}")
    configure_file(
        "${_ROCR_BUILD_FLAGS_HEADER_TEMPLATE}"
        "${output_file}"
        @ONLY
    )
endfunction()
