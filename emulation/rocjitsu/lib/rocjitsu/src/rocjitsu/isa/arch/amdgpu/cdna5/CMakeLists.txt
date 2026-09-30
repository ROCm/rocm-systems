# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

rj_add_split_amdgpu_isa_library(cdna5)

if(TARGET rocjitsu_x86_v4_provider_object)
    # This object joins the math kernels in the shared v4 provider. It is never
    # propagated through the ISA interface to standalone execution images.
    rj_add_object_library(
        rocjitsu_cdna5_wmma_v4_object
        ${CMAKE_CURRENT_SOURCE_DIR}/../generated/cdna5/wmma_v4_exec.cpp
    )
    target_compile_options(
        rocjitsu_cdna5_wmma_v4_object
        PRIVATE -O3 -march=x86-64-v4
    )
    target_compile_definitions(
        rocjitsu_cdna5_wmma_v4_object
        PRIVATE
            ROCJITSU_DEFER_FORCE_SCALAR_INIT=1
            "ROCJITSU_CDNA5_WMMA_BUILD_DIGEST=\"${RJ_X86_V4_BUILD_DIGEST}\""
    )
endif()
