# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Compile each boundary header first in its own translation unit. These
# objects are never linked or executed, so AVX-512 checks also run on AVX2 hosts.
set(_headers
    rocjitsu/isa/arch/amdgpu/shared/mma/layout.h
    rocjitsu/isa/arch/amdgpu/shared/mma/register_access.h
    rocjitsu/isa/arch/amdgpu/shared/mma/extract.h
    rocjitsu/isa/arch/amdgpu/shared/mma/arithmetic.h
    rocjitsu/isa/arch/amdgpu/shared/simd/execute_vop1.h
    rocjitsu/isa/arch/amdgpu/shared/simd/execute_vop2.h
    rocjitsu/isa/arch/amdgpu/shared/simd/execute_vopc.h
    rocjitsu/isa/arch/amdgpu/shared/simd/execute_vop3_integer.h
    rocjitsu/isa/arch/amdgpu/shared/simd/execute_vop3_float.h
    rocjitsu/isa/arch/amdgpu/shared/simd/execute_vop3p.h
    rocjitsu/isa/arch/amdgpu/shared/simd/execute_vopd.h
    rocjitsu/isa/arch/amdgpu/shared/mma/arguments.h
    rocjitsu/isa/arch/amdgpu/shared/mma/staging_limits.h
    rocjitsu/isa/arch/amdgpu/shared/mma/common.h
    rocjitsu/isa/arch/amdgpu/shared/mma/select.h
    rocjitsu/isa/arch/amdgpu/shared/mma_exec.h
    rocjitsu/isa/arch/amdgpu/shared/simd/execute.h
    rocjitsu/isa/arch/amdgpu/shared/simd/select.h
    rocjitsu/isa/arch/amdgpu/shared/simd_glue.h
    rocjitsu/isa/arch/amdgpu/shared/fp_control/select.h
    rocjitsu/isa/detail/register_set/select.h
    util/detail/convert/select.h
    util/detail/simd/select.h
)
set(_sources mma_entry_points.cpp)
foreach(_header IN LISTS _headers)
    string(MAKE_C_IDENTIFIER "${_header}" _name)
    set(_source "${CMAKE_CURRENT_BINARY_DIR}/${_name}.cpp")
    file(CONFIGURE OUTPUT "${_source}" CONTENT "#include \"${_header}\"\n")
    list(APPEND _sources "${_source}")
endforeach()

add_custom_target(rocjitsu_simd_header_checks)
add_dependencies(rocjitsu_tests rocjitsu_simd_header_checks)

function(rj_add_simd_header_check profile)
    set(_target "rocjitsu_simd_headers_${profile}")
    add_library(${_target} OBJECT ${ARGN})
    target_link_libraries(${_target} PRIVATE util simdojo_headers)
    rj_configure_target(${_target} PRIVATE_INCLUDES GENERATED SCHEMAS WARNINGS)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${_target} PRIVATE -O0 -g0)
    endif()
    add_dependencies(rocjitsu_simd_header_checks ${_target})
endfunction()

if(
    CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$"
    AND CMAKE_CXX_COMPILER_ID MATCHES "^(GNU|Clang)$"
)
    include(CheckCXXCompilerFlag)
    foreach(_profile IN ITEMS x86-64 x86-64-v3 x86-64-v4)
        string(MAKE_C_IDENTIFIER "${_profile}" _key)
        check_cxx_compiler_flag("-march=${_profile}" _RJ_SIMD_HEADER_${_key})
        if(_RJ_SIMD_HEADER_${_key})
            rj_add_simd_header_check(${_key} ${_sources})
            target_compile_options(
                rocjitsu_simd_headers_${_key}
                PRIVATE "-march=${_profile}"
            )
        endif()
    endforeach()
else()
    rj_add_simd_header_check(default ${_sources})
endif()

# Check the typed fallback API explicitly, even when experimental/simd exists.
rj_add_simd_header_check(fallbacks mma_entry_points.cpp)
target_compile_definitions(
    rocjitsu_simd_headers_fallbacks
    PRIVATE RJ_TEST_MMA_FALLBACKS=1
)
