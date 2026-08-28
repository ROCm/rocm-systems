# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

rj_add_object_library(rocjitsu_isa
    decoder.cpp
    execution_backend.cpp
    operand.cpp
    register_set.cpp
    rj_decode.cpp
    target_registry.cpp
)

function(rj_add_split_amdgpu_isa_library arch)
    set(options NO_BUILTIN)
    set(one_value_args GENERATED_ROOT INCLUDE_BASE)
    cmake_parse_arguments(ARG "${options}" "${one_value_args}" "" ${ARGN})

    file(
        RELATIVE_PATH
        _provider_header_dir
        "${ROCJITSU_SRC_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}"
    )

    if(ARG_GENERATED_ROOT)
        set(_generated_root "${ARG_GENERATED_ROOT}")
    else()
        set(_generated_root "${CMAKE_CURRENT_SOURCE_DIR}/../generated")
    endif()
    if(ARG_INCLUDE_BASE)
        set(_provider_header_dir "${ARG_INCLUDE_BASE}/${arch}")
    endif()

    set(_target rocjitsu_isa_${arch})
    set(_generated_dir "${_generated_root}/${arch}")

    file(GLOB _model_sources CONFIGURE_DEPENDS "${_generated_dir}/*.cpp")
    list(FILTER _model_sources EXCLUDE REGEX "_exec.*\\.cpp$")
    list(APPEND _model_sources model_target_provider.cpp)
    list(SORT _model_sources)

    file(
        GLOB _execution_sources
        CONFIGURE_DEPENDS
        "${_generated_dir}/*_exec*.cpp"
    )
    list(APPEND _execution_sources addr_calc.cpp target_provider.cpp)
    list(SORT _execution_sources)

    rj_add_object_library(${_target}_model ${_model_sources})
    rj_add_object_library(${_target}_exec ${_execution_sources})

    # The execution sources are the ones that include fp_mode.h and simd_glue.h,
    # so they are the ones that need the compiler to stop assuming a rounding
    # mode across the ScopedFenv. See RJ_STRICT_FP_ROUNDING_OPTIONS in the
    # top-level CMakeLists.txt. The _model sources never see either header.
    target_compile_options(
        ${_target}_exec
        PRIVATE ${RJ_STRICT_FP_ROUNDING_OPTIONS}
    )

    add_library(${_target} INTERFACE)
    target_sources(
        ${_target}
        INTERFACE
            $<TARGET_OBJECTS:${_target}_model>
            $<TARGET_OBJECTS:${_target}_exec>
    )
    target_link_libraries(${_target} INTERFACE ${_target}_model ${_target}_exec)

    if(ARG_NO_BUILTIN)
        rj_add_isa_target_provider(
            ${_target}
            HEADER ${_provider_header_dir}/target_provider.h
        )
    else()
        rj_add_isa_target_provider(
            ${_target}
            HEADER ${_provider_header_dir}/target_provider.h
            BUILTIN
        )
    endif()
    rj_add_isa_target_provider(
        ${_target}_model
        HEADER ${_provider_header_dir}/model_target_provider.h
    )
    if(NOT ARG_NO_BUILTIN)
        set_property(
            GLOBAL
            APPEND
            PROPERTY RJ_BUILTIN_AMDGPU_ISA_MODEL_PROVIDERS ${_target}_model
        )
    endif()
endfunction()

add_subdirectory(arch/amdgpu/cdna1)
add_subdirectory(arch/amdgpu/cdna2)
add_subdirectory(arch/amdgpu/cdna3)
add_subdirectory(arch/amdgpu/cdna4)
add_subdirectory(arch/amdgpu/rdna1)
add_subdirectory(arch/amdgpu/rdna2)
add_subdirectory(arch/amdgpu/rdna3)
add_subdirectory(arch/amdgpu/rdna3_5)
add_subdirectory(arch/amdgpu/rdna4)
add_subdirectory(arch/amdgpu/cdna5)

# Narrow production composition for the fixed-profile gfx1250 B0-to-A0
# library and other model-only consumers. It supplies the default enum entry
# points for each final image that links it without pulling execution objects.
rj_add_isa_target_registry(
    rocjitsu_cdna5_model_registry
    PROVIDERS rocjitsu_isa_cdna5_model
)

get_property(
    _rj_builtin_amdgpu_isa_model_providers
    GLOBAL
    PROPERTY RJ_BUILTIN_AMDGPU_ISA_MODEL_PROVIDERS
)
rj_add_isa_target_registry(
    rocjitsu_amdgpu_isa_model_registry
    PROVIDERS ${_rj_builtin_amdgpu_isa_model_providers}
)

# Keep an AMDGPU-only composition for DBT tools and hooks that intentionally do
# not link the RISC-V VM. Each final image owns exactly one registry and public
# Decoder::create adapter.
get_property(
    _rj_builtin_amdgpu_isa_providers
    GLOBAL
    PROPERTY RJ_BUILTIN_ISA_PROVIDERS
)
set_property(
    GLOBAL
    PROPERTY RJ_BUILTIN_AMDGPU_ISA_PROVIDERS ${_rj_builtin_amdgpu_isa_providers}
)
rj_add_isa_target_registry(
    rocjitsu_amdgpu_isa_registry
    PROVIDERS ${_rj_builtin_amdgpu_isa_providers}
)

add_subdirectory(arch/risc_v)

# Each built-in subdirectory contributes its provider target to this property.
# There is no second central provider/include/switch list to update. The full
# simulator composition includes AMDGPU and RISC-V; narrow consumers select
# only the providers they require.
get_property(_rj_builtin_isa_providers GLOBAL PROPERTY RJ_BUILTIN_ISA_PROVIDERS)
rj_add_isa_target_registry(
    rocjitsu_isa_registry
    PROVIDERS ${_rj_builtin_isa_providers}
)
set(RJ_BUILTIN_ISA_PROVIDERS ${_rj_builtin_isa_providers} PARENT_SCOPE)
unset(_rj_builtin_isa_providers)
unset(_rj_builtin_amdgpu_isa_providers)
unset(_rj_builtin_amdgpu_isa_model_providers)
