# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

add_library(rj_decode_fuzz_core STATIC decode_fuzz_core.cpp)
get_property(
    _rj_decode_fuzz_providers
    GLOBAL
    PROPERTY RJ_BUILTIN_AMDGPU_ISA_PROVIDERS
)
list(REMOVE_ITEM _rj_decode_fuzz_providers rocjitsu_isa_gfx1250)
list(APPEND _rj_decode_fuzz_providers rocjitsu_isa_gfx1250_model)
rj_add_isa_target_registry(
    rj_decode_fuzz_registry
    PROVIDERS ${_rj_decode_fuzz_providers}
)
target_include_directories(
    rj_decode_fuzz_core
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(
    rj_decode_fuzz_core
    PUBLIC
        rocjitsu_dbt_internal
        rj_decode_fuzz_registry
        rocjitsu_vm_amdgpu
        simdojo
        simdojo_headers
)
rj_configure_target(rj_decode_fuzz_core PRIVATE_INCLUDES WARNINGS)

add_executable(rj_decode_fuzz decode_fuzz_main.cpp)
target_link_libraries(rj_decode_fuzz PRIVATE rj_decode_fuzz_core)
rj_configure_target(rj_decode_fuzz PRIVATE_INCLUDES WARNINGS)
