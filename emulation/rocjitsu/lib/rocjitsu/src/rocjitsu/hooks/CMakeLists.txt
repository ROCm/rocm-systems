# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

add_library(rocjitsu_hsa_tool_lifetime OBJECT hsa_tool_lifetime.cpp)
set_target_properties(
    rocjitsu_hsa_tool_lifetime
    PROPERTIES POSITION_INDEPENDENT_CODE ON
)
target_include_directories(
    rocjitsu_hsa_tool_lifetime
    PRIVATE ${ROCJITSU_SRC_DIR}
)
target_link_libraries(rocjitsu_hsa_tool_lifetime PUBLIC ${CMAKE_DL_LIBS})

add_library(
    rocjitsu_hooks
    SHARED
    rj_hsa_dbt_hooks.cpp
    sidecar_registry.cpp
    virtual_lds.cpp
)
add_library(rocjitsu_waitcheck_hooks SHARED rj_hsa_waitcheck_hooks.cpp)

set_target_properties(rocjitsu_hooks PROPERTIES OUTPUT_NAME rocjitsu_hooks)
set_target_properties(
    rocjitsu_waitcheck_hooks
    PROPERTIES OUTPUT_NAME rocjitsu_waitcheck_hooks
)

target_include_directories(
    rocjitsu_hooks
    PRIVATE
        ${ROCJITSU_INCLUDE_DIR}
        ${ROCJITSU_SRC_DIR}
        ${GENERATED_DIR}
        ${RJ_VERSION_INCLUDE_DIR}
)
target_include_directories(rocjitsu_hooks SYSTEM PRIVATE ${HSA_INCLUDE_DIR})

target_include_directories(
    rocjitsu_waitcheck_hooks
    PRIVATE ${ROCJITSU_INCLUDE_DIR} ${ROCJITSU_SRC_DIR}
)
target_include_directories(
    rocjitsu_waitcheck_hooks
    SYSTEM
    PRIVATE ${HSA_INCLUDE_DIR}
)

target_link_libraries(
    rocjitsu_hooks
    PRIVATE
        rocjitsu_analysis
        rocjitsu_code
        rocjitsu_dbt_config
        rocjitsu_isa
        rocjitsu_isa_registry
        rocjitsu_isa_cdna1
        rocjitsu_isa_cdna2
        rocjitsu_isa_cdna3
        rocjitsu_isa_cdna4
        rocjitsu_isa_rdna1
        rocjitsu_isa_rdna2
        rocjitsu_isa_rdna3
        rocjitsu_isa_rdna3_5
        rocjitsu_isa_rdna4
        rocjitsu_isa_gfx1250
        # \NPI new ISA family: link its rocjitsu_isa_<isa> object library here.
        rocjitsu_vm_amdgpu
        util
        flatbuffers
        simdojo
        rocjitsu_hsa_tool_lifetime
        Threads::Threads
        ${CMAKE_DL_LIBS}
)

target_link_libraries(
    rocjitsu_waitcheck_hooks
    PRIVATE
        rocjitsu_waitcheck_objects
        rocjitsu_hsa_tool_lifetime
        util
        Threads::Threads
        ZLIB::ZLIB
        zstd::libzstd
)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_options(
        rocjitsu_waitcheck_hooks
        PRIVATE "LINKER:--gc-sections,--exclude-libs,ALL,--no-undefined"
    )
endif()

if(MSVC)
    target_compile_options(rocjitsu_hsa_tool_lifetime PRIVATE /W4 /WX)
    target_compile_options(rocjitsu_hooks PRIVATE /W4 /WX)
    target_compile_options(rocjitsu_waitcheck_hooks PRIVATE /W4 /WX)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(
        rocjitsu_hsa_tool_lifetime
        PRIVATE -Wall -Wextra -Wpedantic -Werror
    )
    target_compile_options(
        rocjitsu_hooks
        PRIVATE -Wall -Wextra -Wpedantic -Werror
    )
    target_compile_options(
        rocjitsu_waitcheck_hooks
        PRIVATE -Wall -Wextra -Wpedantic -Werror
    )
endif()

add_subdirectory(consan)
