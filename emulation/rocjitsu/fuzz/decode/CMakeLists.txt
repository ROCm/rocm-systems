# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

add_library(rj_decode_fuzz_core STATIC decode_fuzz_core.cpp)
target_include_directories(
    rj_decode_fuzz_core
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(
    rj_decode_fuzz_core
    PUBLIC rocjitsu_amdgpu_isa_model_registry rocjitsu_isa util
)
rj_configure_target(rj_decode_fuzz_core PRIVATE_INCLUDES WARNINGS)

add_executable(rj_decode_fuzz decode_fuzz_main.cpp)
target_link_libraries(rj_decode_fuzz PRIVATE rj_decode_fuzz_core)
rj_configure_target(rj_decode_fuzz PRIVATE_INCLUDES WARNINGS)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_test(
        NAME DecodeFuzzModelOnly.SymbolBoundary
        COMMAND
            ${CMAKE_COMMAND} -DNM=${CMAKE_NM}
            -DMODEL_BINARY=$<TARGET_FILE:rj_decode_fuzz>
            -DREQUIRED_MODEL_ARCHES=cdna1,cdna2,cdna3,cdna4,cdna5,rdna1,rdna2,rdna3,rdna3_5,rdna4
            -P
            ${CMAKE_CURRENT_SOURCE_DIR}/../../tests/check_model_only_symbols.cmake
    )
endif()
