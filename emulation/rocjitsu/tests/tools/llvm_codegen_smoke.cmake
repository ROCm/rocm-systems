# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Opt in to testing a specific external LLVM build, independently of the SDK
# compiler used to build the rest of RocJITsu's fixtures.
set(RJ_LLVM_SMOKE_TOOLS_DIR
    ""
    CACHE PATH
    "LLVM bin directory for compiler compatibility smoke tests"
)
if(NOT RJ_LLVM_SMOKE_TOOLS_DIR)
    return()
endif()
if(NOT HIP_RUNTIME64)
    message(FATAL_ERROR "LLVM compiler smoke tests require the HIP runtime")
endif()
foreach(
    tool
    llc
    llvm-mc
    ld.lld
    llvm-readobj
    llvm-objdump
)
    if(NOT EXISTS "${RJ_LLVM_SMOKE_TOOLS_DIR}/${tool}")
        message(
            FATAL_ERROR
            "LLVM compiler smoke tests require ${RJ_LLVM_SMOKE_TOOLS_DIR}/${tool}"
        )
    endif()
endforeach()

add_executable(llvm_codegen_smoke tools/llvm_codegen_smoke.cpp)
target_include_directories(llvm_codegen_smoke PRIVATE "${ROCM_PATH}/include")
target_compile_definitions(llvm_codegen_smoke PRIVATE __HIP_PLATFORM_AMD__=1)
target_link_libraries(llvm_codegen_smoke PRIVATE ${HIP_RUNTIME64})
rj_configure_target(llvm_codegen_smoke TEST)
add_dependencies(llvm_codegen_smoke rocjitsu_bin rocjitsu_shared)

find_library(
    LLVM_SMOKE_HSA_RUNTIME
    hsa-runtime64
    PATHS "${ROCM_PATH}/lib"
    NO_DEFAULT_PATH
    REQUIRED
)
add_executable(llvm_codegen_hsa_smoke tools/llvm_codegen_hsa_smoke.cpp)
target_include_directories(
    llvm_codegen_hsa_smoke
    SYSTEM
    PRIVATE
        "${ROCM_PATH}/include"
        "${CMAKE_SOURCE_DIR}/lib/rocjitsu/external_headers/hsa_headers"
)
target_link_libraries(llvm_codegen_hsa_smoke PRIVATE ${LLVM_SMOKE_HSA_RUNTIME})
rj_configure_target(llvm_codegen_hsa_smoke TEST)
add_dependencies(llvm_codegen_hsa_smoke rocjitsu_bin rocjitsu_shared)

foreach(
    target
    gfx90a
    gfx950
    gfx1250
    gfx1250-strict
    gfx12-5-generic
)
    if(target STREQUAL "gfx90a")
        set(config gfx90a_mi210_kmd.json)
    elseif(target STREQUAL "gfx950")
        set(config gfx950_mi355x_kmd.json)
    else()
        set(config gfx1250_mi455x.json)
    endif()
    foreach(backend hip hsa)
        if(backend STREQUAL "hip")
            set(helper llvm_codegen_smoke)
            set(test_prefix LlvmCodegenSmoke)
        else()
            set(helper llvm_codegen_hsa_smoke)
            set(test_prefix LlvmCodegenHsaSmoke)
        endif()
        add_test(
            NAME "${test_prefix}.${target}"
            COMMAND
                ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/tools/llvm_codegen_smoke.py"
                --llvm-bin "${RJ_LLVM_SMOKE_TOOLS_DIR}" --helper
                $<TARGET_FILE:${helper}> --rocjitsu $<TARGET_FILE:rocjitsu_bin>
                --config "${CMAKE_SOURCE_DIR}/configs/${config}" --target
                "${target}" --output
                "${CMAKE_CURRENT_BINARY_DIR}/${helper}_objects/${target}"
        )
        set_tests_properties(
            "${test_prefix}.${target}"
            PROPERTIES TIMEOUT 600 LABELS "llvm-smoke"
        )
    endforeach()
endforeach()
