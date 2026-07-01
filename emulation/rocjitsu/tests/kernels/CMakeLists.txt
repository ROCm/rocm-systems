# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Compile HIP device kernels into host+device ELF objects for simulator testing.
#
# The resulting .o files are standard ELF objects containing a .hip_fatbin
# section with the bundled AMDGPU device code. The code object loader
# (Executable) extracts and parses the device ELF from this section.
#
# Requires an ROCm installation with amdclang++. When the compiler is not
# found the kernel targets are silently skipped and tests that depend on them
# are disabled at runtime via the HAS_DEVICE_KERNELS compile definition.

# Prefer the compiler from the ROCm tree selected by ROCM_PATH. Some systems
# also expose /usr/bin/amdclang++ through alternatives; deriving --rocm-path
# from that wrapper gives /usr, which mixes headers and device libraries from
# different installs.
find_program(
    AMDCXX
    amdclang++
    PATHS ${ROCM_PATH}
    PATH_SUFFIXES lib/llvm/bin bin
    DOC "Path to the ROCm amdclang++ compiler"
    NO_DEFAULT_PATH
)

if(NOT AMDCXX)
    message(
        STATUS
        "amdclang++ not found - device kernel tests will be disabled"
    )
    return()
endif()

if(DEFINED SKIP_DEVICE_KERNELS AND SKIP_DEVICE_KERNELS)
    message(
        STATUS
        "SKIP_DEVICE_KERNELS set - device kernel tests will be disabled"
    )
    return()
endif()

message(STATUS "Using device compiler: ${AMDCXX}")

set(KERNEL_OUTPUT_DIR ${CMAKE_BINARY_DIR}/kernels)
file(MAKE_DIRECTORY ${KERNEL_OUTPUT_DIR})

include(rj_add_device_kernel)

rj_add_device_kernel(matmul_naive gfx950)
rj_add_device_kernel(matmul_mfma gfx950)
rj_add_device_kernel(matmul_mfma_16x16 gfx950)
rj_add_device_kernel(matmul_tiled gfx950)
rj_add_device_kernel(vector_add gfx950)
rj_add_device_kernel(dynamic_copy_loop gfx950)

# Triton-generated assembly fixtures. The checked-in assembly records the
# original Triton-generated kernels, so normal test builds only need the ROCm
# assembler/linker and do not depend on a Python/Triton install.
function(rj_add_triton_asm_fixture name target)
    set(asm_obj ${KERNEL_OUTPUT_DIR}/${name}.asm.o)
    set(hsaco ${KERNEL_OUTPUT_DIR}/${name}.hsaco)
    add_custom_command(
        OUTPUT ${asm_obj}
        COMMAND
            ${AMDCXX} -x assembler -target amdgcn-amd-amdhsa -mcpu=${target}
            --rocm-path=${ROCM_PATH} -c -o ${asm_obj}
            ${CMAKE_CURRENT_SOURCE_DIR}/${name}.s
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${name}.s
        COMMENT "Assembling Triton fixture: ${name} (${target})"
    )
    add_custom_command(
        OUTPUT ${hsaco}
        COMMAND
            ${AMDCXX} -target amdgcn-amd-amdhsa -mcpu=${target}
            --rocm-path=${ROCM_PATH} -nostdlib -shared -o ${hsaco} ${asm_obj}
        DEPENDS ${asm_obj}
        COMMENT "Linking Triton fixture: ${name} (${target})"
    )
    add_custom_target(kernel_${name} DEPENDS ${hsaco})
endfunction()

rj_add_triton_asm_fixture(triton_cdna4_matmul_dynamic_32x32x64 gfx950)
rj_add_triton_asm_fixture(triton_cdna4_matmul_buffer_async_1024 gfx950)
rj_add_triton_asm_fixture(triton_cdna4_flash_attention_no_async_1024 gfx950)
rj_add_triton_asm_fixture(triton_cdna4_flash_attention_buffer_async_1024 gfx950)
rj_add_triton_asm_fixture(triton_cdna3_matmul_dynamic_32x32x64 gfx942)
rj_add_triton_asm_fixture(triton_cdna3_matmul_buffer_async_1024 gfx942)
rj_add_triton_asm_fixture(triton_cdna3_flash_attention_no_async_1024 gfx942)
rj_add_triton_asm_fixture(triton_cdna3_flash_attention_buffer_async_1024 gfx942)

# Second build of vector_add for gfx90a, used by the DBI hardware-gated
# smoke test (tests/dbi/hsa_dbi_smoke_test.cpp). Outputs vector_add_gfx90a.o
# so it coexists with the gfx950 build above. (A single multi-target fat
# binary would be cleaner but the Executable loader has a pre-existing
# single-bundle assumption — see executable.cpp::load_hip_fatbin.)
rj_add_device_kernel(vector_add gfx90a OUTPUT_NAME vector_add_gfx90a)

if(RJ_INSTALL_TESTS)
    set(RJ_DEVICE_KERNEL_FIXTURES
        ${KERNEL_OUTPUT_DIR}/matmul_naive.o
        ${KERNEL_OUTPUT_DIR}/matmul_mfma.o
        ${KERNEL_OUTPUT_DIR}/matmul_mfma_16x16.o
        ${KERNEL_OUTPUT_DIR}/matmul_tiled.o
        ${KERNEL_OUTPUT_DIR}/vector_add.o
        ${KERNEL_OUTPUT_DIR}/vector_add_gfx90a.o
        ${KERNEL_OUTPUT_DIR}/dynamic_copy_loop.o
        ${KERNEL_OUTPUT_DIR}/triton_cdna4_matmul_dynamic_32x32x64.hsaco
        ${KERNEL_OUTPUT_DIR}/triton_cdna4_matmul_buffer_async_1024.hsaco
        ${KERNEL_OUTPUT_DIR}/triton_cdna4_flash_attention_no_async_1024.hsaco
        ${KERNEL_OUTPUT_DIR}/triton_cdna4_flash_attention_buffer_async_1024.hsaco
        ${KERNEL_OUTPUT_DIR}/triton_cdna3_matmul_dynamic_32x32x64.hsaco
        ${KERNEL_OUTPUT_DIR}/triton_cdna3_matmul_buffer_async_1024.hsaco
        ${KERNEL_OUTPUT_DIR}/triton_cdna3_flash_attention_no_async_1024.hsaco
        ${KERNEL_OUTPUT_DIR}/triton_cdna3_flash_attention_buffer_async_1024.hsaco
    )
    install(
        FILES ${RJ_DEVICE_KERNEL_FIXTURES}
        DESTINATION ${CMAKE_INSTALL_DATADIR}/rocjitsu/tests/kernels
    )
endif()

# Umbrella target for all device kernels.
add_custom_target(
    device_kernels
    DEPENDS
        kernel_matmul_naive
        kernel_matmul_mfma
        kernel_matmul_mfma_16x16
        kernel_matmul_tiled
        kernel_vector_add
        kernel_vector_add_gfx90a
        kernel_dynamic_copy_loop
        kernel_triton_cdna4_matmul_dynamic_32x32x64
        kernel_triton_cdna4_matmul_buffer_async_1024
        kernel_triton_cdna4_flash_attention_no_async_1024
        kernel_triton_cdna4_flash_attention_buffer_async_1024
        kernel_triton_cdna3_matmul_dynamic_32x32x64
        kernel_triton_cdna3_matmul_buffer_async_1024
        kernel_triton_cdna3_flash_attention_no_async_1024
        kernel_triton_cdna3_flash_attention_buffer_async_1024
)

# Export variables to the parent scope so the test executable can use them.
set(HAS_DEVICE_KERNELS TRUE PARENT_SCOPE)
set(KERNEL_OUTPUT_DIR ${KERNEL_OUTPUT_DIR} PARENT_SCOPE)
