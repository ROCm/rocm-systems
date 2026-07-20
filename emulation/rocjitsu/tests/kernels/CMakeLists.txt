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

include(rj_find_amdcxx)
rj_find_amdcxx(AMDCXX)

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
rj_add_device_kernel(multikernel_indirect_branch gfx950)
rj_add_device_kernel(multikernel_indirect_branch_part0 gfx950)
rj_add_device_kernel(multikernel_indirect_branch_part1 gfx950)

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
# smoke test (tests/dbi/hsa_dbi_nop_asm_test.cpp). Outputs vector_add_gfx90a.o
# so it coexists with the gfx950 build above. (A single multi-target fat
# binary would be cleaner but the Executable loader has a pre-existing
# single-bundle assumption — see executable.cpp::load_hip_fatbin.)
rj_add_device_kernel(vector_add gfx90a OUTPUT_NAME vector_add_gfx90a)

# Register-padded vector_add for the DBI probe-CALL smoke test
# (tests/dbi/hsa_dbi_nop_probe_test.cpp). Forces .sgpr_count >= 32 so the
# probe's link pair s[30:31] is granted; see vector_add_probe.hip.
rj_add_device_kernel(vector_add_probe gfx90a OUTPUT_NAME vector_add_probe_gfx90a)

if(RJ_INSTALL_TESTS)
    set(RJ_DEVICE_KERNEL_FIXTURES
        ${KERNEL_OUTPUT_DIR}/matmul_naive.o
        ${KERNEL_OUTPUT_DIR}/matmul_mfma.o
        ${KERNEL_OUTPUT_DIR}/matmul_mfma_16x16.o
        ${KERNEL_OUTPUT_DIR}/matmul_tiled.o
        ${KERNEL_OUTPUT_DIR}/vector_add.o
        ${KERNEL_OUTPUT_DIR}/vector_add_gfx90a.o
        ${KERNEL_OUTPUT_DIR}/vector_add_probe_gfx90a.o
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

# DBI no-op probe fixture. A lone __device__ function needs the device-only +
# unbundle path (rj_add_probe_object in cmake/rj_add_device_kernel.cmake), which
# additionally requires clang-offload-bundler from the same LLVM tree.
find_program(
    CLANG_OFFLOAD_BUNDLER
    clang-offload-bundler
    HINTS ${ROCM_PATH}/lib/llvm/bin ${ROCM_PATH}/llvm/bin ${ROCM_PATH}/bin
    ENV ROCM_PATH
    PATH_SUFFIXES bin
    DOC "Path to clang-offload-bundler for device-only probe objects"
)
set(HAS_PROBE_FIXTURES FALSE)
if(CLANG_OFFLOAD_BUNDLER)
    rj_add_probe_object(rj_nop_probe gfx90a OUTPUT_NAME rj_nop_probe_gfx90a)
    set(HAS_PROBE_FIXTURES TRUE)
    # Install the probe object alongside the other kernel fixtures so installed
    # runs of probe_fixture_test / hsa_dbi_nop_probe_test can find it via
    # ROCJITSU_KERNEL_DIR. Gated here (not in the list above) because it exists
    # only when the offload bundler is available.
    if(RJ_INSTALL_TESTS)
        install(
            FILES ${KERNEL_OUTPUT_DIR}/rj_nop_probe_gfx90a.hsaco
            DESTINATION ${CMAKE_INSTALL_DATADIR}/rocjitsu/tests/kernels
        )
    endif()
else()
    message(
        STATUS
        "clang-offload-bundler not found - DBI probe object fixtures disabled"
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
        kernel_vector_add_probe_gfx90a
        kernel_dynamic_copy_loop
        kernel_multikernel_indirect_branch
        kernel_multikernel_indirect_branch_part0
        kernel_multikernel_indirect_branch_part1
        kernel_triton_cdna4_matmul_dynamic_32x32x64
        kernel_triton_cdna4_matmul_buffer_async_1024
        kernel_triton_cdna4_flash_attention_no_async_1024
        kernel_triton_cdna4_flash_attention_buffer_async_1024
        kernel_triton_cdna3_matmul_dynamic_32x32x64
        kernel_triton_cdna3_matmul_buffer_async_1024
        kernel_triton_cdna3_flash_attention_no_async_1024
        kernel_triton_cdna3_flash_attention_buffer_async_1024
)

if(HAS_PROBE_FIXTURES)
    add_dependencies(device_kernels probe_rj_nop_probe_gfx90a)
endif()

# Export variables to the parent scope so the test executable can use them.
set(HAS_DEVICE_KERNELS TRUE PARENT_SCOPE)
set(HAS_PROBE_FIXTURES ${HAS_PROBE_FIXTURES} PARENT_SCOPE)
set(KERNEL_OUTPUT_DIR ${KERNEL_OUTPUT_DIR} PARENT_SCOPE)
