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

find_program(AMDCLANG amdclang++
    HINTS /opt/rocm/bin ENV ROCM_PATH
    DOC "Path to the ROCm amdclang++ compiler"
)

if(NOT AMDCLANG)
    message(STATUS "amdclang++ not found - device kernel tests will be disabled")
    return()
endif()

message(STATUS "Found amdclang++: ${AMDCLANG}")

set(KERNEL_OUTPUT_DIR ${CMAKE_BINARY_DIR}/kernels)
file(MAKE_DIRECTORY ${KERNEL_OUTPUT_DIR})

# Locate the ROCm installation root for --rocm-path.
get_filename_component(ROCM_PATH "${AMDCLANG}" DIRECTORY)
get_filename_component(ROCM_PATH "${ROCM_PATH}" DIRECTORY)

include(rj_add_device_kernel)

rj_add_device_kernel(matmul_naive gfx950)
rj_add_device_kernel(matmul_mfma gfx950)
rj_add_device_kernel(matmul_mfma_16x16 gfx950)
rj_add_device_kernel(matmul_tiled gfx950)
rj_add_device_kernel(vector_add gfx950)
rj_add_device_kernel(dynamic_copy_loop gfx950)

# Triton-generated CDNA4 matmul fixtures. Each checked-in .s file records the
# Triton source used to generate it, so normal test builds only need the ROCm
# assembler/linker.
function(rj_add_triton_cdna4_asm_fixture name)
    set(asm_obj ${KERNEL_OUTPUT_DIR}/${name}.asm.o)
    set(hsaco ${KERNEL_OUTPUT_DIR}/${name}.hsaco)
    add_custom_command(
        OUTPUT ${asm_obj}
        COMMAND ${AMDCLANG} -x assembler
                -target amdgcn-amd-amdhsa
                -mcpu=gfx950
                -c
                -o ${asm_obj}
                ${CMAKE_CURRENT_SOURCE_DIR}/${name}.s
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${name}.s
        COMMENT "Assembling Triton CDNA4 fixture: ${name} (gfx950)"
    )
    add_custom_command(
        OUTPUT ${hsaco}
        COMMAND ${AMDCLANG}
                -target amdgcn-amd-amdhsa
                -mcpu=gfx950
                -nostdlib
                -shared
                -o ${hsaco}
                ${asm_obj}
        DEPENDS ${asm_obj}
        COMMENT "Linking Triton CDNA4 fixture: ${name} (gfx950)"
    )
    add_custom_target(kernel_${name} DEPENDS ${hsaco})
endfunction()

set(TRITON_CDNA4_MATMUL_FIXTURES
    triton_cdna4_matmul_static_32x32x64
    triton_cdna4_matmul_static_512x512x512
    triton_cdna4_matmul_static_31x31x64
    triton_cdna4_matmul_static_32x32x63
    triton_cdna4_matmul_dynamic_32x32x64
    triton_cdna4_matmul_async_copy_64x64x64
)

foreach(fixture IN LISTS TRITON_CDNA4_MATMUL_FIXTURES)
    rj_add_triton_cdna4_asm_fixture(${fixture})
    list(APPEND TRITON_CDNA4_MATMUL_TARGETS kernel_${fixture})
endforeach()

# Umbrella target for all device kernels.
add_custom_target(device_kernels DEPENDS
    kernel_matmul_naive
    kernel_matmul_mfma
    kernel_matmul_mfma_16x16
    kernel_matmul_tiled
    kernel_vector_add
    kernel_dynamic_copy_loop
    ${TRITON_CDNA4_MATMUL_TARGETS}
)

# Export variables to the parent scope so the test executable can use them.
set(HAS_DEVICE_KERNELS TRUE PARENT_SCOPE)
set(KERNEL_OUTPUT_DIR ${KERNEL_OUTPUT_DIR} PARENT_SCOPE)
