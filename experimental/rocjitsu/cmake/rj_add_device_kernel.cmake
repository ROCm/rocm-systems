# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Compile a .hip source into a host+device ELF object (.o).
#
# The -c flag produces a relocatable object with a .hip_fatbin section
# that contains the Clang offload bundle with the device code object for
# the specified GPU target. AMDCLANG, ROCM_PATH, and KERNEL_OUTPUT_DIR
# must be set before including this module.
#
# Usage: rj_add_device_kernel(<name> <offload_arch>)
#   name         - base name of the .hip source (without extension)
#   offload_arch - GPU target (e.g. gfx942, gfx950)
function(rj_add_device_kernel name offload_arch)
    set(src ${CMAKE_CURRENT_SOURCE_DIR}/${name}.hip)
    set(out ${KERNEL_OUTPUT_DIR}/${name}.o)

    add_custom_command(
        OUTPUT ${out}
        COMMAND ${AMDCLANG} -x hip
                --offload-arch=${offload_arch}
                --rocm-path=${ROCM_PATH}
                -fPIC -c -O2
                -o ${out} ${src}
        DEPENDS ${src}
        COMMENT "Compiling device kernel: ${name} (${offload_arch})"
    )

    # Per-kernel target so dependents can reference it.
    add_custom_target(kernel_${name} DEPENDS ${out})
endfunction()

# Assemble a checked-in AMDGPU assembly fixture into a standalone code object.
#
# Usage: rj_add_amdgpu_asm_kernel(<name> <offload_arch>)
#   name         - base name of the .s source (without extension)
#   offload_arch - GPU target (e.g. gfx950)
function(rj_add_amdgpu_asm_kernel name offload_arch)
    set(src ${CMAKE_CURRENT_SOURCE_DIR}/${name}.s)
    set(obj ${KERNEL_OUTPUT_DIR}/${name}.asm.o)
    set(out ${KERNEL_OUTPUT_DIR}/${name}.hsaco)

    add_custom_command(
        OUTPUT ${obj}
        COMMAND ${AMDCLANG}
                -target amdgcn-amd-amdhsa
                -mcpu=${offload_arch}
                -x assembler
                -c
                -o ${obj} ${src}
        DEPENDS ${src}
        COMMENT "Assembling AMDGPU kernel: ${name} (${offload_arch})"
    )

    add_custom_command(
        OUTPUT ${out}
        COMMAND ${AMDCLANG}
                -target amdgcn-amd-amdhsa
                -mcpu=${offload_arch}
                -nostdlib
                -Wl,--no-undefined
                -o ${out} ${obj}
        DEPENDS ${obj}
        COMMENT "Linking AMDGPU code object: ${name} (${offload_arch})"
    )

    add_custom_target(kernel_${name} DEPENDS ${out})
endfunction()
