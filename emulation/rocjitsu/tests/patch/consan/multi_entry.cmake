# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

set(_multi_object ${_fixture_dir}/multi-entry.o)
set(_multi_image ${_fixture_dir}/multi-entry.hsaco)
add_custom_command(
    OUTPUT ${_multi_object}
    COMMAND
        ${CMAKE_CXX_COMPILER_LAUNCHER} ${AMDCXX} -x assembler-with-cpp -target
        amdgcn-amd-amdhsa -mcpu=gfx1250 -mcode-object-version=5
        -mno-wavefrontsize64 -c
        ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/multi_entry.s -o ${_multi_object}
    DEPENDS fixtures/multi_entry.s ${AMDCXX}
    VERBATIM
)
add_custom_command(
    OUTPUT ${_multi_image}
    COMMAND
        ${ROCM_PATH}/lib/llvm/bin/ld.lld -shared ${_multi_object} -o
        ${_multi_image}
    DEPENDS ${_multi_object}
    VERBATIM
)
add_custom_target(consan_multi_entry_fixture DEPENDS ${_multi_image})

# Share kernel bodies across synchronization and per-entry ABI fixtures.
foreach(
    _variant
    IN
    ITEMS
        barrier
        loop
        mixed
        native-global
        unreachable-pc
        native-wide-vgprs
        alignment
        trace-exits
        unsupported
)
    set(_variant_object ${_fixture_dir}/multi-entry-${_variant}.o)
    set(_variant_image ${_fixture_dir}/multi-entry-${_variant}.hsaco)
    set(_variant_defines -DMK_BARRIER=1)
    if(_variant STREQUAL "loop")
        list(APPEND _variant_defines -DMK_LOOP=1)
    elseif(_variant STREQUAL "mixed")
        list(APPEND _variant_defines -DMK_MIXED=1)
    elseif(_variant STREQUAL "native-global")
        set(_variant_defines -DMK_NATIVE_GLOBAL=1)
    elseif(_variant STREQUAL "unreachable-pc")
        set(_variant_defines -DMK_NATIVE_GLOBAL=1 -DMK_UNREACHABLE_PC_READ=1)
    elseif(_variant STREQUAL "native-wide-vgprs")
        set(_variant_defines -DMK_NATIVE_VGPRS=384)
    elseif(_variant STREQUAL "alignment")
        set(_variant_defines -DMK_EXTENDED_ABI=1 -DMK_A_KERNARG_ALIGNMENT=32)
    elseif(_variant STREQUAL "trace-exits")
        set(_variant_defines -DMK_TRACE_EXIT_STRESS=1)
    elseif(_variant STREQUAL "unsupported")
        set(_variant_defines -DMK_UNSUPPORTED_C=1)
    endif()
    add_custom_command(
        OUTPUT ${_variant_object}
        COMMAND
            ${CMAKE_CXX_COMPILER_LAUNCHER} ${AMDCXX} -x assembler-with-cpp
            -target amdgcn-amd-amdhsa -mcpu=gfx1250 -mcode-object-version=5
            -mno-wavefrontsize64 ${_variant_defines} -c
            ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/multi_entry.s -o
            ${_variant_object}
        DEPENDS fixtures/multi_entry.s ${AMDCXX}
        VERBATIM
    )
    add_custom_command(
        OUTPUT ${_variant_image}
        COMMAND
            ${ROCM_PATH}/lib/llvm/bin/ld.lld -shared ${_variant_object} -o
            ${_variant_image}
        DEPENDS ${_variant_object}
        VERBATIM
    )
    add_custom_target(
        consan_multi_entry_${_variant}_fixture
        DEPENDS ${_variant_image}
    )
    add_dependencies(
        consan_multi_entry_fixture
        consan_multi_entry_${_variant}_fixture
    )
endforeach()

set(_multi_abi_object ${_fixture_dir}/multi-entry-abi.o)
set(_multi_abi_image ${_fixture_dir}/multi-entry-abi.hsaco)
add_custom_command(
    OUTPUT ${_multi_abi_object}
    COMMAND
        ${CMAKE_CXX_COMPILER_LAUNCHER} ${AMDCXX} -x assembler-with-cpp -target
        amdgcn-amd-amdhsa -mcpu=gfx1250 -mcode-object-version=5
        -mno-wavefrontsize64 -DMK_EXTENDED_ABI=1 -c
        ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/multi_entry.s -o
        ${_multi_abi_object}
    DEPENDS fixtures/multi_entry.s ${AMDCXX}
    VERBATIM
)
add_custom_command(
    OUTPUT ${_multi_abi_image}
    COMMAND
        ${ROCM_PATH}/lib/llvm/bin/ld.lld -shared ${_multi_abi_object} -o
        ${_multi_abi_image}
    DEPENDS ${_multi_abi_object}
    VERBATIM
)
add_custom_target(consan_multi_entry_abi_fixture DEPENDS ${_multi_abi_image})
add_dependencies(consan_multi_entry_fixture consan_multi_entry_abi_fixture)

set(_multi_scalar_object ${_fixture_dir}/multi-entry-native-scalar.o)
set(_multi_scalar_image ${_fixture_dir}/multi-entry-native-scalar.hsaco)
add_custom_command(
    OUTPUT ${_multi_scalar_object}
    COMMAND
        ${CMAKE_CXX_COMPILER_LAUNCHER} ${AMDCXX} -x assembler-with-cpp -target
        amdgcn-amd-amdhsa -mcpu=gfx1250 -mcode-object-version=5
        -mno-wavefrontsize64 -DMK_EXTENDED_ABI=1 -DMK_NATIVE_SCALAR=1 -c
        ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/multi_entry.s -o
        ${_multi_scalar_object}
    DEPENDS fixtures/multi_entry.s ${AMDCXX}
    VERBATIM
)
add_custom_command(
    OUTPUT ${_multi_scalar_image}
    COMMAND
        ${ROCM_PATH}/lib/llvm/bin/ld.lld -shared ${_multi_scalar_object} -o
        ${_multi_scalar_image}
    DEPENDS ${_multi_scalar_object}
    VERBATIM
)
add_custom_target(
    consan_multi_entry_native_scalar_fixture
    DEPENDS ${_multi_scalar_image}
)
add_dependencies(
    consan_multi_entry_fixture
    consan_multi_entry_native_scalar_fixture
)
