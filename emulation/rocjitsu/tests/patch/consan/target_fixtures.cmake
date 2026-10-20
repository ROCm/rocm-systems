# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

set(_target_fixtures)
foreach(
    _gpu
    IN
    ITEMS gfx942 gfx950 gfx1100 gfx1201 gfx1250
)
    foreach(_wave IN ITEMS 32 64)
        set(_legacy 0)
        set(_gfx12 0)
        if(_gpu STREQUAL "gfx942" OR _gpu STREQUAL "gfx950")
            if(_wave EQUAL 32)
                continue()
            endif()
            set(_legacy 1)
        elseif(_gpu STREQUAL "gfx1250" AND _wave EQUAL 64)
            continue()
        endif()
        if(_gpu STREQUAL "gfx1201" OR _gpu STREQUAL "gfx1250")
            set(_gfx12 1)
        endif()
        set(_directory ${_fixture_dir}/targets/${_gpu}/wave${_wave})
        file(MAKE_DIRECTORY ${_directory})
        set(_wave_options)
        if(NOT _legacy)
            if(_wave EQUAL 64)
                set(_wave_options -mwavefrontsize64)
            else()
                set(_wave_options -mno-wavefrontsize64)
            endif()
        endif()
        foreach(
            _case
            IN
            ITEMS 1 2 3 4 6 11 12 13 14
        )
            set(_object ${_directory}/native-${_case}.o)
            set(_image ${_directory}/native-${_case}.hsaco)
            add_custom_command(
                OUTPUT ${_object}
                COMMAND
                    ${CMAKE_CXX_COMPILER_LAUNCHER} ${AMDCXX} -x
                    assembler-with-cpp -P -target amdgcn-amd-amdhsa
                    -mcpu=${_gpu} -mcode-object-version=5 ${_wave_options}
                    -DCONSAN_TARGET="amdgcn-amd-amdhsa--${_gpu}"
                    -DCONSAN_WAVE_SIZE=${_wave} -DCONSAN_LEGACY=${_legacy}
                    -DCONSAN_GFX12=${_gfx12} -DCONSAN_CASE=${_case}
                    -DCONSAN_GROUP_BYTES=128 -c
                    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/native_lds.s -o
                    ${_object}
                DEPENDS fixtures/native_lds.s ${AMDCXX}
                VERBATIM
            )
            add_custom_command(
                OUTPUT ${_image}
                COMMAND
                    ${ROCM_PATH}/lib/llvm/bin/ld.lld -shared ${_object} -o
                    ${_image}
                DEPENDS ${_object}
                VERBATIM
            )
            list(APPEND _target_fixtures ${_image})
        endforeach()
        foreach(_case IN ITEMS 1 2 3)
            set(_bundle ${_directory}/hip-${_case}.bundle)
            set(_image ${_directory}/hip-${_case}.hsaco)
            add_custom_command(
                OUTPUT ${_bundle}
                COMMAND
                    ${CMAKE_CXX_COMPILER_LAUNCHER} ${AMDCXX} -x hip
                    --offload-arch=${_gpu} --rocm-path=${ROCM_PATH}
                    --cuda-device-only -O2 -mcode-object-version=5 -mllvm
                    -amdgpu-enable-vopd=0 ${_wave_options}
                    -DCONSAN_CASE=${_case}
                    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/native_lds.hip -o
                    ${_bundle}
                DEPENDS fixtures/native_lds.hip ${AMDCXX}
                VERBATIM
            )
            add_custom_command(
                OUTPUT ${_image}
                COMMAND
                    ${ROCM_PATH}/lib/llvm/bin/clang-offload-bundler --type=o
                    --unbundle --targets=hipv4-amdgcn-amd-amdhsa--${_gpu}
                    --input=${_bundle} --output=${_image}
                DEPENDS ${_bundle}
                VERBATIM
            )
            list(APPEND _target_fixtures ${_image})
        endforeach()
    endforeach()
endforeach()
add_custom_target(consan_target_fixtures DEPENDS ${_target_fixtures})
