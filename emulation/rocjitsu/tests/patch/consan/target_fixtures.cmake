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
            ITEMS
                1
                2
                3
                4
                6
                11
                12
                13
                14
                unknown-environment
                distant-entry
                dense-sites
                ${_names}
        )
            set(_case_number ${_case})
            set(_layout_options)
            set(_metadata_target "amdgcn-amd-amdhsa--${_gpu}")
            set(_fixture_name native-${_case})
            set(_group_options -DCONSAN_GROUP_BYTES=128)
            list(FIND _names ${_case} _named_index)
            if(NOT _named_index EQUAL -1)
                if(_case STREQUAL "barrier" OR _case STREQUAL "wide")
                    # These rejection-only inputs are not executable scenarios.
                    continue()
                endif()
                math(EXPR _case_number "${_named_index} + 1")
                set(_fixture_name ${_case})
                # Preserve the named controls' original LDS extents and code,
                # including accesses at the end of the eight-byte allocation.
                set(_group_options)
            endif()
            if(_case STREQUAL "unknown-environment")
                set(_case_number 1)
                set(_metadata_target "amdgcn-amd-amdhsa-unknown-${_gpu}")
            endif()
            if(_case STREQUAL "distant-entry")
                set(_case_number 2)
                set(_layout_options -DCONSAN_PREFIX_NOPS=33000)
            elseif(_case STREQUAL "dense-sites")
                set(_case_number 2)
                set(_layout_options -DCONSAN_LDS_ACCESS_PAIRS=256)
            endif()
            set(_object ${_directory}/${_fixture_name}.o)
            set(_image ${_directory}/${_fixture_name}.hsaco)
            add_custom_command(
                OUTPUT ${_object}
                COMMAND
                    ${CMAKE_CXX_COMPILER_LAUNCHER} ${AMDCXX} -x
                    assembler-with-cpp -P -target amdgcn-amd-amdhsa
                    -mcpu=${_gpu} -mcode-object-version=5 ${_wave_options}
                    -DCONSAN_TARGET="amdgcn-amd-amdhsa--${_gpu}"
                    -DCONSAN_METADATA_TARGET="${_metadata_target}"
                    -DCONSAN_WAVE_SIZE=${_wave} -DCONSAN_LEGACY=${_legacy}
                    -DCONSAN_GFX12=${_gfx12} -DCONSAN_CASE=${_case_number}
                    ${_group_options} ${_layout_options} -c
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
