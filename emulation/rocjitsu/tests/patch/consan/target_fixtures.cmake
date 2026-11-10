# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

set(_target_fixtures)
foreach(
    _gpu
    IN
    ITEMS gfx942 gfx950 gfx1100 gfx1201 gfx1250
)
    foreach(_wave_size IN ITEMS 32 64)
        set(_legacy 0)
        set(_gfx12 0)
        if(_gpu STREQUAL "gfx942" OR _gpu STREQUAL "gfx950")
            if(_wave_size EQUAL 32)
                continue()
            endif()
            set(_legacy 1)
        elseif(_gpu STREQUAL "gfx1250" AND _wave_size EQUAL 64)
            continue()
        endif()
        if(_gpu STREQUAL "gfx1201" OR _gpu STREQUAL "gfx1250")
            set(_gfx12 1)
        endif()
        set(_directory ${_fixture_dir}/targets/${_gpu}/wave${_wave_size})
        file(MAKE_DIRECTORY ${_directory})
        set(_wave_size_options)
        if(NOT _legacy)
            if(_wave_size EQUAL 64)
                set(_wave_size_options -mwavefrontsize64)
            else()
                set(_wave_size_options -mno-wavefrontsize64)
            endif()
        endif()
        set(_barrier_cases)
        if(_gpu STREQUAL "gfx1250")
            set(_barrier_cases
                9
                15
                16
                17
                18
                19
                20
                21
                22
                23
                24
                25
                26
                27
                28
                29
                30
                31
                32
                33
                34
                35
                36
                37
                38
                39
                barrier-max31
                barrier-max32
                barrier-max33
                no-barrier-max32
            )
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
                ${_barrier_cases}
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
            elseif(_case MATCHES "^barrier-max(31|32|33)$")
                set(_case_number 15)
                set(_layout_options -DCONSAN_MAX_WORKGROUP=${CMAKE_MATCH_1})
            elseif(_case STREQUAL "no-barrier-max32")
                set(_case_number 1)
                set(_layout_options -DCONSAN_MAX_WORKGROUP=32)
            endif()
            set(_object ${_directory}/${_fixture_name}.o)
            set(_image ${_directory}/${_fixture_name}.hsaco)
            add_custom_command(
                OUTPUT ${_object}
                COMMAND
                    ${CMAKE_CXX_COMPILER_LAUNCHER} ${AMDCXX} -x
                    assembler-with-cpp -P -target amdgcn-amd-amdhsa
                    -mcpu=${_gpu} -mcode-object-version=5 ${_wave_size_options}
                    -DCONSAN_TARGET="amdgcn-amd-amdhsa--${_gpu}"
                    -DCONSAN_METADATA_TARGET="${_metadata_target}"
                    -DCONSAN_WAVE_SIZE=${_wave_size} -DCONSAN_LEGACY=${_legacy}
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
                    -amdgpu-enable-vopd=0 ${_wave_size_options}
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
        foreach(
            _spec
            IN
            ITEMS
                u16:16:0:0:0:0
                i16:16:1:0:0:0
                b64:64:0:0:0:0
                b128:128:0:0:0:0
                u16-high:16:0:1:0:0
                i16-high:16:1:1:0:0
                b64-high:64:0:1:0:0
                b128-high:128:0:1:0:0
                b128-alias:128:0:0:1:0
                b64-collision:64:0:0:0:1
                excluded-b8:8:0:0:0:0
                excluded-b96:96:0:0:0:0
                excluded-partial:17:0:0:0:0
        )
            string(REPLACE ":" ";" _parts ${_spec})
            list(GET _parts 0 _name)
            if(
                NOT _gpu STREQUAL "gfx1250"
                AND (
                    _name STREQUAL "u16-high"
                    OR _name STREQUAL "i16-high"
                    OR _name STREQUAL "b128-high"
                    OR _name STREQUAL "b64-collision"
                )
            )
                continue()
            endif()
            list(GET _parts 1 _width)
            list(GET _parts 2 _signed)
            list(GET _parts 3 _high)
            list(GET _parts 4 _alias)
            list(GET _parts 5 _collision)
            set(_object ${_directory}/width-${_name}.o)
            set(_image ${_directory}/width-${_name}.hsaco)
            add_custom_command(
                OUTPUT ${_object}
                COMMAND
                    ${CMAKE_CXX_COMPILER_LAUNCHER} ${AMDCXX} -x
                    assembler-with-cpp -P -target amdgcn-amd-amdhsa
                    -mcpu=${_gpu} -mcode-object-version=5 ${_wave_size_options}
                    -DCONSAN_TARGET="amdgcn-amd-amdhsa--${_gpu}"
                    -DCONSAN_WAVE_SIZE=${_wave_size} -DCONSAN_LEGACY=${_legacy}
                    -DCONSAN_GFX12=${_gfx12} -DCONSAN_WIDTH=${_width}
                    -DCONSAN_SIGNED=${_signed} -DCONSAN_HIGH=${_high}
                    -DCONSAN_ALIAS=${_alias} -DCONSAN_COLLISION=${_collision} -c
                    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/native_widths.s -o
                    ${_object}
                DEPENDS fixtures/native_widths.s ${AMDCXX}
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
        foreach(
            _case
            IN
            ITEMS 1 2 3 4 5
        )
            if(NOT _gpu STREQUAL "gfx1250")
                continue()
            endif()
            set(_object ${_directory}/mixed-${_case}.o)
            set(_image ${_directory}/mixed-${_case}.hsaco)
            add_custom_command(
                OUTPUT ${_object}
                COMMAND
                    ${CMAKE_CXX_COMPILER_LAUNCHER} ${AMDCXX} -x
                    assembler-with-cpp -P -target amdgcn-amd-amdhsa
                    -mcpu=${_gpu} -mcode-object-version=5 ${_wave_size_options}
                    -DCONSAN_TARGET="amdgcn-amd-amdhsa--${_gpu}"
                    -DCONSAN_WAVE_SIZE=${_wave_size} -DCONSAN_LEGACY=${_legacy}
                    -DCONSAN_GFX12=${_gfx12} -DCONSAN_MIXED=${_case} -c
                    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/mixed_widths.s -o
                    ${_object}
                DEPENDS fixtures/mixed_widths.s ${AMDCXX}
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
        set(_group_cases 1 3 4 6)
        if(_gpu STREQUAL "gfx1250")
            list(APPEND _group_cases 2 5)
        endif()
        foreach(_case IN LISTS _group_cases)
            set(_object ${_directory}/group-${_case}.o)
            set(_image ${_directory}/group-${_case}.hsaco)
            add_custom_command(
                OUTPUT ${_object}
                COMMAND
                    ${CMAKE_CXX_COMPILER_LAUNCHER} ${AMDCXX} -x
                    assembler-with-cpp -P -target amdgcn-amd-amdhsa
                    -mcpu=${_gpu} -mcode-object-version=5 ${_wave_size_options}
                    -DCONSAN_TARGET="amdgcn-amd-amdhsa--${_gpu}"
                    -DCONSAN_WAVE_SIZE=${_wave_size} -DCONSAN_LEGACY=${_legacy}
                    -DCONSAN_GFX12=${_gfx12} -DCONSAN_GROUP_CASE=${_case} -c
                    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/workgroup_lds.s -o
                    ${_object}
                DEPENDS fixtures/workgroup_lds.s ${AMDCXX}
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
        set(_object ${_directory}/native-no-workgroup-id.o)
        set(_image ${_directory}/native-no-workgroup-id.hsaco)
        add_custom_command(
            OUTPUT ${_object}
            COMMAND
                ${CMAKE_CXX_COMPILER_LAUNCHER} ${AMDCXX} -x assembler-with-cpp
                -P -target amdgcn-amd-amdhsa -mcpu=${_gpu}
                -mcode-object-version=5 ${_wave_size_options}
                -DCONSAN_GROUP_BYTES=128 -DCONSAN_NO_WORKGROUP_ID=1
                -DCONSAN_CASE=2 -DCONSAN_LEGACY=${_legacy}
                -DCONSAN_GFX12=${_gfx12} -DCONSAN_WAVE_SIZE=${_wave_size}
                -DCONSAN_TARGET="amdgcn-amd-amdhsa--${_gpu}" -c
                ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/native_lds.s -o ${_object}
            DEPENDS fixtures/native_lds.s ${AMDCXX}
            VERBATIM
        )
        add_custom_command(
            OUTPUT ${_image}
            COMMAND
                ${ROCM_PATH}/lib/llvm/bin/ld.lld -shared ${_object} -o ${_image}
            DEPENDS ${_object}
            VERBATIM
        )
        list(APPEND _target_fixtures ${_image})
    endforeach()
endforeach()
add_custom_target(consan_target_fixtures DEPENDS ${_target_fixtures})
