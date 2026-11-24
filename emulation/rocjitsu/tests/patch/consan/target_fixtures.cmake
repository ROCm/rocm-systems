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
        math(EXPR _barrier_max_below "${_wave_size} - 1")
        math(EXPR _barrier_max_above "${_wave_size} + 1")
        set(_barrier_cases
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
            42
            barrier-max${_barrier_max_below}
            barrier-max${_wave_size}
            barrier-max${_barrier_max_above}
            no-barrier-max${_wave_size}
        )
        if(_gpu STREQUAL "gfx1250")
            list(APPEND _barrier_cases 9 40 41)
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
            elseif(_case MATCHES "^barrier-max([0-9]+)$")
                set(_case_number 15)
                set(_layout_options -DCONSAN_MAX_WORKGROUP=${CMAKE_MATCH_1})
            elseif(_case STREQUAL "no-barrier-max${_wave_size}")
                set(_case_number 1)
                set(_layout_options -DCONSAN_MAX_WORKGROUP=${_wave_size})
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
        set(_atomic_specs)
        foreach(
            _scenario
            IN
            ITEMS count read write atomic adjacent
        )
            set(_name atomic-u32-${_scenario})
            string(TOUPPER "${_scenario}" _scenario_define)
            list(APPEND _atomic_specs "${_name}:32:0:0:0:0")
            list(APPEND _atomic_cases_${_gpu}_${_wave_size} ${_name})
            set(_atomic_options_${_name}
                -DCONSAN_ATOMIC=1
                -DATOMIC_${_scenario_define}=1
            )
        endforeach()
        set(_dual_specs)
        foreach(_bits IN ITEMS 32 64)
            foreach(_spacing IN ITEMS normal stride64)
                set(_suffix "")
                if(_legacy)
                    if(_spacing STREQUAL "stride64")
                        set(_suffix st64)
                    endif()
                    set(_dual_read ds_read2${_suffix}_b${_bits})
                    set(_dual_write ds_write2${_suffix}_b${_bits})
                else()
                    if(_spacing STREQUAL "stride64")
                        set(_suffix _stride64)
                    endif()
                    set(_dual_read ds_load_2addr${_suffix}_b${_bits})
                    set(_dual_write ds_store_2addr${_suffix}_b${_bits})
                endif()
                math(EXPR _scale "${_bits} / 8")
                if(_spacing STREQUAL "stride64")
                    math(EXPR _scale "${_scale} * 64")
                endif()
                set(_dual_cases values-high overlap gap)
                if(_bits EQUAL 32 AND _spacing STREQUAL "normal")
                    list(APPEND _dual_cases collision)
                endif()
                foreach(_case IN LISTS _dual_cases)
                    set(_name dual-b${_bits}-${_spacing}-${_case})
                    set(_high 0)
                    set(_peer 0)
                    set(_colliding 0)
                    if(_case STREQUAL "values-high")
                        set(_high 1)
                    elseif(_case STREQUAL "collision")
                        set(_colliding 1)
                    elseif(_case STREQUAL "overlap")
                        set(_peer 5)
                    else()
                        set(_peer 4)
                    endif()
                    list(
                        APPEND _dual_specs
                        ${_name}:${_bits}:0:${_high}:0:${_colliding}
                    )
                    set(_dual_options_${_name}
                        -DCONSAN_DUAL_READ=${_dual_read}
                        -DCONSAN_DUAL_WRITE=${_dual_write}
                        -DCONSAN_DUAL_SCALE=${_scale}
                        -DCONSAN_DUAL_PEER=${_peer}
                    )
                endforeach()
            endforeach()
        endforeach()
        set(_transpose_specs)
        if(_gpu STREQUAL "gfx950" OR _gpu STREQUAL "gfx1250")
            foreach(_bits IN ITEMS 4 6 8 16)
                set(_transpose_scenarios
                    full
                    partial
                    empty
                    overlap
                    adjacent
                )
                if(_gpu STREQUAL "gfx1250")
                    list(APPEND _transpose_scenarios short)
                endif()
                foreach(_scenario IN LISTS _transpose_scenarios)
                    list(
                        APPEND _transpose_specs
                        "transpose-${_bits}-${_scenario}:0:0:0:0:0"
                    )
                endforeach()
            endforeach()
        endif()
        foreach(
            _spec
            IN
            ITEMS
                ${_atomic_specs}
                ${_transpose_specs}
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
                ${_dual_specs}
                u8:8:0:0:0:0
                i8-alias-high:8:1:1:1:0
                b8-overlap:8:0:0:0:0:0
                b8-adjacent:8:0:0:0:0:1
                b8-disjoint:8:0:0:0:0:2
                b96:96:0:0:0:0
                b96-alias-high:96:0:1:1:0
                b96-overlap:96:0:0:0:0:4
                b96-adjacent:96:0:0:0:0:8
                b96-disjoint:96:0:0:0:0:12
                d16-u8:17:0:0:0:0
                d16-i8-high:17:1:1:0:0
                d16-u16:17:0:0:0:0
                d16-overlap:17:0:0:0:0:0
                d16-adjacent:17:0:0:0:0:4
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
            set(_peer_option)
            list(LENGTH _parts _part_count)
            if(_part_count EQUAL 7)
                list(GET _parts 6 _peer_offset)
                if(_width EQUAL 8)
                    set(_peer_option -DCONSAN_B8_PEER_STRIDE=${_peer_offset})
                elseif(_width EQUAL 17)
                    set(_peer_option -DCONSAN_D16_PEER_OFFSET=${_peer_offset})
                else()
                    set(_peer_option -DCONSAN_B96_PEER_OFFSET=${_peer_offset})
                endif()
            endif()
            if(_name STREQUAL "d16-u16")
                set(_peer_option -DCONSAN_D16_HALF=1)
            endif()
            if(_name MATCHES "^transpose-([0-9]+)-(.*)$")
                set(_form ${CMAKE_MATCH_1})
                set(_scenario ${CMAKE_MATCH_2})
                set(_peer_option -DCONSAN_TRANSPOSE_BITS=${_form})
                if(
                    _scenario STREQUAL "adjacent"
                    OR _scenario STREQUAL "overlap"
                )
                    list(APPEND _peer_option -DTR_PEER=1)
                endif()
                string(TOUPPER "${_scenario}" _scenario_define)
                list(APPEND _peer_option -DTR_${_scenario_define}=1)
            endif()
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
                    -DCONSAN_ALIAS=${_alias} -DCONSAN_COLLISION=${_collision}
                    ${_peer_option} ${_dual_options_${_name}}
                    ${_atomic_options_${_name}} -c
                    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/native_widths.s -o
                    ${_object}
                DEPENDS
                    fixtures/native_widths.s
                    fixtures/native_atomic.inc
                    fixtures/native_d16.inc
                    fixtures/native_dual.inc
                    fixtures/native_transpose.inc
                    ${AMDCXX}
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
        set(_group_cases
            1
            2
            3
            4
            5
            6
        )
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
