# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# ConSan's unit, fuzzing, and live-GPU test registration lives here so the
# shared rocjitsu test manifest remains independent of ConSan's growth.

set(RJ_CONSAN_GFX942_HIP_MOI_BUILD_DIR
    ""
    CACHE PATH
    "Exact hip-moi-build-gfx942-tests directory for simulator CTest coverage"
)
set(RJ_CONSAN_GFX1250_HIP_MOI_BUILD_DIR
    ""
    CACHE PATH
    "Exact hip-moi-build-gfx1250-tests directory for simulator CTest coverage"
)
set(RJ_CONSAN_GFX950_HIP_MOI_BUILD_DIR
    ""
    CACHE PATH
    "Exact hip-moi-build-gfx950-tests directory for simulator CTest coverage"
)

add_executable(
    consan_capability_manifest
    "${CMAKE_CURRENT_LIST_DIR}/tools/consan_capability_manifest.cpp"
)
target_include_directories(
    consan_capability_manifest
    PRIVATE "${CMAKE_SOURCE_DIR}/lib/rocjitsu/src"
)
target_link_libraries(
    consan_capability_manifest
    PRIVATE rocjitsu_dbt_internal
)
rj_configure_target(consan_capability_manifest TEST)
add_custom_target(
    consan_capability_manifest_update
    COMMAND
        ${CMAKE_COMMAND}
        "-DMANIFEST_EXECUTABLE=$<TARGET_FILE:consan_capability_manifest>"
        "-DCAPABILITIES_DOCUMENT=${CMAKE_SOURCE_DIR}/docs/consan/CAPABILITIES.md"
        -DUPDATE=ON -P
        "${CMAKE_CURRENT_LIST_DIR}/check_capability_manifest.cmake"
    DEPENDS consan_capability_manifest
    COMMENT "Updating the generated ConSan capability contract"
)
rj_add_test(
    NAME ConSan.CapabilityManifestMatchesDocumentation
    COMMAND
        ${CMAKE_COMMAND}
        "-DMANIFEST_EXECUTABLE=$<TARGET_FILE:consan_capability_manifest>"
        "-DCAPABILITIES_DOCUMENT=${CMAKE_SOURCE_DIR}/docs/consan/CAPABILITIES.md"
        -P "${CMAKE_CURRENT_LIST_DIR}/check_capability_manifest.cmake"
    LABELS consan
)

function(rj_add_consan_unit_sources target)
    target_sources(
        ${target}
        PRIVATE
            patch/consan/analysis_test.cpp
            patch/consan/atomic_fence_policy_test.cpp
            patch/consan/atomic_classifier_test.cpp
            patch/consan/access_classifier_test.cpp
            patch/consan/barrier_policy_test.cpp
            patch/consan/capability_contract_test.cpp
            patch/consan/composition_test.cpp
            patch/consan/core_test.cpp
            patch/consan/descriptor_test.cpp
            patch/consan/evidence_requirements_test.cpp
            patch/consan/fault_atomic_test.cpp
            patch/consan/fault_barrier_test.cpp
            patch/consan/fault_composition_test.cpp
            patch/consan/fault_injection_test.cpp
            patch/consan/fault_ordinary_test.cpp
            patch/consan/common_test.cpp
            patch/consan/conformance_test.cpp
            patch/consan/lowering_plan_test.cpp
            patch/consan/consan_model_test_support.cpp
            patch/consan/probe_lowering_test.cpp
            patch/consan/observation_plan_test.cpp
            patch/consan/pipeline_test.cpp
            patch/consan/program_inventory_test.cpp
            patch/consan/resource_test.cpp
            patch/consan/request_contract_test.cpp
            patch/consan/supercollider_test.cpp
            patch/consan_adversarial_test.cpp
            patch/consan_report_plan_test.cpp
    )
endfunction()

function(rj_add_consan_gfx1250_jakub_sim_test build_dir)
    if(NOT build_dir)
        return()
    endif()
    if(NOT IS_ABSOLUTE "${build_dir}")
        message(
            FATAL_ERROR
            "RJ_CONSAN_GFX1250_HIP_MOI_BUILD_DIR must be absolute"
        )
    endif()
    if(NOT IS_DIRECTORY "${build_dir}")
        message(
            FATAL_ERROR
            "RJ_CONSAN_GFX1250_HIP_MOI_BUILD_DIR is not a directory: ${build_dir}"
        )
    endif()
    get_filename_component(_build_dir_name "${build_dir}" NAME)
    if(NOT _build_dir_name STREQUAL "hip-moi-build-gfx1250-tests")
        message(
            FATAL_ERROR
            "RJ_CONSAN_GFX1250_HIP_MOI_BUILD_DIR must name hip-moi-build-gfx1250-tests"
        )
    endif()
    get_filename_component(_workspace "${build_dir}" DIRECTORY)
    set(_runner
        "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../dbi/consan/consan_gfx1250_jakub_fault_sim.py"
    )
    set(_test_name "ConSanGfx1250HipMoiSim.JakubProducerSkewBarrierDrop")
    rj_add_test(
        NAME "${_test_name}"
        COMMAND
            "${_runner}"
            --workspace
            "${_workspace}"
            --rocjitsu
            "$<TARGET_FILE:rocjitsu_bin>"
            --config
            "${CMAKE_SOURCE_DIR}/configs/gfx1250_mi455x.json"
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        # The target manifest allows 90 seconds for the producer-skew payload;
        # leave room for the separately bounded pre/post health probes.
        TIMEOUT 180
        LABELS consan gfx1250 hip-moi simulator fault
    )
    if(IS_DIRECTORY "${ROCM_PATH}/bin")
        set_property(
            TEST "${_test_name}"
            APPEND
            PROPERTY
                ENVIRONMENT_MODIFICATION
                    "PATH=path_list_prepend:${ROCM_PATH}/bin"
        )
    endif()
    set_property(TEST "${_test_name}" PROPERTY RUN_SERIAL TRUE)
endfunction()

# Registers one HIP fixture case through a single typed path. Keeping command
# construction, working-directory selection, timeout defaults, and optional
# properties here prevents the live-GPU matrix from drifting as cases are
# added. ENVIRONMENT is a list of KEY=VALUE entries. Every instrumented case
# supplies an environment; loading the hook enables ConSan, and the mode
# selects the analysis. PASS_REGEX is the primary checked-output contract;
# every ALSO_PASS_REGEX expression is an additional conjunctive requirement.
function(rj_add_consan_hip_case)
    set(_options RUN_SERIAL)
    set(_one_value_args
        NAME
        TARGET
        FILTER
        CONFIG
        TIMEOUT
        PASS_REGEX
        FAIL_REGEX
        EXPECTED_EXIT
        RESOURCE_LOCK
    )
    set(_multi_value_args ALSO_PASS_REGEX ENVIRONMENT LABELS)
    cmake_parse_arguments(
        PARSE_ARGV
        0
        _case
        "${_options}"
        "${_one_value_args}"
        "${_multi_value_args}"
    )
    if(_case_UNPARSED_ARGUMENTS)
        message(
            FATAL_ERROR
            "unknown ConSan HIP case arguments: ${_case_UNPARSED_ARGUMENTS}"
        )
    endif()
    foreach(_required IN ITEMS NAME TARGET FILTER)
        if(NOT _case_${_required})
            message(FATAL_ERROR "ConSan HIP case requires ${_required}")
        endif()
    endforeach()
    if(NOT _case_TIMEOUT)
        set(_case_TIMEOUT 30)
    endif()
    if(DEFINED _rj_consan_physical_target)
        if(_case_CONFIG)
            message(
                FATAL_ERROR
                "physical ConSan case ${_case_NAME} cannot use simulator config ${_case_CONFIG}"
            )
        endif()
        list(
            APPEND _case_LABELS
            consan
            consan-device
            "${_rj_consan_physical_target}"
            physical
        )
        list(REMOVE_DUPLICATES _case_LABELS)
        if(NOT _case_RESOURCE_LOCK)
            set(
                _case_RESOURCE_LOCK
                "consan_physical_${_rj_consan_physical_target}"
            )
        endif()
    endif()
    if(
        "consan-device" IN_LIST _case_LABELS
        AND "physical" IN_LIST _case_LABELS
        AND NOT _case_RESOURCE_LOCK
    )
        message(
            FATAL_ERROR
            "physical ConSan device test ${_case_NAME} has no target-scoped CTest resource lock"
        )
    endif()

    # Live ConSan cases are intentionally narrow fixtures. Production ConSan
    # tracks every supported synchronization class by default, but these tests
    # use report buffers and patch limits sized for the class under test. Make
    # the focused default access-only and require barrier/atomic cases to opt in
    # explicitly below.
    set(_case_uses_reports FALSE)
    set(_case_sets_barrier_tracking FALSE)
    set(_case_sets_atomic_tracking FALSE)
    set(_case_sets_runtime_sample_stride FALSE)
    foreach(_environment_entry IN LISTS _case_ENVIRONMENT)
        if(_environment_entry STREQUAL "RJ_CONSAN_MODE=default")
            set(_case_uses_reports TRUE)
        elseif(_environment_entry MATCHES "^RJ_CONSAN_TRACK_BARRIERS=")
            set(_case_sets_barrier_tracking TRUE)
        elseif(_environment_entry MATCHES "^RJ_CONSAN_TRACK_ATOMICS=")
            set(_case_sets_atomic_tracking TRUE)
        elseif(
            _environment_entry MATCHES "^RJ_CONSAN_(PRESET|RUNTIME_SAMPLE_STRIDE|WORKGROUP_SAMPLE_(STRIDE|OFFSET)|CELL_SAMPLE_(STRIDE|OFFSET))="
        )
            set(_case_sets_runtime_sample_stride TRUE)
        endif()
    endforeach()
    if(_case_uses_reports)
        if(NOT _case_sets_barrier_tracking)
            list(APPEND _case_ENVIRONMENT "RJ_CONSAN_TRACK_BARRIERS=0")
        endif()
        if(NOT _case_sets_atomic_tracking)
            list(APPEND _case_ENVIRONMENT "RJ_CONSAN_TRACK_ATOMICS=0")
        endif()
        # Compact runtime fixtures assert on a particular dispatch's records.
        # Keep that semantic contract deterministic unless a test explicitly
        # supplies a sampling stride; production defaults remain covered by
        # configuration tests and large-object validation.
        if(NOT _case_sets_runtime_sample_stride)
            list(
                APPEND _case_ENVIRONMENT
                "RJ_CONSAN_RUNTIME_SAMPLE_STRIDE=1"
            )
        endif()
    endif()

    if(_case_CONFIG)
        set(_case_command
            $<TARGET_FILE:rocjitsu_bin>
            --config
            "${_case_CONFIG}"
            --
            ${CMAKE_CURRENT_BINARY_DIR}/${_case_TARGET}
            "--gtest_filter=${_case_FILTER}"
        )
        add_dependencies(${_case_TARGET}_target rocjitsu_bin)
    else()
        set(_case_command
            ${CMAKE_CURRENT_BINARY_DIR}/${_case_TARGET}
            "--gtest_filter=${_case_FILTER}"
        )
    endif()

    if(_case_ALSO_PASS_REGEX AND NOT _case_PASS_REGEX)
        message(
            FATAL_ERROR
            "ConSan HIP case ${_case_NAME} supplies ALSO_PASS_REGEX without PASS_REGEX"
        )
    endif()
    if(_case_PASS_REGEX)
        # CMake list elements cannot safely carry every regex spelling (for
        # example, an element ending in an escaped parenthesis can absorb the
        # following list separators). Store the pattern in the build tree and
        # pass only its stable path to the checked-output runner.
        string(MD5 _case_pass_regex_id "${_case_PASS_REGEX}")
        set(_case_pass_regex_file
            "${CMAKE_CURRENT_BINARY_DIR}/consan-expected-${_case_pass_regex_id}.txt"
        )
        file(WRITE "${_case_pass_regex_file}" "${_case_PASS_REGEX}")
        set(_case_checked_args
            "-DRJ_EXPECTED_REGEX_FILE=${_case_pass_regex_file}"
        )
        set(_additional_regex_index 0)
        foreach(_additional_regex IN LISTS _case_ALSO_PASS_REGEX)
            string(
                MD5 _additional_regex_id
                "${_case_NAME}|${_additional_regex_index}|${_additional_regex}"
            )
            set(
                _additional_regex_file
                "${CMAKE_CURRENT_BINARY_DIR}/consan-expected-${_additional_regex_id}.txt"
            )
            file(WRITE "${_additional_regex_file}" "${_additional_regex}")
            list(
                APPEND _case_checked_args
                "-DRJ_ADDITIONAL_EXPECTED_REGEX_FILE_${_additional_regex_index}=${_additional_regex_file}"
            )
            math(EXPR _additional_regex_index "${_additional_regex_index} + 1")
        endforeach()
        list(
            APPEND _case_checked_args
            "-DRJ_ADDITIONAL_EXPECTED_REGEX_COUNT=${_additional_regex_index}"
        )
        if(_case_EXPECTED_EXIT)
            list(
                APPEND _case_checked_args
                "-DRJ_EXPECTED_EXIT=${_case_EXPECTED_EXIT}"
            )
        endif()
        set(_case_command
            ${CMAKE_COMMAND}
            ${_case_checked_args}
            -P
            "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/run_checked_test.cmake"
            --
            ${_case_command}
        )
    endif()

    set(_test_args
        NAME
        "${_case_NAME}"
        COMMAND
        ${_case_command}
        WORKING_DIRECTORY
        ${CMAKE_CURRENT_BINARY_DIR}
        TIMEOUT
        ${_case_TIMEOUT}
    )
    if(_case_ENVIRONMENT)
        list(APPEND _test_args ENVIRONMENT ${_case_ENVIRONMENT})
    endif()
    if(_case_LABELS)
        list(APPEND _test_args LABELS ${_case_LABELS})
    endif()
    rj_add_test(${_test_args})

    if(_case_CONFIG AND ROCM_PATH AND UNIX)
        # Simulator configurations may load HSA hotswap libraries from the
        # selected ROCm installation. TheRock keeps those libraries in lib/
        # and their bundled system dependencies in lib/rocm_sysdeps/lib.
        # Amend this test's loader path at execution time, preserving the
        # caller's environment and keeping ROCM_PATH relocatable.
        foreach(
            _rocm_library_dir
            IN
            ITEMS
                "${ROCM_PATH}/lib"
                "${ROCM_PATH}/lib64"
                "${ROCM_PATH}/lib/rocm_sysdeps/lib"
        )
            if(IS_DIRECTORY "${_rocm_library_dir}")
                set_property(
                    TEST "${_case_NAME}"
                    APPEND
                    PROPERTY
                        ENVIRONMENT_MODIFICATION
                            "LD_LIBRARY_PATH=path_list_prepend:${_rocm_library_dir}"
                )
            endif()
        endforeach()
    endif()

    # The checked-output wrapper above keeps an expected diagnostic from
    # overriding a nonzero child exit status.
    set(_case_fail_regex "\\[  FAILED  \\]|FAILED TEST")
    if(_case_FAIL_REGEX)
        list(APPEND _case_fail_regex "${_case_FAIL_REGEX}")
    endif()
    if(_case_uses_reports)
        # Automatic reports are consumed by both the GPU and host. A coarse
        # region may appear to work on a single-GPU host while faulting on a
        # different runtime/topology, so pin the portable allocation contract
        # in every real ConSan device execution.
        list(
            APPEND _case_fail_regex
            "coarse_grained_snapshot_bytes=[1-9][0-9]*"
        )
    endif()
    set_property(
        TEST "${_case_NAME}"
        PROPERTY FAIL_REGULAR_EXPRESSION "${_case_fail_regex}"
    )
    if(_case_RUN_SERIAL)
        set_property(TEST "${_case_NAME}" PROPERTY RUN_SERIAL TRUE)
    endif()
    if(_case_RESOURCE_LOCK)
        set_property(
            TEST "${_case_NAME}"
            PROPERTY RESOURCE_LOCK "${_case_RESOURCE_LOCK}"
        )
    endif()
endfunction()

function(rj_add_consan_device_conformance_cases)
    set(_options PHYSICAL)
    set(_one_value_args PREFIX TARGET CONFIG)
    cmake_parse_arguments(
        PARSE_ARGV
        0
        _device
        "${_options}"
        "${_one_value_args}"
        ""
    )
    if(_device_UNPARSED_ARGUMENTS)
        message(
            FATAL_ERROR
            "unknown ConSan device-conformance arguments: ${_device_UNPARSED_ARGUMENTS}"
        )
    endif()
    foreach(_required IN ITEMS PREFIX TARGET)
        if(NOT _device_${_required})
            message(
                FATAL_ERROR
                "ConSan device-conformance cases require ${_required}"
            )
        endif()
    endforeach()
    if(NOT _device_PHYSICAL AND NOT _device_CONFIG)
        message(
            FATAL_ERROR
            "simulated ConSan device-conformance cases require CONFIG"
        )
    endif()

    set(_config_args)
    set(_resource_lock_args)
    set(_backend simulator)
    if(_device_CONFIG)
        list(APPEND _config_args CONFIG "${_device_CONFIG}")
    else()
        set(_backend physical)
        list(
            APPEND _resource_lock_args
            RESOURCE_LOCK
            "consan_physical_${_device_TARGET}"
        )
    endif()
    set(_labels consan consan-device "${_device_TARGET}" "${_backend}")

    foreach(_epoch_case IN ITEMS LastOrderedPair LastEpochRace ExhaustedOrderedPair RepeatedExhaustion OverflowPolicy)
        set(_epoch_filter "${_epoch_case}")
        set(_epoch_exit 0)
        set(_epoch_forbid_overflow 0)
        set(_epoch_require_records 1)
        if(_epoch_case STREQUAL "OverflowPolicy")
            set(_epoch_filter RepeatedExhaustion)
            set(_epoch_exit 90)
            set(_epoch_forbid_overflow 1)
        endif()
        if(_epoch_filter STREQUAL "RepeatedExhaustion")
            set(_epoch_require_records 0)
        endif()
        set(_epoch_complete true)
        set(_epoch_exhaustion 0)
        set(_epoch_race 0)
        if(_epoch_filter MATCHES "Exhausted|Repeated")
            set(_epoch_complete false)
            if(_device_TARGET MATCHES "gfx942|gfx950")
                set(_epoch_exhaustion 2)
            else()
                set(_epoch_exhaustion 4)
            endif()
        elseif(_epoch_case STREQUAL "LastEpochRace")
            set(_epoch_race 1)
        endif()
        rj_add_consan_hip_case(
            NAME "${_device_PREFIX}.Epoch.${_epoch_case}"
            TARGET hip_consan_device_epoch_exhaustion_${_device_TARGET}_test
            FILTER "ConSanDeviceEpochTest.${_epoch_filter}"
            EXPECTED_EXIT ${_epoch_exit}
            ${_config_args}
            ${_resource_lock_args}
            TIMEOUT 120
            PASS_REGEX "ConSan analysis verdict applicable=true analysis_complete=${_epoch_complete}.*dynamic_complete=${_epoch_complete}.*epoch_exhaustion=${_epoch_exhaustion}"
            ENVIRONMENT
                "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                "HSA_TOOLS_DISABLE_REGISTER=1"
                "RJ_CONSAN_MODE=default"
                "RJ_CONSAN_LOG=1"
                "RJ_CONSAN_PRESET=max"
                "RJ_CONSAN_REQUIRE_RECORDS=${_epoch_require_records}"
                "RJ_CONSAN_REQUIRE_DIAGNOSTICS=${_epoch_race}"
                "RJ_CONSAN_FORBID_DIAGNOSTICS=$<IF:$<BOOL:${_epoch_race}>,0,1>"
                "RJ_CONSAN_FORBID_OVERFLOW=${_epoch_forbid_overflow}"
                "RJ_CONSAN_TRACK_BARRIERS=1"
            LABELS ${_labels} default epoch-exhaustion
        )
    endforeach()

    # Exact masks currently require ordinary VGPR addressing, without the
    # selectable-bank state of gfx1250.
    if(NOT _device_TARGET STREQUAL "gfx1250")
        foreach(_uniform_case IN ITEMS DifferentValues DefaultSelected SingleWriter Disjoint ReadOnly Atomic PartialExec EmptyExec CrossWave)
            set(_uniform_stride 1)
            set(_uniform_race 0)
            set(_uniform_records 1)
            set(_uniform_clean 1)
            set(_uniform_regex "conflicts=0")
            if(_uniform_case MATCHES "^(DifferentValues|DefaultSelected|PartialExec|CrossWave)$")
                set(_uniform_race 1)
                set(_uniform_clean 0)
                set(_uniform_regex "first_lanes=0x0000000000000001 second_lanes=0x[0-9a-f]*[1-9a-f][0-9a-f]*")
            endif()
            if(_uniform_case STREQUAL "EmptyExec")
                set(_uniform_records 0)
                set(_uniform_regex "visible=0.*conflicts=0")
            elseif(_uniform_case STREQUAL "PartialExec")
                set(_uniform_regex "first_lanes=0x0000000000000001 second_lanes=0x0000000000000008")
            elseif(_uniform_case STREQUAL "CrossWave")
                set(_uniform_regex "first_lanes=0x[0f]*ffffffff second_lanes=0x[0f]*ffffffff")
            endif()
            if(_uniform_case STREQUAL "DefaultSelected")
                set(_uniform_stride 256)
            endif()
            rj_add_consan_hip_case(
                NAME "${_device_PREFIX}.Uniform.${_uniform_case}"
                TARGET hip_consan_device_selection_${_device_TARGET}_test
                FILTER "ConSanDeviceUniformTest.${_uniform_case}"
                ${_config_args}
                ${_resource_lock_args}
                TIMEOUT 120
                PASS_REGEX "${_uniform_regex}"
                ALSO_PASS_REGEX "analysis_complete=true static_complete=true dynamic_complete=true"
                ENVIRONMENT
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=default"
                    "RJ_CONSAN_LOG=3"
                    "RJ_CONSAN_FAIL_CLOSED=1"
                    "RJ_CONSAN_REQUIRE_PATCH=1"
                    "RJ_CONSAN_RUNTIME_SAMPLE_STRIDE=${_uniform_stride}"
                    "RJ_CONSAN_REQUIRE_RECORDS=${_uniform_records}"
                    "RJ_CONSAN_REQUIRE_DIAGNOSTICS=${_uniform_race}"
                    "RJ_CONSAN_FORBID_DIAGNOSTICS=${_uniform_clean}"
                LABELS ${_labels} default uniform
            )
        endforeach()
    endif()

    if(NOT _device_TARGET STREQUAL "gfx1250")
        foreach(_preset IN ITEMS high higher max)
            set(_preset_cases DifferentValues CrossWave SingleWriter NonzeroCell)
            if(_preset STREQUAL "high")
                # A single workgroup can be filtered out at stride 16. This
                # case launches 256 groups and exercises both preset selectors.
                set(_preset_cases DefaultSelected)
            endif()
            foreach(_case IN LISTS _preset_cases)
                set(_records 1)
                set(_race 1)
                set(_clean 0)
                set(_regex "conflicts=[1-9][0-9]*")
                if(_case STREQUAL "CrossWave")
                    set(_regex "first_lanes=0x[0f]*ffffffff second_lanes=0x[0f]*ffffffff")
                endif()
                if(_case STREQUAL "SingleWriter")
                    set(_race 0)
                    set(_clean 1)
                    set(_regex "conflicts=0")
                endif()
                if(_case STREQUAL "NonzeroCell" AND _preset STREQUAL "higher")
                    set(_records 0)
                    set(_race 0)
                    set(_clean 1)
                    set(_regex "conflicts=0")
                endif()
                rj_add_consan_hip_case(
                    NAME "${_device_PREFIX}.Preset.${_preset}.${_case}"
                    TARGET hip_consan_device_selection_${_device_TARGET}_test
                    FILTER "ConSanDeviceUniformTest.${_case}"
                    ${_config_args}
                    ${_resource_lock_args}
                    TIMEOUT 120
                    PASS_REGEX "${_regex}"
                    ALSO_PASS_REGEX "analysis_complete=true static_complete=true dynamic_complete=true"
                    ENVIRONMENT
                        "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                        "RJ_CONSAN_MODE=default"
                        "RJ_CONSAN_PRESET=${_preset}"
                        "RJ_CONSAN_TRACK_BARRIERS=1"
                        "RJ_CONSAN_TRACK_ATOMICS=1"
                        "RJ_CONSAN_LOG=3"
                        "RJ_CONSAN_FAIL_CLOSED=1"
                        "RJ_CONSAN_REQUIRE_PATCH=1"
                        "RJ_CONSAN_REQUIRE_RECORDS=${_records}"
                        "RJ_CONSAN_REQUIRE_DIAGNOSTICS=${_race}"
                        "RJ_CONSAN_FORBID_DIAGNOSTICS=${_clean}"
                    LABELS ${_labels} default preset
                )
            endforeach()
        endforeach()
    endif()

    foreach(_allow_uniform IN ITEMS 0 1)

            foreach(_value_case IN ITEMS IdenticalWords VaryingLastWord CrossWave)
                # gfx1250 has no exact-lane evidence for same-instruction collisions.
                if(_device_TARGET STREQUAL "gfx1250"
                   AND (_value_case STREQUAL "VaryingLastWord" OR
                        (_value_case STREQUAL "IdenticalWords" AND NOT _allow_uniform)))
                    continue()
                endif()
                set(_value_race 1)
                set(_value_clean 0)
                if(_value_case STREQUAL "IdenticalWords" AND _allow_uniform)
                    set(_value_race 0)
                    set(_value_clean 1)
                endif()
                set(_value_regex "conflicts=[1-9][0-9]*")
                if(_value_clean)
                    set(_value_regex "conflicts=0")
                endif()
                rj_add_consan_hip_case(
                    NAME "${_device_PREFIX}.UniformValue.default.${_value_case}.Allow${_allow_uniform}"
                    TARGET hip_consan_device_selection_${_device_TARGET}_test
                    FILTER "ConSanDeviceUniformValueTest.${_value_case}"
                    ${_config_args}
                    ${_resource_lock_args}
                    TIMEOUT 120
                    PASS_REGEX "${_value_regex}"
                    ENVIRONMENT
                        "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                        "RJ_CONSAN_MODE=default"
                        "RJ_CONSAN_ALLOW_PROVABLY_SAME_VALUE_WRITE_RACES=${_allow_uniform}"
                        "RJ_CONSAN_LOG=3"
                        "RJ_CONSAN_FAIL_CLOSED=1"
                        "RJ_CONSAN_REQUIRE_PATCH=1"
                        "RJ_CONSAN_RUNTIME_SAMPLE_STRIDE=1"
                        "RJ_CONSAN_TRACK_BARRIERS=1"
                        "RJ_CONSAN_TRACK_ATOMICS=1"
                        "RJ_CONSAN_REQUIRE_RECORDS=1"
                        "RJ_CONSAN_REQUIRE_DIAGNOSTICS=${_value_race}"
                        "RJ_CONSAN_FORBID_DIAGNOSTICS=${_value_clean}"
                    LABELS ${_labels} uniform-value
                )
            endforeach()


    endforeach()

    # Production-rate qualification is separate from the dense semantic
    # matrix. Every selector is explicit to avoid inherited environment and
    # the helper's usual stride-1 override.
    foreach(_selection_case IN ITEMS
            "DefaultSelected|RaceCellZero|256|0|256|0|legacy|race"
            "DefaultUnselectedAddress|RaceCellOne|256|0|256|0|legacy|empty"
            "DefaultOrdered|OrderedCellZero|256|0|256|0|legacy|clean"
            "OffsetSelectsAddress|RaceCellOne|256|0|256|1|independent|race"
            "OtherWorkgroupOffset|RaceCellZero|256|1|1|0|independent|race"
            "DenseSelectsAddress|RaceCellOneSingleWorkgroup|1|0|1|0|independent|race"
            "DenseWorkgroupSparseCell|RaceCellZeroSingleWorkgroup|1|0|256|0|independent|race")
        string(REPLACE "|" ";" _selection_fields "${_selection_case}")
        list(GET _selection_fields 0 _selection_name)
        list(GET _selection_fields 1 _selection_filter)
        list(GET _selection_fields 2 _workgroup_stride)
        list(GET _selection_fields 3 _workgroup_offset)
        list(GET _selection_fields 4 _cell_stride)
        list(GET _selection_fields 5 _cell_offset)
        list(GET _selection_fields 6 _selector_mode)
        list(GET _selection_fields 7 _expectation)
        if(_selector_mode STREQUAL "legacy")
            set(_selection_environment
                "RJ_CONSAN_RUNTIME_SAMPLE_STRIDE=${_workgroup_stride}"
                "RJ_CONSAN_RUNTIME_SAMPLE_OFFSET=${_workgroup_offset}"
                "RJ_CONSAN_WORKGROUP_SAMPLE_STRIDE="
                "RJ_CONSAN_WORKGROUP_SAMPLE_OFFSET="
                "RJ_CONSAN_CELL_SAMPLE_STRIDE="
                "RJ_CONSAN_CELL_SAMPLE_OFFSET=")
        else()
            set(_selection_environment
                "RJ_CONSAN_RUNTIME_SAMPLE_STRIDE="
                "RJ_CONSAN_RUNTIME_SAMPLE_OFFSET="
                "RJ_CONSAN_WORKGROUP_SAMPLE_STRIDE=${_workgroup_stride}"
                "RJ_CONSAN_WORKGROUP_SAMPLE_OFFSET=${_workgroup_offset}"
                "RJ_CONSAN_CELL_SAMPLE_STRIDE=${_cell_stride}"
                "RJ_CONSAN_CELL_SAMPLE_OFFSET=${_cell_offset}")
        endif()
        if(_expectation STREQUAL "race")
            set(_selection_regex "conflicts=[1-9][0-9]*")
            list(APPEND _selection_environment
                "RJ_CONSAN_REQUIRE_RECORDS=1"
                "RJ_CONSAN_REQUIRE_DIAGNOSTICS=1"
                "RJ_CONSAN_FORBID_DIAGNOSTICS=0")
        else()
            list(APPEND _selection_environment
                "RJ_CONSAN_REQUIRE_DIAGNOSTICS=0"
                "RJ_CONSAN_FORBID_DIAGNOSTICS=1")
            if(_expectation STREQUAL "empty")
                set(_selection_regex "visible=0.*conflicts=0")
                list(APPEND _selection_environment "RJ_CONSAN_REQUIRE_RECORDS=0")
            else()
                set(_selection_regex "visible=[1-9][0-9]*.*conflicts=0")
                list(APPEND _selection_environment "RJ_CONSAN_REQUIRE_RECORDS=1")
            endif()
        endif()
        rj_add_consan_hip_case(
            NAME "${_device_PREFIX}.Selection.${_selection_name}"
            TARGET hip_consan_device_selection_${_device_TARGET}_test
            FILTER "ConSanDeviceSelectionTest.${_selection_filter}"
            ${_config_args}
            ${_resource_lock_args}
            TIMEOUT 120
            PASS_REGEX "${_selection_regex}"
            ALSO_PASS_REGEX "analysis_complete=true static_complete=true dynamic_complete=true"
            ENVIRONMENT
                "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                "RJ_CONSAN_MODE=default"
                "RJ_CONSAN_LOG=1"
                "RJ_CONSAN_FAIL_CLOSED=1"
                "RJ_CONSAN_REQUIRE_PATCH=1"
                "RJ_CONSAN_SAMPLE_STRIDE=1"
                "RJ_CONSAN_SAMPLE_OFFSET=0"
                "RJ_CONSAN_WATCHPOINT_BANKS=0"
                "RJ_CONSAN_TRACK_BARRIERS=1"
                "RJ_CONSAN_TRACK_ATOMICS=1"
                "RJ_CONSAN_EPOCH_ANALYSIS=every"
                ${_selection_environment}
            LABELS ${_labels} default sampling-selection
        )
    endforeach()

    # Loading the tool DSO is ConSan's activation contract and selects
    # ConSan by default. Exercise synchronized and racy real LDS
    # workloads and require a fine-grained report snapshot so the
    # default-mode path cannot pass merely by leaving the hook inert or by
    # allocating topology-sensitive coarse memory.
    foreach(_outcome IN ITEMS Correct Incorrect)
        string(TOLOWER "${_outcome}" _outcome_label)
        set(
            _hook_default_environment
            "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
            "RJ_CONSAN_MODE="
            "RJ_CONSAN_POLICY="
            "RJ_CONSAN_LOG=1"
            "RJ_CONSAN_RUNTIME_SAMPLE_STRIDE=1"
        )
        if(_outcome STREQUAL "Correct")
            list(
                APPEND _hook_default_environment
                "RJ_CONSAN_FORBID_DIAGNOSTICS=1"
            )
        else()
            list(
                APPEND _hook_default_environment
                "RJ_CONSAN_REQUIRE_RECORDS=1"
            )
        endif()
        if(_outcome STREQUAL "Correct")
            set(_hook_default_conflicts "conflicts=0")
        else()
            set(_hook_default_conflicts "conflicts=[1-9][0-9]*")
        endif()
        rj_add_consan_hip_case(
            NAME
                "${_device_PREFIX}.HookLoadDefaultsToLds.${_outcome}"
            TARGET
                hip_consan_device_ordered_tile_handoff_${_device_TARGET}_test
            FILTER "ConSanDeviceOrderedTileHandoffTest.${_outcome}"
            ${_config_args}
            ${_resource_lock_args}
            TIMEOUT 60
            PASS_REGEX
                "fine_grained_snapshot_bytes=[1-9][0-9]* coarse_grained_snapshot_bytes=0"
            ALSO_PASS_REGEX "${_hook_default_conflicts}"
            FAIL_REGEX "coarse_grained_snapshot_bytes=[1-9][0-9]*"
            ENVIRONMENT ${_hook_default_environment}
            LABELS
                ${_labels}
                hook-default
                "${_outcome_label}"
                default
                regression-10622
        )
    endforeach()

    if(_device_TARGET STREQUAL "gfx950" OR _device_TARGET STREQUAL "gfx1250")
        rj_add_consan_hip_case(
            NAME
                "${_device_PREFIX}.DispatchIdSgprPressureForcedSpill.Correct"
            TARGET
                hip_consan_device_dispatch_id_sgpr_pressure_${_device_TARGET}_test
            FILTER "ConSanDeviceDispatchIdSgprPressureTest.Correct"
            ${_config_args}
            ${_resource_lock_args}
            TIMEOUT 60
            PASS_REGEX
                "spill=[1-9][0-9]*.*emitted_spill_patches=[1-9][0-9]*"
            ENVIRONMENT
                "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                "RJ_CONSAN_MODE=default"
                "RJ_CONSAN_POLICY=strict"
                "RJ_CONSAN_REQUIRE_PATCH=1"
                "RJ_CONSAN_LOG=1"
                "RJ_CONSAN_TRACK_BARRIERS=1"
                "RJ_CONSAN_TRACK_ATOMICS=0"
                "RJ_CONSAN_REQUIRE_RECORDS=1"
                "RJ_CONSAN_FORBID_DIAGNOSTICS=1"
                "RJ_CONSAN_FORBID_OVERFLOW=1"
                "RJ_CONSAN_TEST_FORCE_VGPR_SPILL=1"
                "RJ_CONSAN_TEST_KERNEL_FILTER=dispatch_id_sgpr_pressure_for_consan"
            LABELS
                ${_labels}
                correct
                default
                regression-11019
        )
    endif()

    if(_device_TARGET STREQUAL "gfx950")
        # Minimal device-level form of the tree-atomic-or regression: inserting
        # queue/dispatch preloads shifts the kernarg pointer, while full scalar
        # liveness forces the private entry prologue to preserve a borrowed
        # guest-SGPR window. Saving that window before ABI repair restores the
        # queue pointer over kernarg and faults on the first guest access.
        rj_add_consan_hip_case(
            NAME
                "${_device_PREFIX}.PrivateEntryAbiSpill.Correct"
            TARGET
                hip_consan_device_dispatch_id_sgpr_pressure_${_device_TARGET}_test
            FILTER "ConSanDeviceDispatchIdSgprPressureTest.PrivateEntryAbiSpill"
            ${_config_args}
            ${_resource_lock_args}
            TIMEOUT 60
            PASS_REGEX
                "spill=[1-9][0-9]*.*emitted_spill_patches=[1-9][0-9]*"
            ENVIRONMENT
                "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                "RJ_CONSAN_MODE=default"
                "RJ_CONSAN_POLICY=strict"
                "RJ_CONSAN_REQUIRE_PATCH=1"
                "RJ_CONSAN_LOG=1"
                "RJ_CONSAN_TRACK_BARRIERS=0"
                "RJ_CONSAN_TRACK_ATOMICS=0"
                "RJ_CONSAN_REQUIRE_RECORDS=1"
                "RJ_CONSAN_FORBID_DIAGNOSTICS=1"
                "RJ_CONSAN_FORBID_OVERFLOW=1"
                "RJ_CONSAN_TEST_FORCE_VGPR_SPILL=1"
                "RJ_CONSAN_TEST_FORCE_PRIVATE_EPOCH=1"
                "RJ_CONSAN_TEST_KERNEL_FILTER=private_entry_abi_spill_for_consan"
            LABELS
                ${_labels}
                correct
                default
                private-entry-abi
        )
    endif()

    # Vary workgroup selection while observing every LDS cell. The correct
    # reduction must remain diagnostic-free; omitting the publication barrier
    # must produce a diagnostic even when only a sparse workgroup is selected.
    foreach(
        _sampling_profile
        IN
        ITEMS
            "Dense|1"
            "Sparse|4"
    )
        string(REPLACE "|" ";" _sampling_fields "${_sampling_profile}")
        list(GET _sampling_fields 0 _sampling_name)
        list(GET _sampling_fields 1 _sampling_stride)
        foreach(_outcome IN ITEMS Correct Incorrect)
            string(TOLOWER "${_outcome}" _outcome_label)
            set(
                _sampling_environment
                "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                "RJ_CONSAN_MODE=default"
                "RJ_CONSAN_POLICY=strict"
                "RJ_CONSAN_REQUIRE_PATCH=1"
                "RJ_CONSAN_LOG=1"
                "RJ_CONSAN_TRACK_BARRIERS=1"
                "RJ_CONSAN_TRACK_ATOMICS=0"
                "RJ_CONSAN_WORKGROUP_SAMPLE_STRIDE=${_sampling_stride}"
                "RJ_CONSAN_WORKGROUP_SAMPLE_OFFSET=0"
                "RJ_CONSAN_CELL_SAMPLE_STRIDE=1"
                "RJ_CONSAN_REQUIRE_RECORDS=1"
                "RJ_CONSAN_FORBID_OVERFLOW=1"
            )
            if(_outcome STREQUAL "Correct")
                list(
                    APPEND _sampling_environment
                    "RJ_CONSAN_FORBID_DIAGNOSTICS=1"
                )
            else()
                list(
                    APPEND _sampling_environment
                    "RJ_CONSAN_REQUIRE_DIAGNOSTICS=1"
                )
            endif()
            rj_add_consan_hip_case(
                NAME
                    "${_device_PREFIX}.ConSan${_sampling_name}SamplingReduction.${_outcome}"
                TARGET
                    hip_consan_device_sampling_reduction_${_device_TARGET}_test
                FILTER
                    "ConSanDeviceSamplingReductionTest.${_sampling_name}${_outcome}"
                ${_config_args}
                ${_resource_lock_args}
                TIMEOUT 60
                PASS_REGEX "visible=[1-9][0-9]*"
                ENVIRONMENT ${_sampling_environment}
                LABELS
                    ${_labels}
                    default
                    sampling
                    "${_outcome_label}"
                    regression-10622
            )
        endforeach()
    endforeach()

    set(_device_workloads
        "OrderedTileHandoff|ordered_tile_handoff|ConSanDeviceOrderedTileHandoffTest"
        "Reduction|reduction|ConSanDeviceReductionTest"
        "SharedHelperOwners|shared_helper_owners|ConSanDeviceSharedHelperOwnersTest"
        "IndependentWorkgroups|independent_workgroups|ConSanDeviceIndependentWorkgroupsTest"
        "RepeatedDispatchIdentity|repeated_dispatch_identity|ConSanDeviceRepeatedDispatchIdentityTest"
        "CrossKernelDispatchIdentity|cross_kernel_dispatch_identity|ConSanDeviceCrossKernelDispatchIdentityTest|default-only"
        "GraphReplay|graph_replay|ConSanDeviceGraphReplayTest"
        "GraphParameterUpdate|graph_parameter_update|ConSanDeviceGraphParameterUpdateTest"
        "DynamicPrivateStack|dynamic_private_stack|ConSanDeviceDynamicPrivateStackTest"
        "KernargPreloadDynamicPrivateStack|kernarg_preload_dynamic_private_stack|ConSanDeviceDynamicPrivateStackTest"
        "MixedPrivateOwners|mixed_private_owners|ConSanDeviceMixedPrivateOwnersTest"
        "ModuleLifecycle|module_lifecycle|ConSanDeviceModuleLifecycleTest"
        "OverlappingSubwords|overlapping_subwords|ConSanDeviceOverlappingSubwordsTest"
        "AdjacentSubwordWriters|adjacent_subword_writers|ConSanDeviceAdjacentSubwordWritersTest"
        "IndexedLdsAlias|indexed_lds_alias|ConSanDeviceIndexedLdsAliasTest"
        "AtomicArrival|atomic_arrival|ConSanDeviceAtomicArrivalTest|atomics"
        "AtomicCasPublication|atomic_cas_publication|ConSanDeviceAtomicCasPublicationTest|atomics"
        "AtomicStorePublication|atomic_store_publication|ConSanDeviceAtomicStorePublicationTest|atomics"
        "FlatAtomicPublication|flat_atomic_publication|ConSanDeviceFlatAtomicPublicationTest|atomics"
        "StreamKLastArriver|streamk_last_arriver|ConSanDeviceStreamKLastArriverTest|atomics"
        "RcclPartialBarrier|rccl_partial_barrier|ConSanDeviceRcclPartialBarrierTest|atomics"
        "TreeAtomicOr|tree_atomic_or|ConSanDeviceTreeAtomicOrTest|atomics"
        "AccessHeavyStreamK|access_heavy_streamk|ConSanDeviceAccessHeavyStreamKTest"
        "DoubleBufferedPipeline|double_buffered_pipeline|ConSanDeviceDoubleBufferedPipelineTest"
        "ReusedLdsGemmPipeline|reused_lds_gemm_pipeline|ConSanDeviceReusedLdsGemmPipelineTest"
        "MultiWaveGemmTile|multiwave_gemm_tile|ConSanDeviceMultiWaveGemmTileTest"
        "CooperativeLdsInitialization|cooperative_lds_initialization|ConSanDeviceCooperativeLdsInitializationTest"
        "SortNetworkExchange|sort_network_exchange|ConSanDeviceSortNetworkExchangeTest"
        "D16HighBytePublication|d16_high_byte_publication|ConSanDeviceD16HighBytePublicationTest"
        "HistogramScatter|histogram_scatter|ConSanDeviceHistogramScatterTest|atomics"
        "SegmentedTopK|segmented_topk|ConSanDeviceSegmentedTopKTest"
        "TwoStageSoftmax|two_stage_softmax|ConSanDeviceTwoStageSoftmaxTest"
        "BackwardOptimizer|backward_optimizer|ConSanDeviceBackwardOptimizerTest"
        "DeterministicStateReplay|deterministic_state_replay|ConSanDeviceDeterministicStateReplayTest"
        "MoeRouting|moe_routing|ConSanDeviceMoeRoutingTest|atomics"
        "ContinuousBatching|continuous_batching|ConSanDeviceContinuousBatchingTest"
        "HeterogeneousObject|heterogeneous_object|ConSanDeviceHeterogeneousObjectTest|atomics"
        "VglobalCachePublication|vglobal_cache_publication|ConSanDeviceVglobalCachePublicationTest|atomics"
        "FenceBarrierPublication|fence_barrier_publication|ConSanDeviceFenceBarrierPublicationTest|atomics"
        "GroupFlatWide|group_flat_wide|ConSanDeviceGroupFlatWideTest"
        "StridedDualAddressPublication|strided_dual_address_publication|ConSanDeviceStridedDualAddressPublicationTest"
        "LongRangeLiveScc|long_range_live_scc|ConSanDeviceLongRangeLiveSccTest"
        "MultiRangeBudget|multi_range_budget|ConSanDeviceMultiRangeBudgetTest|default-only"
        "LargeTextRelay|large_text_relay|ConSanDeviceLargeTextRelayTest"
        "TopKPrefix|topk_prefix|ConSanDeviceTopKPrefixTest|atomics"
        "SubwordSccRoute|subword_scc_route|ConSanDeviceSubwordSccRouteTest"
        "ClobberingLdsReduction|clobbering_lds_reduction|ConSanDeviceClobberingLdsReductionTest"
        "FullBankStreamK|full_bank_streamk|ConSanDeviceFullBankStreamKTest|atomics"
        "LongRangeFullPressure|long_range_full_pressure|ConSanDeviceLongRangeFullPressureTest|default-only"
        "SparseDensePipeline|sparse_dense_pipeline|ConSanDeviceSparseDensePipelineTest|default-only"
        "SparseSpillPressure|sparse_spill_pressure|ConSanDeviceSparseSpillPressureTest|default-only"
        "DenseBranchOnlyPressure|sparse_spill_pressure|ConSanDeviceDenseBranchOnlyPressureTest|default-only"
        "ManyOwnerPressure|many_owner_pressure|ConSanDeviceManyOwnerPressureTest|default-only"
        "SuperColliderDenseThreshold|supercollider_dense_threshold|ConSanDeviceSuperColliderDenseThresholdTest|supercollider-only"
        "IndependentScalarProofs|independent_scalar_proofs|ConSanDeviceIndependentScalarProofsTest|default-only"
    )
    if(_device_TARGET STREQUAL "gfx942" OR _device_TARGET STREQUAL "gfx950")
        list(
            APPEND _device_workloads
            "CdnaMfmaPipeline|cdna_mfma_pipeline|ConSanDeviceCdnaMfmaPipelineTest"
            "CdnaDirectToLdsPublication|cdna_direct_to_lds_publication|ConSanDeviceCdnaDirectToLdsPublicationTest"
            "CdnaB96Boundary|cdna_b96_boundary|ConSanDeviceCdnaB96BoundaryTest"
            "CdnaDenseSccRoutes|cdna_dense_scc_routes|ConSanDeviceCdnaDenseSccRoutesTest"
            "CdnaAccvgprB128|cdna_accvgpr_b128|ConSanDeviceCdnaAccvgprB128Test"
            "CdnaMixedPersistentState|cdna_mixed_persistent_state|ConSanDeviceCdnaMixedPersistentStateTest"
            "CdnaKernargPreloadPrivateState|cdna_kernarg_preload_private_state|ConSanDeviceCdnaMixedPersistentStateTest|default-only"
        )
    endif()
    if(_device_TARGET STREQUAL "gfx950")
        list(
            APPEND _device_workloads
            "DispatchIdSgprPressure|dispatch_id_sgpr_pressure|ConSanDeviceDispatchIdSgprPressureTest|default-only"
            "TwoQueueDispatchIdentity|two_queue_dispatch_identity|ConSanDeviceTwoQueueDispatchIdentityTest|default-only"
            "CdnaTransposePublication|cdna_transpose_publication|ConSanDeviceCdnaTransposePublicationTest"
            "Gfx950LargeLdsPipeline|gfx950_large_lds_pipeline|ConSanDeviceGfx950LargeLdsPipelineTest"
            "HighLdsByteAddress|high_lds_byte_address|ConSanDeviceHighLdsByteAddressTest"
            "HighLdsGenerationRollover|high_lds_byte_address|ConSanDeviceHighLdsGenerationRolloverTest|default-only"
        )
    endif()
    if(
        _device_TARGET STREQUAL "gfx1100"
        OR _device_TARGET STREQUAL "gfx1201"
        OR _device_TARGET STREQUAL "gfx1250"
    )
        list(
            APPEND _device_workloads
            "RdnaB96Boundary|rdna_b96_boundary|ConSanDeviceRdnaB96BoundaryTest"
        )
    endif()
    if(_device_TARGET STREQUAL "gfx1201" OR _device_TARGET STREQUAL "gfx1250")
        list(
            APPEND _device_workloads
            "OrderedAtomicScope|ordered_atomic_scope|ConSanDeviceOrderedAtomicScopeTest|atomics"
        )
    endif()
    if(_device_TARGET STREQUAL "gfx1250")
        list(
            APPEND _device_workloads
            "DispatchIdSgprPressure|dispatch_id_sgpr_pressure|ConSanDeviceDispatchIdSgprPressureTest|default-only"
            "Gfx1250ClusterSync|gfx1250_cluster|ConSanDeviceGfx1250ClusterSyncTest"
            "Gfx1250MultiCluster|gfx1250_cluster|ConSanDeviceGfx1250MultiClusterTest"
            "Gfx1250WideCluster|gfx1250_cluster|ConSanDeviceGfx1250WideClusterTest"
            "Gfx1250TdmWait|gfx1250_cluster|ConSanDeviceGfx1250TdmWaitTest"
            "Gfx1250ClusterMulticast|gfx1250_cluster|ConSanDeviceGfx1250ClusterMulticastTest"
            "Gfx1250HighBankLdsAddress|gfx1250_high_bank_lds_address|ConSanDeviceGfx1250HighBankLdsAddressTest"
            "Gfx1250SparseFp8Pipeline|gfx1250_sparse_fp8_pipeline|ConSanDeviceGfx1250SparseFp8PipelineTest"
            "Gfx1250ScaledWmmaPipeline|gfx1250_scaled_wmma_pipeline|ConSanDeviceGfx1250ScaledWmmaPipelineTest"
            "Gfx1250Fp16WmmaPipeline|gfx1250_fp16_wmma_pipeline|ConSanDeviceGfx1250Fp16WmmaPipelineTest"
            "Gfx1250GlobalAsyncToLds|gfx1250_global_async_to_lds|ConSanDeviceGfx1250GlobalAsyncToLdsTest"
            "Gfx1250GlobalAsyncFromLds|gfx1250_global_async_from_lds|ConSanDeviceGfx1250GlobalAsyncFromLdsTest"
            "Gfx1250TensorDescriptorPipeline|gfx1250_tensor_descriptor_pipeline|ConSanDeviceGfx1250TensorDescriptorPipelineTest"
            "Gfx1250LargeLdsPipeline|gfx950_large_lds_pipeline|ConSanDeviceGfx950LargeLdsPipelineTest"
            "HighLdsByteAddress|high_lds_byte_address|ConSanDeviceHighLdsByteAddressTest"
            "HighLdsGenerationRollover|high_lds_byte_address|ConSanDeviceHighLdsGenerationRolloverTest|default-only"
        )
    endif()
    if(_device_TARGET STREQUAL "gfx1201")
        list(
            APPEND _device_workloads
            "RdnaFp8StagedMatmul|rdna_fp8_staged_matmul|ConSanDeviceRdnaFp8StagedMatmulTest"
        )
    endif()
    if(_device_TARGET STREQUAL "gfx1100" OR _device_TARGET STREQUAL "gfx1201")
        list(
            APPEND _device_workloads
            "RdnaWmmaPipeline|rdna_wmma_pipeline|ConSanDeviceRdnaWmmaPipelineTest"
        )
    endif()
    if(_device_TARGET STREQUAL "gfx1100")
        list(
            APPEND _device_workloads
            "Rdna3ScalarVglobal|rdna3_scalar_vglobal|ConSanDeviceRdna3ScalarVglobalTest|atomics"
        )
    endif()
    # Hardware scheduling regression: SuperCollider borrows a pending WMMA
    # accumulator under full VGPR pressure. The emulator does not model this
    # matrix-to-VMEM latency hazard.
    if(_device_TARGET STREQUAL "gfx1201" AND NOT _device_CONFIG)
        foreach(_spill_mode IN ITEMS Baseline SuperCollider)
            set(_spill_environment)
            string(TOLOWER "${_spill_mode}" _spill_label)
            if(_spill_mode STREQUAL "SuperCollider")
                set(
                    _spill_environment
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=supercollider"
                    "RJ_CONSAN_POLICY=strict"
                    "RJ_CONSAN_REQUIRE_PATCH=1"
                    "RJ_CONSAN_LOG=1"
                )
            endif()
            rj_add_consan_hip_case(
                NAME "${_device_PREFIX}.${_spill_mode}RdnaWmmaSpill.Correct"
                TARGET hip_consan_device_rdna_wmma_spill_gfx1201_test
                FILTER "ConSanDeviceRdnaWmmaSpillTest.Correct"
                ${_resource_lock_args}
                TIMEOUT 60
                ENVIRONMENT ${_spill_environment}
                LABELS ${_labels} ${_spill_label} clean
            )
            if(_spill_mode STREQUAL "SuperCollider")
                list(APPEND _instrumented_tests
                    "${_device_PREFIX}.${_spill_mode}RdnaWmmaSpill.Correct"
                )
            endif()
        endforeach()
    endif()
    foreach(_workload IN LISTS _device_workloads)
        string(REPLACE "|" ";" _workload_fields "${_workload}")
        list(GET _workload_fields 0 _workload_name)
        list(GET _workload_fields 1 _workload_target)
        list(GET _workload_fields 2 _workload_suite)
        set(_baseline_workload_timeout 60)
        if(_workload_name STREQUAL "ModuleLifecycle")
            # The workload is sub-second in isolation, but a simulator process
            # can exceed the default during teardown under the full -j16 gate.
            set(_baseline_workload_timeout 120)
        elseif(_workload_name STREQUAL "KernargPreloadDynamicPrivateStack")
            # This workload is sub-second in isolation, but the gfx1250
            # simulator can exceed the default under the full -j16 gate.
            set(_baseline_workload_timeout 120)
        elseif(_workload_name STREQUAL "RcclPartialBarrier")
            # This workload is sub-second in isolation but can exceed the
            # default while the full -j16 gate runs the large LDS pipelines.
            set(_baseline_workload_timeout 120)
        elseif(_workload_name STREQUAL "FenceBarrierPublication")
            # The body is sub-second in isolation, but a simulator process can
            # exceed the default during teardown under a broad -j16 gate.
            set(_baseline_workload_timeout 120)
        endif()
        foreach(_outcome IN ITEMS Correct Incorrect)
            string(TOLOWER "${_outcome}" _outcome_label)
            rj_add_consan_hip_case(
                NAME
                    "${_device_PREFIX}.Baseline${_workload_name}.${_outcome}"
                TARGET
                    hip_consan_device_${_workload_target}_${_device_TARGET}_test
                FILTER "${_workload_suite}.${_outcome}"
                ${_config_args}
                ${_resource_lock_args}
                TIMEOUT ${_baseline_workload_timeout}
                LABELS ${_labels} baseline "${_outcome_label}"
            )
        endforeach()
    endforeach()

    set(_instrumented_tests)
    foreach(
        _mode
        IN
        ITEMS
            "SuperCollider|supercollider"
            "ConSan|default"
    )
        string(REPLACE "|" ";" _mode_fields "${_mode}")
        list(GET _mode_fields 0 _mode_name)
        list(GET _mode_fields 1 _mode_value)

        foreach(_workload IN LISTS _device_workloads)
            string(REPLACE "|" ";" _workload_fields "${_workload}")
            list(GET _workload_fields 0 _workload_name)
            list(GET _workload_fields 1 _workload_target)
            list(GET _workload_fields 2 _workload_suite)
            set(_workload_timeout 60)
            if(
                _mode_value STREQUAL "supercollider"
                AND _device_TARGET STREQUAL "gfx1250"
                AND (
                    _workload_name STREQUAL "D16HighBytePublication"
                    OR _workload_name STREQUAL "Reduction"
                )
            )
                # These are sub-second in isolation but can exceed the default
                # under the full -j16 simulator gate's gfx1250 contention.
                set(_workload_timeout 120)
            elseif(
                _mode_value STREQUAL "default"
                AND _workload_name STREQUAL "AtomicCasPublication"
            )
                # This workload is sub-second in isolation but can exceed the
                # default while the full -j16 gate runs the large LDS pipelines.
                set(_workload_timeout 120)
            endif()
            list(LENGTH _workload_fields _workload_field_count)
            if(_workload_field_count GREATER 3)
                list(GET _workload_fields 3 _workload_feature)

                if(
                    _workload_feature STREQUAL "default-only"
                    AND NOT _mode_value STREQUAL "default"
                )
                    continue()
                endif()
                if(
                    _workload_feature STREQUAL "supercollider-only"
                    AND NOT _mode_value STREQUAL "supercollider"
                )
                    continue()
                endif()
                if(
                    _workload_feature STREQUAL "default-only"
                    AND _mode_value STREQUAL "supercollider"
                )
                    continue()
                endif()
            endif()

            set(_common_environment
                "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                "RJ_CONSAN_MODE=${_mode_value}"
                "RJ_CONSAN_POLICY=strict"
                "RJ_CONSAN_REQUIRE_PATCH=1"
                "RJ_CONSAN_LOG=1"
            )
            set(_correct_environment ${_common_environment})
            set(_incorrect_environment ${_common_environment})
            if(
                _mode_value STREQUAL "default"
                AND _workload_target STREQUAL "gfx1250_cluster"
            )
                # These fixtures require a retained within-workgroup witness.
                # Cluster-local IDs occupy hash bits 3..4: eight banks alias
                # them, allowing different workgroups to win at each site.
                # 32 banks distinguish all identities in these fixtures (up
                # to four workgroups per cluster, or two clusters of two).
                list(
                    APPEND _correct_environment
                    "RJ_CONSAN_WATCHPOINT_BANKS=32"
                )
                list(
                    APPEND _incorrect_environment
                    "RJ_CONSAN_WATCHPOINT_BANKS=32"
                )
            endif()
            set(_correct_pass_regex_args)
            set(_incorrect_pass_regex_args)
            if(NOT _mode_value STREQUAL "supercollider")
                list(
                    APPEND _correct_environment
                    "RJ_CONSAN_TRACK_BARRIERS=1"
                    "RJ_CONSAN_REQUIRE_RECORDS=1"
                    "RJ_CONSAN_FORBID_DIAGNOSTICS=1"
                    "RJ_CONSAN_FORBID_OVERFLOW=1"
                )
                list(
                    APPEND _incorrect_environment
                    "RJ_CONSAN_TRACK_BARRIERS=1"
                    "RJ_CONSAN_REQUIRE_RECORDS=1"
                    "RJ_CONSAN_FORBID_OVERFLOW=1"
                )
                if(
                    NOT (
                        _mode_value STREQUAL "default"
                        AND _workload_name STREQUAL "TwoQueueDispatchIdentity"
                    )
                )
                    list(
                        APPEND _incorrect_environment
                        "RJ_CONSAN_REQUIRE_DIAGNOSTICS=1"
                    )
                endif()
            endif()

            if(_workload_name STREQUAL "TwoQueueDispatchIdentity")
                # The fixture itself asserts two distinct HSA queues, matching
                # queue-local IDs in two rounds, and both reused and distinct
                # kernarg allocations. Keep those launch preconditions visible
                # in the CTest contract instead of relying on the test name.
                list(
                    APPEND _correct_pass_regex_args
                    PASS_REGEX "raw_dispatch_ids=1,1 same_kernel=1 same_kernarg=0 round=1"
                )
                list(
                    APPEND _incorrect_pass_regex_args
                    PASS_REGEX "raw_dispatch_ids=1,1 same_kernel=1 same_kernarg=0 round=1"
                )

            endif()

            # HIP does not expose its stream-to-HSA-queue mapping to this
            # fixture. Keep the completed launch envelope observable without
            # presenting streams as queue evidence.
            set(_hip_launch_count)
            if(_workload_name STREQUAL "RepeatedDispatchIdentity")
                set(_hip_launch_count 8)
            elseif(_workload_name STREQUAL "GraphReplay")
                set(_hip_launch_count 6)
            elseif(_workload_name STREQUAL "GraphParameterUpdate")
                set(_hip_launch_count 2)
            endif()
            if(_hip_launch_count)
                set(
                    _hip_integration_regex
                    "CONSAN_HIP_INTEGRATION workload=${_workload_target} completed_launches=${_hip_launch_count} underlying_hsa_queues=unobserved raw_dispatch_id_reuse=unobserved"
                )
                list(
                    APPEND _correct_pass_regex_args
                    PASS_REGEX "${_hip_integration_regex}"
                )
                list(
                    APPEND _incorrect_pass_regex_args
                    PASS_REGEX "${_hip_integration_regex}"
                )

            endif()

            if(_workload_name STREQUAL "HistogramScatter")
                # The behavioral oracle depends on the explicit native LDS
                # publication helpers in this kernel. Require its normalized
                # sites to survive compilation, admission, placement, and
                # final validation instead of accepting an empty patch set
                # from unrelated runtime kernels in the same object.
                list(
                    APPEND _correct_environment
                    "RJ_CONSAN_TEST_KERNEL_FILTER=histogram_scatter_for_consan"
                )
                list(
                    APPEND _incorrect_environment
                    "RJ_CONSAN_TEST_KERNEL_FILTER=histogram_scatter_for_consan"
                )
                set(
                    _histogram_coverage_regex
                    "access_discovered=[1-9][0-9]* access_supported=[1-9][0-9]* access_selected=[1-9][0-9]* access_patched=[1-9][0-9]*"
                )
                list(
                    APPEND _correct_pass_regex_args
                    PASS_REGEX "${_histogram_coverage_regex}"
                )
                list(
                    APPEND _incorrect_pass_regex_args
                    PASS_REGEX "${_histogram_coverage_regex}"
                )
            endif()

            if(_workload_field_count GREATER 3)
                list(GET _workload_fields 3 _workload_feature)
                if(_workload_feature STREQUAL "atomics")
                    list(
                        APPEND _correct_environment
                        "RJ_CONSAN_TRACK_ATOMICS=1"
                    )
                    list(
                        APPEND _incorrect_environment
                        "RJ_CONSAN_TRACK_ATOMICS=1"
                    )
                endif()
            endif()

            if(
                (
                    _workload_name STREQUAL "SparseDensePipeline"
                    OR _workload_name STREQUAL "SparseSpillPressure"
                    OR _workload_name STREQUAL "DenseBranchOnlyPressure"
                )
                AND _mode_value STREQUAL "default"
            )
                # Keep one workgroup selected while exercising the cached
                # sparse workgroup predicate distilled from Qwen/TP2. This is
                # deliberately not the dense stride pinned by the generic
                # checked-in fixtures.
                list(
                    APPEND _correct_environment
                    "RJ_CONSAN_RUNTIME_SAMPLE_STRIDE=2"
                )
                list(
                    APPEND _incorrect_environment
                    "RJ_CONSAN_RUNTIME_SAMPLE_STRIDE=2"
                )
            endif()

            if(
                _workload_name STREQUAL "IndependentScalarProofs"
                AND _mode_value STREQUAL "default"
            )
                # Exercise persistent dispatch identity together with the two
                # independent scalar-pressure owners in this workload. One
                # workgroup at offset zero remains selected by this stride.
                list(
                    APPEND _correct_environment
                    "RJ_CONSAN_RUNTIME_SAMPLE_STRIDE=2"
                )
                list(
                    APPEND _incorrect_environment
                    "RJ_CONSAN_RUNTIME_SAMPLE_STRIDE=2"
                )
            endif()

            if(
                _workload_name STREQUAL "CdnaKernargPreloadPrivateState"
                AND _mode_value STREQUAL "default"
            )
                # Qwen's IREE kernels use AMDHSA's +256 kernarg-preload entry.
                # Pin stride one so this contract has no incidental dispatch-ID
                # consumer: the private-state prologue must mirror both entries
                # on its own. The paired result and diagnostic oracles, together
                # with require-patch, validate that behavior without constraining
                # which register-allocation mechanism implements the state.
                list(
                    APPEND _correct_environment
                    "RJ_CONSAN_RUNTIME_SAMPLE_STRIDE=1"
                )
                list(
                    APPEND _incorrect_environment
                    "RJ_CONSAN_RUNTIME_SAMPLE_STRIDE=1"
                )
            endif()

            if(
                _workload_name STREQUAL "MultiRangeBudget"
                AND _mode_value STREQUAL "default"
            )
                # This contract distinguishes the two selected instrumentation
                # sites from the multiple banked report slots consumed by
                # each dual-address LDS instruction.
                list(APPEND _correct_environment "RJ_CONSAN_MAX_PATCHES=2")
                list(APPEND _incorrect_environment "RJ_CONSAN_MAX_PATCHES=2")
            endif()

            if(
                _workload_name STREQUAL "ModuleLifecycle"
                AND NOT _mode_value STREQUAL "supercollider"
            )
                # Four initial objects plus alternating odd/even reloads must
                # all contribute complete dynamic evidence. This count is the
                # workload's lifecycle contract, not a patch-layout detail.
                set(
                    _module_lifecycle_coverage_regex
                    "applicable_code_objects=8 incomplete_code_objects=0 access=32/32"
                )
                list(
                    APPEND _correct_pass_regex_args
                    PASS_REGEX "${_module_lifecycle_coverage_regex}"
                )
                list(
                    APPEND _incorrect_pass_regex_args
                    PASS_REGEX "${_module_lifecycle_coverage_regex}"
                )
            endif()

            set(_correct_test_name
                "${_device_PREFIX}.${_mode_name}${_workload_name}.Correct"
            )
            rj_add_consan_hip_case(
                NAME "${_correct_test_name}"
                TARGET
                    hip_consan_device_${_workload_target}_${_device_TARGET}_test
                FILTER "${_workload_suite}.Correct"
                ${_config_args}
                ${_resource_lock_args}
                TIMEOUT ${_workload_timeout}
                ${_correct_pass_regex_args}
                ENVIRONMENT ${_correct_environment}
                LABELS ${_labels} "${_mode_value}" clean
            )
            list(APPEND _instrumented_tests "${_correct_test_name}")

            set(_incorrect_test_name
                "${_device_PREFIX}.${_mode_name}${_workload_name}.Incorrect"
            )
            set(_incorrect_labels ${_labels} "${_mode_value}" incorrect racy)
            if(_mode_value STREQUAL "supercollider")
                # SuperCollider is a mutation-only mode with no
                # happens-before model. Its explicit negative contract is
                # successful instrumentation and preservation of the
                # independent workload oracle, not a semantic diagnosis.
                list(APPEND _incorrect_labels mutation-only)
            endif()
            rj_add_consan_hip_case(
                NAME "${_incorrect_test_name}"
                TARGET
                    hip_consan_device_${_workload_target}_${_device_TARGET}_test
                FILTER "${_workload_suite}.Incorrect"
                ${_config_args}
                ${_resource_lock_args}
                TIMEOUT ${_workload_timeout}
                ${_incorrect_pass_regex_args}
                ENVIRONMENT ${_incorrect_environment}
                LABELS ${_incorrect_labels}
            )
            list(APPEND _instrumented_tests "${_incorrect_test_name}")
        endforeach()
    endforeach()

    if(_device_TARGET STREQUAL "gfx942" OR _device_TARGET STREQUAL "gfx950")
        # Exercise the compiler's CDNA VGLOBAL release sequence, not only the
        # FLAT encoding used by structural unit fixtures. The incorrect member
        # starts from the same correct device workload and removes exactly its
        # producer release edge; ConSan must then diagnose the consumer's
        # LDS reads.
        set(_atomic_order_common_environment
            "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
            "RJ_CONSAN_MODE=default"
            "RJ_CONSAN_POLICY=strict"
            "RJ_CONSAN_TRACK_ATOMICS=1"
            "RJ_CONSAN_REQUIRE_RECORDS=1"
            "RJ_CONSAN_FORBID_OVERFLOW=1"
            "RJ_CONSAN_TEST_KERNEL_FILTER=atomic_arrival_for_consan"
            "RJ_CONSAN_LOG=1"
            "RJ_CONSAN_REQUIRE_PATCH=1"
        )
        set(_atomic_order_correct_test_name
            "${_device_PREFIX}.GlobalAtomicOrder.Correct"
        )
        rj_add_consan_hip_case(
            NAME "${_atomic_order_correct_test_name}"
            TARGET hip_consan_device_atomic_arrival_${_device_TARGET}_test
            FILTER "ConSanDeviceAtomicArrivalTest.Correct"
            ${_config_args}
            ${_resource_lock_args}
            TIMEOUT 60
            PASS_REGEX "visible_sync=[1-9][0-9]*"
            ENVIRONMENT
                ${_atomic_order_common_environment}
                "RJ_CONSAN_FORBID_DIAGNOSTICS=1"
            LABELS ${_labels} default fault clean
        )
        list(APPEND _instrumented_tests "${_atomic_order_correct_test_name}")

        set(_atomic_order_incorrect_test_name
            "${_device_PREFIX}.GlobalAtomicOrder.Incorrect"
        )
        rj_add_consan_hip_case(
            NAME "${_atomic_order_incorrect_test_name}"
            TARGET hip_consan_device_atomic_arrival_${_device_TARGET}_test
            FILTER "ConSanDeviceAtomicArrivalTest.Correct"
            ${_config_args}
            ${_resource_lock_args}
            TIMEOUT 60
            PASS_REGEX "conflicts=[1-9][0-9]*"
            FAIL_REGEX
                "RJ_CONSAN_REQUIRE_DIAGNOSTICS requested|mutation_installed=false"
            ENVIRONMENT
                ${_atomic_order_common_environment}
                "RJ_CONSAN_REQUIRE_DIAGNOSTICS=1"
                "RJ_CONSAN_FAULT_ATOMIC_WEAKEN_ORDER=1"
                "RJ_CONSAN_FAULT_ATOMIC_ORDER_EDGE=release"
                "RJ_CONSAN_FAULT_ATOMIC_INDEX=1"
                "RJ_CONSAN_FAULT_REQUIRE_EXACTLY_ONE=1"
            LABELS ${_labels} default fault incorrect
        )
        list(APPEND _instrumented_tests "${_atomic_order_incorrect_test_name}")
    endif()

    if(_device_PHYSICAL)
        set(_health_test_name "${_device_PREFIX}.PostInstrumentationHealth")
        rj_add_consan_hip_case(
            NAME "${_health_test_name}"
            TARGET hip_consan_device_ordered_tile_handoff_${_device_TARGET}_test
            FILTER "ConSanDeviceOrderedTileHandoffTest.Correct"
            ${_resource_lock_args}
            TIMEOUT 60
            LABELS ${_labels} health
        )
        set_tests_properties(
            "${_health_test_name}"
            PROPERTIES DEPENDS "${_instrumented_tests}"
        )
    endif()
endfunction()

function(rj_add_consan_device_module_lifecycle_fixture target)
    set(_module_source
        "${CMAKE_CURRENT_SOURCE_DIR}/dbi/consan/device/module_lifecycle_kernels.hip"
    )
    find_program(
        _module_offload_bundler
        clang-offload-bundler
        REQUIRED
        HINTS
            "${ROCM_PATH}/lib/llvm/bin"
            "${ROCM_PATH}/llvm/bin"
            "${ROCM_PATH}/bin"
    )
    set(_module_compiler_command ${AMDCXX})
    if(CMAKE_HIP_COMPILER_LAUNCHER)
        list(PREPEND _module_compiler_command ${CMAKE_HIP_COMPILER_LAUNCHER})
    endif()

    set(_module_hsacos)
    foreach(_variant IN ITEMS 1 2 3 4)
        set(_module_bundle
            "${CMAKE_CURRENT_BINARY_DIR}/hip_consan_device_module_lifecycle_${target}_${_variant}.bundle"
        )
        set(_module_hsaco
            "${CMAKE_CURRENT_BINARY_DIR}/hip_consan_device_module_lifecycle_${target}_${_variant}.hsaco"
        )
        add_custom_command(
            OUTPUT "${_module_bundle}"
            COMMAND
                ${_module_compiler_command} -x hip --cuda-device-only
                --offload-arch=${target} --rocm-path=${ROCM_PATH} -std=c++20 -O2
                -DRJ_CONSAN_MODULE_VARIANT=${_variant} -o "${_module_bundle}"
                "${_module_source}"
            DEPENDS "${_module_source}"
            COMMENT
                "Compiling ${target} module-lifecycle ConSan kernels (object ${_variant})"
            VERBATIM
        )
        add_custom_command(
            OUTPUT "${_module_hsaco}"
            COMMAND
                "${_module_offload_bundler}" --type=o --unbundle
                --targets=hipv4-amdgcn-amd-amdhsa--${target}
                --input=${_module_bundle} --output=${_module_hsaco}
            DEPENDS "${_module_bundle}"
            COMMENT
                "Extracting ${target} module-lifecycle ConSan code object ${_variant}"
            VERBATIM
        )
        list(APPEND _module_hsacos "${_module_hsaco}")
    endforeach()
    list(GET _module_hsacos 0 _module_hsaco_a)
    list(GET _module_hsacos 1 _module_hsaco_b)
    list(GET _module_hsacos 2 _module_hsaco_c)
    list(GET _module_hsacos 3 _module_hsaco_d)
    add_custom_target(
        hip_consan_device_module_lifecycle_${target}_kernels_target
        ALL
        DEPENDS ${_module_hsacos}
    )
    rj_add_hip_test(
        hip_consan_device_module_lifecycle_${target}_test
        ARCH ${target}
        SOURCE dbi/consan/device/module_lifecycle_test.hip
        COMPILE_DEFINITIONS
            "RJ_CONSAN_MODULE_LIFECYCLE_HSACO_A=\"${_module_hsaco_a}\""
            "RJ_CONSAN_MODULE_LIFECYCLE_HSACO_B=\"${_module_hsaco_b}\""
            "RJ_CONSAN_MODULE_LIFECYCLE_HSACO_C=\"${_module_hsaco_c}\""
            "RJ_CONSAN_MODULE_LIFECYCLE_HSACO_D=\"${_module_hsaco_d}\""
        EXTRA_DEPENDS "${_module_hsacos}"
    )
    add_dependencies(
        hip_consan_device_module_lifecycle_${target}_test_target
        hip_consan_device_module_lifecycle_${target}_kernels_target
        rocjitsu_dbi_hooks
    )
endfunction()

function(rj_add_consan_device_fixture_targets target)
    rj_add_consan_device_module_lifecycle_fixture(${target})
    set(_device_workload_sources
        "selection|selection_test.hip"
        "epoch_exhaustion|epoch_exhaustion_test.hip"
        "ordered_tile_handoff|ordered_tile_handoff_test.hip"
        "reduction|reduction_test.hip"
        "shared_helper_owners|shared_helper_owners_test.hip"
        "independent_workgroups|independent_workgroups_test.hip"
        "repeated_dispatch_identity|repeated_dispatch_identity_test.hip"
        "cross_kernel_dispatch_identity|cross_kernel_dispatch_identity_test.hip"
        "graph_replay|graph_replay_test.hip"
        "graph_parameter_update|graph_parameter_update_test.hip"
        "dynamic_private_stack|dynamic_private_stack_test.hip"
        "kernarg_preload_dynamic_private_stack|dynamic_private_stack_test.hip"
        "mixed_private_owners|mixed_private_owners_test.hip"
        "overlapping_subwords|overlapping_subwords_test.hip"
        "adjacent_subword_writers|adjacent_subword_writers_test.hip"
        "indexed_lds_alias|indexed_lds_alias_test.hip"
        "atomic_arrival|atomic_arrival_test.hip"
        "atomic_cas_publication|atomic_cas_publication_test.hip"
        "atomic_store_publication|atomic_store_publication_test.hip"
        "flat_atomic_publication|flat_atomic_publication_test.hip"
        "streamk_last_arriver|streamk_last_arriver_test.hip"
        "rccl_partial_barrier|rccl_partial_barrier_test.hip"
        "tree_atomic_or|tree_atomic_or_test.hip"
        "access_heavy_streamk|access_heavy_streamk_test.hip"
        "double_buffered_pipeline|double_buffered_pipeline_test.hip"
        "reused_lds_gemm_pipeline|reused_lds_gemm_pipeline_test.hip"
        "multiwave_gemm_tile|multiwave_gemm_tile_test.hip"
        "cooperative_lds_initialization|cooperative_lds_initialization_test.hip"
        "sort_network_exchange|sort_network_exchange_test.hip"
        "d16_high_byte_publication|d16_high_byte_publication_test.hip"
        "histogram_scatter|histogram_scatter_test.hip"
        "segmented_topk|segmented_topk_test.hip"
        "two_stage_softmax|two_stage_softmax_test.hip"
        "backward_optimizer|backward_optimizer_test.hip"
        "deterministic_state_replay|deterministic_state_replay_test.hip"
        "moe_routing|moe_routing_test.hip"
        "continuous_batching|continuous_batching_test.hip"
        "heterogeneous_object|heterogeneous_object_test.hip"
        "vglobal_cache_publication|vglobal_cache_publication_test.hip"
        "fence_barrier_publication|vglobal_cache_publication_test.hip"
        "group_flat_wide|group_flat_wide_test.hip"
        "strided_dual_address_publication|strided_dual_address_publication_test.hip"
        "long_range_live_scc|long_range_live_scc_test.hip"
        "multi_range_budget|multi_range_budget_test.hip"
        "large_text_relay|cdna_wave64_large_text_test.hip"
        "topk_prefix|wave32_topk_prefix_test.hip"
        "subword_scc_route|cdna_b16_scc_route_test.hip"
        "clobbering_lds_reduction|cdna_clobbering_lds_reduction_test.hip"
        "full_bank_streamk|cdna_full_bank_streamk_test.hip"
        "long_range_full_pressure|cdna_long_range_full_pressure_test.hip"
        "sparse_dense_pipeline|sparse_dense_pipeline_test.hip"
        "sparse_spill_pressure|sparse_spill_pressure_test.hip"
        "sampling_reduction|sampling_reduction_test.hip"
        "many_owner_pressure|sparse_spill_pressure_test.hip"
        "supercollider_dense_threshold|cdna_supercollider_dense_threshold_test.hip"
        "independent_scalar_proofs|cdna_independent_scalar_proofs_test.hip"
    )
    if(target STREQUAL "gfx942" OR target STREQUAL "gfx950")
        list(
            APPEND _device_workload_sources
            "cdna_mfma_pipeline|cdna_mfma_pipeline_test.hip"
            "cdna_direct_to_lds_publication|cdna_direct_to_lds_publication_test.hip"
            "cdna_b96_boundary|cdna_b96_boundary_test.hip"
            "cdna_dense_scc_routes|cdna_dense_scc_routes_test.hip"
            "cdna_accvgpr_b128|cdna_accvgpr_b128_test.hip"
            "cdna_mixed_persistent_state|cdna_mixed_persistent_state_test.hip"
            "cdna_kernarg_preload_private_state|cdna_mixed_persistent_state_test.hip"
        )
    endif()
    if(target STREQUAL "gfx950")
        list(
            APPEND _device_workload_sources
            "dispatch_id_sgpr_pressure|cdna_dispatch_id_sgpr_pressure_test.hip"
            "cdna_transpose_publication|cdna_transpose_publication_test.hip"
            "gfx950_large_lds_pipeline|gfx950_large_lds_pipeline_test.hip"
            "high_lds_byte_address|high_lds_byte_address_test.hip"
        )
    endif()
    if(
        target STREQUAL "gfx1100"
        OR target STREQUAL "gfx1201"
        OR target STREQUAL "gfx1250"
    )
        list(
            APPEND _device_workload_sources
            "rdna_b96_boundary|rdna_b96_boundary_test.hip"
        )
    endif()
    if(target STREQUAL "gfx1201" OR target STREQUAL "gfx1250")
        list(
            APPEND _device_workload_sources
            "ordered_atomic_scope|ordered_atomic_scope_test.hip"
        )
    endif()
    if(target STREQUAL "gfx1250")
        list(
            APPEND _device_workload_sources
            "dispatch_id_sgpr_pressure|cdna_dispatch_id_sgpr_pressure_test.hip"
            "gfx1250_high_bank_lds_address|gfx1250_high_bank_lds_address_test.hip"
            "gfx1250_sparse_fp8_pipeline|gfx1250_sparse_fp8_pipeline_test.hip"
            "gfx1250_scaled_wmma_pipeline|gfx1250_scaled_wmma_pipeline_test.hip"
            "gfx1250_fp16_wmma_pipeline|gfx1250_fp16_wmma_pipeline_test.hip"
            "gfx1250_global_async_to_lds|gfx1250_global_async_to_lds_test.hip"
            "gfx1250_global_async_from_lds|gfx1250_global_async_from_lds_test.hip"
            "gfx1250_tensor_descriptor_pipeline|gfx1250_tensor_descriptor_pipeline_test.hip"
            "gfx950_large_lds_pipeline|gfx950_large_lds_pipeline_test.hip"
            "high_lds_byte_address|high_lds_byte_address_test.hip"
        )
    endif()
    if(target STREQUAL "gfx1201")
        list(
            APPEND _device_workload_sources
            "rdna_fp8_staged_matmul|rdna_fp8_staged_matmul_test.hip"
            "rdna_wmma_spill|rdna_wmma_spill_test.hip"
        )
    endif()
    if(target STREQUAL "gfx1100" OR target STREQUAL "gfx1201")
        list(
            APPEND _device_workload_sources
            "rdna_wmma_pipeline|rdna_wmma_pipeline_test.hip"
        )
    endif()
    if(target STREQUAL "gfx1100")
        list(
            APPEND _device_workload_sources
            "rdna3_scalar_vglobal|rdna3_scalar_vglobal_test.hip"
        )
    endif()
    foreach(_workload IN LISTS _device_workload_sources)
        string(REPLACE "|" ";" _workload_fields "${_workload}")
        list(GET _workload_fields 0 _workload_name)
        list(GET _workload_fields 1 _workload_source)
        set(_target "hip_consan_device_${_workload_name}_${target}_test")
        set(_workload_compile_definitions)
        set(_workload_compile_options)
        if(_workload_name STREQUAL "fence_barrier_publication")
            list(
                APPEND _workload_compile_definitions
                RJ_CONSAN_DEVICE_FENCE_BARRIER_ONLY=1
            )
        endif()
        if(_workload_name STREQUAL "sampling_reduction")
            # Match the downstream reproducer's optimization level. Its small
            # static reduction shape is materially different at -O2/-O3.
            list(APPEND _workload_compile_options -O1 -g)
        endif()
        if(
            _workload_name STREQUAL "gfx950_large_lds_pipeline"
            AND target STREQUAL "gfx1250"
        )
            # HIP target built-ins are unavailable to the host compilation
            # pass. Keep the shared source's launcher constants identical to
            # its gfx1250 device constants with an explicit two-pass marker.
            list(
                APPEND _workload_compile_definitions
                RJ_CONSAN_LARGE_LDS_GFX1250=1
            )
        endif()
        if(
            _workload_name STREQUAL "large_text_relay"
            AND (
                target STREQUAL "gfx1100"
                OR target STREQUAL "gfx1201"
                OR target STREQUAL "gfx1250"
            )
        )
            list(
                APPEND _workload_compile_definitions
                RJ_CONSAN_LARGE_TEXT_WAVE32=1
            )
        endif()
        if(
            _workload_name STREQUAL "topk_prefix"
            AND (target STREQUAL "gfx942" OR target STREQUAL "gfx950")
        )
            list(
                APPEND _workload_compile_definitions
                RJ_CONSAN_TOPK_WAVE64=1
                RJ_CONSAN_TOPK_LEGACY_LDS=1
            )
        elseif(
            _workload_name STREQUAL "topk_prefix"
            AND target STREQUAL "gfx1100"
        )
            list(
                APPEND _workload_compile_definitions
                RJ_CONSAN_TOPK_LEGACY_LDS=1
            )
        endif()
        if(
            _workload_name STREQUAL "subword_scc_route"
            AND (
                target STREQUAL "gfx1100"
                OR target STREQUAL "gfx1201"
                OR target STREQUAL "gfx1250"
            )
        )
            list(
                APPEND _workload_compile_definitions
                RJ_CONSAN_SUBWORD_SCC_WAVE32=1
            )
        endif()
        if(
            _workload_name STREQUAL "subword_scc_route"
            AND (target STREQUAL "gfx1201" OR target STREQUAL "gfx1250")
        )
            list(
                APPEND _workload_compile_definitions
                RJ_CONSAN_SUBWORD_SCC_MODERN_LDS=1
            )
        endif()
        if(
            _workload_name STREQUAL "clobbering_lds_reduction"
            AND (
                target STREQUAL "gfx1100"
                OR target STREQUAL "gfx1201"
                OR target STREQUAL "gfx1250"
            )
        )
            list(
                APPEND _workload_compile_definitions
                RJ_CONSAN_CLOBBERING_REDUCTION_WAVE32=1
            )
        endif()
        if(
            _workload_name STREQUAL "clobbering_lds_reduction"
            AND (target STREQUAL "gfx1201" OR target STREQUAL "gfx1250")
        )
            list(
                APPEND _workload_compile_definitions
                RJ_CONSAN_CLOBBERING_REDUCTION_MODERN_LDS=1
            )
        endif()
        if(
            _workload_name STREQUAL "full_bank_streamk"
            AND (
                target STREQUAL "gfx1100"
                OR target STREQUAL "gfx1201"
                OR target STREQUAL "gfx1250"
            )
        )
            list(
                APPEND _workload_compile_definitions
                RJ_CONSAN_FULL_BANK_STREAMK_WAVE32=1
            )
        endif()
        if(
            _workload_name STREQUAL "long_range_full_pressure"
            AND (
                target STREQUAL "gfx1100"
                OR target STREQUAL "gfx1201"
                OR target STREQUAL "gfx1250"
            )
        )
            list(
                APPEND _workload_compile_definitions
                RJ_CONSAN_LDS_PRESSURE_WAVE32=1
            )
        endif()
        if(
            _workload_name STREQUAL "sparse_dense_pipeline"
            AND (
                target STREQUAL "gfx1100"
                OR target STREQUAL "gfx1201"
                OR target STREQUAL "gfx1250"
            )
        )
            list(
                APPEND _workload_compile_definitions
                RJ_CONSAN_SPARSE_DENSE_WAVE32=1
            )
        endif()
        if(
            (
                _workload_name STREQUAL "sparse_spill_pressure"
                OR _workload_name STREQUAL "many_owner_pressure"
            )
            AND (
                target STREQUAL "gfx1100"
                OR target STREQUAL "gfx1201"
                OR target STREQUAL "gfx1250"
            )
        )
            list(
                APPEND _workload_compile_definitions
                RJ_CONSAN_SPARSE_SPILL_WAVE32=1
            )
        endif()
        if(
            (
                _workload_name STREQUAL "sparse_spill_pressure"
                OR _workload_name STREQUAL "many_owner_pressure"
            )
            AND (target STREQUAL "gfx1201" OR target STREQUAL "gfx1250")
        )
            list(
                APPEND _workload_compile_definitions
                RJ_CONSAN_SPARSE_SPILL_MODERN_LDS=1
            )
        endif()
        if(
            (
                _workload_name STREQUAL "sparse_spill_pressure"
                OR _workload_name STREQUAL "many_owner_pressure"
            )
            AND target STREQUAL "gfx1250"
        )
            list(
                APPEND _workload_compile_definitions
                RJ_CONSAN_SPARSE_SPILL_GFX1250_COMPACT_SCALAR_PRESSURE=1
            )
        endif()
        if(_workload_name STREQUAL "many_owner_pressure")
            list(
                APPEND _workload_compile_definitions
                RJ_CONSAN_MANY_OWNER_ONLY=1
            )
        endif()
        if(
            _workload_name STREQUAL "sparse_dense_pipeline"
            AND (target STREQUAL "gfx1201" OR target STREQUAL "gfx1250")
        )
            list(
                APPEND _workload_compile_definitions
                RJ_CONSAN_SPARSE_DENSE_MODERN_LDS=1
            )
        endif()
        if(
            _workload_name STREQUAL "long_range_full_pressure"
            AND (target STREQUAL "gfx1201" OR target STREQUAL "gfx1250")
        )
            list(
                APPEND _workload_compile_definitions
                RJ_CONSAN_LDS_PRESSURE_MODERN_LDS=1
            )
        endif()
        if(
            _workload_name STREQUAL "supercollider_dense_threshold"
            AND (
                target STREQUAL "gfx1100"
                OR target STREQUAL "gfx1201"
                OR target STREQUAL "gfx1250"
            )
        )
            list(
                APPEND _workload_compile_definitions
                RJ_CONSAN_DENSE_THRESHOLD_WAVE32=1
            )
        endif()
        if(
            _workload_name STREQUAL "supercollider_dense_threshold"
            AND (target STREQUAL "gfx1201" OR target STREQUAL "gfx1250")
        )
            list(
                APPEND _workload_compile_definitions
                RJ_CONSAN_DENSE_THRESHOLD_MODERN_LDS=1
            )
        endif()
        if(
            _workload_name STREQUAL "independent_scalar_proofs"
            AND (
                target STREQUAL "gfx1100"
                OR target STREQUAL "gfx1201"
                OR target STREQUAL "gfx1250"
            )
        )
            list(
                APPEND _workload_compile_definitions
                RJ_CONSAN_INDEPENDENT_SCALAR_WAVE32=1
            )
        endif()
        if(
            _workload_name STREQUAL "independent_scalar_proofs"
            AND (target STREQUAL "gfx1201" OR target STREQUAL "gfx1250")
        )
            list(
                APPEND _workload_compile_definitions
                RJ_CONSAN_INDEPENDENT_SCALAR_MODERN_LDS=1
            )
        endif()
        if(
            _workload_name STREQUAL "large_text_relay"
            AND (target STREQUAL "gfx1201" OR target STREQUAL "gfx1250")
        )
            list(
                APPEND _workload_compile_definitions
                RJ_CONSAN_LARGE_TEXT_MODERN_LDS=1
            )
        endif()
        if(
            _workload_name STREQUAL "cdna_kernarg_preload_private_state"
            OR _workload_name STREQUAL "kernarg_preload_dynamic_private_stack"
        )
            # Build the same behavioral pair through the compiler's real
            # kernarg-preload ABI. One pointer argument is enough to select the
            # secondary +256 hardware entry on a preloaded dispatch.
            list(
                APPEND _workload_compile_options
                -mllvm
                -amdgpu-kernarg-preload-count=1
            )
        endif()
        rj_add_hip_test(
            ${_target}
            ARCH ${target}
            SOURCE "dbi/consan/device/${_workload_source}"
            COMPILE_DEFINITIONS ${_workload_compile_definitions}
            COMPILE_OPTIONS ${_workload_compile_options}
        )
        add_dependencies(${_target}_target rocjitsu_dbi_hooks)
    endforeach()

    if(target STREQUAL "gfx950")
        set(_two_queue_kernel_source
            "${CMAKE_CURRENT_SOURCE_DIR}/dbi/consan/device/gfx950_two_queue_dispatch_identity_kernels.hip"
        )
        set(_two_queue_hsaco
            "${CMAKE_CURRENT_BINARY_DIR}/hip_consan_device_gfx950_two_queue_dispatch_identity.hsaco"
        )
        set(_two_queue_bundle
            "${CMAKE_CURRENT_BINARY_DIR}/hip_consan_device_gfx950_two_queue_dispatch_identity.bundle"
        )
        find_program(
            _two_queue_offload_bundler
            clang-offload-bundler
            REQUIRED
            HINTS
                "${ROCM_PATH}/lib/llvm/bin"
                "${ROCM_PATH}/llvm/bin"
                "${ROCM_PATH}/bin"
        )
        set(_two_queue_compiler_command ${AMDCXX})
        if(CMAKE_HIP_COMPILER_LAUNCHER)
            list(
                PREPEND _two_queue_compiler_command
                ${CMAKE_HIP_COMPILER_LAUNCHER}
            )
        endif()
        add_custom_command(
            OUTPUT "${_two_queue_bundle}"
            COMMAND
                ${_two_queue_compiler_command} -x hip --cuda-device-only
                --offload-arch=gfx950 --rocm-path=${ROCM_PATH} -std=c++20 -O2
                -o "${_two_queue_bundle}" "${_two_queue_kernel_source}"
            DEPENDS "${_two_queue_kernel_source}"
            COMMENT "Compiling gfx950 two-queue ConSan kernel bundle"
            VERBATIM
        )
        add_custom_command(
            OUTPUT "${_two_queue_hsaco}"
            COMMAND
                "${_two_queue_offload_bundler}" --type=o --unbundle
                --targets=hipv4-amdgcn-amd-amdhsa--gfx950
                --input=${_two_queue_bundle} --output=${_two_queue_hsaco}
            DEPENDS "${_two_queue_bundle}"
            COMMENT "Extracting gfx950 two-queue ConSan code object"
            VERBATIM
        )
        add_custom_target(
            hip_consan_device_gfx950_two_queue_dispatch_identity_kernels_target
            ALL
            DEPENDS "${_two_queue_hsaco}"
        )
        rj_add_hip_test(
            hip_consan_device_two_queue_dispatch_identity_gfx950_test
            ARCH gfx950
            SOURCE dbi/consan/device/gfx950_two_queue_dispatch_identity_test.cpp
            COMPILE_DEFINITIONS
                "RJ_CONSAN_GFX950_TWO_QUEUE_HSACO=\"${_two_queue_hsaco}\""
            EXTRA_DEPENDS "${_two_queue_hsaco}"
            EXTRA_LIBS -lhsa-runtime64
        )
        add_dependencies(
            hip_consan_device_two_queue_dispatch_identity_gfx950_test_target
            hip_consan_device_gfx950_two_queue_dispatch_identity_kernels_target
            rocjitsu_dbi_hooks
        )
        if(RJ_INSTALL_TESTS)
            install(
                FILES "${_two_queue_hsaco}"
                DESTINATION "${RJ_INSTALL_TEST_BIN_DIR}"
            )
        endif()
    endif()

    if(target STREQUAL "gfx1250")
        set(_cluster_kernel_source
            "${CMAKE_CURRENT_SOURCE_DIR}/dbi/consan/device/gfx1250_cluster_device_kernels.hip"
        )
        set(_cluster_hsaco
            "${CMAKE_CURRENT_BINARY_DIR}/hip_consan_device_gfx1250_cluster_kernels.hsaco"
        )
        set(_cluster_bundle
            "${CMAKE_CURRENT_BINARY_DIR}/hip_consan_device_gfx1250_cluster_kernels.bundle"
        )
        find_program(
            _cluster_offload_bundler
            clang-offload-bundler
            REQUIRED
            HINTS
                "${ROCM_PATH}/lib/llvm/bin"
                "${ROCM_PATH}/llvm/bin"
                "${ROCM_PATH}/bin"
        )
        set(_cluster_compiler_command ${AMDCXX})
        if(CMAKE_HIP_COMPILER_LAUNCHER)
            list(
                PREPEND _cluster_compiler_command
                ${CMAKE_HIP_COMPILER_LAUNCHER}
            )
        endif()
        add_custom_command(
            OUTPUT "${_cluster_bundle}"
            COMMAND
                ${_cluster_compiler_command} -x hip --cuda-device-only
                --offload-arch=gfx1250 --rocm-path=${ROCM_PATH} -std=c++20 -O2
                -o "${_cluster_bundle}" "${_cluster_kernel_source}"
            DEPENDS "${_cluster_kernel_source}"
            COMMENT "Compiling gfx1250 clustered ConSan kernel bundle"
            VERBATIM
        )
        add_custom_command(
            OUTPUT "${_cluster_hsaco}"
            COMMAND
                "${_cluster_offload_bundler}" --type=o --unbundle
                --targets=hipv4-amdgcn-amd-amdhsa--gfx1250
                --input=${_cluster_bundle} --output=${_cluster_hsaco}
            DEPENDS "${_cluster_bundle}"
            COMMENT "Extracting gfx1250 clustered ConSan code object"
            VERBATIM
        )
        add_custom_target(
            hip_consan_device_gfx1250_cluster_kernels_target
            ALL
            DEPENDS "${_cluster_hsaco}"
        )
        rj_add_hip_test(
            hip_consan_device_gfx1250_cluster_gfx1250_test
            ARCH gfx1250
            SOURCE dbi/consan/device/gfx1250_cluster_device_test.cpp
            COMPILE_DEFINITIONS
                "RJ_CONSAN_GFX1250_CLUSTER_HSACO=\"${_cluster_hsaco}\""
            EXTRA_DEPENDS "${_cluster_hsaco}"
            EXTRA_LIBS -lhsa-runtime64
        )
        add_dependencies(
            hip_consan_device_gfx1250_cluster_gfx1250_test_target
            hip_consan_device_gfx1250_cluster_kernels_target
            rocjitsu_dbi_hooks
        )
        if(RJ_INSTALL_TESTS)
            install(
                FILES "${_cluster_hsaco}"
                DESTINATION "${RJ_INSTALL_TEST_BIN_DIR}"
            )
        endif()
    endif()
endfunction()

function(
    rj_add_consan_cdna_hip_sim_case
    target_name
    target
    build_dir
    name
    suite
)
    set(_runner
        "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../dbi/consan/consan_cdna_hip_sim.py"
    )
    set(_test_name "ConSan${target_name}HipMoiSim.${name}")
    rj_add_test(
        NAME "${_test_name}"
        COMMAND
            "${_runner}"
            --target
            "${target}"
            --rocjitsu
            "$<TARGET_FILE:rocjitsu_bin>"
            --hip-moi-build
            "${build_dir}"
            --suite
            "${suite}"
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        TIMEOUT 75
        LABELS consan "${target}" hip-moi simulator
    )
    set_property(TEST "${_test_name}" PROPERTY RUN_SERIAL TRUE)
endfunction()

function(
    rj_add_consan_cdna_hip_sim_tests
    target_name
    target
    option_name
    build_dir
)
    if(NOT build_dir)
        return()
    endif()
    if(NOT IS_ABSOLUTE "${build_dir}")
        message(FATAL_ERROR "${option_name} must be absolute")
    endif()
    if(NOT IS_DIRECTORY "${build_dir}")
        message(FATAL_ERROR "${option_name} is not a directory: ${build_dir}")
    endif()
    get_filename_component(_build_dir_name "${build_dir}" NAME)
    set(_expected_build_dir_name "hip-moi-build-${target}-tests")
    if(NOT _build_dir_name STREQUAL _expected_build_dir_name)
        message(
            FATAL_ERROR
            "${option_name} must name ${_expected_build_dir_name}"
        )
    endif()

    set(_runner
        "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../dbi/consan/consan_cdna_hip_sim.py"
    )
    set(_suite_registry
        "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../dbi/consan/consan_cdna_hip_registry.py"
    )
    set(_validation_registry
        "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../dbi/consan/consan_validation.py"
    )
    set(_validation_modules
        "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../dbi/consan/consan_validation_catalog.py"
        "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../dbi/consan/consan_validation_commands.py"
        "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../dbi/consan/consan_validation_diagnostics.py"
        "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../dbi/consan/consan_validation_empirical.py"
        "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../dbi/consan/consan_validation_execution.py"
        "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../dbi/consan/consan_validation_faults.py"
        "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../dbi/consan/consan_validation_statistics.py"
    )
    # These scripts generate the CTest rows below at configure time. Make any
    # registry or validation-contract edit regenerate those rows automatically.
    set_property(
        DIRECTORY
        APPEND
        PROPERTY
            CMAKE_CONFIGURE_DEPENDS
                "${_runner}"
                "${_suite_registry}"
                "${_validation_registry}"
                ${_validation_modules}
    )
    execute_process(
        COMMAND "${_runner}" --list-suites
        RESULT_VARIABLE _suite_registry_result
        OUTPUT_VARIABLE _suite_registry
        ERROR_VARIABLE _suite_registry_error
        OUTPUT_STRIP_TRAILING_WHITESPACE
        TIMEOUT 10
    )
    if(NOT _suite_registry_result EQUAL 0)
        message(
            FATAL_ERROR
            "failed to read the CDNA hip-moi suite registry: "
            "${_suite_registry_error}"
        )
    endif()
    string(REPLACE "\r\n" "\n" _suite_registry "${_suite_registry}")
    string(REPLACE "\n" ";" _suite_specs "${_suite_registry}")
    foreach(_suite_spec IN LISTS _suite_specs)
        string(REPLACE "|" ";" _suite_fields "${_suite_spec}")
        list(LENGTH _suite_fields _suite_field_count)
        if(NOT _suite_field_count EQUAL 2)
            message(
                FATAL_ERROR
                "malformed CDNA hip-moi suite registry row: ${_suite_spec}"
            )
        endif()
        list(GET _suite_fields 0 _ctest_name)
        list(GET _suite_fields 1 _suite_id)
        rj_add_consan_cdna_hip_sim_case(
            "${target_name}"
            "${target}"
            "${build_dir}"
            "${_ctest_name}"
            "${_suite_id}"
        )
    endforeach()
endfunction()

set(RJ_CONSAN_LDS_CLEAN_TEST "ConSanLdsTest.PaddedCleanStoreCompletes")
set(RJ_CONSAN_LDS_FLAT_RACY_TEST
    "ConSanLdsTest.PaddedCrossWaveFlatLdsStoreCompletes"
)
set(RJ_CONSAN_LDS_2D_RACY_TEST
    "ConSanLdsTest.PaddedTwoDimensionalCrossWaveStoreCompletes"
)
set(RJ_CONSAN_LDS_3D_RACY_TEST
    "ConSanLdsTest.PaddedThreeDimensionalCrossWaveStoreCompletes"
)
set(RJ_CONSAN_LDS_HIGH_D16_TEST
    "ConSanLdsTest.PaddedHighHalfD16FlatLoadsProduceExpectedResults"
)
set(RJ_CONSAN_LDS_HIGH_D16_STORE_TEST
    "ConSanLdsTest.PaddedHighHalfD16FlatStoresProduceExpectedResults"
)
set(RJ_CONSAN_LDS_ZERO_EXEC_SCALAR_SPILL_TEST
    "ConSanLdsTest.ZeroExecFlatSpillPreservesLiveScalarPair"
)
set(RJ_CONSAN_LDS_FULL_PRESSURE_FLAT_SPILL_TEST
    "ConSanLdsTest.FullPressureFlatSpillPreservesLiveRegisterFiles"
)
set(RJ_CONSAN_LDS_DYNAMIC_STACK_FULL_PRESSURE_FLAT_SPILL_TEST
    "ConSanLdsTest.DynamicStackFullPressureFlatSpillPreservesLiveRegistersAndPrivateValues"
)
set(RJ_CONSAN_LDS_ZERO_EXEC_NATIVE_SCALAR_SPILL_TEST
    "ConSanLdsTest.ZeroExecNativeLdsSpillPreservesAllocatedScalars"
)
set(RJ_CONSAN_STATIC_LDS_BYTE_ADJACENT_TEST
    "ConSanLdsTest.StaticLdsByteAdjacentBytesStayIndependent"
)
set(RJ_CONSAN_STATIC_LDS_BYTE_OVERLAP_TEST
    "ConSanLdsTest.StaticLdsByteOverlappingByteCompletes"
)
set(RJ_CONSAN_DYNAMIC_LDS_BYTE_ADJACENT_TEST
    "ConSanLdsTest.DynamicLdsByteAdjacentBytesStayIndependent"
)
set(RJ_CONSAN_DYNAMIC_LDS_BYTE_OVERLAP_TEST
    "ConSanLdsTest.DynamicLdsByteOverlappingByteCompletes"
)
set(RJ_CONSAN_STATIC_LDS_BYTE_TAIL_B16_TEST
    "ConSanLdsTest.StaticLdsByteTailB16StaysWithinCoverage"
)
set(RJ_CONSAN_DYNAMIC_LDS_BYTE_UNEQUAL_TEST
    "ConSanLdsTest.DynamicLdsByteUnequalHighByteOverlapCompletes"
)

set(RJ_CONSAN_CDNA_FORCED_SPILL_TEST
    "ConSanCdnaTest.ForcedSpillPreservesLiveVgprsAndWritesEntry"
)

set(RJ_CONSAN_CDNA_SHARED_HELPER_TEST
    "ConSanCdnaTest.SharedHelperForcedSpillPreservesBothCallers"
)
set(RJ_CONSAN_CDNA_BARRIER_TEST
    "ConSanCdnaTest.BarrierForcedSpillPreservesLiveVgprsAndAdvancesEpoch"
)
set(RJ_CONSAN_CDNA_ATOMIC_TEST
    "ConSanCdnaTest.AtomicForcedSpillExecutesVglobalAddressForms"
)
set(RJ_CONSAN_CDNA_DYNAMIC_STACK_TEST
    "ConSanCdnaTest.DynamicStackSpillPreservesPrivateValues"
)
set(RJ_CONSAN_CDNA_ZERO_EXEC_SCALAR_SPILL_CORRECT_TEST
    "ConSanCdnaTest.ZeroExecScalarSpillCorrectPreservesLiveScalars"
)
set(RJ_CONSAN_CDNA_NONEMPTY_SCALAR_SPILL_INCORRECT_TEST
    "ConSanCdnaTest.NonemptyScalarSpillIncorrectReportsRace"
)
set(RJ_CONSAN_CDNA_ADDRESS_GROUP_TEST
    "ConSanCdnaTest.SecondAddressGroupReportsExactLaneConflict"
)
set(RJ_CONSAN_CDNA_DISJOINT_B128_TEST
    "ConSanCdnaTest.DisjointLaneB128StoresDoNotConflict"
)

set(RJ_CONSAN_FORCED_SPILL_TEST
    "ConSanHipTest.ForcedSpillPreservesLiveVgprsAndWritesEntry"
)
set(RJ_CONSAN_SHARED_NOINLINE_HELPER_TEST
    "ConSanHipTest.SharedNoinlineHelperPreservesBothCallers"
)
set(RJ_CONSAN_SHARED_ATOMIC_HELPER_TEST
    "ConSanHipTest.SharedAtomicNoinlineHelperPreservesBothCallers"
)
set(RJ_CONSAN_DYNAMIC_PRIVATE_STACK_TEST
    "ConSanHipTest.DynamicPrivateStackPreservesOutput"
)
set(RJ_CONSAN_ADDRESS_GROUP_TEST
    "ConSanHipTest.SecondAddressGroupReportsExactLaneConflict"
)
set(RJ_CONSAN_DISJOINT_B128_TEST
    "ConSanHipTest.DisjointLaneB128StoresDoNotConflict"
)

function(
    rj_add_consan_address_group_sim_cases
    prefix
    target
    conflict_filter
    disjoint_filter
    config
)
    # Exact second-address-group provenance is currently implemented on RDNA4.
    if(prefix MATCHES "Gfx1201" OR prefix STREQUAL "ConSanHipTest")
    rj_add_consan_hip_case(
        NAME "${prefix}.SecondAddressGroupRace"
        TARGET ${target}
        FILTER "${conflict_filter}"
        CONFIG "${config}"
        TIMEOUT 60
        RUN_SERIAL
        PASS_REGEX
            "first_lds=\\[4,8\\) second_lds=\\[4,8\\).*first_lane_mask=0x2 second_lane_mask=0x4"
        FAIL_REGEX "RJ_CONSAN_REQUIRE_(RECORDS|DIAGNOSTICS) requested"
        ENVIRONMENT
            "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
            "RJ_CONSAN_MODE=default"
            "RJ_CONSAN_REQUIRE_RECORDS=1"
            "RJ_CONSAN_REQUIRE_DIAGNOSTICS=1"
            "RJ_CONSAN_FORBID_OVERFLOW=1"
            "RJ_CONSAN_TRACK_BARRIERS=0"
            "RJ_CONSAN_TRACK_ATOMICS=0"
            "RJ_CONSAN_TEST_KERNEL_FILTER=second_address_group_race_for_instrumentation"
            "RJ_CONSAN_LOG=3"
            "RJ_CONSAN_MAX_PATCHES=1"
            "RJ_CONSAN_REQUIRE_PATCH=1"
    )
    endif()
    rj_add_consan_hip_case(
        NAME "${prefix}.DisjointLaneB128StoresAreClean"
        TARGET ${target}
        FILTER "${disjoint_filter}"
        CONFIG "${config}"
        TIMEOUT 60
        RUN_SERIAL
        PASS_REGEX "visible=[1-9][0-9]*"
        FAIL_REGEX "RJ_CONSAN_(REQUIRE_RECORDS|FORBID_DIAGNOSTICS) requested"
        ENVIRONMENT
            "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
            "RJ_CONSAN_MODE=default"
            "RJ_CONSAN_REQUIRE_RECORDS=1"
            "RJ_CONSAN_FORBID_DIAGNOSTICS=1"
            "RJ_CONSAN_FORBID_OVERFLOW=1"
            "RJ_CONSAN_TRACK_BARRIERS=0"
            "RJ_CONSAN_TRACK_ATOMICS=0"
            "RJ_CONSAN_TEST_KERNEL_FILTER=disjoint_lane_b128_stores_for_instrumentation"
            "RJ_CONSAN_LOG=1"
            "RJ_CONSAN_MAX_PATCHES=1"
            "RJ_CONSAN_REQUIRE_PATCH=1"
    )
endfunction()

function(rj_add_consan_byte_overlap_cases prefix target config)
    set(_exact_common_environment
        "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
        "RJ_CONSAN_MODE=default"
        "RJ_CONSAN_AUTO_REPORT_BUFFER_SIZE=67108864"
        "RJ_CONSAN_FLAT_PROVENANCE=strict"
        "RJ_CONSAN_REQUIRE_RECORDS=1"
        "RJ_CONSAN_FORBID_OVERFLOW=1"
        "RJ_CONSAN_TRACK_BARRIERS=0"
        "RJ_CONSAN_TRACK_ATOMICS=0"
        "RJ_CONSAN_LOG=3"
        "RJ_CONSAN_REQUIRE_PATCH=1"
    )
    rj_add_consan_hip_case(
        NAME "${prefix}.StaticLdsByteAdjacentClean"
        TARGET ${target}
        FILTER "${RJ_CONSAN_STATIC_LDS_BYTE_ADJACENT_TEST}"
        CONFIG "${config}"
        TIMEOUT 60
        RUN_SERIAL
        PASS_REGEX "visible=[1-9][0-9]*"
        FAIL_REGEX "RJ_CONSAN_(REQUIRE_RECORDS|FORBID_DIAGNOSTICS) requested"
        ENVIRONMENT
            ${_exact_common_environment}
            "RJ_CONSAN_FORBID_DIAGNOSTICS=1"
            "RJ_CONSAN_TEST_KERNEL_FILTER=padded_static_lds_byte_adjacent_bytes"
            "RJ_CONSAN_MAX_PATCHES=1"
    )
    rj_add_consan_hip_case(
        NAME "${prefix}.StaticLdsByteOverlapReports"
        TARGET ${target}
        FILTER "${RJ_CONSAN_STATIC_LDS_BYTE_OVERLAP_TEST}"
        CONFIG "${config}"
        TIMEOUT 60
        RUN_SERIAL
        PASS_REGEX "conflicts=[1-9][0-9]*"
        FAIL_REGEX "RJ_CONSAN_REQUIRE_(RECORDS|DIAGNOSTICS) requested"
        ENVIRONMENT
            ${_exact_common_environment}
            "RJ_CONSAN_REQUIRE_DIAGNOSTICS=1"
            "RJ_CONSAN_TEST_KERNEL_FILTER=padded_static_lds_byte_overlapping_byte"
            "RJ_CONSAN_MAX_PATCHES=1"
    )
    rj_add_consan_hip_case(
        NAME "${prefix}.DynamicLdsByteAdjacentClean"
        TARGET ${target}
        FILTER "${RJ_CONSAN_DYNAMIC_LDS_BYTE_ADJACENT_TEST}"
        CONFIG "${config}"
        TIMEOUT 60
        RUN_SERIAL
        PASS_REGEX "visible=[1-9][0-9]*"
        FAIL_REGEX "RJ_CONSAN_(REQUIRE_RECORDS|FORBID_DIAGNOSTICS) requested"
        ENVIRONMENT
            ${_exact_common_environment}
            "RJ_CONSAN_FORBID_DIAGNOSTICS=1"
            "RJ_CONSAN_TEST_KERNEL_FILTER=padded_dynamic_lds_byte_adjacent_bytes"
            "RJ_CONSAN_MAX_PATCHES=1"
    )
    rj_add_consan_hip_case(
        NAME "${prefix}.DynamicLdsByteOverlapReports"
        TARGET ${target}
        FILTER "${RJ_CONSAN_DYNAMIC_LDS_BYTE_OVERLAP_TEST}"
        CONFIG "${config}"
        TIMEOUT 60
        RUN_SERIAL
        PASS_REGEX "conflicts=[1-9][0-9]*"
        FAIL_REGEX "RJ_CONSAN_REQUIRE_(RECORDS|DIAGNOSTICS) requested"
        ENVIRONMENT
            ${_exact_common_environment}
            "RJ_CONSAN_REQUIRE_DIAGNOSTICS=1"
            "RJ_CONSAN_TEST_KERNEL_FILTER=padded_dynamic_lds_byte_overlapping_byte"
            "RJ_CONSAN_MAX_PATCHES=1"
    )
    rj_add_consan_hip_case(
        NAME "${prefix}.StaticLdsByteTailB16Clean"
        TARGET ${target}
        FILTER "${RJ_CONSAN_STATIC_LDS_BYTE_TAIL_B16_TEST}"
        CONFIG "${config}"
        TIMEOUT 60
        RUN_SERIAL
        PASS_REGEX "visible=[1-9][0-9]*"
        FAIL_REGEX "RJ_CONSAN_(REQUIRE_RECORDS|FORBID_DIAGNOSTICS|FORBID_OVERFLOW) requested"
        ENVIRONMENT
            ${_exact_common_environment}
            "RJ_CONSAN_FORBID_DIAGNOSTICS=1"
            "RJ_CONSAN_TEST_KERNEL_FILTER=padded_static_lds_byte_tail_b16"
            "RJ_CONSAN_MAX_PATCHES=1"
    )
    rj_add_consan_hip_case(
        NAME "${prefix}.DynamicLdsByteUnequalHighByteOverlapReports"
        TARGET ${target}
        FILTER "${RJ_CONSAN_DYNAMIC_LDS_BYTE_UNEQUAL_TEST}"
        CONFIG "${config}"
        TIMEOUT 60
        RUN_SERIAL
        PASS_REGEX
            "conflicts=[1-9][0-9]*"
        FAIL_REGEX "RJ_CONSAN_REQUIRE_(RECORDS|DIAGNOSTICS) requested"
        ENVIRONMENT
            ${_exact_common_environment}
            "RJ_CONSAN_REQUIRE_DIAGNOSTICS=1"
            "RJ_CONSAN_TEST_KERNEL_FILTER=padded_dynamic_lds_byte_unequal_high_byte_overlap"
            "RJ_CONSAN_MAX_PATCHES=2"
    )
endfunction()

function(rj_add_consan_multidimensional_cases prefix target config)
    set(_multidimensional_common_environment
        "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
        "RJ_CONSAN_MODE=default"
        # Resident-wave IDs are sparse; retain all ten owner bits in the bank.
        "RJ_CONSAN_WATCHPOINT_BANKS=1024"
        "RJ_CONSAN_AUTO_REPORT_BUFFER_SIZE=1048576"
        "RJ_CONSAN_REQUIRE_RECORDS=1"
        "RJ_CONSAN_REQUIRE_DIAGNOSTICS=1"
        "RJ_CONSAN_FORBID_OVERFLOW=1"
        "RJ_CONSAN_TRACK_BARRIERS=0"
        "RJ_CONSAN_TRACK_ATOMICS=0"
        "RJ_CONSAN_LOG=3"
        "RJ_CONSAN_MAX_PATCHES=1"
        "RJ_CONSAN_REQUIRE_PATCH=1"
    )
    rj_add_consan_hip_case(
        NAME "${prefix}.TwoDimensionalCrossWaveRace"
        TARGET ${target}
        FILTER "${RJ_CONSAN_LDS_2D_RACY_TEST}"
        CONFIG "${config}"
        TIMEOUT 60
        RUN_SERIAL
        PASS_REGEX
            "conflicts=[1-9][0-9]*"
        FAIL_REGEX "RJ_CONSAN_REQUIRE_(RECORDS|DIAGNOSTICS) requested"
        ENVIRONMENT
            ${_multidimensional_common_environment}
            "RJ_CONSAN_OWNER_SOURCE=hwid"
            "RJ_CONSAN_TEST_KERNEL_FILTER=padded_2d_cross_wave_lds_store"
    )
    rj_add_consan_hip_case(
        NAME "${prefix}.ThreeDimensionalCrossWaveRace"
        TARGET ${target}
        FILTER "${RJ_CONSAN_LDS_3D_RACY_TEST}"
        CONFIG "${config}"
        TIMEOUT 60
        RUN_SERIAL
        PASS_REGEX
            "conflicts=[1-9][0-9]*"
        FAIL_REGEX "RJ_CONSAN_REQUIRE_(RECORDS|DIAGNOSTICS) requested"
        ENVIRONMENT
            ${_multidimensional_common_environment}
            "RJ_CONSAN_OWNER_SOURCE=hwid"
            "RJ_CONSAN_TEST_KERNEL_FILTER=padded_3d_cross_wave_lds_store"
    )
endfunction()

function(rj_add_consan_common_sim_cases)
    set(_one_value_args
        PREFIX
        LDS_TARGET
        TARGET
        CONFIG

        TEST
        SHARED_TEST
        ACCESS_KERNEL
        SHARED_KERNEL
        SHARED_INIT_OWNER_EPOCH
    )
    cmake_parse_arguments(PARSE_ARGV 0 _sim "" "${_one_value_args}" "")
    if(_sim_UNPARSED_ARGUMENTS)
        message(
            FATAL_ERROR
            "unknown common ConSan simulator arguments: ${_sim_UNPARSED_ARGUMENTS}"
        )
    endif()
    foreach(
        _required
        IN
        ITEMS
            PREFIX
            LDS_TARGET
            TARGET
            CONFIG

            TEST
            SHARED_TEST
            ACCESS_KERNEL
            SHARED_KERNEL
    )
        if(NOT _sim_${_required})
            message(
                FATAL_ERROR
                "common ConSan simulator cases require ${_required}"
            )
        endif()
    endforeach()
    if(
        NOT DEFINED _sim_SHARED_INIT_OWNER_EPOCH
        OR NOT "${_sim_SHARED_INIT_OWNER_EPOCH}" MATCHES "^[01]$"
    )
        message(
            FATAL_ERROR
            "common ConSan simulator cases require "
            "SHARED_INIT_OWNER_EPOCH to be 0 or 1"
        )
    endif()

    rj_add_consan_hip_case(
        NAME "${_sim_PREFIX}.SuperColliderClean"
        TARGET ${_sim_LDS_TARGET}
        FILTER "${RJ_CONSAN_LDS_CLEAN_TEST}"
        CONFIG "${_sim_CONFIG}"
        TIMEOUT 60
        RUN_SERIAL
        PASS_REGEX "\\[  PASSED  \\] 1 test"
        ENVIRONMENT
            "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
            "RJ_CONSAN_MODE=supercollider"
            "RJ_CONSAN_LOG=1"
            "RJ_CONSAN_SC_DELAY=2"
            "RJ_CONSAN_TMP_VGPR=3"
            "RJ_CONSAN_REQUIRE_PATCH=1"
    )
    rj_add_consan_hip_case(
        NAME "${_sim_PREFIX}.SuperColliderHighHalfD16Load"
        TARGET ${_sim_LDS_TARGET}
        FILTER "${RJ_CONSAN_LDS_HIGH_D16_TEST}"
        CONFIG "${_sim_CONFIG}"
        TIMEOUT 60
        RUN_SERIAL
        PASS_REGEX "kind=inline-flat-load-check-trap"
        ENVIRONMENT
            "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
            "RJ_CONSAN_MODE=supercollider"
            "RJ_CONSAN_FLAT_PROVENANCE=strict"
            "RJ_CONSAN_LOG=3"
            "RJ_CONSAN_SC_DELAY=2"
            "RJ_CONSAN_TEST_KERNEL_FILTER=padded_flat_high_d16_loads_kernel_for_instrumentation"
            "RJ_CONSAN_MAX_PATCHES=8"
            "RJ_CONSAN_REQUIRE_PATCH=1"
    )
    rj_add_consan_hip_case(
        NAME "${_sim_PREFIX}.SuperColliderHighHalfD16Store"
        TARGET ${_sim_LDS_TARGET}
        FILTER "${RJ_CONSAN_LDS_HIGH_D16_STORE_TEST}"
        CONFIG "${_sim_CONFIG}"
        TIMEOUT 60
        RUN_SERIAL
        PASS_REGEX "kind=inline-flat-store-check-trap"
        ENVIRONMENT
            "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
            "RJ_CONSAN_MODE=supercollider"
            "RJ_CONSAN_FLAT_PROVENANCE=strict"
            "RJ_CONSAN_LOG=3"
            "RJ_CONSAN_SC_DELAY=2"
            "RJ_CONSAN_TEST_KERNEL_FILTER=padded_flat_high_d16_store"
            "RJ_CONSAN_MAX_PATCHES=12"
            "RJ_CONSAN_REQUIRE_PATCH=1"
    )
    rj_add_consan_hip_case(
        NAME "${_sim_PREFIX}.FlatLdsForcedSpill"
        TARGET ${_sim_LDS_TARGET}
        FILTER "${RJ_CONSAN_LDS_FLAT_RACY_TEST}"
        CONFIG "${_sim_CONFIG}"
        TIMEOUT 60
        RUN_SERIAL
        PASS_REGEX "ConSan resources.*spill=[1-9][0-9]*.*emitted_spill_patches=[1-9][0-9]*"
        FAIL_REGEX "RJ_CONSAN_REQUIRE_(RECORDS|DIAGNOSTICS) requested"
        ENVIRONMENT
            "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
            "RJ_CONSAN_MODE=default"
            "RJ_CONSAN_AUTO_REPORT_BUFFER_SIZE=1048576"
            "RJ_CONSAN_FLAT_PROVENANCE=strict"

            "RJ_CONSAN_REQUIRE_RECORDS=1"
            "RJ_CONSAN_REQUIRE_DIAGNOSTICS=1"
            "RJ_CONSAN_TRACK_BARRIERS=0"
            "RJ_CONSAN_TRACK_ATOMICS=0"
            "RJ_CONSAN_TEST_FORCE_VGPR_SPILL=1"
            "RJ_CONSAN_TEST_KERNEL_FILTER=padded_cross_wave_flat_lds_store"
            "RJ_CONSAN_LOG=3"
            "RJ_CONSAN_MAX_PATCHES=1"
            "RJ_CONSAN_REQUIRE_PATCH=1"
    )

    rj_add_consan_hip_case(
        NAME "${_sim_PREFIX}.ForcedSpill"
        TARGET ${_sim_TARGET}
        FILTER "${_sim_TEST}"
        CONFIG "${_sim_CONFIG}"
        TIMEOUT 60
        RUN_SERIAL
        PASS_REGEX "ConSan resources.*spill=[1-9][0-9]*.*emitted_spill_patches=[1-9][0-9]*"
        ENVIRONMENT
            "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
            "RJ_CONSAN_MODE=default"
            "RJ_CONSAN_TRACK_BARRIERS=0"
            "RJ_CONSAN_TRACK_ATOMICS=0"
            "RJ_CONSAN_TEST_FORCE_VGPR_SPILL=1"
            "RJ_CONSAN_TEST_KERNEL_FILTER=${_sim_ACCESS_KERNEL}"
            "RJ_CONSAN_LOG=3"
            "RJ_CONSAN_MAX_PATCHES=1"
            "RJ_CONSAN_REQUIRE_PATCH=1"
    )
    rj_add_consan_hip_case(
        NAME "${_sim_PREFIX}.SharedHelperForcedSpill"
        TARGET ${_sim_TARGET}
        FILTER "${_sim_SHARED_TEST}"
        CONFIG "${_sim_CONFIG}"
        TIMEOUT 60
        RUN_SERIAL
        PASS_REGEX "ConSan resources.*spill=[1-9][0-9]*.*emitted_spill_patches=[1-9][0-9]*"
        ENVIRONMENT
            "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
            "RJ_CONSAN_MODE=default"
            "RJ_CONSAN_INIT_OWNER_EPOCH=${_sim_SHARED_INIT_OWNER_EPOCH}"
            "RJ_CONSAN_TRACK_BARRIERS=0"
            "RJ_CONSAN_TRACK_ATOMICS=0"
            "RJ_CONSAN_TEST_FORCE_VGPR_SPILL=1"
            "RJ_CONSAN_TEST_KERNEL_FILTER=${_sim_SHARED_KERNEL}"
            "RJ_CONSAN_LOG=3"
            "RJ_CONSAN_MAX_PATCHES=1"
            "RJ_CONSAN_REQUIRE_PATCH=1"
    )
    rj_add_consan_byte_overlap_cases(
        "${_sim_PREFIX}"
        ${_sim_LDS_TARGET}
        "${_sim_CONFIG}"
    )
    rj_add_consan_multidimensional_cases(
        "${_sim_PREFIX}"
        ${_sim_LDS_TARGET}
        "${_sim_CONFIG}"
    )
endfunction()

function(rj_add_consan_scalar_spill_exec_cases)
    set(_options PHYSICAL)
    set(
        _one_value_args
        PREFIX TARGET CONFIG ARCH_LABEL CORRECT_TEST INCORRECT_TEST
    )
    cmake_parse_arguments(
        PARSE_ARGV
        0
        _spill
        "${_options}"
        "${_one_value_args}"
        ""
    )
    if(_spill_UNPARSED_ARGUMENTS)
        message(
            FATAL_ERROR
            "unknown ConSan scalar-spill arguments: ${_spill_UNPARSED_ARGUMENTS}"
        )
    endif()
    foreach(
        _required
        IN
        ITEMS PREFIX TARGET ARCH_LABEL CORRECT_TEST INCORRECT_TEST
    )
        if(NOT _spill_${_required})
            message(
                FATAL_ERROR
                "ConSan scalar-spill cases require ${_required}"
            )
        endif()
    endforeach()

    set(_config_args)
    if(_spill_CONFIG)
        list(APPEND _config_args CONFIG "${_spill_CONFIG}")
    endif()
    set(_resource_args)
    set(_execution_label simulator)
    if(_spill_PHYSICAL)
        list(
            APPEND _resource_args
            RESOURCE_LOCK "consan_physical_${_spill_ARCH_LABEL}"
        )
        set(_execution_label physical)
    endif()

    # The correct and incorrect workloads intentionally share the same SGPR
    # pressure and ConSan instrumentation. Empty EXEC must preserve the guest's
    # scalars without publishing evidence, while two nonempty waves must retain
    # those scalars and diagnose their conflicting LDS stores.
    rj_add_consan_hip_case(
        NAME "${_spill_PREFIX}.BaselineScalarSpillExec.Correct"
        TARGET ${_spill_TARGET}
        FILTER "${_spill_CORRECT_TEST}"
        ${_config_args}
        ${_resource_args}
        TIMEOUT 60
        LABELS
            consan consan-device scalar-spill baseline
            "${_spill_ARCH_LABEL}" "${_execution_label}" correct
    )
    rj_add_consan_hip_case(
        NAME "${_spill_PREFIX}.BaselineScalarSpillExec.Incorrect"
        TARGET ${_spill_TARGET}
        FILTER "${_spill_INCORRECT_TEST}"
        ${_config_args}
        ${_resource_args}
        TIMEOUT 60
        LABELS
            consan consan-device scalar-spill baseline
            "${_spill_ARCH_LABEL}" "${_execution_label}" incorrect racy
    )

    rj_add_consan_hip_case(
        NAME "${_spill_PREFIX}.ScalarSpillExec.Correct"
        TARGET ${_spill_TARGET}
        FILTER "${_spill_CORRECT_TEST}"
        ${_config_args}
        ${_resource_args}
        TIMEOUT 60
        ENVIRONMENT
            "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
            "RJ_CONSAN_MODE=default"
            "RJ_CONSAN_AUTO_REPORT_BUFFER_SIZE=67108864"
            "RJ_CONSAN_FORBID_DIAGNOSTICS=1"
            "RJ_CONSAN_FORBID_OVERFLOW=1"
            "RJ_CONSAN_RUNTIME_SAMPLE_STRIDE=1"
            "RJ_CONSAN_DEVICE_CONFLICT_CHECK=1"
            "RJ_CONSAN_TRACK_BARRIERS=0"
            "RJ_CONSAN_TRACK_ATOMICS=0"
            "RJ_CONSAN_TEST_FORCE_VGPR_SPILL=1"
            "RJ_CONSAN_TEST_KERNEL_FILTER=zero_exec_scalar_spill_correct_kernel_for_instrumentation"
            "RJ_CONSAN_LOG=3"
            "RJ_CONSAN_MAX_PATCHES=8"
            "RJ_CONSAN_REQUIRE_PATCH=1"
        LABELS
            consan consan-device default scalar-spill
            "${_spill_ARCH_LABEL}" "${_execution_label}" correct
    )
    rj_add_consan_hip_case(
        NAME "${_spill_PREFIX}.ScalarSpillExec.Incorrect"
        TARGET ${_spill_TARGET}
        FILTER "${_spill_INCORRECT_TEST}"
        ${_config_args}
        ${_resource_args}
        TIMEOUT 60
        ENVIRONMENT
            "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
            "RJ_CONSAN_MODE=default"
            "RJ_CONSAN_AUTO_REPORT_BUFFER_SIZE=67108864"
            "RJ_CONSAN_REQUIRE_RECORDS=1"
            "RJ_CONSAN_REQUIRE_DIAGNOSTICS=1"
            "RJ_CONSAN_FORBID_OVERFLOW=1"
            "RJ_CONSAN_RUNTIME_SAMPLE_STRIDE=1"
            "RJ_CONSAN_DEVICE_CONFLICT_CHECK=1"
            "RJ_CONSAN_TRACK_BARRIERS=0"
            "RJ_CONSAN_TRACK_ATOMICS=0"
            "RJ_CONSAN_TEST_FORCE_VGPR_SPILL=1"
            "RJ_CONSAN_TEST_KERNEL_FILTER=nonempty_scalar_spill_incorrect_kernel_for_instrumentation"
            "RJ_CONSAN_LOG=3"
            "RJ_CONSAN_MAX_PATCHES=8"
            "RJ_CONSAN_REQUIRE_PATCH=1"
        LABELS
            consan consan-device default scalar-spill
            "${_spill_ARCH_LABEL}" "${_execution_label}" incorrect racy
    )
endfunction()

function(rj_add_consan_cdna_sim_cases)
    set(_one_value_args PREFIX LDS_TARGET TARGET CONFIG ARCH_LABEL)
    cmake_parse_arguments(PARSE_ARGV 0 _sim "" "${_one_value_args}" "")
    if(_sim_UNPARSED_ARGUMENTS)
        message(
            FATAL_ERROR
            "unknown ConSan CDNA simulator arguments: ${_sim_UNPARSED_ARGUMENTS}"
        )
    endif()
    foreach(_required IN ITEMS PREFIX LDS_TARGET TARGET CONFIG ARCH_LABEL)
        if(NOT _sim_${_required})
            message(
                FATAL_ERROR
                "ConSan CDNA simulator cases require ${_required}"
            )
        endif()
    endforeach()

    rj_add_consan_common_sim_cases(
        PREFIX "${_sim_PREFIX}"
        LDS_TARGET ${_sim_LDS_TARGET}
        TARGET ${_sim_TARGET}
        CONFIG "${_sim_CONFIG}"

        TEST "${RJ_CONSAN_CDNA_FORCED_SPILL_TEST}"
        SHARED_TEST "${RJ_CONSAN_CDNA_SHARED_HELPER_TEST}"
        ACCESS_KERNEL "cdna_forced_spill_test_kernel_for_instrumentation"
        SHARED_KERNEL "cdna_shared_lds_helper_for_instrumentation"
        SHARED_INIT_OWNER_EPOCH 1
    )
    rj_add_consan_address_group_sim_cases(
        "${_sim_PREFIX}"
        ${_sim_TARGET}
        "${RJ_CONSAN_CDNA_ADDRESS_GROUP_TEST}"
        "${RJ_CONSAN_CDNA_DISJOINT_B128_TEST}"
        "${_sim_CONFIG}"
    )
    rj_add_consan_scalar_spill_exec_cases(
        PREFIX "${_sim_PREFIX}"
        TARGET ${_sim_TARGET}
        CONFIG "${_sim_CONFIG}"
        ARCH_LABEL "${_sim_ARCH_LABEL}"
        CORRECT_TEST
            "${RJ_CONSAN_CDNA_ZERO_EXEC_SCALAR_SPILL_CORRECT_TEST}"
        INCORRECT_TEST
            "${RJ_CONSAN_CDNA_NONEMPTY_SCALAR_SPILL_INCORRECT_TEST}"
    )

    # CDNA-specific synchronization, atomic, and dynamic-stack spill cases.

    rj_add_consan_hip_case(
        NAME "${_sim_PREFIX}.BarrierForcedSpill"
        TARGET ${_sim_TARGET}
        FILTER "${RJ_CONSAN_CDNA_BARRIER_TEST}"
        CONFIG "${_sim_CONFIG}"
        TIMEOUT 60
        RUN_SERIAL
        PASS_REGEX "ConSan proof patch.*kind=trampoline-sync-metadata.*spilled_vgprs=[1-9][0-9]*"
        ENVIRONMENT
            "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
            "RJ_CONSAN_MODE=default"
            "RJ_CONSAN_INIT_OWNER_EPOCH=1"
            "RJ_CONSAN_TRACK_BARRIERS=1"
            "RJ_CONSAN_TRACK_ATOMICS=0"
            "RJ_CONSAN_TEST_FORCE_VGPR_SPILL=1"
            "RJ_CONSAN_TEST_KERNEL_FILTER=cdna_barrier_forced_spill_kernel_for_instrumentation"
            "RJ_CONSAN_LOG=3"
            "RJ_CONSAN_MAX_PATCHES=8"
            "RJ_CONSAN_REQUIRE_PATCH=1"
    )
    rj_add_consan_hip_case(
        NAME "${_sim_PREFIX}.AtomicForcedSpill"
        TARGET ${_sim_TARGET}
        FILTER "${RJ_CONSAN_CDNA_ATOMIC_TEST}"
        CONFIG "${_sim_CONFIG}"
        TIMEOUT 60
        RUN_SERIAL
        PASS_REGEX "ConSan proof patch.*kind=trampoline-sync-metadata.*spilled_vgprs=[1-9][0-9]*"
        ENVIRONMENT
            "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
            "RJ_CONSAN_MODE=default"
            "RJ_CONSAN_TRACK_BARRIERS=0"
            "RJ_CONSAN_TRACK_ATOMICS=1"
            "RJ_CONSAN_TEST_FORCE_VGPR_SPILL=1"
            "RJ_CONSAN_TEST_KERNEL_FILTER=cdna_atomic_forced_spill_kernel_for_instrumentation"
            "RJ_CONSAN_LOG=3"
            "RJ_CONSAN_MAX_PATCHES=8"
            "RJ_CONSAN_REQUIRE_PATCH=1"
    )

    rj_add_consan_hip_case(
        NAME "${_sim_PREFIX}.DynamicStackForcedSpill"
        TARGET ${_sim_TARGET}
        FILTER "${RJ_CONSAN_CDNA_DYNAMIC_STACK_TEST}"
        CONFIG "${_sim_CONFIG}"
        TIMEOUT 60
        RUN_SERIAL
        PASS_REGEX "ConSan resources.*spill=[1-9][0-9]*.*emitted_spill_patches=[1-9][0-9]*"
        ENVIRONMENT
            "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
            "RJ_CONSAN_MODE=default"
            "RJ_CONSAN_REQUIRE_RECORDS=1"
            "RJ_CONSAN_REQUIRE_DIAGNOSTICS=1"
            "RJ_CONSAN_TRACK_BARRIERS=0"
            "RJ_CONSAN_TRACK_ATOMICS=0"
            "RJ_CONSAN_TEST_FORCE_VGPR_SPILL=1"
            "RJ_CONSAN_TEST_KERNEL_FILTER=cdna_dynamic_stack_spill_kernel_for_instrumentation"
            "RJ_CONSAN_LOG=3"
            "RJ_CONSAN_MAX_PATCHES=1"
            "RJ_CONSAN_REQUIRE_PATCH=1"
    )
endfunction()

function(rj_add_consan_rdna4_b96_cases)
    set(_one_value_args PREFIX TARGET CONFIG TIMEOUT)
    cmake_parse_arguments(PARSE_ARGV 0 _b96 "" "${_one_value_args}" "")
    if(_b96_UNPARSED_ARGUMENTS)
        message(
            FATAL_ERROR
            "unknown ConSan RDNA4 B96 arguments: ${_b96_UNPARSED_ARGUMENTS}"
        )
    endif()
    foreach(_required IN ITEMS PREFIX TARGET TIMEOUT)
        if(NOT _b96_${_required})
            message(FATAL_ERROR "ConSan RDNA4 B96 cases require ${_required}")
        endif()
    endforeach()
    set(_config_args)
    if(_b96_CONFIG)
        list(APPEND _config_args CONFIG "${_b96_CONFIG}")
    endif()
    foreach(
        _mode
        IN
        ITEMS
            "ConSan|default"
    )
        string(REPLACE "|" ";" _mode_fields "${_mode}")
        list(GET _mode_fields 0 _mode_name)
        list(GET _mode_fields 1 _mode_mode)
        foreach(
            _case
            IN
            ITEMS
                "Racy|ConSanHipTest.NativeB96RacyAliasingLoadCompletesWithValidTupleWords|RJ_CONSAN_REQUIRE_DIAGNOSTICS=1"
                "Ordered|ConSanHipTest.NativeB96BarrierOrderedAliasingLoadPreservesOutput|RJ_CONSAN_FORBID_DIAGNOSTICS=1"
        )
            string(REPLACE "|" ";" _case_fields "${_case}")
            list(GET _case_fields 0 _case_name)
            list(GET _case_fields 1 _case_filter)
            list(GET _case_fields 2 _case_diagnostic_policy)
            if(_case_name STREQUAL "Ordered")
                set(_case_track_barriers 1)
                set(_case_max_patches 8)
            else()
                set(_case_track_barriers 0)
                set(_case_max_patches 2)
            endif()
            rj_add_consan_hip_case(
                NAME "${_b96_PREFIX}.NativeB96${_mode_name}${_case_name}"
                TARGET ${_b96_TARGET}
                FILTER "${_case_filter}"
                ${_config_args}
                TIMEOUT ${_b96_TIMEOUT}
                RUN_SERIAL
                PASS_REGEX "\\[  PASSED  \\] 1 test"
                ENVIRONMENT
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=${_mode_mode}"
                    "RJ_CONSAN_POLICY=strict"
                    "RJ_CONSAN_TRACK_BARRIERS=${_case_track_barriers}"
                    "RJ_CONSAN_TRACK_ATOMICS=0"
                    "RJ_CONSAN_REQUIRE_RECORDS=1"
                    "${_case_diagnostic_policy}"
                    "RJ_CONSAN_FORBID_OVERFLOW=1"
                    "RJ_CONSAN_TEST_KERNEL_FILTER=native_b96_aliasing_access_for_instrumentation"
                    "RJ_CONSAN_LOG=3"
                    "RJ_CONSAN_MAX_PATCHES=${_case_max_patches}"
            )
        endforeach()
    endforeach()
endfunction()

function(rj_add_consan_gfx1250_sim_cases)
    set(_one_value_args PREFIX LDS_TARGET TARGET CONFIG)
    cmake_parse_arguments(PARSE_ARGV 0 _sim "" "${_one_value_args}" "")
    if(_sim_UNPARSED_ARGUMENTS)
        message(
            FATAL_ERROR
            "unknown ConSan gfx1250 simulator arguments: ${_sim_UNPARSED_ARGUMENTS}"
        )
    endif()
    foreach(_required IN ITEMS PREFIX LDS_TARGET TARGET CONFIG)
        if(NOT _sim_${_required})
            message(
                FATAL_ERROR
                "ConSan gfx1250 simulator cases require ${_required}"
            )
        endif()
    endforeach()

    rj_add_consan_common_sim_cases(
        PREFIX "${_sim_PREFIX}"
        LDS_TARGET ${_sim_LDS_TARGET}
        TARGET ${_sim_TARGET}
        CONFIG "${_sim_CONFIG}"

        TEST "${RJ_CONSAN_FORCED_SPILL_TEST}"
        SHARED_TEST "${RJ_CONSAN_SHARED_NOINLINE_HELPER_TEST}"
        ACCESS_KERNEL "forced_spill_preserves_live_vgprs_test_kernel_for_instrumentation"
        SHARED_KERNEL "shared_noinline_lds_helper_for_instrumentation"
        SHARED_INIT_OWNER_EPOCH 0
    )
    rj_add_consan_address_group_sim_cases(
        "${_sim_PREFIX}"
        ${_sim_TARGET}
        "${RJ_CONSAN_ADDRESS_GROUP_TEST}"
        "${RJ_CONSAN_DISJOINT_B128_TEST}"
        "${_sim_CONFIG}"
    )
    rj_add_consan_scalar_spill_exec_cases(
        PREFIX "${_sim_PREFIX}"
        TARGET ${_sim_TARGET}
        CONFIG "${_sim_CONFIG}"
        ARCH_LABEL gfx1250
        CORRECT_TEST
            "ConSanHipTest.ZeroExecScalarSpillCorrectPreservesLiveScalars"
        INCORRECT_TEST
            "ConSanHipTest.NonemptyScalarSpillIncorrectReportsRace"
    )
    rj_add_consan_rdna4_b96_cases(
        PREFIX "${_sim_PREFIX}"
        TARGET ${_sim_TARGET}
        CONFIG "${_sim_CONFIG}"
        TIMEOUT 60
    )

    rj_add_consan_hip_case(
        NAME "${_sim_PREFIX}.SuperColliderZeroExecScalarSpill"
        TARGET ${_sim_LDS_TARGET}
        FILTER "${RJ_CONSAN_LDS_ZERO_EXEC_SCALAR_SPILL_TEST}"
        CONFIG "${_sim_CONFIG}"
        TIMEOUT 60
        RUN_SERIAL
        PASS_REGEX
            "kind=inline-flat-store-check-trap[ -~]*scalar_vcc_spill_vgpr=[0-9]+"
        ENVIRONMENT
            "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
            "RJ_CONSAN_MODE=supercollider"
            "RJ_CONSAN_FLAT_PROVENANCE=strict"
            "RJ_CONSAN_SC_DELAY_MODE=sleep_var"
            "RJ_CONSAN_SC_DELAY_VAR_SSRC=0"
            "RJ_CONSAN_SC_DELAY=1"
            "RJ_CONSAN_TEST_KERNEL_FILTER=padded_zero_exec_flat_lds_store_preserves_scalar_pair_for_instrumentation"
            "RJ_CONSAN_LOG=3"
            "RJ_CONSAN_MAX_PATCHES=1"
            "RJ_CONSAN_REQUIRE_PATCH=1"
    )

    rj_add_consan_hip_case(
        NAME "${_sim_PREFIX}.SuperColliderFullPressureFlatSpill"
        TARGET ${_sim_LDS_TARGET}
        FILTER "${RJ_CONSAN_LDS_FULL_PRESSURE_FLAT_SPILL_TEST}"
        CONFIG "${_sim_CONFIG}"
        TIMEOUT 60
        RUN_SERIAL
        PASS_REGEX
            "kind=[a-z-]*flat-store-check-trap[ -~]*scalar_vcc_spill_vgpr_count=2[ -~]*spilled_vgprs=[1-9][0-9]*"
        ENVIRONMENT
            "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
            "RJ_CONSAN_MODE=supercollider"
            "RJ_CONSAN_FLAT_PROVENANCE=strict"
            "RJ_CONSAN_SC_DELAY_MODE=sleep_var"
            "RJ_CONSAN_SC_DELAY_VAR_SSRC=0"
            "RJ_CONSAN_SC_DELAY=1"
            "RJ_CONSAN_TEST_KERNEL_FILTER=padded_full_pressure_flat_lds_store_preserves_registers_for_instrumentation"
            "RJ_CONSAN_LOG=3"
            "RJ_CONSAN_MAX_PATCHES=1"
            "RJ_CONSAN_REQUIRE_PATCH=1"
    )

    rj_add_consan_hip_case(
        NAME "${_sim_PREFIX}.SuperColliderDynamicStackFullPressureFlatSpill"
        TARGET ${_sim_LDS_TARGET}
        FILTER "${RJ_CONSAN_LDS_DYNAMIC_STACK_FULL_PRESSURE_FLAT_SPILL_TEST}"
        CONFIG "${_sim_CONFIG}"
        TIMEOUT 60
        RUN_SERIAL
        PASS_REGEX
            "kind=[a-z-]*flat-store-check-trap[ -~]*scalar_vcc_spill_vgpr_count=4[ -~]*spilled_vgprs=[1-9][0-9]*"
        ENVIRONMENT
            "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
            "RJ_CONSAN_MODE=supercollider"
            "RJ_CONSAN_FLAT_PROVENANCE=strict"
            "RJ_CONSAN_SC_DELAY_MODE=sleep_var"
            "RJ_CONSAN_SC_DELAY_VAR_SSRC=0"
            "RJ_CONSAN_SC_DELAY=1"
            "RJ_CONSAN_TEST_KERNEL_FILTER=padded_dynamic_stack_full_pressure_flat_lds_store_preserves_registers_for_instrumentation"
            "RJ_CONSAN_LOG=3"
            "RJ_CONSAN_MAX_PATCHES=1"
            "RJ_CONSAN_REQUIRE_PATCH=1"
    )

    rj_add_consan_hip_case(
        NAME "${_sim_PREFIX}.SuperColliderZeroExecNativeLdsScalarSpill"
        TARGET ${_sim_LDS_TARGET}
        FILTER "${RJ_CONSAN_LDS_ZERO_EXEC_NATIVE_SCALAR_SPILL_TEST}"
        CONFIG "${_sim_CONFIG}"
        TIMEOUT 60
        RUN_SERIAL
        PASS_REGEX
            "kind=inline-lds-load-check-trap[ -~]*scalar_vcc_spill_sgpr=[1-9][0-9]*[ -~]*scalar_vcc_spill_vgpr=[0-9]+"
        ENVIRONMENT
            "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
            "RJ_CONSAN_MODE=supercollider"
            "RJ_CONSAN_SC_DELAY_MODE=sleep_var"
            "RJ_CONSAN_SC_DELAY_VAR_SSRC=0"
            "RJ_CONSAN_SC_DELAY=1"
            "RJ_CONSAN_TEST_KERNEL_FILTER=padded_zero_exec_native_lds_load_preserves_allocated_scalars_for_instrumentation"
            "RJ_CONSAN_LOG=3"
            "RJ_CONSAN_MAX_PATCHES=1"
            "RJ_CONSAN_REQUIRE_PATCH=1"
    )

    rj_add_consan_hip_case(
        NAME "${_sim_PREFIX}.RelocatedB128Load"
        TARGET ${_sim_TARGET}
        FILTER
            "ConSanHipTest.RelocatedB128LoadAfterBarrierPreservesLoadedValue"
        CONFIG "${_sim_CONFIG}"
        TIMEOUT 60
        RUN_SERIAL
        ENVIRONMENT
            "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
            "RJ_CONSAN_MODE=default"
            "RJ_CONSAN_AUTO_REPORT_BUFFER_SIZE=1048576"

            "RJ_CONSAN_INIT_OWNER_EPOCH=1"
            "RJ_CONSAN_TRACK_BARRIERS=1"
            "RJ_CONSAN_TRACK_ATOMICS=0"
            "RJ_CONSAN_REQUIRE_RECORDS=1"
            "RJ_CONSAN_TEST_KERNEL_FILTER=relocated_b128_load_after_barrier_for_instrumentation"
            "RJ_CONSAN_LOG=1"
            "RJ_CONSAN_MAX_PATCHES=1"
            "RJ_CONSAN_REQUIRE_PATCH=1"
    )

    # gfx1250 uses the general RDNA fixture, whose focused inline,
    # synchronization, atomic, and dynamic-stack cases differ from CDNA.
    rj_add_consan_hip_case(
        NAME "${_sim_PREFIX}.Atomic"
        TARGET ${_sim_TARGET}
        FILTER "${RJ_CONSAN_SHARED_ATOMIC_HELPER_TEST}"
        CONFIG "${_sim_CONFIG}"
        TIMEOUT 60
        RUN_SERIAL
        PASS_REGEX "visible_sync=[1-9][0-9]*"
        ENVIRONMENT
            "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
            "RJ_CONSAN_MODE=default"
            "RJ_CONSAN_REQUIRE_RECORDS=1"
            "RJ_CONSAN_FORBID_DIAGNOSTICS=1"
            "RJ_CONSAN_TRACK_BARRIERS=0"
            "RJ_CONSAN_TRACK_ATOMICS=1"
            "RJ_CONSAN_TEST_FORCE_VGPR_SPILL=1"
            "RJ_CONSAN_TEST_KERNEL_FILTER=shared_noinline_atomic_add_for_instrumentation"
            "RJ_CONSAN_LOG=1"
            "RJ_CONSAN_MAX_PATCHES=8"
            "RJ_CONSAN_REQUIRE_PATCH=1"
    )

    rj_add_consan_hip_case(
        NAME "${_sim_PREFIX}.DynamicStackForcedSpill"
        TARGET ${_sim_TARGET}
        FILTER "${RJ_CONSAN_DYNAMIC_PRIVATE_STACK_TEST}"
        CONFIG "${_sim_CONFIG}"
        TIMEOUT 60
        RUN_SERIAL
        PASS_REGEX
            "kind=trampoline-[a-z-]*[ -~]*spilled_vgprs=[1-9][0-9]*[ -~]*dynamic_private_addend=[1-9][0-9]*"
        ENVIRONMENT
            "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
            "RJ_CONSAN_MODE=default"
            "RJ_CONSAN_TRACK_BARRIERS=0"
            "RJ_CONSAN_TRACK_ATOMICS=0"
            "RJ_CONSAN_TEST_FORCE_VGPR_SPILL=1"
            "RJ_CONSAN_TEST_KERNEL_FILTER=dynamic_private_stack_test_kernel_for_instrumentation"
            "RJ_CONSAN_LOG=3"
            "RJ_CONSAN_MAX_PATCHES=1"
            "RJ_CONSAN_REQUIRE_PATCH=1"
    )
endfunction()

function(rj_add_consan_rdna3_compact_sim_cases)
    set(_one_value_args PREFIX LDS_TARGET TARGET CONFIG)
    cmake_parse_arguments(PARSE_ARGV 0 _sim "" "${_one_value_args}" "")
    if(_sim_UNPARSED_ARGUMENTS)
        message(
            FATAL_ERROR
            "unknown compact ConSan RDNA3 simulator arguments: ${_sim_UNPARSED_ARGUMENTS}"
        )
    endif()
    foreach(_required IN ITEMS PREFIX LDS_TARGET TARGET CONFIG)
        if(NOT _sim_${_required})
            message(
                FATAL_ERROR
                "compact ConSan RDNA3 simulator cases require ${_required}"
            )
        endif()
    endforeach()

    # This is intentionally a target-native smoke slice rather than the common
    # cross-family simulator matrix. The common matrix depends on generic
    # high-half D16, forced-spill, shared-helper, and exact-byte fixtures that
    # are not RDNA3-native qualification artifacts. Broad transformation
    # coverage stays in the host suite; this slice proves that native RDNA3
    # objects execute ConSan and SuperCollider and attribute a real conflict.

    rj_add_consan_scalar_spill_exec_cases(
        PREFIX "${_sim_PREFIX}"
        TARGET ${_sim_TARGET}
        CONFIG "${_sim_CONFIG}"
        ARCH_LABEL gfx1100
        CORRECT_TEST
            "ConSanRdna3Test.ZeroExecScalarSpillCorrectPreservesLiveScalars"
        INCORRECT_TEST
            "ConSanRdna3Test.NonemptyScalarSpillIncorrectReportsRace"
    )

    rj_add_consan_hip_case(
        NAME "${_sim_PREFIX}.SuperColliderAllSupportedSites"
        TARGET ${_sim_LDS_TARGET}
        FILTER "ConSanLdsTest.PaddedRacyStoreReportsMarker"
        CONFIG "${_sim_CONFIG}"
        TIMEOUT 60
        RUN_SERIAL
        ENVIRONMENT
            "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
            "RJ_CONSAN_MODE=supercollider"
            "RJ_CONSAN_SC_DELAY=2"
            "RJ_CONSAN_LOG=1"
            "RJ_CONSAN_REQUIRE_PATCH=1"
    )

    set(_sim_clean_test
        "ConSanRdna3Test.BarrierOrderedAccessesPreserveOutput"
    )


        rj_add_consan_hip_case(
            NAME "${_sim_PREFIX}.Detection"
            TARGET ${_sim_TARGET}
            FILTER "${_sim_clean_test}"
            CONFIG "${_sim_CONFIG}"
            TIMEOUT 60
            RUN_SERIAL
            PASS_REGEX "kind=trampoline-"
            ENVIRONMENT
                "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                "RJ_CONSAN_MODE=default"
                "RJ_CONSAN_TRACK_BARRIERS=1"
                "RJ_CONSAN_TEST_KERNEL_FILTER=rdna3_barrier_ordered_accesses_for_instrumentation"
                "RJ_CONSAN_LOG=3"
                "RJ_CONSAN_MAX_PATCHES=3"
                "RJ_CONSAN_REQUIRE_PATCH=1"
                "RJ_CONSAN_REQUIRE_RECORDS=1"
                "RJ_CONSAN_FORBID_DIAGNOSTICS=1"
        )


    rj_add_consan_hip_case(
        NAME "${_sim_PREFIX}.DetectsRace"
        TARGET ${_sim_TARGET}
        FILTER "ConSanRdna3Test.TwoWaveRacyStoresComplete"
        CONFIG "${_sim_CONFIG}"
        TIMEOUT 60
        RUN_SERIAL
        PASS_REGEX
            "conflicts=[1-9][0-9]*"
        ENVIRONMENT
            "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
            "RJ_CONSAN_MODE=default"
            "RJ_CONSAN_TRACK_BARRIERS=0"
            "RJ_CONSAN_TRACK_ATOMICS=0"
            "RJ_CONSAN_TEST_KERNEL_FILTER=rdna3_two_wave_racy_stores_for_instrumentation"
            "RJ_CONSAN_LOG=2"
            "RJ_CONSAN_MAX_PATCHES=1"
            "RJ_CONSAN_REQUIRE_PATCH=1"
            "RJ_CONSAN_REQUIRE_RECORDS=1"
            "RJ_CONSAN_REQUIRE_DIAGNOSTICS=1"
    )
endfunction()

function(rj_add_consan_fuzzers)
    if(RJ_BUILD_FUZZERS)
        set(_rj_consan_transform_fuzz_targets)
        foreach(
            _rj_profile
            IN
            ITEMS supercollider default
        )
            set(_rj_fuzz_target "consan_transform_${_rj_profile}_fuzz")
            add_executable(${_rj_fuzz_target} fuzz/consan_transform_fuzz.cpp)
            target_link_libraries(
                ${_rj_fuzz_target}
                PRIVATE
                    ${RJ_OBJECT_LIBS}
                    rocjitsu_kmd
                    util
                    flatbuffers
                    Threads::Threads
            )
            string(TOUPPER "${_rj_profile}" _rj_profile_upper)
            target_compile_definitions(
                ${_rj_fuzz_target}
                PRIVATE "RJ_CONSAN_FUZZ_PROFILE_${_rj_profile_upper}=1"
            )
            target_compile_options(${_rj_fuzz_target} PRIVATE -fsanitize=fuzzer)
            target_link_options(${_rj_fuzz_target} PRIVATE -fsanitize=fuzzer)
            rj_configure_target(${_rj_fuzz_target} TEST)
            list(APPEND _rj_consan_transform_fuzz_targets ${_rj_fuzz_target})
        endforeach()
        add_custom_target(
            consan_transform_fuzz
            DEPENDS ${_rj_consan_transform_fuzz_targets}
        )

        add_executable(consan_placement_fuzz fuzz/consan_placement_fuzz.cpp)
        target_link_libraries(
            consan_placement_fuzz
            PRIVATE
                ${RJ_OBJECT_LIBS}
                rocjitsu_kmd
                util
                flatbuffers
                Threads::Threads
        )
        target_compile_options(consan_placement_fuzz PRIVATE -fsanitize=fuzzer)
        target_link_options(consan_placement_fuzz PRIVATE -fsanitize=fuzzer)
        rj_configure_target(consan_placement_fuzz TEST)
    endif()
endfunction()

function(rj_add_consan_hip_tests)
    set(_rj_consan_include_dir
        "${CMAKE_CURRENT_SOURCE_DIR}/../lib/rocjitsu/src"
    )
    set(_rj_consan_abi_header
        "${_rj_consan_include_dir}/rocjitsu/code/patch/consan/consan_abi.h"
    )
    set(_rj_consan_hip_fixture_headers
        "${_rj_consan_abi_header};${CMAKE_CURRENT_SOURCE_DIR}/dbi/consan/hip_consan_address_group_test.h;${CMAKE_CURRENT_SOURCE_DIR}/dbi/consan/hip_consan_empty_exec_scalar_spill.h"
    )
    if(RJ_HIP_RUNTIME_SUPPORTS_GFX942)
        rj_add_consan_device_fixture_targets(gfx942)
        rj_add_consan_device_conformance_cases(
            PREFIX "ConSanDeviceGfx942Sim"
            TARGET gfx942
            CONFIG "${CMAKE_SOURCE_DIR}/configs/gfx942_cdna3_kmd.json"
        )
        # Keep a native gfx942 code object available on every host so CDNA3
        # ConSan can be exercised through the RocJITsu simulator.
        rj_add_hip_test(
                hip_consan_lds_gfx942_test
                ARCH gfx942
                SOURCE dbi/consan/hip_consan_lds_test.hip
        )
        rj_add_hip_test(
                hip_consan_gfx942_test
                ARCH gfx942
                SOURCE dbi/consan/hip_consan_cdna_test.hip
                INCLUDE_DIRS "${_rj_consan_include_dir}"
                EXTRA_DEPENDS "${_rj_consan_hip_fixture_headers}"
        )
        add_dependencies(hip_consan_lds_gfx942_test_target rocjitsu_dbi_hooks)
        add_dependencies(hip_consan_gfx942_test_target rocjitsu_dbi_hooks)
        rj_add_consan_cdna_sim_cases(
            PREFIX "ConSanGfx942Sim"
            LDS_TARGET hip_consan_lds_gfx942_test
            TARGET hip_consan_gfx942_test
            CONFIG "${CMAKE_SOURCE_DIR}/configs/gfx942_cdna3_kmd.json"
            ARCH_LABEL gfx942
        )
    endif()
    if(RJ_HIP_RUNTIME_SUPPORTS_GFX950)
        rj_add_consan_device_fixture_targets(gfx950)
        rj_add_consan_device_conformance_cases(
            PREFIX "ConSanDeviceGfx950Sim"
            TARGET gfx950
            CONFIG "${CMAKE_SOURCE_DIR}/configs/gfx950_mi355x_kmd.json"
        )
        rj_add_hip_test(
                hip_consan_lds_gfx950_test
                ARCH gfx950
                SOURCE dbi/consan/hip_consan_lds_test.hip
        )
        rj_add_hip_test(
                hip_consan_spill_gfx950_test
                ARCH gfx950
                SOURCE dbi/consan/hip_consan_spill_gfx950_test.hip
        )
        rj_add_hip_test(
                hip_consan_gfx950_test
                ARCH gfx950
                SOURCE dbi/consan/hip_consan_cdna_test.hip
                INCLUDE_DIRS "${_rj_consan_include_dir}"
                EXTRA_DEPENDS "${_rj_consan_hip_fixture_headers}"
        )
        foreach(_spill_case IN ITEMS Gfx950VgprScratchRoundTrip
                Gfx950PartialExecVgprScratchRoundTrip Gfx950RepeatedScratchRoundTrips)
            rj_add_consan_hip_case(
                NAME "ConSanSpillGfx950Sim.${_spill_case}"
                TARGET hip_consan_spill_gfx950_test
                FILTER "ConSanSpillHipTest.${_spill_case}"
                CONFIG "${CMAKE_SOURCE_DIR}/configs/gfx950_mi355x_kmd.json"
                LABELS consan consan-device gfx950 simulator
            )
        endforeach()
        add_dependencies(hip_consan_lds_gfx950_test_target rocjitsu_dbi_hooks)
        add_dependencies(hip_consan_gfx950_test_target rocjitsu_dbi_hooks)
        rj_add_consan_cdna_sim_cases(
            PREFIX "ConSanGfx950Sim"
            LDS_TARGET hip_consan_lds_gfx950_test
            TARGET hip_consan_gfx950_test
            CONFIG "${CMAKE_SOURCE_DIR}/configs/gfx950_mi355x_kmd.json"
            ARCH_LABEL gfx950
        )
        # Keep cross-compiled gfx950 fixtures available on other hosts,
        # but only register native HIP execution when gfx950 is present.
        if(HAS_GFX950_GPU)
            rj_add_consan_device_conformance_cases(
                PREFIX "ConSanDeviceGfx950Physical"
                TARGET gfx950
                PHYSICAL
            )
            rj_add_consan_scalar_spill_exec_cases(
                PREFIX "ConSanGfx950Physical"
                TARGET hip_consan_gfx950_test
                ARCH_LABEL gfx950
                CORRECT_TEST
                    "${RJ_CONSAN_CDNA_ZERO_EXEC_SCALAR_SPILL_CORRECT_TEST}"
                INCORRECT_TEST
                    "${RJ_CONSAN_CDNA_NONEMPTY_SCALAR_SPILL_INCORRECT_TEST}"
                PHYSICAL
            )
            # Successful per-site resource plans are intentionally summarized
            # to keep normal hook logging bounded. Spill cases match the
            # aggregate emission proof or a typed patch record at log level 3.
            foreach(_spill_case IN ITEMS Gfx950VgprScratchRoundTrip
                    Gfx950PartialExecVgprScratchRoundTrip Gfx950RepeatedScratchRoundTrips)
                rj_add_consan_hip_case(
                    NAME "ConSanSpillHipTest.${_spill_case}"
                    TARGET hip_consan_spill_gfx950_test
                    FILTER "ConSanSpillHipTest.${_spill_case}"
                    LABELS consan consan-device gfx950 physical
                    RESOURCE_LOCK consan_physical_gfx950
                    RUN_SERIAL
                )
            endforeach()

            rj_add_consan_hip_case(
                NAME "ConSanCdnaTest.DbiForcedSpill"
                TARGET hip_consan_gfx950_test
                FILTER "${RJ_CONSAN_CDNA_FORCED_SPILL_TEST}"
                TIMEOUT 30
                RUN_SERIAL
                PASS_REGEX "ConSan resources.*spill=[1-9][0-9]*.*emitted_spill_patches=[1-9][0-9]*"
                ENVIRONMENT
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=default"
                    "RJ_CONSAN_TRACK_BARRIERS=0"
                    "RJ_CONSAN_TRACK_ATOMICS=0"
                    "RJ_CONSAN_TEST_FORCE_VGPR_SPILL=1"
                    "RJ_CONSAN_TEST_KERNEL_FILTER=cdna_forced_spill_test_kernel_for_instrumentation"
                    "RJ_CONSAN_LOG=3"
                    "RJ_CONSAN_MAX_PATCHES=1"
                    "RJ_CONSAN_REQUIRE_PATCH=1"
            )

            rj_add_consan_hip_case(
                NAME "ConSanCdnaTest.DbiSharedHelperForcedSpill"
                TARGET hip_consan_gfx950_test
                FILTER "${RJ_CONSAN_CDNA_SHARED_HELPER_TEST}"
                TIMEOUT 30
                RUN_SERIAL
                PASS_REGEX "ConSan resources.*spill=[1-9][0-9]*.*emitted_spill_patches=[1-9][0-9]*"
                ENVIRONMENT
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=default"
                    "RJ_CONSAN_TRACK_BARRIERS=0"
                    "RJ_CONSAN_TRACK_ATOMICS=0"
                    "RJ_CONSAN_TEST_FORCE_VGPR_SPILL=1"
                    "RJ_CONSAN_TEST_KERNEL_FILTER=cdna_shared_lds_helper_for_instrumentation"
                    "RJ_CONSAN_LOG=3"
                    "RJ_CONSAN_MAX_PATCHES=1"
                    "RJ_CONSAN_REQUIRE_PATCH=1"
            )
            rj_add_consan_hip_case(
                NAME "ConSanCdnaTest.DbiBarrierForcedSpill"
                TARGET hip_consan_gfx950_test
                FILTER "${RJ_CONSAN_CDNA_BARRIER_TEST}"
                TIMEOUT 30
                RUN_SERIAL
                PASS_REGEX "ConSan proof patch.*kind=trampoline-sync-metadata.*spilled_vgprs=[1-9][0-9]*"
                ENVIRONMENT
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=default"
                    "RJ_CONSAN_INIT_OWNER_EPOCH=1"
                    "RJ_CONSAN_TRACK_BARRIERS=1"
                    "RJ_CONSAN_TRACK_ATOMICS=0"
                    "RJ_CONSAN_TEST_FORCE_VGPR_SPILL=1"
                    "RJ_CONSAN_TEST_KERNEL_FILTER=cdna_barrier_forced_spill_kernel_for_instrumentation"
                    "RJ_CONSAN_LOG=3"
                    "RJ_CONSAN_MAX_PATCHES=8"
                    "RJ_CONSAN_REQUIRE_PATCH=1"
            )
            rj_add_consan_hip_case(
                NAME "ConSanCdnaTest.DbiAtomicForcedSpill"
                TARGET hip_consan_gfx950_test
                FILTER "${RJ_CONSAN_CDNA_ATOMIC_TEST}"
                TIMEOUT 30
                RUN_SERIAL
                PASS_REGEX "ConSan proof patch.*kind=trampoline-sync-metadata.*spilled_vgprs=[1-9][0-9]*"
                ENVIRONMENT
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=default"
                    "RJ_CONSAN_TRACK_BARRIERS=0"
                    "RJ_CONSAN_TRACK_ATOMICS=1"
                    "RJ_CONSAN_TEST_FORCE_VGPR_SPILL=1"
                    "RJ_CONSAN_TEST_KERNEL_FILTER=cdna_atomic_forced_spill_kernel_for_instrumentation"
                    "RJ_CONSAN_LOG=3"
                    "RJ_CONSAN_MAX_PATCHES=8"
                    "RJ_CONSAN_REQUIRE_PATCH=1"
            )
            rj_add_consan_hip_case(
                NAME "ConSanCdnaTest.DbiDynamicStackForcedSpill"
                TARGET hip_consan_gfx950_test
                FILTER "${RJ_CONSAN_CDNA_DYNAMIC_STACK_TEST}"
                TIMEOUT 30
                RUN_SERIAL
                PASS_REGEX "ConSan resources.*spill=[1-9][0-9]*.*emitted_spill_patches=[1-9][0-9]*"
                ENVIRONMENT
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=default"
                    "RJ_CONSAN_TRACK_BARRIERS=0"
                    "RJ_CONSAN_TRACK_ATOMICS=0"
                    "RJ_CONSAN_REQUIRE_RECORDS=1"
                    "RJ_CONSAN_REQUIRE_DIAGNOSTICS=1"
                    "RJ_CONSAN_TEST_FORCE_VGPR_SPILL=1"
                    "RJ_CONSAN_TEST_KERNEL_FILTER=cdna_dynamic_stack_spill_kernel_for_instrumentation"
                    "RJ_CONSAN_LOG=3"
                    "RJ_CONSAN_MAX_PATCHES=1"
                    "RJ_CONSAN_REQUIRE_PATCH=1"
            )
        endif()
    endif()
    if(RJ_HIP_RUNTIME_SUPPORTS_GFX1100)
        rj_add_consan_device_fixture_targets(gfx1100)
        rj_add_consan_device_conformance_cases(
            PREFIX "ConSanDeviceGfx1100Sim"
            TARGET gfx1100
            CONFIG "${CMAKE_SOURCE_DIR}/configs/gfx1100_w7900.json"
        )
        rj_add_hip_test(
                hip_consan_lds_gfx1100_test
                ARCH gfx1100
                SOURCE dbi/consan/hip_consan_lds_test.hip
        )
        rj_add_hip_test(
                hip_consan_gfx1100_test
                ARCH gfx1100
                SOURCE dbi/consan/hip_consan_rdna3_test.hip
                EXTRA_DEPENDS "${_rj_consan_hip_fixture_headers}"
        )
        add_dependencies(hip_consan_lds_gfx1100_test_target rocjitsu_dbi_hooks)
        add_dependencies(hip_consan_gfx1100_test_target rocjitsu_dbi_hooks)
        rj_add_consan_rdna3_compact_sim_cases(
            PREFIX "ConSanGfx1100Sim"
            LDS_TARGET hip_consan_lds_gfx1100_test
            TARGET hip_consan_gfx1100_test
            CONFIG "${CMAKE_SOURCE_DIR}/configs/gfx1100_w7900.json"
        )
        if(HAS_GFX1100_GPU AND GFX1100_GPU_UUID)
            set(_rj_consan_gfx1100_device
                "ROCR_VISIBLE_DEVICES=${GFX1100_GPU_UUID}"
            )
            set(_rj_consan_gfx1100_clean
                "ConSanRdna3Test.BarrierOrderedAccessesPreserveOutput"
            )
            rj_add_consan_hip_case(
                NAME "ConSanGfx1100Physical.BaselineClean"
                TARGET hip_consan_gfx1100_test
                FILTER "${_rj_consan_gfx1100_clean}"
                RUN_SERIAL
                ENVIRONMENT "${_rj_consan_gfx1100_device}"
            )
            rj_add_consan_hip_case(
                NAME "ConSanGfx1100Physical.SuperColliderClean"
                TARGET hip_consan_gfx1100_test
                FILTER "${_rj_consan_gfx1100_clean}"
                RUN_SERIAL
                ENVIRONMENT
                    "${_rj_consan_gfx1100_device}"
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=supercollider"
                    "RJ_CONSAN_SC_DELAY=2"
                    "RJ_CONSAN_TEST_KERNEL_FILTER=rdna3_barrier_ordered_accesses_for_instrumentation"
                    "RJ_CONSAN_LOG=1"
                    "RJ_CONSAN_MAX_PATCHES=2"
                    "RJ_CONSAN_REQUIRE_PATCH=1"
            )
            rj_add_consan_hip_case(
                NAME "ConSanGfx1100Physical.SuperColliderAllSupportedSites"
                TARGET hip_consan_lds_gfx1100_test
                FILTER "ConSanLdsTest.PaddedRacyStoreReportsMarker"
                RUN_SERIAL
                ENVIRONMENT
                    "${_rj_consan_gfx1100_device}"
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=supercollider"
                    "RJ_CONSAN_SC_DELAY=2"
                    "RJ_CONSAN_LOG=1"
                    "RJ_CONSAN_REQUIRE_PATCH=1"
            )


                rj_add_consan_hip_case(
                    NAME "ConSanGfx1100Physical.Detection"
                    TARGET hip_consan_gfx1100_test
                    FILTER "${_rj_consan_gfx1100_clean}"
                    PASS_REGEX "kind=trampoline-"
                    RUN_SERIAL
                    ENVIRONMENT
                        "${_rj_consan_gfx1100_device}"
                        "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                        "RJ_CONSAN_MODE=default"
                        "RJ_CONSAN_TRACK_BARRIERS=1"
                        "RJ_CONSAN_TEST_KERNEL_FILTER=rdna3_barrier_ordered_accesses_for_instrumentation"
                        "RJ_CONSAN_LOG=3"
                        "RJ_CONSAN_MAX_PATCHES=3"
                        "RJ_CONSAN_REQUIRE_PATCH=1"
                        "RJ_CONSAN_REQUIRE_RECORDS=1"
                        "RJ_CONSAN_FORBID_DIAGNOSTICS=1"
                )

            rj_add_consan_hip_case(
                NAME "ConSanGfx1100Physical.DetectsRace"
                TARGET hip_consan_gfx1100_test
                FILTER "ConSanRdna3Test.TwoWaveRacyStoresComplete"
                PASS_REGEX "conflicts=[1-9][0-9]*"
                RUN_SERIAL
                ENVIRONMENT
                    "${_rj_consan_gfx1100_device}"
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=default"
                    "RJ_CONSAN_TRACK_BARRIERS=0"
                    "RJ_CONSAN_TRACK_ATOMICS=0"
                    "RJ_CONSAN_TEST_KERNEL_FILTER=rdna3_two_wave_racy_stores_for_instrumentation"
                    "RJ_CONSAN_LOG=2"
                    "RJ_CONSAN_MAX_PATCHES=1"
                    "RJ_CONSAN_REQUIRE_PATCH=1"
                    "RJ_CONSAN_REQUIRE_RECORDS=1"
                    "RJ_CONSAN_REQUIRE_DIAGNOSTICS=1"
            )
            rj_add_consan_hip_case(
                NAME "ConSanGfx1100Physical.PostInstrumentationHealth"
                TARGET hip_consan_gfx1100_test
                FILTER "${_rj_consan_gfx1100_clean}"
                RUN_SERIAL
                ENVIRONMENT "${_rj_consan_gfx1100_device}"
            )
            set_tests_properties(
                "ConSanGfx1100Physical.PostInstrumentationHealth"
                PROPERTIES
                    DEPENDS
                        "ConSanGfx1100Physical.SuperColliderClean;ConSanGfx1100Physical.SuperColliderAllSupportedSites;ConSanGfx1100Physical.Detection;ConSanGfx1100Physical.DetectsRace"
            )
            unset(_rj_consan_gfx1100_clean)
            unset(_rj_consan_gfx1100_device)
        elseif(HAS_GFX1100_GPU)
            message(
                WARNING
                "gfx1100 hardware was detected, but its stable UUID could not be resolved; physical ConSan tests will not be registered"
            )
        endif()
    endif()
    if(RJ_HIP_RUNTIME_SUPPORTS_GFX1250)
        rj_add_consan_device_fixture_targets(gfx1250)
        rj_add_consan_device_conformance_cases(
            PREFIX "ConSanDeviceGfx1250Sim"
            TARGET gfx1250
            CONFIG "${CMAKE_SOURCE_DIR}/configs/gfx1250_mi455x.json"
        )
        # Keep native gfx1250 code objects available on every host so the
        # target-specific ConSan path can be regression-tested in simulation.
        rj_add_hip_test(
                hip_consan_lds_gfx1250_test
                ARCH gfx1250
                SOURCE dbi/consan/hip_consan_lds_test.hip
        )
        rj_add_hip_test(
                hip_consan_gfx1250_test
                ARCH gfx1250
                SOURCE dbi/consan/hip_consan_test.hip
                INCLUDE_DIRS "${_rj_consan_include_dir}"
                EXTRA_DEPENDS "${_rj_consan_hip_fixture_headers}"
        )
        add_dependencies(hip_consan_lds_gfx1250_test_target rocjitsu_dbi_hooks)
        add_dependencies(hip_consan_gfx1250_test_target rocjitsu_dbi_hooks)
        rj_add_consan_gfx1250_sim_cases(
            PREFIX "ConSanGfx1250Sim"
            LDS_TARGET hip_consan_lds_gfx1250_test
            TARGET hip_consan_gfx1250_test
            CONFIG "${CMAKE_SOURCE_DIR}/configs/gfx1250_mi455x.json"
        )
    endif()
    rj_add_consan_device_fixture_targets(gfx1201)
    rj_add_consan_device_conformance_cases(
        PREFIX "ConSanDeviceGfx1201Sim"
        TARGET gfx1201
        CONFIG "${CMAKE_SOURCE_DIR}/configs/gfx1201_r9700.json"
    )
    rj_add_hip_test(
            hip_consan_lds_test
            ARCH gfx1201
            SOURCE dbi/consan/hip_consan_lds_test.hip
    )
    add_dependencies(hip_consan_lds_test_target rocjitsu_dbi_hooks)
    rj_add_hip_test(
            hip_consan_lds_diagnostics_test
            ARCH gfx1201
            SOURCE dbi/consan/hip_consan_lds_diagnostics_test.hip
    )
    add_dependencies(hip_consan_lds_diagnostics_test_target rocjitsu_dbi_hooks)
    rj_add_hip_test(
            hip_consan_test
            ARCH gfx1201
            SOURCE dbi/consan/hip_consan_test.hip
            INCLUDE_DIRS "${_rj_consan_include_dir}"
            # The packaged gfx1250 runtime translator processes the complete
            # test object before gtest filtering and cannot yet translate this
            # staged DS_LOAD_ADDTID sequence. Registration filtering alone is
            # therefore insufficient to keep the physical-only discriminator
            # out of the aggregate gfx1250 object.
            COMPILE_DEFINITIONS RJ_CONSAN_ENABLE_STAGED_DSCNT_TEST=1
            EXTRA_DEPENDS "${_rj_consan_hip_fixture_headers}"
    )
    add_dependencies(hip_consan_test_target rocjitsu_dbi_hooks)
    rj_add_consan_scalar_spill_exec_cases(
        PREFIX "ConSanGfx1201Sim"
        TARGET hip_consan_test
        CONFIG "${CMAKE_SOURCE_DIR}/configs/gfx1201_r9700.json"
        ARCH_LABEL gfx1201
        CORRECT_TEST
            "ConSanHipTest.ZeroExecScalarSpillCorrectPreservesLiveScalars"
        INCORRECT_TEST
            "ConSanHipTest.NonemptyScalarSpillIncorrectReportsRace"
    )
    # Deliberately ill-formed GPU synchronization. The gtests are disabled
    # by default and must only run through consan_fault_runner.py with the
    # destructive opt-in, timeout, health checks, and quarantine enabled.
    rj_add_hip_test(
            hip_consan_malformed_sync_test
            ARCH gfx1201
            SOURCE dbi/consan/hip_consan_malformed_sync_test.hip
    )
    # Keep a small RDNA4 execution gate independent of physical-host
    # availability. The zero-EXEC case specifically exercises the fixed-lane
    # scalar preservation used by SuperCollider's full-pressure fallback.
    rj_add_consan_hip_case(
        NAME "ConSanGfx1201Sim.SuperColliderClean"
        TARGET hip_consan_lds_test
        FILTER "${RJ_CONSAN_LDS_CLEAN_TEST}"
        CONFIG "${CMAKE_SOURCE_DIR}/configs/gfx1201_r9700.json"
        TIMEOUT 60
        RUN_SERIAL
        PASS_REGEX "\\[  PASSED  \\] 1 test"
        ENVIRONMENT
            "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
            "RJ_CONSAN_MODE=supercollider"
            "RJ_CONSAN_LOG=1"
            "RJ_CONSAN_SC_DELAY=2"
            "RJ_CONSAN_TMP_VGPR=3"
            "RJ_CONSAN_REQUIRE_PATCH=1"
    )
    rj_add_consan_hip_case(
        NAME "ConSanGfx1201Sim.SuperColliderZeroExecNativeLdsScalarSpill"
        TARGET hip_consan_lds_test
        FILTER "${RJ_CONSAN_LDS_ZERO_EXEC_NATIVE_SCALAR_SPILL_TEST}"
        CONFIG "${CMAKE_SOURCE_DIR}/configs/gfx1201_r9700.json"
        TIMEOUT 60
        RUN_SERIAL
        PASS_REGEX
            "kind=inline-lds-load-check-trap[ -~]*scalar_vcc_spill_sgpr=[1-9][0-9]*[ -~]*scalar_vcc_spill_vgpr=[0-9]+"
        ENVIRONMENT
            "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
            "RJ_CONSAN_MODE=supercollider"
            "RJ_CONSAN_SC_DELAY_MODE=sleep_var"
            "RJ_CONSAN_SC_DELAY_VAR_SSRC=0"
            "RJ_CONSAN_SC_DELAY=1"
            "RJ_CONSAN_TEST_KERNEL_FILTER=padded_zero_exec_native_lds_load_preserves_allocated_scalars_for_instrumentation"
            "RJ_CONSAN_LOG=3"
            "RJ_CONSAN_MAX_PATCHES=1"
            "RJ_CONSAN_REQUIRE_PATCH=1"
    )
    rj_add_consan_byte_overlap_cases(
        "ConSanGfx1201Sim"
        hip_consan_lds_test
        "${CMAKE_SOURCE_DIR}/configs/gfx1201_r9700.json"
    )
    rj_add_consan_multidimensional_cases(
        "ConSanGfx1201Sim"
        hip_consan_lds_test
        "${CMAKE_SOURCE_DIR}/configs/gfx1201_r9700.json"
    )
    if(HAS_GFX1201_GPU)
        # Every case registered in this scope executes against the native
        # gfx1201 device. Supply the common physical labels and target-scoped
        # resource lock to direct cases and nested registration helpers alike.
        set(_rj_consan_physical_target gfx1201)
        rj_add_consan_device_conformance_cases(
            PREFIX "ConSanDeviceGfx1201Physical"
            TARGET gfx1201
            PHYSICAL
        )
        rj_add_consan_scalar_spill_exec_cases(
            PREFIX "ConSanGfx1201Physical"
            TARGET hip_consan_test
            ARCH_LABEL gfx1201
            CORRECT_TEST
                "ConSanHipTest.ZeroExecScalarSpillCorrectPreservesLiveScalars"
            INCORRECT_TEST
                "ConSanHipTest.NonemptyScalarSpillIncorrectReportsRace"
            PHYSICAL
        )
        rj_add_consan_byte_overlap_cases(
            "ConSanGfx1201Physical"
            hip_consan_lds_test
            ""
        )
        rj_add_consan_multidimensional_cases(
            "ConSanGfx1201Physical"
            hip_consan_lds_test
            ""
        )
        set(RJ_CONSAN_LDS_RACY_TEST
            "ConSanLdsTest.PaddedRacyStoreCompletesOrTripsDbiTrap"
        )
        set(RJ_CONSAN_LDS_CLEAN_REPORT_TEST
            "ConSanLdsTest.PaddedCleanStoreReportBufferStaysZero"
        )
        set(RJ_CONSAN_LDS_RACY_REPORT_TEST
            "ConSanLdsTest.PaddedRacyStoreReportsMarker"
        )
        set(RJ_CONSAN_LDS_DIAGNOSTICS_RACY_TEST
            "ConSanLdsDiagnosticsTest.CrossWaveRacyStoreCompletes"
        )
        set(RJ_CONSAN_LDS_DIAGNOSTICS_B128_RACY_TEST
            "ConSanLdsDiagnosticsTest.CrossWaveRacyB128StoreCompletes"
        )
        set(RJ_CONSAN_LDS_DIAGNOSTICS_FLAT_RACY_TEST
            "ConSanLdsDiagnosticsTest.CrossWaveRacyFlatLdsStoreCompletes"
        )
        set(RJ_CONSAN_LDS_DIAGNOSTICS_BARRIER_TEST
            "ConSanLdsDiagnosticsTest.BarrierOrderedCrossWaveStoreCompletes"
        )
        set(RJ_CONSAN_LDS_DIAGNOSTICS_FENCE_BARRIER_TEST
            "ConSanLdsDiagnosticsTest.FenceAndBarrierOrderedAccessesPreserveExactOutput"
        )
        set(RJ_CONSAN_LDS_DIAGNOSTICS_ATOMIC_SAME_ADDRESS_TEST
            "ConSanLdsDiagnosticsTest.SameAddressAtomicHandoffCompletes"
        )
        set(RJ_CONSAN_LDS_DIAGNOSTICS_ATOMIC_WRONG_ADDRESS_TEST
            "ConSanLdsDiagnosticsTest.WrongAddressAtomicHandoffCompletes"
        )

        set(RJ_CONSAN_LDS_DIAGNOSTICS_ATOMIC_CAS_TEST
            "ConSanLdsDiagnosticsTest.FailedCasDoesNotReplaceReleaseCompletes"
        )

        set(RJ_CONSAN_LDS_DIRECT_TEST
            "ConSanHipTest.BackendWritesDirectEntries"
        )

        set(RJ_CONSAN_LDS_AUTO_REPLAY_CONFLICT_TEST
            "ConSanHipTest.TwoWaveAccessesRunWithAutoReportBuffer"
        )
        set(RJ_CONSAN_LDS_AUTO_REPEATED_DISPATCH_TEST
            "ConSanHipTest.RepeatedDispatchesRetainAutomaticRecords"
        )
        set(RJ_CONSAN_LDS_SATURATION_OUTPUT_TEST
            "ConSanHipTest.SaturationPreservesRepeatedDispatchOutput"
        )
        set(RJ_CONSAN_LDS_STAGED_DSCNT_TEST
            "ConSanHipTest.StagedDscntLoadPreservesOutput"
        )

        set(RJ_CONSAN_LDS_WORKGROUP_2D_RECORD_TEST
            "ConSanHipTest.EquivalentAccessesInDifferent2DWorkgroupsDoNotConflict"
        )
        set(RJ_CONSAN_LDS_WORKGROUP_3D_RECORD_TEST
            "ConSanHipTest.EquivalentAccessesInDifferent3DWorkgroupsDoNotConflict"
        )
        set(RJ_CONSAN_LDS_LIMITED_RECORD_TEST
            "ConSanHipTest.OneSlotPreservesBufferBounds"
        )

        set(RJ_CONSAN_GFX1201_VGPR_SPILL_TEST
            "ConSanSpillHipTest.Gfx1201VgprScratchRoundTrip"
        )
        set(RJ_CONSAN_FULL_VGPR_FILE_TEST
            "ConSanHipTest.FullVgprFileForcedSpillPreservesOutput"
        )
        set(RJ_CONSAN_STATIC_PRIVATE_STACK_TEST
            "ConSanHipTest.StaticPrivateStackForcedSpillPreservesOutput"
        )
        rj_add_consan_rdna4_b96_cases(
            PREFIX "ConSanHipTest"
            TARGET hip_consan_test
            TIMEOUT 30
        )
        rj_add_consan_hip_case(
                NAME "ConSanSpillHipTest.Gfx1201VgprScratchRoundTrip"
                TARGET hip_consan_test
                FILTER "${RJ_CONSAN_GFX1201_VGPR_SPILL_TEST}"
                TIMEOUT 30
        )

        rj_add_consan_hip_case(
                NAME "ConSanHipTest.DbiForcedSpillPreservesLiveVgprs"
                TARGET hip_consan_test
                FILTER "${RJ_CONSAN_FORCED_SPILL_TEST}"
                TIMEOUT 30
                ENVIRONMENT
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=default"
                    "RJ_CONSAN_TRACK_BARRIERS=0"
                    "RJ_CONSAN_TRACK_ATOMICS=0"
                    "RJ_CONSAN_TEST_FORCE_VGPR_SPILL=1"
                    "RJ_CONSAN_TEST_KERNEL_FILTER=forced_spill_preserves_live_vgprs_test_kernel_for_instrumentation"
                    "RJ_CONSAN_LOG=1"
                    "RJ_CONSAN_MAX_PATCHES=1"
                    "RJ_CONSAN_REQUIRE_PATCH=1"
        )
        rj_add_consan_hip_case(
                NAME "ConSanHipTest.BaselineFullVgprFilePreservesOutput"
                TARGET hip_consan_test
                FILTER "${RJ_CONSAN_FULL_VGPR_FILE_TEST}"
                TIMEOUT 30
        )
        # Successful resource plans are intentionally summarized to keep hook
        # logging bounded. These cases match aggregate emission proofs at log
        # level 3 instead of depending on removed per-site success dumps.
        rj_add_consan_hip_case(
                NAME "ConSanHipTest.DbiFullVgprFileForcedSpillPreservesOutput"
                TARGET hip_consan_test
                FILTER "${RJ_CONSAN_FULL_VGPR_FILE_TEST}"
                TIMEOUT 30
                PASS_REGEX
                    "ConSan resources.*spill=[1-9][0-9]*.*emitted_spill_patches=[1-9][0-9]*"
                ENVIRONMENT
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=default"
                    "RJ_CONSAN_INIT_OWNER_EPOCH=0"
                    "RJ_CONSAN_TEST_FORCE_VGPR_SPILL=1"
                    "RJ_CONSAN_TEST_KERNEL_FILTER=full_vgpr_file_test_kernel_for_instrumentation"
                    "RJ_CONSAN_LOG=3"
                    "RJ_CONSAN_MAX_PATCHES=1"
                    "RJ_CONSAN_REQUIRE_PATCH=1"
        )
        rj_add_consan_hip_case(
                NAME "ConSanHipTest.DbiSharedNoinlineHelperPatchPreservesOutput"
                TARGET hip_consan_test
                FILTER "${RJ_CONSAN_SHARED_NOINLINE_HELPER_TEST}"
                TIMEOUT 30
                PASS_REGEX "ConSan resources.*descriptor_growth=[1-9][0-9]*"
                ENVIRONMENT
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=default"
                    "RJ_CONSAN_TEST_KERNEL_FILTER=shared_noinline_lds_helper_for_instrumentation"
                    "RJ_CONSAN_LOG=3"
                    "RJ_CONSAN_MAX_PATCHES=1"
                    "RJ_CONSAN_REQUIRE_PATCH=1"
        )
        rj_add_consan_hip_case(
                NAME "ConSanHipTest.DbiSharedNoinlineHelperSpillPreservesOutput"
                TARGET hip_consan_test
                FILTER "${RJ_CONSAN_SHARED_NOINLINE_HELPER_TEST}"
                TIMEOUT 30
                PASS_REGEX "ConSan resources.*spill=[1-9][0-9]*.*emitted_spill_patches=[1-9][0-9]*"
                ENVIRONMENT
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=default"
                    "RJ_CONSAN_INIT_OWNER_EPOCH=0"
                    "RJ_CONSAN_TEST_FORCE_VGPR_SPILL=1"
                    "RJ_CONSAN_TEST_KERNEL_FILTER=shared_noinline_lds_helper_for_instrumentation"
                    "RJ_CONSAN_LOG=3"
                    "RJ_CONSAN_MAX_PATCHES=1"
                    "RJ_CONSAN_REQUIRE_PATCH=1"
        )
        rj_add_consan_hip_case(
                NAME "ConSanHipTest.DbiSharedAtomicHelperForcedSpill"
                TARGET hip_consan_test
                FILTER "${RJ_CONSAN_SHARED_ATOMIC_HELPER_TEST}"
                TIMEOUT 30
                PASS_REGEX "ConSan resources.*spill=[1-9][0-9]*.*emitted_spill_patches=[1-9][0-9]*"
                ALSO_PASS_REGEX
                    "kind=trampoline-sync-metadata.*spilled_vgprs=[1-9]"
                    "visible_sync=[1-9]"
                ENVIRONMENT
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=default"
                    "RJ_CONSAN_TRACK_ATOMICS=1"
                    "RJ_CONSAN_TEST_FORCE_VGPR_SPILL=1"
                    "RJ_CONSAN_TEST_KERNEL_FILTER=shared_noinline_atomic_add_for_instrumentation"
                    "RJ_CONSAN_REQUIRE_RECORDS=1"
                    "RJ_CONSAN_LOG=3"
                    "RJ_CONSAN_MAX_PATCHES=8"
                    "RJ_CONSAN_REQUIRE_PATCH=1"
        )

        rj_add_consan_hip_case(
                NAME "ConSanHipTest.BaselineStaticPrivateStackPreservesOutput"
                TARGET hip_consan_test
                FILTER "${RJ_CONSAN_STATIC_PRIVATE_STACK_TEST}"
                TIMEOUT 30
        )
        rj_add_consan_hip_case(
                NAME "ConSanHipTest.DbiStaticPrivateStackForcedSpillPreservesOutput"
                TARGET hip_consan_test
                FILTER "${RJ_CONSAN_STATIC_PRIVATE_STACK_TEST}"
                TIMEOUT 30
                PASS_REGEX "owner_names=.*static_private_stack.*private_bytes=[1-9][0-9]*"
                ENVIRONMENT
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=default"
                    "RJ_CONSAN_TEST_FORCE_VGPR_SPILL=1"
                    "RJ_CONSAN_TEST_KERNEL_FILTER=static_private_stack_test_kernel_for_instrumentation"
                    "RJ_CONSAN_LOG=3"
                    "RJ_CONSAN_MAX_PATCHES=1"
                    "RJ_CONSAN_REQUIRE_PATCH=1"
        )
        rj_add_consan_hip_case(
                NAME "ConSanHipTest.BaselineDynamicPrivateStackPreservesOutput"
                TARGET hip_consan_test
                FILTER "${RJ_CONSAN_DYNAMIC_PRIVATE_STACK_TEST}"
                TIMEOUT 30
        )

        rj_add_consan_hip_case(
                NAME "ConSanHipTest.DbiDynamicPrivateStackForcedSpillPreservesOutput"
                TARGET hip_consan_test
                FILTER "${RJ_CONSAN_DYNAMIC_PRIVATE_STACK_TEST}"
                TIMEOUT 30
                PASS_REGEX
                    "kind=trampoline-[a-z-]*[ -~]*spilled_vgprs=[1-9][0-9]*[ -~]*dynamic_private_addend=[1-9][0-9]*"
                ENVIRONMENT
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=default"
                    "RJ_CONSAN_TEST_FORCE_VGPR_SPILL=1"
                    "RJ_CONSAN_TEST_KERNEL_FILTER=dynamic_private_stack_test_kernel_for_instrumentation"
                    "RJ_CONSAN_LOG=3"
                    "RJ_CONSAN_MAX_PATCHES=1"
                    "RJ_CONSAN_REQUIRE_PATCH=1"
        )
        rj_add_consan_hip_case(
                NAME "ConSanLdsTest.BaselinePaddedCleanStore"
                TARGET hip_consan_lds_test
                FILTER "${RJ_CONSAN_LDS_CLEAN_TEST}"
                TIMEOUT 30
        )
        rj_add_consan_hip_case(
                NAME "ConSanLdsTest.DbiPaddedCleanStoreDoesNotTrap"
                TARGET hip_consan_lds_test
                FILTER "${RJ_CONSAN_LDS_CLEAN_TEST}"
                TIMEOUT 30
                ENVIRONMENT
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=supercollider"
                    "RJ_CONSAN_LOG=1"
                    "RJ_CONSAN_SC_DELAY=2"
                    "RJ_CONSAN_TMP_VGPR=3"
                    "RJ_CONSAN_REQUIRE_PATCH=1"
        )
        rj_add_consan_hip_case(
                NAME "ConSanLdsTest.BaselineHighHalfD16FlatLoad"
                TARGET hip_consan_lds_test
                FILTER "${RJ_CONSAN_LDS_HIGH_D16_TEST}"
                TIMEOUT 30
        )
        rj_add_consan_hip_case(
                NAME "ConSanLdsTest.DbiHighHalfD16FlatLoadDoesNotTrap"
                TARGET hip_consan_lds_test
                FILTER "${RJ_CONSAN_LDS_HIGH_D16_TEST}"
                TIMEOUT 30
                PASS_REGEX "kind=inline-flat-load-check-trap"
                ENVIRONMENT
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=supercollider"
                    "RJ_CONSAN_FLAT_PROVENANCE=strict"
                    "RJ_CONSAN_LOG=3"
                    "RJ_CONSAN_SC_DELAY=2"
                    "RJ_CONSAN_TEST_KERNEL_FILTER=padded_flat_high_d16_loads_kernel_for_instrumentation"
                    "RJ_CONSAN_MAX_PATCHES=8"
                    "RJ_CONSAN_REQUIRE_PATCH=1"
        )
        rj_add_consan_hip_case(
                NAME "ConSanLdsTest.BaselineHighHalfD16FlatStore"
                TARGET hip_consan_lds_test
                FILTER "${RJ_CONSAN_LDS_HIGH_D16_STORE_TEST}"
                TIMEOUT 30
        )
        rj_add_consan_hip_case(
                NAME "ConSanLdsTest.DbiHighHalfD16FlatStoreDoesNotTrap"
                TARGET hip_consan_lds_test
                FILTER "${RJ_CONSAN_LDS_HIGH_D16_STORE_TEST}"
                TIMEOUT 30
                PASS_REGEX "kind=inline-flat-store-check-trap"
                ENVIRONMENT
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=supercollider"
                    "RJ_CONSAN_FLAT_PROVENANCE=strict"
                    "RJ_CONSAN_LOG=3"
                    "RJ_CONSAN_SC_DELAY=2"
                    "RJ_CONSAN_TEST_KERNEL_FILTER=padded_flat_high_d16_store"
                    "RJ_CONSAN_MAX_PATCHES=12"
                    "RJ_CONSAN_REQUIRE_PATCH=1"
        )
        rj_add_consan_hip_case(
                NAME "ConSanLdsTest.BaselinePaddedRacyStore"
                TARGET hip_consan_lds_test
                FILTER "${RJ_CONSAN_LDS_RACY_TEST}"
                TIMEOUT 30
        )
        rj_add_consan_hip_case(
                NAME "ConSanLdsTest.DbiPaddedRacyStoreTripsTrap"
                TARGET hip_consan_lds_test
                FILTER "${RJ_CONSAN_LDS_RACY_TEST}"
                TIMEOUT 30
                ENVIRONMENT
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=supercollider"
                    "RJ_CONSAN_SC_REPORT_MODE=trap"
                    "RJ_CONSAN_LOG=1"
                    "RJ_CONSAN_SC_DELAY=2"
                    "RJ_CONSAN_TMP_VGPR=3"
                    "RJ_CONSAN_REQUIRE_PATCH=1"
                    "RJ_CONSAN_EXPECT_TRAP=1"
        )
        rj_add_consan_hip_case(
                NAME "ConSanLdsTest.DbiPaddedCleanStoreReportBufferStaysZero"
                TARGET hip_consan_lds_test
                FILTER "${RJ_CONSAN_LDS_CLEAN_REPORT_TEST}"
                TIMEOUT 30
                ENVIRONMENT
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=supercollider"
                    "RJ_CONSAN_LOG=1"
                    "RJ_CONSAN_SC_DELAY=2"
                    "RJ_CONSAN_TMP_VGPR=3"
                    "RJ_CONSAN_REQUIRE_PATCH=1"
        )
        rj_add_consan_hip_case(
                NAME "ConSanLdsTest.DbiPaddedRacyStoreReportsMarker"
                TARGET hip_consan_lds_test
                FILTER "${RJ_CONSAN_LDS_RACY_REPORT_TEST}"
                TIMEOUT 30
                ENVIRONMENT
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=supercollider"
                    "RJ_CONSAN_LOG=1"
                    "RJ_CONSAN_SC_DELAY=2"
                    "RJ_CONSAN_TMP_VGPR=3"
                    "RJ_CONSAN_REQUIRE_PATCH=1"
        )
        rj_add_consan_hip_case(
                NAME "ConSanLdsDiagnosticsTest.DbiReportsCrossWaveRace"
                TARGET hip_consan_lds_diagnostics_test
                FILTER "${RJ_CONSAN_LDS_DIAGNOSTICS_RACY_TEST}"
                TIMEOUT 30
                PASS_REGEX
                    "conflicts=[1-9][0-9]*"
                ENVIRONMENT
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=default"
                    "RJ_CONSAN_REQUIRE_RECORDS=1"
                    "RJ_CONSAN_REQUIRE_DIAGNOSTICS=1"
                    "RJ_CONSAN_LOG=1"
                    "RJ_CONSAN_REQUIRE_PATCH=1"
        )

        rj_add_consan_hip_case(
                NAME "ConSanLdsDiagnosticsTest.DbiRepeatedLaunchConflict"
                TARGET hip_consan_lds_diagnostics_test
                FILTER "${RJ_CONSAN_LDS_DIAGNOSTICS_RACY_TEST}"
                TIMEOUT 60
                PASS_REGEX
                    "conflicts=[1-9][0-9]*"
                RUN_SERIAL
                ENVIRONMENT
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=default"
                    "RJ_CONSAN_REQUIRE_RECORDS=1"
                    "RJ_CONSAN_REQUIRE_DIAGNOSTICS=1"
                    "RJ_CONSAN_LOG=1"
                    "RJ_CONSAN_TEST_KERNEL_FILTER=padded_cross_wave_lds_store"
                    "RJ_CONSAN_MAX_PATCHES=4"
                    "RJ_CONSAN_REQUIRE_PATCH=1"
                    "CONSAN_DIAGNOSTIC_LAUNCHES=100"
        )
        rj_add_consan_hip_case(
                NAME "ConSanLdsDiagnosticsTest.DbiReportsCrossWaveB128Race"
                TARGET hip_consan_lds_diagnostics_test
                FILTER "${RJ_CONSAN_LDS_DIAGNOSTICS_B128_RACY_TEST}"
                TIMEOUT 30
                ENVIRONMENT
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=default"
                    "RJ_CONSAN_REQUIRE_RECORDS=1"
                    "RJ_CONSAN_REQUIRE_DIAGNOSTICS=1"
                    "RJ_CONSAN_LOG=1"
                    "RJ_CONSAN_TEST_KERNEL_FILTER=padded_cross_wave_lds_store_b128"
                    "RJ_CONSAN_MAX_PATCHES=1"
                    "RJ_CONSAN_REQUIRE_PATCH=1"
        )
        rj_add_consan_hip_case(
                NAME "ConSanLdsDiagnosticsTest.DbiReportsCrossWaveFlatLdsRace"
                TARGET hip_consan_lds_diagnostics_test
                FILTER "${RJ_CONSAN_LDS_DIAGNOSTICS_FLAT_RACY_TEST}"
                TIMEOUT 30
                ENVIRONMENT
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=default"
                    "RJ_CONSAN_FLAT_PROVENANCE=strict"
                    "RJ_CONSAN_REQUIRE_RECORDS=1"
                    "RJ_CONSAN_REQUIRE_DIAGNOSTICS=1"
                    "RJ_CONSAN_LOG=1"
                    "RJ_CONSAN_TEST_KERNEL_FILTER=padded_cross_wave_flat_lds_store"
                    "RJ_CONSAN_MAX_PATCHES=1"
                    "RJ_CONSAN_REQUIRE_PATCH=1"
        )

        rj_add_consan_hip_case(
                NAME "ConSanLdsDiagnosticsTest.DbiReportsCrossWaveRaceHwIdOwnerSource"
                TARGET hip_consan_lds_diagnostics_test
                FILTER "${RJ_CONSAN_LDS_DIAGNOSTICS_RACY_TEST}"
                TIMEOUT 30
                ENVIRONMENT
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=default"
                    "RJ_CONSAN_REQUIRE_RECORDS=1"
                    "RJ_CONSAN_REQUIRE_DIAGNOSTICS=1"
                    "RJ_CONSAN_LOG=1"
                    "RJ_CONSAN_OWNER_SOURCE=hw_id"
                    "RJ_CONSAN_MAX_PATCHES=4"
                    "RJ_CONSAN_REQUIRE_PATCH=1"
        )
        rj_add_consan_hip_case(
                NAME "ConSanLdsDiagnosticsTest.DbiBarrierEpochOrdersCrossWaveAccesses"
                TARGET hip_consan_lds_diagnostics_test
                FILTER "${RJ_CONSAN_LDS_DIAGNOSTICS_BARRIER_TEST}"
                TIMEOUT 30
                ENVIRONMENT
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=default"
                    "RJ_CONSAN_FORBID_DIAGNOSTICS=1"
                    "RJ_CONSAN_TRACK_BARRIERS=1"
                    "RJ_CONSAN_TEST_KERNEL_FILTER=padded_barrier_ordered_lds_store"
                    "RJ_CONSAN_LOG=1"
                    "RJ_CONSAN_MAX_PATCHES=8"
                    "RJ_CONSAN_REQUIRE_PATCH=1"
        )
        rj_add_consan_hip_case(
                NAME "ConSanLdsDiagnosticsTest.DbiRepeatedLaunchClean"
                TARGET hip_consan_lds_diagnostics_test
                FILTER "${RJ_CONSAN_LDS_DIAGNOSTICS_BARRIER_TEST}"
                TIMEOUT 60
                ENVIRONMENT
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=default"
                    "RJ_CONSAN_REQUIRE_RECORDS=1"
                    "RJ_CONSAN_FORBID_DIAGNOSTICS=1"
                    "RJ_CONSAN_FORBID_OVERFLOW=1"
                    "RJ_CONSAN_TRACK_BARRIERS=1"
                    "RJ_CONSAN_LOG=1"
                    "RJ_CONSAN_TEST_KERNEL_FILTER=padded_barrier_ordered_lds_store"
                    "RJ_CONSAN_MAX_PATCHES=8"
                    "RJ_CONSAN_REQUIRE_PATCH=1"
                    "CONSAN_DIAGNOSTIC_LAUNCHES=100"
        )
        set_tests_properties(
            "ConSanLdsDiagnosticsTest.DbiRepeatedLaunchClean"
            PROPERTIES RUN_SERIAL TRUE
        )
        rj_add_consan_hip_case(
                NAME "ConSanLdsDiagnosticsTest.BaselineFenceAndBarrierPreservesExactOutput"
                TARGET hip_consan_lds_diagnostics_test
                FILTER "${RJ_CONSAN_LDS_DIAGNOSTICS_FENCE_BARRIER_TEST}"
                TIMEOUT 30
        )
        rj_add_consan_hip_case(
                NAME "ConSanLdsDiagnosticsTest.DbiFenceAndBarrierPreservesExactOutput"
                TARGET hip_consan_lds_diagnostics_test
                FILTER "${RJ_CONSAN_LDS_DIAGNOSTICS_FENCE_BARRIER_TEST}"
                TIMEOUT 30
                ENVIRONMENT
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=default"
                    "RJ_CONSAN_REQUIRE_RECORDS=1"
                    "RJ_CONSAN_FORBID_DIAGNOSTICS=1"
                    "RJ_CONSAN_TRACK_BARRIERS=1"
                    "RJ_CONSAN_TEST_KERNEL_FILTER=padded_fence_barrier_ordered_lds_accesses"
                    "RJ_CONSAN_LOG=1"
                    "RJ_CONSAN_MAX_PATCHES=8"
                    "RJ_CONSAN_REQUIRE_PATCH=1"
        )
        rj_add_consan_hip_case(
                NAME "ConSanLdsDiagnosticsTest.DbiMovedBarrierReportsCrossWaveRace"
                TARGET hip_consan_lds_diagnostics_test
                FILTER "${RJ_CONSAN_LDS_DIAGNOSTICS_BARRIER_TEST}"
                TIMEOUT 30
                ENVIRONMENT
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=default"
                    "RJ_CONSAN_REQUIRE_RECORDS=1"
                    "RJ_CONSAN_REQUIRE_DIAGNOSTICS=1"
                    "RJ_CONSAN_TRACK_BARRIERS=1"
                    "RJ_CONSAN_FAULT_MOVE_BARRIER=1"
                    "RJ_CONSAN_FAULT_BARRIER_INDEX=2"
                    "RJ_CONSAN_TEST_KERNEL_FILTER=padded_barrier_ordered_lds_store"
                    "RJ_CONSAN_LOG=1"
                    "RJ_CONSAN_MAX_PATCHES=8"
                    "RJ_CONSAN_REQUIRE_PATCH=1"
        )
        rj_add_consan_hip_case(
                NAME "ConSanLdsDiagnosticsTest.DbiPrivateEpochBarrierOrdersCrossWaveAccesses"
                TARGET hip_consan_lds_diagnostics_test
                FILTER "${RJ_CONSAN_LDS_DIAGNOSTICS_BARRIER_TEST}"
                TIMEOUT 30
                ENVIRONMENT
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=default"
                    "RJ_CONSAN_REQUIRE_RECORDS=1"
                    "RJ_CONSAN_FORBID_DIAGNOSTICS=1"
                    "RJ_CONSAN_TRACK_BARRIERS=1"
                    "RJ_CONSAN_LOG=1"
                    "RJ_CONSAN_TEST_FORCE_PRIVATE_EPOCH=1"
                    "RJ_CONSAN_TEST_KERNEL_FILTER=padded_barrier_ordered_lds_store"
                    "RJ_CONSAN_MAX_PATCHES=8"
                    "RJ_CONSAN_REQUIRE_PATCH=1"
        )

        rj_add_consan_hip_case(
                NAME "ConSanLdsDiagnosticsTest.DbiAtomicHandoffSameAddressIsClean"
                TARGET hip_consan_lds_diagnostics_test
                FILTER "${RJ_CONSAN_LDS_DIAGNOSTICS_ATOMIC_SAME_ADDRESS_TEST}"
                TIMEOUT 30
                ENVIRONMENT
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=default"
                    "RJ_CONSAN_FORBID_DIAGNOSTICS=1"
                    "RJ_CONSAN_TRACK_ATOMICS=1"
                    "RJ_CONSAN_TEST_KERNEL_FILTER=padded_atomic_ordered_lds_accesses"
                    "RJ_CONSAN_LOG=1"
                    "RJ_CONSAN_MAX_PATCHES=8"
                    "RJ_CONSAN_REQUIRE_PATCH=1"
        )
        rj_add_consan_hip_case(
                NAME "ConSanLdsDiagnosticsTest.DbiInjectedAtomicWrongAddressReports"
                TARGET hip_consan_lds_diagnostics_test
                FILTER "${RJ_CONSAN_LDS_DIAGNOSTICS_ATOMIC_SAME_ADDRESS_TEST}"
                TIMEOUT 30
                ENVIRONMENT
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=default"
                    "RJ_CONSAN_REQUIRE_RECORDS=1"
                    "RJ_CONSAN_REQUIRE_DIAGNOSTICS=1"
                    "RJ_CONSAN_TRACK_ATOMICS=1"
                    "RJ_CONSAN_FAULT_ATOMIC_WRONG_ADDRESS=1"
                    "RJ_CONSAN_FAULT_ATOMIC_VALID_ADDRESS_DELTA=4"
                    "RJ_CONSAN_FAULT_REQUIRE_EXACTLY_ONE=1"
                    "RJ_CONSAN_FAULT_ATOMIC_INDEX=1"
                    "RJ_CONSAN_TEST_KERNEL_FILTER=padded_atomic_ordered_lds_accesses"
                    "RJ_CONSAN_LOG=1"
                    "RJ_CONSAN_MAX_PATCHES=8"
                    "RJ_CONSAN_REQUIRE_PATCH=1"
        )
        rj_add_consan_hip_case(
                NAME "ConSanLdsDiagnosticsTest.DbiInjectedAtomicWeakenedOrderReports"
                TARGET hip_consan_lds_diagnostics_test
                FILTER "${RJ_CONSAN_LDS_DIAGNOSTICS_ATOMIC_SAME_ADDRESS_TEST}"
                TIMEOUT 30
                ENVIRONMENT
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=default"
                    "RJ_CONSAN_REQUIRE_RECORDS=1"
                    "RJ_CONSAN_REQUIRE_DIAGNOSTICS=1"
                    "RJ_CONSAN_TRACK_ATOMICS=1"
                    "RJ_CONSAN_FAULT_ATOMIC_WEAKEN_ORDER=1"
                    "RJ_CONSAN_FAULT_ATOMIC_INDEX=1"
                    "RJ_CONSAN_FAULT_REQUIRE_EXACTLY_ONE=1"
                    "RJ_CONSAN_TEST_KERNEL_FILTER=padded_atomic_ordered_lds_accesses"
                    "RJ_CONSAN_LOG=1"
                    "RJ_CONSAN_MAX_PATCHES=8"
                    "RJ_CONSAN_REQUIRE_PATCH=1"
        )
        rj_add_consan_hip_case(
                NAME "ConSanLdsDiagnosticsTest.DbiInjectedAtomicWeakenedScopeReports"
                TARGET hip_consan_lds_diagnostics_test
                FILTER "${RJ_CONSAN_LDS_DIAGNOSTICS_ATOMIC_SAME_ADDRESS_TEST}"
                TIMEOUT 30
                ENVIRONMENT
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=default"
                    "RJ_CONSAN_REQUIRE_RECORDS=1"
                    "RJ_CONSAN_REQUIRE_DIAGNOSTICS=1"
                    "RJ_CONSAN_TRACK_ATOMICS=1"
                    "RJ_CONSAN_FAULT_ATOMIC_WEAKEN_SCOPE=1"
                    "RJ_CONSAN_FAULT_ATOMIC_INDEX=1"
                    "RJ_CONSAN_FAULT_REQUIRE_EXACTLY_ONE=1"
                    "RJ_CONSAN_TEST_KERNEL_FILTER=padded_atomic_ordered_lds_accesses"
                    "RJ_CONSAN_LOG=1"
                    "RJ_CONSAN_MAX_PATCHES=8"
                    "RJ_CONSAN_REQUIRE_PATCH=1"
        )
        rj_add_consan_hip_case(
                NAME "ConSanLdsDiagnosticsTest.DbiAtomicHandoffSameAddressIsCleanHwIdOwnerSource"
                TARGET hip_consan_lds_diagnostics_test
                FILTER "${RJ_CONSAN_LDS_DIAGNOSTICS_ATOMIC_SAME_ADDRESS_TEST}"
                TIMEOUT 30
                ENVIRONMENT
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=default"
                    "RJ_CONSAN_FORBID_DIAGNOSTICS=1"
                    "RJ_CONSAN_TRACK_ATOMICS=1"
                    "RJ_CONSAN_TEST_KERNEL_FILTER=padded_atomic_ordered_lds_accesses"
                    "RJ_CONSAN_LOG=1"
                    "RJ_CONSAN_OWNER_SOURCE=hw_id"
                    # Stress the physical HW_ID zero-sentinel remap while the
                    # structural unit tests cover its exact instruction shape.
                    "CONSAN_SC_CLEAN_LAUNCHES=100"
                    "RJ_CONSAN_MAX_PATCHES=8"
                    "RJ_CONSAN_REQUIRE_PATCH=1"
        )
        rj_add_consan_hip_case(
                NAME "ConSanLdsDiagnosticsTest.DbiAtomicHandoffWrongAddressReports"
                TARGET hip_consan_lds_diagnostics_test
                FILTER "${RJ_CONSAN_LDS_DIAGNOSTICS_ATOMIC_WRONG_ADDRESS_TEST}"
                TIMEOUT 30
                ENVIRONMENT
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=default"
                    "RJ_CONSAN_REQUIRE_RECORDS=1"
                    "RJ_CONSAN_REQUIRE_DIAGNOSTICS=1"
                    "RJ_CONSAN_TRACK_ATOMICS=1"
                    "RJ_CONSAN_TEST_KERNEL_FILTER=padded_atomic_ordered_lds_accesses"
                    "RJ_CONSAN_LOG=1"
                    "RJ_CONSAN_MAX_PATCHES=8"
                    "RJ_CONSAN_REQUIRE_PATCH=1"
        )
        rj_add_consan_hip_case(
                NAME "ConSanLdsDiagnosticsTest.DbiAtomicHandoffWrongAddressReportsHwIdOwnerSource"
                TARGET hip_consan_lds_diagnostics_test
                FILTER "${RJ_CONSAN_LDS_DIAGNOSTICS_ATOMIC_WRONG_ADDRESS_TEST}"
                TIMEOUT 30
                ENVIRONMENT
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=default"
                    "RJ_CONSAN_REQUIRE_RECORDS=1"
                    "RJ_CONSAN_REQUIRE_DIAGNOSTICS=1"
                    "RJ_CONSAN_TRACK_ATOMICS=1"
                    "RJ_CONSAN_TEST_KERNEL_FILTER=padded_atomic_ordered_lds_accesses"
                    "RJ_CONSAN_LOG=1"
                    "RJ_CONSAN_OWNER_SOURCE=hw_id"
                    "RJ_CONSAN_MAX_PATCHES=8"
                    "RJ_CONSAN_REQUIRE_PATCH=1"
        )

        rj_add_consan_hip_case(
                NAME "ConSanLdsDiagnosticsTest.DbiAtomicCasFailureDoesNotPublish"
                TARGET hip_consan_lds_diagnostics_test
                FILTER "${RJ_CONSAN_LDS_DIAGNOSTICS_ATOMIC_CAS_TEST}"
                TIMEOUT 30
                ENVIRONMENT
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=default"
                    "RJ_CONSAN_FORBID_DIAGNOSTICS=1"
                    "RJ_CONSAN_TRACK_ATOMICS=1"
                    "RJ_CONSAN_TEST_KERNEL_FILTER=padded_atomic_cas_failure_does_not_publish"
                    "RJ_CONSAN_LOG=1"
                    "RJ_CONSAN_MAX_PATCHES=8"
                    "RJ_CONSAN_REQUIRE_PATCH=1"
        )
        rj_add_consan_hip_case(
                NAME "ConSanLdsDiagnosticsTest.DbiInjectedAtomicCasWeakenedOrderReports"
                TARGET hip_consan_lds_diagnostics_test
                FILTER "${RJ_CONSAN_LDS_DIAGNOSTICS_ATOMIC_CAS_TEST}"
                TIMEOUT 30
                ENVIRONMENT
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=default"
                    "RJ_CONSAN_REQUIRE_RECORDS=1"
                    "RJ_CONSAN_REQUIRE_DIAGNOSTICS=1"
                    "RJ_CONSAN_TRACK_ATOMICS=1"
                    "RJ_CONSAN_FAULT_ATOMIC_WEAKEN_ORDER=1"
                    "RJ_CONSAN_FAULT_ATOMIC_INDEX=1"
                    "RJ_CONSAN_FAULT_REQUIRE_EXACTLY_ONE=1"
                    "RJ_CONSAN_TEST_KERNEL_FILTER=padded_atomic_cas_failure_does_not_publish"
                    "RJ_CONSAN_LOG=1"
                    "RJ_CONSAN_MAX_PATCHES=8"
                    "RJ_CONSAN_REQUIRE_PATCH=1"
        )

        rj_add_consan_hip_case(
                NAME "ConSanHipTest.DbiEquivalentAccessesInDifferent2DWorkgroupsDoNotConflict"
                TARGET hip_consan_test
                FILTER "${RJ_CONSAN_LDS_WORKGROUP_2D_RECORD_TEST}"
                TIMEOUT 30
                ENVIRONMENT
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=default"
                    "RJ_CONSAN_TEST_KERNEL_FILTER=equivalent_lds_accesses_in_different_2d_workgroups"
                    "RJ_CONSAN_LOG=1"
                    "RJ_CONSAN_MAX_PATCHES=10"
                    "RJ_CONSAN_REQUIRE_PATCH=1"
        )
        rj_add_consan_hip_case(
                NAME "ConSanHipTest.DbiEquivalentAccessesInDifferent3DWorkgroupsDoNotConflict"
                TARGET hip_consan_test
                FILTER "${RJ_CONSAN_LDS_WORKGROUP_3D_RECORD_TEST}"
                TIMEOUT 30
                ENVIRONMENT
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=default"
                    "RJ_CONSAN_TEST_KERNEL_FILTER=equivalent_lds_accesses_in_different_3d_workgroups"
                    "RJ_CONSAN_LOG=1"
                    "RJ_CONSAN_MAX_PATCHES=12"
                    "RJ_CONSAN_REQUIRE_PATCH=1"
        )
        rj_add_consan_hip_case(
                NAME "ConSanHipTest.DbiBackendWritesDirectEntries"
                TARGET hip_consan_test
                FILTER "${RJ_CONSAN_LDS_DIRECT_TEST}"
                TIMEOUT 30
                ENVIRONMENT
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=default"
                    "RJ_CONSAN_TEST_KERNEL_FILTER=two_wave_lds_accesses"
                    "RJ_CONSAN_LOG=1"
                    "RJ_CONSAN_MAX_PATCHES=2"
                    "RJ_CONSAN_REQUIRE_PATCH=1"
        )
        rj_add_consan_hip_case(
                NAME "ConSanHipTest.DbiRuntimeBackendRetainsAllWavesForSelectedAddress"
                TARGET hip_consan_test
                FILTER "ConSanHipTest.RuntimeBackendRetainsAllWavesForSelectedAddress"
                TIMEOUT 30
                RUN_SERIAL
                ENVIRONMENT
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=default"
                    "RJ_CONSAN_RUNTIME_SAMPLE_STRIDE=2"
                    "RJ_CONSAN_RUNTIME_SAMPLE_OFFSET=0"
                    "RJ_CONSAN_TEST_KERNEL_FILTER=two_wave_lds_accesses"
                    "RJ_CONSAN_LOG=1"
                    "RJ_CONSAN_MAX_PATCHES=2"
                    "RJ_CONSAN_REQUIRE_PATCH=1"
        )
        rj_add_consan_hip_case(
                NAME "ConSanHipTest.DbiInKernelCheckerReportsConflictImmediately"
                TARGET hip_consan_test
                FILTER "ConSanHipTest.InKernelCheckerReportsConflictImmediately"
                TIMEOUT 30
                ENVIRONMENT
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=default"
                    "RJ_CONSAN_DEVICE_CONFLICT_CHECK=1"
                    "RJ_CONSAN_TEST_KERNEL_FILTER=two_wave_lds_accesses"
                    "RJ_CONSAN_LOG=1"
                    "RJ_CONSAN_MAX_PATCHES=2"
                    "RJ_CONSAN_REQUIRE_PATCH=1"
        )
        rj_add_consan_hip_case(
                NAME "ConSanHipTest.DbiAutoBufferWithDeviceCheckReportsConflict"
                TARGET hip_consan_test
                FILTER "${RJ_CONSAN_LDS_AUTO_REPLAY_CONFLICT_TEST}"
                TIMEOUT 30
                # Immediate adjacency checking is a lower-fidelity optional
                # path; the portable contract is the exact host-side conflict.
                PASS_REGEX "conflicts=[1-9]"
                ENVIRONMENT
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=default"
                    "RJ_CONSAN_DEVICE_CONFLICT_CHECK=1"
                    "RJ_CONSAN_REQUIRE_RECORDS=1"
                    "RJ_CONSAN_REQUIRE_DIAGNOSTICS=1"
                    "RJ_CONSAN_TEST_KERNEL_FILTER=two_wave_lds_accesses"
                    "RJ_CONSAN_LOG=1"
                    "RJ_CONSAN_MAX_PATCHES=2"
                    "RJ_CONSAN_REQUIRE_PATCH=1"
        )
        rj_add_consan_hip_case(
                NAME "ConSanHipTest.DbiAutoBufferReportsConflict"
                TARGET hip_consan_test
                FILTER "${RJ_CONSAN_LDS_AUTO_REPLAY_CONFLICT_TEST}"
                TIMEOUT 30
                ENVIRONMENT
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=default"
                    "RJ_CONSAN_REQUIRE_RECORDS=1"
                    "RJ_CONSAN_REQUIRE_DIAGNOSTICS=1"
                    "RJ_CONSAN_TEST_KERNEL_FILTER=two_wave_lds_accesses"
                    "RJ_CONSAN_LOG=1"
                    "RJ_CONSAN_MAX_PATCHES=2"
                    "RJ_CONSAN_REQUIRE_PATCH=1"
        )
        rj_add_consan_hip_case(
                NAME "ConSanHipTest.DbiCausalMultiWorkgroupClean"
                TARGET hip_consan_test
                FILTER "ConSanHipTest.DirectRetainsExactMultiWorkgroupWindows"
                TIMEOUT 30
                PASS_REGEX
                    "SA2C sampled evidence ready_windows=2 live_entries=2 workgroups=0,1 dropped_windows=0"
                ENVIRONMENT
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=default"
                    "RJ_CONSAN_TEST_KERNEL_FILTER=equivalent_lds_accesses_in_different_workgroups"
                    "RJ_CONSAN_LOG=1"
                    "RJ_CONSAN_MAX_PATCHES=4"
                    "RJ_CONSAN_REQUIRE_PATCH=1"
        )
        rj_add_consan_hip_case(
                NAME "ConSanHipTest.DbiCausalFaultReportsConflict"
                TARGET hip_consan_test
                FILTER "${RJ_CONSAN_LDS_AUTO_REPLAY_CONFLICT_TEST}"
                TIMEOUT 30
                PASS_REGEX "visible=[1-9].*conflicts=[1-9]"
                ENVIRONMENT
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=default"
                    "RJ_CONSAN_REQUIRE_RECORDS=1"
                    "RJ_CONSAN_REQUIRE_DIAGNOSTICS=1"
                    "RJ_CONSAN_FORBID_OVERFLOW=1"
                    "RJ_CONSAN_TEST_KERNEL_FILTER=two_wave_lds_accesses"
                    "RJ_CONSAN_LOG=1"
                    "RJ_CONSAN_MAX_PATCHES=2"
                    "RJ_CONSAN_REQUIRE_PATCH=1"
        )

        rj_add_consan_hip_case(
                NAME "ConSanHipTest.DbiAutoReportRetainsRepeatedDispatches"
                TARGET hip_consan_test
                FILTER "${RJ_CONSAN_LDS_AUTO_REPEATED_DISPATCH_TEST}"
                TIMEOUT 30
                PASS_REGEX
                    "visible=[1-9]"
                ALSO_PASS_REGEX
                    "completed_epochs=4 discarded_epochs=0"
                    "visible_evidence=8 dynamic_incomplete=0"
                ENVIRONMENT
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=default"
                    "RJ_CONSAN_REQUIRE_RECORDS=1"
                    "RJ_CONSAN_FORBID_OVERFLOW=1"
                    "RJ_CONSAN_TEST_KERNEL_FILTER=two_wave_lds_accesses"
                    "RJ_CONSAN_LOG=1"
                    "RJ_CONSAN_MAX_PATCHES=2"
                    "RJ_CONSAN_REQUIRE_PATCH=1"
        )
        rj_add_consan_hip_case(
                NAME "ConSanHipTest.DbiSaturationPreservesRepeatedDispatchOutput"
                TARGET hip_consan_test
                FILTER "${RJ_CONSAN_LDS_SATURATION_OUTPUT_TEST}"
                TIMEOUT 30
                PASS_REGEX "saturated_windows=[1-9]"
                ENVIRONMENT
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=default"
                    # Dispatch zero is not selected by stride two on every
                    # runtime. Select every workgroup so this remains a
                    # saturation/output test rather than a sampling-residue test.
                    "RJ_CONSAN_RUNTIME_SAMPLE_STRIDE=1"
                    "RJ_CONSAN_TRACK_BARRIERS=1"
                    "RJ_CONSAN_REQUIRE_RECORDS=1"
                    "RJ_CONSAN_FORBID_DIAGNOSTICS=1"
                    "RJ_CONSAN_TEST_KERNEL_FILTER=repeated_dispatch_saturation_preserves_output_for_instrumentation"
                    "RJ_CONSAN_LOG=1"
                    "RJ_CONSAN_MAX_PATCHES=4"
                    "RJ_CONSAN_REQUIRE_PATCH=1"
        )
        rj_add_consan_hip_case(
                NAME "ConSanHipTest.DbiStagedDscntLoadPreservesOutput"
                TARGET hip_consan_test
                FILTER "${RJ_CONSAN_LDS_STAGED_DSCNT_TEST}"
                TIMEOUT 30
                ENVIRONMENT
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=default"
                    "RJ_CONSAN_RUNTIME_SAMPLE_STRIDE=2"
                    "RJ_CONSAN_RUNTIME_SAMPLE_OFFSET=0"
                    "RJ_CONSAN_TRACK_BARRIERS=1"
                    "RJ_CONSAN_FORBID_DIAGNOSTICS=1"
                    "RJ_CONSAN_TEST_KERNEL_FILTER=staged_dscnt_load_preserves_output_for_instrumentation"
                    "RJ_CONSAN_LOG=1"
                    "RJ_CONSAN_MAX_PATCHES=2"
                    "RJ_CONSAN_REQUIRE_PATCH=1"
        )

        rj_add_consan_hip_case(
                NAME "ConSanHipTest.DbiOneSlotPreservesBufferBounds"
                TARGET hip_consan_test
                FILTER "${RJ_CONSAN_LDS_LIMITED_RECORD_TEST}"
                TIMEOUT 30
                ENVIRONMENT
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=default"
                    "RJ_CONSAN_TEST_KERNEL_FILTER=two_wave_lds_accesses"
                    "RJ_CONSAN_LOG=1"
                    "RJ_CONSAN_MAX_PATCHES=2"
                    "RJ_CONSAN_REQUIRE_PATCH=1"
        )

        # RR3C5C's retained harness runs these rows under the global GPU
        # lock and applies stricter log accounting. Keep the underlying
        # controls independently build-wired so their exact GTest names
        # cannot silently drift.
        rj_add_consan_hip_case(
                NAME "ConSanHipTest.DbiFenceHandoffSameAddressIsClean"
                TARGET hip_consan_lds_diagnostics_test
                FILTER "${RJ_CONSAN_LDS_DIAGNOSTICS_ATOMIC_SAME_ADDRESS_TEST}"
                TIMEOUT 60
                ENVIRONMENT
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=default"
                    "RJ_CONSAN_AUTO_REPORT_BUFFER_SIZE=1048576"
                    "RJ_CONSAN_TRACK_ATOMICS=1"

                    "RJ_CONSAN_TMP_VGPR=8"
                    "RJ_CONSAN_TEST_KERNEL_FILTER=padded_atomic_ordered_lds_accesses"
                    "RJ_CONSAN_MAX_PATCHES=16"
                    "RJ_CONSAN_REQUIRE_PATCH=1"
                    "RJ_CONSAN_REQUIRE_RECORDS=1"
                    "RJ_CONSAN_FORBID_DIAGNOSTICS=1"
                    "RJ_CONSAN_FORBID_OVERFLOW=1"
                    "RJ_CONSAN_LOG=2"
        )
        rj_add_consan_hip_case(
                NAME "ConSanHipTest.DbiFenceHandoffWrongAddressConflicts"
                TARGET hip_consan_lds_diagnostics_test
                FILTER "${RJ_CONSAN_LDS_DIAGNOSTICS_ATOMIC_WRONG_ADDRESS_TEST}"
                TIMEOUT 60
                ENVIRONMENT
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=default"
                    "RJ_CONSAN_AUTO_REPORT_BUFFER_SIZE=1048576"
                    "RJ_CONSAN_TRACK_ATOMICS=1"

                    "RJ_CONSAN_TMP_VGPR=8"
                    "RJ_CONSAN_TEST_KERNEL_FILTER=padded_atomic_ordered_lds_accesses"
                    "RJ_CONSAN_MAX_PATCHES=16"
                    "RJ_CONSAN_REQUIRE_PATCH=1"
                    "RJ_CONSAN_REQUIRE_RECORDS=1"
                    "RJ_CONSAN_REQUIRE_DIAGNOSTICS=1"
                    "RJ_CONSAN_FORBID_OVERFLOW=1"
                    "RJ_CONSAN_LOG=2"
        )
        rj_add_consan_hip_case(
                NAME "ConSanHipTest.DbiAtomicHandoffAutoReportIsClean"
                TARGET hip_consan_test
                FILTER "ConSanHipTest.AtomicHandoffRunsWithAutoReportBuffer"
                TIMEOUT 30
                PASS_REGEX
                    "visible=[1-9].*conflicts=0"
                ENVIRONMENT
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=default"
                    "RJ_CONSAN_AUTO_REPORT_BUFFER_SIZE=1048576"
                    "RJ_CONSAN_TRACK_ATOMICS=1"

                    "RJ_CONSAN_REQUIRE_RECORDS=1"
                    "RJ_CONSAN_FORBID_DIAGNOSTICS=1"
                    "RJ_CONSAN_FORBID_OVERFLOW=1"
                    "RJ_CONSAN_TEST_KERNEL_FILTER=atomic_handoff_lds_accesses"
                    "RJ_CONSAN_LOG=1"
                    "RJ_CONSAN_MAX_PATCHES=8"
                    "RJ_CONSAN_REQUIRE_PATCH=1"
        )
        unset(_rj_consan_physical_target)
    endif()
endfunction()
