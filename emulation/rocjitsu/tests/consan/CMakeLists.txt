# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# ConSan's unit, fuzzing, and live-GPU test registration lives here so the
# shared rocjitsu test manifest remains independent of ConSan's growth.

set(RJ_CONSAN_GFX942_HIP_MOI_BUILD_DIR
    ""
    CACHE PATH
    "Exact hip-moi-build-gfx942-tests directory for simulator CTest coverage"
)
set(RJ_CONSAN_GFX950_HIP_MOI_BUILD_DIR
    ""
    CACHE PATH
    "Exact hip-moi-build-gfx950-tests directory for simulator CTest coverage"
)

function(rj_add_consan_unit_sources target)
    target_sources(
        ${target}
        PRIVATE
            patch/consan/analysis_test.cpp
            patch/consan/composition_test.cpp
            patch/consan/core_test.cpp
            patch/consan/fault_atomic_test.cpp
            patch/consan/fault_barrier_test.cpp
            patch/consan/fault_composition_test.cpp
            patch/consan/fault_injection_test.cpp
            patch/consan/fault_ordinary_test.cpp
            patch/consan/moi_common_test.cpp
            patch/consan/moi_engine_conformance_test.cpp
            patch/consan/moi_inline_atomic_test.cpp
            patch/consan/moi_inline_shadow_test.cpp
            patch/consan/moi_record_replay_model_test.cpp
            patch/consan/moi_record_replay_test.cpp
            patch/consan/moi_sampled_test.cpp
            patch/consan/resource_test.cpp
            patch/consan/supercollider_test.cpp
            patch/consan_moi_adversarial_test.cpp
            patch/consan_moi_report_plan_test.cpp
    )
endfunction()

# Registers one HIP fixture case through a single typed path. Keeping command
# construction, working-directory selection, timeout defaults, and optional
# properties here prevents the live-GPU matrix from drifting as cases are
# added. ENVIRONMENT is a list of KEY=VALUE entries. Every instrumented case
# supplies an environment; loading the hook enables ConSan, and the mode
# selects the analysis.
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
    )
    set(_multi_value_args ENVIRONMENT)
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

    # Live MOI cases are intentionally narrow fixtures. Production ConSan
    # tracks every supported synchronization class by default, but these tests
    # use report buffers and patch limits sized for the class under test. Make
    # the focused default access-only and require barrier/atomic cases to opt in
    # explicitly below.
    set(_case_is_moi FALSE)
    set(_case_sets_barrier_tracking FALSE)
    set(_case_sets_atomic_tracking FALSE)
    foreach(_environment_entry IN LISTS _case_ENVIRONMENT)
        if(
            _environment_entry
                MATCHES
                "^RJ_CONSAN_MODE=(record-replay|inline-shadow|sampled)$"
        )
            set(_case_is_moi TRUE)
        elseif(_environment_entry MATCHES "^RJ_CONSAN_MOI_TRACK_BARRIERS=")
            set(_case_sets_barrier_tracking TRUE)
        elseif(_environment_entry MATCHES "^RJ_CONSAN_MOI_TRACK_ATOMICS=")
            set(_case_sets_atomic_tracking TRUE)
        endif()
    endforeach()
    if(_case_is_moi)
        if(NOT _case_sets_barrier_tracking)
            list(APPEND _case_ENVIRONMENT "RJ_CONSAN_MOI_TRACK_BARRIERS=0")
        endif()
        if(NOT _case_sets_atomic_tracking)
            list(APPEND _case_ENVIRONMENT "RJ_CONSAN_MOI_TRACK_ATOMICS=0")
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
    rj_add_test(${_test_args})

    if(_case_PASS_REGEX)
        set_property(
            TEST "${_case_NAME}"
            PROPERTY PASS_REGULAR_EXPRESSION "${_case_PASS_REGEX}"
        )
    endif()
    # CTest's PASS_REGULAR_EXPRESSION can otherwise override a nonzero GTest
    # exit status. Never let an expected diagnostic hide a failed assertion.
    set(_case_fail_regex "\\[  FAILED  \\]|FAILED TEST")
    if(_case_FAIL_REGEX)
        list(APPEND _case_fail_regex "${_case_FAIL_REGEX}")
    endif()
    set_property(
        TEST "${_case_NAME}"
        PROPERTY FAIL_REGULAR_EXPRESSION "${_case_fail_regex}"
    )
    if(_case_RUN_SERIAL)
        set_property(TEST "${_case_NAME}" PROPERTY RUN_SERIAL TRUE)
    endif()
endfunction()

function(
    rj_add_consan_cdna_hip_moi_sim_case
    target_name
    target
    build_dir
    name
    suite
)
    set(_runner
        "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../dbi/consan/consan_cdna_hip_moi_sim.py"
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
    rj_add_consan_cdna_hip_moi_sim_tests
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
        "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../dbi/consan/consan_cdna_hip_moi_sim.py"
    )
    set(_suite_registry
        "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../dbi/consan/consan_cdna_hip_moi_registry.py"
    )
    set(_validation_registry
        "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../dbi/consan/consan_validation.py"
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
        rj_add_consan_cdna_hip_moi_sim_case(
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
set(RJ_CONSAN_CDNA_MOI_FORCED_SPILL_RECORD_TEST
    "ConSanMoiCdnaTest.ForcedSpillRecordReplayPreservesLiveVgprsAndWritesRecord"
)
set(RJ_CONSAN_CDNA_MOI_FORCED_SPILL_SAMPLED_TEST
    "ConSanMoiCdnaTest.ForcedSpillSampledPreservesLiveVgprsAndWritesEntry"
)
set(RJ_CONSAN_CDNA_MOI_FORCED_SPILL_INLINE_TEST
    "ConSanMoiCdnaTest.ForcedSpillInlineShadowPreservesLiveVgprs"
)
set(RJ_CONSAN_CDNA_MOI_SHARED_HELPER_TEST
    "ConSanMoiCdnaTest.SharedHelperForcedSpillPreservesBothCallers"
)
set(RJ_CONSAN_CDNA_MOI_BARRIER_TEST
    "ConSanMoiCdnaTest.BarrierForcedSpillPreservesLiveVgprsAndWritesRecords"
)
set(RJ_CONSAN_CDNA_MOI_ATOMIC_TEST
    "ConSanMoiCdnaTest.AtomicForcedSpillPreservesLiveVgprsAndWritesRecord"
)
set(RJ_CONSAN_CDNA_MOI_DYNAMIC_STACK_TEST
    "ConSanMoiCdnaTest.DynamicStackInlineSpillPreservesPrivateValues"
)
set(RJ_CONSAN_LDS_MOI_BARRIER_RECORD_TEST
    "ConSanMoiHipTest.BarrierRecordOrdersTwoWaveAccesses"
)
set(RJ_CONSAN_LDS_MOI_ATOMIC_RECORD_TEST
    "ConSanMoiHipTest.FlatAtomicWritesDynamicRecord"
)
set(RJ_CONSAN_MOI_FORCED_SPILL_RECORD_TEST
    "ConSanMoiHipTest.ForcedSpillRecordReplayPreservesLiveVgprsAndWritesRecord"
)
set(RJ_CONSAN_MOI_FORCED_SPILL_SAMPLED_TEST
    "ConSanMoiHipTest.ForcedSpillSampledPreservesLiveVgprsAndWritesEntry"
)
set(RJ_CONSAN_MOI_SHARED_NOINLINE_HELPER_TEST
    "ConSanMoiHipTest.SharedNoinlineHelperPreservesBothCallers"
)
set(RJ_CONSAN_MOI_SHARED_ATOMIC_HELPER_TEST
    "ConSanMoiHipTest.SharedAtomicNoinlineHelperPreservesBothCallers"
)
set(RJ_CONSAN_MOI_DYNAMIC_PRIVATE_STACK_TEST
    "ConSanMoiHipTest.DynamicPrivateStackPreservesOutput"
)

function(rj_add_consan_common_sim_cases)
    set(_one_value_args
        PREFIX
        LDS_TARGET
        MOI_TARGET
        CONFIG
        RECORD_TEST
        SAMPLED_TEST
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
            MOI_TARGET
            CONFIG
            RECORD_TEST
            SAMPLED_TEST
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
            "RJ_CONSAN_DELAY=2"
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
            "RJ_CONSAN_DELAY=2"
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
            "RJ_CONSAN_DELAY=2"
            "RJ_CONSAN_TEST_KERNEL_FILTER=padded_flat_high_d16_store"
            "RJ_CONSAN_MAX_PATCHES=12"
            "RJ_CONSAN_REQUIRE_PATCH=1"
    )
    rj_add_consan_hip_case(
        NAME "${_sim_PREFIX}.FlatLdsRecordReplayForcedSpill"
        TARGET ${_sim_LDS_TARGET}
        FILTER "${RJ_CONSAN_LDS_FLAT_RACY_TEST}"
        CONFIG "${_sim_CONFIG}"
        TIMEOUT 60
        RUN_SERIAL
        PASS_REGEX "site=access.*source=spill reason=none"
        FAIL_REGEX "RJ_CONSAN_MOI_REQUIRE_(RECORDS|REPLAY_CONFLICT) requested"
        ENVIRONMENT
            "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
            "RJ_CONSAN_MODE=record-replay"
            "RJ_CONSAN_MOI_AUTO_REPORT_BUFFER_SIZE=1048576"
            "RJ_CONSAN_FLAT_PROVENANCE=strict"
            "RJ_CONSAN_MOI_DYNAMIC_ACCESS_RECORDS=1"
            "RJ_CONSAN_MOI_REQUIRE_RECORDS=1"
            "RJ_CONSAN_MOI_REQUIRE_REPLAY_CONFLICT=1"
            "RJ_CONSAN_MOI_TRACK_BARRIERS=0"
            "RJ_CONSAN_MOI_TRACK_ATOMICS=0"
            "RJ_CONSAN_TEST_FORCE_VGPR_SPILL=1"
            "RJ_CONSAN_TEST_KERNEL_FILTER=padded_cross_wave_flat_lds_store"
            "RJ_CONSAN_LOG=3"
            "RJ_CONSAN_MAX_PATCHES=1"
            "RJ_CONSAN_REQUIRE_PATCH=1"
    )
    rj_add_consan_hip_case(
        NAME "${_sim_PREFIX}.FlatLdsInlineShadowForcedSpill"
        TARGET ${_sim_LDS_TARGET}
        FILTER "${RJ_CONSAN_LDS_FLAT_RACY_TEST}"
        CONFIG "${_sim_CONFIG}"
        TIMEOUT 60
        RUN_SERIAL
        PASS_REGEX
            "ConSan proof patch[ -~]*kind=trampoline-moi-exact-shadow-store[ -~]*spilled_vgprs=[1-9][0-9]*.*ConSan MOI auto diagnostic[ -~]*backend=1 kind=1[ -~]*first_owner=0 second_owner=[1-9][0-9]*[ -~]*first_kind=2 second_kind=2[ -~]*second_lanes=0x[1-9a-fA-F][0-9a-fA-F]*[ -~]*first_lds=\\[0,4\\) second_lds=\\[0,4\\)"
        FAIL_REGEX "RJ_CONSAN_MOI_REQUIRE_(RECORDS|DIAGNOSTICS) requested"
        ENVIRONMENT
            "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
            "RJ_CONSAN_MODE=inline-shadow"
            "RJ_CONSAN_MOI_AUTO_REPORT_BUFFER_SIZE=1048576"
            "RJ_CONSAN_FLAT_PROVENANCE=strict"
            "RJ_CONSAN_MOI_REQUIRE_RECORDS=1"
            "RJ_CONSAN_MOI_REQUIRE_DIAGNOSTICS=1"
            "RJ_CONSAN_MOI_TRACK_BARRIERS=0"
            "RJ_CONSAN_MOI_TRACK_ATOMICS=0"
            "RJ_CONSAN_TEST_FORCE_VGPR_SPILL=1"
            "RJ_CONSAN_TEST_KERNEL_FILTER=padded_cross_wave_flat_lds_store"
            "RJ_CONSAN_LOG=3"
            "RJ_CONSAN_MAX_PATCHES=1"
            "RJ_CONSAN_REQUIRE_PATCH=1"
    )
    rj_add_consan_hip_case(
        NAME "${_sim_PREFIX}.MoiRecordReplayForcedSpill"
        TARGET ${_sim_MOI_TARGET}
        FILTER "${_sim_RECORD_TEST}"
        CONFIG "${_sim_CONFIG}"
        TIMEOUT 60
        RUN_SERIAL
        PASS_REGEX "site=access.*source=spill reason=none"
        ENVIRONMENT
            "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
            "RJ_CONSAN_MODE=record-replay"
            "RJ_CONSAN_MOI_TRACK_BARRIERS=0"
            "RJ_CONSAN_MOI_TRACK_ATOMICS=0"
            "RJ_CONSAN_TEST_FORCE_VGPR_SPILL=1"
            "RJ_CONSAN_TEST_KERNEL_FILTER=${_sim_ACCESS_KERNEL}"
            "RJ_CONSAN_LOG=3"
            "RJ_CONSAN_MAX_PATCHES=1"
            "RJ_CONSAN_REQUIRE_PATCH=1"
    )
    rj_add_consan_hip_case(
        NAME "${_sim_PREFIX}.MoiSampledForcedSpill"
        TARGET ${_sim_MOI_TARGET}
        FILTER "${_sim_SAMPLED_TEST}"
        CONFIG "${_sim_CONFIG}"
        TIMEOUT 60
        RUN_SERIAL
        PASS_REGEX "site=access.*source=spill reason=none"
        ENVIRONMENT
            "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
            "RJ_CONSAN_MODE=sampled"
            "RJ_CONSAN_MOI_TRACK_BARRIERS=0"
            "RJ_CONSAN_MOI_TRACK_ATOMICS=0"
            "RJ_CONSAN_TEST_FORCE_VGPR_SPILL=1"
            "RJ_CONSAN_TEST_KERNEL_FILTER=${_sim_ACCESS_KERNEL}"
            "RJ_CONSAN_LOG=3"
            "RJ_CONSAN_MAX_PATCHES=1"
            "RJ_CONSAN_REQUIRE_PATCH=1"
    )
    rj_add_consan_hip_case(
        NAME "${_sim_PREFIX}.MoiSharedHelperForcedSpill"
        TARGET ${_sim_MOI_TARGET}
        FILTER "${_sim_SHARED_TEST}"
        CONFIG "${_sim_CONFIG}"
        TIMEOUT 60
        RUN_SERIAL
        PASS_REGEX "site=access.*source=spill reason=none owners=2"
        ENVIRONMENT
            "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
            "RJ_CONSAN_MODE=record-replay"
            "RJ_CONSAN_MOI_INIT_OWNER_EPOCH=${_sim_SHARED_INIT_OWNER_EPOCH}"
            "RJ_CONSAN_MOI_TRACK_BARRIERS=0"
            "RJ_CONSAN_MOI_TRACK_ATOMICS=0"
            "RJ_CONSAN_TEST_FORCE_VGPR_SPILL=1"
            "RJ_CONSAN_TEST_KERNEL_FILTER=${_sim_SHARED_KERNEL}"
            "RJ_CONSAN_LOG=3"
            "RJ_CONSAN_MAX_PATCHES=1"
            "RJ_CONSAN_REQUIRE_PATCH=1"
    )
endfunction()

function(rj_add_consan_cdna_sim_cases)
    set(_one_value_args PREFIX LDS_TARGET MOI_TARGET CONFIG)
    cmake_parse_arguments(PARSE_ARGV 0 _sim "" "${_one_value_args}" "")
    if(_sim_UNPARSED_ARGUMENTS)
        message(
            FATAL_ERROR
            "unknown ConSan CDNA simulator arguments: ${_sim_UNPARSED_ARGUMENTS}"
        )
    endif()
    foreach(_required IN ITEMS PREFIX LDS_TARGET MOI_TARGET CONFIG)
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
        MOI_TARGET ${_sim_MOI_TARGET}
        CONFIG "${_sim_CONFIG}"
        RECORD_TEST "${RJ_CONSAN_CDNA_MOI_FORCED_SPILL_RECORD_TEST}"
        SAMPLED_TEST "${RJ_CONSAN_CDNA_MOI_FORCED_SPILL_SAMPLED_TEST}"
        SHARED_TEST "${RJ_CONSAN_CDNA_MOI_SHARED_HELPER_TEST}"
        ACCESS_KERNEL "moi_cdna_forced_spill_test_kernel_for_instrumentation"
        SHARED_KERNEL "moi_cdna_shared_lds_helper_for_instrumentation"
        SHARED_INIT_OWNER_EPOCH 1
    )

    # CDNA's focused MOI fixture has spill-specific inline, synchronization,
    # atomic, and dynamic-stack cases; keep those family-specific below.
    rj_add_consan_hip_case(
        NAME "${_sim_PREFIX}.MoiInlineShadowForcedSpill"
        TARGET ${_sim_MOI_TARGET}
        FILTER "${RJ_CONSAN_CDNA_MOI_FORCED_SPILL_INLINE_TEST}"
        CONFIG "${_sim_CONFIG}"
        TIMEOUT 60
        RUN_SERIAL
        PASS_REGEX "site=access.*source=spill reason=none"
        ENVIRONMENT
            "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
            "RJ_CONSAN_MODE=inline-shadow"
            "RJ_CONSAN_MOI_REQUIRE_RECORDS=1"
            "RJ_CONSAN_MOI_FORBID_DIAGNOSTICS=1"
            "RJ_CONSAN_MOI_TRACK_BARRIERS=0"
            "RJ_CONSAN_MOI_TRACK_ATOMICS=0"
            "RJ_CONSAN_TEST_FORCE_VGPR_SPILL=1"
            "RJ_CONSAN_TEST_KERNEL_FILTER=moi_cdna_forced_spill_test_kernel_for_instrumentation"
            "RJ_CONSAN_LOG=3"
            "RJ_CONSAN_MAX_PATCHES=1"
            "RJ_CONSAN_REQUIRE_PATCH=1"
    )
    rj_add_consan_hip_case(
        NAME "${_sim_PREFIX}.MoiBarrierForcedSpill"
        TARGET ${_sim_MOI_TARGET}
        FILTER "${RJ_CONSAN_CDNA_MOI_BARRIER_TEST}"
        CONFIG "${_sim_CONFIG}"
        TIMEOUT 60
        RUN_SERIAL
        PASS_REGEX "site=barrier.*source=spill reason=none"
        ENVIRONMENT
            "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
            "RJ_CONSAN_MODE=record-replay"
            "RJ_CONSAN_MOI_INIT_OWNER_EPOCH=0"
            "RJ_CONSAN_MOI_TRACK_BARRIERS=1"
            "RJ_CONSAN_MOI_TRACK_ATOMICS=0"
            "RJ_CONSAN_TEST_FORCE_VGPR_SPILL=1"
            "RJ_CONSAN_TEST_KERNEL_FILTER=moi_cdna_barrier_forced_spill_kernel_for_instrumentation"
            "RJ_CONSAN_LOG=3"
            "RJ_CONSAN_MAX_PATCHES=2"
            "RJ_CONSAN_REQUIRE_PATCH=1"
    )
    rj_add_consan_hip_case(
        NAME "${_sim_PREFIX}.MoiAtomicForcedSpill"
        TARGET ${_sim_MOI_TARGET}
        FILTER "${RJ_CONSAN_CDNA_MOI_ATOMIC_TEST}"
        CONFIG "${_sim_CONFIG}"
        TIMEOUT 60
        RUN_SERIAL
        PASS_REGEX "site=atomic.*source=spill reason=none"
        ENVIRONMENT
            "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
            "RJ_CONSAN_MODE=record-replay"
            "RJ_CONSAN_MOI_TRACK_BARRIERS=0"
            "RJ_CONSAN_MOI_TRACK_ATOMICS=1"
            "RJ_CONSAN_TEST_FORCE_VGPR_SPILL=1"
            "RJ_CONSAN_TEST_KERNEL_FILTER=moi_cdna_atomic_forced_spill_kernel_for_instrumentation"
            "RJ_CONSAN_LOG=3"
            "RJ_CONSAN_MAX_PATCHES=1"
            "RJ_CONSAN_REQUIRE_PATCH=1"
    )
    rj_add_consan_hip_case(
        NAME "${_sim_PREFIX}.MoiDynamicStackInlineForcedSpill"
        TARGET ${_sim_MOI_TARGET}
        FILTER "${RJ_CONSAN_CDNA_MOI_DYNAMIC_STACK_TEST}"
        CONFIG "${_sim_CONFIG}"
        TIMEOUT 60
        RUN_SERIAL
        PASS_REGEX "site=access.*source=spill reason=none"
        ENVIRONMENT
            "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
            "RJ_CONSAN_MODE=inline-shadow"
            "RJ_CONSAN_MOI_REQUIRE_RECORDS=1"
            "RJ_CONSAN_MOI_FORBID_DIAGNOSTICS=1"
            "RJ_CONSAN_MOI_TRACK_BARRIERS=0"
            "RJ_CONSAN_MOI_TRACK_ATOMICS=0"
            "RJ_CONSAN_TEST_FORCE_VGPR_SPILL=1"
            "RJ_CONSAN_TEST_KERNEL_FILTER=moi_cdna_dynamic_stack_inline_spill_kernel_for_instrumentation"
            "RJ_CONSAN_LOG=3"
            "RJ_CONSAN_MAX_PATCHES=1"
            "RJ_CONSAN_REQUIRE_PATCH=1"
    )
endfunction()

function(rj_add_consan_gfx1250_sim_cases)
    set(_one_value_args PREFIX LDS_TARGET MOI_TARGET CONFIG)
    cmake_parse_arguments(PARSE_ARGV 0 _sim "" "${_one_value_args}" "")
    if(_sim_UNPARSED_ARGUMENTS)
        message(
            FATAL_ERROR
            "unknown ConSan gfx1250 simulator arguments: ${_sim_UNPARSED_ARGUMENTS}"
        )
    endif()
    foreach(_required IN ITEMS PREFIX LDS_TARGET MOI_TARGET CONFIG)
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
        MOI_TARGET ${_sim_MOI_TARGET}
        CONFIG "${_sim_CONFIG}"
        RECORD_TEST "${RJ_CONSAN_MOI_FORCED_SPILL_RECORD_TEST}"
        SAMPLED_TEST "${RJ_CONSAN_MOI_FORCED_SPILL_SAMPLED_TEST}"
        SHARED_TEST "${RJ_CONSAN_MOI_SHARED_NOINLINE_HELPER_TEST}"
        ACCESS_KERNEL "moi_forced_spill_preserves_live_vgprs_test_kernel_for_instrumentation"
        SHARED_KERNEL "moi_shared_noinline_lds_helper_for_instrumentation"
        SHARED_INIT_OWNER_EPOCH 0
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
            "RJ_CONSAN_DELAY_MODE=sleep_var"
            "RJ_CONSAN_DELAY_VAR_SSRC=0"
            "RJ_CONSAN_DELAY=1"
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
            "RJ_CONSAN_DELAY_MODE=sleep_var"
            "RJ_CONSAN_DELAY_VAR_SSRC=0"
            "RJ_CONSAN_DELAY=1"
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
            "RJ_CONSAN_DELAY_MODE=sleep_var"
            "RJ_CONSAN_DELAY_VAR_SSRC=0"
            "RJ_CONSAN_DELAY=1"
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
            "RJ_CONSAN_DELAY_MODE=sleep_var"
            "RJ_CONSAN_DELAY_VAR_SSRC=0"
            "RJ_CONSAN_DELAY=1"
            "RJ_CONSAN_TEST_KERNEL_FILTER=padded_zero_exec_native_lds_load_preserves_allocated_scalars_for_instrumentation"
            "RJ_CONSAN_LOG=3"
            "RJ_CONSAN_MAX_PATCHES=1"
            "RJ_CONSAN_REQUIRE_PATCH=1"
    )

    rj_add_consan_hip_case(
        NAME "${_sim_PREFIX}.MoiRelocatedB128LoadRecordReplay"
        TARGET ${_sim_MOI_TARGET}
        FILTER
            "ConSanMoiHipTest.RelocatedB128LoadAfterBarrierPreservesLoadedValue"
        CONFIG "${_sim_CONFIG}"
        TIMEOUT 60
        RUN_SERIAL
        ENVIRONMENT
            "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
            "RJ_CONSAN_MODE=record-replay"
            "RJ_CONSAN_MOI_AUTO_REPORT_BUFFER_SIZE=1048576"
            "RJ_CONSAN_MOI_DYNAMIC_ACCESS_RECORDS=1"
            "RJ_CONSAN_MOI_INIT_OWNER_EPOCH=1"
            "RJ_CONSAN_MOI_TRACK_BARRIERS=1"
            "RJ_CONSAN_MOI_TRACK_ATOMICS=0"
            "RJ_CONSAN_MOI_REQUIRE_RECORDS=1"
            "RJ_CONSAN_TEST_KERNEL_FILTER=moi_relocated_b128_load_after_barrier_for_instrumentation"
            "RJ_CONSAN_LOG=1"
            "RJ_CONSAN_MAX_PATCHES=1"
            "RJ_CONSAN_REQUIRE_PATCH=1"
    )

    # gfx1250 uses the general RDNA fixture, whose focused inline,
    # synchronization, atomic, and dynamic-stack cases differ from CDNA.
    rj_add_consan_hip_case(
        NAME "${_sim_PREFIX}.MoiInlineShadowAtomic"
        TARGET ${_sim_MOI_TARGET}
        FILTER "${RJ_CONSAN_MOI_SHARED_ATOMIC_HELPER_TEST}"
        CONFIG "${_sim_CONFIG}"
        TIMEOUT 60
        RUN_SERIAL
        PASS_REGEX "ConSan MOI auto inline-atomic-release.*address=0x[1-9a-fA-F]"
        ENVIRONMENT
            "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
            "RJ_CONSAN_MODE=inline-shadow"
            "RJ_CONSAN_MOI_REQUIRE_RECORDS=1"
            "RJ_CONSAN_MOI_FORBID_DIAGNOSTICS=1"
            "RJ_CONSAN_MOI_TRACK_BARRIERS=0"
            "RJ_CONSAN_MOI_TRACK_ATOMICS=1"
            "RJ_CONSAN_TEST_FORCE_VGPR_SPILL=1"
            "RJ_CONSAN_TEST_KERNEL_FILTER=moi_shared_noinline_atomic_add_for_instrumentation"
            "RJ_CONSAN_LOG=1"
            "RJ_CONSAN_MAX_PATCHES=1"
            "RJ_CONSAN_REQUIRE_PATCH=1"
    )
    rj_add_consan_hip_case(
        NAME "${_sim_PREFIX}.MoiBarrierRecordReplay"
        TARGET ${_sim_MOI_TARGET}
        FILTER "${RJ_CONSAN_LDS_MOI_BARRIER_RECORD_TEST}"
        CONFIG "${_sim_CONFIG}"
        TIMEOUT 60
        RUN_SERIAL
        PASS_REGEX "site=barrier.*reason=none"
        ENVIRONMENT
            "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
            "RJ_CONSAN_MODE=record-replay"
            "RJ_CONSAN_MOI_INIT_OWNER_EPOCH=0"
            "RJ_CONSAN_MOI_TRACK_BARRIERS=1"
            "RJ_CONSAN_MOI_TRACK_ATOMICS=0"
            "RJ_CONSAN_TEST_KERNEL_FILTER=moi_two_wave_lds_accesses_with_barrier"
            "RJ_CONSAN_LOG=3"
            "RJ_CONSAN_MAX_PATCHES=8"
            "RJ_CONSAN_REQUIRE_PATCH=1"
    )
    rj_add_consan_hip_case(
        NAME "${_sim_PREFIX}.MoiAtomicRecordReplay"
        TARGET ${_sim_MOI_TARGET}
        FILTER "${RJ_CONSAN_LDS_MOI_ATOMIC_RECORD_TEST}"
        CONFIG "${_sim_CONFIG}"
        TIMEOUT 60
        RUN_SERIAL
        PASS_REGEX "site=atomic.*reason=none"
        ENVIRONMENT
            "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
            "RJ_CONSAN_MODE=record-replay"
            "RJ_CONSAN_MOI_TRACK_BARRIERS=0"
            "RJ_CONSAN_MOI_TRACK_ATOMICS=1"
            "RJ_CONSAN_TEST_KERNEL_FILTER=moi_flat_atomic_add_record"
            "RJ_CONSAN_LOG=3"
            "RJ_CONSAN_MAX_PATCHES=1"
            "RJ_CONSAN_REQUIRE_PATCH=1"
    )
    rj_add_consan_hip_case(
        NAME "${_sim_PREFIX}.MoiDynamicStackRecordReplayForcedSpill"
        TARGET ${_sim_MOI_TARGET}
        FILTER "${RJ_CONSAN_MOI_DYNAMIC_PRIVATE_STACK_TEST}"
        CONFIG "${_sim_CONFIG}"
        TIMEOUT 60
        RUN_SERIAL
        PASS_REGEX
            "kind=trampoline-moi-[a-z-]*[ -~]*spilled_vgprs=[1-9][0-9]*[ -~]*dynamic_private_addend=[1-9][0-9]*"
        ENVIRONMENT
            "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
            "RJ_CONSAN_MODE=record-replay"
            "RJ_CONSAN_MOI_REQUIRE_RECORDS=1"
            "RJ_CONSAN_MOI_FORBID_DIAGNOSTICS=1"
            "RJ_CONSAN_MOI_TRACK_BARRIERS=0"
            "RJ_CONSAN_MOI_TRACK_ATOMICS=0"
            "RJ_CONSAN_TEST_FORCE_VGPR_SPILL=1"
            "RJ_CONSAN_TEST_KERNEL_FILTER=moi_dynamic_private_stack_test_kernel_for_instrumentation"
            "RJ_CONSAN_LOG=3"
            "RJ_CONSAN_MAX_PATCHES=1"
            "RJ_CONSAN_REQUIRE_PATCH=1"
    )
    rj_add_consan_hip_case(
        NAME "${_sim_PREFIX}.MoiDynamicStackSampledForcedSpill"
        TARGET ${_sim_MOI_TARGET}
        FILTER "${RJ_CONSAN_MOI_DYNAMIC_PRIVATE_STACK_TEST}"
        CONFIG "${_sim_CONFIG}"
        TIMEOUT 60
        RUN_SERIAL
        PASS_REGEX
            "kind=trampoline-moi-[a-z-]*[ -~]*spilled_vgprs=[1-9][0-9]*[ -~]*dynamic_private_addend=[1-9][0-9]*"
        ENVIRONMENT
            "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
            "RJ_CONSAN_MODE=sampled"
            "RJ_CONSAN_MOI_REQUIRE_RECORDS=1"
            "RJ_CONSAN_MOI_FORBID_DIAGNOSTICS=1"
            "RJ_CONSAN_MOI_TRACK_BARRIERS=0"
            "RJ_CONSAN_MOI_TRACK_ATOMICS=0"
            "RJ_CONSAN_TEST_FORCE_VGPR_SPILL=1"
            "RJ_CONSAN_TEST_KERNEL_FILTER=moi_dynamic_private_stack_test_kernel_for_instrumentation"
            "RJ_CONSAN_LOG=3"
            "RJ_CONSAN_MAX_PATCHES=1"
            "RJ_CONSAN_REQUIRE_PATCH=1"
    )
endfunction()

function(rj_add_consan_fuzzers)
    if(RJ_BUILD_FUZZERS)
        set(_rj_consan_transform_fuzz_targets)
        foreach(
            _rj_profile
            IN
            ITEMS supercollider record_replay inline_shadow sampled
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
    set(_rj_consan_moi_abi_header
        "${_rj_consan_include_dir}/rocjitsu/code/patch/consan/consan_moi_abi.h"
    )
    if(RJ_HIP_RUNTIME_SUPPORTS_GFX942)
        # Keep a native gfx942 code object available on every host so CDNA3
        # ConSan can be exercised through the RocJITsu simulator.
        rj_add_hip_test(
                hip_consan_lds_gfx942_test
                ARCH gfx942
                SOURCE dbi/consan/hip_consan_lds_test.hip
        )
        rj_add_hip_test(
                hip_consan_moi_gfx942_test
                ARCH gfx942
                SOURCE dbi/consan/hip_consan_moi_cdna_test.hip
                INCLUDE_DIRS "${_rj_consan_include_dir}"
                EXTRA_DEPENDS "${_rj_consan_moi_abi_header}"
        )
        add_dependencies(hip_consan_lds_gfx942_test_target rocjitsu_dbi_hooks)
        add_dependencies(hip_consan_moi_gfx942_test_target rocjitsu_dbi_hooks)
        rj_add_consan_cdna_sim_cases(
            PREFIX "ConSanGfx942Sim"
            LDS_TARGET hip_consan_lds_gfx942_test
            MOI_TARGET hip_consan_moi_gfx942_test
            CONFIG "${CMAKE_SOURCE_DIR}/configs/gfx942_cdna3_kmd.json"
        )
    endif()
    if(RJ_HIP_RUNTIME_SUPPORTS_GFX950)
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
                hip_consan_moi_gfx950_test
                ARCH gfx950
                SOURCE dbi/consan/hip_consan_moi_cdna_test.hip
                INCLUDE_DIRS "${_rj_consan_include_dir}"
                EXTRA_DEPENDS "${_rj_consan_moi_abi_header}"
        )
        add_dependencies(hip_consan_lds_gfx950_test_target rocjitsu_dbi_hooks)
        add_dependencies(hip_consan_moi_gfx950_test_target rocjitsu_dbi_hooks)
        rj_add_consan_cdna_sim_cases(
            PREFIX "ConSanGfx950Sim"
            LDS_TARGET hip_consan_lds_gfx950_test
            MOI_TARGET hip_consan_moi_gfx950_test
            CONFIG "${CMAKE_SOURCE_DIR}/configs/gfx950_cdna4_kmd.json"
        )
        # Keep cross-compiled gfx950 fixtures available on other hosts,
        # but only register native HIP execution when gfx950 is present.
        if(HAS_GFX950_GPU)
            # Per-site resource plans are debug diagnostics. Cases whose pass
            # regex matches site/source/owner details must request log level 3.
            rj_add_consan_hip_case(
                NAME "ConSanSpillHipTest.Gfx950VgprScratchRoundTrip"
                TARGET hip_consan_spill_gfx950_test
                FILTER "ConSanSpillHipTest.Gfx950VgprScratchRoundTrip"
                RUN_SERIAL
            )
            rj_add_consan_hip_case(
                NAME "ConSanSpillHipTest.Gfx950PartialExecVgprScratchRoundTrip"
                TARGET hip_consan_spill_gfx950_test
                FILTER "ConSanSpillHipTest.Gfx950PartialExecVgprScratchRoundTrip"
                RUN_SERIAL
            )
            rj_add_consan_hip_case(
                NAME "ConSanMoiCdnaTest.DbiForcedSpillRecordReplay"
                TARGET hip_consan_moi_gfx950_test
                FILTER "${RJ_CONSAN_CDNA_MOI_FORCED_SPILL_RECORD_TEST}"
                TIMEOUT 30
                RUN_SERIAL
                PASS_REGEX "site=access.*source=spill reason=none"
                ENVIRONMENT
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=record-replay"
                    "RJ_CONSAN_MOI_TRACK_BARRIERS=0"
                    "RJ_CONSAN_MOI_TRACK_ATOMICS=0"
                    "RJ_CONSAN_TEST_FORCE_VGPR_SPILL=1"
                    "RJ_CONSAN_TEST_KERNEL_FILTER=moi_cdna_forced_spill_test_kernel_for_instrumentation"
                    "RJ_CONSAN_LOG=3"
                    "RJ_CONSAN_MAX_PATCHES=1"
                    "RJ_CONSAN_REQUIRE_PATCH=1"
            )
            rj_add_consan_hip_case(
                NAME "ConSanMoiCdnaTest.DbiForcedSpillSampled"
                TARGET hip_consan_moi_gfx950_test
                FILTER "${RJ_CONSAN_CDNA_MOI_FORCED_SPILL_SAMPLED_TEST}"
                TIMEOUT 30
                RUN_SERIAL
                PASS_REGEX "site=access.*source=spill reason=none"
                ENVIRONMENT
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=sampled"
                    "RJ_CONSAN_MOI_TRACK_BARRIERS=0"
                    "RJ_CONSAN_MOI_TRACK_ATOMICS=0"
                    "RJ_CONSAN_TEST_FORCE_VGPR_SPILL=1"
                    "RJ_CONSAN_TEST_KERNEL_FILTER=moi_cdna_forced_spill_test_kernel_for_instrumentation"
                    "RJ_CONSAN_LOG=3"
                    "RJ_CONSAN_MAX_PATCHES=1"
                    "RJ_CONSAN_REQUIRE_PATCH=1"
            )
            rj_add_consan_hip_case(
                NAME "ConSanMoiCdnaTest.DbiForcedSpillInlineShadow"
                TARGET hip_consan_moi_gfx950_test
                FILTER "${RJ_CONSAN_CDNA_MOI_FORCED_SPILL_INLINE_TEST}"
                TIMEOUT 30
                RUN_SERIAL
                PASS_REGEX "site=access.*source=spill reason=none"
                ENVIRONMENT
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=inline-shadow"
                    "RJ_CONSAN_MOI_TRACK_BARRIERS=0"
                    "RJ_CONSAN_MOI_TRACK_ATOMICS=0"
                    "RJ_CONSAN_MOI_REQUIRE_RECORDS=1"
                    "RJ_CONSAN_MOI_FORBID_DIAGNOSTICS=1"
                    "RJ_CONSAN_TEST_FORCE_VGPR_SPILL=1"
                    "RJ_CONSAN_TEST_KERNEL_FILTER=moi_cdna_forced_spill_test_kernel_for_instrumentation"
                    "RJ_CONSAN_LOG=3"
                    "RJ_CONSAN_MAX_PATCHES=1"
                    "RJ_CONSAN_REQUIRE_PATCH=1"
            )
            rj_add_consan_hip_case(
                NAME "ConSanMoiCdnaTest.DbiSharedHelperForcedSpill"
                TARGET hip_consan_moi_gfx950_test
                FILTER "${RJ_CONSAN_CDNA_MOI_SHARED_HELPER_TEST}"
                TIMEOUT 30
                RUN_SERIAL
                PASS_REGEX "site=access.*source=spill reason=none owners=2"
                ENVIRONMENT
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=record-replay"
                    "RJ_CONSAN_MOI_TRACK_BARRIERS=0"
                    "RJ_CONSAN_MOI_TRACK_ATOMICS=0"
                    "RJ_CONSAN_TEST_FORCE_VGPR_SPILL=1"
                    "RJ_CONSAN_TEST_KERNEL_FILTER=moi_cdna_shared_lds_helper_for_instrumentation"
                    "RJ_CONSAN_LOG=3"
                    "RJ_CONSAN_MAX_PATCHES=1"
                    "RJ_CONSAN_REQUIRE_PATCH=1"
            )
            rj_add_consan_hip_case(
                NAME "ConSanMoiCdnaTest.DbiBarrierForcedSpill"
                TARGET hip_consan_moi_gfx950_test
                FILTER "${RJ_CONSAN_CDNA_MOI_BARRIER_TEST}"
                TIMEOUT 30
                RUN_SERIAL
                PASS_REGEX "site=barrier.*source=spill reason=none"
                ENVIRONMENT
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=record-replay"
                    "RJ_CONSAN_MOI_INIT_OWNER_EPOCH=0"
                    "RJ_CONSAN_MOI_TRACK_BARRIERS=1"
                    "RJ_CONSAN_MOI_TRACK_ATOMICS=0"
                    "RJ_CONSAN_TEST_FORCE_VGPR_SPILL=1"
                    "RJ_CONSAN_TEST_KERNEL_FILTER=moi_cdna_barrier_forced_spill_kernel_for_instrumentation"
                    "RJ_CONSAN_LOG=3"
                    "RJ_CONSAN_MAX_PATCHES=2"
                    "RJ_CONSAN_REQUIRE_PATCH=1"
            )
            rj_add_consan_hip_case(
                NAME "ConSanMoiCdnaTest.DbiAtomicForcedSpill"
                TARGET hip_consan_moi_gfx950_test
                FILTER "${RJ_CONSAN_CDNA_MOI_ATOMIC_TEST}"
                TIMEOUT 30
                RUN_SERIAL
                PASS_REGEX "site=atomic.*source=spill reason=none"
                ENVIRONMENT
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=record-replay"
                    "RJ_CONSAN_MOI_TRACK_BARRIERS=0"
                    "RJ_CONSAN_MOI_TRACK_ATOMICS=1"
                    "RJ_CONSAN_TEST_FORCE_VGPR_SPILL=1"
                    "RJ_CONSAN_TEST_KERNEL_FILTER=moi_cdna_atomic_forced_spill_kernel_for_instrumentation"
                    "RJ_CONSAN_LOG=3"
                    "RJ_CONSAN_MAX_PATCHES=1"
                    "RJ_CONSAN_REQUIRE_PATCH=1"
            )
            rj_add_consan_hip_case(
                NAME "ConSanMoiCdnaTest.DbiDynamicStackInlineForcedSpill"
                TARGET hip_consan_moi_gfx950_test
                FILTER "${RJ_CONSAN_CDNA_MOI_DYNAMIC_STACK_TEST}"
                TIMEOUT 30
                RUN_SERIAL
                PASS_REGEX "site=access.*source=spill reason=none"
                ENVIRONMENT
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=inline-shadow"
                    "RJ_CONSAN_MOI_TRACK_BARRIERS=0"
                    "RJ_CONSAN_MOI_TRACK_ATOMICS=0"
                    "RJ_CONSAN_MOI_REQUIRE_RECORDS=1"
                    "RJ_CONSAN_MOI_FORBID_DIAGNOSTICS=1"
                    "RJ_CONSAN_TEST_FORCE_VGPR_SPILL=1"
                    "RJ_CONSAN_TEST_KERNEL_FILTER=moi_cdna_dynamic_stack_inline_spill_kernel_for_instrumentation"
                    "RJ_CONSAN_LOG=3"
                    "RJ_CONSAN_MAX_PATCHES=1"
                    "RJ_CONSAN_REQUIRE_PATCH=1"
            )
        endif()
    endif()
    if(RJ_HIP_RUNTIME_SUPPORTS_GFX1250)
        # Keep native gfx1250 code objects available on every host so the
        # target-specific ConSan path can be regression-tested in simulation.
        rj_add_hip_test(
                hip_consan_lds_gfx1250_test
                ARCH gfx1250
                SOURCE dbi/consan/hip_consan_lds_test.hip
        )
        rj_add_hip_test(
                hip_consan_moi_gfx1250_test
                ARCH gfx1250
                SOURCE dbi/consan/hip_consan_moi_test.hip
                INCLUDE_DIRS "${_rj_consan_include_dir}"
                EXTRA_DEPENDS "${_rj_consan_moi_abi_header}"
        )
        add_dependencies(hip_consan_lds_gfx1250_test_target rocjitsu_dbi_hooks)
        add_dependencies(hip_consan_moi_gfx1250_test_target rocjitsu_dbi_hooks)
        rj_add_consan_gfx1250_sim_cases(
            PREFIX "ConSanGfx1250Sim"
            LDS_TARGET hip_consan_lds_gfx1250_test
            MOI_TARGET hip_consan_moi_gfx1250_test
            CONFIG "${CMAKE_SOURCE_DIR}/configs/gfx1250.json"
        )
    endif()
    rj_add_hip_test(
            hip_consan_lds_test
            ARCH gfx1201
            SOURCE dbi/consan/hip_consan_lds_test.hip
    )
    add_dependencies(hip_consan_lds_test_target rocjitsu_dbi_hooks)
    rj_add_hip_test(
            hip_consan_inline_shadow_test
            ARCH gfx1201
            SOURCE dbi/consan/hip_consan_inline_shadow_test.hip
    )
    add_dependencies(hip_consan_inline_shadow_test_target rocjitsu_dbi_hooks)
    rj_add_hip_test(
            hip_consan_moi_test
            ARCH gfx1201
            SOURCE dbi/consan/hip_consan_moi_test.hip
            INCLUDE_DIRS "${_rj_consan_include_dir}"
            EXTRA_DEPENDS "${_rj_consan_moi_abi_header}"
    )
    add_dependencies(hip_consan_moi_test_target rocjitsu_dbi_hooks)
    # Deliberately ill-formed GPU synchronization. The gtests are disabled
    # by default and must only run through consan_fault_runner.py with the
    # destructive opt-in, timeout, health checks, and quarantine enabled.
    rj_add_hip_test(
            hip_consan_malformed_sync_test
            ARCH gfx1201
            SOURCE dbi/consan/hip_consan_malformed_sync_test.hip
    )
    if(HAS_GFX1201_GPU)
        set(RJ_CONSAN_LDS_RACY_TEST
            "ConSanLdsTest.PaddedRacyStoreCompletesOrTripsDbiTrap"
        )
        set(RJ_CONSAN_LDS_CLEAN_REPORT_TEST
            "ConSanLdsTest.PaddedCleanStoreReportBufferStaysZero"
        )
        set(RJ_CONSAN_LDS_RACY_REPORT_TEST
            "ConSanLdsTest.PaddedRacyStoreReportsMarker"
        )
        set(RJ_CONSAN_LDS_INLINE_SHADOW_RACY_TEST
            "ConSanInlineShadowTest.CrossWaveRacyStoreCompletes"
        )
        set(RJ_CONSAN_LDS_INLINE_SHADOW_B128_RACY_TEST
            "ConSanInlineShadowTest.CrossWaveRacyB128StoreCompletes"
        )
        set(RJ_CONSAN_LDS_INLINE_SHADOW_FLAT_RACY_TEST
            "ConSanInlineShadowTest.CrossWaveRacyFlatLdsStoreCompletes"
        )
        set(RJ_CONSAN_LDS_INLINE_SHADOW_BARRIER_TEST
            "ConSanInlineShadowTest.BarrierOrderedCrossWaveStoreCompletes"
        )
        set(RJ_CONSAN_LDS_INLINE_SHADOW_FENCE_BARRIER_TEST
            "ConSanInlineShadowTest.FenceAndBarrierOrderedAccessesPreserveExactOutput"
        )
        set(RJ_CONSAN_LDS_INLINE_SHADOW_ATOMIC_SAME_ADDRESS_TEST
            "ConSanInlineShadowTest.SameAddressAtomicHandoffCompletes"
        )
        set(RJ_CONSAN_LDS_INLINE_SHADOW_ATOMIC_WRONG_ADDRESS_TEST
            "ConSanInlineShadowTest.WrongAddressAtomicHandoffCompletes"
        )
        set(RJ_CONSAN_LDS_INLINE_SHADOW_ATOMIC_COLLISION_TEST
            "ConSanInlineShadowTest.CollidingReleaseReplacementCompletes"
        )
        set(RJ_CONSAN_LDS_INLINE_SHADOW_ATOMIC_CAS_TEST
            "ConSanInlineShadowTest.FailedCasDoesNotReplaceReleaseCompletes"
        )
        set(RJ_CONSAN_LDS_MOI_RECORD_TEST
            "ConSanMoiHipTest.TwoWaveAccessesWriteDynamicRecords"
        )
        set(RJ_CONSAN_LDS_MOI_SAMPLED_DIRECT_TEST
            "ConSanMoiHipTest.SampledBackendWritesDirectEntries"
        )
        set(RJ_CONSAN_LDS_MOI_DYNAMIC_ACCESS_RECORD_TEST
            "ConSanMoiHipTest.DynamicAccessRecordsAppendActiveLanes"
        )
        set(RJ_CONSAN_LDS_MOI_AUTO_REPLAY_CONFLICT_TEST
            "ConSanMoiHipTest.TwoWaveAccessesRunWithAutoReportBuffer"
        )
        set(RJ_CONSAN_LDS_MOI_AUTO_REPEATED_DISPATCH_TEST
            "ConSanMoiHipTest.RepeatedDispatchesRetainAutomaticRecords"
        )
        set(RJ_CONSAN_LDS_MOI_WORKGROUP_RECORD_TEST
            "ConSanMoiHipTest.EquivalentAccessesInDifferentWorkgroupsDoNotConflict"
        )
        set(RJ_CONSAN_LDS_MOI_WORKGROUP_2D_RECORD_TEST
            "ConSanMoiHipTest.EquivalentAccessesInDifferent2DWorkgroupsDoNotConflict"
        )
        set(RJ_CONSAN_LDS_MOI_WORKGROUP_3D_RECORD_TEST
            "ConSanMoiHipTest.EquivalentAccessesInDifferent3DWorkgroupsDoNotConflict"
        )
        set(RJ_CONSAN_LDS_MOI_LIMITED_RECORD_TEST
            "ConSanMoiHipTest.OneRecordCapacityLimitsAccessPatches"
        )
        set(RJ_CONSAN_LDS_MOI_ATOMIC_HANDOFF_TEST
            "ConSanMoiHipTest.AtomicHandoffOrdersTwoWaveAccesses"
        )
        set(RJ_CONSAN_GFX1201_VGPR_SPILL_TEST
            "ConSanSpillHipTest.Gfx1201VgprScratchRoundTrip"
        )
        set(RJ_CONSAN_MOI_FULL_VGPR_FILE_TEST
            "ConSanMoiHipTest.FullVgprFileForcedSpillPreservesOutput"
        )
        set(RJ_CONSAN_MOI_STATIC_PRIVATE_STACK_TEST
            "ConSanMoiHipTest.StaticPrivateStackForcedSpillPreservesOutput"
        )
        rj_add_consan_hip_case(
                NAME "ConSanSpillHipTest.Gfx1201VgprScratchRoundTrip"
                TARGET hip_consan_moi_test
                FILTER "${RJ_CONSAN_GFX1201_VGPR_SPILL_TEST}"
                TIMEOUT 30
        )
        rj_add_consan_hip_case(
                NAME "ConSanMoiHipTest.DbiForcedSpillRecordReplayPreservesLiveVgprs"
                TARGET hip_consan_moi_test
                FILTER "${RJ_CONSAN_MOI_FORCED_SPILL_RECORD_TEST}"
                TIMEOUT 30
                ENVIRONMENT
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=record-replay"
                    "RJ_CONSAN_MOI_TRACK_BARRIERS=0"
                    "RJ_CONSAN_MOI_TRACK_ATOMICS=0"
                    "RJ_CONSAN_TEST_FORCE_VGPR_SPILL=1"
                    "RJ_CONSAN_TEST_KERNEL_FILTER=moi_forced_spill_preserves_live_vgprs_test_kernel_for_instrumentation"
                    "RJ_CONSAN_LOG=1"
                    "RJ_CONSAN_MAX_PATCHES=1"
                    "RJ_CONSAN_REQUIRE_PATCH=1"
        )
        rj_add_consan_hip_case(
                NAME "ConSanMoiHipTest.DbiForcedSpillSampledPreservesLiveVgprs"
                TARGET hip_consan_moi_test
                FILTER "${RJ_CONSAN_MOI_FORCED_SPILL_SAMPLED_TEST}"
                TIMEOUT 30
                ENVIRONMENT
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=sampled"
                    "RJ_CONSAN_MOI_TRACK_BARRIERS=0"
                    "RJ_CONSAN_MOI_TRACK_ATOMICS=0"
                    "RJ_CONSAN_TEST_FORCE_VGPR_SPILL=1"
                    "RJ_CONSAN_TEST_KERNEL_FILTER=moi_forced_spill_preserves_live_vgprs_test_kernel_for_instrumentation"
                    "RJ_CONSAN_LOG=1"
                    "RJ_CONSAN_MAX_PATCHES=1"
                    "RJ_CONSAN_REQUIRE_PATCH=1"
        )
        rj_add_consan_hip_case(
                NAME "ConSanMoiHipTest.BaselineFullVgprFilePreservesOutput"
                TARGET hip_consan_moi_test
                FILTER "${RJ_CONSAN_MOI_FULL_VGPR_FILE_TEST}"
                TIMEOUT 30
        )
        # Resource-plan cases that match debug-only site/source or
        # owner/private-size diagnostics require log level 3.
        rj_add_consan_hip_case(
                NAME "ConSanMoiHipTest.DbiFullVgprFileForcedSpillPreservesOutput"
                TARGET hip_consan_moi_test
                FILTER "${RJ_CONSAN_MOI_FULL_VGPR_FILE_TEST}"
                TIMEOUT 30
                ENVIRONMENT
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=record-replay"
                    "RJ_CONSAN_MOI_INIT_OWNER_EPOCH=0"
                    "RJ_CONSAN_TEST_FORCE_VGPR_SPILL=1"
                    "RJ_CONSAN_TEST_KERNEL_FILTER=moi_full_vgpr_file_test_kernel_for_instrumentation"
                    "RJ_CONSAN_LOG=3"
                    "RJ_CONSAN_MAX_PATCHES=1"
                    "RJ_CONSAN_REQUIRE_PATCH=1"
        )
        set_tests_properties(
            "ConSanMoiHipTest.DbiFullVgprFileForcedSpillPreservesOutput"
            PROPERTIES
                PASS_REGULAR_EXPRESSION
                    "source=spill reason=none.*current_vgprs=256 max_referenced_vgprs=256"
        )
        rj_add_consan_hip_case(
                NAME "ConSanMoiHipTest.DbiSharedNoinlineHelperPatchPreservesOutput"
                TARGET hip_consan_moi_test
                FILTER "${RJ_CONSAN_MOI_SHARED_NOINLINE_HELPER_TEST}"
                TIMEOUT 30
                ENVIRONMENT
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=record-replay"
                    "RJ_CONSAN_TEST_KERNEL_FILTER=moi_shared_noinline_lds_helper_for_instrumentation"
                    "RJ_CONSAN_LOG=3"
                    "RJ_CONSAN_MAX_PATCHES=1"
                    "RJ_CONSAN_REQUIRE_PATCH=1"
        )
        set_tests_properties(
            "ConSanMoiHipTest.DbiSharedNoinlineHelperPatchPreservesOutput"
            PROPERTIES
                PASS_REGULAR_EXPRESSION
                    "site=access.*source=descriptor-growth reason=none owners=2"
        )
        rj_add_consan_hip_case(
                NAME "ConSanMoiHipTest.DbiSharedNoinlineHelperSpillPreservesOutput"
                TARGET hip_consan_moi_test
                FILTER "${RJ_CONSAN_MOI_SHARED_NOINLINE_HELPER_TEST}"
                TIMEOUT 30
                ENVIRONMENT
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=record-replay"
                    "RJ_CONSAN_MOI_INIT_OWNER_EPOCH=0"
                    "RJ_CONSAN_TEST_FORCE_VGPR_SPILL=1"
                    "RJ_CONSAN_TEST_KERNEL_FILTER=moi_shared_noinline_lds_helper_for_instrumentation"
                    "RJ_CONSAN_LOG=3"
                    "RJ_CONSAN_MAX_PATCHES=1"
                    "RJ_CONSAN_REQUIRE_PATCH=1"
        )
        set_tests_properties(
            "ConSanMoiHipTest.DbiSharedNoinlineHelperSpillPreservesOutput"
            PROPERTIES
                PASS_REGULAR_EXPRESSION
                    "site=access.*source=spill reason=none owners=2"
        )
        rj_add_consan_hip_case(
                NAME "ConSanMoiHipTest.DbiSharedAtomicHelperRecordReplayForcedSpill"
                TARGET hip_consan_moi_test
                FILTER "${RJ_CONSAN_MOI_SHARED_ATOMIC_HELPER_TEST}"
                TIMEOUT 30
                ENVIRONMENT
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=record-replay"
                    "RJ_CONSAN_MOI_TRACK_ATOMICS=1"
                    "RJ_CONSAN_TEST_FORCE_VGPR_SPILL=1"
                    "RJ_CONSAN_TEST_KERNEL_FILTER=moi_shared_noinline_atomic_add_for_instrumentation"
                    "RJ_CONSAN_MOI_REQUIRE_RECORDS=1"
                    "RJ_CONSAN_LOG=3"
                    "RJ_CONSAN_MAX_PATCHES=1"
                    "RJ_CONSAN_REQUIRE_PATCH=1"
        )
        set_tests_properties(
            "ConSanMoiHipTest.DbiSharedAtomicHelperRecordReplayForcedSpill"
            PROPERTIES
                PASS_REGULAR_EXPRESSION
                    "site=atomic.*source=spill reason=none owners=2"
        )
        rj_add_consan_hip_case(
                NAME "ConSanMoiHipTest.DbiSharedAtomicHelperInlineShadowForcedSpill"
                TARGET hip_consan_moi_test
                FILTER "${RJ_CONSAN_MOI_SHARED_ATOMIC_HELPER_TEST}"
                TIMEOUT 30
                ENVIRONMENT
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=inline-shadow"
                    "RJ_CONSAN_MOI_TRACK_ATOMICS=1"
                    "RJ_CONSAN_TEST_FORCE_VGPR_SPILL=1"
                    "RJ_CONSAN_TEST_KERNEL_FILTER=moi_shared_noinline_atomic_add_for_instrumentation"
                    "RJ_CONSAN_MOI_REQUIRE_RECORDS=1"
                    "RJ_CONSAN_MOI_FORBID_DIAGNOSTICS=1"
                    "RJ_CONSAN_LOG=1"
                    "RJ_CONSAN_MAX_PATCHES=1"
                    "RJ_CONSAN_REQUIRE_PATCH=1"
        )
        set_tests_properties(
            "ConSanMoiHipTest.DbiSharedAtomicHelperInlineShadowForcedSpill"
            PROPERTIES
                PASS_REGULAR_EXPRESSION
                    "ConSan MOI auto inline-atomic-release.*address=0x[1-9a-fA-F]"
        )
        rj_add_consan_hip_case(
                NAME "ConSanMoiHipTest.BaselineStaticPrivateStackPreservesOutput"
                TARGET hip_consan_moi_test
                FILTER "${RJ_CONSAN_MOI_STATIC_PRIVATE_STACK_TEST}"
                TIMEOUT 30
        )
        rj_add_consan_hip_case(
                NAME "ConSanMoiHipTest.DbiStaticPrivateStackForcedSpillPreservesOutput"
                TARGET hip_consan_moi_test
                FILTER "${RJ_CONSAN_MOI_STATIC_PRIVATE_STACK_TEST}"
                TIMEOUT 30
                ENVIRONMENT
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=record-replay"
                    "RJ_CONSAN_TEST_FORCE_VGPR_SPILL=1"
                    "RJ_CONSAN_TEST_KERNEL_FILTER=moi_static_private_stack_test_kernel_for_instrumentation"
                    "RJ_CONSAN_LOG=3"
                    "RJ_CONSAN_MAX_PATCHES=1"
                    "RJ_CONSAN_REQUIRE_PATCH=1"
        )
        set_tests_properties(
            "ConSanMoiHipTest.DbiStaticPrivateStackForcedSpillPreservesOutput"
            PROPERTIES
                PASS_REGULAR_EXPRESSION
                    "owner_names=.*moi_static_private_stack.*private_bytes=[1-9][0-9]*"
        )
        rj_add_consan_hip_case(
                NAME "ConSanMoiHipTest.BaselineDynamicPrivateStackPreservesOutput"
                TARGET hip_consan_moi_test
                FILTER "${RJ_CONSAN_MOI_DYNAMIC_PRIVATE_STACK_TEST}"
                TIMEOUT 30
        )
        rj_add_consan_hip_case(
                NAME "ConSanMoiHipTest.DbiDynamicPrivateStackRecordReplayForcedSpillPreservesOutput"
                TARGET hip_consan_moi_test
                FILTER "${RJ_CONSAN_MOI_DYNAMIC_PRIVATE_STACK_TEST}"
                TIMEOUT 30
                PASS_REGEX
                    "kind=trampoline-moi-[a-z-]*[ -~]*spilled_vgprs=[1-9][0-9]*[ -~]*dynamic_private_addend=[1-9][0-9]*"
                ENVIRONMENT
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=record-replay"
                    "RJ_CONSAN_MOI_REQUIRE_RECORDS=1"
                    "RJ_CONSAN_MOI_FORBID_DIAGNOSTICS=1"
                    "RJ_CONSAN_TEST_FORCE_VGPR_SPILL=1"
                    "RJ_CONSAN_TEST_KERNEL_FILTER=moi_dynamic_private_stack_test_kernel_for_instrumentation"
                    "RJ_CONSAN_LOG=3"
                    "RJ_CONSAN_MAX_PATCHES=1"
                    "RJ_CONSAN_REQUIRE_PATCH=1"
        )
        rj_add_consan_hip_case(
                NAME "ConSanMoiHipTest.DbiDynamicPrivateStackSampledForcedSpillPreservesOutput"
                TARGET hip_consan_moi_test
                FILTER "${RJ_CONSAN_MOI_DYNAMIC_PRIVATE_STACK_TEST}"
                TIMEOUT 30
                PASS_REGEX
                    "kind=trampoline-moi-[a-z-]*[ -~]*spilled_vgprs=[1-9][0-9]*[ -~]*dynamic_private_addend=[1-9][0-9]*"
                ENVIRONMENT
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=sampled"
                    "RJ_CONSAN_MOI_REQUIRE_RECORDS=1"
                    "RJ_CONSAN_MOI_FORBID_DIAGNOSTICS=1"
                    "RJ_CONSAN_TEST_FORCE_VGPR_SPILL=1"
                    "RJ_CONSAN_TEST_KERNEL_FILTER=moi_dynamic_private_stack_test_kernel_for_instrumentation"
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
                    "RJ_CONSAN_DELAY=2"
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
                    "RJ_CONSAN_DELAY=2"
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
                    "RJ_CONSAN_DELAY=2"
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
                    "RJ_CONSAN_DELAY=2"
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
                    "RJ_CONSAN_DELAY=2"
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
                    "RJ_CONSAN_DELAY=2"
                    "RJ_CONSAN_TMP_VGPR=3"
                    "RJ_CONSAN_REQUIRE_PATCH=1"
        )
        rj_add_consan_hip_case(
                NAME "ConSanInlineShadowTest.DbiReportsCrossWaveRace"
                TARGET hip_consan_inline_shadow_test
                FILTER "${RJ_CONSAN_LDS_INLINE_SHADOW_RACY_TEST}"
                TIMEOUT 30
                ENVIRONMENT
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=inline-shadow"
                    "RJ_CONSAN_MOI_REQUIRE_RECORDS=1"
                    "RJ_CONSAN_MOI_REQUIRE_DIAGNOSTICS=1"
                    "RJ_CONSAN_LOG=1"
                    "RJ_CONSAN_REQUIRE_PATCH=1"
        )
        set_tests_properties(
            "ConSanInlineShadowTest.DbiReportsCrossWaveRace"
            PROPERTIES
                PASS_REGULAR_EXPRESSION
                    "second_lanes=0x[1-9a-fA-F].*first_lds=\\[0,4\\) second_lds=\\[0,4\\)"
        )
        rj_add_consan_hip_case(
                NAME "ConSanInlineShadowTest.DbiOddExactSlotFailsClosed"
                TARGET hip_consan_inline_shadow_test
                FILTER "${RJ_CONSAN_LDS_INLINE_SHADOW_RACY_TEST}"
                TIMEOUT 30
                ENVIRONMENT
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=inline-shadow"
                    "RJ_CONSAN_TEST_SEED_INLINE_EXACT_ODD=1"
                    "RJ_CONSAN_MOI_FORBID_OVERFLOW=1"
                    "RJ_CONSAN_LOG=1"
                    "RJ_CONSAN_TEST_KERNEL_FILTER=padded_cross_wave_lds_store"
                    "RJ_CONSAN_MAX_PATCHES=1"
                    "RJ_CONSAN_REQUIRE_PATCH=1"
        )
        set_tests_properties(
            "ConSanInlineShadowTest.DbiOddExactSlotFailsClosed"
            PROPERTIES
                PASS_REGULAR_EXPRESSION
                    "ConSan MOI exact snapshot unusable: incomplete=[1-9][0-9]* changed=0 malformed=0"
        )
        rj_add_consan_hip_case(
                NAME "ConSanInlineShadowTest.DbiVersionedPublicationVisibilityConflict"
                TARGET hip_consan_inline_shadow_test
                FILTER "${RJ_CONSAN_LDS_INLINE_SHADOW_RACY_TEST}"
                TIMEOUT 60
                ENVIRONMENT
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=inline-shadow"
                    "RJ_CONSAN_MOI_REQUIRE_RECORDS=1"
                    "RJ_CONSAN_MOI_REQUIRE_DIAGNOSTICS=1"
                    "RJ_CONSAN_LOG=1"
                    "RJ_CONSAN_TEST_KERNEL_FILTER=padded_cross_wave_lds_store"
                    "RJ_CONSAN_MAX_PATCHES=4"
                    "RJ_CONSAN_REQUIRE_PATCH=1"
                    "CONSAN_INLINE_VERSIONED_LAUNCHES=100"
        )
        set_tests_properties(
            "ConSanInlineShadowTest.DbiVersionedPublicationVisibilityConflict"
            PROPERTIES
                RUN_SERIAL TRUE
                PASS_REGULAR_EXPRESSION
                    "second_lanes=0x[1-9a-fA-F].*first_lds=\\[0,4\\) second_lds=\\[0,4\\)"
        )
        rj_add_consan_hip_case(
                NAME "ConSanInlineShadowTest.DbiReportsCrossWaveB128Race"
                TARGET hip_consan_inline_shadow_test
                FILTER "${RJ_CONSAN_LDS_INLINE_SHADOW_B128_RACY_TEST}"
                TIMEOUT 30
                ENVIRONMENT
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=inline-shadow"
                    "RJ_CONSAN_MOI_REQUIRE_RECORDS=1"
                    "RJ_CONSAN_MOI_REQUIRE_DIAGNOSTICS=1"
                    "RJ_CONSAN_LOG=1"
                    "RJ_CONSAN_TEST_KERNEL_FILTER=padded_cross_wave_lds_store_b128"
                    "RJ_CONSAN_MAX_PATCHES=1"
                    "RJ_CONSAN_REQUIRE_PATCH=1"
        )
        rj_add_consan_hip_case(
                NAME "ConSanInlineShadowTest.DbiReportsCrossWaveFlatLdsRace"
                TARGET hip_consan_inline_shadow_test
                FILTER "${RJ_CONSAN_LDS_INLINE_SHADOW_FLAT_RACY_TEST}"
                TIMEOUT 30
                ENVIRONMENT
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=inline-shadow"
                    "RJ_CONSAN_FLAT_PROVENANCE=strict"
                    "RJ_CONSAN_MOI_REQUIRE_RECORDS=1"
                    "RJ_CONSAN_MOI_REQUIRE_DIAGNOSTICS=1"
                    "RJ_CONSAN_LOG=1"
                    "RJ_CONSAN_TEST_KERNEL_FILTER=padded_cross_wave_flat_lds_store"
                    "RJ_CONSAN_MAX_PATCHES=1"
                    "RJ_CONSAN_REQUIRE_PATCH=1"
        )
        rj_add_consan_hip_case(
                NAME "ConSanInlineShadowTest.RecordReplayReportsCrossWaveFlatLdsRace"
                TARGET hip_consan_inline_shadow_test
                FILTER "${RJ_CONSAN_LDS_INLINE_SHADOW_FLAT_RACY_TEST}"
                TIMEOUT 30
                ENVIRONMENT
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=record-replay"
                    "RJ_CONSAN_MOI_AUTO_REPORT_BUFFER_SIZE=1048576"
                    "RJ_CONSAN_FLAT_PROVENANCE=strict"
                    "RJ_CONSAN_MOI_DYNAMIC_ACCESS_RECORDS=1"
                    "RJ_CONSAN_MOI_REQUIRE_RECORDS=1"
                    "RJ_CONSAN_MOI_REQUIRE_REPLAY_CONFLICT=1"
                    "RJ_CONSAN_LOG=1"
                    "RJ_CONSAN_TEST_KERNEL_FILTER=padded_cross_wave_flat_lds_store"
                    "RJ_CONSAN_MAX_PATCHES=1"
                    "RJ_CONSAN_REQUIRE_PATCH=1"
        )
        rj_add_consan_hip_case(
                NAME "ConSanInlineShadowTest.DbiReportsCrossWaveRaceHwIdOwnerSource"
                TARGET hip_consan_inline_shadow_test
                FILTER "${RJ_CONSAN_LDS_INLINE_SHADOW_RACY_TEST}"
                TIMEOUT 30
                ENVIRONMENT
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=inline-shadow"
                    "RJ_CONSAN_MOI_REQUIRE_RECORDS=1"
                    "RJ_CONSAN_MOI_REQUIRE_DIAGNOSTICS=1"
                    "RJ_CONSAN_LOG=1"
                    "RJ_CONSAN_MOI_OWNER_SOURCE=hw_id"
                    "RJ_CONSAN_MAX_PATCHES=4"
                    "RJ_CONSAN_REQUIRE_PATCH=1"
        )
        rj_add_consan_hip_case(
                NAME "ConSanInlineShadowTest.DbiBarrierEpochOrdersCrossWaveAccesses"
                TARGET hip_consan_inline_shadow_test
                FILTER "${RJ_CONSAN_LDS_INLINE_SHADOW_BARRIER_TEST}"
                TIMEOUT 30
                ENVIRONMENT
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=inline-shadow"
                    "RJ_CONSAN_MOI_FORBID_DIAGNOSTICS=1"
                    "RJ_CONSAN_MOI_TRACK_BARRIERS=1"
                    "RJ_CONSAN_TEST_KERNEL_FILTER=padded_barrier_ordered_lds_store"
                    "RJ_CONSAN_LOG=1"
                    "RJ_CONSAN_MAX_PATCHES=8"
                    "RJ_CONSAN_REQUIRE_PATCH=1"
        )
        rj_add_consan_hip_case(
                NAME "ConSanInlineShadowTest.DbiVersionedPublicationVisibilityClean"
                TARGET hip_consan_inline_shadow_test
                FILTER "${RJ_CONSAN_LDS_INLINE_SHADOW_BARRIER_TEST}"
                TIMEOUT 60
                ENVIRONMENT
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=inline-shadow"
                    "RJ_CONSAN_MOI_REQUIRE_RECORDS=1"
                    "RJ_CONSAN_MOI_FORBID_DIAGNOSTICS=1"
                    "RJ_CONSAN_MOI_FORBID_OVERFLOW=1"
                    "RJ_CONSAN_MOI_TRACK_BARRIERS=1"
                    "RJ_CONSAN_LOG=1"
                    "RJ_CONSAN_TEST_KERNEL_FILTER=padded_barrier_ordered_lds_store"
                    "RJ_CONSAN_MAX_PATCHES=8"
                    "RJ_CONSAN_REQUIRE_PATCH=1"
                    "CONSAN_INLINE_VERSIONED_LAUNCHES=100"
        )
        set_tests_properties(
            "ConSanInlineShadowTest.DbiVersionedPublicationVisibilityClean"
            PROPERTIES RUN_SERIAL TRUE
        )
        rj_add_consan_hip_case(
                NAME "ConSanInlineShadowTest.BaselineFenceAndBarrierPreservesExactOutput"
                TARGET hip_consan_inline_shadow_test
                FILTER "${RJ_CONSAN_LDS_INLINE_SHADOW_FENCE_BARRIER_TEST}"
                TIMEOUT 30
        )
        rj_add_consan_hip_case(
                NAME "ConSanInlineShadowTest.DbiFenceAndBarrierPreservesExactOutput"
                TARGET hip_consan_inline_shadow_test
                FILTER "${RJ_CONSAN_LDS_INLINE_SHADOW_FENCE_BARRIER_TEST}"
                TIMEOUT 30
                ENVIRONMENT
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=inline-shadow"
                    "RJ_CONSAN_MOI_REQUIRE_RECORDS=1"
                    "RJ_CONSAN_MOI_FORBID_DIAGNOSTICS=1"
                    "RJ_CONSAN_MOI_TRACK_BARRIERS=1"
                    "RJ_CONSAN_TEST_KERNEL_FILTER=padded_fence_barrier_ordered_lds_accesses"
                    "RJ_CONSAN_LOG=1"
                    "RJ_CONSAN_MAX_PATCHES=8"
                    "RJ_CONSAN_REQUIRE_PATCH=1"
        )
        rj_add_consan_hip_case(
                NAME "ConSanInlineShadowTest.DbiMovedBarrierReportsCrossWaveRace"
                TARGET hip_consan_inline_shadow_test
                FILTER "${RJ_CONSAN_LDS_INLINE_SHADOW_BARRIER_TEST}"
                TIMEOUT 30
                ENVIRONMENT
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=inline-shadow"
                    "RJ_CONSAN_MOI_REQUIRE_RECORDS=1"
                    "RJ_CONSAN_MOI_REQUIRE_DIAGNOSTICS=1"
                    "RJ_CONSAN_MOI_TRACK_BARRIERS=1"
                    "RJ_CONSAN_FAULT_MOVE_BARRIER=1"
                    "RJ_CONSAN_FAULT_BARRIER_INDEX=2"
                    "RJ_CONSAN_TEST_KERNEL_FILTER=padded_barrier_ordered_lds_store"
                    "RJ_CONSAN_LOG=1"
                    "RJ_CONSAN_MAX_PATCHES=8"
                    "RJ_CONSAN_REQUIRE_PATCH=1"
        )
        rj_add_consan_hip_case(
                NAME "ConSanInlineShadowTest.DbiPrivateEpochBarrierOrdersCrossWaveAccesses"
                TARGET hip_consan_inline_shadow_test
                FILTER "${RJ_CONSAN_LDS_INLINE_SHADOW_BARRIER_TEST}"
                TIMEOUT 30
                ENVIRONMENT
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=inline-shadow"
                    "RJ_CONSAN_MOI_REQUIRE_RECORDS=1"
                    "RJ_CONSAN_MOI_FORBID_DIAGNOSTICS=1"
                    "RJ_CONSAN_MOI_TRACK_BARRIERS=1"
                    "RJ_CONSAN_LOG=1"
                    "RJ_CONSAN_TEST_FORCE_PRIVATE_EPOCH=1"
                    "RJ_CONSAN_TEST_KERNEL_FILTER=padded_barrier_ordered_lds_store"
                    "RJ_CONSAN_MAX_PATCHES=8"
                    "RJ_CONSAN_REQUIRE_PATCH=1"
        )
        rj_add_consan_hip_case(
                NAME "ConSanInlineShadowTest.DbiBarrierEpochOrdersCrossWaveAccessesHwIdOwnerSource"
                TARGET hip_consan_inline_shadow_test
                FILTER "${RJ_CONSAN_LDS_INLINE_SHADOW_BARRIER_TEST}"
                TIMEOUT 30
                ENVIRONMENT
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=inline-shadow"
                    "RJ_CONSAN_MOI_FORBID_DIAGNOSTICS=1"
                    "RJ_CONSAN_MOI_TRACK_BARRIERS=1"
                    "RJ_CONSAN_TEST_KERNEL_FILTER=padded_barrier_ordered_lds_store"
                    "RJ_CONSAN_LOG=1"
                    "RJ_CONSAN_MOI_OWNER_SOURCE=hw_id"
                    "RJ_CONSAN_MAX_PATCHES=8"
                    "RJ_CONSAN_REQUIRE_PATCH=1"
        )
        rj_add_consan_hip_case(
                NAME "ConSanInlineShadowTest.DbiAtomicHandoffSameAddressIsClean"
                TARGET hip_consan_inline_shadow_test
                FILTER "${RJ_CONSAN_LDS_INLINE_SHADOW_ATOMIC_SAME_ADDRESS_TEST}"
                TIMEOUT 30
                ENVIRONMENT
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=inline-shadow"
                    "RJ_CONSAN_MOI_FORBID_DIAGNOSTICS=1"
                    "RJ_CONSAN_MOI_TRACK_ATOMICS=1"
                    "RJ_CONSAN_TEST_KERNEL_FILTER=padded_atomic_ordered_lds_accesses"
                    "RJ_CONSAN_LOG=1"
                    "RJ_CONSAN_MAX_PATCHES=8"
                    "RJ_CONSAN_REQUIRE_PATCH=1"
        )
        rj_add_consan_hip_case(
                NAME "ConSanInlineShadowTest.DbiInjectedAtomicWrongAddressReports"
                TARGET hip_consan_inline_shadow_test
                FILTER "${RJ_CONSAN_LDS_INLINE_SHADOW_ATOMIC_SAME_ADDRESS_TEST}"
                TIMEOUT 30
                ENVIRONMENT
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=inline-shadow"
                    "RJ_CONSAN_MOI_REQUIRE_RECORDS=1"
                    "RJ_CONSAN_MOI_REQUIRE_DIAGNOSTICS=1"
                    "RJ_CONSAN_MOI_TRACK_ATOMICS=1"
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
                NAME "ConSanInlineShadowTest.DbiInjectedAtomicWeakenedOrderReports"
                TARGET hip_consan_inline_shadow_test
                FILTER "${RJ_CONSAN_LDS_INLINE_SHADOW_ATOMIC_SAME_ADDRESS_TEST}"
                TIMEOUT 30
                ENVIRONMENT
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=inline-shadow"
                    "RJ_CONSAN_MOI_REQUIRE_RECORDS=1"
                    "RJ_CONSAN_MOI_REQUIRE_DIAGNOSTICS=1"
                    "RJ_CONSAN_MOI_TRACK_ATOMICS=1"
                    "RJ_CONSAN_FAULT_ATOMIC_WEAKEN_ORDER=1"
                    "RJ_CONSAN_FAULT_ATOMIC_INDEX=1"
                    "RJ_CONSAN_FAULT_REQUIRE_EXACTLY_ONE=1"
                    "RJ_CONSAN_TEST_KERNEL_FILTER=padded_atomic_ordered_lds_accesses"
                    "RJ_CONSAN_LOG=1"
                    "RJ_CONSAN_MAX_PATCHES=8"
                    "RJ_CONSAN_REQUIRE_PATCH=1"
        )
        rj_add_consan_hip_case(
                NAME "ConSanInlineShadowTest.DbiInjectedAtomicWeakenedScopeReports"
                TARGET hip_consan_inline_shadow_test
                FILTER "${RJ_CONSAN_LDS_INLINE_SHADOW_ATOMIC_SAME_ADDRESS_TEST}"
                TIMEOUT 30
                ENVIRONMENT
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=inline-shadow"
                    "RJ_CONSAN_MOI_REQUIRE_RECORDS=1"
                    "RJ_CONSAN_MOI_REQUIRE_DIAGNOSTICS=1"
                    "RJ_CONSAN_MOI_TRACK_ATOMICS=1"
                    "RJ_CONSAN_FAULT_ATOMIC_WEAKEN_SCOPE=1"
                    "RJ_CONSAN_FAULT_ATOMIC_INDEX=1"
                    "RJ_CONSAN_FAULT_REQUIRE_EXACTLY_ONE=1"
                    "RJ_CONSAN_TEST_KERNEL_FILTER=padded_atomic_ordered_lds_accesses"
                    "RJ_CONSAN_LOG=1"
                    "RJ_CONSAN_MAX_PATCHES=8"
                    "RJ_CONSAN_REQUIRE_PATCH=1"
        )
        rj_add_consan_hip_case(
                NAME "ConSanInlineShadowTest.DbiAtomicHandoffSameAddressIsCleanHwIdOwnerSource"
                TARGET hip_consan_inline_shadow_test
                FILTER "${RJ_CONSAN_LDS_INLINE_SHADOW_ATOMIC_SAME_ADDRESS_TEST}"
                TIMEOUT 30
                ENVIRONMENT
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=inline-shadow"
                    "RJ_CONSAN_MOI_FORBID_DIAGNOSTICS=1"
                    "RJ_CONSAN_MOI_TRACK_ATOMICS=1"
                    "RJ_CONSAN_TEST_KERNEL_FILTER=padded_atomic_ordered_lds_accesses"
                    "RJ_CONSAN_LOG=1"
                    "RJ_CONSAN_MOI_OWNER_SOURCE=hw_id"
                    "RJ_CONSAN_MAX_PATCHES=8"
                    "RJ_CONSAN_REQUIRE_PATCH=1"
        )
        rj_add_consan_hip_case(
                NAME "ConSanInlineShadowTest.DbiAtomicHandoffWrongAddressReports"
                TARGET hip_consan_inline_shadow_test
                FILTER "${RJ_CONSAN_LDS_INLINE_SHADOW_ATOMIC_WRONG_ADDRESS_TEST}"
                TIMEOUT 30
                ENVIRONMENT
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=inline-shadow"
                    "RJ_CONSAN_MOI_REQUIRE_RECORDS=1"
                    "RJ_CONSAN_MOI_REQUIRE_DIAGNOSTICS=1"
                    "RJ_CONSAN_MOI_TRACK_ATOMICS=1"
                    "RJ_CONSAN_TEST_KERNEL_FILTER=padded_atomic_ordered_lds_accesses"
                    "RJ_CONSAN_LOG=1"
                    "RJ_CONSAN_MAX_PATCHES=8"
                    "RJ_CONSAN_REQUIRE_PATCH=1"
        )
        rj_add_consan_hip_case(
                NAME "ConSanInlineShadowTest.DbiAtomicHandoffWrongAddressReportsHwIdOwnerSource"
                TARGET hip_consan_inline_shadow_test
                FILTER "${RJ_CONSAN_LDS_INLINE_SHADOW_ATOMIC_WRONG_ADDRESS_TEST}"
                TIMEOUT 30
                ENVIRONMENT
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=inline-shadow"
                    "RJ_CONSAN_MOI_REQUIRE_RECORDS=1"
                    "RJ_CONSAN_MOI_REQUIRE_DIAGNOSTICS=1"
                    "RJ_CONSAN_MOI_TRACK_ATOMICS=1"
                    "RJ_CONSAN_TEST_KERNEL_FILTER=padded_atomic_ordered_lds_accesses"
                    "RJ_CONSAN_LOG=1"
                    "RJ_CONSAN_MOI_OWNER_SOURCE=hw_id"
                    "RJ_CONSAN_MAX_PATCHES=8"
                    "RJ_CONSAN_REQUIRE_PATCH=1"
        )
        rj_add_consan_hip_case(
                NAME "ConSanInlineShadowTest.DbiAtomicCollisionReplacementReports"
                TARGET hip_consan_inline_shadow_test
                FILTER "${RJ_CONSAN_LDS_INLINE_SHADOW_ATOMIC_COLLISION_TEST}"
                TIMEOUT 30
                PASS_REGEX "atomic_patched=[1-9][0-9]*"
                ENVIRONMENT
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=inline-shadow"
                    "RJ_CONSAN_MOI_REQUIRE_RECORDS=1"
                    "RJ_CONSAN_MOI_REQUIRE_DIAGNOSTICS=1"
                    "RJ_CONSAN_MOI_TRACK_ATOMICS=1"
                    "RJ_CONSAN_TEST_KERNEL_FILTER=padded_atomic_collision_replaces_release"
                    "RJ_CONSAN_LOG=1"
                    "RJ_CONSAN_MAX_PATCHES=8"
                    "RJ_CONSAN_REQUIRE_PATCH=1"
        )
        rj_add_consan_hip_case(
                NAME "ConSanInlineShadowTest.DbiAtomicCasFailureDoesNotPublish"
                TARGET hip_consan_inline_shadow_test
                FILTER "${RJ_CONSAN_LDS_INLINE_SHADOW_ATOMIC_CAS_TEST}"
                TIMEOUT 30
                ENVIRONMENT
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=inline-shadow"
                    "RJ_CONSAN_MOI_FORBID_DIAGNOSTICS=1"
                    "RJ_CONSAN_MOI_TRACK_ATOMICS=1"
                    "RJ_CONSAN_TEST_KERNEL_FILTER=padded_atomic_cas_failure_does_not_publish"
                    "RJ_CONSAN_LOG=1"
                    "RJ_CONSAN_MAX_PATCHES=8"
                    "RJ_CONSAN_REQUIRE_PATCH=1"
        )
        rj_add_consan_hip_case(
                NAME "ConSanInlineShadowTest.DbiInjectedAtomicCasWeakenedOrderReports"
                TARGET hip_consan_inline_shadow_test
                FILTER "${RJ_CONSAN_LDS_INLINE_SHADOW_ATOMIC_CAS_TEST}"
                TIMEOUT 30
                ENVIRONMENT
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=inline-shadow"
                    "RJ_CONSAN_MOI_REQUIRE_RECORDS=1"
                    "RJ_CONSAN_MOI_REQUIRE_DIAGNOSTICS=1"
                    "RJ_CONSAN_MOI_TRACK_ATOMICS=1"
                    "RJ_CONSAN_FAULT_ATOMIC_WEAKEN_ORDER=1"
                    "RJ_CONSAN_FAULT_ATOMIC_INDEX=1"
                    "RJ_CONSAN_FAULT_REQUIRE_EXACTLY_ONE=1"
                    "RJ_CONSAN_TEST_KERNEL_FILTER=padded_atomic_cas_failure_does_not_publish"
                    "RJ_CONSAN_LOG=1"
                    "RJ_CONSAN_MAX_PATCHES=8"
                    "RJ_CONSAN_REQUIRE_PATCH=1"
        )
        rj_add_consan_hip_case(
                NAME "ConSanMoiHipTest.DbiTwoWaveAccessesWriteDynamicRecords"
                TARGET hip_consan_moi_test
                FILTER "${RJ_CONSAN_LDS_MOI_RECORD_TEST}"
                TIMEOUT 30
                ENVIRONMENT
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=record-replay"
                    "RJ_CONSAN_TEST_KERNEL_FILTER=moi_two_wave_lds_accesses"
                    "RJ_CONSAN_LOG=1"
                    "RJ_CONSAN_MAX_PATCHES=2"
                    "RJ_CONSAN_REQUIRE_PATCH=1"
        )
        rj_add_consan_hip_case(
                NAME "ConSanMoiHipTest.DbiEquivalentAccessesInDifferentWorkgroupsDoNotConflict"
                TARGET hip_consan_moi_test
                FILTER "${RJ_CONSAN_LDS_MOI_WORKGROUP_RECORD_TEST}"
                TIMEOUT 30
                ENVIRONMENT
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=record-replay"
                    "RJ_CONSAN_TEST_KERNEL_FILTER=moi_equivalent_lds_accesses_in_different_workgroups"
                    "RJ_CONSAN_LOG=1"
                    "RJ_CONSAN_MAX_PATCHES=8"
                    "RJ_CONSAN_REQUIRE_PATCH=1"
        )
        rj_add_consan_hip_case(
                NAME "ConSanMoiHipTest.DbiEquivalentAccessesInDifferent2DWorkgroupsDoNotConflict"
                TARGET hip_consan_moi_test
                FILTER "${RJ_CONSAN_LDS_MOI_WORKGROUP_2D_RECORD_TEST}"
                TIMEOUT 30
                ENVIRONMENT
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=record-replay"
                    "RJ_CONSAN_TEST_KERNEL_FILTER=moi_equivalent_lds_accesses_in_different_2d_workgroups"
                    "RJ_CONSAN_LOG=1"
                    "RJ_CONSAN_MAX_PATCHES=10"
                    "RJ_CONSAN_REQUIRE_PATCH=1"
        )
        rj_add_consan_hip_case(
                NAME "ConSanMoiHipTest.DbiEquivalentAccessesInDifferent3DWorkgroupsDoNotConflict"
                TARGET hip_consan_moi_test
                FILTER "${RJ_CONSAN_LDS_MOI_WORKGROUP_3D_RECORD_TEST}"
                TIMEOUT 30
                ENVIRONMENT
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=record-replay"
                    "RJ_CONSAN_TEST_KERNEL_FILTER=moi_equivalent_lds_accesses_in_different_3d_workgroups"
                    "RJ_CONSAN_LOG=1"
                    "RJ_CONSAN_MAX_PATCHES=12"
                    "RJ_CONSAN_REQUIRE_PATCH=1"
        )
        rj_add_consan_hip_case(
                NAME "ConSanMoiHipTest.DbiSampledBackendWritesDirectEntries"
                TARGET hip_consan_moi_test
                FILTER "${RJ_CONSAN_LDS_MOI_SAMPLED_DIRECT_TEST}"
                TIMEOUT 30
                ENVIRONMENT
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=sampled"
                    "RJ_CONSAN_TEST_KERNEL_FILTER=moi_two_wave_lds_accesses"
                    "RJ_CONSAN_LOG=1"
                    "RJ_CONSAN_MAX_PATCHES=2"
                    "RJ_CONSAN_REQUIRE_PATCH=1"
        )
        rj_add_consan_hip_case(
                NAME "ConSanMoiHipTest.DbiRuntimeSampledBackendRetainsAllWavesForSelectedAddress"
                TARGET hip_consan_moi_test
                FILTER "ConSanMoiHipTest.RuntimeSampledBackendRetainsAllWavesForSelectedAddress"
                TIMEOUT 30
                ENVIRONMENT
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=sampled"
                    "RJ_CONSAN_MOI_RUNTIME_SAMPLE_STRIDE=2"
                    "RJ_CONSAN_MOI_RUNTIME_SAMPLE_OFFSET=0"
                    "RJ_CONSAN_TEST_KERNEL_FILTER=moi_two_wave_lds_accesses"
                    "RJ_CONSAN_LOG=1"
                    "RJ_CONSAN_MAX_PATCHES=2"
                    "RJ_CONSAN_REQUIRE_PATCH=1"
        )
        rj_add_consan_hip_case(
                NAME "ConSanMoiHipTest.DbiInKernelSampledCheckerReportsConflictImmediately"
                TARGET hip_consan_moi_test
                FILTER "ConSanMoiHipTest.InKernelSampledCheckerReportsConflictImmediately"
                TIMEOUT 30
                ENVIRONMENT
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=sampled"
                    "RJ_CONSAN_MOI_SAMPLED_CHECK=1"
                    "RJ_CONSAN_TEST_KERNEL_FILTER=moi_two_wave_lds_accesses"
                    "RJ_CONSAN_LOG=1"
                    "RJ_CONSAN_MAX_PATCHES=2"
                    "RJ_CONSAN_REQUIRE_PATCH=1"
        )
        rj_add_consan_hip_case(
                NAME "ConSanMoiHipTest.DbiAutoBufferReportsImmediateSampledConflict"
                TARGET hip_consan_moi_test
                FILTER "${RJ_CONSAN_LDS_MOI_AUTO_REPLAY_CONFLICT_TEST}"
                TIMEOUT 30
                ENVIRONMENT
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=sampled"
                    "RJ_CONSAN_MOI_SAMPLED_CHECK=1"
                    "RJ_CONSAN_MOI_REQUIRE_RECORDS=1"
                    "RJ_CONSAN_MOI_REQUIRE_DIAGNOSTICS=1"
                    "RJ_CONSAN_TEST_KERNEL_FILTER=moi_two_wave_lds_accesses"
                    "RJ_CONSAN_LOG=1"
                    "RJ_CONSAN_MAX_PATCHES=2"
                    "RJ_CONSAN_REQUIRE_PATCH=1"
        )
        set_tests_properties(
            "ConSanMoiHipTest.DbiAutoBufferReportsImmediateSampledConflict"
            PROPERTIES
                PASS_REGULAR_EXPRESSION "sampled_immediate_conflicts=[1-9]"
        )
        rj_add_consan_hip_case(
                NAME "ConSanMoiHipTest.DbiSampledAutoBufferReportsConflict"
                TARGET hip_consan_moi_test
                FILTER "${RJ_CONSAN_LDS_MOI_AUTO_REPLAY_CONFLICT_TEST}"
                TIMEOUT 30
                ENVIRONMENT
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=sampled"
                    "RJ_CONSAN_MOI_REQUIRE_RECORDS=1"
                    "RJ_CONSAN_MOI_REQUIRE_DIAGNOSTICS=1"
                    "RJ_CONSAN_TEST_KERNEL_FILTER=moi_two_wave_lds_accesses"
                    "RJ_CONSAN_LOG=1"
                    "RJ_CONSAN_MAX_PATCHES=2"
                    "RJ_CONSAN_REQUIRE_PATCH=1"
        )
        rj_add_consan_hip_case(
                NAME "ConSanMoiHipTest.DbiSampledCausalMultiWorkgroupClean"
                TARGET hip_consan_moi_test
                FILTER "ConSanMoiHipTest.DirectSampledRetainsExactMultiWorkgroupWindows"
                TIMEOUT 30
                ENVIRONMENT
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=sampled"
                    "RJ_CONSAN_MOI_REQUIRE_RECORDS=1"
                    "RJ_CONSAN_MOI_FORBID_DIAGNOSTICS=1"
                    "RJ_CONSAN_MOI_FORBID_OVERFLOW=1"
                    "RJ_CONSAN_TEST_KERNEL_FILTER=moi_equivalent_lds_accesses_in_different_workgroups"
                    "RJ_CONSAN_LOG=1"
                    "RJ_CONSAN_MAX_PATCHES=4"
                    "RJ_CONSAN_REQUIRE_PATCH=1"
        )
        set_tests_properties(
            "ConSanMoiHipTest.DbiSampledCausalMultiWorkgroupClean"
            PROPERTIES
                PASS_REGULAR_EXPRESSION
                    "SA2C sampled evidence ready_windows=2 live_entries=2 workgroups=0,1 dropped_windows=0"
        )
        rj_add_consan_hip_case(
                NAME "ConSanMoiHipTest.DbiSampledCausalFaultReportsConflict"
                TARGET hip_consan_moi_test
                FILTER "${RJ_CONSAN_LDS_MOI_AUTO_REPLAY_CONFLICT_TEST}"
                TIMEOUT 30
                ENVIRONMENT
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=sampled"
                    "RJ_CONSAN_MOI_REQUIRE_RECORDS=1"
                    "RJ_CONSAN_MOI_REQUIRE_DIAGNOSTICS=1"
                    "RJ_CONSAN_MOI_FORBID_OVERFLOW=1"
                    "RJ_CONSAN_TEST_KERNEL_FILTER=moi_two_wave_lds_accesses"
                    "RJ_CONSAN_LOG=1"
                    "RJ_CONSAN_MAX_PATCHES=2"
                    "RJ_CONSAN_REQUIRE_PATCH=1"
        )
        set_tests_properties(
            "ConSanMoiHipTest.DbiSampledCausalFaultReportsConflict"
            PROPERTIES
                PASS_REGULAR_EXPRESSION
                    "visible_sampled=[1-9].*sampled_conflicts=[1-9]"
        )
        rj_add_consan_hip_case(
                NAME "ConSanMoiHipTest.DbiDynamicAccessRecordsAppendActiveLanes"
                TARGET hip_consan_moi_test
                FILTER "${RJ_CONSAN_LDS_MOI_DYNAMIC_ACCESS_RECORD_TEST}"
                TIMEOUT 30
                ENVIRONMENT
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=record-replay"
                    "RJ_CONSAN_MOI_DYNAMIC_ACCESS_RECORDS=1"
                    "RJ_CONSAN_TEST_KERNEL_FILTER=moi_two_wave_lds_accesses"
                    "RJ_CONSAN_LOG=1"
                    "RJ_CONSAN_MAX_PATCHES=2"
                    "RJ_CONSAN_REQUIRE_PATCH=1"
        )
        rj_add_consan_hip_case(
                NAME "ConSanMoiHipTest.DbiAutoReportBufferReplaysConflict"
                TARGET hip_consan_moi_test
                FILTER "${RJ_CONSAN_LDS_MOI_AUTO_REPLAY_CONFLICT_TEST}"
                TIMEOUT 30
                ENVIRONMENT
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=record-replay"
                    "RJ_CONSAN_MOI_REQUIRE_RECORDS=1"
                    "RJ_CONSAN_MOI_REQUIRE_REPLAY_CONFLICT=1"
                    "RJ_CONSAN_TEST_KERNEL_FILTER=moi_two_wave_lds_accesses"
                    "RJ_CONSAN_LOG=1"
                    "RJ_CONSAN_MAX_PATCHES=2"
                    "RJ_CONSAN_REQUIRE_PATCH=1"
        )
        rj_add_consan_hip_case(
                NAME "ConSanMoiHipTest.DbiAutoReportRetainsRepeatedDispatches"
                TARGET hip_consan_moi_test
                FILTER "${RJ_CONSAN_LDS_MOI_AUTO_REPEATED_DISPATCH_TEST}"
                TIMEOUT 30
                ENVIRONMENT
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=record-replay"
                    "RJ_CONSAN_MOI_REQUIRE_RECORDS=1"
                    "RJ_CONSAN_MOI_FORBID_OVERFLOW=1"
                    "RJ_CONSAN_TEST_KERNEL_FILTER=moi_two_wave_lds_accesses"
                    "RJ_CONSAN_LOG=1"
                    "RJ_CONSAN_MAX_PATCHES=2"
                    "RJ_CONSAN_REQUIRE_PATCH=1"
        )
        set_tests_properties(
            "ConSanMoiHipTest.DbiAutoReportRetainsRepeatedDispatches"
            PROPERTIES
                PASS_REGULAR_EXPRESSION
                    "committed_records=8.*dispatch_tokens=4/2048.*record_replay_flags=0x0"
                FAIL_REGULAR_EXPRESSION "record_replay_bank_saturated=true"
        )
        rj_add_consan_hip_case(
                NAME "ConSanMoiHipTest.DbiOverflowIsVisible"
                TARGET hip_consan_moi_test
                FILTER "${RJ_CONSAN_LDS_MOI_AUTO_REPLAY_CONFLICT_TEST}"
                TIMEOUT 30
                ENVIRONMENT
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=record-replay"
                    # Header plus one ABI-v12 access record. Keep this case at
                    # exactly one slot so the second active lane overflows.
                    "RJ_CONSAN_MOI_AUTO_REPORT_BUFFER_SIZE=264"
                    "RJ_CONSAN_MOI_DYNAMIC_ACCESS_RECORDS=1"
                    "RJ_CONSAN_TEST_KERNEL_FILTER=moi_two_wave_lds_accesses"
                    "RJ_CONSAN_LOG=1"
                    "RJ_CONSAN_MAX_PATCHES=2"
                    "RJ_CONSAN_REQUIRE_PATCH=1"
        )
        set_tests_properties(
            "ConSanMoiHipTest.DbiOverflowIsVisible"
            PROPERTIES
                PASS_REGULAR_EXPRESSION "ConSan MOI report overflow: dropped"
        )
        rj_add_consan_hip_case(
                NAME "ConSanMoiHipTest.DbiOneRecordCapacityLimitsAccessPatches"
                TARGET hip_consan_moi_test
                FILTER "${RJ_CONSAN_LDS_MOI_LIMITED_RECORD_TEST}"
                TIMEOUT 30
                ENVIRONMENT
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=record-replay"
                    "RJ_CONSAN_TEST_KERNEL_FILTER=moi_two_wave_lds_accesses"
                    "RJ_CONSAN_LOG=1"
                    "RJ_CONSAN_MAX_PATCHES=2"
                    "RJ_CONSAN_REQUIRE_PATCH=1"
        )
        rj_add_consan_hip_case(
                NAME "ConSanMoiHipTest.DbiBarrierRecordOrdersTwoWaveAccesses"
                TARGET hip_consan_moi_test
                FILTER "${RJ_CONSAN_LDS_MOI_BARRIER_RECORD_TEST}"
                TIMEOUT 30
                ENVIRONMENT
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=record-replay"
                    "RJ_CONSAN_MOI_INIT_OWNER_EPOCH=0"
                    "RJ_CONSAN_MOI_TRACK_BARRIERS=1"
                    "RJ_CONSAN_TEST_KERNEL_FILTER=moi_two_wave_lds_accesses_with_barrier"
                    "RJ_CONSAN_LOG=1"
                    "RJ_CONSAN_MAX_PATCHES=8"
                    "RJ_CONSAN_REQUIRE_PATCH=1"
        )
        rj_add_consan_hip_case(
                NAME "ConSanMoiHipTest.DbiFlatAtomicWritesDynamicRecord"
                TARGET hip_consan_moi_test
                FILTER "${RJ_CONSAN_LDS_MOI_ATOMIC_RECORD_TEST}"
                TIMEOUT 30
                ENVIRONMENT
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=record-replay"
                    "RJ_CONSAN_MOI_TRACK_ATOMICS=1"
                    "RJ_CONSAN_TEST_KERNEL_FILTER=moi_flat_atomic_add_record"
                    "RJ_CONSAN_LOG=1"
                    "RJ_CONSAN_MAX_PATCHES=1"
                    "RJ_CONSAN_REQUIRE_PATCH=1"
        )
        rj_add_consan_hip_case(
                NAME "ConSanMoiHipTest.DbiAtomicHandoffOrdersTwoWaveAccesses"
                TARGET hip_consan_moi_test
                FILTER "${RJ_CONSAN_LDS_MOI_ATOMIC_HANDOFF_TEST}"
                TIMEOUT 30
                ENVIRONMENT
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=record-replay"
                    "RJ_CONSAN_MOI_TRACK_ATOMICS=1"
                    "RJ_CONSAN_TEST_KERNEL_FILTER=moi_atomic_handoff_lds_accesses"
                    "RJ_CONSAN_LOG=1"
                    "RJ_CONSAN_MAX_PATCHES=8"
                    "RJ_CONSAN_REQUIRE_PATCH=1"
        )
        # RR3C5C's retained harness runs these rows under the global GPU
        # lock and applies stricter log accounting. Keep the underlying
        # controls independently build-wired so their exact GTest names
        # cannot silently drift.
        rj_add_consan_hip_case(
                NAME "ConSanMoiHipTest.DbiRecordReplayFenceHandoffSameAddressIsClean"
                TARGET hip_consan_inline_shadow_test
                FILTER "${RJ_CONSAN_LDS_INLINE_SHADOW_ATOMIC_SAME_ADDRESS_TEST}"
                TIMEOUT 60
                ENVIRONMENT
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=record-replay"
                    "RJ_CONSAN_MOI_AUTO_REPORT_BUFFER_SIZE=1048576"
                    "RJ_CONSAN_MOI_TRACK_ATOMICS=1"
                    "RJ_CONSAN_MOI_DYNAMIC_ACCESS_RECORDS=1"
                    "RJ_CONSAN_TMP_VGPR=8"
                    "RJ_CONSAN_TEST_KERNEL_FILTER=padded_atomic_ordered_lds_accesses"
                    "RJ_CONSAN_MAX_PATCHES=16"
                    "RJ_CONSAN_REQUIRE_PATCH=1"
                    "RJ_CONSAN_MOI_REQUIRE_RECORDS=1"
                    "RJ_CONSAN_MOI_FORBID_DIAGNOSTICS=1"
                    "RJ_CONSAN_MOI_FORBID_OVERFLOW=1"
                    "RJ_CONSAN_LOG=2"
        )
        rj_add_consan_hip_case(
                NAME "ConSanMoiHipTest.DbiRecordReplayFenceHandoffWrongAddressConflicts"
                TARGET hip_consan_inline_shadow_test
                FILTER "${RJ_CONSAN_LDS_INLINE_SHADOW_ATOMIC_WRONG_ADDRESS_TEST}"
                TIMEOUT 60
                ENVIRONMENT
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=record-replay"
                    "RJ_CONSAN_MOI_AUTO_REPORT_BUFFER_SIZE=1048576"
                    "RJ_CONSAN_MOI_TRACK_ATOMICS=1"
                    "RJ_CONSAN_MOI_DYNAMIC_ACCESS_RECORDS=1"
                    "RJ_CONSAN_TMP_VGPR=8"
                    "RJ_CONSAN_TEST_KERNEL_FILTER=padded_atomic_ordered_lds_accesses"
                    "RJ_CONSAN_MAX_PATCHES=16"
                    "RJ_CONSAN_REQUIRE_PATCH=1"
                    "RJ_CONSAN_MOI_REQUIRE_RECORDS=1"
                    "RJ_CONSAN_MOI_REQUIRE_REPLAY_CONFLICT=1"
                    "RJ_CONSAN_MOI_FORBID_OVERFLOW=1"
                    "RJ_CONSAN_LOG=2"
        )
        set_tests_properties(
            "ConSanMoiHipTest.DbiAtomicHandoffOrdersTwoWaveAccesses"
            PROPERTIES
                PASS_REGULAR_EXPRESSION
                    "RR3C5C fence evidence release_token=0x[0-9a-fA-F]*[1-9a-fA-F][0-9a-fA-F]* acquire_token=0x[0-9a-fA-F]*[1-9a-fA-F][0-9a-fA-F]*"
        )
        rj_add_consan_hip_case(
                NAME "ConSanMoiHipTest.DbiAtomicHandoffAutoReportReplaysFences"
                TARGET hip_consan_moi_test
                FILTER "ConSanMoiHipTest.AtomicHandoffRunsWithAutoReportBuffer"
                TIMEOUT 30
                ENVIRONMENT
                    "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_dbi_hooks>"
                    "RJ_CONSAN_MODE=record-replay"
                    "RJ_CONSAN_MOI_AUTO_REPORT_BUFFER_SIZE=1048576"
                    "RJ_CONSAN_MOI_TRACK_ATOMICS=1"
                    "RJ_CONSAN_MOI_DYNAMIC_ACCESS_RECORDS=1"
                    "RJ_CONSAN_MOI_REQUIRE_RECORDS=1"
                    "RJ_CONSAN_MOI_FORBID_DIAGNOSTICS=1"
                    "RJ_CONSAN_MOI_FORBID_OVERFLOW=1"
                    "RJ_CONSAN_TEST_KERNEL_FILTER=moi_atomic_handoff_lds_accesses"
                    "RJ_CONSAN_LOG=1"
                    "RJ_CONSAN_MAX_PATCHES=8"
                    "RJ_CONSAN_REQUIRE_PATCH=1"
        )
        set_tests_properties(
            "ConSanMoiHipTest.DbiAtomicHandoffAutoReportReplaysFences"
            PROPERTIES
                PASS_REGULAR_EXPRESSION
                    "processed_fences=[1-9][0-9]*.*unsupported_fences=0.*conflict=false"
        )
    endif()
endfunction()
