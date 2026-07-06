# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# --- Race detection HIP tests ---
# Uses a lightweight config (race_test_config.json) with 1 XCD and 4 CUs
# instead of the full MI350X topology (8 XCDs, 256 CUs). Each race test
# spawns a complete simulation engine, and with ctest -j the full config
# causes OOM kills at moderate parallelism (~15+ concurrent tests). The
# race detector only needs a few CUs to exercise cross-wave and
# cross-workgroup synchronization checking.
rj_add_hip_test(hip_race_tests SOURCE hip_race_tests.hip)
set(RJ_RACE_CONFIG ${CMAKE_CURRENT_SOURCE_DIR}/race_test_config.json)
if(RJ_INSTALL_TESTS)
    install(
        FILES ${RJ_RACE_CONFIG}
        DESTINATION ${CMAKE_INSTALL_DATADIR}/rocjitsu/configs
    )
endif()

function(rj_add_race_test_case TEST_CASE)
    set(SINK_DIR ${CMAKE_CURRENT_BINARY_DIR}/racetest_${TEST_CASE})
    set(RUNTIME_DIR ${SINK_DIR}/runtime)
    file(MAKE_DIRECTORY ${SINK_DIR})
    file(MAKE_DIRECTORY ${RUNTIME_DIR})
    rj_add_hip_test_case(
        hip_race_tests "RaceTest.${TEST_CASE}"
        CONFIG ${RJ_RACE_CONFIG}
        TIMEOUT 30
        ENVIRONMENT
            "ROCJITSU_RUNTIME_DIR=${RUNTIME_DIR}"
            "RJ_RACE=1"
            "RJ_SINKS=file"
            "RJ_SINK_DIR=${SINK_DIR}"
    )
endfunction()

rj_add_race_test_case(vgpr_waitcnt)
rj_add_race_test_case(vgpr_waitcnt_race)
rj_add_race_test_case(sgpr_waitcnt)
rj_add_race_test_case(sgpr_waitcnt_race)
rj_add_race_test_case(lds_cross_wave)
rj_add_race_test_case(lds_cross_wave_race)
rj_add_race_test_case(global_to_lds_buffer)
rj_add_race_test_case(global_to_lds_buffer_race)
rj_add_race_test_case(partial_vmcnt_race)
rj_add_race_test_case(multiple_race)
rj_add_race_test_case(multi_kernel)
rj_add_race_test_case(multi_kernel_race)
rj_add_race_test_case(multi_workgroup)
rj_add_race_test_case(multi_workgroup_race)
rj_add_race_test_case(mixed_counters_race)
rj_add_race_test_case(stress_wavefront_reuse)
rj_add_race_test_case(stress_wavefront_reuse_race)
rj_add_race_test_case(exec_mask)
rj_add_race_test_case(exec_mask_race)
rj_add_race_test_case(exec_mask_same_wave)
rj_add_race_test_case(exec_mask_cross_wave_race)
rj_add_race_test_case(exec_mask_cross_wave)
rj_add_race_test_case(lds_transpose)
rj_add_race_test_case(lds_transpose_race)
rj_add_race_test_case(dual_offset_lds)
rj_add_race_test_case(dual_offset_lds_race)
rj_add_race_test_case(scratch_vmcnt)
rj_add_race_test_case(scratch_vmcnt_race)
rj_add_race_test_case(f64_vgpr_safe)
rj_add_race_test_case(f64_vgpr_race)
