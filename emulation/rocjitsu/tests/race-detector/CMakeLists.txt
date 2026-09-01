# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# --- Race detection HIP tests ---
# Uses a lightweight config (race_test_config.json) with 1 XCD and 4 CUs
# instead of the full MI350X topology (8 XCDs, 288 physical / 256 active CUs). Each race test
# spawns a complete simulation engine, and with ctest -j the full config
# causes OOM kills at moderate parallelism (~15+ concurrent tests). The
# race detector only needs a few CUs to exercise cross-wave and
# cross-workgroup synchronization checking.
# The gfx950 race detector cases include CDNA-specific inline assembly
# sequences for LDS and waitcnt coverage. Keep that binary on gfx950 even when
# generic HIP smoke tests are overridden for local validation on another ISA.
set(RJ_RACE_TEST_SUPPORT_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/race_log_expectation.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/race_test_support.hpp
)
rj_add_hip_test(
    hip_race_tests_gfx950
    ARCH gfx950
    SOURCE hip_race_gfx950_test.hip
    DEPENDENCIES ${RJ_RACE_TEST_SUPPORT_HEADERS}
)
rj_add_hip_test(
    hip_race_tests_gfx1151
    ARCH gfx1151
    SOURCE hip_race_gfx1151_test.hip
    DEPENDENCIES ${RJ_RACE_TEST_SUPPORT_HEADERS}
)
set(RJ_RACE_GFX950_CONFIG ${CMAKE_CURRENT_SOURCE_DIR}/race_test_config.json)
if(RJ_INSTALL_TESTS)
    install(
        FILES ${RJ_RACE_GFX950_CONFIG}
        DESTINATION ${CMAKE_INSTALL_DATADIR}/rocjitsu/configs
    )
endif()
set(RJ_RACE_GFX1151_CONFIG ${CMAKE_SOURCE_DIR}/configs/gfx1151.json)

# Derive a per-case race-detector config from a base topology config. The
# emulator reads plugin and sink selection exclusively from the config file
# (there is no RJ_RACE / RJ_SINKS environment fallback), so inject the race
# plugin and a file sink here, baking in this case's private output directory
# so each case writes its own race.log. The test binary then locates that log
# via the RJ_SINK_DIR environment variable set on the test.
function(rj_write_race_config BASE_CONFIG SINK_DIR OUT_CONFIG)
    file(READ ${BASE_CONFIG} _race_json)
    string(JSON _race_json SET "${_race_json}" "plugins" "{\"race\": {}}")
    string(
        JSON _race_json
        SET "${_race_json}"
        "sinks"
        "{\"types\": [\"file\"], \"dir\": \"${SINK_DIR}\"}"
    )
    file(WRITE ${OUT_CONFIG} "${_race_json}")
    # Re-run configuration if the base topology config changes.
    set_property(
        DIRECTORY
        APPEND
        PROPERTY CMAKE_CONFIGURE_DEPENDS ${BASE_CONFIG}
    )
endfunction()

function(rj_add_gfx950_race_test_case TEST_CASE)
    set(SINK_DIR ${CMAKE_CURRENT_BINARY_DIR}/racetest_gfx950_${TEST_CASE})
    set(RUNTIME_DIR ${SINK_DIR}/runtime)
    set(INSTALLED_CONFIG_NAME race_gfx950_${TEST_CASE}.json)
    set(INSTALLED_CONFIG
        ${CMAKE_CURRENT_BINARY_DIR}/install-configs/${INSTALLED_CONFIG_NAME}
    )
    file(MAKE_DIRECTORY ${SINK_DIR})
    file(MAKE_DIRECTORY ${RUNTIME_DIR})
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/install-configs)
    rj_write_race_config(${RJ_RACE_GFX950_CONFIG} ${SINK_DIR} ${SINK_DIR}/config.json)
    rj_write_race_config(
        ${RJ_RACE_GFX950_CONFIG}
        runtime/racetest_gfx950_${TEST_CASE}
        ${INSTALLED_CONFIG}
    )
    if(RJ_INSTALL_TESTS)
        install(
            FILES ${INSTALLED_CONFIG}
            DESTINATION ${CMAKE_INSTALL_DATADIR}/rocjitsu/configs/tests
        )
    endif()
    rj_add_hip_test_case(
        hip_race_tests_gfx950 "Gfx950RaceTest.${TEST_CASE}"
        CONFIG ${SINK_DIR}/config.json
        INSTALL_CONFIG "\${RJ_INSTALLED_CONFIG_DIR}/tests/${INSTALLED_CONFIG_NAME}"
        NAME "RaceTest.gfx950_${TEST_CASE}"
        TIMEOUT 30
        ENVIRONMENT
            "ROCJITSU_RUNTIME_DIR=${RUNTIME_DIR}"
            "RJ_SINK_DIR=${SINK_DIR}"
    )
endfunction()

function(rj_add_gfx1151_race_test_case TEST_CASE)
    set(SINK_DIR ${CMAKE_CURRENT_BINARY_DIR}/racetest_gfx1151_${TEST_CASE})
    set(RUNTIME_DIR ${SINK_DIR}/runtime)
    set(INSTALLED_CONFIG_NAME race_gfx1151_${TEST_CASE}.json)
    set(INSTALLED_CONFIG
        ${CMAKE_CURRENT_BINARY_DIR}/install-configs/${INSTALLED_CONFIG_NAME}
    )
    file(MAKE_DIRECTORY ${SINK_DIR})
    file(MAKE_DIRECTORY ${RUNTIME_DIR})
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/install-configs)
    rj_write_race_config(${RJ_RACE_GFX1151_CONFIG} ${SINK_DIR} ${SINK_DIR}/config.json)
    rj_write_race_config(
        ${RJ_RACE_GFX1151_CONFIG}
        runtime/racetest_gfx1151_${TEST_CASE}
        ${INSTALLED_CONFIG}
    )
    if(RJ_INSTALL_TESTS)
        install(
            FILES ${INSTALLED_CONFIG}
            DESTINATION ${CMAKE_INSTALL_DATADIR}/rocjitsu/configs/tests
        )
    endif()
    rj_add_hip_test_case(
        hip_race_tests_gfx1151 "Gfx1151RaceTest.${TEST_CASE}"
        CONFIG ${SINK_DIR}/config.json
        INSTALL_CONFIG "\${RJ_INSTALLED_CONFIG_DIR}/tests/${INSTALLED_CONFIG_NAME}"
        NAME "RaceTest.gfx1151_${TEST_CASE}"
        TIMEOUT 30
        ENVIRONMENT
            "ROCJITSU_RUNTIME_DIR=${RUNTIME_DIR}"
            "RJ_SINK_DIR=${SINK_DIR}"
    )
endfunction()

rj_add_gfx950_race_test_case(vgpr_waitcnt)
rj_add_gfx950_race_test_case(vgpr_waitcnt_race)
rj_add_gfx950_race_test_case(vmcnt_no_wait_race)
rj_add_gfx950_race_test_case(lgkmcnt_no_wait_race)
rj_add_gfx950_race_test_case(vmcnt_capacity_boundary)
rj_add_gfx950_race_test_case(vmcnt_below_capacity_race)
rj_add_gfx950_race_test_case(lgkmcnt_capacity_boundary)
rj_add_gfx950_race_test_case(lgkmcnt_below_capacity_race)
rj_add_gfx950_race_test_case(sgpr_waitcnt)
rj_add_gfx950_race_test_case(sgpr_waitcnt_race)
rj_add_gfx950_race_test_case(lds_cross_wave)
rj_add_gfx950_race_test_case(lds_cross_wave_race)
rj_add_gfx950_race_test_case(lds_same_wave_order)
rj_add_gfx950_race_test_case(global_to_lds_buffer)
rj_add_gfx950_race_test_case(global_to_lds_buffer_race)
rj_add_gfx950_race_test_case(partial_vmcnt_race)
rj_add_gfx950_race_test_case(multiple_race)
rj_add_gfx950_race_test_case(multi_kernel)
rj_add_gfx950_race_test_case(multi_kernel_race)
rj_add_gfx950_race_test_case(multi_workgroup)
rj_add_gfx950_race_test_case(multi_workgroup_race)
rj_add_gfx950_race_test_case(mixed_counters_race)
rj_add_gfx950_race_test_case(stress_wavefront_reuse)
rj_add_gfx950_race_test_case(stress_wavefront_reuse_race)
rj_add_gfx950_race_test_case(exec_mask)
rj_add_gfx950_race_test_case(exec_mask_race)
rj_add_gfx950_race_test_case(exec_mask_same_wave)
rj_add_gfx950_race_test_case(exec_mask_cross_wave_race)
rj_add_gfx950_race_test_case(exec_mask_cross_wave)
rj_add_gfx950_race_test_case(lds_transpose)
rj_add_gfx950_race_test_case(lds_transpose_race)
rj_add_gfx950_race_test_case(dual_offset_lds)
rj_add_gfx950_race_test_case(dual_offset_lds_race)
rj_add_gfx950_race_test_case(scratch_vmcnt)
rj_add_gfx950_race_test_case(scratch_vmcnt_race)
rj_add_gfx950_race_test_case(waw_global_load_then_alu)
rj_add_gfx950_race_test_case(waw_global_load_then_alu_safe)
rj_add_gfx950_race_test_case(waw_global_load_then_mfma)
rj_add_gfx950_race_test_case(f64_vgpr_safe)
rj_add_gfx950_race_test_case(f64_vgpr_race)
rj_add_gfx950_race_test_case(f64_vgpr_waw)

rj_add_gfx1151_race_test_case(vgpr_waitcnt)
rj_add_gfx1151_race_test_case(vgpr_waitcnt_race)
rj_add_gfx1151_race_test_case(standalone_vmcnt)
rj_add_gfx1151_race_test_case(standalone_vmcnt_does_not_wait_lgkm)
rj_add_gfx1151_race_test_case(standalone_lgkmcnt)
rj_add_gfx1151_race_test_case(standalone_lgkmcnt_does_not_wait_vmcnt)
rj_add_gfx1151_race_test_case(d16_global_load_merge_no_race)
rj_add_gfx1151_race_test_case(d16_global_load_hi_merge_no_race)
rj_add_gfx1151_race_test_case(d16_global_load_missing_waitcnt_race)
rj_add_gfx1151_race_test_case(lds_wave32_cross_wave)
rj_add_gfx1151_race_test_case(lds_wave32_cross_wave_race)
rj_add_gfx1151_race_test_case(waw_global_load_then_wmma)
