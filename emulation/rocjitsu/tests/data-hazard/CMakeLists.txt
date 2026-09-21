# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# --- Data hazard HIP tests ---
# Uses the same lightweight topology rationale as the race detector: each case
# spawns a complete simulation engine, so data_hazard_test_config.json
# describes 1 XCD and 4 CUs rather than the full MI350X topology.
rj_add_hip_test(hip_data_hazard_tests_gfx950 ARCH gfx950 SOURCE hip_data_hazard_gfx950_test.hip)

set(RJ_DATA_HAZARD_GFX950_CONFIG
    ${CMAKE_CURRENT_SOURCE_DIR}/data_hazard_test_config.json
)
if(RJ_INSTALL_TESTS)
    install(
        FILES ${RJ_DATA_HAZARD_GFX950_CONFIG}
        DESTINATION ${CMAKE_INSTALL_DATADIR}/rocjitsu/configs
    )
endif()

# Derive a per-case config from the base topology config. The emulator reads
# plugin and sink selection exclusively from the config file, so the data hazard
# plugin and a file sink are injected here with this case's private output
# directory baked in. The plugin streams every finding to
# <dir>/data_hazard.log as it is made, which is what the test binary reads
# (locating it through RJ_SINK_DIR); the JSON report is only written when the
# run ends.
function(rj_write_data_hazard_config BASE_CONFIG SINK_DIR OUT_CONFIG)
    file(READ ${BASE_CONFIG} _hazard_json)
    string(
        JSON _hazard_json
        SET "${_hazard_json}"
        "plugins"
        "{\"data_hazard\": {}}"
    )
    string(
        JSON _hazard_json
        SET "${_hazard_json}"
        "sinks"
        "{\"types\": [\"file\"], \"dir\": \"${SINK_DIR}\"}"
    )
    file(WRITE ${OUT_CONFIG} "${_hazard_json}")
    # Re-run configuration if the base topology config changes.
    set_property(
        DIRECTORY
        APPEND
        PROPERTY CMAKE_CONFIGURE_DEPENDS ${BASE_CONFIG}
    )
endfunction()

function(rj_add_data_hazard_test_case TEST_CASE)
    set(SINK_DIR ${CMAKE_CURRENT_BINARY_DIR}/hazardtest_gfx950_${TEST_CASE})
    set(RUNTIME_DIR ${SINK_DIR}/runtime)
    set(INSTALLED_CONFIG_NAME data_hazard_gfx950_${TEST_CASE}.json)
    set(INSTALLED_CONFIG
        ${CMAKE_CURRENT_BINARY_DIR}/install-configs/${INSTALLED_CONFIG_NAME}
    )
    file(MAKE_DIRECTORY ${SINK_DIR})
    file(MAKE_DIRECTORY ${RUNTIME_DIR})
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/install-configs)
    rj_write_data_hazard_config(
        ${RJ_DATA_HAZARD_GFX950_CONFIG}
        ${SINK_DIR}
        ${SINK_DIR}/config.json
    )
    # The installed copy resolves its sink directory relative to the runtime
    # directory the installed test is launched with.
    rj_write_data_hazard_config(
        ${RJ_DATA_HAZARD_GFX950_CONFIG}
        runtime/hazardtest_gfx950_${TEST_CASE}
        ${INSTALLED_CONFIG}
    )
    if(RJ_INSTALL_TESTS)
        install(
            FILES ${INSTALLED_CONFIG}
            DESTINATION ${CMAKE_INSTALL_DATADIR}/rocjitsu/configs/tests
        )
    endif()
    rj_add_hip_test_case(
        hip_data_hazard_tests_gfx950 "Gfx950DataHazardTest.${TEST_CASE}"
        CONFIG ${SINK_DIR}/config.json
        INSTALL_CONFIG "\${RJ_INSTALLED_CONFIG_DIR}/tests/${INSTALLED_CONFIG_NAME}"
        NAME "DataHazardTest.gfx950_${TEST_CASE}"
        TIMEOUT 30
        ENVIRONMENT
            "ROCJITSU_RUNTIME_DIR=${RUNTIME_DIR}"
            "RJ_SINK_DIR=${SINK_DIR}"
    )
endfunction()

rj_add_data_hazard_test_case(vgpr_waitcnt)
rj_add_data_hazard_test_case(vgpr_waitcnt_hazard)
