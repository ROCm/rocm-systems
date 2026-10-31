# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

add_executable(
    consan_input_test
    input_test.cpp
    prepare_test.cpp
    row_layout_test.cpp
    callable_probe_test.cpp
)
target_link_libraries(
    consan_input_test
    PRIVATE
        rocjitsu_consan_compiler
        rocjitsu_amdgpu_isa_model_registry
        GTest::gtest_main
)
rj_configure_target(consan_input_test PRIVATE_INCLUDES WARNINGS)

if(AMDCXX AND EXISTS "${ROCM_PATH}/lib/llvm/bin/ld.lld")
    set(_fixture_dir ${CMAKE_CURRENT_BINARY_DIR}/fixtures)
    file(MAKE_DIRECTORY ${_fixture_dir})
    set(_fixtures)
    set(_names
        race
        disjoint
        read_read
        collision
        single_lane
        alias
        barrier
        immediate
        wide
        four_ranges
        exhausted
        empty
        branched
    )
    set(_case 0)
    foreach(_name IN LISTS _names)
        math(EXPR _case "${_case} + 1")
        set(_object ${_fixture_dir}/${_name}.o)
        set(_image ${_fixture_dir}/${_name}.hsaco)
        add_custom_command(
            OUTPUT ${_object}
            COMMAND
                ${CMAKE_CXX_COMPILER_LAUNCHER} ${AMDCXX} -x assembler-with-cpp
                -target amdgcn-amd-amdhsa -mcpu=gfx1100 -mcode-object-version=5
                -DCONSAN_CASE=${_case} -c
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
        list(APPEND _fixtures ${_image})
    endforeach()
    add_custom_command(
        OUTPUT ${_fixture_dir}/renamed.o
        COMMAND
            ${CMAKE_CXX_COMPILER_LAUNCHER} ${AMDCXX} -x assembler-with-cpp
            -target amdgcn-amd-amdhsa -mcpu=gfx1100 -mcode-object-version=5
            -DCONSAN_CASE=1 -DCONSAN_KERNEL=renamed_input -c
            ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/native_lds.s -o
            ${_fixture_dir}/renamed.o
        DEPENDS fixtures/native_lds.s ${AMDCXX}
        VERBATIM
    )
    add_custom_command(
        OUTPUT ${_fixture_dir}/renamed.hsaco
        COMMAND
            ${ROCM_PATH}/lib/llvm/bin/ld.lld -shared ${_fixture_dir}/renamed.o
            -o ${_fixture_dir}/renamed.hsaco
        DEPENDS ${_fixture_dir}/renamed.o
        VERBATIM
    )
    list(APPEND _fixtures ${_fixture_dir}/renamed.hsaco)
    include(target_fixtures.cmake)
    target_sources(consan_input_test PRIVATE target_test.cpp)
    add_custom_target(consan_original_fixtures DEPENDS ${_fixtures})
    add_dependencies(consan_input_test consan_original_fixtures)
    add_dependencies(consan_input_test consan_target_fixtures)
    target_compile_definitions(
        consan_input_test
        PRIVATE CONSAN_FIXTURE_DIR="${_fixture_dir}"
    )
else()
    message(
        STATUS
        "ConSan original HSACO fixtures unavailable (need AMD LLVM assembler/linker)"
    )
endif()
# Preserve the explicit target/wave suffix across CMake versions. Older GoogleTest
# discovery otherwise appends GetParam comments to custom parameter names.
rj_add_gtest_tests(consan_input_test NO_PRETTY_VALUES)

add_test(
    NAME ConSanConfigure.EnvironmentOnlySdk
    COMMAND
        ${Python3_EXECUTABLE} -B
        ${PROJECT_SOURCE_DIR}/tests/code/consan/test_configure.py --build
        ${PROJECT_BINARY_DIR} --rocm ${ROCM_PATH}
)

set(RJ_CONSAN_TEST_ARTIFACT_ROOT
    "${PROJECT_BINARY_DIR}/Testing/ConSan"
    CACHE PATH
    "Directory for preserved ConSan integration evidence"
)
if(TARGET consan_original_fixtures)
    add_test(
        NAME ConSanPrepareCli.RejectNonExecutableOriginalBeforePublication
        COMMAND
            ${Python3_EXECUTABLE} -B
            ${CMAKE_CURRENT_SOURCE_DIR}/prepare_cli_test.py
            $<TARGET_FILE:rj_consan_prepare> ${_fixture_dir}/race.hsaco
            ${RJ_CONSAN_TEST_ARTIFACT_ROOT}
    )
    add_test(
        NAME ConSanPrepareCli.RejectBoundedInputAndFilesystemFailures
        COMMAND
            ${Python3_EXECUTABLE} -B
            ${CMAKE_CURRENT_SOURCE_DIR}/prepare_cli_io_test.py
            $<TARGET_FILE:rj_consan_prepare> ${_fixture_dir}/race.hsaco
            ${RJ_CONSAN_TEST_ARTIFACT_ROOT}
    )
endif()

add_executable(consan_bootstrap_sim_test bootstrap_sim_test.cpp)
target_include_directories(
    consan_bootstrap_sim_test
    PRIVATE ${PROJECT_SOURCE_DIR}/tests
)
target_link_libraries(
    consan_bootstrap_sim_test
    PRIVATE
        rocjitsu_consan_compiler
        ${RJ_OBJECT_LIBS}
        flatbuffers
        GTest::gtest_main
)
rj_configure_target(consan_bootstrap_sim_test PRIVATE_INCLUDES GENERATED WARNINGS SCHEMAS)
if(TARGET consan_original_fixtures)
    add_dependencies(consan_bootstrap_sim_test consan_original_fixtures)
    add_dependencies(consan_bootstrap_sim_test consan_target_fixtures)
    target_compile_definitions(
        consan_bootstrap_sim_test
        PRIVATE CONSAN_FIXTURE_DIR="${_fixture_dir}"
    )
endif()
rj_add_gtest_tests(consan_bootstrap_sim_test NO_PRETTY_VALUES)

if(TARGET consan_original_fixtures)
    set(_artifact ${CMAKE_CURRENT_BINARY_DIR}/prepared/race)
    add_custom_command(
        OUTPUT ${_artifact}/facts.bin
        BYPRODUCTS ${_artifact}/prepared.hsaco ${_artifact}/ready
        COMMAND
            ${CMAKE_COMMAND} -E make_directory
            ${CMAKE_CURRENT_BINARY_DIR}/prepared
        COMMAND ${CMAKE_COMMAND} -E rm -rf ${_artifact}
        COMMAND
            $<TARGET_FILE:rj_consan_prepare> ${_fixture_dir}/race.hsaco gfx1100
            lds_producer_consumer ${_artifact}
        DEPENDS rj_consan_prepare ${_fixture_dir}/race.hsaco
        VERBATIM
    )
    add_custom_target(consan_artifact_fixture DEPENDS ${_artifact}/facts.bin)
    add_executable(
        consan_image_test
        ${PROJECT_SOURCE_DIR}/tests/code/consan/artifact_test.cpp
    )
    target_link_libraries(
        consan_image_test
        PRIVATE rocjitsu_consan_image GTest::gtest_main
    )
    rj_configure_target(consan_image_test PRIVATE_INCLUDES WARNINGS)
    add_dependencies(consan_image_test consan_artifact_fixture)
    target_compile_definitions(
        consan_image_test
        PRIVATE
            CONSAN_ORIGINAL_FIXTURE="${_fixture_dir}/race.hsaco"
            CONSAN_ARTIFACT_FIXTURE="${_artifact}"
    )
    rj_add_gtest_tests(consan_image_test)
endif()

if(RJ_BUILD_CONSAN_RUNTIME AND TARGET consan_original_fixtures)
    add_executable(
        consan_hsa_plugin_test
        ${PROJECT_SOURCE_DIR}/tests/code/consan/hsa_plugin_test.cpp
    )
    target_link_libraries(
        consan_hsa_plugin_test
        PRIVATE
            rocjitsu_hsa_tools_headers
            rocjitsu_consan_contract
            GTest::gtest_main
            Threads::Threads
            ${CMAKE_DL_LIBS}
    )
    rj_configure_target(consan_hsa_plugin_test PRIVATE_INCLUDES WARNINGS)
    add_dependencies(
        consan_hsa_plugin_test
        hsa_consan_testable
        consan_artifact_fixture
    )
    target_compile_definitions(
        consan_hsa_plugin_test
        PRIVATE
            CONSAN_PLUGIN_FIXTURE="$<TARGET_FILE:hsa_consan_testable>"
            CONSAN_ORIGINAL_FIXTURE="${_fixture_dir}/race.hsaco"
            CONSAN_ARTIFACT_FIXTURE="${_artifact}"
    )
    rj_add_gtest_tests(consan_hsa_plugin_test)
endif()

if(
    RJ_BUILD_CONSAN_RUNTIME
    AND TARGET consan_original_fixtures
    AND EXISTS "${ROCM_PATH}/lib/libhsa-runtime64.so"
)
    add_executable(
        consan_hsa_runner
        ${CMAKE_CURRENT_SOURCE_DIR}/consan_hsa_runner.cpp
    )
    target_include_directories(
        consan_hsa_runner
        PRIVATE ${ROCJITSU_SRC_DIR} ${ROCJITSU_INCLUDE_DIR}
    )
    target_include_directories(
        consan_hsa_runner
        SYSTEM
        PRIVATE ${ROCM_PATH}/include
    )
    target_link_libraries(
        consan_hsa_runner
        PRIVATE ${ROCM_PATH}/lib/libhsa-runtime64.so ${CMAKE_DL_LIBS}
    )
    rj_configure_target(consan_hsa_runner WARNINGS)
    add_test(
        NAME ConSanRuntimeIntegration.Guards
        COMMAND
            ${Python3_EXECUTABLE} -B
            ${CMAKE_CURRENT_SOURCE_DIR}/test_runtime_e2e.py
    )
endif()

# Both environments consume the same scenarios. Runtime cases remain available
# without an analyzer, and simulator cases never treat an absent agent as a skip.
if(TARGET consan_hsa_runner)
    function(
        consan_add_target_e2e
        environment
        gpu
        wave_size
        kind
        scenario
        threads
        variant
        stage
    )
        set(_plugin hsa_consan_testable)
        set(_statistics --test-statistics)
        set(_variant_name)
        if(variant STREQUAL "production")
            set(_plugin hsa_consan)
            set(_statistics)
            set(_variant_name Production)
        elseif(NOT variant STREQUAL "testable")
            message(FATAL_ERROR "Unknown ConSan plugin variant: ${variant}")
        endif()
        set(_stage_name)
        set(_analysis_options)
        if(stage STREQUAL "runtime")
            set(_stage_name Runtime)
            set(_driver consan_runtime_e2e.py)
        elseif(stage STREQUAL "analysis")
            set(_driver consan_e2e.py)
            set(_analysis_options --analyze $<TARGET_FILE:rj_consan_analyze>)
        else()
            message(FATAL_ERROR "Unknown ConSan test stage: ${stage}")
        endif()
        set(_name
            ConSan${environment}${_stage_name}${_variant_name}E2E.${gpu}.wave${wave_size}.${kind}.${scenario}
        )
        set(_case ${scenario})
        set(_entry_options)
        set(_signal_options)
        if(_case STREQUAL "partial")
            set(_case disjoint)
        elseif(_case STREQUAL "utf8_entry")
            set(_case disjoint)
            set(_entry_options --utf8-entry)
        elseif(_case STREQUAL "amd-signals")
            set(_case disjoint)
            set(_signal_options --amd-signals)
        elseif(_case STREQUAL "amd-late-barrier-and")
            set(_case late-barrier-and)
            set(_signal_options --amd-signals)
        elseif(_case STREQUAL "amd-late-barrier-or")
            set(_case late-barrier-or)
            set(_signal_options --amd-signals)
        endif()
        set(_execution_options)
        if(environment STREQUAL "Sim")
            if(gpu STREQUAL "gfx942")
                set(_config gfx942_cdna3_kmd)
            elseif(gpu STREQUAL "gfx950")
                set(_config gfx950_mi355x_kmd)
            elseif(gpu STREQUAL "gfx1100")
                set(_config gfx1100_w7900)
            elseif(gpu STREQUAL "gfx1201")
                set(_config gfx1201_r9700)
            else()
                set(_config gfx1250_mi455x)
            endif()
            list(
                APPEND _execution_options
                --launcher
                $<TARGET_FILE:rocjitsu_bin>
                --config
                ${PROJECT_SOURCE_DIR}/configs/${_config}.json
                --require-agent
            )
        endif()
        add_test(
            NAME ${_name}
            COMMAND
                ${Python3_EXECUTABLE} -B ${CMAKE_CURRENT_SOURCE_DIR}/${_driver}
                --prepare $<TARGET_FILE:rj_consan_prepare> --runner
                $<TARGET_FILE:consan_hsa_runner> --inspect
                $<TARGET_FILE:consan_capture_inspect> --plugin
                $<TARGET_FILE:${_plugin}> ${_statistics} ${_analysis_options}
                --fixtures ${_fixture_dir} --source ${PROJECT_SOURCE_DIR} --root
                ${RJ_CONSAN_TEST_ARTIFACT_ROOT} --case ${_case} --target ${gpu}
                --wave-size ${wave_size} --threads ${threads} --fixture-kind
                ${kind} ${_entry_options} ${_signal_options}
                ${_execution_options}
        )
        set_tests_properties(${_name} PROPERTIES TIMEOUT 120)
        if(environment STREQUAL "Sim")
            rj_test_runtime_dir_env(_runtime ${_name})
            set_tests_properties(
                ${_name}
                PROPERTIES
                    LABELS "consan;simulator;required-consan"
                    ENVIRONMENT "${_runtime}"
            )
        else()
            set_tests_properties(
                ${_name}
                PROPERTIES
                    LABELS "consan;physical-gpu"
                    SKIP_RETURN_CODE 77
                    RESOURCE_LOCK consan-${gpu}
            )
        endif()
    endfunction()
    set(_named_runtime_cases
        race
        disjoint
        read_read
        collision
        single_lane
        repeat
        alias
        immediate
        four_ranges
        exhausted
        empty
        branched
        wrong-artifact
        bad-geometry
        profiling
        reused-signal
    )
    set(_named_analysis_cases ${_named_runtime_cases} repeat_race utf8_entry)
    foreach(
        _mode
        IN
        ITEMS
            gfx942:64
            gfx950:64
            gfx1100:32
            gfx1100:64
            gfx1201:32
            gfx1201:64
            gfx1250:32
    )
        string(REPLACE ":" ";" _parts ${_mode})
        list(GET _parts 0 _gpu)
        list(GET _parts 1 _wave_size)
        math(EXPR _threads "2 * ${_wave_size}")
        math(EXPR _partial_threads "3 * ${_wave_size} + 1")
        set(_kind hip)
        if(_gpu STREQUAL "gfx1250")
            set(_kind assembly)
        endif()
        foreach(_environment IN ITEMS Sim Physical)
            foreach(_scenario IN LISTS _named_runtime_cases)
                consan_add_target_e2e(${_environment} ${_gpu} ${_wave_size} named-assembly
                    ${_scenario} ${_threads} testable runtime
                )
            endforeach()
            if(RJ_BUILD_CONSAN_ANALYZER)
                if(_gpu STREQUAL "gfx1250")
                    foreach(
                        _barrier_case
                        IN
                        ITEMS
                            barrier
                            barrier_empty
                            barrier_high
                            barrier_separated
                            barrier_multiple
                    )
                        consan_add_target_e2e(${_environment} ${_gpu} ${_wave_size} assembly
                            ${_barrier_case} 65 production analysis
                        )
                    endforeach()
                endif()
                foreach(_scenario IN LISTS _named_analysis_cases)
                    consan_add_target_e2e(${_environment} ${_gpu} ${_wave_size} named-assembly
                        ${_scenario} ${_threads} testable analysis
                    )
                endforeach()
                foreach(_scenario IN ITEMS race disjoint read_read)
                    consan_add_target_e2e(${_environment} ${_gpu} ${_wave_size} ${_kind}
                        ${_scenario} ${_threads} testable analysis
                    )
                endforeach()
                consan_add_target_e2e(${_environment} ${_gpu} ${_wave_size} assembly
                    high_lane ${_threads} testable analysis
                )
                foreach(
                    _scenario
                    IN
                    ITEMS
                        disjoint
                        missing-statistics
                        batch-barrier-and
                        batch-barrier-or
                        late-barrier-and
                        late-barrier-or
                        amd-signals
                        amd-late-barrier-and
                        amd-late-barrier-or
                )
                    consan_add_target_e2e(${_environment} ${_gpu} ${_wave_size} ${_kind}
                        ${_scenario} ${_threads} production analysis
                    )
                endforeach()
                consan_add_target_e2e(${_environment} ${_gpu} ${_wave_size} ${_kind}
                    partial ${_partial_threads} testable analysis
                )
            endif()
        endforeach()
    endforeach()
endif()

if(RJ_BUILD_CONSAN_ANALYZER)
    add_test(
        NAME ConSanRequiredCoverage.Guards
        COMMAND
            ${Python3_EXECUTABLE} -B
            ${PROJECT_SOURCE_DIR}/tests/code/consan/test_required.py
    )
endif()
