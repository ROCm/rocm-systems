# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

add_executable(
    consan_input_test
    input_test.cpp
    prepare_test.cpp
    row_layout_test.cpp
    callable_probe_test.cpp
    application_instruction_test.cpp
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
    include(multi_entry.cmake)
    set(_composition_object ${_fixture_dir}/compilation-limits.o)
    set(_composition_image ${_fixture_dir}/compilation-limits.hsaco)
    add_custom_command(
        OUTPUT ${_composition_object}
        COMMAND
            ${CMAKE_CXX_COMPILER_LAUNCHER} ${AMDCXX} -x assembler-with-cpp
            -target amdgcn-amd-amdhsa -mcpu=gfx1250 -mcode-object-version=5
            -mno-wavefrontsize64 -c
            ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/compilation_limits.s -o
            ${_composition_object}
        DEPENDS fixtures/compilation_limits.s ${AMDCXX}
        VERBATIM
    )
    add_custom_command(
        OUTPUT ${_composition_image}
        COMMAND
            ${ROCM_PATH}/lib/llvm/bin/ld.lld -shared ${_composition_object} -o
            ${_composition_image}
        DEPENDS ${_composition_object}
        VERBATIM
    )
    set(_composition_prologue_object ${_fixture_dir}/compilation-prologue.o)
    set(_composition_prologue_image ${_fixture_dir}/compilation-prologue.hsaco)
    add_custom_command(
        OUTPUT ${_composition_prologue_object}
        COMMAND
            ${CMAKE_CXX_COMPILER_LAUNCHER} ${AMDCXX} -x assembler-with-cpp
            -target amdgcn-amd-amdhsa -mcpu=gfx1250 -mcode-object-version=5
            -mno-wavefrontsize64 -DCONSAN_COMPOSITION_SITES=1
            -DCONSAN_COMPOSITION_PADDING_WORDS=32768 -c
            ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/compilation_limits.s -o
            ${_composition_prologue_object}
        DEPENDS fixtures/compilation_limits.s ${AMDCXX}
        VERBATIM
    )
    add_custom_command(
        OUTPUT ${_composition_prologue_image}
        COMMAND
            ${ROCM_PATH}/lib/llvm/bin/ld.lld -shared
            ${_composition_prologue_object} -o ${_composition_prologue_image}
        DEPENDS ${_composition_prologue_object}
        VERBATIM
    )
    set(_composition_split_prologue_object
        ${_fixture_dir}/compilation-split-prologue.o
    )
    set(_composition_split_prologue_image
        ${_fixture_dir}/compilation-split-prologue.hsaco
    )
    add_custom_command(
        OUTPUT ${_composition_split_prologue_object}
        COMMAND
            ${CMAKE_CXX_COMPILER_LAUNCHER} ${AMDCXX} -x assembler-with-cpp
            -target amdgcn-amd-amdhsa -mcpu=gfx1250 -mcode-object-version=5
            -mno-wavefrontsize64 -DCONSAN_COMPOSITION_SITES=1
            -DCONSAN_COMPOSITION_GAP_WORDS=32768 -c
            ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/compilation_limits.s -o
            ${_composition_split_prologue_object}
        DEPENDS fixtures/compilation_limits.s ${AMDCXX}
        VERBATIM
    )
    add_custom_command(
        OUTPUT ${_composition_split_prologue_image}
        COMMAND
            ${ROCM_PATH}/lib/llvm/bin/ld.lld -shared
            ${_composition_split_prologue_object} -o
            ${_composition_split_prologue_image}
        DEPENDS ${_composition_split_prologue_object}
        VERBATIM
    )
    add_custom_target(
        consan_composition_fixture
        DEPENDS
            ${_composition_image}
            ${_composition_prologue_image}
            ${_composition_split_prologue_image}
    )
    add_dependencies(consan_multi_entry_fixture consan_composition_fixture)
    target_sources(consan_input_test PRIVATE target_test.cpp)
    add_custom_target(consan_original_fixtures DEPENDS ${_fixtures})
    add_dependencies(consan_input_test consan_original_fixtures)
    add_dependencies(
        consan_input_test
        consan_target_fixtures
        consan_multi_entry_fixture
    )
    target_compile_definitions(
        consan_input_test
        PRIVATE
            CONSAN_FIXTURE_DIR="${_fixture_dir}"
            CONSAN_MULTI_ENTRY_FIXTURE="${_fixture_dir}/multi-entry.hsaco"
            CONSAN_MULTI_ENTRY_BARRIER_FIXTURE="${_fixture_dir}/multi-entry-barrier.hsaco"
            CONSAN_MULTI_ENTRY_LOOP_FIXTURE="${_fixture_dir}/multi-entry-loop.hsaco"
            CONSAN_MULTI_ENTRY_MIXED_FIXTURE="${_fixture_dir}/multi-entry-mixed.hsaco"
            CONSAN_MULTI_ENTRY_NATIVE_GLOBAL_FIXTURE="${_fixture_dir}/multi-entry-native-global.hsaco"
            CONSAN_MULTI_ENTRY_UNREACHABLE_PC_FIXTURE="${_fixture_dir}/multi-entry-unreachable-pc.hsaco"
            CONSAN_MULTI_ENTRY_UNSUPPORTED_FIXTURE="${_fixture_dir}/multi-entry-unsupported.hsaco"
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
        NAME ConSanPrepareCli.CompositionCapacity
        COMMAND
            ${Python3_EXECUTABLE} -B
            ${CMAKE_CURRENT_SOURCE_DIR}/prepare_cli_composition_test.py
            --prepare $<TARGET_FILE:rj_consan_prepare> --fixture
            ${_composition_image} --prologue-fixture
            ${_composition_prologue_image} --split-prologue-fixture
            ${_composition_split_prologue_image} --root
            ${RJ_CONSAN_TEST_ARTIFACT_ROOT}
    )
    add_test(
        NAME ConSanPrepareCli.BestEffortCoverage
        COMMAND
            ${Python3_EXECUTABLE} -B
            ${CMAKE_CURRENT_SOURCE_DIR}/prepare_cli_coverage_test.py
            $<TARGET_FILE:rj_consan_prepare>
            ${_fixture_dir}/multi-entry-unsupported.hsaco
            ${RJ_CONSAN_TEST_ARTIFACT_ROOT}
    )
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
if(RJ_BUILD_CONSAN_ANALYZER)
    target_link_libraries(
        consan_bootstrap_sim_test
        PRIVATE rocjitsu_consan_analysis
    )
    target_compile_definitions(
        consan_bootstrap_sim_test
        PRIVATE CONSAN_ANALYZER_ENABLED=1
    )
endif()
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
    set(_application_lds_artifacts)
    foreach(_scope IN ITEMS barrier loop)
        if(_scope STREQUAL "barrier")
            set(_case 15)
        else()
            set(_case 27)
        endif()
        set(_application_${_scope}_artifact
            ${CMAKE_CURRENT_BINARY_DIR}/prepared/application-${_scope}
        )
        set(_scope_artifact ${_application_${_scope}_artifact})
        set(_scope_image
            ${_fixture_dir}/targets/gfx1100/wave32/native-${_case}.hsaco
        )
        add_custom_command(
            OUTPUT ${_scope_artifact}/facts.bin
            BYPRODUCTS
                ${_scope_artifact}/prepared.hsaco
                ${_scope_artifact}/ready
            COMMAND
                ${CMAKE_COMMAND} -E make_directory
                ${CMAKE_CURRENT_BINARY_DIR}/prepared
            COMMAND ${CMAKE_COMMAND} -E rm -rf ${_scope_artifact}
            COMMAND
                $<TARGET_FILE:rj_consan_prepare> ${_scope_image} gfx1100
                lds_producer_consumer ${_scope_artifact}
            DEPENDS rj_consan_prepare ${_scope_image}
            VERBATIM
        )
        list(APPEND _application_lds_artifacts ${_scope_artifact}/facts.bin)
    endforeach()
    add_custom_target(
        consan_application_lds_artifacts
        DEPENDS ${_application_lds_artifacts}
    )
    set(_multi_artifact ${CMAKE_CURRENT_BINARY_DIR}/prepared/multi-entry)
    add_custom_command(
        OUTPUT ${_multi_artifact}/facts.bin
        BYPRODUCTS ${_multi_artifact}/prepared.hsaco ${_multi_artifact}/ready
        COMMAND
            ${CMAKE_COMMAND} -E make_directory
            ${CMAKE_CURRENT_BINARY_DIR}/prepared
        COMMAND ${CMAKE_COMMAND} -E rm -rf ${_multi_artifact}
        COMMAND
            $<TARGET_FILE:rj_consan_prepare> ${_fixture_dir}/multi-entry.hsaco
            gfx1250 ${_multi_artifact} --entry mk_a --entry mk_b
        DEPENDS rj_consan_prepare ${_fixture_dir}/multi-entry.hsaco
        VERBATIM
    )
    add_custom_target(
        consan_multi_artifact_fixture
        DEPENDS ${_multi_artifact}/facts.bin
    )
    set(_native_shape_artifact
        ${CMAKE_CURRENT_BINARY_DIR}/prepared/native-required-shape
    )
    add_custom_command(
        OUTPUT ${_native_shape_artifact}/facts.bin
        BYPRODUCTS
            ${_native_shape_artifact}/prepared.hsaco
            ${_native_shape_artifact}/ready
        COMMAND ${CMAKE_COMMAND} -E rm -rf ${_native_shape_artifact}
        COMMAND
            $<TARGET_FILE:rj_consan_prepare>
            ${_fixture_dir}/multi-entry-abi.hsaco gfx1250
            ${_native_shape_artifact} --entry mk_b
        DEPENDS rj_consan_prepare ${_fixture_dir}/multi-entry-abi.hsaco
        VERBATIM
    )
    add_custom_target(
        consan_native_shape_artifact_fixture
        DEPENDS ${_native_shape_artifact}/facts.bin
    )
    foreach(_alignment_disposition IN ITEMS selected native)
        set(_alignment_artifact
            ${CMAKE_CURRENT_BINARY_DIR}/prepared/kernarg-alignment-${_alignment_disposition}
        )
        set(_alignment_${_alignment_disposition}_artifact
            ${_alignment_artifact}
        )
        if(_alignment_disposition STREQUAL "selected")
            set(_alignment_entry mk_a)
        else()
            set(_alignment_entry mk_b)
        endif()
        add_custom_command(
            OUTPUT ${_alignment_artifact}/facts.bin
            BYPRODUCTS
                ${_alignment_artifact}/prepared.hsaco
                ${_alignment_artifact}/ready
            COMMAND ${CMAKE_COMMAND} -E rm -rf ${_alignment_artifact}
            COMMAND
                $<TARGET_FILE:rj_consan_prepare>
                ${_fixture_dir}/multi-entry-alignment.hsaco gfx1250
                ${_alignment_artifact} --entry ${_alignment_entry}
            DEPENDS
                rj_consan_prepare
                ${_fixture_dir}/multi-entry-alignment.hsaco
            VERBATIM
        )
        add_custom_target(
            consan_${_alignment_disposition}_alignment_artifact_fixture
            DEPENDS ${_alignment_artifact}/facts.bin
        )
    endforeach()
    set(_trace_artifact ${CMAKE_CURRENT_BINARY_DIR}/prepared/multi-trace)
    add_custom_command(
        OUTPUT ${_trace_artifact}/facts.bin
        BYPRODUCTS ${_trace_artifact}/prepared.hsaco ${_trace_artifact}/ready
        COMMAND ${CMAKE_COMMAND} -E rm -rf ${_trace_artifact}
        COMMAND
            $<TARGET_FILE:rj_consan_prepare> ${_fixture_dir}/multi-entry.hsaco
            gfx1250 ${_trace_artifact} --profile execution-trace --entry mk_a
            --entry mk_b
        DEPENDS rj_consan_prepare ${_fixture_dir}/multi-entry.hsaco
        VERBATIM
    )
    add_custom_target(
        consan_trace_artifact_fixture
        DEPENDS ${_trace_artifact}/facts.bin
    )
    add_executable(
        consan_image_test
        ${PROJECT_SOURCE_DIR}/tests/code/consan/artifact_test.cpp
    )
    target_link_libraries(
        consan_image_test
        PRIVATE rocjitsu_consan_image GTest::gtest_main
    )
    rj_configure_target(consan_image_test PRIVATE_INCLUDES WARNINGS)
    add_dependencies(
        consan_image_test
        consan_artifact_fixture
        consan_multi_entry_fixture
    )
    target_compile_definitions(
        consan_image_test
        PRIVATE
            CONSAN_ORIGINAL_FIXTURE="${_fixture_dir}/race.hsaco"
            CONSAN_ARTIFACT_FIXTURE="${_artifact}"
            CONSAN_MULTI_ENTRY_FIXTURE="${_fixture_dir}/multi-entry.hsaco"
            CONSAN_MULTI_ENTRY_BARRIER_FIXTURE="${_fixture_dir}/multi-entry-barrier.hsaco"
            CONSAN_MULTI_ENTRY_LOOP_FIXTURE="${_fixture_dir}/multi-entry-loop.hsaco"
            CONSAN_MULTI_ENTRY_MIXED_FIXTURE="${_fixture_dir}/multi-entry-mixed.hsaco"
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
            rocjitsu_consan_image
            GTest::gtest_main
            Threads::Threads
            ${CMAKE_DL_LIBS}
    )
    rj_configure_target(consan_hsa_plugin_test PRIVATE_INCLUDES WARNINGS)
    add_dependencies(
        consan_hsa_plugin_test
        hsa_consan_testable
        consan_artifact_fixture
        consan_multi_artifact_fixture
        consan_application_lds_artifacts
        consan_native_shape_artifact_fixture
        consan_selected_alignment_artifact_fixture
        consan_native_alignment_artifact_fixture
        consan_trace_artifact_fixture
    )
    target_compile_definitions(
        consan_hsa_plugin_test
        PRIVATE
            CONSAN_PLUGIN_FIXTURE="$<TARGET_FILE:hsa_consan_testable>"
            CONSAN_ORIGINAL_FIXTURE="${_fixture_dir}/race.hsaco"
            CONSAN_ARTIFACT_FIXTURE="${_artifact}"
            CONSAN_MULTI_ORIGINAL_FIXTURE="${_fixture_dir}/multi-entry.hsaco"
            CONSAN_MULTI_ARTIFACT_FIXTURE="${_multi_artifact}"
            CONSAN_APPLICATION_BARRIER_ORIGINAL="${_fixture_dir}/targets/gfx1100/wave32/native-15.hsaco"
            CONSAN_APPLICATION_BARRIER_ARTIFACT="${_application_barrier_artifact}"
            CONSAN_APPLICATION_LOOP_ARTIFACT="${_application_loop_artifact}"
            CONSAN_NATIVE_SHAPE_ORIGINAL_FIXTURE="${_fixture_dir}/multi-entry-abi.hsaco"
            CONSAN_NATIVE_SHAPE_ARTIFACT_FIXTURE="${_native_shape_artifact}"
            CONSAN_ALIGNMENT_ORIGINAL_FIXTURE="${_fixture_dir}/multi-entry-alignment.hsaco"
            CONSAN_SELECTED_ALIGNMENT_ARTIFACT_FIXTURE="${_alignment_selected_artifact}"
            CONSAN_NATIVE_ALIGNMENT_ARTIFACT_FIXTURE="${_alignment_native_artifact}"
            CONSAN_MULTI_TRACE_ARTIFACT_FIXTURE="${_trace_artifact}"
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
    add_executable(consan_multi_entry_runner consan_multi_entry_runner.cpp)
    target_include_directories(
        consan_multi_entry_runner
        PRIVATE ${ROCJITSU_SRC_DIR} ${ROCJITSU_INCLUDE_DIR}
    )
    target_include_directories(
        consan_multi_entry_runner
        SYSTEM
        PRIVATE ${ROCM_PATH}/include
    )
    target_link_libraries(
        consan_multi_entry_runner
        PRIVATE ${ROCM_PATH}/lib/libhsa-runtime64.so ${CMAKE_DL_LIBS}
    )
    rj_configure_target(consan_multi_entry_runner WARNINGS)
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
        set(_geometry_options)
        if(ARGC GREATER 8)
            list(APPEND _geometry_options --workgroups ${ARGV8})
            string(APPEND _name ".groups${ARGV8}")
        endif()
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
                ${_execution_options} ${_geometry_options}
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
        math(EXPR _width_threads "2 * ${_wave_size} + 1")
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
                if(_gpu STREQUAL "gfx950" OR _gpu STREQUAL "gfx1250")
                    foreach(_bits IN ITEMS 4 6 8 16)
                        foreach(
                            _scenario
                            IN
                            ITEMS full partial empty overlap adjacent short
                        )
                            if(
                                _scenario STREQUAL "short"
                                AND NOT _gpu STREQUAL "gfx1250"
                            )
                                continue()
                            endif()
                            if(
                                _gpu STREQUAL "gfx950"
                                AND _scenario STREQUAL "partial"
                                AND _environment STREQUAL "Physical"
                            )
                                continue() # The gfx950 ISA defines only full and empty EXEC.
                            endif()
                            set(_launch_size ${_wave_size})
                            if(
                                _scenario STREQUAL "overlap"
                                OR _scenario STREQUAL "adjacent"
                            )
                                set(_launch_size ${_threads})
                            elseif(_scenario STREQUAL "short")
                                set(_launch_size 4)
                            endif()
                            consan_add_target_e2e(${_environment} ${_gpu} ${_wave_size} assembly
                                transpose_${_bits}_${_scenario} ${_launch_size} production analysis
                            )
                        endforeach()
                    endforeach()
                endif()
                foreach(_atomic IN LISTS _atomic_cases_${_gpu}_${_wave_size})
                    string(REPLACE "-" "_" _case ${_atomic})
                    consan_add_target_e2e(${_environment} ${_gpu} ${_wave_size} assembly
                        ${_case} ${_threads} production analysis
                    )
                endforeach()
                set(_width_cases
                    wide_u8
                    wide_i8_alias_high
                    wide_d16_u8
                    wide_d16_i8_high
                    wide_d16_u16
                    wide_u16
                    wide_i16
                    wide_b64
                    wide_b96
                    wide_b96_alias_high
                    wide_b128
                )
                if(_gpu STREQUAL "gfx1250")
                    list(
                        APPEND _width_cases
                        wide_u16_high
                        wide_i16_high
                        wide_b64_high
                        wide_b128_high
                        wide_alias
                        wide_collision
                        mixed_overlap
                        mixed_b16
                        mixed_disjoint
                        mixed_read_read
                    )
                endif()
                foreach(_width_case IN LISTS _width_cases)
                    consan_add_target_e2e(${_environment} ${_gpu} ${_wave_size} assembly
                        ${_width_case} ${_width_threads} production analysis
                    )
                endforeach()
                foreach(_peer IN ITEMS overlap adjacent)
                    consan_add_target_e2e(${_environment} ${_gpu} ${_wave_size} assembly
                        wide_d16_${_peer} ${_threads} production analysis
                    )
                endforeach()
                foreach(_bits IN ITEMS 32 64)
                    foreach(_spacing IN ITEMS normal stride64)
                        set(_dual wide_dual_b${_bits}_${_spacing})
                        consan_add_target_e2e(${_environment} ${_gpu} ${_wave_size} assembly
                            ${_dual}_values_high ${_width_threads} production analysis
                        )
                        foreach(_peer IN ITEMS overlap gap)
                            consan_add_target_e2e(${_environment} ${_gpu} ${_wave_size} assembly
                                ${_dual}_${_peer} ${_threads} production analysis
                            )
                        endforeach()
                    endforeach()
                endforeach()
                foreach(_peer IN ITEMS overlap adjacent disjoint)
                    consan_add_target_e2e(${_environment} ${_gpu} ${_wave_size} assembly
                        wide_b8_${_peer} ${_threads} production analysis
                    )
                    consan_add_target_e2e(${_environment} ${_gpu} ${_wave_size} assembly
                        wide_b96_${_peer} ${_threads} production analysis
                    )
                endforeach()
                foreach(_groups IN ITEMS 1 4 16)
                    consan_add_target_e2e(${_environment} ${_gpu} ${_wave_size} assembly
                        grouped_disjoint ${_width_threads} production analysis ${_groups}
                    )
                endforeach()
                foreach(_groups IN ITEMS 4 16)
                    foreach(
                        _scenario
                        IN
                        ITEMS grouped_race grouped_read_read grouped_isolation
                    )
                        consan_add_target_e2e(${_environment} ${_gpu} ${_wave_size} assembly
                            ${_scenario} ${_width_threads} production analysis ${_groups}
                        )
                    endforeach()
                endforeach()
                consan_add_target_e2e(${_environment} ${_gpu} ${_wave_size} assembly
                    grouped_repeat ${_width_threads} testable analysis 4
                )
                consan_add_target_e2e(${_environment} ${_gpu} ${_wave_size} assembly
                    grouped_oversized ${_width_threads} testable analysis 17
                )
                consan_add_target_e2e(${_environment} ${_gpu} ${_wave_size} assembly
                    grouped_edge ${_width_threads} testable analysis 4
                )
                consan_add_target_e2e(${_environment} ${_gpu} ${_wave_size} assembly
                    grouped_missing_id ${_width_threads} testable analysis 4
                )
                consan_add_target_e2e(${_environment} ${_gpu} ${_wave_size} assembly
                    single_missing_id ${_width_threads} production analysis 1
                )
                if(_gpu STREQUAL "gfx1250")
                    consan_add_target_e2e(${_environment} ${_gpu} ${_wave_size} assembly
                        mixed_barrier ${_width_threads} production analysis
                    )
                endif()
                foreach(_groups IN ITEMS 1 4 16)
                    foreach(_scenario IN ITEMS grouped_barrier grouped_empty)
                        consan_add_target_e2e(${_environment} ${_gpu} ${_wave_size} assembly
                            ${_scenario} ${_width_threads} production analysis ${_groups}
                        )
                    endforeach()
                endforeach()
                foreach(
                    _barrier_case
                    IN
                    ITEMS
                        barrier
                        barrier_empty
                        barrier_high
                        barrier_separated
                        barrier_multiple
                        barrier_missing_second
                )
                    consan_add_target_e2e(${_environment} ${_gpu} ${_wave_size} assembly
                        ${_barrier_case} ${_width_threads} production analysis
                    )
                endforeach()
                foreach(
                    _loop_case
                    IN
                    ITEMS
                        loop_zero
                        loop_one
                        loop_three
                        loop_access_loss
                        loop_sync_loss
                        loop_no_b0
                        loop_no_b0_full_journal
                        loop_no_b0_sync_loss
                        loop_no_b1
                        loop_late
                )
                    consan_add_target_e2e(${_environment} ${_gpu} ${_wave_size} assembly
                        ${_loop_case} ${_threads} production analysis
                    )
                endforeach()
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

if(TARGET consan_multi_entry_runner AND TARGET rj_consan_analyze)
    add_dependencies(consan_multi_entry_runner consan_multi_entry_fixture)
    foreach(
        _profile
        IN
        ITEMS
            SelectedAB
            WrongProfile
            BarrierAB
            LoopAB
            NativeApplication
            ApplicationLds
            ApplicationLdsBarrier
    )
        set(_profile_options)
        set(_multi_case_image ${_fixture_dir}/multi-entry.hsaco)
        set(_multi_variant no-sync)
        if(_profile STREQUAL "WrongProfile")
            set(_profile_options --wrong-profile)
        elseif(_profile STREQUAL "BarrierAB")
            set(_multi_case_image ${_fixture_dir}/multi-entry-barrier.hsaco)
            set(_multi_variant barrier)
        elseif(_profile STREQUAL "LoopAB")
            set(_multi_case_image ${_fixture_dir}/multi-entry-loop.hsaco)
            set(_multi_variant loop)
        elseif(_profile STREQUAL "NativeApplication")
            set(_profile_options --native-application)
        elseif(_profile STREQUAL "ApplicationLds")
            set(_profile_options --application)
        elseif(_profile STREQUAL "ApplicationLdsBarrier")
            set(_profile_options --application)
            set(_multi_case_image ${_fixture_dir}/multi-entry-barrier.hsaco)
            set(_multi_variant barrier)
        endif()
        set(_test ConSanMultiEntrySim.${_profile})
        add_test(
            NAME ${_test}
            COMMAND
                ${Python3_EXECUTABLE} -B
                ${CMAKE_CURRENT_SOURCE_DIR}/consan_multi_entry_e2e.py --prepare
                $<TARGET_FILE:rj_consan_prepare> --runner
                $<TARGET_FILE:consan_multi_entry_runner> --inspect
                $<TARGET_FILE:consan_capture_inspect> --analyze
                $<TARGET_FILE:rj_consan_analyze> --plugin
                $<TARGET_FILE:hsa_consan_testable> --launcher
                $<TARGET_FILE:rocjitsu_bin> --fixture ${_multi_case_image}
                --variant ${_multi_variant} --config
                ${PROJECT_SOURCE_DIR}/configs/gfx1250_mi455x.json --root
                ${RJ_CONSAN_TEST_ARTIFACT_ROOT} --source ${PROJECT_SOURCE_DIR}
                ${_profile_options}
        )
        rj_test_runtime_dir_env(_runtime ${_test})
        set_tests_properties(
            ${_test}
            PROPERTIES
                TIMEOUT 240
                LABELS "consan;simulator;required-consan"
                ENVIRONMENT "${_runtime}"
        )
    endforeach()
endif()

add_test(
    NAME ConSanPrefillHarness.Guards
    COMMAND
        ${Python3_EXECUTABLE} -B
        ${CMAKE_CURRENT_SOURCE_DIR}/test_iree_prefill_e2e.py
)
set_tests_properties(
    ConSanPrefillHarness.Guards
    PROPERTIES TIMEOUT 30 LABELS "consan;required-consan"
)

if(TARGET consan_multi_entry_runner AND TARGET rj_consan_analyze)
    foreach(_groups IN ITEMS 1 16 512)
        set(_test ConSanExecutionTraceSim.Groups${_groups})
        add_test(
            NAME ${_test}
            COMMAND
                ${Python3_EXECUTABLE} -B
                ${CMAKE_CURRENT_SOURCE_DIR}/consan_execution_trace_e2e.py
                --prepare $<TARGET_FILE:rj_consan_prepare> --runner
                $<TARGET_FILE:consan_multi_entry_runner> --inspect
                $<TARGET_FILE:consan_capture_inspect> --analyze
                $<TARGET_FILE:rj_consan_analyze> --plugin
                $<TARGET_FILE:hsa_consan_testable> --launcher
                $<TARGET_FILE:rocjitsu_bin> --fixture
                ${_fixture_dir}/multi-entry-trace-exits.hsaco --config
                ${PROJECT_SOURCE_DIR}/configs/gfx1250_mi455x.json --root
                ${RJ_CONSAN_TEST_ARTIFACT_ROOT} --source ${PROJECT_SOURCE_DIR}
                --workgroups ${_groups}
        )
        rj_test_runtime_dir_env(_runtime ${_test})
        set_tests_properties(
            ${_test}
            PROPERTIES
                TIMEOUT 240
                LABELS "consan;simulator;required-consan"
                ENVIRONMENT "${_runtime}"
        )
    endforeach()
endif()
