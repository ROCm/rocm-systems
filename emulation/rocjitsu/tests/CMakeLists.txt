# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

include(rj_configure_target)
include(rj_find_amdcxx)

# Device kernels (compiled from HIP sources for simulator testing).
add_subdirectory(kernels)

# Common object libraries linked by all test/tool executables.
set(RJ_OBJECT_LIBS
    rocjitsu_analysis
    rocjitsu_code
    rocjitsu_isa
    rocjitsu_isa_registry
    simdojo
    rocjitsu_kmd
    rocjitsu_vm
    rocjitsu_vm_amdgpu
    rocjitsu_plugin_logging
    rocjitsu_plugin_race_detector
    rocjitsu_plugin_throughput
    rocjitsu_plugin_loader
    rocjitsu_vm_risc_v
    rocjitsu_config
    rocjitsu_dbt_config
    rocjitsu_pci
)

if(RJ_INSTALL_TESTS)
    set(RJ_INSTALL_TEST_DIR "${CMAKE_INSTALL_DATADIR}/rocjitsu/tests")
    set(RJ_INSTALL_TEST_BIN_DIR "${RJ_INSTALL_TEST_DIR}/bin")
    set(RJ_INSTALL_CTEST_FILE
        "${CMAKE_CURRENT_BINARY_DIR}/install/CTestTestfile.cmake"
    )
    set(RJ_INSTALL_GTEST_GENERATOR
        "${PROJECT_SOURCE_DIR}/cmake/rj_generate_installed_gtest_ctest.cmake"
    )
    if(IS_ABSOLUTE "${CMAKE_INSTALL_BINDIR}")
        set(_rj_installed_bindir "${CMAKE_INSTALL_BINDIR}")
    else()
        set(_rj_installed_bindir "../../../${CMAKE_INSTALL_BINDIR}")
    endif()
    if(IS_ABSOLUTE "${CMAKE_INSTALL_DATADIR}")
        set(_rj_installed_datadir "${CMAKE_INSTALL_DATADIR}")
    else()
        set(_rj_installed_datadir "../..")
    endif()
    if(IS_ABSOLUTE "${CMAKE_INSTALL_LIBDIR}")
        set(_rj_installed_libdir "${CMAKE_INSTALL_LIBDIR}")
    else()
        set(_rj_installed_libdir "../../../${CMAKE_INSTALL_LIBDIR}")
    endif()
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/install")
    file(WRITE "${RJ_INSTALL_CTEST_FILE}" "# Installed rocjitsu CTest tests\n")
    file(
        APPEND "${RJ_INSTALL_CTEST_FILE}"
        "set(RJ_INSTALLED_BINDIR \"${_rj_installed_bindir}\")\n"
        "set(RJ_INSTALLED_CONFIG_DIR \"${_rj_installed_datadir}/rocjitsu/configs\")\n\n"
        "set(RJ_INSTALLED_KERNEL_DIR \"kernels\")\n\n"
        "set(RJ_INSTALLED_LIBDIR \"${_rj_installed_libdir}\")\n\n"
        "include(\"rocjitsu_tests.cmake\")\n\n"
    )

    # Install a compiled test/tool target into the installed CTest bin directory.
    # The installed CTest file references these binaries with paths relative to
    # ${RJ_INSTALL_TEST_DIR}, e.g. bin/rocjitsu_tests.
    function(rj_install_test_target target)
        install(
            TARGETS ${target}
            RUNTIME DESTINATION ${RJ_INSTALL_TEST_BIN_DIR}
        )
    endfunction()

    # Install a non-CMake executable produced by a custom command, such as a HIP
    # test binary compiled directly with hipcc.
    function(rj_install_test_program path)
        install(PROGRAMS ${path} DESTINATION ${RJ_INSTALL_TEST_BIN_DIR})
    endfunction()

    # Append one installed CTest entry to ${RJ_INSTALL_CTEST_FILE}. This mirrors
    # the subset of add_test()/set_tests_properties() used by rj_add_test, but
    # writes plain CMake code because the file is consumed later by
    # `ctest --test-dir <installed tests dir>`.
    function(rj_install_ctest name)
        set(oneValueArgs TIMEOUT SKIP_REGULAR_EXPRESSION)
        set(multiValueArgs COMMAND ENVIRONMENT LABELS)
        cmake_parse_arguments(
            ARG
            ""
            "${oneValueArgs}"
            "${multiValueArgs}"
            ${ARGN}
        )
        if(NOT ARG_COMMAND)
            message(FATAL_ERROR "rj_install_ctest(${name}) requires COMMAND")
        endif()

        set(_content "add_test(\"${name}\"")
        foreach(_arg IN LISTS ARG_COMMAND)
            string(APPEND _content " \"${_arg}\"")
        endforeach()
        string(APPEND _content ")\n")

        set(_props)
        if(ARG_TIMEOUT)
            string(APPEND _props " \"TIMEOUT\" \"${ARG_TIMEOUT}\"")
        endif()
        if(ARG_SKIP_REGULAR_EXPRESSION)
            string(
                APPEND _props
                " \"SKIP_REGULAR_EXPRESSION\" [=[${ARG_SKIP_REGULAR_EXPRESSION}]=]"
            )
        endif()
        if(ARG_LABELS)
            list(JOIN ARG_LABELS ";" _labels)
            string(APPEND _props " \"LABELS\" \"${_labels}\"")
        endif()
        if(ARG_ENVIRONMENT)
            list(JOIN ARG_ENVIRONMENT ";" _env)
            string(APPEND _props " \"ENVIRONMENT\" \"${_env}\"")
        endif()
        if(_props)
            string(
                APPEND _content
                "set_tests_properties(\"${name}\" PROPERTIES"
            )
            string(APPEND _content "${_props})\n")
        endif()

        file(APPEND "${RJ_INSTALL_CTEST_FILE}" "${_content}\n")
    endfunction()

    # Generate and install fine-grained CTest entries for every concrete
    # GoogleTest case in a target. The target executable must already exist, so
    # this is a build-time custom command that runs --gtest_list_tests and writes
    # an installed CTest fragment like rocjitsu_tests.cmake.
    function(rj_install_gtest_cases target)
        set(_fragment "${CMAKE_CURRENT_BINARY_DIR}/install/${target}.cmake")
        cmake_parse_arguments(ARG "" "TEST_FILTER" "" ${ARGN})
        add_custom_command(
            OUTPUT "${_fragment}"
            COMMAND
                ${CMAKE_COMMAND} "-DTEST_EXECUTABLE=$<TARGET_FILE:${target}>"
                "-DINSTALLED_EXECUTABLE=bin/${target}"
                "-DTEST_FILTER=${ARG_TEST_FILTER}" "-DOUTPUT_FILE=${_fragment}"
                -P "${RJ_INSTALL_GTEST_GENERATOR}"
            DEPENDS ${target} "${RJ_INSTALL_GTEST_GENERATOR}"
            COMMENT "Generating installed CTest entries for ${target}"
            VERBATIM
        )
        add_custom_target(${target}_installed_ctest ALL DEPENDS "${_fragment}")
        install(FILES "${_fragment}" DESTINATION ${RJ_INSTALL_TEST_DIR})
    endfunction()

    install(FILES "${RJ_INSTALL_CTEST_FILE}" DESTINATION ${RJ_INSTALL_TEST_DIR})
    install(
        FILES "${PROJECT_SOURCE_DIR}/cmake/rj_run_filtered_gtest.cmake"
        DESTINATION ${RJ_INSTALL_TEST_DIR}
    )
endif()

include(GoogleTest)

set(_rj_installed_test_bin_dir "${CMAKE_INSTALL_DATADIR}/rocjitsu/tests/bin")
function(_rj_path_from_installed_test_bin out install_dir file_name)
    if(IS_ABSOLUTE "${install_dir}")
        set(_path "${install_dir}/${file_name}")
    elseif(IS_ABSOLUTE "${_rj_installed_test_bin_dir}")
        set(_path "${CMAKE_INSTALL_PREFIX}/${install_dir}/${file_name}")
    else()
        file(
            RELATIVE_PATH
            _rel_dir
            "/${_rj_installed_test_bin_dir}"
            "/${install_dir}"
        )
        set(_path "${_rel_dir}/${file_name}")
    endif()
    set(${out} "${_path}" PARENT_SCOPE)
endfunction()

function(rj_test_runtime_dir_env out name)
    string(REGEX REPLACE "[^A-Za-z0-9_.-]" "_" _runtime_name "${name}")
    set(${out}
        "ROCJITSU_RUNTIME_DIR=${CMAKE_CURRENT_BINARY_DIR}/runtime/${_runtime_name}"
        PARENT_SCOPE
    )
endfunction()

function(rj_installed_test_runtime_dir_env out name)
    string(REGEX REPLACE "[^A-Za-z0-9_.-]" "_" _runtime_name "${name}")
    set(${out} "ROCJITSU_RUNTIME_DIR=runtime/${_runtime_name}" PARENT_SCOPE)
endfunction()

function(rj_environment_has_runtime_dir out)
    set(_has_runtime_dir FALSE)
    foreach(_entry IN LISTS ARGN)
        if(_entry MATCHES "^ROCJITSU_RUNTIME_DIR=")
            set(_has_runtime_dir TRUE)
        endif()
    endforeach()
    set(${out} "${_has_runtime_dir}" PARENT_SCOPE)
endfunction()

function(rj_set_default_runtime_dir name)
    get_property(_env TEST "${name}" PROPERTY ENVIRONMENT)
    rj_environment_has_runtime_dir(_has_runtime_dir ${_env})
    if(NOT _has_runtime_dir)
        rj_test_runtime_dir_env(_runtime_dir_env "${name}")
        list(APPEND _env "${_runtime_dir_env}")
        set_tests_properties("${name}" PROPERTIES ENVIRONMENT "${_env}")
    endif()
endfunction()

# Register a test for both the build tree and, when RJ_INSTALL_TESTS is ON, the
# installed CTest tree. COMMAND is the normal build-tree command passed to
# add_test(). INSTALL_COMMAND is optional and should use paths relative to the
# installed test directory, e.g. bin/my_test or ${RJ_INSTALLED_BINDIR}/rocjitsu.
# Shared properties TIMEOUT, SKIP_REGULAR_EXPRESSION, and LABELS are applied to
# both registrations. ENVIRONMENT applies only to the build-tree test;
# INSTALL_ENVIRONMENT adds to the default installed test environment, which
# points installed tests at packaged configs, kernels, binaries, and runtimes.
function(rj_add_test)
    set(oneValueArgs NAME WORKING_DIRECTORY TIMEOUT SKIP_REGULAR_EXPRESSION)
    set(multiValueArgs
        COMMAND
        ENVIRONMENT
        INSTALL_COMMAND
        INSTALL_ENVIRONMENT
        LABELS
    )
    cmake_parse_arguments(ARG "" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    if(NOT ARG_NAME)
        message(FATAL_ERROR "rj_add_test requires NAME")
    endif()
    if(NOT ARG_COMMAND)
        message(FATAL_ERROR "rj_add_test(${ARG_NAME}) requires COMMAND")
    endif()

    rj_environment_has_runtime_dir(_has_runtime_dir ${ARG_ENVIRONMENT})
    if(NOT _has_runtime_dir)
        # Each launcher test writes the active config into its runtime dir for
        # the preloaded interposer. Use a per-test default so ctest -j runs
        # cannot overwrite another test's config or daemon socket.
        rj_test_runtime_dir_env(_runtime_dir_env "${ARG_NAME}")
        list(APPEND ARG_ENVIRONMENT "${_runtime_dir_env}")
    endif()

    add_test(NAME "${ARG_NAME}" COMMAND ${ARG_COMMAND})

    if(ARG_WORKING_DIRECTORY)
        set_tests_properties(
            "${ARG_NAME}"
            PROPERTIES WORKING_DIRECTORY "${ARG_WORKING_DIRECTORY}"
        )
    endif()
    if(ARG_TIMEOUT)
        set_tests_properties("${ARG_NAME}" PROPERTIES TIMEOUT "${ARG_TIMEOUT}")
    endif()
    if(ARG_SKIP_REGULAR_EXPRESSION)
        set_tests_properties(
            "${ARG_NAME}"
            PROPERTIES SKIP_REGULAR_EXPRESSION "${ARG_SKIP_REGULAR_EXPRESSION}"
        )
    endif()
    if(ARG_LABELS)
        list(JOIN ARG_LABELS ";" _labels)
        set_tests_properties("${ARG_NAME}" PROPERTIES LABELS "${_labels}")
    endif()
    if(ARG_ENVIRONMENT)
        list(JOIN ARG_ENVIRONMENT ";" _env)
        set_tests_properties("${ARG_NAME}" PROPERTIES ENVIRONMENT "${_env}")
    endif()

    if(RJ_INSTALL_TESTS AND ARG_INSTALL_COMMAND)
        set(_install_args COMMAND ${ARG_INSTALL_COMMAND})
        if(ARG_TIMEOUT)
            list(APPEND _install_args TIMEOUT "${ARG_TIMEOUT}")
        endif()
        if(ARG_SKIP_REGULAR_EXPRESSION)
            list(
                APPEND _install_args
                SKIP_REGULAR_EXPRESSION
                "${ARG_SKIP_REGULAR_EXPRESSION}"
            )
        endif()
        if(ARG_LABELS)
            list(APPEND _install_args LABELS ${ARG_LABELS})
        endif()
        set(_install_env
            "ROCJITSU_CONFIG_DIR=\${RJ_INSTALLED_CONFIG_DIR}"
            "ROCJITSU_KERNEL_DIR=\${RJ_INSTALLED_KERNEL_DIR}"
        )
        rj_environment_has_runtime_dir(
            _install_has_runtime_dir
            ${ARG_INSTALL_ENVIRONMENT}
        )
        if(NOT _install_has_runtime_dir)
            rj_installed_test_runtime_dir_env(
                _install_runtime_dir_env
                "${ARG_NAME}"
            )
            list(APPEND _install_env "${_install_runtime_dir_env}")
        endif()
        list(APPEND _install_env ${ARG_INSTALL_ENVIRONMENT})
        list(APPEND _install_args ENVIRONMENT ${_install_env})
        rj_install_ctest("${ARG_NAME}" ${_install_args})
    endif()
endfunction()

# Register a GoogleTest target with fine-grained tests in both the build tree
# and installed CTest tree. Build-tree discovery delegates to
# gtest_discover_tests(); installed discovery delegates to rj_install_gtest_cases
# so parameterized TEST_P cases are expanded from the built executable.
#
# TEST_FILTER reaches BOTH paths. It used to reach only the build tree, while the
# installed generator carried its own hard-coded copy, so a case excluded here to be
# registered manually was still discovered on the installed side and ended up
# registered twice under one name.
function(rj_add_gtest_tests target)
    cmake_parse_arguments(ARG "" "TEST_FILTER" "" ${ARGN})
    gtest_discover_tests(${target} ${ARGN})
    if(RJ_INSTALL_TESTS)
        rj_install_test_target(${target})
        rj_install_gtest_cases(${target} TEST_FILTER "${ARG_TEST_FILTER}")
    endif()
endfunction()

# --- Unit / integration test suite ---

add_test(NAME RocjitsuVersion COMMAND $<TARGET_FILE:rocjitsu_bin> --version)
set_tests_properties(
    RocjitsuVersion
    PROPERTIES
        PASS_REGULAR_EXPRESSION
            "rocjitsu ${PROJECT_VERSION}[\r\n]+git revision: .+[\r\n]+git commit: .+"
)

add_executable(rj_decode_fuzz_test fuzz/decode/decode_fuzz_test.cpp)
target_link_libraries(
    rj_decode_fuzz_test
    PRIVATE rj_decode_fuzz_core GTest::gtest_main
)
rj_configure_target(rj_decode_fuzz_test PRIVATE_INCLUDES WARNINGS)
add_dependencies(rj_decode_fuzz_test rj_decode_fuzz)
rj_add_gtest_tests(rj_decode_fuzz_test)

find_package(Python3 REQUIRED COMPONENTS Interpreter)
add_test(
    NAME DecodeDifferentialUnitTest
    COMMAND
        ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/fuzz/decode/test_decode_differential.py -v
)
add_test(
    NAME DecodeSeedExtractorUnitTest
    COMMAND
        ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/fuzz/decode/test_prepare_decode_seeds.py -v
)
add_test(
    NAME DecodeFuzzCliTest
    COMMAND
        ${CMAKE_COMMAND} -E env RJ_DECODE_FUZZ=$<TARGET_FILE:rj_decode_fuzz>
        ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/fuzz/decode/test_decode_cli.py -v
)
add_test(
    NAME PretranslateScriptUnitTest
    COMMAND
        ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tools/test_pretranslate.py -v
)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(PLUGIN_LOADER_FIXTURE_DIR
        "${CMAKE_CURRENT_BINARY_DIR}/plugin_loader_fixtures"
    )
    function(rj_add_plugin_loader_fixture name)
        add_library(
            plugin_loader_fixture_${name}
            MODULE
            plugin_loader_fixture.cpp
        )
        target_include_directories(
            plugin_loader_fixture_${name}
            PRIVATE
                ${ROCJITSU_INCLUDE_DIR}
                ${ROCJITSU_SRC_DIR}
                ${HSA_INCLUDE_DIR}
                $<TARGET_PROPERTY:util,INTERFACE_INCLUDE_DIRECTORIES>
                $<TARGET_PROPERTY:simdojo,INTERFACE_INCLUDE_DIRECTORIES>
        )
        target_compile_definitions(
            plugin_loader_fixture_${name}
            PRIVATE PLUGIN_FIXTURE_NAME="${name}" ${ARGN}
        )
        set_target_properties(
            plugin_loader_fixture_${name}
            PROPERTIES
                OUTPUT_NAME rocjitsu_plugin_${name}
                LIBRARY_OUTPUT_DIRECTORY ${PLUGIN_LOADER_FIXTURE_DIR}
                CXX_VISIBILITY_PRESET hidden
                VISIBILITY_INLINES_HIDDEN ON
        )
        rj_configure_target(plugin_loader_fixture_${name} TEST)
    endfunction()

    rj_add_plugin_loader_fixture(good)
    rj_add_plugin_loader_fixture(missing PLUGIN_FIXTURE_MISSING_DESTROY=1)
    rj_add_plugin_loader_fixture(duplicate)
endif()

add_executable(
    rocjitsu_tests
    rvsim_test.cpp
    amdgpu_vm_test.cpp
    config_test.cpp
    daemon_api_test.cpp
    matmul_test.cpp
    vector_add_test.cpp
    xcd_distribution_test.cpp
    xcd_fanout_kfd_test.cpp
    xcd_shard_test.cpp
    cdna5_architecture_test.cpp
    cdna5_cluster_test.cpp
    cdna5_dispatch_memory_test.cpp
    cdna5_execution_test.cpp
    cdna5_instruction_test.cpp
    cdna5_sdma_test.cpp
    cdna5_tensor_dma_test.cpp
    lds_test.cpp
    l2_cache_test.cpp
    instruction_cache_test.cpp
    scaling_thread_counts_test.cpp
    sparse_memory_test.cpp
    simdojo_sim_test.cpp
    partitioning_test.cpp
    simulated_kfd_test.cpp
    decode_smoke_test.cpp
    decoder_c_api_test.cpp
    buffer_operand_test.cpp
    smem_sbase_operand_test.cpp
    shared_infra_test.cpp
    transcendental_test.cpp
    amdgpu_elf_test_support.cpp
    analysis/free_registers_test.cpp
    analysis/liveness_test.cpp
    patch/spill_manager_test.cpp
    patch/log_buffer_test.cpp
    patch/instruction_builder_test.cpp
    patch/kernarg_extension_test.cpp
    patch/sidecar_metadata_test.cpp
    patch/trampoline_builder_test.cpp
    patch/instrumentor_test.cpp
    patch/probe_symbol_test.cpp
    patch/probe_callable_test.cpp
    patch/probe_clobber_test.cpp
    patch/error_report_test.cpp
    dbi/dbi_spill_sim_test.cpp
    code/code_object_target_id_test.cpp
    code/amdgpu_code_object_test.cpp
    code/relocation_function_table_test.cpp
    code/kernel_descriptor_scan_test.cpp
    code/kernel_scope_test.cpp
    dbt/semantic_scratch_test.cpp
    dbt/waitcnt_translator_test.cpp
    dbt/virtual_lds_dispatch_rewrite_test.cpp
    dbt/virtual_lds_metadata_test.cpp
    dbt/encoding_translator_test.cpp
    dbt/legalization_table_test.cpp
    dbt/translate_gfx1250_test.cpp
    dbt/translate_test.cpp
    mtype_flags_test.cpp
    sysfs_test.cpp
    kfd_ioctl_test.cpp
    waitcnt_counter_test.cpp
    register_access_test.cpp
    register_file_test.cpp
    simd_correctness/mfma_lazy_storage_test.cpp
    vgpr_redispatch_test.cpp
    instruction_execution_harness_test.cpp
    isa_target_registry_test.cpp
    pseudo_scalar_exec_test.cpp
    scalar_scc_test.cpp
    cdna3_predicate_mask_test.cpp
    scoped_temp_test.cpp
    wave_debug_test.cpp
    decode_execute_benchmark.cpp
    util_observable_shared_mutex_test.cpp
    util_result_test.cpp
    util_unique_handle_test.cpp
    util_simd_test.cpp
    v_add_simd_benchmark.cpp
    mfma_simd_benchmark.cpp
    wmma_simd_benchmark.cpp
    simd_correctness/vop1_cvt_f64_scalar_correctness_test.cpp
    simd_correctness/vopc_cmp_class_scalar_correctness_test.cpp
    simd_correctness/vop2_simd_correctness_test.cpp
    simd_correctness/vop2_carry_simd_correctness_test.cpp
    simd_correctness/vop2_fma_simd_correctness_test.cpp
    simd_correctness/vop2_fma_f16_rdna_simd_correctness_test.cpp
    simd_correctness/vop2_fma_f64_simd_correctness_test.cpp
    simd_correctness/vop2_cndmask_simd_correctness_test.cpp
    simd_correctness/vop2_minmax_simd_correctness_test.cpp
    simd_correctness/vop1_simd_correctness_test.cpp
    simd_correctness/vop1_cvt_f8_simd_correctness_test.cpp
    simd_correctness/vop1_f64_simd_correctness_test.cpp
    simd_correctness/vop1_cvt_f64_simd_correctness_test.cpp
    simd_correctness/vopc_simd_correctness_test.cpp
    simd_correctness/vopc_cmp_class_simd_correctness_test.cpp
    simd_correctness/vop3_modifier_simd_test.cpp
    simd_correctness/vop3_binary_simd_correctness_test.cpp
    simd_correctness/vop3_unary_simd_correctness_test.cpp
    simd_correctness/vopc_vop3_int_simd_correctness_test.cpp
    simd_correctness/vopc_vop3_fp32_simd_correctness_test.cpp
    simd_correctness/vopc_vop3_fp16_simd_correctness_test.cpp
    simd_correctness/vopc_vop3_fp64_simd_correctness_test.cpp
    simd_correctness/vop3_ternary_int_simd_correctness_test.cpp
    simd_correctness/vop3_cvt_f64_simd_correctness_test.cpp
    simd_correctness/vop3_int_binary_extra_simd_correctness_test.cpp
    simd_correctness/vop3_int_widening_ternary_simd_correctness_test.cpp
    simd_correctness/vop3_fp64_simd_correctness_test.cpp
    simd_correctness/vop3_carry_simd_correctness_test.cpp
    simd_correctness/vop_alias_rdna_simd_correctness_test.cpp
    simd_correctness/vop3_shift64_simd_correctness_test.cpp
    simd_correctness/vop3_shift64_scalar_correctness_test.cpp
    simd_correctness/vop3_fp16_unary_simd_correctness_test.cpp
    simd_correctness/vop3_fp16_transcendental_simd_correctness_test.cpp
    simd_correctness/vop3_cndmask_pack_simd_correctness_test.cpp
    simd_correctness/vop3_div_helpers_simd_correctness_test.cpp
    simd_correctness/vop3_b16_rdna_simd_correctness_test.cpp
    simd_correctness/vop3_cvt_pk_f32_rdna_correctness_test.cpp
    simd_correctness/vop3_ternary_fp_simd_correctness_test.cpp
    simd_correctness/vop3_fmac_simd_correctness_test.cpp
    simd_correctness/vop3_ldexp_simd_correctness_test.cpp
    simd_correctness/vop3p_pk_binary_int_simd_correctness_test.cpp
    simd_correctness/vop3p_pk_binary_fp16_simd_correctness_test.cpp
    simd_correctness/vop3p_pk_inline_constant_correctness_test.cpp
    simd_correctness/vop3p_fma_mix_simd_correctness_test.cpp
    simd_correctness/vop3p_mov_b32_simd_correctness_test.cpp
    simd_correctness/vop3p_dot_simd_correctness_test.cpp
    execution_plugin_test.cpp
    plugin_config_resolver_test.cpp
    plugin_loader_test.cpp
    race-detector/race_detector_event_registry_test.cpp
    race-detector/race_detector_tests.cpp
    race-detector/interval_set_tests.cpp
    race-detector/race_log_expectation_test.cpp
    big_int_test.cpp
    data_types_test.cpp
    drm_info_layout_test.cpp
    pci/bar_access_trace_test.cpp
    pci/gpu_pci_device_test.cpp
    pci/scratch_pci_device_test.cpp
    pci/ip_discovery_test.cpp
)
if(HAS_DEVICE_KERNELS)
    target_sources(
        rocjitsu_tests
        PRIVATE
            dbt/device_kernels_translate_tests.cpp
            dbt/multikernel_indirect_branch_test.cpp
    )
endif()
if(HAS_GFX1250_DEVICE_KERNELS)
    target_sources(
        rocjitsu_tests
        PRIVATE dbt/relocation_function_table_hip_test.cpp
    )
    target_compile_definitions(
        rocjitsu_tests
        PRIVATE HAS_GFX1250_DEVICE_KERNELS=1
    )
endif()
# The vfio-user backend is optional, so its tests build only when it does.
if(ROCJITSU_ENABLE_VFIO)
    target_sources(
        rocjitsu_tests
        PRIVATE
            vmm/bus_plan_test.cpp
            vmm/guest_region_map_test.cpp
            vmm/vfio_user_client.cpp
            vmm/vfio_device_host_test.cpp
    )
    target_link_libraries(
        rocjitsu_tests
        PRIVATE rocjitsu_vmm_vfu libvfio-user::libvfio-user
    )
endif()
if(RJ_ENABLE_EXPENSIVE_CHECKS)
    target_sources(
        rocjitsu_tests
        PRIVATE
            simd_correctness/mfma_simd_exact_test.cpp
            simd_correctness/wmma_simd_exact_test.cpp
    )
endif()
target_link_libraries(
    rocjitsu_tests
    PRIVATE
        ${RJ_OBJECT_LIBS}
        rocjitsu_kmd
        util
        flatbuffers
        Threads::Threads
        GTest::gmock
        GTest::gtest_main
)
target_include_directories(rocjitsu_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
rj_configure_target(rocjitsu_tests TEST)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_compile_definitions(
        rocjitsu_tests
        PRIVATE PLUGIN_LOADER_FIXTURE_DIR="${PLUGIN_LOADER_FIXTURE_DIR}"
    )
    add_dependencies(
        rocjitsu_tests
        plugin_loader_fixture_good
        plugin_loader_fixture_missing
        plugin_loader_fixture_duplicate
    )
endif()
# Benchmarks are timing tools, not correctness tests; keep them out of the
# default ctest run (invoke them directly via
# ./rocjitsu_tests --gtest_filter='*Benchmark*'). The request-exit stress test
# and the concurrent-close wait test are registered separately so CTest can
# bound a deadlocked child process.
rj_add_gtest_tests(
    rocjitsu_tests
    TEST_FILTER
        "-*Benchmark*:*RequestExitWakesAllPartitions*:*DispatchedWaitSurvivesAConcurrentFinalClose*"
)

# RequestExitWakesAllPartitions may require several runs to show regression
# (deadlock). Run 30 times with a 10 second timeout to boost chances of seeing
# the failures. Usually fails within the first 5 times and 30 passing tests
# takes about 2.5s in CI.
rj_add_test(
    NAME "TerminationTest.RequestExitWakesAllPartitions"
    COMMAND
        $<TARGET_FILE:rocjitsu_tests>
        "--gtest_filter=TerminationTest.RequestExitWakesAllPartitions"
        "--gtest_repeat=30"
    TIMEOUT 10
    INSTALL_COMMAND
        "bin/rocjitsu_tests"
        "--gtest_filter=TerminationTest.RequestExitWakesAllPartitions"
        "--gtest_repeat=30"
)

# DispatchedWaitSurvivesAConcurrentFinalClose races an indefinite WAIT_EVENTS
# against a final close. Its in-test recovery wake bounds a close that blocks
# while releasing the driver mutex, but nothing in-process can bound one that
# blocks while HOLDING it -- releasing the wait's lease needs that same mutex.
# CTest is the only backstop there, so register it with an explicit timeout.
rj_add_test(
    NAME "SimulatedKfdTest.DispatchedWaitSurvivesAConcurrentFinalClose"
    COMMAND
        $<TARGET_FILE:rocjitsu_tests>
        "--gtest_filter=SimulatedKfdTest.DispatchedWaitSurvivesAConcurrentFinalClose"
    TIMEOUT 30
    INSTALL_COMMAND
        "bin/rocjitsu_tests"
        "--gtest_filter=SimulatedKfdTest.DispatchedWaitSurvivesAConcurrentFinalClose"
)

# Exercise the Linux-only model ISA contract in a decoder-only final link.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_executable(rocjitsu_model_only_test model_only_isa_test.cpp)
    target_include_directories(
        rocjitsu_model_only_test
        PRIVATE ${ROCJITSU_INCLUDE_DIR} ${ROCJITSU_SRC_DIR}
    )
    target_link_libraries(
        rocjitsu_model_only_test
        PRIVATE
            rocjitsu_dbt_internal
            rocjitsu_cdna5_model_registry
            simdojo_headers
            GTest::gtest_main
    )
    rj_configure_target(rocjitsu_model_only_test TEST)
    rj_add_gtest_tests(rocjitsu_model_only_test)
    add_test(
        NAME ModelOnlyIsaTest.SymbolBoundary
        COMMAND
            ${CMAKE_COMMAND} -DNM=${CMAKE_NM}
            -DMODEL_BINARY=$<TARGET_FILE:rocjitsu_model_only_test> -P
            ${CMAKE_CURRENT_SOURCE_DIR}/check_model_only_symbols.cmake
    )

    add_executable(
        gfx1250_b0_to_a0_library_test
        dbt/gfx1250_b0_to_a0_library_test.cpp
    )
    target_link_libraries(
        gfx1250_b0_to_a0_library_test
        PRIVATE rocjitsu_gfx1250_b0_to_a0 GTest::gtest_main
    )
    if(HAS_GFX1250_DBT_FIXTURE)
        target_compile_definitions(
            gfx1250_b0_to_a0_library_test
            PRIVATE
                GFX1250_B0_TO_A0_FIXTURE="${KERNEL_OUTPUT_DIR}/relocation_function_table_dispatch_gfx1250.hsaco"
        )
        add_dependencies(
            gfx1250_b0_to_a0_library_test
            probe_relocation_function_table_dispatch_gfx1250
        )
    endif()
    rj_configure_target(
        gfx1250_b0_to_a0_library_test
        PRIVATE_INCLUDES
        WARNINGS
    )
    rj_add_gtest_tests(gfx1250_b0_to_a0_library_test)
    # Checked on the translator rather than the hook. The hook used to link the
    # translator as a static archive and so carried the model code itself; now it
    # loads a shared object, and the boundary this guards -- model in, execution
    # out -- belongs to whichever artifact holds that code. Pointed at the hook it
    # would now pass vacuously, which is the failure mode the required-symbol
    # check at the end of the script exists to catch.
    add_test(
        NAME Gfx1250B0ToA0Library.SymbolBoundary
        COMMAND
            ${CMAKE_COMMAND} -DNM=${CMAKE_NM}
            -DMODEL_BINARY=$<TARGET_FILE:rocjitsu_gfx1250_b0_to_a0> -P
            ${CMAKE_CURRENT_SOURCE_DIR}/check_model_only_symbols.cmake
    )

    # This source-integrated NPI fixture is deliberately absent from the
    # built-in provider list. It exercises per-consumer composition by canonical
    # name; it does not model an open plugin ABI or a complete new architecture.
    rj_add_object_library(
        rocjitsu_downstream_isa_fixture
        fixtures/downstream_isa_target_provider.cpp
    )
    target_include_directories(
        rocjitsu_downstream_isa_fixture
        PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/fixtures
    )
    rj_add_isa_target_provider(
        rocjitsu_downstream_isa_fixture
        HEADER downstream_isa_target_provider.h
    )
    rj_add_isa_target_registry(
        rocjitsu_downstream_registry_fixture
        PROVIDERS rocjitsu_downstream_isa_fixture
    )
    add_library(
        rocjitsu_downstream_isa_registry_fixture
        SHARED
        fixtures/downstream_isa_registry_fixture.cpp
    )
    target_link_libraries(
        rocjitsu_downstream_isa_registry_fixture
        PRIVATE
            rocjitsu_isa
            rocjitsu_downstream_registry_fixture
            simdojo_headers
    )
    rj_configure_target(rocjitsu_downstream_isa_registry_fixture LIB)

    # These shared objects model independently composed final images. The
    # driver links the all-target registry while each DSO retains its own
    # default subset.
    add_library(
        rocjitsu_narrow_isa_registry_fixture
        SHARED
        fixtures/narrow_isa_registry_fixture.cpp
    )
    target_link_libraries(
        rocjitsu_narrow_isa_registry_fixture
        PRIVATE
            rocjitsu_dbt_internal
            rocjitsu_cdna5_model_registry
            simdojo_headers
    )
    rj_configure_target(rocjitsu_narrow_isa_registry_fixture LIB)

    add_executable(
        isa_registry_composition_test
        isa_registry_composition_test.cpp
    )
    target_link_libraries(
        isa_registry_composition_test
        PRIVATE
            ${RJ_OBJECT_LIBS}
            rocjitsu_downstream_isa_registry_fixture
            rocjitsu_narrow_isa_registry_fixture
            util
            flatbuffers
            Threads::Threads
            GTest::gtest_main
    )
    rj_configure_target(isa_registry_composition_test TEST)
    rj_add_gtest_tests(isa_registry_composition_test)
    add_test(
        NAME IsaRegistryCompositionTest.NarrowSymbolBoundary
        COMMAND
            ${CMAKE_COMMAND} -DNM=${CMAKE_NM}
            -DNARROW_LIBRARY=$<TARGET_FILE:rocjitsu_narrow_isa_registry_fixture>
            -P ${CMAKE_CURRENT_SOURCE_DIR}/check_narrow_registry_symbols.cmake
    )
endif()

# --- CLI launcher preload tests ---

add_executable(launch_preload_test launch_preload_test.cpp)
target_link_libraries(
    launch_preload_test
    PRIVATE rocjitsu_cli_launch_preload GTest::gtest_main
)
if(RJ_ENABLE_ASAN AND RJ_SANITIZER_USE_SHARED)
    target_compile_definitions(
        launch_preload_test
        PRIVATE
            RJ_EXPECT_SHARED_ASAN_RUNTIME=1
            RJ_EXPECTED_SHARED_ASAN_RUNTIME="${RJ_ASAN_SHARED_LIBRARY}"
    )
endif()
if(RJ_ENABLE_TSAN AND RJ_SANITIZER_USE_SHARED)
    target_compile_definitions(
        launch_preload_test
        PRIVATE
            RJ_EXPECT_SHARED_TSAN_RUNTIME=1
            RJ_EXPECTED_SHARED_TSAN_RUNTIME="${RJ_TSAN_SHARED_LIBRARY}"
    )
endif()
rj_configure_target(launch_preload_test TEST)
rj_add_gtest_tests(launch_preload_test)

# --- CLI ROCm visibility tests ---
# rocm_visibility.cpp ships in the launch-preload library for build convenience,
# but its tests get their own executable so they mirror the source structure.

add_executable(rocm_visibility_test rocm_visibility_test.cpp)
target_link_libraries(
    rocm_visibility_test
    PRIVATE rocjitsu_cli_launch_preload GTest::gtest_main
)
rj_configure_target(rocm_visibility_test TEST)
rj_add_gtest_tests(rocm_visibility_test)

# --- DBI probe fixture test ---
# Loads a real amdclang++-compiled rj_nop_probe device ELF and exercises the
# probe symbol resolver against it. Needs the compiled fixture (amdclang++ +
# clang-offload-bundler) but no GPU, so it registers with CTest unconditionally
# when the fixture is available.
if(HAS_DEVICE_KERNELS AND HAS_PROBE_FIXTURES)
    add_executable(probe_fixture_test dbi/probe_fixture_test.cpp)
    target_include_directories(
        probe_fixture_test
        PRIVATE
            ${CMAKE_SOURCE_DIR}/lib/rocjitsu/include
            ${CMAKE_SOURCE_DIR}/lib/rocjitsu/src
    )
    target_link_libraries(
        probe_fixture_test
        PRIVATE
            ${RJ_OBJECT_LIBS}
            util
            flatbuffers
            Threads::Threads
            GTest::gtest_main
    )
    target_compile_definitions(
        probe_fixture_test
        PRIVATE KERNEL_DIR="${KERNEL_OUTPUT_DIR}"
    )
    add_dependencies(
        probe_fixture_test
        probe_rj_nop_probe_gfx90a
        probe_callable_sgpr_probe_gfx950
    )
    rj_configure_target(probe_fixture_test TEST)
    if(RJ_INSTALL_TESTS)
        rj_install_test_target(probe_fixture_test)
    endif()
    rj_add_test(
        NAME "ProbeFixture.NopProbeResolvesAndReturns"
        COMMAND probe_fixture_test
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        INSTALL_COMMAND "bin/probe_fixture_test"
    )
    rj_add_test(
        NAME "ProbeFixture.CallableSgprUseExceedsKernelDescriptor"
        COMMAND
            probe_fixture_test
            --gtest_filter=ProbeFixture.CallableSgprUseExceedsKernelDescriptor
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        INSTALL_COMMAND
            "bin/probe_fixture_test"
            "--gtest_filter=ProbeFixture.CallableSgprUseExceedsKernelDescriptor"
    )
endif()

# Pin the zero-test failure behavior independently of GPU availability. This is
# deliberately a negative CMake-script test: success means the guard rejected
# the nonexistent filter before a launcher could report a false pass.
add_test(
    NAME GtestFilterGuard.RejectsMissingTest
    COMMAND
        ${CMAKE_COMMAND} "-DTEST_EXECUTABLE=$<TARGET_FILE:rocjitsu_tests>"
        "-DTEST_FILTER=DefinitelyMissingSuite.DefinitelyMissingTest"
        -DLIST_ONLY=ON -P
        "${PROJECT_SOURCE_DIR}/cmake/rj_run_filtered_gtest.cmake"
)
set_tests_properties(
    GtestFilterGuard.RejectsMissingTest
    PROPERTIES WILL_FAIL TRUE
)
add_test(
    NAME GtestFilterGuard.AcceptsExistingTest
    COMMAND
        ${CMAKE_COMMAND} "-DTEST_EXECUTABLE=$<TARGET_FILE:rocjitsu_tests>"
        "-DTEST_FILTER=SidecarMetadata.HeaderBytesPinMagicAndVersion"
        -DLIST_ONLY=ON -P
        "${PROJECT_SOURCE_DIR}/cmake/rj_run_filtered_gtest.cmake"
)

add_executable(
    rj_dbt_translate_smoke_test
    tools/rj_dbt_translate_smoke_test.cpp
)
target_include_directories(
    rj_dbt_translate_smoke_test
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/lib/rocjitsu/include
        ${CMAKE_SOURCE_DIR}/lib/rocjitsu/src
)
target_link_libraries(
    rj_dbt_translate_smoke_test
    PRIVATE rocjitsu_translate_cli GTest::gtest
)
rj_configure_target(rj_dbt_translate_smoke_test TEST)
add_dependencies(
    rj_dbt_translate_smoke_test
    rj_dbt_translate
    rj_decode_benchmark
)
if(RJ_INSTALL_TESTS)
    rj_install_test_target(rj_dbt_translate_smoke_test)
    rj_install_test_target(rj_dbt_translate)
    rj_install_test_target(rj_decode_benchmark)
endif()
rj_add_test(
    NAME "RjDbtTranslate.Smoke"
    COMMAND rj_dbt_translate_smoke_test $<TARGET_FILE:rj_dbt_translate>
            $<TARGET_FILE:rj_decode_benchmark>
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    INSTALL_COMMAND
        "bin/rj_dbt_translate_smoke_test"
        "bin/rj_dbt_translate"
        "bin/rj_decode_benchmark"
)

# --- Scaling test (standalone executable) ---

add_executable(scaling_test scaling_test.cpp)
target_link_libraries(
    scaling_test
    PRIVATE ${RJ_OBJECT_LIBS} util flatbuffers Threads::Threads
)
target_compile_options(scaling_test PRIVATE -O2)
rj_configure_target(scaling_test TEST)

# --- Interposer tests run through the rocjitsu CLI launcher ---
# All tests that need the simulated GPU go through:
#   rocjitsu --config <cfg> -- <test_binary> [args]
# The CLI handles LD_PRELOAD and config discovery. rj_add_test gives launcher
# tests isolated runtime directories, so no RESOURCE_LOCK is needed.

set(RJ_CDNA4_KMD_CONFIG "${CMAKE_SOURCE_DIR}/configs/gfx950_mi355x_kmd.json")
set(RJ_CDNA4_KMD_2GPU_CONFIG
    "${CMAKE_SOURCE_DIR}/configs/gfx950_mi355x_kmd_2gpu.json"
)
set(RJ_CDNA3_KMD_CONFIG "${CMAKE_SOURCE_DIR}/configs/gfx942_cdna3_kmd.json")
set(RJ_CDNA2_KMD_CONFIG "${CMAKE_SOURCE_DIR}/configs/gfx90a_mi210_kmd.json")
set(RJ_CDNA4_TO_CDNA3_SIM_DBT_CONFIG
    "${CMAKE_SOURCE_DIR}/configs/guest_gfx950_on_simulated_gfx942.json"
)
set(RJ_HIP_TEST_ARCH
    "gfx950"
    CACHE STRING
    "GPU ISA used to build basic HIP smoke tests"
)
set(RJ_HIP_TEST_CONFIG
    "${RJ_CDNA4_KMD_CONFIG}"
    CACHE FILEPATH
    "rocjitsu config used by basic HIP smoke tests"
)
set(RJ_HIP_TEST_2GPU_CONFIG
    "${RJ_CDNA4_KMD_2GPU_CONFIG}"
    CACHE FILEPATH
    "rocjitsu config used by basic two-GPU HIP tests"
)
get_filename_component(RJ_CDNA4_KMD_CONFIG_NAME "${RJ_CDNA4_KMD_CONFIG}" NAME)
get_filename_component(RJ_CDNA3_KMD_CONFIG_NAME "${RJ_CDNA3_KMD_CONFIG}" NAME)
get_filename_component(RJ_CDNA2_KMD_CONFIG_NAME "${RJ_CDNA2_KMD_CONFIG}" NAME)
get_filename_component(
    RJ_CDNA4_TO_CDNA3_SIM_DBT_CONFIG_NAME
    "${RJ_CDNA4_TO_CDNA3_SIM_DBT_CONFIG}"
    NAME
)
get_filename_component(
    RJ_CDNA4_KMD_2GPU_CONFIG_NAME
    "${RJ_CDNA4_KMD_2GPU_CONFIG}"
    NAME
)
get_filename_component(RJ_HIP_TEST_CONFIG_NAME "${RJ_HIP_TEST_CONFIG}" NAME)
get_filename_component(
    RJ_HIP_TEST_2GPU_CONFIG_NAME
    "${RJ_HIP_TEST_2GPU_CONFIG}"
    NAME
)
set(RJ_CLI "$<TARGET_FILE:rocjitsu_bin>")

function(rj_detect_kfd_gfx_target_versions out_var)
    set(_versions)
    foreach(
        _nodes_dir
        IN
        ITEMS
            "/sys/devices/virtual/kfd/kfd/topology/nodes"
            "/sys/class/kfd/kfd/topology/nodes"
    )
        if(NOT IS_DIRECTORY "${_nodes_dir}")
            continue()
        endif()

        file(GLOB _node_dirs LIST_DIRECTORIES true "${_nodes_dir}/*")
        foreach(_node_dir IN LISTS _node_dirs)
            if(NOT IS_DIRECTORY "${_node_dir}")
                continue()
            endif()
            if(NOT EXISTS "${_node_dir}/properties")
                continue()
            endif()

            # DBT guest configs name a concrete host ISA. Read KFD topology at
            # configure time so hardware-backed tests are not registered on
            # hosts where /dev/kfd exists but the required ISA is absent.
            file(READ "${_node_dir}/properties" _properties)
            if(_properties MATCHES "(^|[\r\n])gfx_target_version[ \t]+([0-9]+)")
                list(APPEND _versions "${CMAKE_MATCH_2}")
            endif()
        endforeach()
    endforeach()
    if(_versions)
        list(REMOVE_DUPLICATES _versions)
    endif()
    set(${out_var} "${_versions}" PARENT_SCOPE)
endfunction()

rj_detect_kfd_gfx_target_versions(RJ_KFD_GFX_TARGET_VERSIONS)
set(RJ_HAS_KFD_DEVICE FALSE)
if(EXISTS "/dev/kfd")
    set(RJ_HAS_KFD_DEVICE TRUE)
endif()
set(RJ_HAS_GFX1201_GPU FALSE)
if("120001" IN_LIST RJ_KFD_GFX_TARGET_VERSIONS)
    set(RJ_HAS_GFX1201_GPU TRUE)
endif()

set(RJ_SANITIZER_PRELOAD_PREFIX "")
if(RJ_ASAN_SHARED_LIBRARY)
    set(RJ_SANITIZER_PRELOAD_PREFIX "${RJ_ASAN_SHARED_LIBRARY}:")
elseif(RJ_TSAN_SHARED_LIBRARY)
    set(RJ_SANITIZER_PRELOAD_PREFIX "${RJ_TSAN_SHARED_LIBRARY}:")
endif()

set(RJ_KMD_PRELOAD_ENV
    "LD_PRELOAD=${RJ_SANITIZER_PRELOAD_PREFIX}$<TARGET_FILE:rocjitsu_shared>"
    "RJ_CONFIG=${CMAKE_SOURCE_DIR}/configs/gfx950_mi355x.json"
)

add_executable(guest_kfd_test guest_kfd_test.cpp)
target_include_directories(
    guest_kfd_test
    PRIVATE ${CMAKE_SOURCE_DIR}/lib/rocjitsu/include ${HSA_INCLUDE_DIR}
)
target_link_libraries(guest_kfd_test PRIVATE Threads::Threads GTest::gtest_main)
rj_configure_target(guest_kfd_test TEST)
add_dependencies(guest_kfd_test rocjitsu_bin rocjitsu_shared rocjitsu_hooks)
rj_add_test(
    NAME "GuestKfdFailureTest.NonexistentSimulatorConfigFailsCleanly"
    COMMAND
        ${RJ_CLI} --config ${RJ_CDNA3_KMD_CONFIG} --
        $<TARGET_FILE:guest_kfd_test>
        --gtest_filter=GuestKfdFailureTest.NonexistentSimulatorConfigFailsCleanly
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    TIMEOUT 60
    INSTALL_COMMAND
        "\${RJ_INSTALLED_BINDIR}/rocjitsu"
        "--config"
        "\${RJ_INSTALLED_CONFIG_DIR}/${RJ_CDNA3_KMD_CONFIG_NAME}"
        "--"
        "bin/guest_kfd_test"
        "--gtest_filter=GuestKfdFailureTest.NonexistentSimulatorConfigFailsCleanly"
)
rj_add_test(
    NAME "GuestKfdFailureTest.SimulatorHostIsaMismatchFailsCleanly"
    COMMAND
        ${RJ_CLI} --config ${RJ_CDNA3_KMD_CONFIG} --
        $<TARGET_FILE:guest_kfd_test>
        --gtest_filter=GuestKfdFailureTest.SimulatorHostIsaMismatchFailsCleanly
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    TIMEOUT 60
    INSTALL_COMMAND
        "\${RJ_INSTALLED_BINDIR}/rocjitsu"
        "--config"
        "\${RJ_INSTALLED_CONFIG_DIR}/${RJ_CDNA3_KMD_CONFIG_NAME}"
        "--"
        "bin/guest_kfd_test"
        "--gtest_filter=GuestKfdFailureTest.SimulatorHostIsaMismatchFailsCleanly"
)
rj_add_test(
    NAME "GuestKfdFailureTest.SimulatorOversizedLdsFailsCleanly"
    COMMAND
        ${RJ_CLI} --config ${RJ_CDNA3_KMD_CONFIG} --
        $<TARGET_FILE:guest_kfd_test>
        --gtest_filter=GuestKfdFailureTest.SimulatorOversizedLdsFailsCleanly
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    TIMEOUT 60
    INSTALL_COMMAND
        "\${RJ_INSTALLED_BINDIR}/rocjitsu"
        "--config"
        "\${RJ_INSTALLED_CONFIG_DIR}/${RJ_CDNA3_KMD_CONFIG_NAME}"
        "--"
        "bin/guest_kfd_test"
        "--gtest_filter=GuestKfdFailureTest.SimulatorOversizedLdsFailsCleanly"
)
rj_add_test(
    NAME "GuestKfdFailureTest.PathOnlyAutomaticDbtHandoffFailsClosed"
    COMMAND
        ${RJ_CLI} --config ${RJ_CDNA3_KMD_CONFIG} --
        $<TARGET_FILE:guest_kfd_test>
        --gtest_filter=GuestKfdFailureTest.PathOnlyAutomaticDbtHandoffFailsClosed
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    TIMEOUT 60
    INSTALL_COMMAND
        "\${RJ_INSTALLED_BINDIR}/rocjitsu"
        "--config"
        "\${RJ_INSTALLED_CONFIG_DIR}/${RJ_CDNA3_KMD_CONFIG_NAME}"
        "--"
        "bin/guest_kfd_test"
        "--gtest_filter=GuestKfdFailureTest.PathOnlyAutomaticDbtHandoffFailsClosed"
)
rj_add_test(
    NAME "GuestKfdConfigTest.ReadsRuntimeHandoffLargerThan4095Bytes"
    COMMAND
        ${RJ_CLI} --config ${RJ_CDNA3_KMD_CONFIG} --
        $<TARGET_FILE:guest_kfd_test>
        --gtest_filter=GuestKfdConfigTest.ReadsRuntimeHandoffLargerThan4095Bytes
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    TIMEOUT 60
    # The test skips when the runtime directory it is handed is already deeper than
    # the >4095-byte path it needs to build. Without this the skip would be reported
    # as a pass and the lost coverage would be invisible.
    SKIP_REGULAR_EXPRESSION "\\[  SKIPPED \\]"
    INSTALL_COMMAND
        "\${RJ_INSTALLED_BINDIR}/rocjitsu"
        "--config"
        "\${RJ_INSTALLED_CONFIG_DIR}/${RJ_CDNA3_KMD_CONFIG_NAME}"
        "--"
        "bin/guest_kfd_test"
        "--gtest_filter=GuestKfdConfigTest.ReadsRuntimeHandoffLargerThan4095Bytes"
)
# Pure harness logic: no guest device, so no launcher wrapper.
rj_add_test(
    NAME "GuestKfdConfigTest.LongPathHandoffSkipsWhenRuntimeDirIsTooDeep"
    COMMAND
        $<TARGET_FILE:guest_kfd_test>
        --gtest_filter=GuestKfdConfigTest.LongPathHandoffSkipsWhenRuntimeDirIsTooDeep
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    TIMEOUT 60
    INSTALL_COMMAND
        "bin/guest_kfd_test"
        "--gtest_filter=GuestKfdConfigTest.LongPathHandoffSkipsWhenRuntimeDirIsTooDeep"
)
if(RJ_HAS_GFX1201_GPU)
    add_test(
        NAME
            "GuestKfdMultiprocessTest.ForkedChildrenRefuseGpuAndLeaveParentOverlayIntact"
        COMMAND
            ${RJ_CLI} --config
            ${CMAKE_SOURCE_DIR}/configs/guest_gfx950_on_gfx1201.json --
            $<TARGET_FILE:guest_kfd_test>
            --gtest_filter=GuestKfdMultiprocessTest.ForkedChildrenRefuseGpuAndLeaveParentOverlayIntact
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
    add_test(
        NAME "GuestKfdMultiprocessTest.ForkedChildCannotUseInheritedGuestDriver"
        COMMAND
            ${RJ_CLI} --config
            ${CMAKE_SOURCE_DIR}/configs/guest_gfx950_on_gfx1201.json --
            $<TARGET_FILE:guest_kfd_test>
            --gtest_filter=GuestKfdMultiprocessTest.ForkedChildCannotUseInheritedGuestDriver
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
    add_test(
        NAME "GuestKfdConcurrencyTest.ConcurrentOpenIoctlAndSysfsRedirect"
        COMMAND
            ${RJ_CLI} --config
            ${CMAKE_SOURCE_DIR}/configs/guest_gfx950_on_gfx1201.json --
            $<TARGET_FILE:guest_kfd_test>
            --gtest_filter=GuestKfdConcurrencyTest.ConcurrentOpenIoctlAndSysfsRedirect
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
    add_test(
        NAME "GuestKfdLifecycleTest.ReopensAfterLastClose"
        COMMAND
            ${RJ_CLI} --config
            ${CMAKE_SOURCE_DIR}/configs/guest_gfx950_on_gfx1201.json --
            $<TARGET_FILE:guest_kfd_test>
            --gtest_filter=GuestKfdLifecycleTest.ReopensAfterLastClose
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
    add_test(
        NAME "GuestKfdMemoryTest.GuestAllocationMmapOffsetIsRejected"
        COMMAND
            ${RJ_CLI} --config
            ${CMAKE_SOURCE_DIR}/configs/guest_gfx950_on_gfx1201.json --
            $<TARGET_FILE:guest_kfd_test>
            --gtest_filter=GuestKfdMemoryTest.GuestAllocationMmapOffsetIsRejected
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
    add_test(
        NAME "GuestKfdFdReuseTest.OverwritingHiddenRealFdKeepsRoutingCorrect"
        COMMAND
            ${RJ_CLI} --config
            ${CMAKE_SOURCE_DIR}/configs/guest_gfx950_on_gfx1201.json --
            $<TARGET_FILE:guest_kfd_test>
            --gtest_filter=GuestKfdFdReuseTest.OverwritingHiddenRealFdKeepsRoutingCorrect
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
    set_tests_properties(
        "GuestKfdMultiprocessTest.ForkedChildrenRefuseGpuAndLeaveParentOverlayIntact"
        "GuestKfdMultiprocessTest.ForkedChildCannotUseInheritedGuestDriver"
        "GuestKfdConcurrencyTest.ConcurrentOpenIoctlAndSysfsRedirect"
        "GuestKfdLifecycleTest.ReopensAfterLastClose"
        "GuestKfdMemoryTest.GuestAllocationMmapOffsetIsRejected"
        "GuestKfdFdReuseTest.OverwritingHiddenRealFdKeepsRoutingCorrect"
        PROPERTIES TIMEOUT 60 SKIP_REGULAR_EXPRESSION "\\[  SKIPPED \\]"
    )
    foreach(
        _guest_kfd_test
        IN
        ITEMS
            "GuestKfdMultiprocessTest.ForkedChildrenRefuseGpuAndLeaveParentOverlayIntact"
            "GuestKfdMultiprocessTest.ForkedChildCannotUseInheritedGuestDriver"
            "GuestKfdConcurrencyTest.ConcurrentOpenIoctlAndSysfsRedirect"
            "GuestKfdLifecycleTest.ReopensAfterLastClose"
            "GuestKfdMemoryTest.GuestAllocationMmapOffsetIsRejected"
            "GuestKfdFdReuseTest.OverwritingHiddenRealFdKeepsRoutingCorrect"
    )
        rj_set_default_runtime_dir("${_guest_kfd_test}")
    endforeach()
    message(
        STATUS
        "GuestKfd DBT guest tests enabled (gfx1201 host GPU detected)"
    )
else()
    message(
        STATUS
        "GuestKfd DBT guest tests built but NOT registered with CTest "
        "(no gfx1201 KFD host GPU)"
    )
endif()

add_executable(
    hsa_hooks_unit_test
    dbt/hsa_hooks_unit_test.cpp
    dbt/sidecar_registry_test.cpp
    ${CMAKE_SOURCE_DIR}/lib/rocjitsu/src/rocjitsu/hooks/sidecar_registry.cpp
)
target_compile_definitions(
    hsa_hooks_unit_test
    PRIVATE
        RJ_HOOK_UNIT_CONFIG_PATH="${CMAKE_SOURCE_DIR}/configs/guest_gfx950_on_gfx1201.json"
)
target_include_directories(
    hsa_hooks_unit_test
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/lib/rocjitsu/include
        ${CMAKE_SOURCE_DIR}/lib/rocjitsu/src
        ${CMAKE_BINARY_DIR}/include
        ${GENERATED_DIR}
)
target_include_directories(
    hsa_hooks_unit_test
    SYSTEM
    PRIVATE ${HSA_INCLUDE_DIR}
)
target_link_libraries(
    hsa_hooks_unit_test
    PRIVATE
        # Test-only hook variant: same objects as rocjitsu_hooks plus the
        # exported rj_hsa_dbt_set_topology_nodes_root_for_test seam.
        rocjitsu_hooks_testing
        ${RJ_OBJECT_LIBS}
        util
        flatbuffers
        Threads::Threads
        GTest::gtest_main
)
rj_configure_target(hsa_hooks_unit_test TEST)
gtest_discover_tests(hsa_hooks_unit_test)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_executable(
        hsa_hotswap_rocjitsu_unit_test
        dbt/hsa_hotswap_rocjitsu_unit_test.cpp
    )
    target_include_directories(
        hsa_hotswap_rocjitsu_unit_test
        PRIVATE
            ${CMAKE_SOURCE_DIR}/lib/rocjitsu/include
            ${CMAKE_SOURCE_DIR}/lib/rocjitsu/src
    )
    target_include_directories(
        hsa_hotswap_rocjitsu_unit_test
        SYSTEM
        PRIVATE ${HSA_INCLUDE_DIR}
    )
    target_link_libraries(
        hsa_hotswap_rocjitsu_unit_test
        PRIVATE hsa_hotswap_rocjitsu_testable GTest::gtest_main
    )
    if(HAS_GFX1250_DBT_FIXTURE)
        target_compile_definitions(
            hsa_hotswap_rocjitsu_unit_test
            PRIVATE
                GFX1250_B0_TO_A0_FIXTURE="${KERNEL_OUTPUT_DIR}/relocation_function_table_dispatch_gfx1250.hsaco"
        )
        add_dependencies(
            hsa_hotswap_rocjitsu_unit_test
            probe_relocation_function_table_dispatch_gfx1250
        )
        # The pre-translated tier is filled by a separate program and read by the
        # hook. Running the real tool is the only way to test that the two agree
        # on a key; writing entries through a test seam would exercise the hook's
        # own derivation twice and prove nothing about the pair.
        target_compile_definitions(
            hsa_hotswap_rocjitsu_unit_test
            PRIVATE RJ_PRETRANSLATE_TOOL="$<TARGET_FILE:rj_pretranslate>"
        )
        add_dependencies(hsa_hotswap_rocjitsu_unit_test rj_pretranslate)
    endif()
    rj_configure_target(hsa_hotswap_rocjitsu_unit_test TEST)
    gtest_discover_tests(hsa_hotswap_rocjitsu_unit_test)

    # Compiles the translation store as a plain DBT component rather than
    # reaching it through a hook, which is how a second consumer would use it.
    add_executable(
        translation_store_test
        dbt/translation_store_test.cpp
        ${CMAKE_SOURCE_DIR}/lib/rocjitsu/src/rocjitsu/code/dbt/translation_store.cpp
    )
    target_include_directories(
        translation_store_test
        PRIVATE
            ${CMAKE_SOURCE_DIR}/lib/rocjitsu/include
            ${CMAKE_SOURCE_DIR}/lib/rocjitsu/src
    )
    target_compile_definitions(
        translation_store_test
        PRIVATE RJ_TRANSLATION_STORE_TEST_HOOKS
    )
    # The store keys entries on this note and disables itself without it.
    target_link_options(translation_store_test PRIVATE "LINKER:--build-id=sha1")
    target_link_libraries(
        translation_store_test
        PRIVATE GTest::gtest_main ${CMAKE_DL_LIBS}
    )
    rj_configure_target(translation_store_test TEST)
    gtest_discover_tests(translation_store_test)

    # One source, two modules differing only in build id -- which is what a
    # rebuilt translator is. Entries written against one must be invisible to the
    # other, and no single binary can show that: the store derives identity from
    # the module containing an address, so two identities need two modules.
    #
    # The ids are fixed rather than content-derived so this stays a test of the
    # store rather than of the linker. Content-derived ids would also differ, but
    # only incidentally, and would churn whenever either file was touched.
    foreach(_probe_suffix a b)
        set(_probe rj_translator_build_probe_${_probe_suffix})
        add_library(${_probe} MODULE dbt/translator_build_probe.cpp)
        rj_configure_target(${_probe} TEST)
        target_compile_definitions(
            translation_store_test
            PRIVATE
                RJ_TRANSLATOR_PROBE_${_probe_suffix}="$<TARGET_FILE:${_probe}>"
        )
        add_dependencies(translation_store_test ${_probe})
    endforeach()
    target_link_options(
        rj_translator_build_probe_a
        PRIVATE "LINKER:--build-id=0xaaaaaaaaaaaaaaaa"
    )
    target_link_options(
        rj_translator_build_probe_b
        PRIVATE "LINKER:--build-id=0xbbbbbbbbbbbbbbbb"
    )
endif()

# A separate loadable module is the only way to observe the force-scalar gate's
# per-module scope: the gate is an inline variable with hidden visibility, so
# nothing linked into the test executable can see the boundary. The test dlopens
# this and compares what it reports against its own copy.
add_library(util_force_scalar_probe MODULE util_force_scalar_probe.cpp)
target_link_libraries(util_force_scalar_probe PRIVATE util)
rj_configure_target(util_force_scalar_probe TEST)
add_dependencies(rocjitsu_tests util_force_scalar_probe)
target_compile_definitions(
    rocjitsu_tests
    PRIVATE
        RJ_FORCE_SCALAR_PROBE_MODULE="$<TARGET_FILE:util_force_scalar_probe>"
)

# --- Interposer loader initialization regression tests ---

add_library(early_mmap_preload SHARED early_mmap_preload_test.cpp)
rj_configure_target(early_mmap_preload TEST)
add_dependencies(early_mmap_preload rocjitsu_shared)

add_executable(early_mmap_driver early_mmap_driver_test.cpp)
rj_configure_target(early_mmap_driver TEST)

if(RJ_INSTALL_TESTS)
    rj_install_test_target(early_mmap_driver)
    install(
        TARGETS early_mmap_preload
        LIBRARY DESTINATION ${RJ_INSTALL_TEST_BIN_DIR}
    )
endif()

# Keep rocjitsu before early_mmap_preload in LD_PRELOAD (after any shared
# sanitizer runtimes). The loader maps rocjitsu first, but runs
# early_mmap_preload's constructor before init_interposer(). That constructor
# calls mmap/munmap early enough to exercise the pre-init fallback; verifying the
# exact path directly would require adding observable test state to the
# interposer.
rj_add_test(
    NAME "InterposerTest.EarlyMmapBeforeRocjitsuConstructor"
    COMMAND $<TARGET_FILE:early_mmap_driver>
    ENVIRONMENT
        "LD_PRELOAD=${RJ_SANITIZER_PRELOAD_PREFIX}$<TARGET_FILE:rocjitsu_shared>:$<TARGET_FILE:early_mmap_preload>"
    INSTALL_COMMAND "bin/early_mmap_driver"
    INSTALL_ENVIRONMENT
        "LD_PRELOAD=${RJ_SANITIZER_PRELOAD_PREFIX}\${RJ_INSTALLED_LIBDIR}/${CMAKE_SHARED_LIBRARY_PREFIX}rocjitsu${CMAKE_SHARED_LIBRARY_SUFFIX}:bin/${CMAKE_SHARED_LIBRARY_PREFIX}early_mmap_preload${CMAKE_SHARED_LIBRARY_SUFFIX}"
)

# --- interposer dup/backend bookkeeping test (CLI launcher) ---
# Runs a small gtest binary via the rocjitsu CLI launcher (which preloads the
# interposer and writes the config_path file it reads) so open("/dev/kfd") is
# serviced by the simulated driver, then exercises dup/dup2/dup3/fcntl(F_DUPFD)
# fd tracking that the SimulatedKfd-level unit tests cannot reach.
add_executable(interposer_dup_test interposer_dup_test.cpp)
rj_configure_target(interposer_dup_test TEST)
target_include_directories(
    interposer_dup_test
    PRIVATE ${CMAKE_SOURCE_DIR}/lib/rocjitsu/external_headers/hsa_headers
)
target_link_libraries(
    interposer_dup_test
    PRIVATE GTest::gtest_main ${CMAKE_DL_LIBS}
)
add_dependencies(interposer_dup_test rocjitsu_bin rocjitsu_shared)
if(RJ_INSTALL_TESTS)
    rj_install_test_target(interposer_dup_test)
endif()
function(rj_add_interposer_test_suite _suite)
    foreach(_case IN LISTS ARGN)
        rj_add_test(
            NAME "${_suite}.${_case}"
            COMMAND
                ${RJ_CLI}
                --config
                ${RJ_CDNA4_KMD_CONFIG}
                --
                $<TARGET_FILE:interposer_dup_test>
                "--gtest_filter=${_suite}.${_case}"
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            SKIP_REGULAR_EXPRESSION "\\[  SKIPPED \\]"
            INSTALL_COMMAND
                "\${RJ_INSTALLED_BINDIR}/rocjitsu"
                "--config"
                "\${RJ_INSTALLED_CONFIG_DIR}/${RJ_CDNA4_KMD_CONFIG_NAME}"
                "--"
                "bin/interposer_dup_test"
                "--gtest_filter=${_suite}.${_case}"
        )
    endforeach()
endfunction()

set(_rj_interposer_dup_tests
    DupKeepsKfdRoutingAfterPrimaryClose
    FcntlDupfdKeepsKfdRouting
    Dup2OverPrimaryInvalidatesKfdIdentity
    Dup2OntoFreshFdRoutesKfd
    Dup3RoutesKfdAndRejectsSameFd
    ReopenAfterPrimaryOverwriteKeepsBackend
    SerializedReopenUnderContentionStaysRoutable
    FcntlDupfdReplacesStaleDrmTracking
    NonDrmDuplicatesClearStaleDrmTracking
    VforkChildCloseKeepsParentKfdRoutable
    ProcMapsNamesRemoteKfdMarker
)
# TSan models concurrent close/dup on one descriptor as an application-level fd
# race even though the interposer deliberately serializes the kernel operation.
# Keep the lifecycle regression in regular and ASan builds, where it exercises
# the intended close/reuse schedule without a sanitizer false positive.
if(NOT RJ_ENABLE_TSAN)
    list(
        APPEND _rj_interposer_dup_tests
        DrmCloseReuseRaceNeverMisclassifiesDuplicate
    )
endif()
rj_add_interposer_test_suite(InterposerDupTest ${_rj_interposer_dup_tests})

rj_add_interposer_test_suite(
    InterposerDrmTest
    DeviceInfoReportsActiveCuCount
)

rj_add_interposer_test_suite(
    InterposerForkTest
    ChildRefusesGpuEndpointsBeforeExec
    ChildRefusesEveryGpuEndpointSpelling
    ChildRefusesAnEmulatedEndpointTheHostDoesNotHave
    ChildFailsClosedOnAnUnclassifiablePath
    ChildClassifiesGpuEndpointsByIdentityNotName
    AppDescriptorCarriesNoRealHardwareAuthority
    ChildRefusesOperationsOnInheritedRealGpuDescriptors
    ChildCannotLaunderInheritedGpuFdAcrossExec
    ChildRefusesMmap64OnInheritedGpuFd
    ChildRefusesDup3OnInheritedGpuFd
    ChildFopenPreservesLibcModeSemantics
    MultithreadedForkExecAlwaysReachesExec
    SystemAndPopenWorkFromAMultithreadedParent
    PosixSpawnFromAMultithreadedParent
)

rj_add_interposer_test_suite(
    InterposerGemTest
    GemNamespaceIsPrivateButSharedByDuplicates
    IndependentDrmFilesRejectOverlappingMappings
    ReusedDmabufFdMintsDistinctHandles
    UnmapWithWrongHandleFails
    MapSucceedsAfterExportFdClosed
    ReplaceEvictsOverlappingDifferentSizeRange
    ReplaceTransfersOwnershipAcrossOldOwnerClose
    ClearUpdatesOwnerBookkeeping
)

set(_rj_interposer_syncobj_tests
    VmTimelineWaitObservesSynchronousMapAndUnmap
    VmDelayUpdateSuppressesTimelinePublication
    GemVaSignalsBinarySyncobjsAtPointZero
    PointZeroOutputReplacesTimelinePayload
    DuplicateDrmFdsShareNamespaceAndLifetime
    IndependentDrmOpensHaveSeparateNamespaces
    CreateSignaledSeedsOnlyTheInitialPoint
    WaitValidationMatchesDrm
    DestroyRejectsInvalidInput
    TimelineWaitBlocksUntilSignalOrDeadline
    TimelineWaitReturnsEintrForSignal
    ConcurrentVmUpdatesShareTimeline
    OutOfOrderTimelinePointPreservesWatermark
)
rj_add_interposer_test_suite(
    InterposerSyncobjTest
    ${_rj_interposer_syncobj_tests}
)

# --- rocminfo test (requires rocminfo binary + HSA runtime + CLI launcher) ---
# Spawns rocminfo via the rocjitsu CLI and validates output.

_rj_path_from_installed_test_bin(
    _rj_installed_rocjitsu_bin
    "${CMAKE_INSTALL_BINDIR}"
    "rocjitsu"
)
_rj_path_from_installed_test_bin(
    _rj_installed_cdna4_kmd_config
    "${CMAKE_INSTALL_DATADIR}/rocjitsu/configs"
    "${RJ_CDNA4_KMD_CONFIG_NAME}"
)
_rj_path_from_installed_test_bin(
    _rj_installed_dbt_guest_config
    "${CMAKE_INSTALL_DATADIR}/rocjitsu/configs"
    "guest_gfx950_on_gfx1201.json"
)

find_program(
    ROCMINFO_EXECUTABLE
    NAMES rocminfo
    HINTS "${ROCM_PATH}"
    PATH_SUFFIXES bin
    DOC "Path to the rocminfo executable"
)
if(ROCMINFO_EXECUTABLE AND RJ_HAS_KFD_DEVICE)
    add_executable(rocminfo_test rocminfo_test.cpp)
    target_compile_definitions(
        rocminfo_test
        PRIVATE
            ROCMINFO_PATH="${ROCMINFO_EXECUTABLE}"
            ROCJITSU_BIN="$<TARGET_FILE:rocjitsu_bin>"
            RJ_CONFIG_PATH="${RJ_CDNA4_KMD_CONFIG}"
            RJ_INSTALLED_ROCJITSU_BIN="${_rj_installed_rocjitsu_bin}"
            RJ_INSTALLED_CONFIG_PATH="${_rj_installed_cdna4_kmd_config}"
    )
    target_link_libraries(rocminfo_test PRIVATE GTest::gtest_main)
    add_dependencies(rocminfo_test rocjitsu_bin rocjitsu_shared)

    if(RJ_INSTALL_TESTS)
        rj_install_test_target(rocminfo_test)
    endif()
    set(_rocminfo_install_environment_arg)
    if(ROCM_PATH AND EXISTS "${ROCM_PATH}/bin/rocminfo")
        list(
            APPEND _rocminfo_install_environment_arg
            INSTALL_ENVIRONMENT
            "ROCMINFO_PATH=${ROCM_PATH}/bin/rocminfo"
        )
    endif()
    foreach(
        _rocminfo_test
        IN
        ITEMS
            ExitsSuccessfully
            InterposerActive
            HsaSystemAttributes
            DetectsGpuAgent
            ReportsGfx950
            ReportsWavefrontSize
            ReportsMi350xActiveComputeUnits
            ReportsMi350xMaxClockFrequency
    )
        rj_add_test(
            NAME "RocminfoTest.${_rocminfo_test}"
            COMMAND
                rocminfo_test
                "--gtest_filter=RocminfoTest.${_rocminfo_test}"
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            ENVIRONMENT ${RJ_KMD_PRELOAD_ENV}
            INSTALL_COMMAND
                "bin/rocminfo_test"
                "--gtest_filter=RocminfoTest.${_rocminfo_test}"
            ${_rocminfo_install_environment_arg}
        )
    endforeach()

    add_executable(dbt_guest_rocminfo_test rocminfo_test.cpp)
    target_compile_definitions(
        dbt_guest_rocminfo_test
        PRIVATE
            ROCMINFO_PATH="${ROCMINFO_EXECUTABLE}"
            ROCJITSU_BIN="$<TARGET_FILE:rocjitsu_bin>"
            RJ_CONFIG_PATH="${CMAKE_SOURCE_DIR}/configs/guest_gfx950_on_gfx1201.json"
            RJ_INSTALLED_ROCJITSU_BIN="${_rj_installed_rocjitsu_bin}"
            RJ_INSTALLED_CONFIG_PATH="${_rj_installed_dbt_guest_config}"
    )
    target_link_libraries(dbt_guest_rocminfo_test PRIVATE GTest::gtest_main)
    add_dependencies(
        dbt_guest_rocminfo_test
        rocjitsu_bin
        rocjitsu_shared
        rocjitsu_hooks
    )

    if(RJ_HAS_GFX1201_GPU)
        add_test(
            NAME "DbtGuestRocminfoTest.ExitsSuccessfully"
            COMMAND
                dbt_guest_rocminfo_test
                --gtest_filter=RocminfoTest.ExitsSuccessfully
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        )
        add_test(
            NAME "DbtGuestRocminfoTest.InterposerActive"
            COMMAND
                dbt_guest_rocminfo_test
                --gtest_filter=RocminfoTest.InterposerActive
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        )
        add_test(
            NAME "DbtGuestRocminfoTest.HsaSystemAttributes"
            COMMAND
                dbt_guest_rocminfo_test
                --gtest_filter=RocminfoTest.HsaSystemAttributes
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        )
        add_test(
            NAME "DbtGuestRocminfoTest.DetectsGpuAgent"
            COMMAND
                dbt_guest_rocminfo_test
                --gtest_filter=RocminfoTest.DetectsGpuAgent
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        )
        add_test(
            NAME "DbtGuestRocminfoTest.ReportsGfx950"
            COMMAND
                dbt_guest_rocminfo_test
                --gtest_filter=RocminfoTest.ReportsGfx950
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        )
        add_test(
            NAME "DbtGuestRocminfoTest.ReportsWavefrontSize"
            COMMAND
                dbt_guest_rocminfo_test
                --gtest_filter=RocminfoTest.ReportsWavefrontSize
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        )
        foreach(
            _dbt_guest_rocminfo_test
            IN
            ITEMS
                "DbtGuestRocminfoTest.ExitsSuccessfully"
                "DbtGuestRocminfoTest.InterposerActive"
                "DbtGuestRocminfoTest.HsaSystemAttributes"
                "DbtGuestRocminfoTest.DetectsGpuAgent"
                "DbtGuestRocminfoTest.ReportsGfx950"
                "DbtGuestRocminfoTest.ReportsWavefrontSize"
        )
            rj_set_default_runtime_dir("${_dbt_guest_rocminfo_test}")
        endforeach()
        message(
            STATUS
            "DBT guest rocminfo tests enabled (gfx1201 host GPU detected)"
        )
    else()
        message(
            STATUS
            "DBT guest rocminfo tests built but NOT registered with CTest "
            "(no gfx1201 KFD host GPU)"
        )
    endif()
    message(STATUS "rocminfo test enabled (found ${ROCMINFO_EXECUTABLE})")
elseif(ROCMINFO_EXECUTABLE)
    message(
        STATUS
        "rocminfo test disabled (/dev/kfd not found; rocminfo exits before "
        "simulated KFD discovery)"
    )
else()
    message(STATUS "rocminfo test disabled (rocminfo not found)")
endif()

# --- HSA init test (requires ROCm HSA runtime + CLI launcher) ---

find_library(HSA_RUNTIME64 hsa-runtime64 PATHS "${ROCM_PATH}" PATH_SUFFIXES lib)
if(HSA_RUNTIME64)
    get_filename_component(RJ_HSA_LIBRARY_DIR "${HSA_RUNTIME64}" DIRECTORY)
    get_filename_component(RJ_HSA_ROCM_ROOT "${RJ_HSA_LIBRARY_DIR}" DIRECTORY)
    set(RJ_HSA_INCLUDE_DIR "${RJ_HSA_ROCM_ROOT}/include")

    add_executable(hsa_init_test hsa_init_test.cpp)
    target_include_directories(
        hsa_init_test
        PRIVATE ${RJ_HSA_INCLUDE_DIR} ${HSA_INCLUDE_DIR}
    )
    target_link_libraries(hsa_init_test PRIVATE ${HSA_RUNTIME64} GTest::gtest)
    target_link_options(hsa_init_test PRIVATE -Wl,-rpath,${RJ_HSA_LIBRARY_DIR})
    add_dependencies(hsa_init_test rocjitsu_bin rocjitsu_shared)
    if(RJ_INSTALL_TESTS)
        rj_install_test_target(hsa_init_test)
    endif()
    foreach(_hsa_test IN ITEMS InitSucceeded GpuAgentFound)
        rj_add_test(
            NAME "HsaTest.${_hsa_test}"
            COMMAND
                ${RJ_CLI}
                --config
                ${RJ_CDNA4_KMD_CONFIG}
                --
                $<TARGET_FILE:hsa_init_test>
                "--gtest_filter=HsaTest.${_hsa_test}"
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            SKIP_REGULAR_EXPRESSION "\\[  SKIPPED \\]"
            INSTALL_COMMAND
                "\${RJ_INSTALLED_BINDIR}/rocjitsu"
                "--config"
                "\${RJ_INSTALLED_CONFIG_DIR}/${RJ_CDNA4_KMD_CONFIG_NAME}"
                "--"
                "bin/hsa_init_test"
                "--gtest_filter=HsaTest.${_hsa_test}"
        )
    endforeach()
    message(STATUS "HSA init test enabled (found ${HSA_RUNTIME64})")

    # --- HSA translate test: CDNA4→RDNA4 on-device verification ---
    # Requires both compiled device kernels AND an RDNA4 GPU on the host.
    if(HAS_DEVICE_KERNELS)
        # Detect supported DBT host GPUs at configure time via rocm_agent_enumerator.
        find_program(
            ROCM_AGENT_ENUM
            rocm_agent_enumerator
            HINTS "${ROCM_PATH}"
            PATH_SUFFIXES bin
        )
        set(HAS_DBT_VECTOR_ADD_GPU FALSE)
        set(HAS_CDNA3_GPU FALSE)
        set(HAS_RDNA4_GPU FALSE)
        set(HAS_GFX1201_GPU FALSE)
        set(HAS_CDNA2_GPU FALSE)
        if(ROCM_AGENT_ENUM)
            execute_process(
                COMMAND ${ROCM_AGENT_ENUM}
                OUTPUT_VARIABLE _agents
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET
                RESULT_VARIABLE _agent_rc
            )
            if(
                _agent_rc EQUAL 0
                AND _agents MATCHES "gfx94[012]|gfx1100|gfx120[01]"
            )
                set(HAS_DBT_VECTOR_ADD_GPU TRUE)
            endif()
            if(_agent_rc EQUAL 0 AND _agents MATCHES "gfx94[012]")
                set(HAS_CDNA3_GPU TRUE)
            endif()
            if(_agent_rc EQUAL 0 AND _agents MATCHES "gfx120[01]")
                set(HAS_RDNA4_GPU TRUE)
            endif()
            if(_agent_rc EQUAL 0 AND _agents MATCHES "gfx1201")
                set(HAS_GFX1201_GPU TRUE)
            endif()
            if(_agent_rc EQUAL 0 AND _agents MATCHES "gfx90a")
                set(HAS_CDNA2_GPU TRUE)
            endif()
        endif()

        add_executable(hsa_translate_test dbt/hsa_translate_test.cpp)
        target_include_directories(
            hsa_translate_test
            PRIVATE
                ${CMAKE_SOURCE_DIR}/lib/rocjitsu/include
                ${CMAKE_SOURCE_DIR}/lib/rocjitsu/src
                ${RJ_HSA_INCLUDE_DIR}
        )
        target_link_libraries(
            hsa_translate_test
            PRIVATE
                ${RJ_OBJECT_LIBS}
                util
                ${HSA_RUNTIME64}
                GTest::gtest
                GTest::gtest_main
        )
        target_link_options(
            hsa_translate_test
            PRIVATE -Wl,-rpath,${RJ_HSA_LIBRARY_DIR}
        )
        target_compile_definitions(
            hsa_translate_test
            PRIVATE HAS_HOST_AMDGPU=1 KERNEL_DIR="${KERNEL_OUTPUT_DIR}"
        )
        add_dependencies(hsa_translate_test device_kernels)
        rj_configure_target(hsa_translate_test TEST)
        if(RJ_INSTALL_TESTS)
            rj_install_test_target(hsa_translate_test)
        endif()

        add_executable(hsa_hooks_test dbt/hsa_hooks_test.cpp)
        target_include_directories(
            hsa_hooks_test
            PRIVATE
                ${CMAKE_SOURCE_DIR}/lib/rocjitsu/include
                ${CMAKE_SOURCE_DIR}/lib/rocjitsu/src
                ${RJ_HSA_INCLUDE_DIR}
        )
        target_link_libraries(
            hsa_hooks_test
            PRIVATE
                ${RJ_OBJECT_LIBS}
                util
                ${HSA_RUNTIME64}
                GTest::gtest
                GTest::gtest_main
        )
        target_link_options(
            hsa_hooks_test
            PRIVATE -Wl,-rpath,${RJ_HSA_LIBRARY_DIR}
        )
        target_compile_definitions(
            hsa_hooks_test
            PRIVATE HAS_HOST_AMDGPU=1 KERNEL_DIR="${KERNEL_OUTPUT_DIR}"
        )
        add_dependencies(
            hsa_hooks_test
            device_kernels
            rocjitsu_bin
            rocjitsu_hooks
        )
        rj_configure_target(hsa_hooks_test TEST)
        if(RJ_INSTALL_TESTS)
            rj_install_test_target(hsa_hooks_test)
        endif()

        if(HAS_RDNA4_GPU)
            rj_add_test(
                NAME "HsaTranslateTest.TranslateAndDispatchVectorAdd"
                COMMAND hsa_translate_test --gtest_filter=*VectorAdd*
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                INSTALL_COMMAND
                    "bin/hsa_translate_test"
                    "--gtest_filter=*VectorAdd*"
            )
        endif()

        if(HAS_GFX1201_GPU)
            rj_add_test(
                NAME "HsaHooksTest.TranslateGfx950Mfma16x16ThroughToolsLib"
                COMMAND
                    $<TARGET_FILE:rocjitsu_bin> --config
                    ${CMAKE_SOURCE_DIR}/configs/guest_gfx950_on_gfx1201.json --
                    $<TARGET_FILE:hsa_hooks_test>
                    --gtest_filter=HsaHooksTest.TranslateGfx950Mfma16x16ThroughToolsLib
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                TIMEOUT 60
                INSTALL_COMMAND
                    "\${RJ_INSTALLED_BINDIR}/rocjitsu"
                    "--config"
                    "\${RJ_INSTALLED_CONFIG_DIR}/guest_gfx950_on_gfx1201.json"
                    "--"
                    "bin/hsa_hooks_test"
                    "--gtest_filter=HsaHooksTest.TranslateGfx950Mfma16x16ThroughToolsLib"
            )
        endif()

        if(HAS_RDNA4_GPU)
            rj_add_test(
                NAME "HsaTranslateTest.TranslateAndDispatchMfma16x16"
                COMMAND hsa_translate_test --gtest_filter=*Mfma16x16*
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                INSTALL_COMMAND
                    "bin/hsa_translate_test"
                    "--gtest_filter=*Mfma16x16*"
            )
            message(STATUS "HSA translate test enabled (RDNA4 GPU detected)")
        else()
            message(
                STATUS
                "HSA translate test built but NOT registered with CTest (no RDNA4 GPU)"
            )
        endif()

        set(RJ_CDNA4_TO_CDNA3_DISPATCH_TEST_EXE cdna4_to_cdna3_dispatch_test)
        add_executable(
            ${RJ_CDNA4_TO_CDNA3_DISPATCH_TEST_EXE}
            dbt/cdna4_to_cdna3_dispatch_test.cpp
        )
        target_include_directories(
            ${RJ_CDNA4_TO_CDNA3_DISPATCH_TEST_EXE}
            PRIVATE
                ${CMAKE_SOURCE_DIR}/lib/rocjitsu/include
                ${CMAKE_SOURCE_DIR}/lib/rocjitsu/src
                ${RJ_HSA_INCLUDE_DIR}
        )
        target_link_libraries(
            ${RJ_CDNA4_TO_CDNA3_DISPATCH_TEST_EXE}
            PRIVATE
                ${RJ_OBJECT_LIBS}
                util
                ${HSA_RUNTIME64}
                GTest::gtest
                GTest::gtest_main
        )
        target_link_options(
            ${RJ_CDNA4_TO_CDNA3_DISPATCH_TEST_EXE}
            PRIVATE -Wl,-rpath,${RJ_HSA_LIBRARY_DIR}
        )
        target_compile_definitions(
            ${RJ_CDNA4_TO_CDNA3_DISPATCH_TEST_EXE}
            PRIVATE HAS_HOST_AMDGPU=1 KERNEL_DIR="${KERNEL_OUTPUT_DIR}"
        )
        add_dependencies(
            ${RJ_CDNA4_TO_CDNA3_DISPATCH_TEST_EXE}
            device_kernels
            rocjitsu_bin
            rocjitsu_shared
            rocjitsu_hooks
        )
        rj_configure_target(${RJ_CDNA4_TO_CDNA3_DISPATCH_TEST_EXE} TEST)
        if(RJ_INSTALL_TESTS)
            rj_install_test_target(${RJ_CDNA4_TO_CDNA3_DISPATCH_TEST_EXE})
        endif()

        set(RJ_CDNA4_TO_CDNA3_TRANSLATE_TESTS
            DynamicCopyLoopTranslates
            VCvtPkBf16F32Translates
            VirtualLdsSmokeTranslatesWithSidecarMetadata
            TritonDynamicMatmulTranslates
            TritonBufferAsyncMatmulTranslates
            TritonFlashAttentionNoAsyncTranslates
            TritonFlashAttentionBufferAsyncTranslates
            HipKittensBf16Matmul16x32TranslatesWithVirtualLdsMetadata
            HipKittensBf16Matmul32x16TranslatesWithVirtualLdsMetadata
        )
        foreach(_test_name IN LISTS RJ_CDNA4_TO_CDNA3_TRANSLATE_TESTS)
            rj_add_test(
                NAME "Cdna4ToCdna3DispatchTest.${_test_name}"
                COMMAND
                    ${RJ_CDNA4_TO_CDNA3_DISPATCH_TEST_EXE}
                    --gtest_filter=Cdna4ToCdna3DispatchTest.${_test_name}
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                TIMEOUT 120
                INSTALL_COMMAND
                    "bin/${RJ_CDNA4_TO_CDNA3_DISPATCH_TEST_EXE}"
                    "--gtest_filter=Cdna4ToCdna3DispatchTest.${_test_name}"
            )
        endforeach()

        set(RJ_CDNA4_TO_CDNA3_DISPATCH_RUN_TESTS
            DynamicCopyLoopDispatchAndRun
            VCvtPkBf16F32DispatchAndRun
            VirtualLdsSmokeDispatchAndRun
            TritonDynamicMatmulDispatchAndRun
            TritonBufferAsyncMatmulDispatchAndRun
            TritonBufferAsyncMatmulConservativeLivenessDispatchAndRun
            TritonFlashAttentionNoAsyncDispatchAndRun
            TritonFlashAttentionBufferAsyncDispatchAndRun
        )
        foreach(_test_name IN LISTS RJ_CDNA4_TO_CDNA3_DISPATCH_RUN_TESTS)
            rj_add_test(
                NAME "Cdna4ToCdna3SimulatedDispatchTest.${_test_name}"
                COMMAND
                    ${RJ_CLI} --config ${RJ_CDNA3_KMD_CONFIG} --
                    $<TARGET_FILE:${RJ_CDNA4_TO_CDNA3_DISPATCH_TEST_EXE}>
                    --gtest_filter=Cdna4ToCdna3DispatchTest.${_test_name}
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                TIMEOUT 300
                INSTALL_COMMAND
                    "\${RJ_INSTALLED_BINDIR}/rocjitsu"
                    "--config"
                    "\${RJ_INSTALLED_CONFIG_DIR}/${RJ_CDNA3_KMD_CONFIG_NAME}"
                    "--"
                    "bin/${RJ_CDNA4_TO_CDNA3_DISPATCH_TEST_EXE}"
                    "--gtest_filter=Cdna4ToCdna3DispatchTest.${_test_name}"
            )
        endforeach()
        message(STATUS "CDNA4->CDNA3 simulator dispatch tests enabled")

        set(RJ_CDNA4_TO_CDNA3_DBT_GUEST_TESTS
            DynamicCopyLoopDispatchAndRun
            TritonDynamicMatmulDispatchAndRun
            TritonBufferAsyncMatmulDispatchAndRun
            TritonFlashAttentionNoAsyncDispatchAndRun
            TritonFlashAttentionBufferAsyncDispatchAndRun
        )
        # The conservative-liveness variant injects BinaryTranslatorOptions
        # directly and cannot be expressed through the public DBT hook. Running
        # it here would duplicate the default buffer-async guest path.
        foreach(_test_name IN LISTS RJ_CDNA4_TO_CDNA3_DBT_GUEST_TESTS)
            rj_add_test(
                NAME "Cdna4ToCdna3DbtGuestTest.${_test_name}"
                COMMAND
                    ${RJ_CLI} --config ${RJ_CDNA4_TO_CDNA3_SIM_DBT_CONFIG} --
                    $<TARGET_FILE:${RJ_CDNA4_TO_CDNA3_DISPATCH_TEST_EXE}>
                    --gtest_filter=Cdna4ToCdna3DbtGuestTest.${_test_name}
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                TIMEOUT 300
                INSTALL_COMMAND
                    "\${RJ_INSTALLED_BINDIR}/rocjitsu"
                    "--config"
                    "\${RJ_INSTALLED_CONFIG_DIR}/${RJ_CDNA4_TO_CDNA3_SIM_DBT_CONFIG_NAME}"
                    "--"
                    "bin/${RJ_CDNA4_TO_CDNA3_DISPATCH_TEST_EXE}"
                    "--gtest_filter=Cdna4ToCdna3DbtGuestTest.${_test_name}"
            )
        endforeach()

        if(HAS_CDNA3_GPU)
            foreach(_test_name IN LISTS RJ_CDNA4_TO_CDNA3_DISPATCH_RUN_TESTS)
                rj_add_test(
                    NAME "Cdna4ToCdna3DispatchTest.${_test_name}"
                    COMMAND
                        ${RJ_CDNA4_TO_CDNA3_DISPATCH_TEST_EXE}
                        --gtest_filter=Cdna4ToCdna3DispatchTest.${_test_name}
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                    TIMEOUT 120
                    INSTALL_COMMAND
                        "bin/${RJ_CDNA4_TO_CDNA3_DISPATCH_TEST_EXE}"
                        "--gtest_filter=Cdna4ToCdna3DispatchTest.${_test_name}"
                )
            endforeach()
            message(
                STATUS
                "CDNA4->CDNA3 dispatch tests enabled (CDNA3 GPU detected)"
            )
        else()
            message(
                STATUS
                "CDNA4->CDNA3 hardware dispatch tests built but NOT registered with CTest "
                "(no CDNA3 GPU)"
            )
        endif()

        # --- HSA DBI nop-asm test: gfx90a kernel patched with inlined nop ---
        # Build always; register with CTest only when a gfx90a GPU is present.
        add_executable(hsa_dbi_nop_asm_test dbi/hsa_dbi_nop_asm_test.cpp)
        target_include_directories(
            hsa_dbi_nop_asm_test
            PRIVATE
                ${CMAKE_SOURCE_DIR}/lib/rocjitsu/include
                ${CMAKE_SOURCE_DIR}/lib/rocjitsu/src
                ${RJ_HSA_INCLUDE_DIR}
        )
        target_link_libraries(
            hsa_dbi_nop_asm_test
            PRIVATE
                ${RJ_OBJECT_LIBS}
                util
                ${HSA_RUNTIME64}
                GTest::gtest
                GTest::gtest_main
        )
        target_link_options(
            hsa_dbi_nop_asm_test
            PRIVATE -Wl,-rpath,${RJ_HSA_LIBRARY_DIR}
        )
        target_compile_definitions(
            hsa_dbi_nop_asm_test
            PRIVATE HAS_HOST_AMDGPU=1 KERNEL_DIR="${KERNEL_OUTPUT_DIR}"
        )
        # rocjitsu_bin/rocjitsu_shared are needed by the Sim registrations
        # below, which launch this binary through the CLI. A generator
        # expression in a test COMMAND does not create a build dependency.
        add_dependencies(
            hsa_dbi_nop_asm_test
            device_kernels
            rocjitsu_bin
            rocjitsu_shared
        )
        rj_configure_target(hsa_dbi_nop_asm_test TEST)
        if(RJ_INSTALL_TESTS)
            rj_install_test_target(hsa_dbi_nop_asm_test)
        endif()

        # Local helper: register one TEST_F case from hsa_dbi_nop_asm_test as a
        # CTest entry. The gtest suite is HsaDbiNopAsm<arch><policy>, where
        # <arch> names the target (Cdna2, ...) and <policy> is Static or
        # Hardware. Scoped here because it's only used by the calls immediately
        # below.
        #
        # SIM runs the same gtest case through the rocjitsu CLI against a
        # simulated agent instead of the host GPU, using CONFIG (defaults to the
        # CDNA2 config). find_gpu_agent() matches on the HSA ISA name, so the
        # fixture binds to whichever agent the launcher supplies and the test
        # source needs no backend knowledge. Only the CTest name encodes the
        # backend; the gtest filter still names the <arch><policy> fixture, as
        # with Cdna4ToCdna3SimulatedDispatchTest.
        function(register_hsa_dbi_nop_asm_case arch policy case)
            cmake_parse_arguments(
                RJ_DBI
                "SIM"
                "TIMEOUT;CONFIG;CONFIG_NAME"
                ""
                ${ARGN}
            )
            set(name "HsaDbiNopAsm${arch}${policy}.${case}")
            if(NOT RJ_DBI_CONFIG)
                set(RJ_DBI_CONFIG "${RJ_CDNA2_KMD_CONFIG}")
                set(RJ_DBI_CONFIG_NAME "${RJ_CDNA2_KMD_CONFIG_NAME}")
            endif()
            if(RJ_DBI_SIM)
                set(_ctest_name "HsaDbiNopAsm${arch}Sim.${case}")
                set(_command
                    ${CMAKE_COMMAND}
                    "-DTEST_EXECUTABLE=$<TARGET_FILE:hsa_dbi_nop_asm_test>"
                    "-DTEST_FILTER=${name}"
                    "-DTEST_LAUNCHER=${RJ_CLI}"
                    "-DTEST_CONFIG=${RJ_DBI_CONFIG}"
                    "-DFAIL_ON_SKIP=ON"
                    -P
                    "${PROJECT_SOURCE_DIR}/cmake/rj_run_filtered_gtest.cmake"
                )
                set(_install_command
                    "${CMAKE_COMMAND}"
                    "-DTEST_EXECUTABLE=bin/hsa_dbi_nop_asm_test"
                    "-DTEST_FILTER=${name}"
                    "-DTEST_LAUNCHER=\${RJ_INSTALLED_BINDIR}/rocjitsu"
                    "-DTEST_CONFIG=\${RJ_INSTALLED_CONFIG_DIR}/${RJ_DBI_CONFIG_NAME}"
                    "-DFAIL_ON_SKIP=ON"
                    "-P"
                    "rj_run_filtered_gtest.cmake"
                )
            else()
                set(_ctest_name "${name}")
                set(_command hsa_dbi_nop_asm_test --gtest_filter=${name})
                set(_install_command
                    "bin/hsa_dbi_nop_asm_test"
                    "--gtest_filter=${name}"
                )
            endif()
            set(_timeout)
            if(RJ_DBI_TIMEOUT)
                set(_timeout TIMEOUT ${RJ_DBI_TIMEOUT})
            endif()
            rj_add_test(
                NAME "${_ctest_name}"
                COMMAND ${_command}
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                INSTALL_COMMAND ${_install_command}
                ${_timeout}
            )
        endfunction()

        # gfx90a / CDNA2.

        # Static-only test: no GPU needed, runs anywhere the binary is built.
        register_hsa_dbi_nop_asm_case(Cdna2 Static PatchedElfActuallyContainsInstrumentation)

        # The same three cases on the simulator. Registered unconditionally so
        # CI covers the DBI dispatch path without a CDNA2 host; the timeouts are
        # the hardware ones scaled for simulated execution.
        register_hsa_dbi_nop_asm_case(
            Cdna2
            Hardware
            PatchedElfLoadsAndValidatesInHsaExecutable
            SIM
            TIMEOUT 300
        )
        register_hsa_dbi_nop_asm_case(
            Cdna2
            Hardware
            PatchedKernelDispatchMatchesOriginal
            SIM
            TIMEOUT 300
        )
        register_hsa_dbi_nop_asm_case(
            Cdna2
            Hardware
            TrampolineIsActuallyExecutedByGpu
            SIM
            TIMEOUT 300
        )

        if(HAS_CDNA2_GPU)
            # Hardware-gated tests. The two that dispatch get an explicit
            # timeout; the load+validate one runs quickly enough for the
            # default.
            register_hsa_dbi_nop_asm_case(
                Cdna2
                Hardware
                PatchedElfLoadsAndValidatesInHsaExecutable
            )
            register_hsa_dbi_nop_asm_case(
                Cdna2
                Hardware
                PatchedKernelDispatchMatchesOriginal
                TIMEOUT 30
            )
            register_hsa_dbi_nop_asm_case(
                Cdna2
                Hardware
                TrampolineIsActuallyExecutedByGpu
                TIMEOUT 30
            )
            message(STATUS "HSA DBI nop-asm test enabled (gfx90a detected)")
        else()
            message(
                STATUS
                "HSA DBI nop-asm hardware cases NOT registered with CTest "
                "(no gfx90a); the Sim cases still run"
            )
        endif()

        # Pin the FAIL_ON_SKIP guard, the way GtestFilterGuard.RejectsMissingTest
        # pins the zero-test guard. Launching the gfx90a fixture against a
        # simulated gfx942 selects one test, find_gpu_agent() finds no match,
        # and GoogleTest exits 0 after skipping it -- so success here means the
        # wrapper turned that skip into a failure instead of a false pass.
        add_test(
            NAME GtestFilterGuard.RejectsSkippedTest
            COMMAND
                ${CMAKE_COMMAND}
                "-DTEST_EXECUTABLE=$<TARGET_FILE:hsa_dbi_nop_asm_test>"
                "-DTEST_FILTER=HsaDbiNopAsmCdna2Hardware.PatchedElfLoadsAndValidatesInHsaExecutable"
                "-DTEST_LAUNCHER=${RJ_CLI}"
                "-DTEST_CONFIG=${RJ_CDNA3_KMD_CONFIG}" -DFAIL_ON_SKIP=ON -P
                "${PROJECT_SOURCE_DIR}/cmake/rj_run_filtered_gtest.cmake"
        )
        set_tests_properties(
            GtestFilterGuard.RejectsSkippedTest
            PROPERTIES WILL_FAIL TRUE TIMEOUT 300
        )

        # --- HSA DBI nop-probe smoke test: gfx90a kernel patched to CALL the
        # compiled rj_nop_probe via an s_swappc_b64 trampoline (PC01-B). Needs
        # the compiled probe fixture (amdclang++ + clang-offload-bundler), so it
        # is built only when HAS_PROBE_FIXTURES. Static case registers anywhere;
        # hardware cases register only with a gfx90a GPU.
        if(HAS_PROBE_FIXTURES)
            add_executable(
                hsa_dbi_nop_probe_test
                dbi/hsa_dbi_nop_probe_test.cpp
            )
            target_include_directories(
                hsa_dbi_nop_probe_test
                PRIVATE
                    ${CMAKE_SOURCE_DIR}/lib/rocjitsu/include
                    ${CMAKE_SOURCE_DIR}/lib/rocjitsu/src
                    ${RJ_HSA_INCLUDE_DIR}
            )
            target_link_libraries(
                hsa_dbi_nop_probe_test
                PRIVATE
                    ${RJ_OBJECT_LIBS}
                    util
                    ${HSA_RUNTIME64}
                    GTest::gtest
                    GTest::gtest_main
            )
            target_link_options(
                hsa_dbi_nop_probe_test
                PRIVATE -Wl,-rpath,${RJ_HSA_LIBRARY_DIR}
            )
            target_compile_definitions(
                hsa_dbi_nop_probe_test
                PRIVATE HAS_HOST_AMDGPU=1 KERNEL_DIR="${KERNEL_OUTPUT_DIR}"
            )
            # rocjitsu_bin/rocjitsu_shared are needed by the Sim registrations
            # below, which launch this binary through the CLI. A generator
            # expression in a test COMMAND does not create a build dependency.
            add_dependencies(
                hsa_dbi_nop_probe_test
                device_kernels
                probe_rj_nop_probe_gfx90a
                rocjitsu_bin
                rocjitsu_shared
            )
            rj_configure_target(hsa_dbi_nop_probe_test TEST)
            if(RJ_INSTALL_TESTS)
                rj_install_test_target(hsa_dbi_nop_probe_test)
            endif()

            # Local helper: register one TEST_F case as a CTest entry, in both the
            # build tree and (under RJ_INSTALL_TESTS) the installed CTest tree.
            # The gtest suite is HsaDbiNopProbe<arch><policy>, as in
            # register_hsa_dbi_nop_asm_case above; SIM and CONFIG behave the same.
            function(register_hsa_dbi_nop_probe_case arch policy case)
                cmake_parse_arguments(
                    RJ_DBIP
                    "SIM"
                    "TIMEOUT;CONFIG;CONFIG_NAME"
                    ""
                    ${ARGN}
                )
                set(name "HsaDbiNopProbe${arch}${policy}.${case}")
                if(NOT RJ_DBIP_CONFIG)
                    set(RJ_DBIP_CONFIG "${RJ_CDNA2_KMD_CONFIG}")
                    set(RJ_DBIP_CONFIG_NAME "${RJ_CDNA2_KMD_CONFIG_NAME}")
                endif()
                if(RJ_DBIP_SIM)
                    set(_ctest_name "HsaDbiNopProbe${arch}Sim.${case}")
                    set(_command
                        ${CMAKE_COMMAND}
                        "-DTEST_EXECUTABLE=$<TARGET_FILE:hsa_dbi_nop_probe_test>"
                        "-DTEST_FILTER=${name}"
                        "-DTEST_LAUNCHER=${RJ_CLI}"
                        "-DTEST_CONFIG=${RJ_DBIP_CONFIG}"
                        "-DFAIL_ON_SKIP=ON"
                        -P
                        "${PROJECT_SOURCE_DIR}/cmake/rj_run_filtered_gtest.cmake"
                    )
                    set(_install_command
                        "${CMAKE_COMMAND}"
                        "-DTEST_EXECUTABLE=bin/hsa_dbi_nop_probe_test"
                        "-DTEST_FILTER=${name}"
                        "-DTEST_LAUNCHER=\${RJ_INSTALLED_BINDIR}/rocjitsu"
                        "-DTEST_CONFIG=\${RJ_INSTALLED_CONFIG_DIR}/${RJ_DBIP_CONFIG_NAME}"
                        "-DFAIL_ON_SKIP=ON"
                        "-P"
                        "rj_run_filtered_gtest.cmake"
                    )
                else()
                    set(_ctest_name "${name}")
                    set(_command hsa_dbi_nop_probe_test --gtest_filter=${name})
                    set(_install_command
                        "bin/hsa_dbi_nop_probe_test"
                        "--gtest_filter=${name}"
                    )
                endif()
                set(_timeout)
                if(RJ_DBIP_TIMEOUT)
                    set(_timeout TIMEOUT ${RJ_DBIP_TIMEOUT})
                endif()
                rj_add_test(
                    NAME "${_ctest_name}"
                    COMMAND ${_command}
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                    INSTALL_COMMAND ${_install_command}
                    ${_timeout}
                )
            endfunction()

            # gfx90a / CDNA2.

            # Static-only test: no GPU needed, runs anywhere the binary builds.
            register_hsa_dbi_nop_probe_case(
                Cdna2
                Static
                PatchedElfContainsProbeCallInstrumentation
            )

            # The same three cases on the simulator, registered unconditionally
            # so CI covers the probe-call dispatch path without a CDNA2 host.
            register_hsa_dbi_nop_probe_case(
                Cdna2
                Hardware
                PatchedElfLoadsAndValidatesInHsaExecutable
                SIM
                TIMEOUT 300
            )
            register_hsa_dbi_nop_probe_case(
                Cdna2
                Hardware
                PatchedKernelDispatchMatchesOriginal
                SIM
                TIMEOUT 300
            )
            register_hsa_dbi_nop_probe_case(
                Cdna2
                Hardware
                ProbeBodyIsActuallyCalledByGpu
                SIM
                TIMEOUT 300
            )

            if(HAS_CDNA2_GPU)
                register_hsa_dbi_nop_probe_case(
                    Cdna2
                    Hardware
                    PatchedElfLoadsAndValidatesInHsaExecutable
                )
                # The dispatch + sabotage cases prove the probe executes on the
                # GPU. They (temporarily) require the register-padded fixture
                # kernel vector_add_probe, granting the s[30:31] link pair
                register_hsa_dbi_nop_probe_case(
                    Cdna2
                    Hardware
                    PatchedKernelDispatchMatchesOriginal
                    TIMEOUT 30
                )
                register_hsa_dbi_nop_probe_case(
                    Cdna2
                    Hardware
                    ProbeBodyIsActuallyCalledByGpu
                    TIMEOUT 30
                )
                message(
                    STATUS
                    "HSA DBI nop-probe smoke test enabled (gfx90a detected)"
                )
            else()
                message(
                    STATUS
                    "HSA DBI nop-probe hardware cases NOT registered with "
                    "CTest (no gfx90a); the Sim cases still run"
                )
            endif()
        else()
            message(
                STATUS
                "HSA DBI nop-probe smoke test disabled (no probe fixtures; "
                "needs amdclang++ + clang-offload-bundler)"
            )
        endif()
    endif()
else()
    message(STATUS "HSA init test disabled (libhsa-runtime64 not found)")
endif()

# --- HIP tests (requires amdclang++ + CLI launcher) ---

rj_find_amdcxx(AMDCXX)
set(RJ_ENABLE_HIP_TESTS ON)
find_library(HIP_RUNTIME64 amdhip64 PATHS "${ROCM_PATH}" PATH_SUFFIXES lib)
set(RJ_HIP_RUNTIME_SUPPORTS_TEST_ARCH FALSE)
set(RJ_HIP_RUNTIME_SUPPORTS_GFX950 FALSE)
set(RJ_HIP_RUNTIME_SUPPORTS_GFX942 FALSE)

function(rj_hip_runtime_supports_arch out_var arch)
    set(_supports FALSE)
    if(HIP_RUNTIME64)
        # Recent HIP runtimes reject agents whose PCI ID/ISA are not in their
        # device table, even when ROCR can enumerate the agent. Gate CTest
        # registration by the user-space runtime's advertised architectures so
        # unsupported targets do not fail before exercising rocjitsu.
        file(STRINGS "${HIP_RUNTIME64}" _rj_hip_arch_strings REGEX "${arch}")
        if(_rj_hip_arch_strings)
            set(_supports TRUE)
        endif()
    endif()
    set(${out_var} "${_supports}" PARENT_SCOPE)
endfunction()

if(NOT AMDCXX)
    set(RJ_ENABLE_HIP_TESTS OFF)
    message(STATUS "HIP tests disabled (amdclang++ not found)")
elseif(NOT HSA_RUNTIME64)
    set(RJ_ENABLE_HIP_TESTS OFF)
    message(STATUS "HIP tests disabled (libhsa-runtime64 not found)")
elseif(NOT HIP_RUNTIME64)
    set(RJ_ENABLE_HIP_TESTS OFF)
    message(STATUS "HIP tests disabled (libamdhip64 not found)")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND RJ_SANITIZER_LINK_OPTIONS)
    set(RJ_ENABLE_HIP_TESTS OFF)
    message(STATUS "HIP tests disabled (GNU host compiler with sanitizers)")
endif()

if(RJ_ENABLE_HIP_TESTS)
    rj_hip_runtime_supports_arch(
        RJ_HIP_RUNTIME_SUPPORTS_TEST_ARCH
        "${RJ_HIP_TEST_ARCH}"
    )
    rj_hip_runtime_supports_arch(RJ_HIP_RUNTIME_SUPPORTS_GFX950 "gfx950")
    rj_hip_runtime_supports_arch(RJ_HIP_RUNTIME_SUPPORTS_GFX942 "gfx942")
endif()

if(RJ_ENABLE_HIP_TESTS)
    set(GTEST_INC ${googletest_SOURCE_DIR}/googletest/include)
    set(GTEST_LIB_DIR ${CMAKE_BINARY_DIR}/lib)
    set(RJ_HIP_SANITIZER_FLAGS)
    foreach(_rj_sanitizer_flag IN LISTS RJ_SANITIZER_COMPILE_OPTIONS)
        list(APPEND RJ_HIP_SANITIZER_FLAGS -Xarch_host ${_rj_sanitizer_flag})
    endforeach()
    set(RJ_HIP_SANITIZER_LINK_FLAGS)
    foreach(_rj_sanitizer_flag IN LISTS RJ_SANITIZER_LINK_OPTIONS)
        list(
            APPEND RJ_HIP_SANITIZER_LINK_FLAGS
            -Xarch_host
            ${_rj_sanitizer_flag}
        )
    endforeach()

    # Helper: build a HIP test binary with amdclang++ + gtest.
    function(rj_add_hip_test TEST_NAME)
        cmake_parse_arguments(
            ARG
            ""
            "ARCH;SOURCE"
            "DEPENDENCIES;EXTRA_LIBS"
            ${ARGN}
        )
        if(NOT ARG_ARCH)
            set(ARG_ARCH ${RJ_HIP_TEST_ARCH})
        endif()
        if(ARG_SOURCE)
            set(HIP_TEST_SRC ${CMAKE_CURRENT_SOURCE_DIR}/${ARG_SOURCE})
        else()
            set(HIP_TEST_SRC ${CMAKE_CURRENT_SOURCE_DIR}/${TEST_NAME}.cpp)
        endif()
        set(HIP_TEST_BIN ${CMAKE_CURRENT_BINARY_DIR}/${TEST_NAME})
        set(HIP_EXTRA_LIBS)
        if(ARG_EXTRA_LIBS)
            set(HIP_EXTRA_LIBS -x none ${ARG_EXTRA_LIBS})
        endif()
        add_custom_command(
            OUTPUT ${HIP_TEST_BIN}
            COMMAND
                ${AMDCXX} -x hip --offload-arch=${ARG_ARCH}
                --rocm-path=${ROCM_PATH} -std=c++20 -O2
                ${RJ_HIP_SANITIZER_FLAGS} -isystem ${GTEST_INC}
                -I${CMAKE_CURRENT_SOURCE_DIR} -o ${HIP_TEST_BIN} ${HIP_TEST_SRC}
                ${HIP_EXTRA_LIBS} -L${GTEST_LIB_DIR} -lgtest -lpthread
                -Wl,-rpath,${GTEST_LIB_DIR} -Wl,-rpath,${RJ_HSA_LIBRARY_DIR}
                ${RJ_HIP_SANITIZER_LINK_FLAGS}
            DEPENDS
                ${HIP_TEST_SRC}
                ${AMDCXX}
                ${ARG_DEPENDENCIES}
                GTest::gtest
                GTest::gtest_main
            COMMENT "Building ${TEST_NAME} with amdclang++ -x hip (${ARG_ARCH})"
            VERBATIM
        )
        add_custom_target(${TEST_NAME}_target ALL DEPENDS ${HIP_TEST_BIN})
        add_dependencies(${TEST_NAME}_target rocjitsu_bin rocjitsu_shared)
        if(RJ_INSTALL_TESTS)
            rj_install_test_program(${HIP_TEST_BIN})
        endif()
    endfunction()

    # Helper: register a HIP test case run through the CLI launcher.
    function(rj_add_hip_test_case TEST_BIN TEST_CASE)
        cmake_parse_arguments(
            ARG
            ""
            "CONFIG;INSTALL_CONFIG;NAME;TIMEOUT"
            "ENVIRONMENT"
            ${ARGN}
        )
        if(NOT ARG_CONFIG)
            set(ARG_CONFIG ${RJ_CDNA4_KMD_CONFIG})
        endif()
        if(NOT ARG_NAME)
            set(ARG_NAME ${TEST_CASE})
        endif()
        if(ARG_INSTALL_CONFIG)
            set(_install_config "${ARG_INSTALL_CONFIG}")
        else()
            get_filename_component(_install_config_name "${ARG_CONFIG}" NAME)
            set(_install_config
                "\${RJ_INSTALLED_CONFIG_DIR}/${_install_config_name}"
            )
        endif()
        set(_install_env)
        foreach(_env_entry IN LISTS ARG_ENVIRONMENT)
            if(_env_entry MATCHES "^ROCJITSU_HIPKITTENS_KERNEL_DIR=")
                # The build-tree value points at KERNEL_OUTPUT_DIR, which does not
                # exist in an install tree. The hipkittens .hsaco fixtures are
                # installed alongside the other kernels, so point the installed
                # test at the installed kernel directory instead.
                list(
                    APPEND _install_env
                    "ROCJITSU_HIPKITTENS_KERNEL_DIR=\${RJ_INSTALLED_KERNEL_DIR}"
                )
            elseif(
                _env_entry MATCHES "^(ROCJITSU_RUNTIME_DIR|RJ_SINK_DIR)=(.*)$"
            )
                set(_env_name "${CMAKE_MATCH_1}")
                set(_env_value "${CMAKE_MATCH_2}")
                if(IS_ABSOLUTE "${_env_value}")
                    file(
                        RELATIVE_PATH
                        _env_relpath
                        "${CMAKE_CURRENT_BINARY_DIR}"
                        "${_env_value}"
                    )
                    if(_env_relpath MATCHES "^\\.\\.")
                        get_filename_component(
                            _env_relpath
                            "${_env_value}"
                            NAME
                        )
                    endif()
                    list(
                        APPEND _install_env
                        "${_env_name}=runtime/${_env_relpath}"
                    )
                else()
                    list(APPEND _install_env "${_env_entry}")
                endif()
            else()
                list(APPEND _install_env "${_env_entry}")
            endif()
        endforeach()
        set(_test_args
            NAME
            ${ARG_NAME}
            COMMAND
            ${CMAKE_COMMAND}
            "-DTEST_EXECUTABLE=${CMAKE_CURRENT_BINARY_DIR}/${TEST_BIN}"
            "-DTEST_FILTER=${TEST_CASE}"
            "-DTEST_LAUNCHER=${RJ_CLI}"
            "-DTEST_CONFIG=${ARG_CONFIG}"
            -P
            "${PROJECT_SOURCE_DIR}/cmake/rj_run_filtered_gtest.cmake"
            INSTALL_COMMAND
            "${CMAKE_COMMAND}"
            "-DTEST_EXECUTABLE=bin/${TEST_BIN}"
            "-DTEST_FILTER=${TEST_CASE}"
            "-DTEST_LAUNCHER=\${RJ_INSTALLED_BINDIR}/rocjitsu"
            "-DTEST_CONFIG=${_install_config}"
            "-P"
            "rj_run_filtered_gtest.cmake"
        )
        if(RJ_INSTALL_TESTS AND _install_env)
            list(APPEND _test_args INSTALL_ENVIRONMENT ${_install_env})
        endif()
        if(ARG_ENVIRONMENT)
            list(APPEND _test_args ENVIRONMENT ${ARG_ENVIRONMENT})
        endif()
        if(ARG_TIMEOUT)
            list(APPEND _test_args TIMEOUT ${ARG_TIMEOUT})
        endif()
        rj_add_test(${_test_args})
    endfunction()

    if(RJ_HIP_RUNTIME_SUPPORTS_TEST_ARCH)
        rj_add_hip_test(hip_memcpy_test ARCH ${RJ_HIP_TEST_ARCH})
        rj_add_hip_test_case(
            hip_memcpy_test
            "HipMemcpyTest.RoundTripFloat"
            CONFIG ${RJ_HIP_TEST_CONFIG}
        )
        rj_add_hip_test_case(
            hip_memcpy_test
            "HipMemcpyTest.RoundTripInt"
            CONFIG ${RJ_HIP_TEST_CONFIG}
        )
        # The only case that reaches ROCr's pinned SDMA blit, and both sanitizers
        # report that library rather than rocjitsu. TSan sees the blit's memcpy
        # into the ring but not the release store on the doorbell that follows
        # it, because libhsa-runtime64 is not instrumented; the pair it forms
        # with the acquire load in poll_doorbells() is therefore invisible and
        # the fetch looks unordered. LSan reports ~5.7 KB allocated entirely
        # within that library, in frames with no rocjitsu caller. Neither is
        # actionable from here, and suppressing them by file would hide the same
        # classes of defect on the paths rocjitsu does own. The case still runs
        # in regular builds, in both local and daemon mode, which is where it
        # regression-tests the transfer.
        if(NOT RJ_ENABLE_ASAN AND NOT RJ_ENABLE_TSAN)
            rj_add_hip_test_case(
                hip_memcpy_test
                "HipMemcpyTest.RoundTripPageableAbovePinThreshold"
                CONFIG ${RJ_HIP_TEST_CONFIG}
            )
        endif()
        rj_add_hip_test_case(
            hip_memcpy_test
            "HipMemcpyTest.DeviceToDevice"
            CONFIG ${RJ_HIP_TEST_CONFIG}
        )
        rj_add_hip_test(hip_vector_add_test ARCH ${RJ_HIP_TEST_ARCH})
        rj_add_hip_test_case(
            hip_vector_add_test
            "HipVectorAddTest.CorrectResult"
            CONFIG ${RJ_HIP_TEST_CONFIG}
        )

        # --- Daemon RPC tests (gtest-driven) ---
        # A gtest fixture manages the daemon lifecycle: fork+exec the daemon in
        # SetUp, run the HIP test binary as a subprocess, tear down in TearDown.
        _rj_path_from_installed_test_bin(
            _rj_installed_daemon_config
            "${CMAKE_INSTALL_DATADIR}/rocjitsu/configs"
            "${RJ_HIP_TEST_CONFIG_NAME}"
        )
        _rj_path_from_installed_test_bin(
            _rj_installed_daemon_config_2gpu
            "${CMAKE_INSTALL_DATADIR}/rocjitsu/configs"
            "${RJ_HIP_TEST_2GPU_CONFIG_NAME}"
        )
        _rj_path_from_installed_test_bin(
            _rj_installed_preload_lib
            "${CMAKE_INSTALL_LIBDIR}"
            "${CMAKE_SHARED_LIBRARY_PREFIX}rocjitsu${CMAKE_SHARED_LIBRARY_SUFFIX}"
        )

        # Daemon-mode plugin coverage uses a placeholder sink directory that
        # the test rewrites to a private runtime path.
        set(RJ_TEST_SINK_DIR "__RJ_DAEMON_SINK_DIR__")
        configure_file(
            ${CMAKE_CURRENT_SOURCE_DIR}/logging/logging_test_config.json.in
            ${CMAKE_CURRENT_BINARY_DIR}/daemon_logging_config.json
            @ONLY
        )
        target_compile_definitions(
            rocjitsu_tests
            PRIVATE
                RJ_DAEMON_LOGGING_CONFIG_PATH="${CMAKE_CURRENT_BINARY_DIR}/daemon_logging_config.json"
        )
        if(RJ_INSTALL_TESTS)
            install(
                FILES ${CMAKE_CURRENT_BINARY_DIR}/daemon_logging_config.json
                DESTINATION ${CMAKE_INSTALL_DATADIR}/rocjitsu/configs
            )
        endif()
        _rj_path_from_installed_test_bin(
            _rj_installed_daemon_logging_config
            "${CMAKE_INSTALL_DATADIR}/rocjitsu/configs"
            "daemon_logging_config.json"
        )

        add_executable(daemon_test daemon_test.cpp)
        target_compile_definitions(
            daemon_test
            PRIVATE
                RJ_DAEMON_BIN="$<TARGET_FILE:rocjitsu_bin>"
                RJ_DAEMON_CONFIG="${RJ_HIP_TEST_CONFIG}"
                RJ_DAEMON_CONFIG_2GPU="${RJ_HIP_TEST_2GPU_CONFIG}"
                RJ_DAEMON_SANITIZER_PRELOAD="${RJ_SANITIZER_PRELOAD_PREFIX}"
                RJ_PRELOAD_LIB="$<TARGET_FILE:rocjitsu_shared>"
                RJ_HIP_VECTOR_ADD_BIN="${CMAKE_CURRENT_BINARY_DIR}/hip_vector_add_test"
                RJ_HIP_MEMCPY_BIN="${CMAKE_CURRENT_BINARY_DIR}/hip_memcpy_test"
                RJ_HIP_RCCL_BIN="${CMAKE_CURRENT_BINARY_DIR}/hip_rccl_test"
                RJ_INTERPOSER_DUP_BIN="$<TARGET_FILE:interposer_dup_test>"
                RJ_DAEMON_LOGGING_CONFIG="${CMAKE_CURRENT_BINARY_DIR}/daemon_logging_config.json"
                RJ_DAEMON_LOGGING_SINK_TOKEN="__RJ_DAEMON_SINK_DIR__"
                RJ_INSTALLED_DAEMON_BIN="${_rj_installed_rocjitsu_bin}"
                RJ_INSTALLED_DAEMON_CONFIG="${_rj_installed_daemon_config}"
                RJ_INSTALLED_DAEMON_CONFIG_2GPU="${_rj_installed_daemon_config_2gpu}"
                RJ_INSTALLED_PRELOAD_LIB="${_rj_installed_preload_lib}"
                RJ_INSTALLED_HIP_VECTOR_ADD_BIN="hip_vector_add_test"
                RJ_INSTALLED_HIP_MEMCPY_BIN="hip_memcpy_test"
                RJ_INSTALLED_HIP_RCCL_BIN="hip_rccl_test"
                RJ_INSTALLED_INTERPOSER_DUP_BIN="interposer_dup_test"
                RJ_INSTALLED_DAEMON_LOGGING_CONFIG="${_rj_installed_daemon_logging_config}"
        )
        target_link_libraries(daemon_test PRIVATE GTest::gtest_main)
        add_dependencies(
            daemon_test
            rocjitsu_bin
            hip_vector_add_test_target
            hip_memcpy_test_target
            interposer_dup_test
        )
        if(RJ_INSTALL_TESTS)
            rj_install_test_target(daemon_test)
        endif()
        set(RJ_DAEMON_TESTS
            RocjitsuCliDaemon.LaunchesApplicationAfterDaemonIsReady
            DaemonTest.HipVectorAdd
            DaemonTest.AttachExecFailurePreservesDaemonSocket
            DaemonTest.HipMemcpyRoundTripFloat
            DaemonTest.HipMemcpyRoundTripInt
            DaemonTest.HipMemcpyDeviceToDevice
            DaemonTest.TwoIndependentClients
            DaemonTest.InterposerDupReopenAfterPrimaryOverwriteRemote
            DaemonTest.InterposerDupSerializedReopenUnderContentionRemote
            DaemonTest.InterposerDupKeepsRoutingAfterPrimaryCloseRemote
            DaemonTest.InterposerDup2OverPrimaryInvalidatesRemote
            DaemonTest.InterposerProcMapsNamesRemoteKfdMarker
        )
        # Excluded under sanitizers for the reason given at its local-mode
        # registration above: the pinned SDMA path this case exists to cover
        # runs inside an uninstrumented libhsa-runtime64, and both the race TSan
        # infers and the allocations LSan finds are that library's.
        if(NOT RJ_ENABLE_ASAN AND NOT RJ_ENABLE_TSAN)
            list(
                APPEND RJ_DAEMON_TESTS
                DaemonTest.HipMemcpyRoundTripPageableAbovePinThreshold
            )
        endif()
        foreach(TC IN LISTS RJ_DAEMON_TESTS)
            rj_add_test(
                NAME ${TC}
                COMMAND daemon_test --gtest_filter=${TC}
                TIMEOUT 120
                LABELS daemon
                INSTALL_COMMAND
                    "bin/daemon_test"
                    "--gtest_filter=${TC}"
            )
        endforeach()

        rj_add_test(
            NAME DaemonTest.LoggingPlugin
            COMMAND daemon_test --gtest_filter=DaemonPluginTest.LoggingPluginDispatchLogged
            TIMEOUT 120
            LABELS daemon
            INSTALL_COMMAND
                "bin/daemon_test"
                "--gtest_filter=DaemonPluginTest.LoggingPluginDispatchLogged"
        )

        # --- RCCL collective test ---
        find_library(RCCL_LIB rccl PATHS "${ROCM_PATH}" PATH_SUFFIXES lib)
        if(RCCL_LIB AND RJ_HIP_RUNTIME_SUPPORTS_GFX950)
            rj_add_hip_test(hip_rccl_test ARCH gfx950 EXTRA_LIBS ${RCCL_LIB})
            add_dependencies(daemon_test hip_rccl_test_target)
            set(RJ_RCCL_TESTS
                RcclDaemonTest.AllReduce
                RcclDaemonTest.Broadcast
                RcclDaemonTest.AllGather
                RcclDaemonTest.ReduceScatter
                RcclDaemonTest.SendRecv
            )
            foreach(TC IN LISTS RJ_RCCL_TESTS)
                rj_add_test(
                    NAME ${TC}
                    COMMAND daemon_test --gtest_filter=${TC}
                    TIMEOUT 180
                    LABELS daemon rccl
                    INSTALL_COMMAND
                        "bin/daemon_test"
                        "--gtest_filter=${TC}"
                )
            endforeach()
            message(STATUS "RCCL collective test enabled (found ${RCCL_LIB})")
        elseif(RCCL_LIB)
            message(
                STATUS
                "RCCL collective test disabled (libamdhip64 does not advertise gfx950 support)"
            )
        else()
            message(STATUS "RCCL collective test disabled (librccl not found)")
        endif()
    else()
        message(
            STATUS
            "HIP smoke/daemon tests disabled (libamdhip64 does not advertise ${RJ_HIP_TEST_ARCH} support)"
        )
    endif()

    if(RJ_HIP_RUNTIME_SUPPORTS_GFX950)
        # This test exercises CDNA ACCVGPR/DS transpose instructions directly
        # in inline assembly, so it must stay on a CDNA target even when the
        # generic HIP smoke tests are built for a local RDNA host such as
        # gfx1201.
        rj_add_hip_test(hip_ds_transpose_acc_test ARCH gfx950)

        rj_add_hip_test(
            cdna4_to_cdna3_lds_hip_test
            ARCH gfx950
            SOURCE dbt/cdna4_to_cdna3_lds_hip_test.cpp
        )
        add_dependencies(cdna4_to_cdna3_lds_hip_test_target rocjitsu_hooks)
        rj_add_hip_test(
            cdna4_to_cdna3_hipkittens_test
            ARCH gfx950
            SOURCE dbt/cdna4_to_cdna3_hipkittens_test.cpp
        )
        add_dependencies(
            cdna4_to_cdna3_hipkittens_test_target
            device_kernels
            rocjitsu_hooks
        )
        if(DEFINED HAS_CDNA3_GPU AND HAS_CDNA3_GPU)
            foreach(
                TC
                IN
                ITEMS
                    LargeStaticLdsCopyLowOffset
                    LargeStaticLdsCopyHighOffset
                    LargeStaticLdsCopyHighOffsetWithBorrowedHighSgpr
                    LargeStaticLdsReverseHighOffset
                    LargeStaticLdsMultiBlockHighOffset
                    LargeStaticLdsCrossWaveExchangeHighOffset
                    LargeStaticLdsB128ReverseHighOffset
                    LargeStaticLdsB128ReverseImmediateOffset
                    LargeStaticLdsReadB64TrB16
                    LargeStaticLdsZeroKernargWrapperState
                    Static64Dynamic64ZeroKernargWrapperState
                    Static64Dynamic64ZeroKernargWrapperStateWithBorrowedHighSgpr
                    MiopenC220ZeroKernargWrapperState
                    MiopenC220ZeroKernargWrapperStateWithBorrowedHighSgpr
                    MiopenC220V116ZeroKernargWrapperState
                    MiopenC220V116ZeroKernargWrapperStateWithBorrowedHighSgpr
                    Static64Dynamic64DispatchPointerWrapperState
                    Static64Dynamic64DispatchPointerWrapperStateWithBorrowedHighSgpr
            )
                rj_add_hip_test_case(
                    cdna4_to_cdna3_lds_hip_test
                    "HipLdsCopyDbtTest.${TC}"
                    CONFIG ${CMAKE_SOURCE_DIR}/configs/guest_gfx950_on_gfx942.json
                    NAME "HipLdsCopyDbtTest.${TC}"
                    TIMEOUT 120
                )
            endforeach()
            foreach(
                TC
                IN
                ITEMS
                    With16x32DispatchesAndMatchesReference
                    With32x16DispatchesAndMatchesReference
            )
                rj_add_hip_test_case(
                    cdna4_to_cdna3_hipkittens_test
                    "HipKittensBf16MatmulDbtTest.${TC}"
                    CONFIG ${CMAKE_SOURCE_DIR}/configs/guest_gfx950_on_gfx942.json
                    NAME "HipKittensBf16MatmulDbtTest.${TC}"
                    TIMEOUT 240
                    ENVIRONMENT "ROCJITSU_HIPKITTENS_KERNEL_DIR=${KERNEL_OUTPUT_DIR}"
                )
            endforeach()
        endif()

        rj_add_hip_test(hip_bfe_2d_test ARCH gfx950)
        rj_add_hip_test_case(hip_bfe_2d_test "BfeRegressionTest.ThreadIdxY")

        # --- Logging plugin end-to-end test ---
        rj_add_hip_test(
            hip_logging_test
            ARCH gfx950
            SOURCE logging/hip_logging_test.hip
        )
        set(LOGGING_SINK_DIR ${CMAKE_CURRENT_BINARY_DIR}/logging_test_output)
        set(LOGGING_RUNTIME_DIR ${LOGGING_SINK_DIR}/runtime)
        file(MAKE_DIRECTORY ${LOGGING_SINK_DIR})
        file(MAKE_DIRECTORY ${LOGGING_RUNTIME_DIR})
        set(RJ_TEST_SINK_DIR ${LOGGING_SINK_DIR})
        configure_file(
            ${CMAKE_CURRENT_SOURCE_DIR}/logging/logging_test_config.json.in
            ${LOGGING_SINK_DIR}/config.json
            @ONLY
        )
        set(RJ_TEST_SINK_DIR runtime/logging_test_output)
        set(LOGGING_INSTALLED_CONFIG_DIR
            ${CMAKE_CURRENT_BINARY_DIR}/install-configs
        )
        set(LOGGING_INSTALLED_CONFIG
            ${LOGGING_INSTALLED_CONFIG_DIR}/logging.json
        )
        file(MAKE_DIRECTORY ${LOGGING_INSTALLED_CONFIG_DIR})
        configure_file(
            ${CMAKE_CURRENT_SOURCE_DIR}/logging/logging_test_config.json.in
            ${LOGGING_INSTALLED_CONFIG}
            @ONLY
        )
        target_compile_definitions(
            rocjitsu_tests
            PRIVATE
                RJ_LOGGING_CONFIG_PATH="${LOGGING_SINK_DIR}/config.json"
                RJ_INSTALLED_LOGGING_CONFIG_PATH="${LOGGING_INSTALLED_CONFIG}"
        )
        if(RJ_INSTALL_TESTS)
            install(
                FILES ${LOGGING_INSTALLED_CONFIG}
                DESTINATION ${CMAKE_INSTALL_DATADIR}/rocjitsu/configs/tests
            )
        endif()
        rj_add_hip_test_case(
            hip_logging_test "LoggingTest.dispatch_logged"
            CONFIG ${LOGGING_SINK_DIR}/config.json
            INSTALL_CONFIG "\${RJ_INSTALLED_CONFIG_DIR}/tests/logging.json"
            TIMEOUT 30
            ENVIRONMENT
                "ROCJITSU_RUNTIME_DIR=${LOGGING_RUNTIME_DIR}"
                "RJ_SINK_DIR=${LOGGING_SINK_DIR}"
        )

        add_subdirectory(race-detector)
    else()
        message(
            STATUS
            "CDNA4 HIP tests disabled (libamdhip64 does not advertise gfx950 support)"
        )
    endif()

    message(STATUS "HIP tests enabled (found ${AMDCXX})")
endif()

# --- Narrow conversion HIP CTS tests ---
# Uses the same HIP compiler path as the other CDNA4 HIP tests.
if(RJ_ENABLE_HIP_TESTS AND RJ_HIP_RUNTIME_SUPPORTS_GFX950)
    set(CVT_NARROW_BIN ${CMAKE_CURRENT_BINARY_DIR}/hip_cvt_narrow_test)
    add_custom_command(
        OUTPUT ${CVT_NARROW_BIN}
        COMMAND
            ${AMDCXX} -x hip --offload-arch=gfx950 --rocm-path=${ROCM_PATH}
            -std=c++20 ${RJ_HIP_SANITIZER_FLAGS}
            "-I$<JOIN:$<TARGET_PROPERTY:util,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            -o ${CVT_NARROW_BIN}
            ${CMAKE_CURRENT_SOURCE_DIR}/hip_cvt_narrow_test.cpp
            ${RJ_HIP_SANITIZER_LINK_FLAGS}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/hip_cvt_narrow_test.cpp ${AMDCXX}
        COMMENT "Building hip_cvt_narrow_test with amdclang++ -x hip (gfx950)"
        VERBATIM
    )
    add_custom_target(hip_cvt_narrow_test_target ALL DEPENDS ${CVT_NARROW_BIN})
    add_dependencies(hip_cvt_narrow_test_target rocjitsu_bin rocjitsu_shared)
    if(RJ_INSTALL_TESTS)
        rj_install_test_program(${CVT_NARROW_BIN})
    endif()

    rj_add_test(
        NAME "CvtNarrow.AllTests"
        COMMAND ${RJ_CLI} --config ${RJ_CDNA4_KMD_CONFIG} -- ${CVT_NARROW_BIN}
        TIMEOUT 300
        INSTALL_COMMAND
            "\${RJ_INSTALLED_BINDIR}/rocjitsu"
            "--config"
            "\${RJ_INSTALLED_CONFIG_DIR}/${RJ_CDNA4_KMD_CONFIG_NAME}"
            "--"
            "bin/hip_cvt_narrow_test"
    )
endif()

# --- rocgdb wave-debugging smoke test (requires rocgdb + hipcc + mirage) ---
# Drives real ROCgdb through `mirage run` to set a GPU kernel breakpoint, stop,
# inspect wave registers, and continue to a correct result (see
# tests/rocgdb/debug_kernel.sh). Gated on the debugger and HIP compiler being
# present; the harness itself skips (printing "SKIP:") if mirage is not found.
find_program(
    ROCGDB_EXECUTABLE
    NAMES rocgdb
    HINTS "${ROCM_PATH}"
    PATH_SUFFIXES bin
)
find_program(
    HIPCC_EXECUTABLE
    NAMES hipcc
    HINTS "${ROCM_PATH}"
    PATH_SUFFIXES bin
)
# NO_DEFAULT_PATH: only an in-tree Mirage qualifies. An unrelated one installed
# on PATH would be a different build than the librocjitsu.so staged below, and
# the mismatch surfaces as an opaque "enable_debug failed (STATUS_ERROR)" rather
# than as a version error.
#
# A fresh cache entry name (not MIRAGE_EXECUTABLE) is deliberate: find_program is
# a no-op when its cache variable is already set, so any build tree configured
# while this used HINTS -- which does not suppress the default PATH search --
# would keep an out-of-tree hit forever and NO_DEFAULT_PATH would never apply to
# the very trees it was added for.
#
# build/manylinux/bin is where emulation/mirage/scripts/mirage.sh installs, so it
# belongs beside the two cargo target dirs; PATH_SUFFIXES covers it without
# reopening the default search.
find_program(
    RJ_MIRAGE_EXECUTABLE
    NAMES mirage
    PATHS
        "${CMAKE_SOURCE_DIR}/../mirage/target/release"
        "${CMAKE_SOURCE_DIR}/../mirage/target/debug"
        "${CMAKE_SOURCE_DIR}/../mirage/build/manylinux"
    PATH_SUFFIXES bin
    NO_DEFAULT_PATH
)
set(MIRAGE_EXECUTABLE "${RJ_MIRAGE_EXECUTABLE}")
# Mirage is as much a prerequisite as rocgdb and hipcc: without it the harness
# has nothing to run the debuggee under. Registering the test anyway produced a
# test that could only skip, or -- worse -- one that ran a Mirage which then
# loaded whatever librocjitsu.so it happened to find, or none at all and exited
# 127 long before reaching ROCgdb.
# A Mirage binary older than Mirage's own sources is not usable here: this
# branch changes the ctl and rocjitsu_sys crates, and a prebuilt binary that
# predates them fails attachment with an opaque "enable_debug failed
# (STATUS_ERROR)". CMake deliberately does not drive cargo, so refuse to
# register rather than test a binary that does not match the tree, and say what
# to run. Configure-time only, so `cargo build` needs a re-run of cmake -- which
# the message states.
#
# No CONFIGURE_DEPENDS: it re-runs cmake when the *set* of matched files changes,
# never when an existing file's mtime does, which is the only thing this gate
# reads -- so it cannot fire for the case that matters and costs a recursive walk
# of the whole mirage tree on every build.
set(_rocgdb_mirage_stale FALSE)
if(MIRAGE_EXECUTABLE)
    # mirage is a workspace whose root package is the `mirage` binary itself
    # (src/main.rs, declared by the top-level Cargo.toml), so the per-crate
    # patterns alone would miss the entry point most likely to change.
    file(
        GLOB_RECURSE _rocgdb_mirage_sources
        "${CMAKE_SOURCE_DIR}/../mirage/*/src/*.rs"
        "${CMAKE_SOURCE_DIR}/../mirage/*/Cargo.toml"
        "${CMAKE_SOURCE_DIR}/../mirage/src/*.rs"
        "${CMAKE_SOURCE_DIR}/../mirage/build.rs"
        "${CMAKE_SOURCE_DIR}/../mirage/Cargo.toml"
        "${CMAKE_SOURCE_DIR}/../mirage/Cargo.lock"
    )
    # Phrased as "the binary is not newer than the source" rather than the
    # reverse: CMake's IS_NEWER_THAN is also true when the two timestamps are
    # *equal*, so testing the source directly calls a tie stale. Ties are the
    # common case after a checkout, a container copy, or a CI cache restore, and
    # calling them stale silently deregisters the test with a message telling the
    # developer to run a build they already ran.
    foreach(_src IN LISTS _rocgdb_mirage_sources)
        if(NOT "${MIRAGE_EXECUTABLE}" IS_NEWER_THAN "${_src}")
            set(_rocgdb_mirage_stale TRUE)
            break()
        endif()
    endforeach()
endif()

if(
    ROCGDB_EXECUTABLE
    AND HIPCC_EXECUTABLE
    AND MIRAGE_EXECUTABLE
    AND NOT _rocgdb_mirage_stale
)
    # Stage the librocjitsu.so this build just produced into a stub ROCm root
    # and point Mirage at it. $ROCM_HOME/lib is one of Mirage's documented
    # search locations (emulation/mirage/rocjitsu/src/lib.rs kmd_search_dirs),
    # and the search is additive, so a stub root cannot shadow anything else.
    set(_rocgdb_rocm_home ${CMAKE_CURRENT_BINARY_DIR}/rocgdb_rocm_home)
    # Stage Mirage next to the library as <prefix>/bin/mirage + <prefix>/lib.
    # kmd_search_dirs (emulation/mirage/rocjitsu/src/lib.rs) probes
    # <exe_dir>/../..{,/..}/rocjitsu/build first and only then <exe_dir>/../lib,
    # with $ROCM_HOME/lib after both. Run in place from
    # emulation/mirage/target/debug, the third of those probes resolves to
    # emulation/rocjitsu/build -- the developer's own tree -- so ROCM_HOME never
    # got a look in. Running the staged copy puts none of the rocjitsu/build
    # probes on a real directory, and <exe_dir>/../lib is this build's staged
    # librocjitsu.so.
    add_custom_command(
        OUTPUT
            ${_rocgdb_rocm_home}/lib/librocjitsu.so
            ${_rocgdb_rocm_home}/bin/mirage
        COMMAND ${CMAKE_COMMAND} -E make_directory ${_rocgdb_rocm_home}/lib
        COMMAND ${CMAKE_COMMAND} -E make_directory ${_rocgdb_rocm_home}/bin
        COMMAND
            ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:rocjitsu_shared>
            ${_rocgdb_rocm_home}/lib/librocjitsu.so
        COMMAND
            ${CMAKE_COMMAND} -E copy_if_different ${MIRAGE_EXECUTABLE}
            ${_rocgdb_rocm_home}/bin/mirage
        # copy_if_different leaves the destination's mtime alone when the bytes
        # already match, so without this the outputs stay older than their inputs
        # and the generator re-issues all four commands -- a byte-compare of a
        # 69 MB binary and the shared library -- on every incremental build.
        COMMAND
            ${CMAKE_COMMAND} -E touch ${_rocgdb_rocm_home}/lib/librocjitsu.so
            ${_rocgdb_rocm_home}/bin/mirage
        DEPENDS rocjitsu_shared ${MIRAGE_EXECUTABLE}
        COMMENT
            "Staging mirage + librocjitsu.so for the rocgdb debug smoke test"
        VERBATIM
    )
    add_custom_target(
        rocgdb_demo_rocm_home
        ALL
        DEPENDS
            ${_rocgdb_rocm_home}/lib/librocjitsu.so
            ${_rocgdb_rocm_home}/bin/mirage
    )

    rj_add_test(
        NAME "RocgdbDebug.KernelBreakpoint"
        COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/rocgdb/debug_kernel.sh"
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        TIMEOUT 300
        ENVIRONMENT
            "MIRAGE_BIN=${_rocgdb_rocm_home}/bin/mirage"
            "ROCM_HOME=${_rocgdb_rocm_home}"
            "MIRAGE_PROFILE=mi350x"
            "OFFLOAD_ARCH=gfx950"
            "XDG_CONFIG_HOME=${CMAKE_CURRENT_BINARY_DIR}/rocgdb_mirage_gfx950/config"
            "XDG_STATE_HOME=${CMAKE_CURRENT_BINARY_DIR}/rocgdb_mirage_gfx950/state"
        SKIP_REGULAR_EXPRESSION "SKIP:"
    )
    rj_add_test(
        NAME "RocgdbDebug.Gfx1250KernelBreakpoint"
        COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/rocgdb/debug_kernel.sh"
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        TIMEOUT 300
        ENVIRONMENT
            "MIRAGE_BIN=${_rocgdb_rocm_home}/bin/mirage"
            "ROCM_HOME=${_rocgdb_rocm_home}"
            "MIRAGE_PROFILE=mi450x"
            "OFFLOAD_ARCH=gfx1250"
            "XDG_CONFIG_HOME=${CMAKE_CURRENT_BINARY_DIR}/rocgdb_mirage_gfx1250/config"
            "XDG_STATE_HOME=${CMAKE_CURRENT_BINARY_DIR}/rocgdb_mirage_gfx1250/state"
        SKIP_REGULAR_EXPRESSION "SKIP:"
    )
    message(
        STATUS
        "rocgdb debug smoke test enabled (found ${ROCGDB_EXECUTABLE})"
    )
else()
    # One message naming every unmet reason. Reporting only staleness when rocgdb
    # is also missing sends the developer to `cargo build`, after which the test
    # still does not register and nothing new is printed to say why.
    set(_rocgdb_missing "")
    if(NOT ROCGDB_EXECUTABLE)
        list(APPEND _rocgdb_missing "rocgdb")
    endif()
    if(NOT HIPCC_EXECUTABLE)
        list(APPEND _rocgdb_missing "hipcc")
    endif()
    if(NOT MIRAGE_EXECUTABLE)
        list(APPEND _rocgdb_missing "mirage")
    endif()
    if(_rocgdb_mirage_stale)
        list(
            APPEND _rocgdb_missing
            "a mirage newer than its sources (run `cargo build` in emulation/mirage, then re-run cmake)"
        )
    endif()
    list(JOIN _rocgdb_missing ", " _rocgdb_missing)
    message(STATUS "rocgdb debug smoke test disabled (need ${_rocgdb_missing})")
endif()
