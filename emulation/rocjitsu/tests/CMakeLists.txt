# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

include(rj_configure_target)

# Device kernels (compiled from HIP sources for simulator testing).
add_subdirectory(kernels)

# Common object libraries linked by all test/tool executables.
set(RJ_OBJECT_LIBS
    rocjitsu_analysis
    rocjitsu_code
    rocjitsu_isa
    rocjitsu_isa_cdna1
    rocjitsu_isa_cdna2
    rocjitsu_isa_cdna3
    rocjitsu_isa_cdna4
    rocjitsu_isa_rdna1
    rocjitsu_isa_rdna2
    rocjitsu_isa_rdna3
    rocjitsu_isa_rdna3_5
    rocjitsu_isa_rdna4
    rocjitsu_isa_gfx1250
    rocjitsu_isa_risc_v
    simdojo
    rocjitsu_kmd
    rocjitsu_vm
    rocjitsu_vm_amdgpu
    rocjitsu_plugin_logging
    rocjitsu_plugin_race_detector
    rocjitsu_vm_risc_v
    rocjitsu_config
)

# --- Unit / integration test suite ---

add_executable(
    rocjitsu_tests
    rvsim_test.cpp
    amdgpu_vm_test.cpp
    config_test.cpp
    matmul_test.cpp
    vector_add_test.cpp
    gfx1250_sim_test.cpp
    simdojo_sim_test.cpp
    simulated_driver_test.cpp
    decode_smoke_test.cpp
    shared_infra_test.cpp
    transcendental_test.cpp
    analysis/liveness_test.cpp
    patch/spill_manager_test.cpp
    dbt/translate_test.cpp
    mtype_flags_test.cpp
    kfd_ioctl_test.cpp
    waitcnt_counter_test.cpp
    instruction_execution_harness_test.cpp
    decode_execute_benchmark.cpp
    util_simd_test.cpp
    v_add_simd_benchmark.cpp
    simd_correctness/vop1_cvt_f64_scalar_correctness_test.cpp
    simd_correctness/vopc_cmp_class_scalar_correctness_test.cpp
    simd_correctness/vop2_simd_correctness_test.cpp
    simd_correctness/vop2_carry_simd_correctness_test.cpp
    simd_correctness/vop2_fma_simd_correctness_test.cpp
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
    simd_correctness/vop3_fp16_unary_simd_correctness_test.cpp
    simd_correctness/vop3_fp16_transcendental_simd_correctness_test.cpp
    simd_correctness/vop3_cndmask_pack_simd_correctness_test.cpp
    simd_correctness/vop3_div_helpers_simd_correctness_test.cpp
    simd_correctness/vop3_b16_rdna_simd_correctness_test.cpp
    simd_correctness/vop3_ternary_fp_simd_correctness_test.cpp
    simd_correctness/vop3_fmac_simd_correctness_test.cpp
    simd_correctness/vop3_ldexp_simd_correctness_test.cpp
    simd_correctness/vop3p_pk_binary_int_simd_correctness_test.cpp
    simd_correctness/vop3p_pk_binary_fp16_simd_correctness_test.cpp
    simd_correctness/vop3p_fma_mix_simd_correctness_test.cpp
    simd_correctness/vop3p_mov_b32_simd_correctness_test.cpp
    simd_correctness/vop3p_dot_simd_correctness_test.cpp
    execution_plugin_test.cpp
    race-detector/race_detector_tests.cpp
    race-detector/interval_set_tests.cpp
    data_types_test.cpp
)
target_link_libraries(
    rocjitsu_tests
    PRIVATE
        ${RJ_OBJECT_LIBS}
        rocjitsu_kmd
        util
        flatbuffers
        Threads::Threads
        GTest::gtest_main
)
rj_configure_target(rocjitsu_tests TEST)

include(GoogleTest)
gtest_discover_tests(rocjitsu_tests)

add_executable(
    rj_dbt_translate_smoke_test
    tools/rj_dbt_translate_smoke_test.cpp
)
target_include_directories(
    rj_dbt_translate_smoke_test
    PRIVATE
        ${CMAKE_SOURCE_DIR}/lib/rocjitsu/include
        ${CMAKE_SOURCE_DIR}/lib/rocjitsu/src
)
target_link_libraries(rj_dbt_translate_smoke_test PRIVATE GTest::gtest)
rj_configure_target(rj_dbt_translate_smoke_test TEST)
add_dependencies(rj_dbt_translate_smoke_test rj_dbt_translate)
add_test(
    NAME "RjDbtTranslate.Smoke"
    COMMAND rj_dbt_translate_smoke_test $<TARGET_FILE:rj_dbt_translate>
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
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
# The CLI handles LD_PRELOAD and config discovery. No RESOURCE_LOCK needed.

set(RJ_CDNA4_KMD_CONFIG "${CMAKE_SOURCE_DIR}/configs/amdgpu_cdna4_kmd.json")
set(RJ_CDNA3_KMD_CONFIG "${CMAKE_SOURCE_DIR}/configs/amdgpu_cdna3_kmd.json")
set(RJ_CLI "$<TARGET_FILE:rocjitsu_bin>")

set(RJ_KMD_PRELOAD_ENV
    "LD_PRELOAD=$<TARGET_FILE:rocjitsu_kmd_shim>"
    "RJ_CONFIG=${CMAKE_SOURCE_DIR}/configs/amdgpu_cdna4.json"
)

# --- rocminfo test (requires rocminfo binary + HSA runtime + CLI launcher) ---
# Spawns rocminfo via the rocjitsu CLI and validates output.

find_program(
    ROCMINFO_EXECUTABLE
    NAMES rocminfo
    HINTS ${CMAKE_SOURCE_DIR}/.venv
    ENV ROCM_PATH
    /opt/rocm
    PATH_SUFFIXES bin
    DOC "Path to the rocminfo executable"
)
if(ROCMINFO_EXECUTABLE)
    add_executable(rocminfo_test rocminfo_test.cpp)
    target_compile_definitions(
        rocminfo_test
        PRIVATE
            ROCMINFO_PATH="${ROCMINFO_EXECUTABLE}"
            ROCJITSU_BIN="$<TARGET_FILE:rocjitsu_bin>"
            RJ_CONFIG_PATH="${RJ_CDNA4_KMD_CONFIG}"
    )
    target_link_libraries(rocminfo_test PRIVATE GTest::gtest_main)
    add_dependencies(rocminfo_test rocjitsu_bin rocjitsu_kmd_shim)

    add_test(
        NAME "RocminfoTest.ExitsSuccessfully"
        COMMAND rocminfo_test --gtest_filter=RocminfoTest.ExitsSuccessfully
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
    add_test(
        NAME "RocminfoTest.InterposerActive"
        COMMAND rocminfo_test --gtest_filter=RocminfoTest.InterposerActive
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
    add_test(
        NAME "RocminfoTest.HsaSystemAttributes"
        COMMAND rocminfo_test --gtest_filter=RocminfoTest.HsaSystemAttributes
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
    add_test(
        NAME "RocminfoTest.DetectsGpuAgent"
        COMMAND rocminfo_test --gtest_filter=RocminfoTest.DetectsGpuAgent
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
    add_test(
        NAME "RocminfoTest.ReportsGfx950"
        COMMAND rocminfo_test --gtest_filter=RocminfoTest.ReportsGfx950
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
    add_test(
        NAME "RocminfoTest.ReportsWavefrontSize"
        COMMAND rocminfo_test --gtest_filter=RocminfoTest.ReportsWavefrontSize
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
    set_tests_properties(
        "RocminfoTest.ExitsSuccessfully"
        "RocminfoTest.InterposerActive"
        "RocminfoTest.HsaSystemAttributes"
        "RocminfoTest.DetectsGpuAgent"
        "RocminfoTest.ReportsGfx950"
        "RocminfoTest.ReportsWavefrontSize"
        PROPERTIES ENVIRONMENT "${RJ_KMD_PRELOAD_ENV}"
    )
    message(STATUS "rocminfo test enabled (found ${ROCMINFO_EXECUTABLE})")
else()
    message(STATUS "rocminfo test disabled (rocminfo not found)")
endif()

# --- HSA init test (requires ROCm HSA runtime + CLI launcher) ---

find_library(
    HSA_RUNTIME64
    hsa-runtime64
    PATHS /opt/rocm/lib
    ENV ROCM_PATH
    PATH_SUFFIXES lib
)
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
    add_dependencies(hsa_init_test rocjitsu_bin rocjitsu_kmd_shim)
    add_test(
        NAME "HsaTest.InitSucceeded"
        COMMAND
            ${RJ_CLI} --config ${RJ_CDNA4_KMD_CONFIG} --
            $<TARGET_FILE:hsa_init_test> --gtest_filter=HsaTest.InitSucceeded
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
    add_test(
        NAME "HsaTest.GpuAgentFound"
        COMMAND
            ${RJ_CLI} --config ${RJ_CDNA4_KMD_CONFIG} --
            $<TARGET_FILE:hsa_init_test> --gtest_filter=HsaTest.GpuAgentFound
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
    set_tests_properties(
        "HsaTest.InitSucceeded"
        "HsaTest.GpuAgentFound"
        PROPERTIES SKIP_REGULAR_EXPRESSION "\\[  SKIPPED \\]"
    )
    message(STATUS "HSA init test enabled (found ${HSA_RUNTIME64})")

    # --- HSA translate test: CDNA4→RDNA4 on-device verification ---
    # Requires both compiled device kernels AND an RDNA4 GPU on the host.
    if(HAS_DEVICE_KERNELS)
        # Detect supported DBT host GPUs at configure time via rocm_agent_enumerator.
        find_program(
            ROCM_AGENT_ENUM
            rocm_agent_enumerator
            HINTS
            ENV ROCM_PATH
            ${RJ_HSA_ROCM_ROOT}
            /opt/rocm
            PATH_SUFFIXES bin
        )
        set(HAS_DBT_VECTOR_ADD_GPU FALSE)
        set(HAS_CDNA3_GPU FALSE)
        set(HAS_RDNA4_GPU FALSE)
        set(HAS_GFX1201_GPU FALSE)
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
        endif()

        add_executable(hsa_translate_test dbt/hsa_translate_test.cpp)
        target_include_directories(
            hsa_translate_test
            PRIVATE
                ${CMAKE_SOURCE_DIR}/lib/rocjitsu/include
                ${CMAKE_SOURCE_DIR}/lib/rocjitsu/src
                ${CMAKE_SOURCE_DIR}/lib/util/include
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

        add_executable(hsa_hooks_test dbt/hsa_hooks_test.cpp)
        target_include_directories(
            hsa_hooks_test
            PRIVATE
                ${CMAKE_SOURCE_DIR}/lib/rocjitsu/include
                ${CMAKE_SOURCE_DIR}/lib/rocjitsu/src
                ${CMAKE_SOURCE_DIR}/lib/util/include
                ${RJ_HSA_INCLUDE_DIR}
        )
        target_link_libraries(
            hsa_hooks_test
            PRIVATE
                ${RJ_OBJECT_LIBS}
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
        add_dependencies(hsa_hooks_test device_kernels rocjitsu_hooks)
        rj_configure_target(hsa_hooks_test TEST)

        if(HAS_RDNA4_GPU)
            add_test(
                NAME "HsaTranslateTest.TranslateAndDispatchVectorAdd"
                COMMAND hsa_translate_test --gtest_filter=*VectorAdd*
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            )
        endif()

        if(HAS_GFX1201_GPU)
            add_test(
                NAME "HsaHooksTest.TranslateGfx950Mfma16x16ThroughToolsLib"
                COMMAND
                    hsa_hooks_test
                    --gtest_filter=HsaHooksTest.TranslateGfx950Mfma16x16ThroughToolsLib
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            )
            set_tests_properties(
                "HsaHooksTest.TranslateGfx950Mfma16x16ThroughToolsLib"
                PROPERTIES
                    TIMEOUT 60
                    ENVIRONMENT
                        "HSA_TOOLS_LIB=$<TARGET_FILE:rocjitsu_hooks>;RJ_DBT_TARGET_ISA=gfx1201;RJ_DBT_LOG=1"
            )
        endif()

        if(HAS_RDNA4_GPU)
            add_test(
                NAME "HsaTranslateTest.TranslateAndDispatchMfma16x16"
                COMMAND hsa_translate_test --gtest_filter=*Mfma16x16*
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            )
            message(STATUS "HSA translate test enabled (RDNA4 GPU detected)")
        else()
            message(
                STATUS
                "HSA translate test built but NOT registered with CTest (no RDNA4 GPU)"
            )
        endif()

        add_executable(
            cdna4_to_cdna3_dispatch_test
            dbt/cdna4_to_cdna3_dispatch_test.cpp
        )
        target_include_directories(
            cdna4_to_cdna3_dispatch_test
            PRIVATE
                ${CMAKE_SOURCE_DIR}/lib/rocjitsu/include
                ${CMAKE_SOURCE_DIR}/lib/rocjitsu/src
                ${CMAKE_SOURCE_DIR}/lib/util/include
                ${RJ_HSA_INCLUDE_DIR}
        )
        target_link_libraries(
            cdna4_to_cdna3_dispatch_test
            PRIVATE
                ${RJ_OBJECT_LIBS}
                util
                ${HSA_RUNTIME64}
                GTest::gtest
                GTest::gtest_main
        )
        target_link_options(
            cdna4_to_cdna3_dispatch_test
            PRIVATE -Wl,-rpath,${RJ_HSA_LIBRARY_DIR}
        )
        target_compile_definitions(
            cdna4_to_cdna3_dispatch_test
            PRIVATE HAS_HOST_AMDGPU=1 KERNEL_DIR="${KERNEL_OUTPUT_DIR}"
        )
        add_dependencies(cdna4_to_cdna3_dispatch_test device_kernels)
        rj_configure_target(cdna4_to_cdna3_dispatch_test TEST)

        add_test(
            NAME "Cdna4ToCdna3DispatchTest.TritonBufferAsyncMatmulTranslates"
            COMMAND
                cdna4_to_cdna3_dispatch_test
                --gtest_filter=Cdna4ToCdna3DispatchTest.TritonBufferAsyncMatmulTranslates
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        )
        set_tests_properties(
            "Cdna4ToCdna3DispatchTest.TritonBufferAsyncMatmulTranslates"
            PROPERTIES TIMEOUT 120
        )

        if(HAS_CDNA3_GPU)
            add_test(
                NAME "Cdna4ToCdna3DispatchTest.DynamicCopyLoopDispatchAndRun"
                COMMAND
                    cdna4_to_cdna3_dispatch_test
                    --gtest_filter=Cdna4ToCdna3DispatchTest.DynamicCopyLoopDispatchAndRun
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            )
            add_test(
                NAME
                    "Cdna4ToCdna3DispatchTest.TritonDynamicMatmulDispatchAndRun"
                COMMAND
                    cdna4_to_cdna3_dispatch_test
                    --gtest_filter=Cdna4ToCdna3DispatchTest.TritonDynamicMatmulDispatchAndRun
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            )
            add_test(
                NAME
                    "Cdna4ToCdna3DispatchTest.TritonBufferAsyncMatmulDispatchAndRun"
                COMMAND
                    cdna4_to_cdna3_dispatch_test
                    --gtest_filter=Cdna4ToCdna3DispatchTest.TritonBufferAsyncMatmulDispatchAndRun
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            )
            add_test(
                NAME
                    "Cdna4ToCdna3DispatchTest.TritonBufferAsyncMatmulConservativeLivenessDispatchAndRun"
                COMMAND
                    cdna4_to_cdna3_dispatch_test
                    --gtest_filter=Cdna4ToCdna3DispatchTest.TritonBufferAsyncMatmulConservativeLivenessDispatchAndRun
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            )
            add_test(
                NAME
                    "Cdna4ToCdna3DispatchTest.TritonFlashAttentionNoAsyncDispatchAndRun"
                COMMAND
                    cdna4_to_cdna3_dispatch_test
                    --gtest_filter=Cdna4ToCdna3DispatchTest.TritonFlashAttentionNoAsyncDispatchAndRun
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            )
            add_test(
                NAME
                    "Cdna4ToCdna3DispatchTest.TritonFlashAttentionBufferAsyncDispatchAndRun"
                COMMAND
                    cdna4_to_cdna3_dispatch_test
                    --gtest_filter=Cdna4ToCdna3DispatchTest.TritonFlashAttentionBufferAsyncDispatchAndRun
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            )
            set_tests_properties(
                "Cdna4ToCdna3DispatchTest.DynamicCopyLoopDispatchAndRun"
                "Cdna4ToCdna3DispatchTest.TritonDynamicMatmulDispatchAndRun"
                "Cdna4ToCdna3DispatchTest.TritonBufferAsyncMatmulDispatchAndRun"
                "Cdna4ToCdna3DispatchTest.TritonBufferAsyncMatmulConservativeLivenessDispatchAndRun"
                "Cdna4ToCdna3DispatchTest.TritonFlashAttentionNoAsyncDispatchAndRun"
                "Cdna4ToCdna3DispatchTest.TritonFlashAttentionBufferAsyncDispatchAndRun"
                PROPERTIES TIMEOUT 120
            )
            message(
                STATUS
                "CDNA4->CDNA3 dispatch tests enabled (CDNA3 GPU detected)"
            )
        else()
            message(
                STATUS
                "CDNA4->CDNA3 dispatch tests built but NOT registered with CTest "
                "(no CDNA3 GPU)"
            )
        endif()
    endif()
else()
    message(STATUS "HSA init test disabled (libhsa-runtime64 not found)")
endif()

# --- HIP tests (requires hipcc + CLI launcher) ---

find_program(
    HIPCC_EXECUTABLE
    hipcc
    PATHS /opt/rocm/bin
    ENV ROCM_PATH
    PATH_SUFFIXES bin
)
if(HIPCC_EXECUTABLE AND HSA_RUNTIME64)
    set(GTEST_INC ${googletest_SOURCE_DIR}/googletest/include)
    set(GTEST_LIB_DIR ${CMAKE_BINARY_DIR}/lib)

    # Helper: build a HIP test binary with hipcc + gtest.
    function(rj_add_hip_test TEST_NAME)
        cmake_parse_arguments(ARG "" "ARCH;SOURCE" "EXTRA_LIBS" ${ARGN})
        if(NOT ARG_ARCH)
            set(ARG_ARCH gfx950)
        endif()
        if(ARG_SOURCE)
            set(HIP_TEST_SRC ${CMAKE_CURRENT_SOURCE_DIR}/${ARG_SOURCE})
        else()
            set(HIP_TEST_SRC ${CMAKE_CURRENT_SOURCE_DIR}/${TEST_NAME}.cpp)
        endif()
        set(HIP_TEST_BIN ${CMAKE_CURRENT_BINARY_DIR}/${TEST_NAME})
        add_custom_command(
            OUTPUT ${HIP_TEST_BIN}
            COMMAND
                ${HIPCC_EXECUTABLE} --offload-arch=${ARG_ARCH} -std=c++20
                -isystem ${GTEST_INC} -I${CMAKE_CURRENT_SOURCE_DIR}
                -L${GTEST_LIB_DIR} -lgtest -lpthread ${ARG_EXTRA_LIBS}
                -Wl,-rpath,${GTEST_LIB_DIR} -o ${HIP_TEST_BIN} ${HIP_TEST_SRC}
            DEPENDS ${HIP_TEST_SRC} GTest::gtest GTest::gtest_main
            COMMENT "Building ${TEST_NAME} with hipcc (${ARG_ARCH})"
            VERBATIM
        )
        add_custom_target(${TEST_NAME}_target ALL DEPENDS ${HIP_TEST_BIN})
        add_dependencies(${TEST_NAME}_target rocjitsu_bin rocjitsu_kmd_shim)
    endfunction()

    # Helper: register a HIP test case run through the CLI launcher.
    function(rj_add_hip_test_case TEST_BIN TEST_CASE)
        cmake_parse_arguments(
            ARG
            ""
            "CONFIG;NAME;TIMEOUT"
            "ENVIRONMENT"
            ${ARGN}
        )
        if(NOT ARG_CONFIG)
            set(ARG_CONFIG ${RJ_CDNA4_KMD_CONFIG})
        endif()
        if(NOT ARG_NAME)
            set(ARG_NAME ${TEST_CASE})
        endif()
        add_test(
            NAME ${ARG_NAME}
            COMMAND
                ${RJ_CLI} --config ${ARG_CONFIG} --
                ${CMAKE_CURRENT_BINARY_DIR}/${TEST_BIN}
                --gtest_filter=${TEST_CASE}
        )
        if(ARG_ENVIRONMENT)
            list(JOIN ARG_ENVIRONMENT ";" _ENV_STR)
            set_tests_properties(
                ${ARG_NAME}
                PROPERTIES ENVIRONMENT "${_ENV_STR}"
            )
        endif()
        if(ARG_TIMEOUT)
            set_tests_properties(${ARG_NAME} PROPERTIES TIMEOUT ${ARG_TIMEOUT})
        endif()
    endfunction()

    rj_add_hip_test(hip_memcpy_test)
    rj_add_hip_test_case(hip_memcpy_test "HipMemcpyTest.RoundTripFloat")
    rj_add_hip_test_case(hip_memcpy_test "HipMemcpyTest.RoundTripInt")
    rj_add_hip_test_case(hip_memcpy_test "HipMemcpyTest.DeviceToDevice")
    rj_add_hip_test(hip_vector_add_test)
    rj_add_hip_test_case(hip_vector_add_test "HipVectorAddTest.CorrectResult")
    rj_add_hip_test(hip_ds_transpose_acc_test)

    # --- RCCL collective test ---
    find_library(
        RCCL_LIB
        rccl
        PATHS /opt/rocm/lib
        ENV ROCM_PATH
        PATH_SUFFIXES lib
    )
    if(RCCL_LIB)
        rj_add_hip_test(hip_rccl_test EXTRA_LIBS ${RCCL_LIB})
        message(STATUS "RCCL collective test enabled (found ${RCCL_LIB})")
    else()
        message(STATUS "RCCL collective test disabled (librccl not found)")
    endif()

    # --- Daemon RPC tests (gtest-driven) ---
    # A gtest fixture manages the daemon lifecycle: fork+exec the daemon in
    # SetUp, run the HIP test binary as a subprocess, tear down in TearDown.
    add_executable(daemon_test daemon_test.cpp)
    target_compile_definitions(
        daemon_test
        PRIVATE
            RJ_DAEMON_BIN="$<TARGET_FILE:rocjitsu_bin>"
            RJ_DAEMON_CONFIG="${CMAKE_SOURCE_DIR}/configs/amdgpu_cdna4_kmd.json"
            RJ_DAEMON_CONFIG_2GPU="${CMAKE_SOURCE_DIR}/configs/amdgpu_cdna4_kmd_2gpu.json"
            RJ_PRELOAD_LIB="$<TARGET_FILE:rocjitsu_kmd_shim>"
            RJ_HIP_VECTOR_ADD_BIN="${CMAKE_CURRENT_BINARY_DIR}/hip_vector_add_test"
            RJ_HIP_MEMCPY_BIN="${CMAKE_CURRENT_BINARY_DIR}/hip_memcpy_test"
            RJ_HIP_RCCL_BIN="${CMAKE_CURRENT_BINARY_DIR}/hip_rccl_test"
    )
    target_link_libraries(daemon_test PRIVATE GTest::gtest_main)
    add_dependencies(
        daemon_test
        rocjitsu_bin
        hip_vector_add_test_target
        hip_memcpy_test_target
    )
    if(RCCL_LIB)
        add_dependencies(daemon_test hip_rccl_test_target)
    endif()

    set(RJ_DAEMON_TESTS
        DaemonTest.HipVectorAdd
        DaemonTest.HipMemcpyRoundTripFloat
        DaemonTest.HipMemcpyRoundTripInt
        DaemonTest.HipMemcpyDeviceToDevice
        DaemonTest.TwoIndependentClients
    )
    foreach(TC IN LISTS RJ_DAEMON_TESTS)
        add_test(NAME ${TC} COMMAND daemon_test --gtest_filter=${TC})
        set_tests_properties(${TC} PROPERTIES TIMEOUT 120 LABELS "daemon")
    endforeach()

    if(RCCL_LIB)
        set(RJ_RCCL_TESTS
            RcclDaemonTest.AllReduce
            RcclDaemonTest.Broadcast
            RcclDaemonTest.AllGather
            RcclDaemonTest.ReduceScatter
            RcclDaemonTest.SendRecv
        )
        foreach(TC IN LISTS RJ_RCCL_TESTS)
            add_test(NAME ${TC} COMMAND daemon_test --gtest_filter=${TC})
            set_tests_properties(
                ${TC}
                PROPERTIES TIMEOUT 180 LABELS "daemon;rccl"
            )
        endforeach()
    endif()

    # --- rocBLAS GEMM test ---
    find_library(
        ROCBLAS_LIB
        rocblas
        PATHS /opt/rocm/lib
        ENV ROCM_PATH
        PATH_SUFFIXES lib
    )
    if(ROCBLAS_LIB)
        rj_add_hip_test(hip_gemm_test EXTRA_LIBS ${ROCBLAS_LIB})
        rj_add_hip_test(hip_gemm_test_cdna3 ARCH gfx942 SOURCE hip_gemm_test.cpp EXTRA_LIBS ${ROCBLAS_LIB})

        foreach(
            TC
            IN
            ITEMS
                Tiny_2x2x3
                Square_4x4
                Square_8x8
                Square_16x16
                Square_32x32
                Square_64x64
                Rect_16x32x8
                Rect_1x64x1
                Rect_64x1x64
                Rect_7x11x13
                AlphaBeta
                BetaZero
                Large_2048x2048
        )
            rj_add_hip_test_case(hip_gemm_test "RocblasGemmTest.${TC}")
        endforeach()
        set_tests_properties(
            RocblasGemmTest.Large_2048x2048
            PROPERTIES TIMEOUT 300
        )
        foreach(I RANGE 0 4)
            rj_add_hip_test_case(hip_gemm_test "RocblasGemmFuzz.Iter${I}")
        endforeach()

        foreach(TC IN ITEMS Tiny_2x2x3 Square_16x16 Square_32x32 Rect_7x11x13)
            rj_add_hip_test_case(hip_gemm_test_cdna3 "RocblasGemmTest.${TC}"
                CONFIG ${RJ_CDNA3_KMD_CONFIG} NAME "RocblasGemmCdna3.${TC}"
            )
        endforeach()
        foreach(I RANGE 0 2)
            rj_add_hip_test_case(hip_gemm_test_cdna3 "RocblasGemmFuzz.Iter${I}"
                CONFIG ${RJ_CDNA3_KMD_CONFIG} NAME "RocblasGemmCdna3.Fuzz_Iter${I}"
            )
        endforeach()
        message(STATUS "rocBLAS GEMM test enabled (found ${ROCBLAS_LIB})")
    else()
        message(STATUS "rocBLAS GEMM test disabled (librocblas not found)")
    endif()

    rj_add_hip_test(hip_bfe_2d_test)
    rj_add_hip_test_case(hip_bfe_2d_test "BfeRegressionTest.ThreadIdxY")

    # --- Logging plugin end-to-end test ---
    rj_add_hip_test(hip_logging_test SOURCE logging/hip_logging_test.hip)
    set(LOGGING_SINK_DIR ${CMAKE_CURRENT_BINARY_DIR}/logging_test_output)
    set(LOGGING_RUNTIME_DIR ${LOGGING_SINK_DIR}/runtime)
    file(MAKE_DIRECTORY ${LOGGING_SINK_DIR})
    file(MAKE_DIRECTORY ${LOGGING_RUNTIME_DIR})
    rj_add_hip_test_case(
        hip_logging_test "LoggingTest.dispatch_logged"
        TIMEOUT 30
        ENVIRONMENT
            "ROCJITSU_RUNTIME_DIR=${LOGGING_RUNTIME_DIR}"
            "RJ_LOG=1"
            "RJ_SINKS=file"
            "RJ_SINK_DIR=${LOGGING_SINK_DIR}"
    )

    # --- Profiled plugin group end-to-end test ---
    rj_add_hip_test(
        hip_profiled_plugin_test
        SOURCE logging/hip_profiled_plugin_test.hip
    )
    set(PROFILED_SINK_DIR ${CMAKE_CURRENT_BINARY_DIR}/profiled_test_output)
    set(PROFILED_RUNTIME_DIR ${PROFILED_SINK_DIR}/runtime)
    file(MAKE_DIRECTORY ${PROFILED_SINK_DIR})
    file(MAKE_DIRECTORY ${PROFILED_RUNTIME_DIR})
    rj_add_hip_test_case(
        hip_profiled_plugin_test "ProfiledPluginTest.hook_profile_output"
        TIMEOUT 30
        ENVIRONMENT
            "ROCJITSU_RUNTIME_DIR=${PROFILED_RUNTIME_DIR}"
            "RJ_USE_PROFILED_EXECUTION_PLUGIN_GROUP=1"
            "RJ_SINKS=file"
            "RJ_SINK_DIR=${PROFILED_SINK_DIR}"
    )

    add_subdirectory(race-detector)

    message(STATUS "HIP tests enabled (found ${HIPCC_EXECUTABLE})")
else()
    message(STATUS "HIP tests disabled (hipcc or libhsa-runtime64 not found)")
endif()

# --- Narrow conversion HIP CTS tests ---
# Uses hipcc (rocjitsu emulator does not yet support amdclang++ -x hip binaries).
if(HIPCC_EXECUTABLE AND HSA_RUNTIME64)
    set(CVT_NARROW_BIN ${CMAKE_CURRENT_BINARY_DIR}/hip_cvt_narrow_test)
    add_custom_command(
        OUTPUT ${CVT_NARROW_BIN}
        COMMAND
            ${HIPCC_EXECUTABLE} --offload-arch=gfx950 -std=c++20
            -I${CMAKE_SOURCE_DIR}/lib/util/include -o ${CVT_NARROW_BIN}
            ${CMAKE_CURRENT_SOURCE_DIR}/hip_cvt_narrow_test.cpp
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/hip_cvt_narrow_test.cpp
        COMMENT "Building hip_cvt_narrow_test with hipcc (gfx950)"
        VERBATIM
    )
    add_custom_target(hip_cvt_narrow_test_target ALL DEPENDS ${CVT_NARROW_BIN})
    add_dependencies(hip_cvt_narrow_test_target rocjitsu_bin rocjitsu_kmd_shim)

    add_test(
        NAME "CvtNarrow.AllTests"
        COMMAND ${RJ_CLI} --config ${RJ_CDNA4_KMD_CONFIG} -- ${CVT_NARROW_BIN}
    )
    set_tests_properties("CvtNarrow.AllTests" PROPERTIES TIMEOUT 300)
endif()
