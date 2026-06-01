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
    rocjitsu_isa_risc_v
    simdojo
    rocjitsu_vm
    rocjitsu_vm_amdgpu
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
    execution_plugin_test.cpp
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
if(HAS_DEVICE_KERNELS)
    target_sources(
        rocjitsu_tests
        PRIVATE dbt/device_kernels_translate_tests.cpp
    )
endif()
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

# --- LD_PRELOAD test environment for KMD interposer tests ---

set(RJ_KMD_PRELOAD_ENV
    "LD_PRELOAD=$<TARGET_FILE:rocjitsu_kmd_shim>"
    "RJ_CONFIG=${CMAKE_SOURCE_DIR}/configs/amdgpu_cdna4_kmd.json"
    "HSA_ENABLE_SDMA=1"
)

# --- rocminfo test (requires rocminfo binary + HSA runtime + LD_PRELOAD) ---
# Spawns rocminfo as a subprocess against the simulated GPU and validates output.

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
        PRIVATE ROCMINFO_PATH="${ROCMINFO_EXECUTABLE}"
    )
    target_link_libraries(rocminfo_test PRIVATE GTest::gtest_main)

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

# --- HSA init test (requires ROCm HSA runtime + LD_PRELOAD at runtime) ---
# Standalone binary (not gtest) to avoid library initialization order issues
# with the interposer on machines that have a real GPU.

find_library(
    HSA_RUNTIME64
    hsa-runtime64
    HINTS
    ENV ROCM_PATH
    /opt/rocm
    PATH_SUFFIXES lib
)
if(HSA_RUNTIME64)
    # Keep include paths and rpaths tied to the HSA runtime that CMake actually
    # found. Test machines often install ROCm under versioned roots such as
    # /opt/rocm-7.2.0 without a /opt/rocm symlink.
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
    # Use explicit add_test rather than gtest_discover_tests: gtest_discover_tests
    # generates set_tests_properties with three separate bracket-quoted strings
    # after ENVIRONMENT, but CMake's PROPERTIES key-value parser only picks up
    # the first one (LD_PRELOAD). add_test + set_tests_properties emits a single
    # semicolon-delimited ENVIRONMENT value, matching how the HIP tests work.
    add_test(
        NAME "HsaTest.InitSucceeded"
        COMMAND hsa_init_test --gtest_filter=HsaTest.InitSucceeded
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
    add_test(
        NAME "HsaTest.GpuAgentFound"
        COMMAND hsa_init_test --gtest_filter=HsaTest.GpuAgentFound
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
    set_tests_properties(
        "HsaTest.InitSucceeded"
        "HsaTest.GpuAgentFound"
        PROPERTIES
            ENVIRONMENT "${RJ_KMD_PRELOAD_ENV}"
            SKIP_REGULAR_EXPRESSION "\\[  SKIPPED \\]"
    )
    message(STATUS "HSA init test enabled (found ${HSA_RUNTIME64})")

    # --- HSA translate test: CDNA4→host on-device verification ---
    # Requires compiled device kernels and a host GPU with a wired DBT target.
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
        set(HAS_RDNA4_GPU FALSE)
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
            if(_agent_rc EQUAL 0 AND _agents MATCHES "gfx120[01]")
                set(HAS_RDNA4_GPU TRUE)
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

        if(HAS_DBT_VECTOR_ADD_GPU)
            add_test(
                NAME "HsaTranslateTest.TranslateAndDispatchVectorAdd"
                COMMAND hsa_translate_test --gtest_filter=*VectorAdd*
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            )
            message(
                STATUS
                "HSA translate test enabled (supported DBT GPU detected)"
            )
        else()
            message(
                STATUS
                "HSA translate vector_add test built but NOT registered with CTest "
                "(no supported DBT GPU)"
            )
        endif()

        if(HAS_RDNA4_GPU)
            add_test(
                NAME "HsaTranslateTest.TranslateAndDispatchMfma16x16"
                COMMAND hsa_translate_test --gtest_filter=*Mfma16x16*
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            )
            set_tests_properties(
                "HsaTranslateTest.TranslateAndDispatchMfma16x16"
                PROPERTIES TIMEOUT 60
            )
            message(
                STATUS
                "HSA translate MFMA test enabled (RDNA4 GPU detected)"
            )
        else()
            message(
                STATUS
                "HSA translate MFMA test built but NOT registered with CTest "
                "(no RDNA4 GPU)"
            )
        endif()
    endif()
else()
    message(STATUS "HSA init test disabled (libhsa-runtime64 not found)")
endif()

# --- HIP vector_add test (requires hipcc + LD_PRELOAD at runtime) ---
# Standalone binary compiled with hipcc for device code generation.

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

    # Helper: build a HIP test with hipcc + gtest
    function(rj_add_hip_test TEST_NAME)
        set(HIP_TEST_SRC ${CMAKE_CURRENT_SOURCE_DIR}/${TEST_NAME}.cpp)
        set(HIP_TEST_BIN ${CMAKE_CURRENT_BINARY_DIR}/${TEST_NAME})
        add_custom_command(
            OUTPUT ${HIP_TEST_BIN}
            COMMAND
                ${HIPCC_EXECUTABLE} --offload-arch=gfx950 -std=c++20 -isystem
                ${GTEST_INC} -L${GTEST_LIB_DIR} -lgtest -lpthread
                -Wl,-rpath,${GTEST_LIB_DIR} -o ${HIP_TEST_BIN} ${HIP_TEST_SRC}
            DEPENDS ${HIP_TEST_SRC} GTest::gtest GTest::gtest_main
            COMMENT "Building ${TEST_NAME} with hipcc + gtest"
            VERBATIM
        )
        add_custom_target(${TEST_NAME}_target ALL DEPENDS ${HIP_TEST_BIN})
        add_test(NAME ${TEST_NAME} COMMAND ${HIP_TEST_BIN})
        set_tests_properties(
            ${TEST_NAME}
            PROPERTIES ENVIRONMENT "${RJ_KMD_PRELOAD_ENV}"
        )
    endfunction()

    # Each HIP test registers as a separate ctest so gtest doesn't try to run
    # multiple tests in the same process. Running multiple HIP dispatches
    # sequentially in one process hangs due to comgr-related state issues.
    function(rj_add_hip_test_case TEST_BIN TEST_CASE)
        add_test(
            NAME ${TEST_CASE}
            COMMAND
                ${CMAKE_CURRENT_BINARY_DIR}/${TEST_BIN}
                --gtest_filter=${TEST_CASE}
        )
        set_tests_properties(
            ${TEST_CASE}
            PROPERTIES ENVIRONMENT "${RJ_KMD_PRELOAD_ENV}"
        )
    endfunction()

    rj_add_hip_test(hip_memcpy_test)
    rj_add_hip_test_case(hip_memcpy_test "HipMemcpyTest.RoundTripFloat")
    rj_add_hip_test_case(hip_memcpy_test "HipMemcpyTest.RoundTripInt")
    rj_add_hip_test_case(hip_memcpy_test "HipMemcpyTest.DeviceToDevice")
    rj_add_hip_test_case(hip_memcpy_test "HipMemcpyTest.LargeCopy_16MB")
    rj_add_hip_test_case(hip_memcpy_test "HipMemcpyTest.LargeCopy_256MB")
    rj_add_hip_test_case(hip_memcpy_test "HipMemcpyTest.ReuseAfterFree")
    rj_add_hip_test(hip_vector_add_test)

    # --- rocBLAS GEMM test ---
    find_library(
        ROCBLAS_LIB
        rocblas
        PATHS /opt/rocm/lib
        ENV ROCM_PATH
        PATH_SUFFIXES lib
    )
    if(ROCBLAS_LIB)
        set(GEMM_SRC ${CMAKE_CURRENT_SOURCE_DIR}/hip_gemm_test.cpp)
        set(GEMM_BIN ${CMAKE_CURRENT_BINARY_DIR}/hip_gemm_test)
        add_custom_command(
            OUTPUT ${GEMM_BIN}
            COMMAND
                ${HIPCC_EXECUTABLE} --offload-arch=gfx950 -std=c++20 -isystem
                ${GTEST_INC} -L${GTEST_LIB_DIR} -lgtest -lpthread -lrocblas
                -Wl,-rpath,${GTEST_LIB_DIR} -o ${GEMM_BIN} ${GEMM_SRC}
            DEPENDS ${GEMM_SRC} GTest::gtest
            COMMENT "Building hip_gemm_test with hipcc + rocblas"
            VERBATIM
        )
        add_custom_target(hip_gemm_test_target ALL DEPENDS ${GEMM_BIN})

        # CDNA4 test cases (default KMD config).
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
        )
            rj_add_hip_test_case(hip_gemm_test "RocblasGemmTest.${TC}")
        endforeach()
        # Large GEMM stress test — validates SDMA + LDS isolation for big K.
        add_test(
            NAME "RocblasGemmTest.Large_2048x2048"
            COMMAND ${GEMM_BIN} --gtest_filter=RocblasGemmTest.Large_2048x2048
        )
        set_tests_properties(
            "RocblasGemmTest.Large_2048x2048"
            PROPERTIES ENVIRONMENT "${RJ_KMD_PRELOAD_ENV}"
        )
        foreach(I RANGE 0 4)
            rj_add_hip_test_case(hip_gemm_test "RocblasGemmFuzz.Iter${I}")
        endforeach()

        # CDNA3 test cases (same binary, CDNA3 KMD config).
        set(RJ_CDNA3_ENV
            "LD_PRELOAD=$<TARGET_FILE:rocjitsu_kmd_shim>"
            "RJ_CONFIG=${CMAKE_SOURCE_DIR}/configs/amdgpu_cdna3_kmd.json"
            "HSA_ENABLE_SDMA=1"
        )
        foreach(TC IN ITEMS Tiny_2x2x3 Square_16x16 Square_32x32 Rect_7x11x13)
            add_test(
                NAME "RocblasGemmCdna3.${TC}"
                COMMAND ${GEMM_BIN} --gtest_filter=RocblasGemmTest.${TC}
            )
            set_tests_properties(
                "RocblasGemmCdna3.${TC}"
                PROPERTIES ENVIRONMENT "${RJ_CDNA3_ENV}"
            )
        endforeach()
        foreach(I RANGE 0 2)
            add_test(
                NAME "RocblasGemmCdna3.Fuzz_Iter${I}"
                COMMAND ${GEMM_BIN} --gtest_filter=RocblasGemmFuzz.Iter${I}
            )
            set_tests_properties(
                "RocblasGemmCdna3.Fuzz_Iter${I}"
                PROPERTIES ENVIRONMENT "${RJ_CDNA3_ENV}"
            )
        endforeach()
        message(STATUS "rocBLAS GEMM test enabled (found ${ROCBLAS_LIB})")
    else()
        message(STATUS "rocBLAS GEMM test disabled (librocblas not found)")
    endif()

    rj_add_hip_test(hip_bfe_2d_test)
    rj_add_hip_test_case(hip_bfe_2d_test "BfeRegressionTest.ThreadIdxY")

    message(STATUS "HIP tests enabled (found ${HIPCC_EXECUTABLE})")
else()
    message(STATUS "HIP tests disabled (hipcc or libhsa-runtime64 not found)")
endif()
