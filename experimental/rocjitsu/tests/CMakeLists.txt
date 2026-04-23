# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

include(rj_configure_target)

# Device kernels (compiled from HIP sources for simulator testing).
add_subdirectory(kernels)

# Common object libraries linked by all test/tool executables.
set(RJ_OBJECT_LIBS
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

add_executable(rocjitsu_tests
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
    mtype_flags_test.cpp
    kfd_ioctl_test.cpp
    waitcnt_counter_test.cpp
    instruction_execution_harness_test.cpp
    decode_execute_benchmark.cpp
    execution_plugin_test.cpp
)
target_link_libraries(rocjitsu_tests PRIVATE
    ${RJ_OBJECT_LIBS} rocjitsu_kmd util flatbuffers Threads::Threads GTest::gtest_main)
rj_configure_target(rocjitsu_tests TEST)

include(GoogleTest)
gtest_discover_tests(rocjitsu_tests)

# --- Scaling test (standalone executable) ---

add_executable(scaling_test scaling_test.cpp)
target_link_libraries(scaling_test PRIVATE
    ${RJ_OBJECT_LIBS} util flatbuffers Threads::Threads)
target_compile_options(scaling_test PRIVATE -O2)
rj_configure_target(scaling_test TEST)

# --- LD_PRELOAD test environment for KMD interposer tests ---

set(RJ_KMD_PRELOAD_ENV
    "LD_PRELOAD=$<TARGET_FILE:rocjitsu_kmd_shim>"
    "RJ_CONFIG=${CMAKE_SOURCE_DIR}/configs/amdgpu_cdna4_kmd.json"
    "RJ_SCHEMA=${CMAKE_SOURCE_DIR}/schemas/simulation_config.fbs"
)

# --- rocminfo test (requires rocminfo binary + HSA runtime + LD_PRELOAD) ---
# Spawns rocminfo as a subprocess against the simulated GPU and validates output.

find_program(ROCMINFO_EXECUTABLE NAMES rocminfo
    HINTS ${CMAKE_SOURCE_DIR}/.venv ENV ROCM_PATH /opt/rocm
    PATH_SUFFIXES bin
    DOC "Path to the rocminfo executable"
)
if(ROCMINFO_EXECUTABLE)
    add_executable(rocminfo_test rocminfo_test.cpp)
    target_compile_definitions(rocminfo_test PRIVATE
        ROCMINFO_PATH="${ROCMINFO_EXECUTABLE}")
    target_link_libraries(rocminfo_test PRIVATE GTest::gtest_main)

    add_test(NAME "RocminfoTest.ExitsSuccessfully"
        COMMAND rocminfo_test --gtest_filter=RocminfoTest.ExitsSuccessfully
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    add_test(NAME "RocminfoTest.InterposerActive"
        COMMAND rocminfo_test --gtest_filter=RocminfoTest.InterposerActive
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    add_test(NAME "RocminfoTest.HsaSystemAttributes"
        COMMAND rocminfo_test --gtest_filter=RocminfoTest.HsaSystemAttributes
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    add_test(NAME "RocminfoTest.DetectsGpuAgent"
        COMMAND rocminfo_test --gtest_filter=RocminfoTest.DetectsGpuAgent
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    add_test(NAME "RocminfoTest.ReportsGfx950"
        COMMAND rocminfo_test --gtest_filter=RocminfoTest.ReportsGfx950
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    add_test(NAME "RocminfoTest.ReportsWavefrontSize"
        COMMAND rocminfo_test --gtest_filter=RocminfoTest.ReportsWavefrontSize
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    set_tests_properties(
        "RocminfoTest.ExitsSuccessfully"
        "RocminfoTest.InterposerActive"
        "RocminfoTest.HsaSystemAttributes"
        "RocminfoTest.DetectsGpuAgent"
        "RocminfoTest.ReportsGfx950"
        "RocminfoTest.ReportsWavefrontSize"
        PROPERTIES
            ENVIRONMENT "${RJ_KMD_PRELOAD_ENV}"
            TIMEOUT 30)
    message(STATUS "rocminfo test enabled (found ${ROCMINFO_EXECUTABLE})")
else()
    message(STATUS "rocminfo test disabled (rocminfo not found)")
endif()


# --- HSA init test (requires ROCm HSA runtime + LD_PRELOAD at runtime) ---
# Standalone binary (not gtest) to avoid library initialization order issues
# with the interposer on machines that have a real GPU.

find_library(HSA_RUNTIME64 hsa-runtime64 PATHS /opt/rocm/lib ENV ROCM_PATH PATH_SUFFIXES lib)
if(HSA_RUNTIME64)
    add_executable(hsa_init_test hsa_init_test.cpp)
    target_include_directories(hsa_init_test PRIVATE /opt/rocm/include ${HSA_INCLUDE_DIR})
    target_link_libraries(hsa_init_test PRIVATE ${HSA_RUNTIME64} GTest::gtest)
    target_link_options(hsa_init_test PRIVATE -Wl,-rpath,/opt/rocm/lib)
    # Use explicit add_test rather than gtest_discover_tests: gtest_discover_tests
    # generates set_tests_properties with three separate bracket-quoted strings
    # after ENVIRONMENT, but CMake's PROPERTIES key-value parser only picks up
    # the first one (LD_PRELOAD). add_test + set_tests_properties emits a single
    # semicolon-delimited ENVIRONMENT value, matching how the HIP tests work.
    add_test(NAME "HsaTest.InitSucceeded"
        COMMAND hsa_init_test --gtest_filter=HsaTest.InitSucceeded
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    add_test(NAME "HsaTest.GpuAgentFound"
        COMMAND hsa_init_test --gtest_filter=HsaTest.GpuAgentFound
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    set_tests_properties("HsaTest.InitSucceeded" "HsaTest.GpuAgentFound" PROPERTIES
        ENVIRONMENT "${RJ_KMD_PRELOAD_ENV}"
        SKIP_REGULAR_EXPRESSION "\\[  SKIPPED \\]"
        TIMEOUT 30)
    message(STATUS "HSA init test enabled (found ${HSA_RUNTIME64})")
else()
    message(STATUS "HSA init test disabled (libhsa-runtime64 not found)")
endif()

# --- HIP vector_add test (requires hipcc + LD_PRELOAD at runtime) ---
# Standalone binary compiled with hipcc for device code generation.

find_program(HIPCC_EXECUTABLE hipcc PATHS /opt/rocm/bin ENV ROCM_PATH PATH_SUFFIXES bin)
if(HIPCC_EXECUTABLE AND HSA_RUNTIME64)
    set(GTEST_INC ${CMAKE_SOURCE_DIR}/third_party/googletest-src/googletest/include)
    set(GTEST_LIB_DIR ${CMAKE_BINARY_DIR}/lib)

    # Helper: build a HIP test with hipcc + gtest
    function(rj_add_hip_test TEST_NAME)
        set(HIP_TEST_SRC ${CMAKE_CURRENT_SOURCE_DIR}/${TEST_NAME}.cpp)
        set(HIP_TEST_BIN ${CMAKE_CURRENT_BINARY_DIR}/${TEST_NAME})
        add_custom_command(
            OUTPUT ${HIP_TEST_BIN}
            COMMAND ${HIPCC_EXECUTABLE} --offload-arch=gfx950 -std=c++20
                -isystem ${GTEST_INC}
                -L${GTEST_LIB_DIR} -lgtest -lpthread
                -Wl,-rpath,${GTEST_LIB_DIR}
                -o ${HIP_TEST_BIN} ${HIP_TEST_SRC}
            DEPENDS ${HIP_TEST_SRC} GTest::gtest GTest::gtest_main
            COMMENT "Building ${TEST_NAME} with hipcc + gtest"
            VERBATIM
        )
        add_custom_target(${TEST_NAME}_target ALL DEPENDS ${HIP_TEST_BIN})
        add_test(NAME ${TEST_NAME} COMMAND ${HIP_TEST_BIN})
        set_tests_properties(${TEST_NAME} PROPERTIES
            ENVIRONMENT "${RJ_KMD_PRELOAD_ENV}"
            TIMEOUT 120)
    endfunction()

    # Each HIP test registers as a separate ctest so gtest doesn't try to run
    # multiple tests in the same process. Running multiple HIP dispatches
    # sequentially in one process hangs due to comgr-related state issues.
    function(rj_add_hip_test_case TEST_BIN TEST_CASE)
        add_test(NAME ${TEST_CASE} COMMAND ${CMAKE_CURRENT_BINARY_DIR}/${TEST_BIN}
                 --gtest_filter=${TEST_CASE})
        set_tests_properties(${TEST_CASE} PROPERTIES
            ENVIRONMENT "${RJ_KMD_PRELOAD_ENV}"
            TIMEOUT 120)
    endfunction()

    rj_add_hip_test(hip_memcpy_test)
    rj_add_hip_test_case(hip_memcpy_test "HipMemcpyTest.RoundTripFloat")
    rj_add_hip_test_case(hip_memcpy_test "HipMemcpyTest.RoundTripInt")
    rj_add_hip_test_case(hip_memcpy_test "HipMemcpyTest.DeviceToDevice")
    rj_add_hip_test(hip_vector_add_test)
    message(STATUS "HIP tests enabled (found ${HIPCC_EXECUTABLE})")
else()
    message(STATUS "HIP tests disabled (hipcc or libhsa-runtime64 not found)")
endif()
