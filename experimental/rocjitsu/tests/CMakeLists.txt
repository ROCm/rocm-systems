# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

include(rj_configure_target)

# Device kernels (compiled from HIP sources for simulator testing).
add_subdirectory(kernels)

# Common object libraries linked by all test/tool executables.
set(RJ_OBJECT_LIBS
    rocjitsu_code
    rocjitsu_isa
    rocjitsu_isa_cdna3
    rocjitsu_isa_cdna4
    rocjitsu_isa_risc_v
    simdojo
    rocjitsu_vm
    rocjitsu_vm_amdgpu
    rocjitsu_vm_risc_v
    rocjitsu_config
)

# --- Unit / integration test suite ---

enable_testing()

add_executable(rocjitsu_tests
    rvsim_test.cpp
    amdgpu_vm_test.cpp
    config_test.cpp
    matmul_test.cpp
    vector_add_test.cpp
    simdojo_sim_test.cpp
)
target_link_libraries(rocjitsu_tests PRIVATE
    ${RJ_OBJECT_LIBS} util flatbuffers Threads::Threads GTest::gtest_main)
rj_configure_target(rocjitsu_tests TEST)

include(GoogleTest)
gtest_discover_tests(rocjitsu_tests)

# --- Scaling test (standalone executable) ---

add_executable(scaling_test scaling_test.cpp)
target_link_libraries(scaling_test PRIVATE
    ${RJ_OBJECT_LIBS} util flatbuffers Threads::Threads)
target_compile_options(scaling_test PRIVATE -O2)
rj_configure_target(scaling_test TEST)

# --- rocminfo test (requires rocminfo binary + HSA runtime + LD_PRELOAD) ---
# Spawns rocminfo as a subprocess against the simulated GPU and validates output.

find_program(ROCMINFO_EXECUTABLE rocminfo PATHS /opt/rocm/bin ENV ROCM_PATH PATH_SUFFIXES bin)
if(ROCMINFO_EXECUTABLE AND HSA_RUNTIME64)
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
    message(STATUS "rocminfo test disabled (rocminfo or libhsa-runtime64 not found)")
endif()

