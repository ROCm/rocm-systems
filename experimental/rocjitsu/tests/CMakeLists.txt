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
