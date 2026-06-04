# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

include(rj_configure_target)

add_executable(
    rocjitsu_afl_amdgpu_elf_reader_test
    tests/amdgpu_elf_reader_test.cpp
)
target_link_libraries(
    rocjitsu_afl_amdgpu_elf_reader_test
    PRIVATE
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
        util
        flatbuffers
        Threads::Threads
        GTest::gtest_main
)
rj_configure_target(rocjitsu_afl_amdgpu_elf_reader_test TEST)
add_test(
    NAME RocjitsuAflDbi.AmdGpuElfReader
    COMMAND rocjitsu_afl_amdgpu_elf_reader_test
)
