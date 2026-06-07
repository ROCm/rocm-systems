# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

rj_add_object_library(rocjitsu_kmd
    events.cpp
    guest_kfd.cpp
    libc_passthrough.cpp
    linux_kfd.cpp
    remote_driver.cpp
    simulated_kfd.cpp
    sysfs.cpp
    transport.cpp
)
target_include_directories(rocjitsu_kmd PRIVATE ${GENERATED_DIR})
target_link_libraries(rocjitsu_kmd PUBLIC ${CMAKE_DL_LIBS})

add_library(rocjitsu_waitcheck_shim SHARED waitcheck_preload.cpp)
target_include_directories(
    rocjitsu_waitcheck_shim
    PRIVATE ${ROCJITSU_INCLUDE_DIR} ${ROCJITSU_SRC_DIR} ${HSA_INCLUDE_DIR}
)
target_link_libraries(
    rocjitsu_waitcheck_shim
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
        rocjitsu_isa_gfx1250
        rocjitsu_isa_risc_v
        rocjitsu_vm_amdgpu
        simdojo
        util
        dl
)
target_compile_definitions(rocjitsu_waitcheck_shim PRIVATE _GNU_SOURCE)
target_compile_options(
    rocjitsu_waitcheck_shim
    PRIVATE -Wall -Wextra -Wpedantic -Werror
)
set_target_properties(
    rocjitsu_waitcheck_shim
    PROPERTIES OUTPUT_NAME "rocjitsu_waitcheck"
)
