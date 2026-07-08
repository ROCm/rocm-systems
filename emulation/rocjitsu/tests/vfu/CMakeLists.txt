# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# vfio-user integration tests. Requires RJ_ENABLE_VFIO_USER=ON and a guest
# VM image (set RJ_GUEST_IMAGE env var or CTest property before running).

if(NOT RJ_ENABLE_VFIO_USER)
    return()
endif()

include(rj_configure_target)

add_executable(hip_vfu_vector_add_test hip_vfu_vector_add_test.cpp)

target_include_directories(
    hip_vfu_vector_add_test
    PRIVATE ${ROCJITSU_INCLUDE_DIR} ${ROCJITSU_SRC_DIR}
)

target_link_libraries(
    hip_vfu_vector_add_test
    PRIVATE
        rocjitsu_vfu
        rocjitsu_kmd
        rocjitsu_vm
        rocjitsu_vm_amdgpu
        util
        flatbuffers
        Threads::Threads
        GTest::gtest_main
)

rj_configure_target(hip_vfu_vector_add_test TOOL)

include(GoogleTest)
gtest_discover_tests(
    hip_vfu_vector_add_test
    PROPERTIES LABELS "vfu"
)

# Shell script smoke test: registered as a CTest test that skips when the
# guest image is not available.
add_test(
    NAME vfu_smoke_test_sh
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/vfu_smoke_test.sh
            --vfu-bin $<TARGET_FILE:rocjitsu_vfu_bin>
            --config ${PROJECT_SOURCE_DIR}/configs/gfx950_mi350p_kmd.json
)
set_tests_properties(vfu_smoke_test_sh PROPERTIES LABELS "vfu")
