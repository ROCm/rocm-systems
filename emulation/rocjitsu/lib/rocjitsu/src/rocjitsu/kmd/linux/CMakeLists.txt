# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

rj_add_object_library(rocjitsu_kmd
    cwsr.cpp
    events.cpp
    gpu_alias_ingress.cpp
    guest_kfd.cpp
    legacy_gpu_vm.cpp
    libc_passthrough.cpp
    linux_kfd.cpp
    memory_backing.cpp
    memory_keeper_protocol.cpp
    remote_driver.cpp
    rj_daemon.cpp
    simulated_kfd.cpp
    sysfs.cpp
    transport.cpp
)
target_include_directories(rocjitsu_kmd PRIVATE ${GENERATED_DIR})
target_link_libraries(rocjitsu_kmd PUBLIC ${CMAKE_DL_LIBS} PRIVATE util)

file(
    RELATIVE_PATH
    _rj_keeper_relative
    "${CMAKE_INSTALL_FULL_LIBDIR}"
    "${CMAKE_INSTALL_FULL_LIBEXECDIR}/rocjitsu/rocjitsu-memory"
)
file(
    RELATIVE_PATH
    _rj_keeper_bin_relative
    "${CMAKE_INSTALL_FULL_BINDIR}"
    "${CMAKE_INSTALL_FULL_LIBEXECDIR}/rocjitsu/rocjitsu-memory"
)
file(
    RELATIVE_PATH
    _rj_keeper_test_relative
    "${CMAKE_INSTALL_FULL_DATADIR}/rocjitsu/tests/bin"
    "${CMAKE_INSTALL_FULL_LIBEXECDIR}/rocjitsu/rocjitsu-memory"
)
rj_add_build_path(
    rocjitsu_kmd RJ_MEMORY_KEEPER_BUILD_LIBDIR "$<TARGET_FILE_DIR:rocjitsu_shared>"
)
rj_add_build_path(
    rocjitsu_kmd RJ_MEMORY_KEEPER_BUILD_BINDIR "$<TARGET_FILE_DIR:rocjitsu_bin>"
)
rj_add_build_path(
    rocjitsu_kmd
    RJ_MEMORY_KEEPER_BUILD_TESTDIR
    "$<$<BOOL:${BUILD_TESTING}>:$<TARGET_FILE_DIR:rocjitsu_tests>>"
)
rj_add_build_path(
    rocjitsu_kmd RJ_MEMORY_KEEPER_BUILD_PATH "$<TARGET_FILE:rocjitsu_memory>"
)
rj_add_build_path(rocjitsu_kmd RJ_MEMORY_KEEPER_INSTALL_PATH "${_rj_keeper_relative}")
rj_add_build_path(rocjitsu_kmd RJ_MEMORY_KEEPER_BIN_INSTALL_PATH "${_rj_keeper_bin_relative}")
rj_add_build_path(rocjitsu_kmd RJ_MEMORY_KEEPER_TEST_INSTALL_PATH "${_rj_keeper_test_relative}")
