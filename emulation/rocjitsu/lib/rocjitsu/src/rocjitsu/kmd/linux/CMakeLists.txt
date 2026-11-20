# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

rj_add_object_library(rocjitsu_kmd
    cwsr.cpp
    gpu_alias_ingress.cpp
    guest_kfd.cpp
    libc_passthrough.cpp
    linux_kfd.cpp
    memory_keeper_protocol.cpp
    remote_driver.cpp
    sysfs.cpp
    transport.cpp
)
target_include_directories(rocjitsu_kmd PRIVATE ${GENERATED_DIR})
target_link_libraries(rocjitsu_kmd PUBLIC ${CMAKE_DL_LIBS} PRIVATE util)
rj_add_host_policy_sources(
    rocjitsu_kmd events.cpp legacy_gpu_vm.cpp memory_backing.cpp rj_daemon.cpp simulated_kfd.cpp
)

# Locate the helper relative to the object that owns the backing manager. The
# embedding component has its own private install path and can be relocated
# independently of the launcher.
function(rj_add_keeper_paths target library helper_dir with_frontends)
    set(_helper
        "${CMAKE_INSTALL_FULL_LIBEXECDIR}/${helper_dir}/rocjitsu-memory"
    )
    file(RELATIVE_PATH _from_lib "${CMAKE_INSTALL_FULL_LIBDIR}" "${_helper}")
    rj_add_build_path(${target} RJ_MEMORY_KEEPER_BUILD_LIBDIR "$<TARGET_FILE_DIR:${library}>")
    rj_add_build_path(${target} RJ_MEMORY_KEEPER_BUILD_PATH "$<TARGET_FILE:rocjitsu_memory>")
    rj_add_build_path(${target} RJ_MEMORY_KEEPER_INSTALL_PATH "${_from_lib}")
    if(with_frontends)
        file(
            RELATIVE_PATH
            _from_bin
            "${CMAKE_INSTALL_FULL_BINDIR}"
            "${_helper}"
        )
        file(
            RELATIVE_PATH
            _from_test
            "${CMAKE_INSTALL_FULL_DATADIR}/rocjitsu/tests/bin"
            "${_helper}"
        )
        rj_add_build_path(${target} RJ_MEMORY_KEEPER_BUILD_BINDIR "$<TARGET_FILE_DIR:rocjitsu_bin>")
        rj_add_build_path(${target} RJ_MEMORY_KEEPER_BUILD_TESTDIR "$<$<BOOL:${BUILD_TESTING}>:$<TARGET_FILE_DIR:rocjitsu_tests>>")
        rj_add_build_path(${target} RJ_MEMORY_KEEPER_BIN_INSTALL_PATH "${_from_bin}")
        rj_add_build_path(${target} RJ_MEMORY_KEEPER_TEST_INSTALL_PATH "${_from_test}")
    endif()
endfunction()

rj_add_keeper_paths(rocjitsu_kmd_host rocjitsu_shared rocjitsu ON)
if(ROCJITSU_BUILD_EMBED_LIBRARY)
    rj_add_keeper_paths(rocjitsu_kmd_embed rocjitsu_embed rocjitsu-embed OFF)
endif()
