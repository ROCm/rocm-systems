# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.24)

# The prototype can also be built against an existing ROCm installation.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(rocprofv3-att-gdb LANGUAGES CXX)
    include(GNUInstallDirs)
    include(CTest)
endif()

find_package(Threads REQUIRED)
add_library(rocprofv3-att-gdb-helper SHARED helper.cpp)
target_compile_features(rocprofv3-att-gdb-helper PRIVATE cxx_std_17)
target_compile_options(rocprofv3-att-gdb-helper PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(rocprofv3-att-gdb-helper PRIVATE Threads::Threads ${CMAKE_DL_LIBS})
set_target_properties(
    rocprofv3-att-gdb-helper
    PROPERTIES CXX_VISIBILITY_PRESET hidden
               VISIBILITY_INLINES_HIDDEN ON
               LIBRARY_OUTPUT_DIRECTORY
               "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/rocprofiler-sdk")
install(TARGETS rocprofv3-att-gdb-helper
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/rocprofiler-sdk COMPONENT tools)

configure_file(
    launcher.py
    ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}/rocprofv3-att-gdb
    COPYONLY
    FILE_PERMISSIONS
    OWNER_READ
    OWNER_WRITE
    OWNER_EXECUTE
    GROUP_READ
    GROUP_EXECUTE
    WORLD_READ
    WORLD_EXECUTE)
configure_file(
    extension.py
    ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_DATAROOTDIR}/rocprofiler-sdk/rocprofv3_att_gdb.py
    COPYONLY)
configure_file(
    rocprofv3_att_gdb_common.py
    ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_DATAROOTDIR}/rocprofiler-sdk/rocprofv3_att_gdb_common.py
    COPYONLY)
install(
    PROGRAMS ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}/rocprofv3-att-gdb
    DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT tools)
install(
    FILES extension.py
    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/rocprofiler-sdk
    RENAME rocprofv3_att_gdb.py
    COMPONENT tools)
install(
    FILES rocprofv3_att_gdb_common.py
    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/rocprofiler-sdk
    COMPONENT tools)

if(BUILD_TESTING OR ROCPROFILER_BUILD_TESTS)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    set(ROCPROFV3_ATT_GDB_ROCM_ROOT
        "/opt/rocm"
        CACHE PATH "ROCm installation for ATT debugger tests")
    add_test(
        NAME rocprofv3-att-gdb-prototype-tests
        COMMAND
            ${Python3_EXECUTABLE} -m pytest -q
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_helper.py
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_launcher.py --helper
            $<TARGET_FILE:rocprofv3-att-gdb-helper> --launcher
            ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}/rocprofv3-att-gdb)
    set_tests_properties(rocprofv3-att-gdb-prototype-tests PROPERTIES LABELS unittests
                                                                      TIMEOUT 90)
    add_test(
        NAME rocprofv3-att-gdb-debugger-tests
        COMMAND
            ${Python3_EXECUTABLE} -m pytest -q
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_debugger.py --helper
            $<TARGET_FILE:rocprofv3-att-gdb-helper> --launcher
            ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}/rocprofv3-att-gdb --rocm
            ${ROCPROFV3_ATT_GDB_ROCM_ROOT})
    set_tests_properties(rocprofv3-att-gdb-debugger-tests
                         PROPERTIES LABELS integration-tests TIMEOUT 90)

    option(ROCPROFV3_ATT_GDB_GPU_TESTS
           "Enable ROCgdb/HIP integration tests for the ATT prototype" OFF)
    set(ROCPROFV3_ATT_GDB_GPU_ARCH
        "native"
        CACHE STRING "HIP GPU target for ATT debugger tests")
    if(ROCPROFV3_ATT_GDB_GPU_TESTS)
        add_test(
            NAME rocprofv3-att-gdb-gpu-tests
            COMMAND
                ${Python3_EXECUTABLE} -m pytest -q
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_gpu.py --helper
                $<TARGET_FILE:rocprofv3-att-gdb-helper> --launcher
                ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}/rocprofv3-att-gdb --gpu
                --gpu-arch ${ROCPROFV3_ATT_GDB_GPU_ARCH} --rocm
                ${ROCPROFV3_ATT_GDB_ROCM_ROOT})
        set_tests_properties(
            rocprofv3-att-gdb-gpu-tests
            PROPERTIES LABELS "integration-tests;thread-trace" TIMEOUT 180 RUN_SERIAL ON)
    endif()
endif()
