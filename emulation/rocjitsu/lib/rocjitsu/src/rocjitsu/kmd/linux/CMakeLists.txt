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
