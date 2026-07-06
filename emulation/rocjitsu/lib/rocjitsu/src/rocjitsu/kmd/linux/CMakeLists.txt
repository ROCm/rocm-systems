# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

rj_add_object_library(rocjitsu_kmd
    events.cpp
    remote_driver.cpp
    simulated_driver.cpp
    sysfs.cpp
    transport.cpp
)
target_include_directories(rocjitsu_kmd PRIVATE ${GENERATED_DIR})
