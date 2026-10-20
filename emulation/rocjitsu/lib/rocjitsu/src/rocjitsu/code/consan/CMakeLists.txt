# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# A leaf dependency for all three ConSan stages. Do not add compiler, ISA,
# simulator or live HSA dependencies here.
add_library(rocjitsu_consan_contract STATIC contract.cpp capture.cpp)
target_include_directories(
    rocjitsu_consan_contract
    PUBLIC
        $<BUILD_INTERFACE:${ROCJITSU_SRC_DIR}>
        $<BUILD_INTERFACE:${ROCJITSU_INCLUDE_DIR}>
)
target_link_libraries(rocjitsu_consan_contract PUBLIC util)
rj_configure_target(rocjitsu_consan_contract WARNINGS HIDDEN)

if(RJ_BUILD_CONSAN_COMPILER OR RJ_BUILD_CONSAN_RUNTIME)
    add_subdirectory(image)
endif()
