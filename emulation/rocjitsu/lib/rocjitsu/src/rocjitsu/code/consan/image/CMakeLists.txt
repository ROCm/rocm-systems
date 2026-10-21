# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Shared ELF/metadata checks are not the compiler's CFG analysis or lowering.
# They depend on declarative descriptor types, never an ISA registry or VM.
add_library(rocjitsu_consan_image STATIC metadata.cpp artifact.cpp)
target_link_libraries(
    rocjitsu_consan_image
    PUBLIC rocjitsu_consan_contract rocjitsu_kernel_descriptor_scan
)
rj_configure_target(rocjitsu_consan_image PRIVATE_INCLUDES WARNINGS HIDDEN)
