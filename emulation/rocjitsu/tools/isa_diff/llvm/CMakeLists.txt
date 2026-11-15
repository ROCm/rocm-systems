# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# LLVM plugin for rj_isa_diff: drives an external LLVM toolchain (llc, llvm-mc, ld.lld)
# as the assembler/compiler adapter. It does not link LLVM.
add_library(rj_isa_diff_llvm STATIC llvm_frontend.cpp)
target_include_directories(rj_isa_diff_llvm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(rj_isa_diff_llvm PUBLIC rj_isa_diff_core)
rj_configure_target(rj_isa_diff_llvm PRIVATE_INCLUDES WARNINGS)
