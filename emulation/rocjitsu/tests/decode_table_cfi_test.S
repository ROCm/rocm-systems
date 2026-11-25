// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/isa/arch/amdgpu/shared/decode_table_asm.inc"

#if RJ_HOST_HAS_X86_64_ELF_REL32
#define RJ_FACTORY_NAME(scope, name) rj_test_##scope##_##name
rj_table_begin rj_test_cfi_table
RJ_DECODE_CASE(0, valid, decode)
RJ_DECODE_CASE(1, mistyped, decode)
rj_table_end rj_test_cfi_table, 2
#endif
