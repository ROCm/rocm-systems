// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file decode_table_asm.inc
/// @brief Emit relative AMDGPU decoder tables and preserve ELF security properties.

#include "util/host_target.h"

#if RJ_HOST_ARCH_X86 && defined(__CET__)
#include <cet.h>
#endif
// Data-only objects must preserve the rest of the binary's security properties.
.section .note.GNU-stack,"",%progbits

#if RJ_HOST_HAS_X86_64_ELF_REL32
// A standalone assembly object owns each table and references ordinary hidden
// factory symbols. The static archive index and LTO linker see those references.
.macro rj_table_begin name
    .section .rodata.\name,"a",@progbits
    .balign 4
    .hidden \name
    .globl \name
    .type \name,@object
\name:
0:
.endm

// Numbered subsections restore slot order after grouping switch aliases.
// The local label 0b is the start of the table.
.macro rj_decode_run first, count, factory
    // Bind the factory locally even if a DSO exports the same name.
    .hidden \factory
    .subsection (1 + \first)
    .rept \count
    .long \factory - 0b
    .endr
.endm

.macro rj_table_end name, slots
    .subsection (1 + \slots)
    .size \name, . - 0b
.endm

#define RJ_DECODE_CASE(slot, scope, name) rj_decode_run slot, 1, RJ_FACTORY_NAME(scope, name);
#define RJ_DECODE_ALIAS(slot, scope, name) RJ_DECODE_CASE(slot, scope, name)
#define RJ_DECODE_DEFAULT(scope, name)
#define RJ_DECODE_GAP(first, count, scope, name) \
    rj_decode_run first, count, RJ_FACTORY_NAME(scope, name);
#endif
