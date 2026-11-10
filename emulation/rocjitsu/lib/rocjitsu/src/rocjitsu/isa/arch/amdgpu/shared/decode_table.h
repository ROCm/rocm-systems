// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file decode_table.h
/// @brief Dispatch table storage and emission for generated AMDGPU decoders.
#pragma once

#include "util/host_target.h"

#if RJ_HOST_HAS_X86_64_ELF_REL32
#include <cstddef>
#include <cstdint>
#endif

// Give assembly and C++ the same private linker names without depending on C++ mangling.
#if defined(__ELF__) && (defined(__GNUC__) || defined(__clang__))
#define RJ_DECODE_SYMBOL(name) __asm__(#name) __attribute__((visibility("hidden")))
#else
#define RJ_DECODE_SYMBOL(name)
#endif
// Assembly takes these addresses outside C++. Keep the factories visible to
// LTO and bind their symbols to checked jump-table entries even when Clang
// uses noncanonical CFI jump tables for other functions.
#if RJ_HOST_HAS_X86_64_ELF_REL32 && defined(__clang__)
#define RJ_DECODE_ASM_TARGET __attribute__((used, cfi_canonical_jump_table))
#elif RJ_HOST_HAS_X86_64_ELF_REL32
#define RJ_DECODE_ASM_TARGET __attribute__((used))
#else
#define RJ_DECODE_ASM_TARGET
#endif
#define RJ_DECODE_FACTORY(arch, scope, name)                                                       \
  DecodeFactory name RJ_DECODE_SYMBOL(rj_##arch##_##scope##_##name) RJ_DECODE_ASM_TARGET

#if RJ_HOST_HAS_X86_64_ELF_REL32
namespace rocjitsu::amdgpu {

/// @brief Link-time displacements from the start of an AMDGPU decoder table.
template <typename Function, std::size_t Size> class DecodeTable {
public:
  /// @brief Resolve a factory by adding its signed displacement to the table address.
  Function operator[](std::size_t index) const {
    return reinterpret_cast<Function>(reinterpret_cast<std::uintptr_t>(this) + offsets_[index]);
  }

private:
  // The assembly definition emits exactly Size consecutive signed 32-bit words.
  std::int32_t offsets_[Size];
};

} // namespace rocjitsu::amdgpu

#else
/// @brief Return the default factory result using the generated function's opcode and emit_error.
#define RJ_DECODE_DEFAULT(scope, name)                                                             \
  default:                                                                                         \
    return scope::name(opcode, emit_error);
/// @brief Return the selected factory result using the same opcode and emit_error.
#define RJ_DECODE_CASE(slot, scope, name)                                                          \
  case slot:                                                                                       \
    return scope::name(opcode, emit_error);
#define RJ_DECODE_ALIAS(slot, scope, name) case slot:
#define RJ_DECODE_GAP(first, count, scope, name)
#endif
