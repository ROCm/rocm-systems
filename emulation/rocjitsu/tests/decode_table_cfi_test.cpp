// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/isa/arch/amdgpu/shared/decode_table.h"

#include <csignal>
#include <cstdlib>

#if RJ_HOST_HAS_X86_64_ELF_REL32
namespace valid {
using DecodeFactory = int(int);
RJ_DECODE_FACTORY(test, valid, decode);
int decode(int value) { return value + 1; }
} // namespace valid

namespace mistyped {
using DecodeFactory = int(long);
RJ_DECODE_FACTORY(test, mistyped, decode);
int decode(long value) { return static_cast<int>(value) + 1; }
} // namespace mistyped

extern const rocjitsu::amdgpu::DecodeTable<valid::DecodeFactory *, 2>
    table RJ_DECODE_SYMBOL(rj_test_cfi_table);

// Clang traps an indirect call with the wrong signature on x86.
int main() {
  if (table[0](1) != 2) {
    return EXIT_FAILURE;
  }
  auto expected_trap = [](int) { std::_Exit(EXIT_SUCCESS); };
  std::signal(SIGILL, expected_trap);
  std::signal(SIGTRAP, expected_trap);
  table[1](1);
  return EXIT_FAILURE;
}
#else
int main() { return 77; }
#endif
