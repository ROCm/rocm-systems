// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Mapping captured CPUID and XCR0 state to supported x86-64 CPU levels.
// Keeps this shared policy independent of instruction-specific register layouts.

#include "common/internal.h"
#include "rj_arith/rj_arith.h"

#include <stdint.h>

namespace rocjitsu::arith {

// Returns the highest CPU level supported by the supplied CPUID and XCR0 state.
uint64_t decode_cpu(const CpuState &s) {
  // CPUID(1, 0).ECX requirements for x86-64-v3:
  // SSE3 (0), SSSE3 (9), FMA (12), CMPXCHG16B (13), SSE4.1 (19), SSE4.2 (20),
  // MOVBE (22), POPCNT (23), XSAVE (26), OSXSAVE (27), AVX (28), F16C (29).
  constexpr uint32_t v3_leaf1_ecx = 0x3cd83201;

  // CPUID(7, 0).EBX: BMI1 (3), AVX2 (5), BMI2 (8).
  constexpr uint32_t v3_leaf7_ebx = 0x00000128;

  // CPUID(0x80000001, 0).ECX: LAHF/SAHF in 64-bit mode (0), LZCNT (5).
  constexpr uint32_t v3_ext1_ecx = 0x00000021;

  // XCR0: the OS enables saving/restoring XMM (1) and YMM upper halves (2).
  constexpr uint64_t v3_xcr0 = 0x06;

  if ((s.leaf1_ecx & v3_leaf1_ecx) != v3_leaf1_ecx ||
      (s.leaf7_ebx & v3_leaf7_ebx) != v3_leaf7_ebx || (s.ext1_ecx & v3_ext1_ecx) != v3_ext1_ecx ||
      (s.xcr0 & v3_xcr0) != v3_xcr0) {
    return RJ_ARITH_CPU_BASELINE;
  }

  // CPUID(7, 0).EBX additions for x86-64-v4:
  // AVX512F (16), AVX512DQ (17), AVX512CD (28), AVX512BW (30), AVX512VL (31).
  constexpr uint32_t v4_leaf7_ebx = 0xd0030000;

  // XCR0: XMM (1), YMM upper halves (2), opmask (5), ZMM upper halves (6),
  // and the full ZMM16-31 registers (7) must all be enabled by the OS.
  constexpr uint64_t v4_xcr0 = 0xe6;

  if ((s.leaf7_ebx & v4_leaf7_ebx) != v4_leaf7_ebx || (s.xcr0 & v4_xcr0) != v4_xcr0) {
    return RJ_ARITH_CPU_X86_64_V3;
  }

  // CPUID(7, 0).EBX addition for rj-arith's Zen4 tier: AVX512IFMA (21).
  constexpr uint32_t zen4_leaf7_ebx = 0x00200000;

  // CPUID(7, 0).ECX additions for rj-arith's Zen4 tier:
  // AVX512VBMI (1), AVX512VBMI2 (6), GFNI (8), VAES (9), VPCLMULQDQ (10),
  // AVX512VNNI (11), AVX512BITALG (12), AVX512VPOPCNTDQ (14).
  constexpr uint32_t zen4_leaf7_ecx = 0x00005f42;

  // CPUID(7, 1).EAX addition for rj-arith's Zen4 tier: AVX512BF16 (5).
  constexpr uint32_t zen4_leaf71_eax = 0x00000020;

  if ((s.leaf7_ebx & zen4_leaf7_ebx) != zen4_leaf7_ebx ||
      (s.leaf7_ecx & zen4_leaf7_ecx) != zen4_leaf7_ecx ||
      (s.leaf71_eax & zen4_leaf71_eax) != zen4_leaf71_eax) {
    return RJ_ARITH_CPU_X86_64_V4;
  }
  return RJ_ARITH_CPU_X86_64_ZEN4;
}

} // namespace rocjitsu::arith
