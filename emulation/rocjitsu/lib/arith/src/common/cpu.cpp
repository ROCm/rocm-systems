// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Host CPU/OS feature detection and public CPU-level names.

#include "common/cpu_decode.h"
#include "rj_arith/rj_arith.h"

#include <stdint.h>

#if defined(RJ_ARITH_X86_64_QUERY)
#include <cpuid.h>
#include <immintrin.h>
#endif

#if defined(RJ_ARITH_X86_64_QUERY)
namespace {

// Returns XCR0. Requires CPUID to report both XSAVE and OSXSAVE support.
[[gnu::target("xsave")]] uint64_t read_xcr0(void) { return _xgetbv(0); }

} // namespace
#endif

// Query host CPU and OS vector-state support and return usable CPU flags.
uint64_t rj_arith_init_cpu_flags(void) {
#if defined(RJ_ARITH_X86_64_QUERY)
  rocjitsu::arith::CpuState s;
  uint32_t a, b, c, d;
  const uint32_t max_leaf = __get_cpuid_max(0, nullptr);
  if (max_leaf >= 1) {
    __cpuid_count(1, 0, a, b, c, d);
    s.leaf1_ecx = c;
  }

  if (max_leaf >= 7) {
    __cpuid_count(7, 0, a, b, c, d);
    s.leaf7_ebx = b;
    s.leaf7_ecx = c;
    if (a >= 1) {
      __cpuid_count(7, 1, a, b, c, d);
      s.leaf71_eax = a;
    }
  }

  if (__get_cpuid_max(0x80000000u, nullptr) >= 0x80000001u) {
    __cpuid_count(0x80000001u, 0, a, b, c, d);
    s.ext1_ecx = c;
  }

  // CPUID(1, 0).ECX: XSAVE (26) and OSXSAVE (27) make XGETBV available.
  constexpr uint32_t xgetbv_leaf1_ecx = 0x0c000000;
  if ((s.leaf1_ecx & xgetbv_leaf1_ecx) == xgetbv_leaf1_ecx) {
    s.xcr0 = read_xcr0();
  }

  return rocjitsu::arith::decode_cpu(s);
#else
  return RJ_ARITH_CPU_BASELINE;
#endif
}

// Return a stable CPU-level name, or "unknown" for an unrecognized level.
const char *rj_arith_cpu_name(uint64_t flags) {
  switch (flags & RJ_ARITH_CPU_MASK) {
  case RJ_ARITH_CPU_BASELINE:
    return "scalar";
  case RJ_ARITH_CPU_X86_64_V3:
    return "x86-64-v3";
  case RJ_ARITH_CPU_X86_64_V4:
    return "x86-64-v4";
  case RJ_ARITH_CPU_X86_64_ZEN4:
    return "x86-64-zen4";
  default:
    return "unknown";
  }
}
