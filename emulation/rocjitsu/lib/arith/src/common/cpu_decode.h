// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Captured CPU state and the baseline decoder used by detection and tests.
#pragma once

#include <stdint.h>

namespace rocjitsu::arith {

// Snapshot of CPUID and XCR0 for deterministic OS-state/feature decoder tests.
struct CpuState {
  uint32_t leaf1_ecx = 0, leaf7_ebx = 0, leaf7_ecx = 0, leaf71_eax = 0, ext1_ecx = 0;
  uint64_t xcr0 = 0;
};

// Return the highest supported CPU level from captured CPUID/XCR0 state; performs no host
// queries.
uint64_t decode_cpu(const CpuState &s);

} // namespace rocjitsu::arith
