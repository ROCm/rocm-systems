// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_TESTS_KERNELS_MULTIKERNEL_INDIRECT_BRANCH_COMMON_H_
#define ROCJITSU_TESTS_KERNELS_MULTIKERNEL_INDIRECT_BRANCH_COMMON_H_

#include <hip/hip_runtime.h>

/// Shared device routine used to force real `s_swappc_b64` calls in a HIP-built
/// code object.
///
/// The DBT indirect-branch recovery path needs to see multiple kernel entries
/// call into one reachable helper body.  Keeping this routine noinline prevents
/// the HIP optimizer from folding the helper into each kernel, and `used` keeps
/// the body present even though the only references are device-side calls.
__device__ __attribute__((noinline, used)) unsigned
multikernel_indirect_branch_shared_block(unsigned value, unsigned salt) {
  unsigned mixed = value ^ (salt * 0x45d9f3bu);
  mixed ^= mixed >> 16;
  mixed *= 0x119de1f3u;
  mixed ^= mixed >> 15;
  return mixed + salt;
}

/// Emit one recoverable `s_setpc_b64` static-skip island.
///
/// The literal addend is intentionally 0x100. Smaller deltas can be encoded as
/// inline integer operands, but the DBT recovery logic looks for the common
/// literal form emitted by real libraries before an indirect `s_setpc_b64`.
#define RJ_STATIC_SETPC_ISLAND(PC_LO, PC_HI)                                                       \
  asm volatile("s_getpc_b64 s[" #PC_LO ":" #PC_HI "]\n"                                            \
               "s_add_u32 s" #PC_LO ", s" #PC_LO ", 0x100\n"                                       \
               "s_addc_u32 s" #PC_HI ", s" #PC_HI ", 0\n"                                          \
               "s_setpc_b64 s[" #PC_LO ":" #PC_HI "]\n"                                            \
               ".rept 60\n"                                                                        \
               "s_nop 0\n"                                                                         \
               ".endr\n" ::                                                                        \
                   : "memory")

/// Emit one direct `s_call_b64` island whose target is unrelated to the shared
/// helper.  The target branches to the local join instead of returning; this is
/// translation coverage for the instruction and its direct target, not a helper
/// call ABI test.
#define RJ_STATIC_SCALL_ISLAND(RET_LO, RET_HI)                                                     \
  asm volatile("s_call_b64 s[" #RET_LO ":" #RET_HI "], 1f\n"                                       \
               "s_branch 2f\n"                                                                     \
               "1:\n"                                                                              \
               "s_nop 0\n"                                                                         \
               "s_branch 2f\n"                                                                     \
               "2:\n" ::                                                                           \
                   : "memory")

#endif // ROCJITSU_TESTS_KERNELS_MULTIKERNEL_INDIRECT_BRANCH_COMMON_H_
