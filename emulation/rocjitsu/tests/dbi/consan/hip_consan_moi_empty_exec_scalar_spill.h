// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file hip_consan_moi_empty_exec_scalar_spill.h
/// @brief Cross-architecture device workloads for empty-wave Inline spills.
///
/// These compact workloads distill a physical-gfx950 PyTorch torch.sort
/// failure. A full-pressure radix-sort access borrowed a transient scalar
/// window and reached the displaced LDS operation with EXEC=0. Per-lane
/// private save/restore must not run for that empty wave; the paired nonempty
/// workload proves that the guard does not suppress a real conflict.

#pragma once

#ifndef RJ_CONSAN_MOI_RECORD_NOP_CAVE
#error "define RJ_CONSAN_MOI_RECORD_NOP_CAVE before including this fixture"
#endif

#if defined(__gfx942__) || defined(__gfx950__)
#define RJ_CONSAN_MOI_SCALAR_SPILL_LDS_STORE "ds_write_b32"
#else
#define RJ_CONSAN_MOI_SCALAR_SPILL_LDS_STORE "ds_store_b32"
#endif

#define RJ_CONSAN_MOI_SCALAR_SPILL_LIVE_VALUES                                                     \
  ".irp r,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,"              \
  "33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,56,57,58,59,60,61,62,63,"                          \
  "64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,"           \
  "92,93,94,95,96,97,98,99\n\t"                                                                    \
  "s_mov_b32 s\\r, 0x13579bdf\n\t"                                                                 \
  ".endr\n\t"                                                                                      \
  "s_mov_b32 s6, 0x2468ace0\n\t"

#define RJ_CONSAN_MOI_SCALAR_SPILL_CHECKSUM                                                        \
  "s_mov_b32 s48, 0\n\t"                                                                           \
  ".irp r,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,"              \
  "33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,56,57,58,59,60,61,62,63,"                          \
  "64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,"           \
  "92,93,94,95,96,97,98,99\n\t"                                                                    \
  "s_xor_b32 s48, s48, s\\r\n\t"                                                                   \
  ".endr\n\t"                                                                                      \
  "v_mov_b32 %[checksum], s48\n\t"

#define RJ_CONSAN_MOI_SCALAR_SPILL_CLOBBERS                                                        \
  "s6", "s7", "s8", "s9", "s10", "s11", "s12", "s13", "s14", "s15", "s16", "s17", "s18", "s19",    \
      "s20", "s21", "s22", "s23", "s24", "s25", "s26", "s27", "s28", "s29", "s30", "s31", "s33",   \
      "s34", "s35", "s36", "s37", "s38", "s39", "s40", "s41", "s42", "s43", "s44", "s45", "s46",   \
      "s47", "s48", "s56", "s57", "s58", "s59", "s60", "s61", "s62", "s63", "s64", "s65", "s66",   \
      "s67", "s68", "s69", "s70", "s71", "s72", "s73", "s74", "s75", "s76", "s77", "s78", "s79",   \
      "s80", "s81", "s82", "s83", "s84", "s85", "s86", "s87", "s88", "s89", "s90", "s91", "s92",   \
      "s93", "s94", "s95", "s96", "s97", "s98", "s99"

// Eight access sites select Inline Shadow's scalable appended-body router.
// Keeping an ordinary NOP cave after every site also models the large PyTorch
// object without making the checked-in workload large or slow.
#define RJ_CONSAN_MOI_SCALAR_SPILL_EIGHT_LDS_STORES                                                \
  RJ_CONSAN_MOI_SCALAR_SPILL_LDS_STORE                                                             \
  " %[lds_addr], "                                                                                 \
  "%[value]\n\t" RJ_CONSAN_MOI_RECORD_NOP_CAVE RJ_CONSAN_MOI_SCALAR_SPILL_LDS_STORE                \
  " %[lds_addr], "                                                                                 \
  "%[value]\n\t" RJ_CONSAN_MOI_RECORD_NOP_CAVE RJ_CONSAN_MOI_SCALAR_SPILL_LDS_STORE                \
  " %[lds_addr], "                                                                                 \
  "%[value]\n\t" RJ_CONSAN_MOI_RECORD_NOP_CAVE RJ_CONSAN_MOI_SCALAR_SPILL_LDS_STORE                \
  " %[lds_addr], "                                                                                 \
  "%[value]\n\t" RJ_CONSAN_MOI_RECORD_NOP_CAVE RJ_CONSAN_MOI_SCALAR_SPILL_LDS_STORE                \
  " %[lds_addr], "                                                                                 \
  "%[value]\n\t" RJ_CONSAN_MOI_RECORD_NOP_CAVE RJ_CONSAN_MOI_SCALAR_SPILL_LDS_STORE                \
  " %[lds_addr], "                                                                                 \
  "%[value]\n\t" RJ_CONSAN_MOI_RECORD_NOP_CAVE RJ_CONSAN_MOI_SCALAR_SPILL_LDS_STORE                \
  " %[lds_addr], "                                                                                 \
  "%[value]\n\t" RJ_CONSAN_MOI_RECORD_NOP_CAVE RJ_CONSAN_MOI_SCALAR_SPILL_LDS_STORE                \
  " %[lds_addr], %[value]\n\t" RJ_CONSAN_MOI_RECORD_NOP_CAVE

namespace {

__global__ __launch_bounds__(64) void moi_zero_exec_scalar_spill_correct_kernel_for_instrumentation(
    uint32_t *out) {
  __shared__ uint32_t lds[1];
  const auto lds_addr = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&lds[0]) & 0xffffu);
  const uint32_t value = 0xc011ec7u;
  uint32_t checksum = 0;
  // Live ordinary SGPRs cover every 30-register candidate window while
  // s48:s55 remains available for compact PC/SCC routing state. The production
  // allocator therefore borrows and privately spills a scalar window.
  asm volatile(RJ_CONSAN_MOI_SCALAR_SPILL_LIVE_VALUES
               "s_mov_b64 vcc, exec\n\t"
               "s_mov_b64 exec, 0\n\t" RJ_CONSAN_MOI_SCALAR_SPILL_EIGHT_LDS_STORES
               "s_mov_b64 exec, vcc\n\t" RJ_CONSAN_MOI_SCALAR_SPILL_CHECKSUM
               : [checksum] "=&v"(checksum)
               : [lds_addr] "v"(lds_addr), [value] "v"(value)
               : RJ_CONSAN_MOI_SCALAR_SPILL_CLOBBERS, "vcc", "memory");
  if (threadIdx.x == 0)
    out[0] = checksum;
}

__global__ __launch_bounds__(
    128) void moi_nonempty_scalar_spill_incorrect_kernel_for_instrumentation(uint32_t *out) {
  __shared__ uint32_t lds[1];
  const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
  if ((tid & 63u) == 0u) {
    const auto lds_addr = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&lds[0]) & 0xffffu);
    const uint32_t value = 0x1cc00000u | tid;
    uint32_t checksum = 0;
    // One active lane in each of two waves performs the conflicting store.
    // This is the behavioral counterpart to the empty-EXEC case above: the
    // guard must not suppress instrumentation for a nonempty wave.
    asm volatile(RJ_CONSAN_MOI_SCALAR_SPILL_LIVE_VALUES RJ_CONSAN_MOI_SCALAR_SPILL_EIGHT_LDS_STORES
                     RJ_CONSAN_MOI_SCALAR_SPILL_CHECKSUM
                 : [checksum] "=&v"(checksum)
                 : [lds_addr] "v"(lds_addr), [value] "v"(value)
                 : RJ_CONSAN_MOI_SCALAR_SPILL_CLOBBERS, "memory");
    out[tid / 64u] = checksum;
  }
}

} // namespace

#undef RJ_CONSAN_MOI_SCALAR_SPILL_CLOBBERS
#undef RJ_CONSAN_MOI_SCALAR_SPILL_EIGHT_LDS_STORES
#undef RJ_CONSAN_MOI_SCALAR_SPILL_CHECKSUM
#undef RJ_CONSAN_MOI_SCALAR_SPILL_LIVE_VALUES
#undef RJ_CONSAN_MOI_SCALAR_SPILL_LDS_STORE
