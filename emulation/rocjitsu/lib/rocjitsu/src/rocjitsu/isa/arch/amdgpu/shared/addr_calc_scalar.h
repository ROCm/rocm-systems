// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_SHARED_ADDR_CALC_SCALAR_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_SHARED_ADDR_CALC_SCALAR_H_

/// @file Shared address calculation for SMEM and DS instructions.
///
/// SMEM is scalar memory (constant cache / kernarg loads).
/// DS is local data share (LDS) operations.
///
/// These functions are templated on the machine instruction type so they work
/// with any ISA family whose encoding struct exposes the required field names.

#include "rocjitsu/isa/arch/amdgpu/shared/buffer_address.h"
#include "rocjitsu/isa/arch/amdgpu/shared/scalar_operand_read.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/mem_state.h"
#include "rocjitsu/vm/amdgpu/register_access.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "util/host_target.h"
#include "util/log.h"

#include <array>
#include <cstdint>
#include <optional>
#include <span>

#if RJ_HOST_HAS_AVX2
#include <immintrin.h>
#endif

namespace rocjitsu {
namespace amdgpu {
namespace addr_calc {

/// Mark each in-range DWORD separately so partial buffer loads zero only the
/// out-of-range destinations. Byte and halfword loads use one destination DWORD.
inline void scalar_buffer_load_mask(ScalarMemState *state, int64_t offset, uint64_t bound) {
  if (!state)
    return;
  state->load_dword_mask = 0;
  for (uint32_t i = 0; i < state->num_dwords; ++i) {
    const int64_t part = offset + i * 4;
    if (part >= 0 && static_cast<uint64_t>(part) < bound)
      state->load_dword_mask |= 1u << i;
  }
}

/// Scalar buffer resource layout shared by GFX9 and RDNA: a 48-bit base,
/// 14-bit stride and 32-bit record count. Stride affects bounds, not the address.
inline std::optional<uint64_t> scalar_buffer_address(Wavefront &wf, uint32_t selector,
                                                     uint64_t base, int64_t offset,
                                                     ScalarMemState *state, uint64_t align_mask = 3,
                                                     uint64_t size_align_mask = 0) {
  auto records = try_read_scalar_selector(wf, selector + 2);
  if (!records)
    return std::nullopt;
  const uint64_t stride = (base >> 48) & 0x3fff;
  scalar_buffer_load_mask(state, offset, (*records * (stride ? stride : 1)) & ~size_align_mask);
  return buffer_virtual_address(base + offset) & ~align_mask;
}

/// @brief Scalar-buffer load opcode families (GFX9/RDNA through GFX11, then GFX12).
constexpr bool smem_is_buffer_load_op(uint32_t op) { return op >= 8 && op <= 12; }
/// @brief GFX9 scalar-buffer loads, stores and atomics.
constexpr bool gfx9_smem_is_buffer_op(uint32_t op) {
  return smem_is_buffer_load_op(op) || (op >= 24 && op <= 26) || (op >= 64 && op <= 76) ||
         (op >= 96 && op <= 108);
}
/// @brief Ordinary GFX12 scalar loads; prefetches have defined drop behavior.
constexpr bool gfx12_smem_is_ordinary_load_op(uint32_t op) {
  return op <= 5 || (op >= 8 && op <= 11);
}
constexpr bool gfx12_smem_is_buffer_load_op(uint32_t op) {
  return (op >= 16 && op <= 21) || (op >= 24 && op <= 27);
}

/// @brief GFX9 SMEM s_scratch_{load,store}_dword{,x2,x4} opcodes.
constexpr bool smem_is_scratch_op(uint32_t op) {
  return (op >= 5 && op <= 7) || (op >= 21 && op <= 23);
}

/// @brief Compute scalar address for SMEM encoding.
///
/// @details GFX9 scalar memory addressing:
/// ADDR = SGPR[base] + inst_offset + {SGPR[offset] or M0 or 0}, with the register
/// component scaled by 64 for s_scratch_*. The two LSBs of each byte component are
/// ignored. SBASE, OFFSET[6:0] (IMM=0) and SOFFSET (SOE=1) are scalar-source selectors.
///
/// Requires: inst.op, inst.sbase, inst.soffset_en, inst.soffset, inst.imm, inst.offset.
template <typename SmemInst>
std::optional<uint64_t> smem_calculate_address(const SmemInst &inst, amdgpu::Wavefront &wf,
                                               ScalarMemState *state = nullptr,
                                               bool check_negative_offset = false) {
  constexpr uint64_t kDwordMask = ~0x3ULL;
  auto base = amdgpu::try_read_scalar_selector64(wf, inst.sbase * 2);
  if (!base)
    return std::nullopt;
  *base &= kDwordMask;

  int64_t inst_offset = 0;
  if (inst.imm)
    inst_offset = static_cast<int64_t>(static_cast<int32_t>(inst.offset << 11) >> 11);

  uint64_t reg_offset = 0;
  if (inst.soffset_en || !inst.imm) {
    const uint32_t selector = inst.soffset_en ? inst.soffset : inst.offset & 0x7F;
    auto value = amdgpu::try_read_scalar_selector(wf, selector);
    if (!value)
      return std::nullopt;
    reg_offset = *value;
  }
  if (smem_is_scratch_op(inst.op))
    reg_offset *= 64;
  // Qualify the non-buffer rule before alignment, so compensating low bits
  // cannot turn a nonnegative raw offset into a warning.
  if (check_negative_offset && !gfx9_smem_is_buffer_op(inst.op) &&
      inst_offset + static_cast<int64_t>(reg_offset) < 0)
    wf.report_undefined_behavior("negative combined scalar-memory offset");
  inst_offset &= ~0x3LL;
  reg_offset &= kDwordMask;
  if (gfx9_smem_is_buffer_op(inst.op))
    return scalar_buffer_address(wf, inst.sbase * 2, *base, inst_offset + reg_offset, state);
  const uint64_t addr = *base + inst_offset + reg_offset;
  util::Logger::vm([&](auto &os) {
    static thread_local uint64_t smem_count = 0;
    if (++smem_count <= 12 || (smem_count % 240) == 0)
      os << std::format("SMEM #{} op={} base={:#x} inst_off={:#x} reg_off={:#x} imm={} soff_en={} "
                        "addr={:#x} raw_off={}",
                        smem_count, inst.op, *base, inst_offset, reg_offset, inst.imm,
                        inst.soffset_en, addr, inst.offset);
  });
  return addr;
}

#if RJ_HOST_HAS_AVX2
namespace detail {

/// @brief Write addresses[lane] + bias for lanes [0, WF), eight lanes per step.
///
/// @details Inactive lanes are written as 0 rather than skipped: an AND with
/// the EXEC mask replaces masked stores, which cost more than the plain stores
/// they guard. That matches skipping because callers either pass a freshly
/// constructed VectorMemState, whose per_lane_addr is value-initialized, or
/// never read inactive lanes.
template <uint32_t WF>
inline void ds_lane_addresses_avx2(const uint32_t *addresses, uint64_t exec, uint32_t bias,
                                   uint64_t *per_lane_addr) {
  static_assert(WF % 8 == 0 && WF <= 64);
  const __m256i vbias = _mm256_set1_epi32(static_cast<int>(bias));
  // Moves bit i of an 8-lane EXEC chunk into the sign bit of 32-bit element i.
  const __m256i bit_to_sign = _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24);
#pragma GCC unroll 8
  for (uint32_t lane = 0; lane < WF; lane += 8) {
    const __m256i chunk = _mm256_set1_epi32(static_cast<int>((exec >> lane) & 0xFF));
    const __m256i active = _mm256_srai_epi32(_mm256_sllv_epi32(chunk, bit_to_sign), 31);
    const __m256i lanes = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(addresses + lane));
    // 32-bit add, then zero-extend: same wraparound as the scalar uint32_t sum.
    const __m256i sum = _mm256_and_si256(_mm256_add_epi32(lanes, vbias), active);
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(per_lane_addr + lane),
                        _mm256_cvtepu32_epi64(_mm256_castsi256_si128(sum)));
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(per_lane_addr + lane + 4),
                        _mm256_cvtepu32_epi64(_mm256_extracti128_si256(sum, 1)));
  }
}

} // namespace detail
#endif

/// @brief Write addresses[lane] + bias, a 32-bit sum, for the active lanes.
/// @details Lanes inactive in @p exec are either left unchanged or written as 0.
inline void ds_lane_addresses(std::span<const uint32_t> addresses, uint64_t exec, uint32_t wf_size,
                              uint32_t bias, uint64_t *per_lane_addr) {
  uint32_t lane = 0;
#if RJ_HOST_HAS_AVX2
  // Constant wave sizes let the AVX2 loop unroll fully.
  if (wf_size == 64) {
    detail::ds_lane_addresses_avx2<64>(addresses.data(), exec, bias, per_lane_addr);
    lane = 64;
  } else if (wf_size == 32) {
    detail::ds_lane_addresses_avx2<32>(addresses.data(), exec, bias, per_lane_addr);
    lane = 32;
  }
#endif
  for (; lane < wf_size; ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    per_lane_addr[lane] = addresses[lane] + bias;
  }
}

/// @brief Compute per-lane addresses for DS encoding.
///
/// @details Populates d.per_lane_addr, d.lane_mask, and d.exec_mask.
/// The DS encoding splits the 16-bit offset into two 8-bit fields (offset0
/// and offset1). For non-dual operations (ds_write_b32, ds_read_b32, etc.),
/// these form a single 16-bit byte offset: (offset1 << 8) | offset0.
template <typename DsInst>
void ds_calculate_addresses(const DsInst &inst, amdgpu::Wavefront &wf, VectorMemState &d) {
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  d.lane_mask = exec;
  d.exec_mask = exec;
  d.wf_size = wf.wf_size();
  d.wg_id = wf.wg_id();
  d.wf_id = wf.wf_id();
  uint32_t offset = (static_cast<uint32_t>(inst.offset1) << 8) | inst.offset0;
  RegisterAccess regs(cu);
  auto addr_region = regs.read_vgpr_region(wf.vgpr_alloc().base + inst.addr, 1, exec);
  // Inactive lanes of per_lane_addr end up 0, as constructed.
  ds_lane_addresses(addr_region.lanes(), exec, wf.wf_size(), offset + wf.lds_base(),
                    d.per_lane_addr.data());

  util::Logger::vm([&](auto &os) {
    static uint64_t ds_addr_count = 0;
    if (++ds_addr_count > 100)
      return;
    os << std::format("DS addr: {} wg[{}] wf[{}] v{}+{:#x} lds_base={} is_load={}",
                      wf.cu().full_path(), wf.wg_id(), wf.wf_id(), inst.addr, offset, wf.lds_base(),
                      d.is_load);
    for (uint32_t ln = 0; ln < wf.wf_size(); ++ln) {
      if (!(d.lane_mask & (1ULL << ln)))
        continue;
      os << std::format(" L{}:{:#x}", ln, d.per_lane_addr[ln]);
    }
  });
}

} // namespace addr_calc
} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_ISA_ARCH_AMDGPU_SHARED_ADDR_CALC_SCALAR_H_
