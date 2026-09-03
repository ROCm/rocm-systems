// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_SHARED_ADDR_CALC_BUFFER_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_SHARED_ADDR_CALC_BUFFER_H_

/// @file Shared address calculation for MUBUF and MTBUF (buffer) instructions.
///
/// @details These are vector memory operations that access global memory through
/// a buffer resource descriptor (SRD). Templated on the machine instruction
/// type so they work with any ISA family whose encoding struct exposes the
/// required field names.

#include "rocjitsu/isa/arch/amdgpu/shared/scalar_operand_read.h"
#include "rocjitsu/isa/isa_traits.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/mem_state.h"
#include "rocjitsu/vm/amdgpu/register_access.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "util/log.h"

#include <algorithm>
#include <bit>
#include <cassert>
#include <cstdint>
#include <optional>

namespace rocjitsu {
namespace amdgpu {
namespace addr_calc {

constexpr uint32_t buffer_offset_part(uint32_t voffset, int64_t inst_offset) {
  // Hardware forms this sub-expression in 32-bit offset space before it is
  // widened and added to the descriptor base address.
  return voffset + static_cast<uint32_t>(inst_offset);
}

constexpr uint64_t buffer_total_offset(uint32_t index, uint32_t stride, uint32_t offset_part,
                                       uint32_t soffset) {
  return static_cast<uint64_t>(index) * stride + offset_part + soffset;
}

/// @brief GFX9 MUBUF buffer_{load,store}_format[_d16][_hi]_* opcodes.
constexpr bool gfx9_mubuf_is_format_op(uint32_t op) { return op <= 15 || op == 38 || op == 39; }

/// @brief GFX9/CDNA V# fields that shape the buffer address (ISA "Buffer Resource").
struct Gfx9BufferResource {
  uint32_t stride;       ///< STRIDE, word1[29:16].
  bool swizzle_enable;   ///< SWIZZLE_ENABLE, word1[31].
  uint32_t data_format;  ///< DATA_FORMAT, word3[18:15]; STRIDE[17:14] for non-format ADD_TID ops.
  uint32_t index_stride; ///< INDEX_STRIDE, word3[22:21], in indices (8, 16, 32 or 64).
  bool add_tid_enable;   ///< ADD_TID_ENABLE, word3[23].
};

constexpr Gfx9BufferResource gfx9_buffer_resource(uint32_t srd1, uint32_t srd3) {
  return {.stride = (srd1 >> 16) & 0x3FFFu,
          .swizzle_enable = (srd1 >> 31) != 0,
          .data_format = (srd3 >> 15) & 0xFu,
          .index_stride = 8u << ((srd3 >> 21) & 0x3u),
          .add_tid_enable = ((srd3 >> 23) & 1) != 0};
}

/// @brief Swizzle element size. GFX9 has no ELEMENT_SIZE field (word3[20:19] are User VM bits).
constexpr uint32_t kGfx9SwizzleElemSize = 4;

/// @brief Byte offset from the V# base of @p lane's access on GFX9/CDNA (ISA "Buffer Addressing").
///
/// @details ADD_TID_ENABLE adds the lane to the index (gfx950 drops it for a linear buffer with
/// IDXEN) and then, for non-format ops, extends STRIDE with DATA_FORMAT; SWIZZLE_ENABLE selects
/// the swizzled layout. SOFFSET is added last and is never swizzled.
constexpr uint64_t gfx9_buffer_lane_offset(const Gfx9BufferResource &srd, bool idxen,
                                           bool format_op, uint32_t index, uint32_t offset_part,
                                           uint32_t soffset, uint32_t lane) {
  const bool add_lane = srd.add_tid_enable && (srd.swizzle_enable || !idxen);
  if (add_lane)
    index += lane;
  const uint32_t stride =
      add_lane && !format_op ? (srd.data_format << 14) | srd.stride : srd.stride;
  if (!srd.swizzle_enable)
    return buffer_total_offset(index, stride, offset_part, soffset);
  const uint32_t index_msb = index / srd.index_stride;
  const uint32_t index_lsb = index % srd.index_stride;
  const uint32_t offset_msb = offset_part / kGfx9SwizzleElemSize;
  const uint32_t offset_lsb = offset_part % kGfx9SwizzleElemSize;
  return (uint64_t{index_msb} * stride + uint64_t{offset_msb} * kGfx9SwizzleElemSize) *
             srd.index_stride +
         index_lsb * kGfx9SwizzleElemSize + offset_lsb + soffset;
}

/// @brief SWIZZLE_ENABLE: consecutive dwords of a lane sit INDEX_STRIDE elements apart in memory.
inline void gfx9_buffer_apply_swizzle(VectorMemState &d, const Gfx9BufferResource &srd,
                                      uint64_t exec) {
  if (!srd.swizzle_enable)
    return;
  d.scratch_swizzle = true;
  d.scratch_lane_mask = exec;
  d.scratch_addr_stride = kGfx9SwizzleElemSize * srd.index_stride;
}

/// @brief Operands of the GFX9/CDNA buffer range check for one lane.
struct Gfx9BufferRange {
  bool add_tid_enable;  ///< V# ADD_TID_ENABLE.
  bool idxen;           ///< Instruction IDXEN.
  uint32_t stride;      ///< V# STRIDE in bytes.
  uint32_t num_records; ///< V# NUM_RECORDS: records when index-checked, else bytes.
  uint32_t index;       ///< VGPR index (0 unless IDXEN).
  uint64_t byte_offset; ///< buffer_offset (VOFFSET + inst_offset, 32-bit) + SOFFSET; not wrapped.
};

/// @brief Number of leading elements of a lane's access that pass the GFX9/CDNA range check.
///
/// @details GFX9 V# has no OOB_SELECT: ADD_TID_ENABLE && !IDXEN is unchecked, IDXEN && STRIDE != 0
/// checks index < NUM_RECORDS, else element i needs byte_offset + (i + 1) * elem_size <=
/// NUM_RECORDS. SGPR-offset participation and the per-dword clamp follow gfx950 hardware.
constexpr uint32_t gfx9_buffer_elems_in_range(const Gfx9BufferRange &range, uint32_t elem_size,
                                              uint32_t num_elems) {
  if (range.add_tid_enable && !range.idxen)
    return num_elems;
  if (range.idxen && range.stride != 0)
    return range.index < range.num_records ? num_elems : 0;
  if (range.byte_offset >= range.num_records)
    return 0;
  return static_cast<uint32_t>(
      std::min<uint64_t>(num_elems, (range.num_records - range.byte_offset) / elem_size));
}

/// @brief Apply the GFX9/CDNA range check to @p lane; returns whether the lane stays active.
///
/// @details Dwordx{2,3,4} loads and stores are clamped per element through
/// d.element_lane_masks (sized by the caller); format ops, atomics and MTBUF are all or nothing.
inline bool gfx9_buffer_range_check_lane(VectorMemState &d, uint32_t lane,
                                         const Gfx9BufferRange &range, bool per_element) {
  assert(d.elem_size != 0 && d.num_elems != 0);
  const uint32_t in_range = gfx9_buffer_elems_in_range(range, d.elem_size, d.num_elems);
  if (!per_element)
    return in_range == d.num_elems;
  for (uint32_t elem = in_range; elem < d.num_elems; ++elem)
    d.element_lane_masks[elem] &= ~(uint64_t{1} << lane);
  return in_range != 0;
}

/// @brief Compute per-lane addresses for MUBUF encoding.
///
/// @details Populates d.per_lane_addr, d.lane_mask, d.exec_mask and, on GFX9/CDNA,
/// d.element_lane_masks. Out-of-bounds lanes are excluded from lane_mask (loads
/// return 0, stores are dropped); d.elem_size and d.num_elems must be set.
///
/// Requires: inst.op, inst.srsrc, inst.soffset, inst.idxen, inst.offen, inst.vaddr,
///           inst.offset.
template <typename MubufInst>
void mubuf_calculate_addresses(const MubufInst &inst, amdgpu::Wavefront &wf, VectorMemState &d) {
  RegisterAccess regs(wf);
  uint64_t exec = wf.exec();
  d.lane_mask = exec;
  d.exec_mask = exec;
  d.wf_size = wf.wf_size();
  d.wg_id = wf.wg_id();
  d.wf_id = wf.wf_id();
  const uint32_t sb_sel = inst.srsrc * 4;
  if (!amdgpu::scalar_selector_range_is_backed(wf, sb_sel, 4)) {
    reject_vector_memory_access(d);
    return;
  }
  uint32_t srd0 = amdgpu::read_scalar_selector(wf, sb_sel);
  uint32_t srd1 = amdgpu::read_scalar_selector(wf, sb_sel + 1);
  uint32_t srd2 = amdgpu::read_scalar_selector(wf, sb_sel + 2);
  uint32_t srd3 = amdgpu::read_scalar_selector(wf, sb_sel + 3);
  uint64_t base_addr = (static_cast<uint64_t>(srd1 & 0xFFFF) << 32) | srd0;
  // soffset field: 0-105 = SGPR index, 128 (0x80) = inline constant 0.
  uint32_t soffset_val = 0;
  if (inst.soffset != 0x80) {
    auto soffset = amdgpu::try_read_scalar_selector(wf, inst.soffset);
    if (!soffset) {
      reject_vector_memory_access(d);
      return;
    }
    soffset_val = *soffset;
  }
  // Buffer bounds checking: OOB loads return 0, OOB stores are dropped.
  // num_records is the buffer size in bytes, or in records when index-checked.
  uint32_t num_records = srd2;
  util::Logger::vm([&](auto &os) {
    uint32_t wgid = wf.wg_id();
    if (wgid == 0)
      os << std::format("{} wg[{}] wf[{}] MUBUF addr: srsrc=s[{}:{}]"
                        " srd=[{:#x},{:#x},{:#x},{:#x}] base={:#x}"
                        " soff={:#x} offset={:#x} offen={} idxen={} vaddr=v{}"
                        " num_records={}",
                        wf.cu().full_path(), wf.wg_id(), wf.wf_id(), inst.srsrc * 4,
                        inst.srsrc * 4 + 3, srd0, srd1, srd2, srd3, base_addr, soffset_val,
                        inst.offset, inst.offen, inst.idxen, inst.vaddr, num_records);
  });
  // GFX9 MUBUF address calculation per ISA Table 42 / Section 9.1.5.2:
  //
  // VGPR assignment (idxen × offen):
  //   0,0 → no VGPRs          0,1 → vaddr = offset
  //   1,0 → vaddr = index     1,1 → vaddr = index, vaddr+1 = offset
  //
  // Address = base + soffset + (index * stride) + voffset + inst_offset
  //
  // RDNA OOB modes (srd[3] bit 31); GFX9/CDNA: gfx9_buffer_lane_offset, gfx9_buffer_elems_in_range.
  //   1 = raw:        OOB if (voffset + inst_offset) >= num_records
  //   0 = structured: OOB if stride > 0 ? (index >= num_records)
  //                              else    (voffset + inst_offset) >= num_records
  uint32_t stride = (srd1 >> 16) & 0x3FFF;
  bool oob_raw = (srd3 >> 31) & 1;
  const bool gfx9 = arch_is_cdna_4_or_lower(wf.cu().arch());
  const Gfx9BufferResource gfx9_srd = gfx9_buffer_resource(srd1, srd3);
  const bool format_op = gfx9_mubuf_is_format_op(inst.op);
  const bool per_element = gfx9 && d.num_elems > 1 && d.atomic_op == AtomicOp::NONE && !format_op;
  d.element_lane_masks.clear();
  if (per_element)
    d.element_lane_masks.assign(d.num_elems, exec);
  if (gfx9)
    gfx9_buffer_apply_swizzle(d, gfx9_srd, exec);
  uint32_t vgpr_base = wf.vgpr_alloc().base + inst.vaddr;
  std::optional<RegisterAccess::VgprReadRegion> vaddr_region;
  if (inst.idxen || inst.offen) {
    uint32_t reg_count = (inst.idxen && inst.offen) ? 2 : 1;
    vaddr_region.emplace(regs.read_vgpr_region(vgpr_base, reg_count, exec));
  }
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t index = 0;
    uint32_t voffset = 0;
    if (inst.idxen && inst.offen) {
      index = vaddr_region->lane(0, lane);
      voffset = vaddr_region->lane(1, lane);
    } else if (inst.idxen) {
      index = vaddr_region->lane(0, lane);
    } else if (inst.offen) {
      voffset = vaddr_region->lane(0, lane);
    }
    uint32_t offset_part = buffer_offset_part(voffset, inst.offset);
    uint64_t total_offset = gfx9 ? gfx9_buffer_lane_offset(gfx9_srd, inst.idxen != 0, format_op,
                                                           index, offset_part, soffset_val, lane)
                                 : buffer_total_offset(index, stride, offset_part, soffset_val);
    // OOB check.
    bool oob;
    if (gfx9) {
      oob = !gfx9_buffer_range_check_lane(d, lane,
                                          {.add_tid_enable = gfx9_srd.add_tid_enable,
                                           .idxen = inst.idxen != 0,
                                           .stride = stride,
                                           .num_records = num_records,
                                           .index = index,
                                           .byte_offset = uint64_t{offset_part} + soffset_val},
                                          per_element);
    } else if (oob_raw) {
      oob = offset_part >= num_records;
    } else if (stride > 0) {
      oob = index >= num_records;
    } else {
      oob = offset_part >= num_records;
    }
    if (oob) {
      d.lane_mask &= ~(1ULL << lane);
      d.per_lane_addr[lane] = 0;
    } else {
      d.per_lane_addr[lane] = (base_addr + total_offset) & 0xFFFFFFFFFFFFULL;
    }
  }
  // Per-lane address trace: log the first 4 active lanes so we can verify
  // that each lane's voffset (and thus effective address) is correct.
  util::Logger::vm([&](auto &os) {
    static uint64_t pl_count = 0;
    if (wf.wg_id() != 0 && ++pl_count > 500)
      return;
    os << std::format("{} wg[{}] wf[{}] MUBUF per-lane: stride={}", wf.cu().full_path(), wf.wg_id(),
                      wf.wf_id(), stride);
    for (uint32_t ln = 0; ln < wf.wf_size(); ++ln) {
      if (!(d.lane_mask & (1ULL << ln)))
        continue;
      os << std::format(" L{}:{:#x}", ln, d.per_lane_addr[ln]);
    }
    os << std::format(" exec={:#x} lane_mask={:#x}", exec, d.lane_mask);
  });
}

/// @brief Compute per-lane addresses for MTBUF encoding.
///
/// @details Populates d.per_lane_addr, d.lane_mask, and d.exec_mask. Typed accesses
/// are range-checked all-or-nothing; d.elem_size and d.num_elems must be set.
///
/// Requires: inst.srsrc, inst.soffset, inst.idxen, inst.offen, inst.vaddr,
///           inst.offset.
template <typename MtbufInst>
void mtbuf_calculate_addresses(const MtbufInst &inst, amdgpu::Wavefront &wf, VectorMemState &d) {
  RegisterAccess regs(wf);
  uint64_t exec = wf.exec();
  d.lane_mask = exec;
  d.exec_mask = exec;
  d.wf_size = wf.wf_size();
  d.wg_id = wf.wg_id();
  d.wf_id = wf.wf_id();
  const uint32_t sb_sel = inst.srsrc * 4;
  if (!amdgpu::scalar_selector_range_is_backed(wf, sb_sel, 4)) {
    reject_vector_memory_access(d);
    return;
  }
  uint32_t srd0 = amdgpu::read_scalar_selector(wf, sb_sel);
  uint32_t srd1 = amdgpu::read_scalar_selector(wf, sb_sel + 1);
  uint32_t srd2 = amdgpu::read_scalar_selector(wf, sb_sel + 2);
  uint32_t srd3 = amdgpu::read_scalar_selector(wf, sb_sel + 3);
  uint64_t base_addr = (static_cast<uint64_t>(srd1 & 0xFFFF) << 32) | srd0;
  uint32_t soffset_val = 0;
  if (inst.soffset != 0x80) {
    auto soffset = amdgpu::try_read_scalar_selector(wf, inst.soffset);
    if (!soffset) {
      reject_vector_memory_access(d);
      return;
    }
    soffset_val = *soffset;
  }
  uint32_t num_records = srd2;
  uint32_t stride = (srd1 >> 16) & 0x3FFF;
  bool oob_raw = (srd3 >> 31) & 1;
  const bool gfx9 = arch_is_cdna_4_or_lower(wf.cu().arch());
  const Gfx9BufferResource gfx9_srd = gfx9_buffer_resource(srd1, srd3);
  if (gfx9)
    gfx9_buffer_apply_swizzle(d, gfx9_srd, exec);
  uint32_t vgpr_base = wf.vgpr_alloc().base + inst.vaddr;
  std::optional<RegisterAccess::VgprReadRegion> vaddr_region;
  if (inst.idxen || inst.offen) {
    uint32_t reg_count = (inst.idxen && inst.offen) ? 2 : 1;
    vaddr_region.emplace(regs.read_vgpr_region(vgpr_base, reg_count, exec));
  }
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t index = 0;
    uint32_t voffset = 0;
    if (inst.idxen && inst.offen) {
      index = vaddr_region->lane(0, lane);
      voffset = vaddr_region->lane(1, lane);
    } else if (inst.idxen) {
      index = vaddr_region->lane(0, lane);
    } else if (inst.offen) {
      voffset = vaddr_region->lane(0, lane);
    }
    uint32_t offset_part = buffer_offset_part(voffset, inst.offset);
    uint64_t total_offset =
        gfx9 ? gfx9_buffer_lane_offset(gfx9_srd, inst.idxen != 0,
                                       /*format_op=*/true, index, offset_part, soffset_val, lane)
             : buffer_total_offset(index, stride, offset_part, soffset_val);
    bool oob;
    if (gfx9) {
      oob = !gfx9_buffer_range_check_lane(d, lane,
                                          {.add_tid_enable = gfx9_srd.add_tid_enable,
                                           .idxen = inst.idxen != 0,
                                           .stride = stride,
                                           .num_records = num_records,
                                           .index = index,
                                           .byte_offset = uint64_t{offset_part} + soffset_val},
                                          /*per_element=*/false);
    } else if (oob_raw) {
      oob = offset_part >= num_records;
    } else if (stride > 0) {
      oob = index >= num_records;
    } else {
      oob = offset_part >= num_records;
    }
    if (oob) {
      d.lane_mask &= ~(1ULL << lane);
      d.per_lane_addr[lane] = 0;
    } else {
      d.per_lane_addr[lane] = (base_addr + total_offset) & 0xFFFFFFFFFFFFULL;
    }
  }
}

} // namespace addr_calc
} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_ISA_ARCH_AMDGPU_SHARED_ADDR_CALC_BUFFER_H_
