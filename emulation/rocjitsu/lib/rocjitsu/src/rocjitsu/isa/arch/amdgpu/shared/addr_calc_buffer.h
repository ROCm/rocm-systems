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

#include "rocjitsu/isa/arch/amdgpu/shared/buffer_address.h"
#include "rocjitsu/isa/arch/amdgpu/shared/scalar_operand_read.h"
#include "rocjitsu/isa/arch/amdgpu/shared/scalar_operand_resolve.h"
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
#include <span>

namespace rocjitsu {
namespace amdgpu {
namespace addr_calc {

/// RDNA buffer resources can end in VCC, as with s[104:107]. Validate the
/// SGPR pair and VCC pair independently; neither may read another wave's storage.
inline bool buffer_resource_range_is_backed(const Wavefront &wf, uint32_t selector) {
  if ((wf.cu().arch() == ROCJITSU_CODE_ARCH_RDNA1 || wf.cu().arch() == ROCJITSU_CODE_ARCH_RDNA2 ||
       wf.cu().arch() == ROCJITSU_CODE_ARCH_RDNA3 || wf.cu().arch() == ROCJITSU_CODE_ARCH_RDNA3_5 ||
       wf.cu().arch() == ROCJITSU_CODE_ARCH_RDNA4 || wf.cu().arch() == ROCJITSU_CODE_ARCH_CDNA5) &&
      selector == 104)
    return scalar_selector_range_is_backed(wf, selector, 2) &&
           scalar_selector_range_is_backed(wf, selector + 2, 2);
  return scalar_selector_range_is_backed(wf, selector, 4);
}

/// MUBUF/MTBUF SOFFSET accepts scalar registers and inline integer/float constants.
/// Constants supply their raw 32-bit value as an unsigned byte offset.
inline std::optional<uint32_t> buffer_scalar_offset(Wavefront &wf, uint32_t selector) {
  if (auto constant = resolve_src_scalar_statically(selector))
    return static_cast<uint32_t>(*constant);
  if (wf.cu().arch() == ROCJITSU_CODE_ARCH_RDNA1 &&
      ((selector >= 235 && selector <= 239) || (selector >= 251 && selector <= 253)))
    return resolve_src_scalar(wf, selector, kLegacyM0Selector);
  return try_read_scalar_selector(wf, selector);
}

constexpr uint32_t buffer_offset_part(uint32_t voffset, int64_t inst_offset) {
  // Hardware forms this sub-expression in 32-bit offset space before it is
  // widened and added to the descriptor base address.
  return voffset + static_cast<uint32_t>(inst_offset);
}

constexpr uint64_t buffer_total_offset(uint32_t index, uint32_t stride, uint32_t offset_part,
                                       uint32_t soffset) {
  return static_cast<uint64_t>(index) * stride + offset_part + soffset;
}

/// @brief RDNA V# OOB_SELECT policies from SQ_BUF_RSRC_WORD3[29:28].
enum class RdnaBufferOobSelect : uint32_t {
  IndexAndOffset = 0,
  IndexOnly = 1,
  NumRecordsZero = 2,
  Complete = 3,
};

constexpr RdnaBufferOobSelect rdna_buffer_oob_select(uint32_t srd3) {
  return static_cast<RdnaBufferOobSelect>((srd3 >> 28) & 3);
}

/// @brief GFX9 MUBUF buffer_{load,store}_format[_d16][_hi]_* opcodes.
constexpr bool gfx9_mubuf_is_format_op(uint32_t op) { return op <= 15 || op == 38 || op == 39; }

/// @brief GFX9/CDNA V# fields that shape the buffer address (ISA "Buffer Resource").
struct Gfx9BufferResource {
  uint32_t stride;           ///< STRIDE, word1[29:16].
  bool swizzle_enable;       ///< Effective address swizzle on this generation.
  bool add_tid_when_indexed; ///< Add the lane to a VGPR-supplied index.
  uint32_t data_format;      ///< DATA_FORMAT, word3[18:15]; 0 unbound except as STRIDE[17:14]
                             ///< for non-format ADD_TID operations.
  uint32_t index_stride;     ///< INDEX_STRIDE, word3[22:21], in indices (8, 16, 32 or 64).
  bool add_tid_enable;       ///< ADD_TID_ENABLE, word3[23].
};

constexpr Gfx9BufferResource gfx9_buffer_resource(rj_code_arch_t arch, uint32_t srd1,
                                                  uint32_t srd3) {
  const bool add_tid = (srd3 & (1u << 23)) != 0;
  const bool swizzle = (srd1 & (1u << 31)) != 0;
  // MI100 ISA tables 36/39 specify independent swizzle and thread-ID fields.
  // Later CDNA targets retain their existing gfx950-qualified policy.
  return {.stride = (srd1 >> 16) & 0x3FFFu,
          .swizzle_enable = swizzle && (arch == ROCJITSU_CODE_ARCH_CDNA1 || add_tid),
          .add_tid_when_indexed = arch == ROCJITSU_CODE_ARCH_CDNA1 || swizzle,
          .data_format = (srd3 >> 15) & 0xFu,
          .index_stride = 8u << ((srd3 >> 21) & 0x3u),
          .add_tid_enable = add_tid};
}

/// @brief Swizzle element size. GFX9 has no ELEMENT_SIZE field (word3[20:19] are User VM bits).
constexpr uint32_t kGfx9SwizzleElemSize = 4;

/// @brief Byte offset from the V# base of @p lane's access on GFX9/CDNA (ISA "Buffer Addressing").
///
/// @details ADD_TID_ENABLE adds the lane to the index (gfx950 drops it for a linear buffer with
/// IDXEN) and then, for non-format ops, extends STRIDE with DATA_FORMAT. The resource decoder
/// selects swizzling and lane participation for the requested architecture. SOFFSET is added last
/// and is never swizzled.
constexpr uint64_t gfx9_buffer_lane_offset(const Gfx9BufferResource &srd, bool idxen,
                                           bool format_op, uint32_t index, uint32_t offset_part,
                                           uint32_t soffset, uint32_t lane) {
  const bool add_lane = srd.add_tid_enable && (srd.add_tid_when_indexed || !idxen);
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

/// @brief A swizzled scratch V# interleaves consecutive dwords by INDEX_STRIDE elements.
inline void gfx9_buffer_apply_swizzle(VectorMemState &d, const Gfx9BufferResource &srd,
                                      uint64_t exec, uint64_t base_addr, uint32_t soffset) {
  if (!srd.swizzle_enable)
    return;
  d.scratch_swizzle = true;
  d.scratch_lane_mask = exec;
  d.scratch_addr_stride = kGfx9SwizzleElemSize * srd.index_stride;
  d.scratch_addr_base_offset = static_cast<uint32_t>((base_addr + soffset) % kGfx9SwizzleElemSize);
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

struct RdnaBufferAddress {
  uint32_t index;
  bool swizzled;
  uint64_t offset;
};

/// IDXEN supplies the index instead of ADD_TID. Swizzling requires an index source.
/// Physical gfx11/gfx12 keep SOFFSET outside the swizzle, including 16-byte elements.
constexpr RdnaBufferAddress rdna_buffer_address(uint32_t srd1, uint32_t srd3, uint32_t stride,
                                                bool idxen, uint32_t index, uint32_t offset,
                                                uint32_t soffset, uint32_t lane) {
  const bool add_tid = (srd3 >> 23) & 1;
  const uint32_t swizzle = srd1 >> 30;
  if (!idxen)
    index = add_tid ? lane : 0;
  const bool swizzled = swizzle != 0 && (idxen || add_tid);
  if (!swizzled)
    return {index, false, buffer_total_offset(index, stride, offset, soffset)};
  const uint32_t size = swizzle == 3 ? 16 : 4;
  const uint32_t group = 8u << ((srd3 >> 21) & 3);
  return {index, true,
          (uint64_t{index / group} * stride + (offset / size) * uint64_t{size}) * group +
              (index % group) * size + offset % size + soffset};
}

inline void rdna_buffer_apply_swizzle(VectorMemState &d, uint32_t srd1, uint32_t srd3, bool idxen,
                                      uint64_t exec, uint64_t base, uint32_t soffset) {
  const uint32_t swizzle = srd1 >> 30;
  if (!swizzle || (!idxen && !(srd3 & (1u << 23))))
    return;
  d.scratch_swizzle = true;
  d.scratch_lane_mask = exec;
  d.scratch_swizzle_unit = swizzle == 3 ? 16 : 4;
  d.scratch_addr_stride = d.scratch_swizzle_unit * (8u << ((srd3 >> 21) & 3));
  d.scratch_addr_base_offset = (base + soffset) % d.scratch_swizzle_unit;
}

/// @brief Apply the GFX10+ buffer bounds policy, hardware-qualified on GFX11/12.
/// @details For GFX10 mode 3, RDNA 1.0 ISA (Sept. 2020), Table 35, specifies
/// reducing NUM_RECORDS by SOFFSET, checking index and payload end against
/// record count and stride for swizzled records, and checking the payload end
/// against the byte count otherwise. Mesa's ac_set_buf_desc_word3 describes
/// a different GFX10 check on the swizzled address or unswizzled start offset.
/// RJ follows the ISA table pending a GFX10 hardware comparison.
/// https://docs.amd.com/api/khub/documents/mU0vhV4IgdmIWSgRqlt70g/content
inline bool rdna_buffer_range_check_lane(VectorMemState &d, uint32_t lane, RdnaBufferOobSelect mode,
                                         uint32_t stride, uint32_t records, uint32_t index,
                                         uint32_t offset, uint32_t soffset, bool swizzle,
                                         bool per_element, rj_code_arch_t arch) {
  if (mode == RdnaBufferOobSelect::Complete)
    records -= std::min(records, soffset);
  const uint32_t count = per_element ? d.num_elems : 1;
  const uint32_t bytes = per_element ? d.elem_size : d.elem_size * d.num_elems;
  bool any = false;
  for (uint32_t element = 0; element < count; ++element) {
    const uint64_t start = uint64_t{offset} + uint64_t{element} * bytes;
    const uint64_t end = start + bytes;
    bool valid = false;
    switch (mode) {
    case RdnaBufferOobSelect::IndexAndOffset:
      // GFX10.1 Table 35 uses "offset > stride" for rejection. GFX10.3
      // Table 35 changes the comparison to ">=". Both use the starting offset.
      if (arch == ROCJITSU_CODE_ARCH_RDNA1)
        valid = index < records && start <= stride;
      else if (arch == ROCJITSU_CODE_ARCH_RDNA2)
        valid = stride == 0 ? start < records : index < records && start < stride;
      else
        valid = index < records && end <= stride;
      break;
    case RdnaBufferOobSelect::IndexOnly:
      valid = index < records;
      break;
    case RdnaBufferOobSelect::NumRecordsZero:
      valid = records != 0;
      break;
    case RdnaBufferOobSelect::Complete:
      valid = swizzle && stride != 0 ? index < records && end <= stride : end <= records;
      break;
    }
    if (per_element && !valid)
      d.element_lane_masks[element] &= ~(uint64_t{1} << lane);
    any |= valid;
  }
  return any;
}

inline bool uses_gfx10_buffer_addressing(rj_code_arch_t arch) {
  return arch == ROCJITSU_CODE_ARCH_RDNA1 || arch == ROCJITSU_CODE_ARCH_RDNA2;
}

inline bool uses_rdna_buffer_range_check(rj_code_arch_t arch) {
  return arch == ROCJITSU_CODE_ARCH_RDNA3 || arch == ROCJITSU_CODE_ARCH_RDNA3_5 ||
         arch == ROCJITSU_CODE_ARCH_RDNA4;
}

inline bool uses_rdna_buffer_addressing(rj_code_arch_t arch) {
  return uses_gfx10_buffer_addressing(arch) || uses_rdna_buffer_range_check(arch);
}

/// @brief RDNA V# FORMAT is word3[18:12] on GFX10 and word3[17:12] on GFX11/GFX12.
/// @details INVALID (0) unbinds the resource for every vector buffer access:
/// gfx1030, gfx1100 and gfx1201 return zero for loads and returning atomics
/// and leave memory untouched for any OOB_SELECT, with or without ADD_TID.
/// Scalar buffer loads ignore it.
inline uint32_t rdna_buffer_resource_format(rj_code_arch_t arch, uint32_t srd3) {
  const bool six_bits = arch == ROCJITSU_CODE_ARCH_RDNA3 || arch == ROCJITSU_CODE_ARCH_RDNA3_5 ||
                        arch == ROCJITSU_CODE_ARCH_RDNA4;
  return (srd3 >> 12) & (six_bits ? 0x3f : 0x7f);
}

/// Dword and wider GFX10.1 buffer requests ignore the low two address bits.
inline uint64_t buffer_access_address(rj_code_arch_t arch, uint64_t address, uint32_t elem_size) {
  if (arch == ROCJITSU_CODE_ARCH_RDNA1 && elem_size >= sizeof(uint32_t))
    address &= ~uint64_t{3};
  return buffer_virtual_address(address);
}

/// @brief Address GFX10 buffers using their single SWIZZLE_ENABLE bit.
/// @details CACHE_SWIZZLE does not change addresses. The IDXEN+ADD_TID rule
/// follows the RDNA1 manual and remains unqualified on GFX10 hardware; the
/// compiler spill path uses IDXEN=0. Later hardware differs from that manual.
constexpr RdnaBufferAddress gfx10_buffer_address(uint32_t srd1, uint32_t srd3, uint32_t stride,
                                                 bool idxen, uint32_t index, uint32_t offset,
                                                 uint32_t soffset, uint32_t lane) {
  index = (idxen ? index : 0) + ((srd3 & (1u << 23)) ? lane : 0);
  if (!(srd1 & (1u << 31)))
    return {index, false, buffer_total_offset(index, stride, offset, soffset)};
  const uint32_t group = 8u << ((srd3 >> 21) & 3);
  return {index, true,
          (uint64_t{index / group} * stride + (offset / 4) * uint64_t{4}) * group +
              (index % group) * 4 + offset % 4 + soffset};
}

inline void gfx10_buffer_apply_swizzle(VectorMemState &d, uint32_t srd1, uint32_t srd3,
                                       uint64_t exec, uint64_t base, uint32_t soffset) {
  if (!(srd1 & (1u << 31)))
    return;
  d.scratch_swizzle = true;
  d.scratch_lane_mask = exec;
  d.scratch_addr_stride = 4 * (8u << ((srd3 >> 21) & 3));
  d.scratch_addr_base_offset = (base + soffset) % 4;
}

/// Address VGPRs selected by a buffer instruction. Read each register once for
/// the active lanes before calculating their individual addresses.
struct BufferAddressRegisters {
  std::span<const uint32_t> indices;
  std::span<const uint32_t> offsets;
};

inline BufferAddressRegisters read_buffer_address_registers(RegisterAccess &regs, uint32_t base,
                                                            bool idxen, bool offen, uint64_t exec) {
  BufferAddressRegisters result;
  if (idxen || offen) {
    const auto region = regs.read_vgpr_region(base, idxen && offen ? 2 : 1, exec);
    if (idxen)
      result.indices = region.lanes(0).value();
    if (offen)
      result.offsets = region.lanes(idxen ? 1 : 0).value();
  }
  return result;
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
  const rj_code_arch_t arch = wf.cu().arch();
  const bool gfx9 = arch_is_cdna_4_or_lower(arch);
  const bool gfx10 = uses_gfx10_buffer_addressing(arch);
  if (!gfx9 && !uses_rdna_buffer_addressing(arch)) {
    reject_vector_memory_access(d);
    return;
  }
  RegisterAccess regs(wf);
  uint64_t exec = wf.exec();
  d.lane_mask = exec;
  d.exec_mask = exec;
  d.wf_size = wf.wf_size();
  d.wg_id = wf.wg_id();
  d.wf_id = wf.wf_id();
  const uint32_t sb_sel = inst.srsrc * 4;
  if (!buffer_resource_range_is_backed(wf, sb_sel)) {
    reject_vector_memory_access(d);
    return;
  }
  uint32_t srd0 = amdgpu::read_scalar_selector(wf, sb_sel);
  uint32_t srd1 = amdgpu::read_scalar_selector(wf, sb_sel + 1);
  uint32_t srd2 = amdgpu::read_scalar_selector(wf, sb_sel + 2);
  uint32_t srd3 = amdgpu::read_scalar_selector(wf, sb_sel + 3);
  if ((arch == ROCJITSU_CODE_ARCH_RDNA3 || arch == ROCJITSU_CODE_ARCH_RDNA3_5) &&
      (srd3 >> 30) != 0) {
    // A resource type mismatch ignores the entire instruction, including load writeback.
    d.exec_mask = d.lane_mask = 0;
    return;
  }
  const RdnaBufferOobSelect mode = rdna_buffer_oob_select(srd3);
  uint64_t base_addr = (static_cast<uint64_t>(srd1 & 0xFFFF) << 32) | srd0;
  auto soffset = buffer_scalar_offset(wf, inst.soffset);
  if (!soffset) {
    reject_vector_memory_access(d);
    return;
  }
  const uint32_t soffset_val = *soffset;
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
  // RDNA OOB_SELECT is srd[3][29:28]. COMPLETE checks the entire final
  // byte range; the other modes retain their index/offset-specific checks.
  // GFX9/CDNA instead use gfx9_buffer_lane_offset and
  // gfx9_buffer_elems_in_range.
  uint32_t stride = (srd1 >> 16) & 0x3FFF;
  const Gfx9BufferResource gfx9_srd = gfx9_buffer_resource(arch, srd1, srd3);
  const bool format_op = gfx9_mubuf_is_format_op(inst.op);
  // Format preparation rejects INVALID for typed accesses. Raw GFX9 buffers
  // need the same no-access behavior here, before opcode-derived formats hide it.
  // The ISA keys the stride-field exemption on ADD_TID itself, independently
  // of the IDXEN qualification used below for adding lanes to linear buffers.
  if (gfx9 && !format_op && gfx9_srd.data_format == 0 && !gfx9_srd.add_tid_enable) {
    reject_vector_memory_access(d);
    return;
  }
  // Raw RDNA accesses too, with no ADD_TID exemption (see rdna_buffer_resource_format).
  if (!gfx9 && rdna_buffer_resource_format(arch, srd3) == 0) {
    reject_vector_memory_access(d);
    return;
  }
  const bool per_element = d.num_elems > 1 && d.atomic_op == AtomicOp::NONE &&
                           (gfx9 ? !format_op : !d.buffer_components);
  d.element_lane_masks.clear();
  if (per_element)
    d.element_lane_masks.assign(d.num_elems, exec);
  if (gfx9)
    gfx9_buffer_apply_swizzle(d, gfx9_srd, exec, base_addr, soffset_val);
  else if (gfx10)
    gfx10_buffer_apply_swizzle(d, srd1, srd3, exec, base_addr, soffset_val);
  else
    rdna_buffer_apply_swizzle(d, srd1, srd3, inst.idxen != 0, exec, base_addr, soffset_val);
  uint32_t vgpr_base = wf.vgpr_alloc().base + inst.vaddr;
  const auto address_regs =
      read_buffer_address_registers(regs, vgpr_base, inst.idxen, inst.offen, exec);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    const uint32_t index = inst.idxen ? address_regs.indices[lane] : 0;
    const uint32_t voffset = inst.offen ? address_regs.offsets[lane] : 0;
    uint32_t offset_part = buffer_offset_part(voffset, inst.offset);
    const auto rdna_addr = gfx10 ? gfx10_buffer_address(srd1, srd3, stride, inst.idxen != 0, index,
                                                        offset_part, soffset_val, lane)
                                 : rdna_buffer_address(srd1, srd3, stride, inst.idxen != 0, index,
                                                       offset_part, soffset_val, lane);
    uint64_t total_offset = gfx9 ? gfx9_buffer_lane_offset(gfx9_srd, inst.idxen != 0, format_op,
                                                           index, offset_part, soffset_val, lane)
                                 : rdna_addr.offset;
    // OOB check.
    bool oob;
    if (gfx9) {
      // MI100 checks the raw buffer offset before adding the scalar offset.
      const bool cdna1_raw = arch == ROCJITSU_CODE_ARCH_CDNA1 && !inst.idxen &&
                             !gfx9_srd.swizzle_enable && !gfx9_srd.add_tid_enable;
      oob = !gfx9_buffer_range_check_lane(
          d, lane,
          {.add_tid_enable = gfx9_srd.add_tid_enable,
           .idxen = inst.idxen != 0,
           .stride = stride,
           .num_records = num_records,
           .index = index,
           .byte_offset = uint64_t{offset_part} + (cdna1_raw ? 0 : soffset_val)},
          per_element);
    } else {
      oob = !rdna_buffer_range_check_lane(d, lane, mode, stride, num_records, rdna_addr.index,
                                          offset_part, soffset_val, rdna_addr.swizzled, per_element,
                                          arch);
    }
    if (oob) {
      d.lane_mask &= ~(1ULL << lane);
      d.per_lane_addr[lane] = 0;
    } else {
      d.per_lane_addr[lane] = buffer_access_address(arch, base_addr + total_offset, d.elem_size);
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
  const rj_code_arch_t arch = wf.cu().arch();
  const bool gfx9 = arch_is_cdna_4_or_lower(arch);
  const bool gfx10 = uses_gfx10_buffer_addressing(arch);
  if (!gfx9 && !uses_rdna_buffer_addressing(arch)) {
    reject_vector_memory_access(d);
    return;
  }
  RegisterAccess regs(wf);
  uint64_t exec = wf.exec();
  d.lane_mask = exec;
  d.exec_mask = exec;
  d.wf_size = wf.wf_size();
  d.wg_id = wf.wg_id();
  d.wf_id = wf.wf_id();
  const uint32_t sb_sel = inst.srsrc * 4;
  if (!buffer_resource_range_is_backed(wf, sb_sel)) {
    reject_vector_memory_access(d);
    return;
  }
  uint32_t srd0 = amdgpu::read_scalar_selector(wf, sb_sel);
  uint32_t srd1 = amdgpu::read_scalar_selector(wf, sb_sel + 1);
  uint32_t srd2 = amdgpu::read_scalar_selector(wf, sb_sel + 2);
  uint32_t srd3 = amdgpu::read_scalar_selector(wf, sb_sel + 3);
  if ((arch == ROCJITSU_CODE_ARCH_RDNA3 || arch == ROCJITSU_CODE_ARCH_RDNA3_5) &&
      (srd3 >> 30) != 0) {
    // A resource type mismatch ignores the entire instruction, including load writeback.
    d.exec_mask = d.lane_mask = 0;
    return;
  }
  const RdnaBufferOobSelect mode = rdna_buffer_oob_select(srd3);
  uint64_t base_addr = (static_cast<uint64_t>(srd1 & 0xFFFF) << 32) | srd0;
  auto soffset = buffer_scalar_offset(wf, inst.soffset);
  if (!soffset) {
    reject_vector_memory_access(d);
    return;
  }
  const uint32_t soffset_val = *soffset;
  uint32_t num_records = srd2;
  uint32_t stride = (srd1 >> 16) & 0x3FFF;
  const Gfx9BufferResource gfx9_srd = gfx9_buffer_resource(arch, srd1, srd3);
  if (gfx9)
    gfx9_buffer_apply_swizzle(d, gfx9_srd, exec, base_addr, soffset_val);
  else if (gfx10)
    gfx10_buffer_apply_swizzle(d, srd1, srd3, exec, base_addr, soffset_val);
  else
    rdna_buffer_apply_swizzle(d, srd1, srd3, inst.idxen != 0, exec, base_addr, soffset_val);
  uint32_t vgpr_base = wf.vgpr_alloc().base + inst.vaddr;
  const auto address_regs =
      read_buffer_address_registers(regs, vgpr_base, inst.idxen, inst.offen, exec);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    const uint32_t index = inst.idxen ? address_regs.indices[lane] : 0;
    const uint32_t voffset = inst.offen ? address_regs.offsets[lane] : 0;
    uint32_t offset_part = buffer_offset_part(voffset, inst.offset);
    const auto rdna_addr = gfx10 ? gfx10_buffer_address(srd1, srd3, stride, inst.idxen != 0, index,
                                                        offset_part, soffset_val, lane)
                                 : rdna_buffer_address(srd1, srd3, stride, inst.idxen != 0, index,
                                                       offset_part, soffset_val, lane);
    uint64_t total_offset =
        gfx9 ? gfx9_buffer_lane_offset(gfx9_srd, inst.idxen != 0,
                                       /*format_op=*/true, index, offset_part, soffset_val, lane)
             : rdna_addr.offset;
    bool oob;
    if (gfx9) {
      // MI100 checks the raw buffer offset before adding the scalar offset.
      const bool cdna1_raw = arch == ROCJITSU_CODE_ARCH_CDNA1 && !inst.idxen &&
                             !gfx9_srd.swizzle_enable && !gfx9_srd.add_tid_enable;
      oob = !gfx9_buffer_range_check_lane(
          d, lane,
          {.add_tid_enable = gfx9_srd.add_tid_enable,
           .idxen = inst.idxen != 0,
           .stride = stride,
           .num_records = num_records,
           .index = index,
           .byte_offset = uint64_t{offset_part} + (cdna1_raw ? 0 : soffset_val)},
          /*per_element=*/false);
    } else {
      oob =
          !rdna_buffer_range_check_lane(d, lane, mode, stride, num_records, rdna_addr.index,
                                        offset_part, soffset_val, rdna_addr.swizzled, false, arch);
    }
    if (oob) {
      d.lane_mask &= ~(1ULL << lane);
      d.per_lane_addr[lane] = 0;
    } else {
      d.per_lane_addr[lane] = buffer_access_address(arch, base_addr + total_offset, d.elem_size);
    }
  }
}

} // namespace addr_calc
} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_ISA_ARCH_AMDGPU_SHARED_ADDR_CALC_BUFFER_H_
