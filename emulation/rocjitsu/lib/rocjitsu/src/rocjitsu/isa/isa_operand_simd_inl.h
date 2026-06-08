// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file isa_operand_simd_inl.h
/// @brief Out-of-line definitions for the SIMD fast-path overrides on
/// `AmdgpuIsaOperand<Isa>`. Pulled into per-arch `operand.cpp` so
/// vtable emission picks up the bodies; not included from the
/// lightweight `operand.h` so analysis-only translation units avoid the
/// heavy Wavefront / ComputeUnit headers.

#ifndef ROCJITSU_ISA_ISA_OPERAND_SIMD_INL_H_
#define ROCJITSU_ISA_ISA_OPERAND_SIMD_INL_H_

#include "rocjitsu/isa/operand.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include <algorithm>
#include <cstring>

namespace rocjitsu {

namespace detail {
template <typename Isa, typename Op>
std::optional<uint32_t> resolved_vgpr_offset_for_operand(const amdgpu::Wavefront &wf,
                                                         const Op &op) {
  if constexpr (requires {
                  Isa::resolved_vgpr_offset(wf, op.opr_type_, op.encoding_value_,
                                            op.vgpr_msb_role());
                }) {
    return Isa::resolved_vgpr_offset(wf, op.opr_type_, op.encoding_value_, op.vgpr_msb_role());
  } else {
    (void)wf;
    return Isa::resolved_vgpr_offset(op.opr_type_, op.encoding_value_);
  }
}
} // namespace detail

template <typename Isa> bool AmdgpuIsaOperand<Isa>::simd_capable() const {
  if (this->delegate())
    return this->delegate()->simd_capable();
  return Isa::simd_capable_value(this->opr_type_, this->encoding_value_);
}

template <typename Isa>
void AmdgpuIsaOperand<Isa>::read_lane_chunk(const amdgpu::Wavefront &wf, uint32_t lane_base,
                                            uint32_t count, uint32_t *out) const {
  if (this->delegate()) {
    this->delegate()->read_lane_chunk(wf, lane_base, count, out);
    return;
  }
  if (auto off = detail::resolved_vgpr_offset_for_operand<Isa>(wf, *this)) {
    const uint8_t *src = wf.cu().vgpr_data(wf.vgpr_alloc().base + *off);
    std::memcpy(out, src + lane_base * sizeof(uint32_t), count * sizeof(uint32_t));
    return;
  }
  std::fill_n(out, count, Isa::simd_broadcast_value(wf, this->opr_type_, this->encoding_value_));
}

template <typename Isa>
void AmdgpuIsaOperand<Isa>::write_lane_chunk(amdgpu::Wavefront &wf, uint32_t lane_base,
                                             uint32_t count, const uint32_t *vals,
                                             uint64_t mask) const {
  auto off = detail::resolved_vgpr_offset_for_operand<Isa>(wf, *this);
  if (!off) {
    Operand::write_lane_chunk(wf, lane_base, count, vals, mask);
    return;
  }
  uint32_t reg = wf.vgpr_alloc().base + *off;
  uint64_t full_mask = util::mask<uint64_t>(static_cast<int>(count));
  if ((mask & full_mask) == full_mask) {
    uint8_t *dst = wf.cu().vgpr_data(reg);
    std::memcpy(dst + lane_base * sizeof(uint32_t), vals, count * sizeof(uint32_t));
    return;
  }
  for (uint32_t i = 0; i < count; ++i)
    if (mask & (1ULL << i))
      wf.cu().write_vgpr(reg, lane_base + i, vals[i]);
}

template <typename Isa>
const uint32_t *AmdgpuIsaOperand<Isa>::simd_lane_ptr(const amdgpu::Wavefront &wf,
                                                     uint32_t lane_base) const {
  if (this->delegate())
    return amdgpu::SimdAccess::lane_ptr(*this->delegate(), wf, lane_base);
  if (auto off = detail::resolved_vgpr_offset_for_operand<Isa>(wf, *this)) {
    const uint8_t *base = wf.cu().vgpr_data(wf.vgpr_alloc().base + *off);
    return reinterpret_cast<const uint32_t *>(base + lane_base * sizeof(uint32_t));
  }
  return nullptr;
}

template <typename Isa>
uint32_t *AmdgpuIsaOperand<Isa>::simd_dst_ptr(amdgpu::Wavefront &wf, uint32_t lane_base) const {
  if (auto off = detail::resolved_vgpr_offset_for_operand<Isa>(wf, *this)) {
    uint8_t *base = wf.cu().vgpr_data(wf.vgpr_alloc().base + *off);
    return reinterpret_cast<uint32_t *>(base + lane_base * sizeof(uint32_t));
  }
  return nullptr;
}

template <typename Isa>
void AmdgpuIsaOperand<Isa>::simd_notify_read(const amdgpu::Wavefront &wf, uint32_t lane_begin,
                                             uint32_t lane_end, uint8_t byte_mask) const {
  if (auto off = Isa::resolved_vgpr_offset(this->opr_type_, this->encoding_value_)) {
    uint32_t physical_reg = wf.vgpr_alloc().base + *off;
    wf.cu().notify_vgpr_read(&wf, physical_reg, lane_begin, lane_end, byte_mask);
  }
}

} // namespace rocjitsu

#endif // ROCJITSU_ISA_ISA_OPERAND_SIMD_INL_H_
