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
#include "rocjitsu/vm/amdgpu/register_access.h"
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
    uint32_t voff = wf.gpr_idx_en() ? amdgpu::apply_gpr_idx(wf, *off, false) : *off;
    assert(lane_base <= wf.wf_size());
    assert(count <= wf.wf_size() - lane_base);
    uint64_t lane_mask =
        count == 0 ? 0 : util::mask<uint64_t>(static_cast<int>(count)) << lane_base;
    auto region =
        amdgpu::RegisterAccess(wf.cu()).read_vgpr_region(wf.vgpr_alloc().base + voff, 1, lane_mask);
    std::copy_n(region.lanes().begin() + lane_base, count, out);
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
  uint32_t voff = wf.gpr_idx_en() ? amdgpu::apply_gpr_idx(wf, *off, true) : *off;
  uint32_t reg = wf.vgpr_alloc().base + voff;
  uint64_t full_mask = util::mask<uint64_t>(static_cast<int>(count));
  if ((mask & full_mask) == full_mask) {
    uint8_t *dst = wf.cu().raw_cu().raw_vgpr_data(reg);
    std::memcpy(dst + lane_base * sizeof(uint32_t), vals, count * sizeof(uint32_t));
    return;
  }
  for (uint32_t i = 0; i < count; ++i)
    if (mask & (1ULL << i))
      wf.cu().raw_cu().write_vgpr(reg, lane_base + i, vals[i]);
}

namespace detail {
template <typename Isa>
void amdgpu_isa_read_lane_chunk_base(const AmdgpuIsaOperand<Isa> &op, const amdgpu::Wavefront &wf,
                                     uint32_t lane_base, uint32_t count, uint32_t *out) {
  op.AmdgpuIsaOperand<Isa>::read_lane_chunk(wf, lane_base, count, out);
}

template <typename Isa>
void amdgpu_isa_write_lane_chunk_base(const AmdgpuIsaOperand<Isa> &op, amdgpu::Wavefront &wf,
                                      uint32_t lane_base, uint32_t count, const uint32_t *vals,
                                      uint64_t mask) {
  op.AmdgpuIsaOperand<Isa>::write_lane_chunk(wf, lane_base, count, vals, mask);
}
} // namespace detail

template <typename Isa>
std::optional<uint32_t>
AmdgpuIsaOperand<Isa>::simd_vgpr_base_impl(const amdgpu::Wavefront &wf) const {
  if (auto off = detail::resolved_vgpr_offset_for_operand<Isa>(wf, *this))
    return wf.vgpr_alloc().base + (wf.gpr_idx_en() ? amdgpu::apply_gpr_idx(wf, *off, false) : *off);
  return std::nullopt;
}

template <typename Isa>
const amdgpu::VgprStorage *
AmdgpuIsaOperand<Isa>::simd_vgpr_storage_impl(const amdgpu::Wavefront &wf) const {
  if (auto off = detail::resolved_vgpr_offset_for_operand<Isa>(wf, *this)) {
    uint32_t voff = wf.gpr_idx_en() ? amdgpu::apply_gpr_idx(wf, *off, false) : *off;
    return &wf.cu().raw_cu().template raw_vgpr_reg<64>(wf.vgpr_alloc().base + voff);
  }
  return nullptr;
}

template <typename Isa>
amdgpu::VgprStorage *
AmdgpuIsaOperand<Isa>::simd_vgpr_storage_mut_impl(amdgpu::Wavefront &wf) const {
  if (auto off = detail::resolved_vgpr_offset_for_operand<Isa>(wf, *this)) {
    uint32_t voff = wf.gpr_idx_en() ? amdgpu::apply_gpr_idx(wf, *off, true) : *off;
    return &wf.cu().raw_cu().template raw_vgpr_reg<64>(wf.vgpr_alloc().base + voff);
  }
  return nullptr;
}

template <typename Isa>
void AmdgpuIsaOperand<Isa>::simd_notify_read_impl(const amdgpu::Wavefront &wf, uint64_t lane_mask,
                                                  uint8_t byte_mask) const {
  if (auto off = detail::resolved_vgpr_offset_for_operand<Isa>(wf, *this)) {
    uint32_t voff = wf.gpr_idx_en() ? amdgpu::apply_gpr_idx(wf, *off, false) : *off;
    uint32_t physical_reg = wf.vgpr_alloc().base + voff;
    wf.cu().raw_cu().notify_vgpr_read(&wf, physical_reg, lane_mask, byte_mask);
  }
}

template <typename Isa>
void AmdgpuIsaOperand<Isa>::simd_notify_read_mut_impl(amdgpu::Wavefront &wf, uint64_t lane_mask,
                                                      uint8_t byte_mask) const {
  if (auto off = detail::resolved_vgpr_offset_for_operand<Isa>(wf, *this)) {
    uint32_t voff = wf.gpr_idx_en() ? amdgpu::apply_gpr_idx(wf, *off, true) : *off;
    uint32_t physical_reg = wf.vgpr_alloc().base + voff;
    wf.cu().raw_cu().notify_vgpr_read(&wf, physical_reg, lane_mask, byte_mask);
  }
}

template <typename Isa>
void AmdgpuIsaOperand<Isa>::simd_notify_read64_impl(const amdgpu::Wavefront &wf, uint64_t lane_mask,
                                                    uint8_t byte_mask) const {
  if (auto off = detail::resolved_vgpr_offset_for_operand<Isa>(wf, *this)) {
    uint32_t voff = wf.gpr_idx_en() ? amdgpu::apply_gpr_idx(wf, *off, false) : *off;
    uint32_t physical_reg = wf.vgpr_alloc().base + voff;
    wf.cu().raw_cu().notify_vgpr_read(&wf, physical_reg, lane_mask, byte_mask);
    wf.cu().raw_cu().notify_vgpr_read(&wf, physical_reg + 1, lane_mask, byte_mask);
  }
}

template <typename Isa>
void AmdgpuIsaOperand<Isa>::simd_notify_read64_mut_impl(amdgpu::Wavefront &wf, uint64_t lane_mask,
                                                        uint8_t byte_mask) const {
  if (auto off = detail::resolved_vgpr_offset_for_operand<Isa>(wf, *this)) {
    uint32_t voff = wf.gpr_idx_en() ? amdgpu::apply_gpr_idx(wf, *off, true) : *off;
    uint32_t physical_reg = wf.vgpr_alloc().base + voff;
    wf.cu().raw_cu().notify_vgpr_read(&wf, physical_reg, lane_mask, byte_mask);
    wf.cu().raw_cu().notify_vgpr_read(&wf, physical_reg + 1, lane_mask, byte_mask);
  }
}

template <typename Isa>
amdgpu::ConstVgprStoragePair64
AmdgpuIsaOperand<Isa>::simd_vgpr_storage64_impl(const amdgpu::Wavefront &wf) const {
  if (auto off = detail::resolved_vgpr_offset_for_operand<Isa>(wf, *this)) {
    uint32_t voff = wf.gpr_idx_en() ? amdgpu::apply_gpr_idx(wf, *off, false) : *off;
    uint32_t reg = wf.vgpr_alloc().base + voff;
    return {&wf.cu().raw_cu().template raw_vgpr_reg<64>(reg),
            &wf.cu().raw_cu().template raw_vgpr_reg<64>(reg + 1)};
  }
  return {nullptr, nullptr};
}

template <typename Isa>
amdgpu::VgprStoragePair64
AmdgpuIsaOperand<Isa>::simd_vgpr_storage64_mut_impl(amdgpu::Wavefront &wf) const {
  if (auto off = detail::resolved_vgpr_offset_for_operand<Isa>(wf, *this)) {
    uint32_t voff = wf.gpr_idx_en() ? amdgpu::apply_gpr_idx(wf, *off, true) : *off;
    uint32_t reg = wf.vgpr_alloc().base + voff;
    return {&wf.cu().raw_cu().template raw_vgpr_reg<64>(reg),
            &wf.cu().raw_cu().template raw_vgpr_reg<64>(reg + 1)};
  }
  return {nullptr, nullptr};
}

} // namespace rocjitsu

#endif // ROCJITSU_ISA_ISA_OPERAND_SIMD_INL_H_
