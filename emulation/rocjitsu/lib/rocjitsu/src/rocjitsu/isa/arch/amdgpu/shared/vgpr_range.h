// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

/// @file vgpr_range.h
/// @brief Shared VGPR-span resolution and boundary policy for instruction execution.

#include "rocjitsu/isa/operand.h"
#include "rocjitsu/vm/amdgpu/register_access.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include <cassert>
#include <cstdint>
#include <optional>

namespace rocjitsu::amdgpu {

/// Result of resolving an ordinary logical operand to an indexed VGPR span.
class ResolvedVgprSpan {
public:
  enum class Kind { NonVgpr, Valid, Invalid };

  static ResolvedVgprSpan non_vgpr() { return ResolvedVgprSpan(Kind::NonVgpr, 0); }
  static ResolvedVgprSpan valid(uint32_t offset) { return ResolvedVgprSpan(Kind::Valid, offset); }
  static ResolvedVgprSpan invalid() { return ResolvedVgprSpan(Kind::Invalid, 0); }

  [[nodiscard]] bool is_non_vgpr() const { return kind_ == Kind::NonVgpr; }
  [[nodiscard]] bool is_valid() const { return kind_ == Kind::Valid; }
  [[nodiscard]] bool is_invalid() const { return kind_ == Kind::Invalid; }
  [[nodiscard]] uint32_t offset() const {
    assert(is_valid());
    return offset_;
  }

private:
  ResolvedVgprSpan(Kind kind, uint32_t offset) : kind_(kind), offset_(offset) {}

  Kind kind_;
  uint32_t offset_;
};

/// Logical operand backed by an already-resolved VGPR allocation offset.
class PhysicalVgprOperand final : public Operand {
public:
  PhysicalVgprOperand(int size_bits, uint32_t offset)
      : Operand(size_bits, offset), offset_(offset) {
    reads_value_ = true;
    writable_ = true;
    is_vgpr_ = true;
  }

  bool simd_capable() const override { return delegate() ? delegate()->simd_capable() : true; }

private:
  [[nodiscard]] uint32_t physical_reg(const Wavefront &wf) const {
    return wf.vgpr_alloc().base + offset_;
  }

  uint32_t read_lane(const Wavefront &wf, uint32_t lane) const override {
    if (delegate())
      return RegisterAccess(wf).read_lane(*delegate(), lane);
    return RegisterAccess(wf.cu()).read_vgpr(wf.vgpr_alloc().base + offset_, lane);
  }

  void write_lane(Wavefront &wf, uint32_t lane, uint32_t value) const override {
    RegisterAccess(wf.cu()).write_vgpr(wf.vgpr_alloc().base + offset_, lane, value);
  }

  uint64_t read_lane64(const Wavefront &wf, uint32_t lane) const override {
    if (delegate())
      return RegisterAccess(wf).read_lane64(*delegate(), lane);
    return RegisterAccess(wf.cu()).read_vgpr64(wf.vgpr_alloc().base + offset_, lane);
  }

  void write_lane64(Wavefront &wf, uint32_t lane, uint64_t value) const override {
    RegisterAccess(wf.cu()).write_vgpr64(wf.vgpr_alloc().base + offset_, lane, value);
  }

  std::optional<uint32_t> simd_vgpr_base_impl(const Wavefront &wf) const override {
    return physical_reg(wf);
  }

  std::optional<uint32_t> simd_vgpr_base_mut_impl(Wavefront &wf) const override {
    return physical_reg(wf);
  }

  const VgprStorage *simd_vgpr_storage_impl(const Wavefront &wf) const override {
    return &wf.cu().raw_cu().raw_vgpr_reg<64>(physical_reg(wf));
  }

  VgprStorage *simd_vgpr_storage_mut_impl(Wavefront &wf) const override {
    return &wf.cu().raw_cu().raw_vgpr_reg<64>(physical_reg(wf));
  }

  ConstVgprStoragePair64 simd_vgpr_storage64_impl(const Wavefront &wf) const override {
    const uint32_t reg = physical_reg(wf);
    return {&wf.cu().raw_cu().raw_vgpr_reg<64>(reg), &wf.cu().raw_cu().raw_vgpr_reg<64>(reg + 1)};
  }

  VgprStoragePair64 simd_vgpr_storage64_mut_impl(Wavefront &wf) const override {
    const uint32_t reg = physical_reg(wf);
    return {&wf.cu().raw_cu().raw_vgpr_reg<64>(reg), &wf.cu().raw_cu().raw_vgpr_reg<64>(reg + 1)};
  }

  void simd_notify_read_impl(const Wavefront &wf, uint64_t lane_mask,
                             uint8_t byte_mask) const override {
    wf.cu().raw_cu().notify_vgpr_read(&wf, physical_reg(wf), lane_mask, byte_mask);
  }

  void simd_notify_read_mut_impl(Wavefront &wf, uint64_t lane_mask,
                                 uint8_t byte_mask) const override {
    wf.cu().raw_cu().notify_vgpr_read(&wf, physical_reg(wf), lane_mask, byte_mask);
  }

  void simd_notify_read64_impl(const Wavefront &wf, uint64_t lane_mask,
                               uint8_t byte_mask) const override {
    const uint32_t reg = physical_reg(wf);
    wf.cu().raw_cu().notify_vgpr_read(&wf, reg, lane_mask, byte_mask);
    wf.cu().raw_cu().notify_vgpr_read(&wf, reg + 1, lane_mask, byte_mask);
  }

  void simd_notify_read64_mut_impl(Wavefront &wf, uint64_t lane_mask,
                                   uint8_t byte_mask) const override {
    const uint32_t reg = physical_reg(wf);
    wf.cu().raw_cu().notify_vgpr_read(&wf, reg, lane_mask, byte_mask);
    wf.cu().raw_cu().notify_vgpr_read(&wf, reg + 1, lane_mask, byte_mask);
  }

  void simd_notify_write_mut_impl(Wavefront &wf, uint64_t lane_mask,
                                  uint8_t byte_mask) const override {
    wf.cu().raw_cu().notify_vgpr_write(&wf, physical_reg(wf), lane_mask, byte_mask);
  }

  void simd_notify_write64_mut_impl(Wavefront &wf, uint64_t lane_mask,
                                    uint8_t byte_mask) const override {
    const uint32_t reg = physical_reg(wf);
    wf.cu().raw_cu().notify_vgpr_write(&wf, reg, lane_mask, byte_mask);
    wf.cu().raw_cu().notify_vgpr_write(&wf, reg + 1, lane_mask, byte_mask);
  }

  uint32_t offset_;
};

[[nodiscard]] inline bool vgpr_span_in_range(const Wavefront &wf, int64_t first, uint32_t count) {
  return first >= 0 && count != 0 && static_cast<uint64_t>(first) < wf.vgpr_alloc().count &&
         count <= wf.vgpr_alloc().count - static_cast<uint64_t>(first);
}

/// Resolve an ordinary operand's VGPR address, including MODE.GPR_IDX, before
/// validating its complete register span. A missing VGPR offset means the
/// operand is a legal scalar or immediate source, not an invalid VGPR.
[[nodiscard]] inline ResolvedVgprSpan resolve_vgpr_span(const Wavefront &wf,
                                                        std::optional<uint32_t> first,
                                                        uint32_t count, bool destination) {
  if (!first)
    return ResolvedVgprSpan::non_vgpr();
  uint32_t indexed = wf.gpr_idx_en() ? apply_gpr_idx(wf, *first, destination) : *first;
  if (!vgpr_span_in_range(wf, indexed, count)) {
    if (!destination && vgpr_span_in_range(wf, 0, count))
      return ResolvedVgprSpan::valid(0);
    return ResolvedVgprSpan::invalid();
  }
  return ResolvedVgprSpan::valid(indexed);
}

[[nodiscard]] inline std::optional<uint32_t> source_vgpr_offset(const Wavefront &wf, int64_t first,
                                                                uint32_t count) {
  if (vgpr_span_in_range(wf, first, count))
    return static_cast<uint32_t>(first);

  // VALU sources outside the wave's allocated VGPR range are read from the
  // corresponding span beginning at VGPR0. This also covers MOVREL's explicit
  // Index > 255 out-of-range rule. Keep destination validation separate: an
  // out-of-range destination suppresses the instruction instead.
  return vgpr_span_in_range(wf, 0, count) ? std::optional<uint32_t>{0} : std::nullopt;
}

[[nodiscard]] inline std::optional<uint32_t>
source_vgpr_offset(const Wavefront &wf, std::optional<uint32_t> first, uint32_t count) {
  return first ? source_vgpr_offset(wf, static_cast<int64_t>(*first), count) : std::nullopt;
}

[[nodiscard]] inline std::optional<uint32_t>
destination_vgpr_offset(const Wavefront &wf, int64_t first, uint32_t count) {
  if (!vgpr_span_in_range(wf, first, count))
    return std::nullopt;
  return static_cast<uint32_t>(first);
}

[[nodiscard]] inline std::optional<uint32_t>
destination_vgpr_offset(const Wavefront &wf, std::optional<uint32_t> first, uint32_t count) {
  return first ? destination_vgpr_offset(wf, static_cast<int64_t>(*first), count) : std::nullopt;
}

} // namespace rocjitsu::amdgpu
